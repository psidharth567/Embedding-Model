import torch
import copy

from packaging import version
import importlib.metadata

from transformers import Gemma3Model, Gemma3ForCausalLM, Gemma3PreTrainedModel, Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3Attention,
    Gemma3MLP,
    Gemma3TextModel,
    Gemma3RMSNorm,
    Gemma3TextScaledWordEmbedding,
    Gemma3RotaryEmbedding,
)

from torch import nn
from transformers.utils import logging
from torch.nn import functional as F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.import_utils import _is_package_available
from transformers.cache_utils import Cache, StaticCache



logger = logging.get_logger(__name__)


def is_transformers_attn_greater_or_equal_4_50():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.50.0"
    )


class ModifiedGemma3Attention(Gemma3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False
        self.sliding_window = None


class NonCausalGemma3DecoderLayer(Gemma3DecoderLayer):
    """Decoder layer with non-causal attention for MNTP."""
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = ModifiedGemma3Attention(config, layer_idx)


class Gemma3BiModel(Gemma3TextModel):
    _no_split_modules = ["Gemma3DecoderLayer", "NonCausalGemma3DecoderLayer"]

    def __init__(self, config: Gemma3Config):
        if not is_transformers_attn_greater_or_equal_4_50():
            raise ValueError(
                "The current implementation of Gemma3BiModel follows modeling_gemma3.py of transformers version >= 4.50.0"
            )
        Gemma3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [NonCausalGemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO: raushan fix this after RoPE refactor. For now we hack it by reassigning thetas
        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache = None,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.zeros(
                (sequence_length, target_length), dtype=dtype, device=device
            )  # in original implementation - torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            # Commenting out next 2 lines to disable causal masking
            # if sequence_length != 1:
            #     causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


class Gemma3BiForMNTP(Gemma3ForCausalLM):
    def __init__(self, config):
        Gemma3PreTrainedModel.__init__(self, config)
        self.model = Gemma3BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing 
        self.post_init()
        self.drop_lm_head(to_cpu=True, set_eval=False)

    # ------------------------------------------------------------------ #
    # Convenience helpers – not part of the original HuggingFace API
    # ------------------------------------------------------------------ #
    def set_tokenizer(self, tokenizer):
        """Attach a tokenizer instance to the model for later use by encode()."""
        self.tokenizer = tokenizer  # type: ignore[attr-defined]

    def encode(
        self,
        texts,
        *,
        batch_size: int = 8,
        pooling: str = "mean",
        max_length: int = 24_000,
    ):
        """Return sentence embeddings.

        Args:
            texts (str | List[str]): Input string(s).
            batch_size (int): Number of texts processed per forward pass.
            pooling (str): Either "mean" or "last" token pooling.
            max_length (int): Truncation length passed to the tokenizer.
        """

        if not hasattr(self, "tokenizer"):
            raise ValueError(
                "Tokenizer missing – call `model.set_tokenizer(tok)` or set the `tokenizer` attribute "
                "on the model instance before calling encode()."
            )

        if isinstance(texts, str):
            texts = [texts]
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'")

        device = next(self.parameters()).device

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # move tokenized tensors once
            inputs = inputs.to(device)
            attn_mask = inputs.get("attention_mask", None)

            model_outputs = self.model(**inputs, output_hidden_states=False, use_cache=False)
            last_hidden = model_outputs.last_hidden_state  # [batch, seq, hidden]

        
            if pooling == "mean":
                if attn_mask is not None:
                    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
                    summed = (last_hidden * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp(min=1e-6)
                    batch_emb = (summed / denom)
                else:
                    batch_emb = last_hidden.mean(dim=1)
            else:
                if attn_mask is not None:
                    # Get index of last non-pad token per row
                    lengths = attn_mask.sum(dim=1) - 1
                    lengths = lengths.clamp(min=0)
                    batch_indices = torch.arange(last_hidden.size(0), device=device)
                    batch_emb = last_hidden[batch_indices, lengths, :]
                else:
                    batch_emb = last_hidden[:, -1, :]

            all_embeddings.append(batch_emb)

        embs = torch.cat(all_embeddings, dim=0)
        embs = F.normalize(embs, dim=-1)
        return embs
 
    
    def drop_lm_head(self, *, to_cpu: bool = True, set_eval: bool = False):
        """Remove the LM head to save memory when only embeddings are needed.

        Args:
            to_cpu (bool): Move head params to CPU before deletion to free GPU VRAM sooner.
            set_eval (bool): Set module to eval mode after surgery to prevent accidental training.
        """
        
        if hasattr(self, "lm_head") and isinstance(self.lm_head, nn.Module):
            try:
                if to_cpu:
                    params = list(self.lm_head.parameters(recurse=True)) + list(self.lm_head.buffers(recurse=True))
                    has_meta = any(getattr(t, "is_meta", False) for t in params)
                    if has_meta:
                        to_empty = getattr(self.lm_head, "to_empty", None)
                        if callable(to_empty):
                            to_empty(device=torch.device("cpu"))
                        # else: keep as-is (meta holds no memory)
                    else:
                        self.lm_head.to(device=torch.device("cpu"))
                del self.lm_head
            finally:
                self.lm_head = nn.Identity()
        if set_eval:
            self.eval()
