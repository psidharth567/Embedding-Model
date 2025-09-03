#!/usr/bin/env python
"""
Train Gemma3BiForMNTP with the SemanticContrastiveLearner on a jsonl file
containing *text1*, *text2* pairs.
"""
from __future__ import annotations
import os, json, argparse, math, time
from types import SimpleNamespace
from pathlib import Path
from typing import List, Dict
import os, json, argparse, math, time, random, numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from torch.optim import AdamW
import inspect
import wandb
# --------------------------------------------------------------------------- #
# Import the learner & backbone
# --------------------------------------------------------------------------- #
from bidirection_gemma3 import Gemma3BiForMNTP          # backbone

from AllGather import AllGather
AllGather = AllGather.apply


class SemanticContrastiveLearner(torch.nn.Module):
    """
    Wrap a sentence-encoder (Gemma3BiForMNTP) and train it with a
    bidirectional InfoNCE objective.

    Args
    ----
    encoder      : the backbone producing per-token hidden states
    task_conf    : config object with .n_gpus and .global_batch_size
    temperature  : softmax temperature (float)
    """

    def __init__(self, encoder, task_conf, temperature: float = 0.05, k_neg: int = 0, margin: float = 0.1):
        super().__init__()
        self.encoder       = encoder
        self.task_conf     = task_conf
        self.temperature   = temperature
        self.k_neg         = int(k_neg) if k_neg is not None else 0
        self.margin        = margin

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    
    def _masked_inbatch_logits(self, sim_matrix: torch.Tensor, pos_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sim_matrix: (B, global_B). Mask suspiciously hard in-batch negatives (> s_pos + margin) as per Qwen-3 Embedding loss function
        # https://arxiv.org/pdf/2506.05176v2

        B, global_B = sim_matrix.size()
        device = sim_matrix.device
        row_idx = torch.arange(B, device=device)
        s_pos = sim_matrix[row_idx, pos_indices]  # (B,)
        if self.margin is None:
            keep = torch.ones_like(sim_matrix, dtype=torch.bool)
        else:
            keep = sim_matrix <= (s_pos.unsqueeze(1) + self.margin)
        keep[row_idx, pos_indices] = True  # always keep positives
        masked = sim_matrix.masked_fill(~keep, float('-inf'))

        return masked, s_pos

    def _encode(self, texts, *, batch_size: int, pooling: str = "mean", max_length: int = 24_000):
        """Return L2-normalised sentence embeddings on the model device.

        The underlying encoder.encode() handles tokenization and returns CPU tensors;
        we move them to the current device for downstream ops (e.g., AllGather).
        """
        out = self.encoder.encode(
            texts,
            batch_size=batch_size,
            pooling=pooling,
            max_length=max_length,
        )
        device = next(self.encoder.parameters()).device
        return out.to(device, non_blocking=True)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, batch, *, args):
        """
        A-> Queries
        B-> Documents
        Expects in batch (Dict):
            texts_a: List[str]
            texts_b: List[str]
            [optional] neg_texts_b: Flat List[str] of size B*K for A→B negatives
            [optional] neg_texts_a: Flat List[str] of size B*K for B→A negatives
        """
        texts_a: list[str] = batch["texts_a"]
        texts_b: list[str] = batch["texts_b"]
        neg_texts_b: list[str] | None = batch.get("neg_texts_b", None)
        neg_texts_a: list[str] | None = batch.get("neg_texts_a", None)

        combined_texts: list[str] = list(texts_a) + list(texts_b) # Queries + Documents in a single list
        has_neg_b = neg_texts_b is not None and len(neg_texts_b) > 0 and (self.k_neg > 0)
        has_neg_a = neg_texts_a is not None and len(neg_texts_a) > 0 and (self.k_neg > 0)
        if has_neg_b:
            combined_texts.extend(neg_texts_b)  # flat B*K; each sample has K negatives and we have B samples
        if has_neg_a:
            combined_texts.extend(neg_texts_a)  # flat B*K; each sample has K negatives and we have B samples
            

        micro_bs = getattr(self.task_conf, "encode_batch_size", 0) or len(combined_texts) # batch size for forward pass
        max_len  = getattr(self.task_conf, "max_length", 24_000) # Truncation length

        # We combine all the texts and encode them together as the kernels needed are called only once and forward-passes are quicker
        all_embs = self._encode( # encode method takes texts and returns embeddings on the model device, it handles tokenization internally
            combined_texts,
            batch_size=micro_bs,
            pooling="mean",
            max_length=max_len,
        )

        B = len(texts_a) # Number of queries
        z1 = all_embs[:B] # Queries embeddings
        z2 = all_embs[B:2*B] # Documents embeddings
        # z1 and z2 are embeddings for the queries and documents respectively and we have the same number of them
        
        cursor = 2 * B # Cursor to keep track of the position of the negatives in the combined embeddings

        neg_z2 = None
        neg_z1 = None

        if has_neg_b:
            neg_total = B * self.k_neg # B*K: Number of negatives for documents
            neg_z2 = all_embs[cursor:cursor+neg_total].view(B, self.k_neg, -1) # (B, K, d)
            cursor += neg_total

        # We will not be using hard-negatives for queries. It's just there as an artifact and can be removed.
        if has_neg_a:
            neg_total = B * self.k_neg
            neg_z1 = all_embs[cursor:cursor+neg_total].view(B, self.k_neg, -1)
            cursor += neg_total

        if self.training and self.task_conf.n_gpus > 1: # Multi-GPU

            all_z1 = AllGather(z1, args)   # (global_B, d)
            all_z2 = AllGather(z2, args)   # (global_B, d)
            # We gather the embeddings across all the GPUs to get the global embeddings
            # It's grad-safe AllGather as it handles the gradients correctly: https://github.com/zsnoob/EfficientDDP-4-Contrastive-Train

             # Number of queries/documents per GPU; local batch = B
            shift = B * dist.get_rank()

            # Shift for rank0: 0
            # Shift for rank1: B
            # Shift for rank2: 2*B
            # ...

        else: # Single GPU case
            all_z1, all_z2 = z1, z2
            shift = 0

        # In-batch similarities
        sim_1 = self.temperature * z1 @ all_z2.T   # (B,d) @ (d, global_B) = (B, global_B)
        sim_2 = self.temperature * z2 @ all_z1.T   # (B,d) @ (d, global_B) = (B, global_B)

        # Mask suspiciously similar in-batch negatives (> s_pos + margin)
        pos_idx_1 = shift + torch.arange(B, device=z1.device)  # pos_idx_1 = [0, 1, 2, 3, ...] + shift
        pos_idx_2 = shift + torch.arange(B, device=z2.device)
        masked_1, s_pos_1 = self._masked_inbatch_logits(sim_1, pos_idx_1)
        masked_2, s_pos_2 = self._masked_inbatch_logits(sim_2, pos_idx_2)

        # Static K hard negatives per query (ALWAYS included; no masking)
        # No AllGather for hard-negatives. They are restricted to the local batch only.

        extra_1 = None
        if neg_z2 is not None:
            extra_1 = self.temperature * torch.bmm(z1.unsqueeze(1), neg_z2.transpose(1, 2)).squeeze(1)  # (B, K)

        extra_2 = None
        if neg_z1 is not None:
            extra_2 = self.temperature * torch.bmm(z2.unsqueeze(1), neg_z1.transpose(1, 2)).squeeze(1)  # (B, K)

        # Concatenate in-batch (masked) + extra negatives, then InfoNCE
        if extra_1 is not None:
            denom_1 = torch.logsumexp(torch.cat([masked_1, extra_1], dim=1), dim=-1)
        else:
            denom_1 = torch.logsumexp(masked_1, dim=-1)
        if extra_2 is not None:
            denom_2 = torch.logsumexp(torch.cat([masked_2, extra_2], dim=1), dim=-1)
        else:
            denom_2 = torch.logsumexp(masked_2, dim=-1)

        loss_1 = -(s_pos_1 - denom_1).mean()
        loss_2 = -(s_pos_2 - denom_2).mean()
        return 0.5 * (loss_1 + loss_2)

        
# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class JsonlPairs(Dataset):
    def __init__(self, path: str | Path, k_neg: int | None = None):
        self.lines: List[str] = Path(path).read_text().splitlines()
        self.k_neg = k_neg

    def __len__(self):  
        return len(self.lines)

    def __getitem__(self, idx):
        obj = json.loads(self.lines[idx])
        item = {
            "text1": obj["text1"],
            "text2": obj["text2"],
            "hard_negs_b": obj.get("hard_negs_b", obj.get("hard_negs", [])),
            "hard_negs_a": obj.get("hard_negs_a", []),
        }
        if self.k_neg is not None:
            item["hard_negs_b"] = item["hard_negs_b"][: self.k_neg]
            item["hard_negs_a"] = item["hard_negs_a"][: self.k_neg]
        return item     # tuple[str, str]

# --------------------------------------------------------------------------- #
# Helper to build batches with dynamic padding
# --------------------------------------------------------------------------- #
def make_collate_fn(k_neg: int | None = None):
    def collate(batch):
        text_a = [ex["text1"] for ex in batch]
        text_b = [ex["text2"] for ex in batch]

        out = {
            "texts_a": text_a,
            "texts_b": text_b,
        }

        if k_neg and k_neg > 0:
            negs_b = [ex.get("hard_negs_b", []) for ex in batch]
            negs_a = [ex.get("hard_negs_a", []) for ex in batch]

            assert all(len(n) >= k_neg for n in negs_b), "Not enough hard_negs_b in a sample"
            # hard_negs_a may be empty for symmetric training if you only mine B-side
            assert all(len(n) >= k_neg for n in negs_a) or all(len(n) == 0 for n in negs_a), "Invalid hard_negs_a"

            flat_b = [s for lst in (nb[:k_neg] for nb in negs_b) for s in lst]
            out["neg_texts_b"] = flat_b

            if any(len(n) >= k_neg for n in negs_a):
                flat_a = [s for lst in (na[:k_neg] for na in negs_a) for s in lst]
                out["neg_texts_a"] = flat_a

        return out
    return collate

# --------------------------------------------------------------------------- #
# Small struct passed to learner.forward for AllGather
# --------------------------------------------------------------------------- #
def ddp_namespace():
    if dist.is_available() and dist.is_initialized():
        return SimpleNamespace(rank=dist.get_rank(),
                               world_size=dist.get_world_size())
    else:
        return SimpleNamespace(rank=0, world_size=1)

# --------------------------------------------------------------------------- #
# WSD LR scheduler: Warmup -> Stable -> Decay
# --------------------------------------------------------------------------- #
def build_wsd_scheduler(optimizer, total_steps: int, *, warmup_ratio: float = 0.1,
                        stable_ratio: float = 0.4, min_lr_ratio: float = 0.1,
                        decay: str = "cosine"):
    warmup = max(1, int(total_steps * warmup_ratio))
    stable = max(0, int(total_steps * stable_ratio))
    decay_steps = max(1, total_steps - warmup - stable)

    import math
    def lr_lambda(step: int):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        elif step < warmup + stable:
            return 1.0
        else:
            t = step - warmup - stable
            if decay == "cosine":
                cos = 0.5 * (1.0 + math.cos(math.pi * float(t) / float(decay_steps)))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cos
            elif decay == "linear":
                return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - float(t) / float(decay_steps))
            else:
                return min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def _collect_random_states():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_training_state(file_path: Path, *, learner, optimizer, scheduler, epoch, step_in_epoch, update_idx, total_updates, ds, dl, cfg, distributed, wandb_run_id=None):
    to_save = {
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "update_idx": update_idx,
        "total_updates": total_updates,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "random_states": _collect_random_states(),
        "cfg": vars(cfg),
        "distributed": distributed,
        "dataset_state": {
            "dataset_len": len(ds),
            "step_in_epoch": step_in_epoch,
            "batch_size": dl.batch_size,
            "accum": cfg.accum,
            "sampler_epoch": getattr(dl.sampler, "epoch", epoch) if dl.sampler is not None else epoch,
        },
    }
    if wandb_run_id is not None:
        to_save["wandb_run_id"] = wandb_run_id
    module = learner.module if isinstance(learner, DDP) else learner
    to_save["model_state_dict"] = module.state_dict()
    to_save["encoder_state_dict"] = module.encoder.state_dict()
    torch.save(to_save, file_path)

def _save_milestone_checkpoint(root: Path, pct: int, update_idx: int, *, learner, optimizer, scheduler, epoch, step_in_epoch, total_updates, ds, dl, cfg, distributed, tokenizer, wandb_run_id=None):
    save_dir = _ensure_dir(root / f"pct_{pct:02d}_update_{update_idx:06d}")
    _save_training_state(save_dir / "state.pt",
                         learner=learner, optimizer=optimizer, scheduler=scheduler,
                         epoch=epoch, step_in_epoch=step_in_epoch,
                         update_idx=update_idx, total_updates=total_updates,
                         ds=ds, dl=dl, cfg=cfg, distributed=distributed, wandb_run_id=wandb_run_id)
    # Also save an HF snapshot for ablations (never deleted)
    module = learner.module if isinstance(learner, DDP) else learner
    hf_dir = _ensure_dir(save_dir / "hf_model")
    module.encoder.save_pretrained(hf_dir)
    try:
        tokenizer.save_pretrained(hf_dir)
    except Exception:
        pass

def _save_step_checkpoint(steps_dir: Path, update_idx: int, *, learner, optimizer, scheduler, epoch, step_in_epoch, total_updates, ds, dl, cfg, distributed, wandb_run_id=None):
    _ensure_dir(steps_dir)
    file_path = steps_dir / f"step_{update_idx:06d}.pt"
    _save_training_state(file_path,
                         learner=learner, optimizer=optimizer, scheduler=scheduler,
                         epoch=epoch, step_in_epoch=step_in_epoch,
                         update_idx=update_idx, total_updates=total_updates,
                         ds=ds, dl=dl, cfg=cfg, distributed=distributed, wandb_run_id=wandb_run_id)

def _prune_step_checkpoints(steps_dir: Path, *, keep: int = 5):
    if not steps_dir.exists():
        return
    ckpts = sorted([p for p in steps_dir.glob("step_*.pt")], key=lambda p: int(p.stem.split("_")[1]))
    if len(ckpts) > keep:
        for p in ckpts[:-keep]:
            try:
                p.unlink()
            except Exception:
                pass

def _latest_step_checkpoint(steps_dir: Path) -> Path | None:
    if not steps_dir.exists():
        return None
    ckpts = sorted([p for p in steps_dir.glob("step_*.pt")], key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None

def _restore_random_states(state: Dict):
    try:
        random.setstate(state["python"])
    except Exception:
        pass
    try:
        np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        torch.set_rng_state(state["torch_cpu"])
    except Exception:
        pass
    try:
        if state.get("torch_cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass

def _maybe_autoresume(steps_dir: Path, *, learner, optimizer, scheduler, device):
    ckpt = _latest_step_checkpoint(steps_dir)
    if ckpt is None:
        return {"resume": False, "epoch": 0, "step_in_epoch": 0, "update_idx": 0, "wandb_run_id": None}
    data = torch.load(ckpt, map_location=device)
    module = learner.module if isinstance(learner, DDP) else learner
    module.load_state_dict(data["model_state_dict"], strict=False)
    optimizer.load_state_dict(data["optimizer"])
    if scheduler is not None and data.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(data["scheduler"])
        except Exception:
            pass
    if "random_states" in data:
        _restore_random_states(data["random_states"])
    return {
        "resume": True,
        "epoch": int(data.get("epoch", 0)),
        "step_in_epoch": int(data.get("step_in_epoch", 0)),
        "update_idx": int(data.get("update_idx", 0)),
        "path": str(ckpt),
        "wandb_run_id": data.get("wandb_run_id", None),
    }

# --------------------------------------------------------------------------- #
# Training routine
# --------------------------------------------------------------------------- #
def main(cfg):
    # ------------- DDP init ------------- #
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                          else "cpu")

    # ------------- data ----------------- #
    ds         = JsonlPairs(cfg.jsonl, k_neg=cfg.k_neg)
    backbone   = Gemma3BiForMNTP.from_pretrained(
                    cfg.ckpt, torch_dtype=torch.bfloat16).to(device)
    # Load tokenizer and attach to backbone for encode()
    tokenizer  = AutoTokenizer.from_pretrained(cfg.ckpt, use_fast=True)
    backbone.set_tokenizer(tokenizer)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) # Enable gradient checkpointing

    collate_fn = make_collate_fn(k_neg=cfg.k_neg)

    # We do not shuffle. The order of the training samples is critical; we must first externally process the dataset and arrange samples in the correct order
    sampler = DistributedSampler(ds, shuffle=False) if distributed else None 
    # Make DataLoader robust to num_workers=0
    dl = DataLoader(
        ds, batch_size=cfg.batch_size,
        sampler=sampler, shuffle=False, drop_last=True,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        pin_memory=True,
    )
    # ------------- learner -------------- #
    class _TaskConf:               # simple container
        def __init__(self, n_gpus, global_bs, max_length, encode_batch_size):
            self.n_gpus = n_gpus
            self.global_batch_size = global_bs
            self.max_length = max_length
            self.encode_batch_size = encode_batch_size
    
    task_conf = _TaskConf(n_gpus=world_size,
                          global_bs=len(dl)*cfg.batch_size,
                          max_length=cfg.max_length,
                          encode_batch_size=cfg.encode_batch_size)

    learner   = SemanticContrastiveLearner(encoder=backbone, task_conf=task_conf,
                                       temperature=cfg.temperature,
                                       k_neg=cfg.k_neg, margin=cfg.margin).to(device)

    if distributed:
        learner = DDP(learner, device_ids=[local_rank],
                      find_unused_parameters=False)

    # ------------- optim & sched -------- #
    no_decay = {"bias", "LayerNorm.weight"}
    params = [
        {"params":[p for n,p in learner.named_parameters()
                   if not any(k in n for k in no_decay)],
         "weight_decay": cfg.wd},
        {"params":[p for n,p in learner.named_parameters()
                   if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    # And build the optimizer like this
    opt_kwargs = {"lr": cfg.lr, "betas": (0.9, 0.98)}

    if "fused" in inspect.signature(AdamW.__init__).parameters:
        opt_kwargs["fused"] = torch.cuda.is_available()

    opt = AdamW(params, **opt_kwargs)

    total_steps  = math.ceil(len(dl)*cfg.epochs / cfg.accum)

    sched = build_wsd_scheduler(
                opt, total_steps,
                warmup_ratio=cfg.warmup_ratio,
                stable_ratio=cfg.stable_ratio,
                min_lr_ratio=cfg.min_lr_ratio,
                decay=cfg.decay_style) # We use a warmup-stable-decay scheduler

    # ---------- checkpoint schedule ----------
    total_updates = total_steps
    milestone_updates = {}
    for pct in range(5, 101, 5):  # every 5% including 100%
        upd = max(1, round(total_updates * (pct / 100.0)))
        milestone_updates[upd] = pct
    steps_dir = Path(cfg.out_dir) / "steps"
    milestones_dir = Path(cfg.out_dir) / "milestones"
    if (not distributed) or (dist.get_rank() == 0):
        _ensure_dir(steps_dir)
        _ensure_dir(milestones_dir)

    # --------- auto-resume from rolling checkpoints ---------
    resume_info = _maybe_autoresume(steps_dir, learner=learner, optimizer=opt, scheduler=sched, device=device)
    start_epoch = resume_info["epoch"] if resume_info.get("resume") else 0
    resume_step_in_epoch = resume_info["step_in_epoch"] if resume_info.get("resume") else 0
    update_idx = resume_info["update_idx"] if resume_info.get("resume") else 0
    if resume_info.get("resume") and ((not distributed) or (local_rank == 0)):
        print(f"Resuming from {resume_info['path']} at epoch={start_epoch}, step_in_epoch={resume_step_in_epoch}, update_idx={update_idx}")
    # --------------------------------------------------------

    # ------------- wandb setup -------- #
    wandb_run = None
    wandb_run_id = None
    if cfg.wandb_key is not None and ((not distributed) or (local_rank == 0)):
        # Login to wandb
        wandb.login(key=cfg.wandb_key)

        # Determine if we're resuming or starting new
        is_resuming = resume_info.get("resume", False)
        existing_wandb_run_id = resume_info.get("wandb_run_id", None) if is_resuming else None

        # Initialize wandb
        wandb_config = {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "wd": cfg.wd,
            "temperature": cfg.temperature,
            "max_length": cfg.max_length,
            "k_neg": cfg.k_neg,
            "margin": cfg.margin,
            "accum": cfg.accum,
            "warmup_ratio": cfg.warmup_ratio,
            "stable_ratio": cfg.stable_ratio,
            "min_lr_ratio": cfg.min_lr_ratio,
            "decay_style": cfg.decay_style,
        }

        if is_resuming and existing_wandb_run_id is not None:
            # Resume existing wandb run
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                id=existing_wandb_run_id,
                resume="must",
                config=wandb_config,
            )
            wandb_run_id = existing_wandb_run_id
            print(f"Resumed wandb run: {wandb_run_id}")
        else:
            # Start new wandb run
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=wandb_config,
            )
            wandb_run_id = wandb_run.id
            print(f"Started new wandb run: {wandb_run_id}")

    # ------------- training loop -------- #
    learner.train()
    for epoch in range(start_epoch, cfg.epochs):
        if distributed: sampler.set_epoch(epoch)
        running_loss = 0.0
        opt.zero_grad(set_to_none=True)

        # skip already-finished mini-batches when resuming mid-epoch
        skip_until = resume_step_in_epoch if (epoch == start_epoch and update_idx > 0) else 0

        for step, batch in enumerate(dl, 1):
            if skip_until and step <= skip_until:
                continue
            # move tensors to GPU (texts remain as Python lists)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            loss = learner(batch, args=ddp_namespace()) / cfg.accum
            loss.backward()
            running_loss += loss.item()

            did_update = False
            if step % cfg.accum == 0:
                torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                update_idx += 1
                did_update = True

            if step % 50 == 0 and (not distributed or local_rank==0):
                avg = running_loss/step
                print(f"[E{epoch+1}] step {step:5d}/{len(dl)}  "
                    f"loss={avg:.4f}")

                # Log to wandb
                if wandb_run is not None:
                    current_lr = opt.param_groups[0]['lr']
                    wandb.log({
                        "epoch": epoch + 1,
                        "step": step,
                        "global_step": update_idx,
                        "loss": avg,
                        "learning_rate": current_lr,
                    }, step=update_idx)

            if did_update and (not distributed or local_rank==0):
                if update_idx % 1000 == 0:
                    _save_step_checkpoint(steps_dir, update_idx,
                                        learner=learner, optimizer=opt, scheduler=sched,
                                        epoch=epoch, step_in_epoch=step,
                                        total_updates=total_updates, ds=ds, dl=dl, cfg=cfg, distributed=distributed,
                                        wandb_run_id=wandb_run_id)
                    _prune_step_checkpoints(steps_dir, keep=5)
                if update_idx in milestone_updates:
                    pct = milestone_updates[update_idx]
                    _save_milestone_checkpoint(milestones_dir, pct, update_idx,
                                            learner=learner, optimizer=opt, scheduler=sched,
                                            epoch=epoch, step_in_epoch=step, total_updates=total_updates,
                                            ds=ds, dl=dl, cfg=cfg, distributed=distributed, tokenizer=tokenizer,
                                            wandb_run_id=wandb_run_id)

    # end-epoch save
    if local_rank==0:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        learner.module.encoder.save_pretrained(out) if distributed \
            else learner.encoder.save_pretrained(out)
        tokenizer.save_pretrained(out)

    # Finish wandb run
    if wandb_run is not None and ((not distributed) or (local_rank == 0)):
        wandb.finish()

    if distributed:
        dist.destroy_process_group()

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",        required=True,
                   help="Path to jsonl file with text1/text2 pairs")
    p.add_argument("--ckpt",         default="google/gemma-2b",
                   help="HF checkpoint or local path")
    p.add_argument("--out_dir",      default="chkpt_semantic_contrastive")
    p.add_argument("--epochs",       type=int, default=1)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--wd",           type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--accum",        type=int, default=1,
                   help="Gradient-accumulation steps")
    p.add_argument("--temperature",  type=float, default=0.05)
    p.add_argument("--max_length",   type=int, default=24000)
    p.add_argument("--encode_batch_size", type=int, default=0,
                   help="Micro-batch size used inside model.encode(); 0 means use full batch.")
    
    p.add_argument("--k_neg",        type=int, default=0,
               help="Number of per-query hard negatives from dataset (kept unmasked).")
    p.add_argument("--margin",       type=float, default=0.1,
               help="Mask in-batch negatives with sim > s_pos + margin.")

    # WSD scheduler knobs
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--stable_ratio", type=float, default=0.40)
    p.add_argument("--min_lr_ratio", type=float, default=0.10)
    p.add_argument("--decay_style", type=str, default="cosine", choices=["cosine", "linear"])

    # Wandb logging arguments
    p.add_argument("--wandb_key",     type=str,   default=None,
                   help="Wandb API key for login")
    p.add_argument("--wandb_project", type=str,   default="Embedding Model",
                   help="Wandb project name")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Wandb run name (if None, will use auto-generated name)")


    cfg = p.parse_args()
    main(cfg) 
