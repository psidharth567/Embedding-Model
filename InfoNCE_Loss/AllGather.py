import torch
import torch.distributed as dist


# With optimizing your similarity matrix result from [global, global] to [local, global]
class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.local_batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Reduce gradients prior to gradient bucket to avoid behavior mismatches with non-distributed training
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)

        return (
            grad_output[ctx.local_batch_size * ctx.rank: ctx.local_batch_size * (ctx.rank + 1)],
            None,
        )

    # GPT-5-thinking-high suggested to use reduce-scatter instead of all gather. 
    # Reduce Scatter is much more efficient but I don't yet know if we can replace that with AllReduce here.

    # @staticmethod
    # def backward(ctx, grad_output):
    #     # local shard after reduction
    #     local = grad_output.narrow(0, ctx.local_batch_size * ctx.rank, ctx.local_batch_size)
    #     out = torch.empty_like(local)
    #     dist.reduce_scatter_tensor(out, grad_output, op=dist.ReduceOp.SUM)
    #     return out, None
