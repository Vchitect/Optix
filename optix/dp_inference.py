import torch
import torch.distributed as dist

def dp_fwd(fwd_fn, input: torch.Tensor, group=None):
    """split the input batch into micro batches, and gather them in the end
    """
    rank = dist.get_rank(group)
    ws = dist.get_world_size(group)
    local_slice = input.chunk(ws, dim=0)[rank]
    local_output = fwd_fn(local_slice)
    output_shape = input.shape[:1] + local_output.shape[1:]
    output = torch.empty(output_shape, device=local_output.device, dtype=local_output.dtype)

    dist.all_gather_into_tensor(output, local_output, group=group)
    return output