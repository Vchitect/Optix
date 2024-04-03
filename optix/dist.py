import torch.distributed as dist

from torchdistpackage import tpc
from .utils import set_seed

# def get_sp_src_rank():
#     """Calculate the global rank corresponding to the first local rank
#     in the tensor model parallel group."""
#     global_rank = dist.get_rank()
#     local_world_size = dist.get_world_size(tpc.get_group('sp'))
#     return (global_rank // local_world_size) * local_world_size

def setup_dp_sp(sp_size, dp_size=-1, set_sp_seed=True):
    ws = dist.get_world_size()
    if sp_size > 0 and dp_size==-1:
        assert ws%sp_size==0, "world size should be multiple of sp_size"
        dp_size = ws//sp_size
    elif dp_size > 0 and sp_size==-1:
        assert ws%dp_size==0, "world size should be multiple of dp_size"
        sp_size = ws//dp_size
    else:
        raise NotImplementedError

    tpc.setup_process_groups([('data', dp_size), ('sp', sp_size)])
    zero_group= dist.group.WORLD # dp and zero group should be global for ulysses
    dp_group =tpc.get_group('data')
    sp_group = tpc.get_group('sp')

    if set_sp_seed:
        dp_rank = dist.get_rank(dp_group)
        set_seed(dp_rank+1024)
    return dp_group, sp_group

def broadcast_input_sp(model_input):
    sp_group = tpc.get_group('sp')
    if dist.get_world_size(sp_group) <=1:
        return
    sp_rank0 = tpc.get_ranks_in_group('sp')[0]
    dist.broadcast(model_input, sp_rank0, sp_group)