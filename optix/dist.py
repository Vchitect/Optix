import torch.distributed as dist

from torchdistpackage import tpc
from .utils import set_seed

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

    tpc.setup_process_groups([('sp', sp_size),('data', dp_size)])
    zero_group= dist.group.WORLD # dp and zero group should be global for ulysses
    dp_group =tpc.get_group('data')
    sp_group = tpc.get_group('sp')

    if set_sp_seed:
        dp_rank = dist.get_rank(dp_group)
        set_seed(dp_rank+1024)
    return dp_group, sp_group

