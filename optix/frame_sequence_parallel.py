import torch

from .sp_comm_utils import _SeqAllToAll
from einops import rearrange


def spatial2temporal(x, b, group=None):
    """transform sptial tensor (b*f/np, s, d) into temporal tensor (b*s/np, f, d)

    Args:
        x (Tensor): input spatial tensor
        group: torch comm group. Defaults to None.
    """
    # _, s, d = x.shape
    # ws = torch.distributed.get_world_size(group)
    out = _SeqAllToAll.apply(group, x, 1, 0)          # out: b*f, s/np, d

    # b*f, s/np, d -> b*s/np, f, d
    out = rearrange(out, '(b f) s_p d -> (b s_p) f d', b=b).contiguous()
    return out


def temporal2spatial(x, b, group=None):
    """transform temporal tensor (b*s/np, f, d) -> sptial tensor (b*f/np, s, d)

    Args:
        x (Tensor): input temporal tensor
        group: torch comm group. Defaults to None.
    """
    # _, f, d = x.shape
    # ws = torch.distributed.get_world_size(group)
    out = _SeqAllToAll.apply(group, x, 1, 0)          # out: b*s, f/np, d

    # b*s, f/np, d -> b*f/np, s, d
    out = rearrange(out, '(b s) f_p d -> (b f_p) s d', b=b).contiguous()
    return out