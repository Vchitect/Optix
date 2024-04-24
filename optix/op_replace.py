import torch
try:
    from apex.normalization import FusedLayerNorm
except ImportError as e:
    try:
        from xformers.triton import FusedLayerNorm
    except ImportError as e:
        FusedLayerNorm = None

from .utils import print_rank0

def replace_all_layernorms(model):
    if FusedLayerNorm is None:
        print_rank0("WARNING: apex.normalization & xformers.triton.FusedLayerNorm is not found, \
              skip using FusedLayerNorm")
        return model
    for name, module in model.named_children():
        if isinstance(module, torch.nn.LayerNorm):
            setattr(model, name, FusedLayerNorm(
                module.normalized_shape, module.eps, module.elementwise_affine))
        else:
            replace_all_layernorms(module)
    return model


def replace_all_groupnorms(model):
    try:
        from apex.contrib.group_norm import GroupNorm
    except ImportError as e:
        print_rank0("WARNING: apex.contrib.group_norm is not found, skip using apex groupnorm")
        return model
    for name, module in model.named_children():
        if isinstance(module, torch.nn.GroupNorm):
            setattr(model, name, GroupNorm(
                module.num_groups, module.num_channels,
                eps=module.eps, affine=module.affine))
        else:
            replace_all_groupnorms(module)
    return model