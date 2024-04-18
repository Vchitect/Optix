import functools

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy,size_based_auto_wrap_policy
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import print_rank0

def setup_fsdp_encoder(model: nn.Module, process_group=None, policy='lambda') -> FSDP:
    if policy == 'size':
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e8)
    elif policy == 'lambda':
        if hasattr(model, 'encoder'):
            lambda_fn = lambda m: m in list(model.encoder.block)
        elif hasattr(model, 'layers'):
            lambda_fn=lambda m: m in list(model.layers)
        else:
            raise NotImplementedError
        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                            lambda_fn=lambda_fn,)
    model = FSDP(
        model,
        auto_wrap_policy= auto_wrap_policy,
        process_group=process_group,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model

def setup_fsdp_training(model: nn.Module, strategy='fsdp', process_group=None,
                        compute_dtype = torch.bfloat16, mix_precision_level=1, policy='size', verbose=False) -> FSDP:
    if policy == 'size':
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e8)
    elif policy == 'lambda':
        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                            lambda_fn=lambda m: m in list(model.layers),)

    if mix_precision_level==2:
        # apex O2, mix precision
        mixed_precision=MixedPrecision(
            param_dtype = compute_dtype,
            reduce_dtype=torch.float32,
            keep_low_precision_grads=False,
        )
        if verbose:
            print_rank0(f"setting up FSDP with model_dtype={compute_dtype} and mixed_precision !")

    elif mix_precision_level==3:
        # O3, model should be in 16bit
        model.to(compute_dtype)
        mixed_precision=MixedPrecision(
            param_dtype = compute_dtype,    # reduce_dtype will be param_dtype
            keep_low_precision_grads=True,
        )
        if verbose:
            print_rank0(f"setting up FSDP with model & grad in {compute_dtype} !")
    else:
        # torch amp, fp32
        model.to(torch.float)
        mixed_precision = None
        if verbose:
            print_rank0("setting up FSDP with model_dtype=fp32 and no mixed_precision !")

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        process_group=process_group,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[strategy],
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model

def setup_ddp_training(model: nn.Module, process_group=None,
                        compute_dtype = torch.bfloat16, mix_precision_level=1):
    if mix_precision_level==1:
        model = model.float()
    elif mix_precision_level==2:
        raise NotImplementedError("DDP _MixedPrecision is not supported")
    else:
        # pure 16bit
        model = model.to(compute_dtype)
    model = DDP(model, process_group=process_group, gradient_as_bucket_view=True)

    return model
