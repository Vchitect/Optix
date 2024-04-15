import functools

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy,size_based_auto_wrap_policy
import torch.nn as nn

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
                        precision = 'bf16', grad_precision=None, policy='size') -> FSDP:
    if policy == 'size':
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e8)
    elif policy == 'lambda':
        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                            lambda_fn=lambda m: m in list(model.layers),)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        process_group=process_group,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[strategy],
        # mixed_precision=MixedPrecision(
        #     param_dtype={
        #         "fp32": torch.float, "tf32": torch.float,
        #         "bf16": torch.bfloat16, "fp16": torch.float16,
        #     }[precision],
        #     reduce_dtype={
        #         "fp32": torch.float, "tf32": torch.float,
        #         "bf16": torch.bfloat16, "fp16": torch.float16,
        #     }[grad_precision or precision],
        # ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model

