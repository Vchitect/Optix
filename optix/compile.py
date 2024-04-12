import types
import copy

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .shardedema import ShardedEMA,FSDPEmaWrapper
from .op_replace import replace_all_layernorms, replace_all_groupnorms
from .utils import setup_node_groups, setup_distributed, print_rank0, enable_tf32
from .fsdp_wrappers import setup_fsdp_training, setup_fsdp_encoder

def get_optimizer(opt_name:str):
    opt_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    return opt_map.get(opt_name.lower())

def get_valid_cfg(config, **kwargs):
    default_kwargs = {
        'use_ema': False,                   # create ema
        'compile_vae': True,                # [PERF] for torch>2.0, recommended to use torch.compile
        'ddp': True,                        # automatically create a ddp module over model
        'dp_group': None,                   # ddp communication group, default is None
        'gradient_checkpointing': True,     # [PERF] grad_ckpt is ON by default; for small batchsize this can be turned off for speedup
        'xformer': False,                   # [PERF] use xformer can speedup a little bit
        'fusedln': True,                    # [PERF] use fusedln can speedup
        'compile_model': False,             # [PERF] this function is not stable so OFF by default
        'vae_channels_last': False,         # [PERF] use channels_last format for vae
        'optim': 'adamw',                   # the optimizer type
        'learning_rate': 1e-5,              # optimizer params
        'weight_decay': 0,                  # optimizer params
        'hybrid_zero': False,               # [PERF] for multi node training, hybrid zero can be faster
        'fsdp': False,
        "fsdp_strategy": 'sdp',
        "model_dtype": torch.bfloat16
    }
    if kwargs:
        default_kwargs.update(kwargs)
    if config is None:
        config = types.SimpleNamespace(**default_kwargs)
    else:
        for key in default_kwargs:
            if not hasattr(config, key):
                setattr(config, key, default_kwargs[key])
    return config

def optimize_sd_vae(vae, config):
    if vae is None:
        return vae
    # vae optimization
    vae.requires_grad_(False)
    vae.to(device='cuda', dtype=torch.float32)
    if config.vae_channels_last:
        vae.encoder = vae.encoder.to(memory_format=torch.channels_last)
        # vae.encoder = replace_all_groupnorms(vae.encoder).to(device='cuda')

    if config.compile_vae:
        vae.encoder = torch.compile(vae.encoder)
    return vae


def compile(model, vae, config=None, **kwargs):
    # env settings
    enable_tf32()
    torch._dynamo.config.suppress_errors = True

    # get a default config if None
    config = get_valid_cfg(config, **kwargs)

    print_rank0(f"Optimization config: {config}")

    vae = optimize_sd_vae(vae, config)

    # model optimizations
    if config.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()

    if config.xformer and hasattr(model, 'enable_xformers_memory_efficient_attention'):
        model.enable_xformers_memory_efficient_attention()

    if config.fusedln:
        model = replace_all_layernorms(model).to(device='cuda')
        # model = replace_all_groupnorms(model).to(device='cuda')

    if config.compile_model:
        model = torch.compile(model)

    optimizer_class=get_optimizer(config.optim)
    ema = None
    if dist.is_initialized() and dist.get_world_size() > 1:
        if config.fsdp:
            if config.use_ema:
                model = model.cpu()
                model_ema = copy.deepcopy(model)
                model.to(config.model_dtype)
                model_ema.to(config.model_dtype)
                fsdp_args = {
                    'strategy': config.fsdp_strategy,
                }
                ema = setup_fsdp_training(model_ema, process_group=config.dp_group, **fsdp_args)
                ema = FSDPEmaWrapper(ema)
            model = setup_fsdp_training(model, process_group=config.dp_group, **fsdp_args)
            opt = optimizer_class(model.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay, fused=True)
        else:
            if config.hybrid_zero and (config.dp_group==None or config.dp_group==dist.group.WORLD):
                zero_group = setup_node_groups()
            else:
                zero_group = config.dp_group

            model=model.to(config.model_dtype).cuda()
            if config.use_ema:
                ema = ShardedEMA(model, config.dp_group)
            model = DDP(model, process_group=config.dp_group, gradient_as_bucket_view=True)
            opt = ZeroRedundancyOptimizer(model.parameters(),
                                        optimizer_class=optimizer_class,
                                        lr=config.learning_rate,
                                        weight_decay=config.weight_decay,
                                        process_group = zero_group,
                                        parameters_as_bucket_view=False,
                                        fused=True)
    else:
        model=model.to(config.model_dtype).cuda()
        if config.use_ema:
            ema = ShardedEMA(model, config.dp_group)
        opt = optimizer_class(model.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay, fused=True)

    return model, vae, opt, ema
