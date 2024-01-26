import types

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .shardedema import ShardedEMA
from .op_replace import replace_all_layernorms, replace_all_groupnorms
from .utils import setup_node_groups, setup_distributed, enable_tf32
from .sliced_vae import sliced_vae

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
        'ddp': True,                        # automatically create a ddp module over unet
        'dp_group': None,                   # ddp communication group, default is None
        'gradient_checkpointing': True,     # [PERF] grad_ckpt is ON by default; for small batchsize this can be turned off for speedup
        'xformer': True,                    # [PERF] use xformer can speedup a little bit
        'fusedln': True,                    # [PERF] use fusedln can speedup
        'compile_unet': False,              # [PERF] this function is not stable so OFF by default
        'vae_channels_last': True,          # [PERF] use channels_last format for vae
        'optim': 'adamw',                   # the optimizer type
        'learning_rate': 1e-5,              # optimizer params
        'weight_decay': 0,                  # optimizer params
        'hybrid_zero': True,                # [PERF] for multi node training, hybrid zero can be faster
        # 'unet_channels_last': False,
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


def compile(unet, vae, config=None, **kwargs):
    # get a default config if None
    config = get_valid_cfg(config, **kwargs)
    print("Optimization config:", config)
    torch._dynamo.config.suppress_errors = True

    # distribute setup
    if config.ddp and not dist.is_initialized():
        setup_distributed()

    # vae optimization
    vae.requires_grad_(False)
    vae.to(device='cuda', dtype=torch.float32)
    if config.vae_channels_last:
        vae.encoder = vae.encoder.to(memory_format=torch.channels_last)
        # vae.encoder = replace_all_groupnorms(vae.encoder).to(device='cuda')


    if config.compile_vae:
        vae.encoder = torch.compile(vae.encoder)

    # unet optimizations
    unet.to(device='cuda')
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if config.xformer:
        unet.enable_xformers_memory_efficient_attention()

    if config.fusedln:
        unet = replace_all_layernorms(unet).to(device='cuda')
        # unet = replace_all_groupnorms(unet).to(device='cuda')

    if config.ddp and not isinstance(unet, DDP) and dist.is_initialized() and dist.get_world_size()>1:
        unet = DDP(unet, process_group=config.dp_group, gradient_as_bucket_view=True)

    if config.compile_unet:
        unet = torch.compile(unet)

    # if config.unet_channels_last:
    #     unet = unet.to(memory_format=torch.channels_last)

    # optimizer setup
    optimizer_class=get_optimizer(config.optim)
    if config.hybrid_zero:
        node_group = setup_node_groups()
        opt = ZeroRedundancyOptimizer(unet.parameters(),
                                      optimizer_class=optimizer_class,
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay,
                                      process_group = node_group,
                                      parameters_as_bucket_view=False,
                                      fused=True)
    else:
        opt = optimizer_class(unet.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay, fused=True)

    # ema config
    if config.use_ema:
        ema = ShardedEMA(unet)
    else:
        ema = None

    return unet, vae, opt, ema