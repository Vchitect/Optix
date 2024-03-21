import types

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .shardedema import ShardedEMA
from .op_replace import replace_all_layernorms, replace_all_groupnorms
from .utils import setup_node_groups, setup_distributed

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
        'xformer': True,                    # [PERF] use xformer can speedup a little bit
        'fusedln': True,                    # [PERF] use fusedln can speedup
        'compile_model': False,              # [PERF] this function is not stable so OFF by default
        'vae_channels_last': True,          # [PERF] use channels_last format for vae
        'optim': 'adamw',                   # the optimizer type
        'learning_rate': 1e-5,              # optimizer params
        'weight_decay': 0,                  # optimizer params
        'hybrid_zero': True,                # [PERF] for multi node training, hybrid zero can be faster
        # 'model_channels_last': False,
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


def compile(model, vae, config=None, **kwargs):
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

    # model optimizations
    model.to(device='cuda')
    if config.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()

    if config.xformer and hasattr(model, 'enable_xformers_memory_efficient_attention'):
        model.enable_xformers_memory_efficient_attention()

    if config.fusedln:
        model = replace_all_layernorms(model).to(device='cuda')
        # model = replace_all_groupnorms(model).to(device='cuda')

    if config.ddp and not isinstance(model, DDP) and dist.is_initialized() and dist.get_world_size()>1:
        model = DDP(model, process_group=config.dp_group, gradient_as_bucket_view=True)

    if config.compile_model:
        model = torch.compile(model)

    # if config.model_channels_last:
    #     model = model.to(memory_format=torch.channels_last)

    # optimizer setup
    optimizer_class=get_optimizer(config.optim)
    if config.hybrid_zero:
        if config.dp_group==None:
            node_group = setup_node_groups()
        else:
            node_group = config.dp_group
        opt = ZeroRedundancyOptimizer(model.parameters(),
                                      optimizer_class=optimizer_class,
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay,
                                      process_group = node_group,
                                      parameters_as_bucket_view=False,
                                      fused=True)
    else:
        opt = optimizer_class(model.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay, fused=True)

    # ema config
    if config.use_ema:
        ema = ShardedEMA(model, config.dp_group)
    else:
        ema = None

    return model, vae, opt, ema