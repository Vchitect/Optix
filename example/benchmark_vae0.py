import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import functools

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from torchdistpackage import setup_distributed_slurm
import optix

from sfast.compilers.diffusion_pipeline_compiler import compile_vae

import types

def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train(model, vae, batchsize, use_amp=True, h=512, w=512, is_xl=False,
          ooopt=False):
    if ooopt:
        model, vae, opt, _ = optix.compile(model, vae, compile_vae=True)
        # vae.encoder = torch.compile(vae.encoder)
    else:
        config=types.SimpleNamespace(enable_jit=True,enable_cnn_optimization=True, enable_cuda_graph=True, enable_fused_linear_geglu=False,
                                     prefer_lowp_gemm=False, enable_triton=True, enable_triton_reshape=False, enable_triton_layer_norm=True,
                                     memory_format=torch.channels_last, enable_xformers=True, enable_jit_freeze=True, preserve_parameters=False,)
        vae.encoder = compile_vae(vae.encoder, config)

    perf_times = []

    for ind in range(8):
        model_input = torch.rand([batchsize, 3, h, w], dtype=torch.float32).cuda()
        torch.cuda.synchronize()
        beg = time.time()

        if not ooopt:
            with torch.no_grad():
                noisy_model_input = vae.encode(model_input).latent_dist.sample().mul_(0.18215)
        else:
            noisy_model_input = optix.sliced_vae(vae, model_input, use_autocast=False, nhwc=False)

        torch.cuda.synchronize()
        # prof.step()
        if ind>4:
           perf_times.append(time.time()-beg)
        beg=time.time()
    print("max mem", torch.cuda.max_memory_allocated()/1e9)
    print(perf_times)

enable_tf32()
rank, world_size, port, addr=setup_distributed_slurm()

pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"

# pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
).cuda()
unet.train()
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").cuda()


train(unet, vae, 8,
      use_amp=False, h=576, w=1024, is_xl ='xl' in pretrained_model_name_or_path,
      ooopt=False)