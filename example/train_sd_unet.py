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

import optix


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train(model, vae, batchsize=1, use_amp=True, h=512, w=512, is_xl=False,
          use_optix=False):
    dt=torch.float32

    if not is_xl:
        timesteps = torch.arange(batchsize, dtype=torch.int64).cuda()+100
        encoder_hidden_states = torch.rand([batchsize,77,768], dtype=dt).cuda()
        # encoder_hidden_states = torch.rand([batchsize,77,1024], dtype=dt).cuda()

    else:
        timesteps = torch.arange(batchsize, dtype=torch.int64).cuda()+100
        prompt_embeds = torch.rand([batchsize,77,2048], dtype=dt).cuda()
        time_ids = torch.rand([batchsize,6], dtype=dt).cuda()
        text_embeds = torch.rand([batchsize,1280], dtype=dt).cuda()
        unet_added_conditions = {
            "time_ids": time_ids,
            "text_embeds": text_embeds
        }

    model.cuda()

    if use_optix:
        model, vae, opt, _ = optix.compile(model, vae, compile_vae=True)
    else:
        model.enable_gradient_checkpointing()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        model = DDP(model)
    perf_times = []

    for ind in range(8):
        model_input = torch.rand([batchsize, 3, h, w], dtype=torch.float32).cuda()
        torch.cuda.synchronize()
        beg = time.time()

        if not use_optix:
            with torch.no_grad():
                noisy_model_input = vae.encode(model_input).latent_dist.sample().mul_(0.18215)
        else:
            noisy_model_input = optix.sliced_vae(vae, model_input, use_autocast=True, nhwc=True)

        with torch.autocast(dtype=torch.float16, device_type='cuda', enabled=use_amp):
            if is_xl:
                model_pred = model(
                            noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                        ).sample
                loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")
            else:
                model_pred = model(noisy_model_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")

        loss.backward()

        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        if ind>4:
           perf_times.append(time.time()-beg)
        beg=time.time()
    print("max mem", torch.cuda.max_memory_allocated()/1e9)
    print(perf_times)

enable_tf32()
rank, world_size, port, addr=optix.utils.setup_distributed()

pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"

# pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
).cuda()
unet.train()
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").cuda()


train(unet, vae, batchsize=4,
      use_amp=False, h=576, w=1024, is_xl ='xl' in pretrained_model_name_or_path,
      use_optix=True)