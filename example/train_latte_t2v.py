# borrowed from https://github.com/Vchitect/Latte to demostrate the usage of Optix

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
import math
import argparse

import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets import get_dataset
from diffusion import create_diffusion
from omegaconf import OmegaConf
from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as Data
from transformers import AutoTokenizer, PretrainedConfig
from transformers import T5EncoderModel, T5Tokenizer
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import numpy as np
import random
from utils import (clip_grad_norm_, create_logger, update_ema,
                   requires_grad, cleanup, create_tensorboard,
                   write_tensorboard, setup_distributed,
                   get_experiment_dir, fetch_files_by_numbers, setup_node_groups,
                   text_preprocessing, )
import re

# read data from ceph
from petrel_client.client import Client

import optix

conf_path = '~/petreloss.conf'
client = Client(conf_path) # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文


from torch.utils.data import DataLoader, Dataset

class DummyClsDataset(Dataset):
    def __init__(self, shape, num_samples=1000):
        self.num_samples = num_samples
        self.shape = shape
        # self.vae = vae

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        del idx
        img = torch.randn(self.shape)
        return {'model_input': img, 'video_prompts': "a video of a panda eating pizza"}

def low_mem_vae(vae, model_input, micro_bs=2):
    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        b, _, _, _, _ = model_input.shape
        chunk_outs = []
        model_input = rearrange(model_input, 'b f c h w -> (b f) c h w').contiguous()
        chunks = model_input.chunk(micro_bs, 0)
        for chunk in chunks:
            chunk_out = vae.encode(chunk).latent_dist.sample().mul_(vae.config.scaling_factor)
            chunk_outs.append(chunk_out)

        model_input = torch.cat(chunk_outs, dim=0)
        model_input = rearrange(model_input, '(b f) c h w -> b c f h w', b=b).contiguous() # for tav unet; b c f h w is for conv3d
    return model_input
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    setup_distributed()
    ntask_per_node = int(os.environ['SLURM_NTASKS']) // int(os.environ['SLURM_NNODES'])
    node_group = setup_node_groups(num_per_node=ntask_per_node)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    args.global_seed = int(random.randint(1, 6666))
    seed = args.global_seed + rank # important
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.num_frames)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    # assert args.image_size[0] % 8 == 0 and args.image_size[1] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    model = get_models(args)
    # Note that parameter initialization is done within the DiT constructor
    if args.engine!='optix':
        if args.use_ema:
            ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
            requires_grad(ema, False)
            logger.info("Using ema mode")
        else:
            logger.info("No using ema mode!")

    # Load the tokenizers
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")

    # Load T5
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")

    # Build train diffusion
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    # Load vae
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")

    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.precision == "fp16":
        weight_dtype = torch.float16
    elif args.precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(device, dtype=torch.float32)
    text_encoder.to(weight_dtype)
    if args.fsdp_encoder:
        text_encoder = optix.setup_fsdp_encoder(text_encoder,policy='lambda', process_group=node_group)
    else:
        text_encoder.to(device)

    train_steps = 0

    if args.use_compile:
        model = torch.compile(model)

    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing!")
        model.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        logger.info("Using Xformers!")
        model.enable_xformers_memory_efficient_attention()

    if args.engine!='optix':
        model = model.to(weight_dtype)
        model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=True)
    else:
        model, vae, opt, ema = optix.compile(model, vae, use_ema=args.use_ema,
                                             dp_group=dist.group.WORLD, model_dtype=weight_dtype,
                                             learning_rate=args.learning_rate, weight_decay=args.weight_decay,)


    logger.info(f"T5 Parameters: {sum(p.numel() for p in text_encoder.parameters()):,}")
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("Loading pretrained stable diffusion models at {}!".format(args.pretrained_model_path))

    # train_dataset = get_dataset(args)
    train_dataset = DummyClsDataset([args.num_frames, 3, args.image_size[0], args.image_size[1]])

    sampler = DistributedSampler(
            train_dataset,
            num_replicas=1, # important
            rank=0, # important
            shuffle=True,
            seed=args.global_seed
        )

    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=int(args.local_batch_size),
                                   shuffle=False,
                                   sampler=sampler,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True)

    lr_scheduler = get_scheduler(
        name='constant',
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    if args.use_ema:
        if args.engine=='optix':
            ema.update(model, decay=0)
        else:
            update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
            ema.eval()  # EMA model should always be in eval mode
    model.train()  # important! This enables embedding dropout for classifier-free guidance


    # Variables for monitoring/logging purposes:
    # train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # training
    amp_enable = True if args.mixed_precision else False
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(train_loader):

            model_input = video_data['model_input'].to(device, non_blocking=True)
            if args.use_image_num != 0:
                video_image_prompts = video_data['video_image_prompts']
            else:
                video_prompts = video_data['video_prompts']

            if args.engine=='optix':
                model_input = optix.sliced_vae(vae, model_input)
            else:
                model_input = low_mem_vae(vae, model_input, micro_bs=4)
            model_input = model_input.to(weight_dtype)
            torch.cuda.synchronize()

            bsz = model_input.shape[0]
            timesteps = torch.randint(0, diffusion.num_timesteps, (bsz,), device=model_input.device).long()
            if args.use_image_num != 0:
                encoder_attention_mask_batch_list = []
                prompt_embeds_batch_list = []
                for prompt in video_image_prompts:
                    prompt = text_preprocessing(prompt)
                    drop_ids = np.random.uniform(0, 1, len(prompt)) < 0.1
                    captions = list(np.where(drop_ids, "", prompt))
                    text_inputs = tokenizer(
                                        captions,
                                        max_length=120,
                                        padding='max_length',
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True,
                                        return_tensors='pt'
                    )
                    text_input_ids = text_inputs.input_ids

                    encoder_attention_mask = text_inputs.attention_mask.to(device)
                    encoder_attention_mask_batch_list.append(encoder_attention_mask.unsqueeze(0))

                    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=encoder_attention_mask)
                    prompt_embeds = prompt_embeds[0].to(device=device)
                    prompt_embeds_batch_list.append(prompt_embeds.unsqueeze(0))

                encoder_attention_mask = torch.cat(encoder_attention_mask_batch_list, dim=0)
                prompt_embeds = torch.cat(prompt_embeds_batch_list, dim=0)

            else:
                max_length = 120
                video_prompts = text_preprocessing(video_prompts)
                drop_ids = np.random.uniform(0, 1, len(video_prompts)) < 0.1
                captions = list(np.where(drop_ids, "", video_prompts))
                with torch.no_grad():
                    text_inputs = tokenizer(captions,
                                            max_length=120,
                                            padding='max_length',
                                            truncation=True,
                                            return_attention_mask=True,
                                            add_special_tokens=True,
                                            return_tensors='pt'
                    )
                    text_input_ids = text_inputs.input_ids
                    encoder_attention_mask = text_inputs.attention_mask.to(device)
                    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=encoder_attention_mask)
                    prompt_embeds = prompt_embeds[0].to(device=device)


            # Predict the noise residual
            added_cond_kwargs = added_cond_kwargs = {"resolution": 1, "aspect_ratio": 1}

            if args.model == 'Transformer3DModelV1' and args.use_image_num != 0:
                attention_mask = torch.zeros(args.num_frames + args.use_image_num, args.num_frames + args.use_image_num).to(device)
                attention_mask.fill_diagonal_(1)
                attention_mask[:args.num_frames, :args.num_frames] = 1
                attention_mask = attention_mask.unsqueeze(0).repeat_interleave(bsz, dim=0)
            else:
                attention_mask = None


            with torch.cuda.amp.autocast(enabled=amp_enable):
                model_kwargs=dict(encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs,
                                encoder_attention_mask=encoder_attention_mask, use_image_num=args.use_image_num,
                                attention_mask=attention_mask,)
                loss_term = diffusion.training_losses(model, model_input, timesteps, model_kwargs)
                loss = loss_term['loss'].mean()

            if args.mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if train_steps < 10000000:
                    gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=False)
                else:
                    gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=True)
                scaler.step(opt)
                scaler.update()
                lr_scheduler.step()
                opt.zero_grad()
                if args.use_ema:
                    if args.engine=='optix':
                        ema.update(model, decay=0.9999)
                    else:
                        update_ema(ema, model.module, decay=0.9999)
                torch.cuda.synchronize()
            else:
                loss.backward()
                if train_steps < 10000000:
                    gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=False)
                else:
                    gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=True)
                opt.step()

                lr_scheduler.step()
                opt.zero_grad()
                if args.use_ema:
                    if args.engine=='optix':
                        ema.update(model, decay=0.9999)
                    else:
                        update_ema(ema, model.module, decay=0.9999)
                torch.cuda.synchronize()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.3f}, lr: {lr_scheduler.get_last_lr()[0]:.6f},"
                            f"mem={torch.cuda.max_memory_allocated()/1024**3}")
                write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm.float(), train_steps)
                write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if args.engine == 'optix':
                    ema_states = ema.state_dict()
                else:
                    ema_states = ema.state_dict()
                if rank == 0:
                    if args.use_ema:
                        checkpoint = {
                            # "model": model.module.state_dict() if not args.deepspeed else model.state_dict(),
                            "ema": ema_states,
                            # "opt": opt.state_dict(),
                            # "args": args
                        }
                    else:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            # "ema": ema.state_dict(),
                            # "opt": opt.state_dict(),
                            # "args": args
                        }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))