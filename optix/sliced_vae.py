import torch
from einops import rearrange
import functools

from .utils import sliced_run, print_once
from .dp_inference import dp_fwd

def auto_micro_bs(inp):
    # based on resolution (and maybe GPU memory too)
    if inp.is_contiguous(memory_format=torch.channels_last):
        h = inp.shape[1]
        w = inp.shape[2]
    else:
        h = inp.shape[2]
        w = inp.shape[3]
    if h>700 and w>700:
        return 2
    elif h > 500 and w>500:
        return 8
    else:
        return 16

def sliced_vae(vae, model_input, micro_bs=None, use_autocast=False, nhwc=False):
    if nhwc:
        vae.encoder.to(memory_format=torch.channels_last)
    input_dim = model_input.dim()

    with torch.no_grad():
        if input_dim == 5:
            b, _, _, _, _ = model_input.shape
            model_input = rearrange(model_input, 'b f c h w -> (b f) c h w').contiguous()
            print_once("sliced_vae received 5D input tensor, the first two dims will be treated as batch and frame")
        if nhwc:
            model_input = model_input.to(memory_format=torch.channels_last).contiguous()
        def vae_fn(chunk):
            with torch.autocast('cuda', enabled=use_autocast):
                chunk_out = vae.encode(chunk).latent_dist.sample().mul_(vae.config.scaling_factor)
            return chunk_out
        if micro_bs==None:
            micro_bs = auto_micro_bs(model_input)
            print_once(f"sliced_vae auto micro_bs={micro_bs}")
        model_input = sliced_run(vae_fn, model_input, micro_bs)
        if input_dim == 5:
            model_input = rearrange(model_input, '(b f) c h w -> b c f h w', b=b).contiguous()
    return model_input

def dp_vae(vae, model_input, use_autocast=False, nhwc=False, group=None):
    input_dim = model_input.dim()
    # import pdb;pdb.set_trace()
    with torch.no_grad():
        if input_dim == 5:
            b, _, _, _, _ = model_input.shape
            model_input = rearrange(model_input, 'b f c h w -> (b f) c h w').contiguous()
            print_once("dp_vae received 5D input tensor, the first two dims will be treated as batch and frame")

        fwd_fn = functools.partial(sliced_vae, vae, nhwc=nhwc, use_autocast=use_autocast)
        output = dp_fwd(fwd_fn, model_input, group=group)
        if input_dim == 5:
            output = rearrange(output, '(b f) c h w -> b c f h w', b=b).contiguous()
        return output
