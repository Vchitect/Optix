import random

import numpy as np
import torch
import torch.distributed as dist
from time import perf_counter

def partition_params(model, num_partitions, return_dict=False):
    """partitions params

    Args:
        model (list): the model
        num_partitions (int): zero dp world size
        numel_per_partition (int): max number of param cnt

    Returns:
        list: list of partitions
    """
    partitions = []
    elcnt = 0
    partition_id = 0
    numel_per_partition = sum([p.numel() for p in model.parameters()]) // num_partitions
    for ind in range(num_partitions):
        if return_dict:
            partitions.append({})
        else:
            partitions.append([])

    for name, param in model.named_parameters():
        if return_dict:
            partitions[partition_id][name] = param
        else:
            partitions[partition_id].append(param)
        elcnt+=param.numel()
        if elcnt > numel_per_partition:
            partition_id+=1
            elcnt=0
    return partitions

def sliced_run(fn, input: torch.Tensor, micro_bs):
    """slice the input into several micro batches and run forward, with no_grad

    Args:
        fn (function): forward function
        input (torch.Tensor): tensor
        micro_bs (integer): micro batchsize

    Returns:
        tensor
    """
    chunk_outs = []
    with torch.no_grad():
        chunks = input.split(micro_bs, 0)
        for chunk in chunks:
            chunk_out = fn(chunk)
            chunk_outs.append(chunk_out)

        out = torch.cat(chunk_outs, dim=0)
    return out

def replace_all_module(model, if_replace_hook, get_new_module):
    for name, module in model.named_children():
        if if_replace_hook(name, module):
            tgt_module = get_new_module(name, module)
            setattr(model, name, tgt_module)
        else:
            replace_all_module(module, if_replace_hook, get_new_module)
    return model


def setup_node_groups(num_per_node=8):
    """every node is build as a comm group

    Args:
        num_per_node (int, optional): num gpu per node. Defaults to 8.

    Returns:
        torch comm group: returns None if world size is illeagal to divide into nodes

    Usage example:
    ```
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, gradient_as_bucket_view=True)
        opt = ZeroRedundancyOptimizer(model.parameters(),
                                    optimizer_class=torch.optim.AdamW,
                                    process_group = setup_node_groups(),
                                    parameters_as_bucket_view=False, fused=True)
    ```

    """
    world_size = dist.get_world_size()
    if world_size % num_per_node != 0 or world_size <= num_per_node:
        return None
    num_nodes = world_size//num_per_node
    for node_ind in range(num_nodes):
        ranks_in_node = [r + node_ind*num_per_node for r in range(0, num_per_node)]
        new_group = dist.new_group(ranks_in_node)
        if dist.get_rank() in ranks_in_node:
            ret = new_group
            print(dist.get_rank(), ranks_in_node)
    return ret


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """

    import os
    import subprocess
    import torch
    import torch.distributed as dist

    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]

        addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            port = 54647
            os.environ["MASTER_PORT"] = str(port)
        else:
            port = int(os.environ["MASTER_PORT"])
        os.environ["WORLD_SIZE"] = str(world_size)

        local_rank = rank % num_gpus
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = rank % num_gpus

    dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"dist init done, world_size = {dist.get_world_size()}")
    return rank, world_size, port, addr

def run_once(fn):
    flg=False
    def wrapper(*args, **kwargs):
        nonlocal flg
        if not flg:
            fn(*args, **kwargs)
            flg=True
    return wrapper

@run_once
def print_once(msg):
    print(msg)

def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def print_rank0(msg, group=dist.group.WORLD):
    if dist.is_initialized():
        if dist.get_rank(group)==0:
            print(msg, flush=True)
    else:
        print(msg, flush=True)


def show_mem(msg='', enable=True):
    if not enable:
        return
    torch.cuda.synchronize()

    printout = f"{msg} ;cur_mem={torch.cuda.memory_allocated()/1024**3:.1f}," \
            f"max_mem={torch.cuda.max_memory_allocated()/1024**3:.1f}"
    print_rank0(printout)



class Timer:
    def __init__(self, msg='', rank0=True, memory=True) -> None:
        self.msg = msg
        self.rank0 = rank0
        self.memory = memory

    def __enter__(self):
        torch.cuda.synchronize()
        # show_mem(msg='before ' + self.msg, enable=self.memory)
        self.start = perf_counter()
        self.memory_begin = (torch.cuda.memory_allocated()/1024**3, torch.cuda.max_memory_allocated()/1024**3)
        return self

    def __exit__(self, type, value, traceback):
        torch.cuda.synchronize()
        self.time = perf_counter() - self.start
        cur_memory = (torch.cuda.memory_allocated()/1024**3, torch.cuda.max_memory_allocated()/1024**3)
        memory_change = (cur_memory[0] - self.memory_begin[0], cur_memory[1] - self.memory_begin[1])
        self.readout = f'{self.msg} Exec Time: {self.time:.3f} s; ' \
                       f'MemoryChange: alloc: {memory_change[0]:.1f}, max: {memory_change[1]:.1f}; '  \
                       f'Current: alloc: {cur_memory[0]:.1f}, max: {cur_memory[1]:.1f}'

        if self.rank0:
            print_rank0(self.readout)
        else:
            print(self.readout, flush=True)


def set_seed(seed: int = 1024, fix_algo=False):
    """Sets seeds for all random libraries.

    Args:
        seed (int): seed for random states
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    assert torch.cuda.is_available()
    torch.cuda.manual_seed(seed)


    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.random.manual_seed(seed)

    if fix_algo:
        torch.backends.cudnn.benchmark = False

        # deterministic algos
        # torch.use_deterministic_algorithms(True)

        # cudnn conv deterministic
        torch.backends.cudnn.deterministic = True