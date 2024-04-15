import types

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .compile import compile
from .shardedema import ShardedEMA
from .op_replace import replace_all_layernorms, replace_all_groupnorms
from .utils import setup_node_groups, setup_distributed, enable_tf32
from .sliced_vae import sliced_vae, dp_vae
from .dp_inference import dp_fwd
from .sp_comm_utils import *
from .frame_sequence_parallel import temporal2spatial, spatial2temporal, slice_encoder_states
from .fsdp_wrappers import setup_fsdp_encoder,setup_fsdp_training

try:
    from .dist import setup_dp_sp,broadcast_input_sp
except:
    setup_dp_sp=None
    broadcast_input_sp=None

try:
    from .modules.mha import MHA
except:
    MHA=None
