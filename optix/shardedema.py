from collections import OrderedDict
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from .utils import partition_params

class ShardedEMA():
    """Shard the ema params across DDP group.
    """
    def __init__(self, model, group=None) -> None:
        self.rank = dist.get_rank(group)
        self.group = group
        # divide param in buckets
        if isinstance(model, DDP):
            model = model.module
        self.all_param_shards = partition_params(model, dist.get_world_size(group), return_dict=True)
        self.param_shard = {}
        self.named_buffers = {}
        for name, p in self.all_param_shards[self.rank].items():
            self.param_shard[name] = p.clone().detach().requires_grad_(False)

    @torch.no_grad()
    def update(self, model, decay=0.9999, only_trainable=True):
        if isinstance(model, DDP):
            model = model.module
        model_params = OrderedDict(model.named_parameters())
        for name in self.param_shard.keys():
            if only_trainable and (not model_params[name].requires_grad):
                continue
            self.param_shard[name].mul_(decay).add_(model_params[name].data, alpha=1 - decay)

        self.named_buffers = model.named_buffers()

    def state_dict_shard(self):
        return self.param_shard

    def state_dict(self):
        """gather the sharded ema param across group to rank0

        Note:
            This needs to be call on all process, not only rank0!

        Returns:
            dict: the full state dict on rank0
        """
        begin = time.time()
        state_dict = OrderedDict(self.named_buffers)
        for k in state_dict.keys():
            state_dict[k] = state_dict[k].cpu()
        for name, val in self.param_shard.items():
            if self.rank==0:
                state_dict[name] = val.cpu()

        for rank in range(1, len(self.all_param_shards)):
            # send from rank to rank0
            params_in_cur_rank = self.all_param_shards[rank]
            for param_name in params_in_cur_rank.keys():
                if self.rank == rank:
                    src_p = self.param_shard[param_name]
                    dist.send(src_p.cuda(), 0, self.group)
                if self.rank == 0:
                    recv_buffer = torch.empty_like(self.all_param_shards[rank][param_name])
                    dist.recv(recv_buffer, rank, self.group)
                    state_dict[param_name] = recv_buffer.cpu()
                dist.barrier()

        print(f"ShardedEMA state_dict time cost: {time.time()-begin} s")
        return state_dict

    def verify_with_gt(self, gt_ema):
        sd_ema_params = self.state_dict()
        if torch.distributed.get_rank()==0:
            for name, gt_param in gt_ema.named_parameters():
                sd_param = sd_ema_params[name]
                if not torch.equal(gt_param.cpu(), sd_param):
                    print(f"param {name} not equal, diff: ", (gt_param.cpu()-sd_param).sum())
                assert torch.equal(gt_param.cpu(), sd_param)


class FSDPEmaWrapper():
    def __init__(self, ema) -> None:
        self.fsdp_ema = ema

    @torch.no_grad()
    def update(self, model, decay=0.9999):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(self.fsdp_ema.named_parameters())
        model_params = OrderedDict(model.named_parameters())
        assert set(ema_params.keys()) == set(model_params.keys())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


    def state_dict(self):
        """get full state dict on rank0 cpu memory

        Note:
            This need to be called on all process!
        """
        with FSDP.state_dict_type(
            self.fsdp_ema,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            consolidated_ema_state_dict = self.fsdp_ema.state_dict()
        return consolidated_ema_state_dict

