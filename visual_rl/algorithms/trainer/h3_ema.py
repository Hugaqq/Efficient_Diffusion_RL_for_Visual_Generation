"""Commit-counted CPU FP32 EMA of rank-local H3 LoRA shards (never rollout)."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import math

import torch

from visual_rl.composition.config.h3_profile import H3EMAConfig

EMA_STATE_SCHEMA = 'h3.lora-ema.local-shards.v1'


def local_parameter(parameter, *, rank, process_group):
    from visual_rl.artifacts.checkpoint.h3_state import _local
    from torch.distributed.tensor import DTensor
    if not isinstance(parameter, DTensor) or parameter.device_mesh.get_rank() != rank:
        raise ValueError('EMA requires this rank\'s prepared WORLD8 DTensor')
    # CUDA 保留原 current_device 校验；只有 CPU 分布式验收显式指定设备。
    local = _local(parameter, process_group, sharded=True,
                   **({'device': 'cpu'} if parameter.device.type == 'cpu' else {}))
    shape = list(parameter.shape)
    chunk = math.ceil(shape[0] / 8)
    offset = min(rank * chunk, shape[0])
    expected = [min(chunk, shape[0]-offset), *shape[1:]]
    if list(local.shape) != expected:
        raise ValueError('EMA local Shard(0) shape/offset mismatch')
    return local, dict(shape=shape, local_shape=expected, offset=offset, dtype='torch.float32')


class H3LoRAEMA:
    def __init__(self, named_parameters, *, config, rank, process_group):
        if not isinstance(config, H3EMAConfig) or not config.enabled:
            raise ValueError('EMA owner requires an enabled H3EMAConfig')
        if type(rank) is not int or not 0 <= rank < 8 or not callable(named_parameters):
            raise ValueError('EMA requires rank 0..7 and a parameter inventory callback')
        self.config, self.rank, self.group = config, rank, process_group
        self.named_parameters = named_parameters
        self.committed_steps = self.updates = 0
        self.swapped = False
        self._ids = None
        local, self.specs = self._inventory()
        self.shadow = {name: p.detach().to(device='cpu', dtype=torch.float32).clone() for name,p in local.items()}
        self.validate()

    def _inventory(self):
        items = tuple(self.named_parameters())
        parameters = dict(items)
        if len(parameters) != len(items) or len({id(p) for p in parameters.values()}) != len(parameters):
            raise ValueError('EMA parameter names/objects must be unique')
        if not parameters or any(not p.requires_grad or p.dtype != torch.float32 for p in parameters.values()):
            raise ValueError('EMA requires nonempty FP32 LoRA masters')
        ids = {name: id(p) for name,p in parameters.items()}
        if self._ids is not None and ids != self._ids:
            raise ValueError('EMA parameter ownership changed')
        self._ids = ids
        local, specs = {}, {}
        for name, p in parameters.items():
            if '.lora_A.' not in name and '.lora_B.' not in name:
                raise ValueError('EMA may own only LoRA A/B parameters')
            local[name], specs[name] = local_parameter(p, rank=self.rank, process_group=self.group)
        if hasattr(self, 'specs') and specs != self.specs:
            raise ValueError('EMA shard metadata changed')
        return local, specs

    def validate(self, *, step=None):
        if self.swapped:
            raise RuntimeError('EMA update/checkpoint forbidden while weights are swapped')
        local, _ = self._inventory()
        if step is not None and self.committed_steps != step:
            raise ValueError('EMA counter differs from committed optimizer step')
        if (type(self.committed_steps) is not int or self.committed_steps < 0
                or type(self.updates) is not int or self.updates != self.committed_steps // self.config.update_interval):
            raise ValueError('EMA update counter mismatch')
        self._validate_tensors(self.shadow)
        # 各 shard 在同一设备，仅将合并后的有限性标量同步到 CPU。
        if not bool(torch.stack([torch.isfinite(p).all() for p in local.values()]).all()):
            raise ValueError('nonfinite live LoRA shard')
        return local

    def _validate_tensors(self, tensors):
        if type(tensors) is not dict or set(tensors) != set(self.specs):
            raise ValueError('EMA shadow key coverage mismatch')
        for name, tensor in tensors.items():
            if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu' or tensor.dtype != torch.float32
                    or list(tensor.shape) != self.specs[name]['local_shape'] or tensor.requires_grad
                    or tensor.grad_fn is not None or not bool(torch.isfinite(tensor).all())):
                raise ValueError('invalid EMA shadow dtype/shape/finite state: ' + name)

    @torch.no_grad()
    def commit(self, step):
        if type(step) is not int or step != self.committed_steps + 1:
            raise ValueError('EMA must see each committed optimizer step exactly once')
        local = self.validate()
        if step % self.config.update_interval == 0:
            decay = min(step / (9 + step), self.config.decay)
            updated = {name: shadow.lerp(local[name].detach().cpu(), 1-decay)
                       for name, shadow in self.shadow.items()}
            self._validate_tensors(updated)
            self.shadow = updated
            self.updates += 1
        self.committed_steps = step

    def state_dict(self):
        self.validate()
        return dict(schema=EMA_STATE_SCHEMA, rank=self.rank, world_size=8, config=asdict(self.config),
                    committed_steps=self.committed_steps, updates=self.updates, specs=deepcopy(self.specs),
                    shadow={name: value.clone() for name, value in self.shadow.items()})

    def load_state_dict(self, state, *, step):
        self.validate()
        if self.committed_steps != 0:
            raise ValueError('EMA restore requires a fresh owner')
        keys = {'schema', 'rank', 'world_size', 'config', 'committed_steps', 'updates', 'specs', 'shadow'}
        if (type(state) is not dict or set(state) != keys or state['schema'] != EMA_STATE_SCHEMA
                or type(state['rank']) is not int or state['rank'] != self.rank
                or type(state['world_size']) is not int or state['world_size'] != 8
                or state['config'] != asdict(self.config) or state['specs'] != self.specs
                or type(step) is not int or step <= 0 or type(state['committed_steps']) is not int
                or state['committed_steps'] != step or type(state['updates']) is not int
                or state['updates'] != step // self.config.update_interval):
            raise ValueError('EMA checkpoint schema/config/topology/counters mismatch')
        if H3EMAConfig(**state['config']) != self.config:
            raise ValueError('EMA checkpoint config mismatch')
        self._validate_tensors(state['shadow'])
        self.shadow = {name: tensor.clone() for name, tensor in state['shadow'].items()}
        self.committed_steps, self.updates = step, state['updates']

    @contextmanager
    def evaluation_weights(self):
        """所有 rank 在无在途图/训练工作的边界调用；异常时仍还原 live local shards。"""
        local = self.validate()
        if any(p.grad is not None for _, p in self.named_parameters()):
            raise RuntimeError('EMA evaluation requires cleared gradients at a committed boundary')
        saved = {name: p.detach().cpu().clone() for name,p in local.items()}
        self.swapped = True
        try:
            with torch.no_grad():
                for name,p in local.items():
                    p.copy_(self.shadow[name].to(p.device))
                yield
        finally:
            try:
                with torch.no_grad():
                    for name,p in local.items():
                        p.copy_(saved[name].to(p.device))
            finally:
                self.swapped = False
