"""Pure local dual-SDE rollout; not a complete training lifecycle.

Every rank must pass the same mode/count and execute the same ordered number of
forward calls. The caller owns distributed admission, behavior version gates,
reward, credit, loss and durable checkpoint publication.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE


SELECTION_VERSION = 'h3-dual-prompt-group-sha256-rejection-v1'


def selected_step(*, seed: int, optimizer_step: int, group_id: str, num_steps: int) -> int:
    """Stateless uniform selection shared by an occurrence group, resume-safe."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be uint64')
    if type(optimizer_step) is not int or not 0 <= optimizer_step < 2**64:
        raise ValueError('optimizer_step must be uint64')
    if type(group_id) is not str or not group_id or len(group_id.encode()) > 256:
        raise ValueError('group_id must be a bounded nonempty string')
    if type(num_steps) is not int or not 2 <= num_steps <= 4096:
        raise ValueError('num_steps must be in [2,4096]')
    prefix = (SELECTION_VERSION.encode() + b'\0' + seed.to_bytes(8,'big') +
              optimizer_step.to_bytes(8,'big') + group_id.encode('utf-8') + b'\0')
    ceiling = (2**256 // num_steps) * num_steps
    for counter in range(100):
        value = int.from_bytes(hashlib.sha256(prefix+counter.to_bytes(4,'big')).digest(),'big')
        if value < ceiling:
            return value % num_steps
    raise RuntimeError('selection rejection limit exceeded')


@dataclass(frozen=True)
class H3DualRecord:
    sample_id: str
    index: int
    x: torch.Tensor
    action: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.sample_id,str) or not self.sample_id or type(self.index) is not int:
            raise ValueError('explicit sample identity/index required')
        for tensor in (self.x,self.action):
            if not isinstance(tensor,torch.Tensor) or tensor.device.type != 'cpu' or tensor.requires_grad or tensor.dtype != torch.float16 or not bool(torch.isfinite(tensor).all()):
                raise ValueError('record requires finite detached CPU FP16 tensors')
        if self.x.shape != self.action.shape or self.x.shape[0] != 1:
            raise ValueError('record requires matching batch-one states')
        object.__setattr__(self,'x',self.x.detach().clone())
        object.__setattr__(self,'action',self.action.detach().clone())


def collect(*, dynamics: H3DualFlowSDE, samples, initial_states, generators,
            mode: str, seed: int, optimizer_step: int, device, predict, read_behavior_id,
            behavior_id: str, gate):
    """Flow: all stochastic transitions. Flash: one stochastic, rest Euler.

    Exactly one sample group stays rank-local. Flow/Flash share the same SDE
    and joint density, but differ in stochastic step count and Flash correction.
    Per-sample state/action is kept only for selected policy steps. A CPU copy
    is retained; inputs are not mutated. Requires caller-imposed memory budget.
    """
    if not isinstance(dynamics,H3DualFlowSDE) or mode not in ('flow','flash'):
        raise ValueError('explicit dual-SDE and flow/flash mode required')
    if type(samples) is not tuple or not samples or len({s for s,_ in samples}) != len(samples):
        raise ValueError('unique sample/group pairs required')
    ids = tuple(s for s,_ in samples)
    if set(initial_states)!=set(ids) or set(generators)!=set(ids):
        raise ValueError('state/generator bindings must cover sample IDs')
    if not all(callable(fn) for fn in (predict, read_behavior_id,gate)) or not behavior_id:
        raise ValueError('model, version and collective gates required')
    device=torch.device(device)
    records=[]; terminals={}; choices={}
    with torch.no_grad():
        if read_behavior_id()!=behavior_id:
            raise ValueError('behavior version changed before rollout')
        for row,(sid,gid) in enumerate(samples):
            if not isinstance(sid,str) or not sid:
                raise ValueError('nonempty sample identity required')
            choice=selected_step(seed=seed,optimizer_step=optimizer_step,group_id=gid,num_steps=dynamics.schedule.num_steps)
            choices[sid]=choice
            x=initial_states[sid]
            if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float16 or x.requires_grad or x.shape[0]!=1:
                raise ValueError('initial state must be detached CPU FP16 batch-one')
            dynamics.layout.unpack(x)
            x=x.to(device)
            pair=generators[sid]
            if type(pair) is not tuple or len(pair)!=2:
                raise ValueError('two per-sample generators required')
            for index in range(dynamics.schedule.num_steps):
                gate(f'rollout/{row}/{index}/forward')
                prediction=predict(sid,x,index)
                stochastic=mode=='flow' or index==choice
                try:
                    if stochastic:
                        action=dynamics.sample(x,prediction,index,video_generator=pair[0],audio_generator=pair[1])
                        records.append(H3DualRecord(sid,index,x.cpu(),action.cpu()))
                    else:
                        action=dynamics.deterministic_step(x,prediction,index)
                except Exception:
                    gate(f'rollout/{row}/{index}/result', local_error=True)
                    raise
                gate(f'rollout/{row}/{index}/result', local_error=False)
                x=action
                del prediction
            terminals[sid]=x.detach().cpu().clone()
        if read_behavior_id()!=behavior_id:
            raise ValueError('behavior version changed during rollout')
    return tuple(records),terminals,choices
