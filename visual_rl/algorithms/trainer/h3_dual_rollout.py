"""Pure local dual-SDE rollout; not a complete training lifecycle.

Every rank must pass the same mode/count and execute the same ordered number of
forward calls. The caller owns distributed admission, behavior version gates,
reward, credit, loss and durable checkpoint publication.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import hashlib

import torch
from visual_rl.runtime import h3_memory as memory

from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE


SELECTION_VERSION = 'h3-dual-prompt-group-first-window-v2'
FLASH_WINDOW_SIZE = 5


def selected_step(*, seed: int, optimizer_step: int, group_id: str, num_steps: int,
                  window_size: int = FLASH_WINDOW_SIZE) -> int:
    """Stateless uniform selection shared by an occurrence group, resume-safe."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be uint64')
    if type(optimizer_step) is not int or not 0 <= optimizer_step < 2**64:
        raise ValueError('optimizer_step must be uint64')
    if type(group_id) is not str or not group_id or len(group_id.encode()) > 256:
        raise ValueError('group_id must be a bounded nonempty string')
    if type(num_steps) is not int or not 2 <= num_steps <= 4096:
        raise ValueError('num_steps must be in [2,4096]')
    if type(window_size) is not int or not 1 <= window_size < num_steps:
        raise ValueError('stochastic window must be positive and exclude the terminal transition')
    prefix = (SELECTION_VERSION.encode() + b'\0' + window_size.to_bytes(2,'big') +
              num_steps.to_bytes(2,'big') + seed.to_bytes(8,'big') +
              optimizer_step.to_bytes(8,'big') + group_id.encode('utf-8') + b'\0')
    ceiling = (2**256 // window_size) * window_size
    for counter in range(100):
        value = int.from_bytes(hashlib.sha256(prefix+counter.to_bytes(4,'big')).digest(),'big')
        if value < ceiling:
            return value % window_size
    raise RuntimeError('selection rejection limit exceeded')


@dataclass(frozen=True)
class H3DualRecord:
    sample_id: str
    index: int
    x: torch.Tensor
    action: torch.Tensor
    old_joint: torch.Tensor | None = None

    def __post_init__(self):
        if not isinstance(self.sample_id,str) or not self.sample_id or type(self.index) is not int:
            raise ValueError('explicit sample identity/index required')
        for tensor in (self.x,self.action):
            if not isinstance(tensor,torch.Tensor) or tensor.device.type != 'cpu' or tensor.requires_grad or tensor.dtype != torch.float16 or not bool(torch.isfinite(tensor).all()):
                raise ValueError('record requires finite detached CPU FP16 tensors')
        if self.x.shape != self.action.shape or self.x.shape[0] != 1:
            raise ValueError('record requires matching batch-one states')
        if self.old_joint is not None:
            score=self.old_joint
            if (not isinstance(score,torch.Tensor) or score.numel()!=1 or score.device.type!='cpu'
                    or score.dtype!=torch.float32 or score.requires_grad or not bool(torch.isfinite(score).all())):
                raise ValueError('stored rollout score must be detached finite CPU FP32 scalar')
            object.__setattr__(self,'old_joint',score.detach().clone())
        object.__setattr__(self,'x',self.x.detach().clone())
        object.__setattr__(self,'action',self.action.detach().clone())


def collect(*, dynamics: H3DualFlowSDE, samples, initial_states, generators,
            mode: str, seed: int, optimizer_step: int, device, predict, read_behavior_id,
            behavior_id: str, gate, window_size: int = FLASH_WINDOW_SIZE, cell_scope=None):
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
    if cell_scope is not None and not callable(cell_scope):
        raise TypeError('optional rollout cell scope must be callable')
    device=torch.device(device)
    records=[]; terminals={}; choices={}
    with torch.no_grad():
        if read_behavior_id()!=behavior_id:
            raise ValueError('behavior version changed before rollout')
        for row,(sid,gid) in enumerate(samples):
            if not isinstance(sid,str) or not sid:
                raise ValueError('nonempty sample identity required')
            try:
                choice=(selected_step(seed=seed,optimizer_step=optimizer_step,group_id=gid,
                                      num_steps=dynamics.schedule.num_steps,window_size=window_size)
                        if mode=='flash' else None)
                choices[sid]=choice
                x=initial_states[sid]
                if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float16 or x.requires_grad or x.shape[0]!=1:
                    raise ValueError('initial state must be detached CPU FP16 batch-one')
                dynamics.layout.unpack(x)
                x=x.to(device)
                pair=generators[sid]
                if type(pair) is not tuple or len(pair)!=2:
                    raise ValueError('two per-sample generators required')
            except Exception:
                gate(f'rollout/{row}/prepare',local_error=True)
                raise
            gate(f'rollout/{row}/prepare',local_error=False)
            for index in range(dynamics.schedule.num_steps):
                scope=(nullcontext() if cell_scope is None else cell_scope(
                    optimizer_step=optimizer_step,row=row,index=index,sample_id=sid,
                    selected_index=choice,gate=gate))
                with scope:
                    gate(f'rollout/{row}/{index}/forward')
                    with memory.span('rollout_forward',synchronize=True,sample_id=sid,selected_index=choice,transition_index=index):
                        prediction=predict(sid,x,index)
                    stochastic=mode=='flow' or index==choice
                    try:
                        with memory.span('rollout_transition',synchronize=True,sample_id=sid,selected_index=choice,transition_index=index,stochastic=stochastic):
                            if stochastic:
                                action=dynamics.sample(x,prediction,index,video_generator=pair[0],audio_generator=pair[1])
                                old=(dynamics.evaluate(x,action,prediction,index)[0].detach().cpu()
                                     if mode=='flash' else None)
                                records.append(H3DualRecord(sid,index,x.cpu(),action.cpu(),old))
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
