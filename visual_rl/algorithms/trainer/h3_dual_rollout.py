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
# Opt-in S1: the paired_4x8 local samples (one prompt group, one selected index)
# share one batched no-grad transformer forward per transition. Everything
# else (initial states, generators, selection, records, per-sample checks)
# is unchanged; batched bf16 GEMMs are NOT bitwise equal to B=1 forwards.
ROLLOUT_BATCH_VERSION = 'h3-rollout-paired-local-b4-v1'
ROLLOUT_BATCH = 4
# Declared (not measured for B=4) admission estimate. Basis: attempt003 rank0
# rollout_forward, 480x320x124 joint layout, B=1: allocated 9.21 GiB at
# segment begin, 12.96 GiB peak, i.e. a 3.75 GiB per-forward transient that
# already includes one block's all-gathered parameters. B=4 is charged a full
# 4 GiB per sample, so the non-scaling parameter part is over-counted.
ROLLOUT_BATCH_ESTIMATE = dict(layout=(5550, 96, 414, 32), transient_bytes_per_sample=4 * 1024**3,
                              basis='attempt003 rank0 B1 rollout_forward peak 12.96 GiB - begin 9.21 GiB')


def rollout_batch_admission(device, *, batch, layout, free_bytes=None):
    """Fail closed BEFORE a batched forward whose declared transient may not fit.

    Allocatable = driver-free + reserved-but-unallocated bytes of this device.
    Only the measured geometry has a declared estimate; CUDA work for another
    geometry is rejected rather than guessed. There is no B=1 fallback: a run
    never changes its rollout numerics mid-run. CPU without an injected free
    byte count has no device memory to admit and is recorded as unchecked.
    """
    device = torch.device(device)
    if type(batch) is not int or batch < 2:
        raise ValueError('batched admission requires an integer batch > 1')
    if device.type != 'cuda' and free_bytes is None:
        return dict(checked=False, device=str(device), batch=batch, reason='no CUDA device memory')
    shape = (layout.video_rows, layout.video_channels, layout.audio_rows, layout.audio_channels)
    if shape != ROLLOUT_BATCH_ESTIMATE['layout']:
        raise ValueError('no declared batched-rollout memory estimate for this geometry')
    required = batch * ROLLOUT_BATCH_ESTIMATE['transient_bytes_per_sample']
    if free_bytes is None:
        driver_free, _ = torch.cuda.mem_get_info(device)
        free_bytes = driver_free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    if type(free_bytes) is not int or free_bytes < 0:
        raise ValueError('free byte count must be a nonnegative integer')
    receipt = dict(checked=True, device=str(device), batch=batch, free_bytes=free_bytes,
                   required_bytes=required, basis=ROLLOUT_BATCH_ESTIMATE['basis'])
    if free_bytes < required:
        raise RuntimeError(f'batched rollout memory admission failed: {free_bytes} free < {required} declared bytes')
    return receipt


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
            behavior_id: str, gate, window_size: int = FLASH_WINDOW_SIZE, cell_scope=None,
            rollout_batch: int = 1, predict_batch=None, admit=None):
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
    if type(rollout_batch) is not int or rollout_batch not in (1, ROLLOUT_BATCH):
        raise ValueError('rollout batch must be 1 or 4')
    if rollout_batch != 1:
        if mode != 'flash' or len(samples) != rollout_batch or not callable(predict_batch) or not callable(admit):
            raise ValueError('batched rollout requires flash, exactly four local samples, batch predict and admission')
        return _collect_batched(dynamics=dynamics, samples=samples, initial_states=initial_states,
            generators=generators, seed=seed, optimizer_step=optimizer_step, device=torch.device(device),
            predict_batch=predict_batch, admit=admit, read_behavior_id=read_behavior_id,
            behavior_id=behavior_id, gate=gate, window_size=window_size, cell_scope=cell_scope)
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


def _collect_batched(*, dynamics, samples, initial_states, generators, seed, optimizer_step, device,
                     predict_batch, admit, read_behavior_id, behavior_id, gate, window_size, cell_scope):
    """S1: B=1 rollout semantics with one batched model forward per transition.

    Row preparation, per-row generators, transitions, stored records (same row
    order, B=1 CPU copies), terminals and gates on per-row results are those
    of ``collect``. The only differences are the batched forward and its
    rank-consistent admission/forward gates. One prompt group means one
    selected index; a mismatch fails before any model call.
    """
    ids = tuple(s for s,_ in samples)
    records=[]; terminals={}; choices={}; states=[]; pairs=[]
    with torch.no_grad():
        if read_behavior_id()!=behavior_id:
            raise ValueError('behavior version changed before rollout')
        for row,(sid,gid) in enumerate(samples):
            if not isinstance(sid,str) or not sid:
                raise ValueError('nonempty sample identity required')
            try:
                choice=selected_step(seed=seed,optimizer_step=optimizer_step,group_id=gid,
                                     num_steps=dynamics.schedule.num_steps,window_size=window_size)
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
            states.append(x); pairs.append(pair)
        try:
            if len({gid for _,gid in samples})!=1 or len(set(choices.values()))!=1:
                raise ValueError('batched rollout requires one prompt group and one selected index')
        except Exception:
            gate('rollout/batch/prepare',local_error=True)
            raise
        gate('rollout/batch/prepare',local_error=False)
        choice=choices[ids[0]]
        for index in range(dynamics.schedule.num_steps):
            # The single B=4 forward contains row 1, so a (row1,index) profile
            # target traces exactly that batched cell.
            scope=(nullcontext() if cell_scope is None else cell_scope(
                optimizer_step=optimizer_step,row=1,index=index,sample_id=ids[1],
                selected_index=choice,gate=gate))
            with scope:
                try:
                    receipt=admit(index)
                except Exception:
                    gate(f'rollout/batch/{index}/admission',local_error=True)
                    raise
                gate(f'rollout/batch/{index}/admission',local_error=False)
                gate(f'rollout/batch/{index}/forward')
                with memory.span('rollout_forward',synchronize=True,sample_ids=list(ids),rollout_batch=len(ids),
                                 selected_index=choice,transition_index=index,admission=receipt):
                    predictions=predict_batch(ids,torch.cat(states),index)
                try:
                    if (type(predictions) is not tuple or len(predictions)!=len(ids)
                            or any(not isinstance(p,torch.Tensor) or p.shape!=x.shape or p.device!=x.device
                                   for p,x in zip(predictions,states))):
                        raise ValueError('batched prediction must return one state-shaped tensor per sample')
                except Exception:
                    gate(f'rollout/batch/{index}/split',local_error=True)
                    raise
                gate(f'rollout/batch/{index}/split',local_error=False)
                stochastic=index==choice
                for row,sid in enumerate(ids):
                    x=states[row];prediction=predictions[row];pair=pairs[row]
                    try:
                        with memory.span('rollout_transition',synchronize=True,sample_id=sid,selected_index=choice,transition_index=index,stochastic=stochastic):
                            if stochastic:
                                action=dynamics.sample(x,prediction,index,video_generator=pair[0],audio_generator=pair[1])
                                old=dynamics.evaluate(x,action,prediction,index)[0].detach().cpu()
                                records.append(H3DualRecord(sid,index,x.cpu(),action.cpu(),old))
                            else:
                                action=dynamics.deterministic_step(x,prediction,index)
                    except Exception:
                        gate(f'rollout/{row}/{index}/result', local_error=True)
                        raise
                    gate(f'rollout/{row}/{index}/result', local_error=False)
                    states[row]=action
                    del prediction
                del predictions
        for row,sid in enumerate(ids):
            terminals[sid]=states[row].detach().cpu().clone()
        if read_behavior_id()!=behavior_id:
            raise ValueError('behavior version changed during rollout')
    return tuple(records),terminals,choices
