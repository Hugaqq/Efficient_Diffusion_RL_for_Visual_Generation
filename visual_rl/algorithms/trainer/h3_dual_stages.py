"""Experimental dual-SDE stages atop the established H3 lifecycle.

Flow and Flash use the same Gaussian dynamics and joint score; they differ in
policy step selection and per-stream Flash gradient rectification. This is an
H3 extension, not a claim of official Flash mathematical equivalence.
"""
from __future__ import annotations

import torch
import functools,json,time,weakref
from contextlib import ExitStack, contextmanager, nullcontext
from visual_rl.runtime import h3_memory as memory


def _timed(method):
    @functools.wraps(method)
    def run(self,*args,**kwargs):
        started=time.monotonic()
        if args:
            value=args[0]
            step=value if type(value)is int else getattr(getattr(value,'identity',None),'optimizer_step',None)
            if step is not None:memory.set_context(step=step+1)
        try:
            with memory.span(method.__name__):return method(self,*args,**kwargs)
        finally:
            if self.rank==0:
                print(json.dumps(dict(event='h3_stage_seconds',stage=method.__name__,seconds=time.monotonic()-started)),flush=True)
    return run

from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.optimization.h3_dual_anchor import H3DualReplayAnchor
from visual_rl.algorithms.optimization.h3_dual_credit import global_stream_means, policy_loss
from visual_rl.algorithms.optimization.h3_world import h3_slot_loss_scale
from visual_rl.algorithms.trainer.h3_dual_rollout import collect
from visual_rl.algorithms.trainer.h3_stages import H3IterationStages, H3UpdateResult, _checked
from visual_rl.algorithms.trainer.interface import StageValue


class H3DualIterationStages(H3IterationStages):
    def __init__(self, *, mode, seed, reward_pipeline=False, media_bytes=None, decode_bytes=None,
                 rollout_cell_scope=None, reuse_video_vae=False, video_reuse_scope=None,
                 reuse_rollout_anchor=False, parallel_owners=False, **kwargs):
        if mode not in ('dual_flow','dual_flash') or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('dual mode and uint63 seed required')
        if type(reward_pipeline) is not bool:
            raise TypeError('reward_pipeline must be bool')
        if reward_pipeline and (mode!='dual_flash' or any(type(n) is not int or n<=0
                                                        for n in (media_bytes,decode_bytes))):
            raise ValueError('reward pipeline requires dual_flash and explicit byte budgets')
        if rollout_cell_scope is not None and not callable(rollout_cell_scope):
            raise TypeError('optional rollout cell scope must be callable')
        if type(reuse_video_vae) is not bool: raise TypeError('reuse_video_vae must be bool')
        if reuse_video_vae and (mode!='dual_flash' or not callable(video_reuse_scope)):
            raise ValueError('video reuse requires dual_flash and a binding scope factory')
        self.reuse_video_vae,self.video_reuse_scope=reuse_video_vae,video_reuse_scope
        if type(reuse_rollout_anchor) is not bool: raise TypeError('reuse_rollout_anchor must be bool')
        if reuse_rollout_anchor and mode!='dual_flash':
            raise ValueError('rollout anchor reuse requires dual_flash (flow records store no old score)')
        self.reuse_rollout_anchor=reuse_rollout_anchor
        if type(parallel_owners) is not bool: raise TypeError('parallel_owners must be bool')
        if parallel_owners and (mode!='dual_flash' or not reward_pipeline):
            raise ValueError('parallel owners require dual_flash with the reward pipeline')
        self.parallel_owners=parallel_owners
        self.rollout_cell_scope=rollout_cell_scope
        self.reward_pipeline=reward_pipeline
        self.media_bytes,self.decode_bytes=media_bytes,decode_bytes
        self.mode=mode
        self.seed=seed
        super().__init__(**kwargs)

    @_timed
    def build(self, optimizer_step):
        value=super().build(optimizer_step)
        ctx=self._context(value)
        # Prevent mismatched modes/seeds before any rank enters model forward.
        ctx.coordinator.gate(f'dual-contract/{self.mode}/{self.seed}')
        ctx.dual=_checked(ctx.coordinator,'dual/dynamics/prepare',lambda:H3DualFlowSDE(
            ctx.request.dynamics.layout,ctx.request.dynamics.schedule))
        ctx.dual_records=None
        ctx.dual_terminals=None
        return value

    @_timed
    def rollout(self, value):
        ctx=self._context(value);r=ctx.request;c=ctx.coordinator
        _checked(c,'dual/rollout/prepare',self.prepared.prepared_root.eval)
        records,terminals,selected=collect(dynamics=ctx.dual,
            samples=tuple((sid,gid) for sid,gid,_ in r.samples),initial_states=r.initial_states,
            generators=r.generators,mode=('flow' if self.mode=='dual_flow' else 'flash'),
            seed=self.seed,optimizer_step=r.identity.optimizer_step,device=self.device,
            predict=lambda sid,x,index:self._predict(ctx,sid,x,index),
            read_behavior_id=lambda:self._version(ctx,'dual/rollout/version'),
            behavior_id=r.behavior_id,gate=lambda event,**kw:c.gate('dual/'+event,**kw),
            cell_scope=self.rollout_cell_scope)
        ctx.dual_records,ctx.dual_terminals,ctx.dual_selection=records,terminals,selected
        c.gate('dual/rollout/complete')
        return StageValue(value.identity,ctx)

    @_timed
    def reward(self,value):
        ctx=self._context(value);r=ctx.request
        parallel=getattr(self,'parallel_owners',False)  # opt-in P2; default serial owner turns
        def admission():
            if (self.reward_pipeline or self.reuse_video_vae) and (self.mode!='dual_flash' or r.group_layout!='paired_4x8'):
                raise ValueError('reward pipeline/video reuse requires paired_4x8 dual_flash')
            if parallel and (not self.reward_pipeline or not callable(getattr(ctx.coordinator,'aux_concurrent',None))):
                raise ValueError('parallel owners require the reward pipeline and concurrent coordination')
        _checked(ctx.coordinator,'dual/reward-pipeline/ready',admission)
        from visual_rl.models.implementations.h3_decode_reuse import VIDEO_CAP, REUSE_VERSION
        ctx.coordinator.gate(f'dual/video-reuse/{int(self.reuse_video_vae)}/{REUSE_VERSION}/{VIDEO_CAP}')
        # Default event text is unchanged; the opt-in suffix makes a rank with a
        # different owner mode fail this gate before any auxiliary work.
        from visual_rl.algorithms.optimization.h3_world import WORLD_SIZE
        from visual_rl.algorithms.trainer.h3_reward_pipeline import MEDIA_PERMITS, MediaLedger
        from visual_rl.models.lifecycle.h3_coordination import HOST_MEDIA_CAP
        ctx.coordinator.gate(f'dual/reward-pipeline/{int(self.reward_pipeline)}/queue1/total2/'
                             f'{self.media_bytes}/{self.decode_bytes}'
                             +(f'/parallel{WORLD_SIZE}/media{HOST_MEDIA_CAP}' if parallel else ''))
        cancel=ledger=None
        if parallel:
            def host_admission():
                # All owners on the one host decode at once: admit their summed
                # budgets once, and the structural media bound before any decode.
                if WORLD_SIZE*MEDIA_PERMITS>HOST_MEDIA_CAP:
                    raise ValueError('host media cap below owners x per-owner permits')
                from visual_rl.models.implementations.h3_loading import _admit_host_memory
                _admit_host_memory(WORLD_SIZE*(self.decode_bytes+MEDIA_PERMITS*self.media_bytes))
            _checked(ctx.coordinator,'dual/parallel-owners/host-admission',host_admission)
            import threading
            cancel,ledger=threading.Event(),MediaLedger()
        def consume_work():
            import math
            from visual_rl.algorithms.trainer.h3_reward_pipeline import RewardSample, reward_pipeline
            samples=tuple(RewardSample(i,sid,gid,member,r.prompts[sid],ctx.dual_selection[sid])
                          for i,(sid,gid,member) in enumerate(r.samples))
            @contextmanager
            def decode(sample):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError('parallel owners aborted; no new decode')
                sid=sample.sample_id
                terminal=ctx.dual_terminals[sid].clone()
                started=time.monotonic()
                print(json.dumps(dict(event='h3_reward_decode_begin',sample_id=sid)),flush=True)
                with ExitStack() as stack:
                    with memory.span('vae_decode',sample_id=sid,selected_index=sample.selected_index):
                        media=stack.enter_context(self.decode(terminal))
                    print(json.dumps(dict(event='h3_reward_decode_end',sample_id=sid,
                                          seconds=time.monotonic()-started)),flush=True)
                    yield media
            def score(sample,media):
                # Called by the P1 consumer thread just before each POST.
                if cancel is not None and cancel.is_set():
                    raise RuntimeError('parallel owners aborted; no new HPS request')
                sid=sample.sample_id
                started=time.monotonic()
                print(json.dumps(dict(event='h3_reward_score_begin',sample_id=sid)),flush=True)
                with memory.span('reward_score',sample_id=sid,selected_index=sample.selected_index):
                    result=self.score_one(sample_id=sid,prompt=sample.prompt,media=media)
                if type(result) is not float or not math.isfinite(result):
                    raise ValueError('reward must be a finite Python float')
                print(json.dumps(dict(event='h3_reward_score_end',sample_id=sid,
                                      seconds=time.monotonic()-started,score=result,
                                      optimizer_step=r.identity.optimizer_step,group_id=sample.group_id,
                                      member=sample.member,stochastic_index=sample.selected_index)),flush=True)
                return result
            if self.reward_pipeline:
                if self.read_behavior_id()!=r.behavior_id:
                    raise ValueError('reward pipeline behavior changed before work')
                rows=reward_pipeline(samples,decode=decode,score=score,
                                     media_bytes=self.media_bytes,decode_bytes=self.decode_bytes,
                                     **({'ledger':ledger} if ledger is not None else {}))
                if self.read_behavior_id()!=r.behavior_id:
                    raise ValueError('reward pipeline behavior changed during work')
                if parallel:
                    # Reassemble by sample_id in the canonical request order on
                    # the worker, so a mismatch is a heartbeat failure everywhere.
                    position={sid:i for i,(sid,_,_) in enumerate(r.samples)}
                    if len(rows)!=len(position) or set(row[0] for row in rows)!=set(position):
                        raise ValueError('parallel owner rows differ from the canonical sample table')
                    rows=tuple(sorted(rows,key=lambda row:position[row[0]]))
                    if tuple(tuple(row[:3]) for row in rows)!=tuple(tuple(x) for x in r.samples):
                        raise ValueError('parallel owner rows differ from the canonical sample table')
                return rows
            rows=[]
            for sample in samples:
                with decode(sample) as media:
                    result=score(sample,media)
                rows.append((sample.sample_id,sample.group_id,sample.member,result,True))
            return tuple(rows)
        def work():
            # Entire P1 helper returns only after its consumer has joined/drained.
            with (self.video_reuse_scope() if self.reuse_video_vae else nullcontext()):
                return consume_work()
        if parallel:
            ctx.rewards=ctx.coordinator.aux_concurrent(work,cancel=cancel,
                inflight=lambda:ledger.held,media_cap=HOST_MEDIA_CAP)
        else:
            ctx.rewards=ctx.coordinator.aux_turns(work)
        return StageValue(value.identity,ctx)

    def credit(self,value):
        ctx=self._context(value)
        def validate():
            if not ctx.dual_records or not set(sid for sid,_,_ in ctx.request.samples).issubset(ctx.advantages):
                raise ValueError('dual records and complete advantages required')
        _checked(ctx.coordinator,'dual/credit/ready',validate)
        return StageValue(value.identity,ctx)

    @_timed
    def optimize(self,value):
        ctx=self._context(value);r=ctx.request;c=ctx.coordinator;records=ctx.dual_records
        keys=tuple((record.sample_id,record.index) for record in records)
        # Every rank must agree whether the anchor forward is skipped (off: 4 no-grad forwards/rank).
        c.gate(f'dual/anchor/reuse/{int(self.reuse_rollout_anchor)}')
        anchor=_checked(c,'dual/anchor/prepare',lambda:H3DualReplayAnchor(dynamics=ctx.dual,
            behavior_id=r.behavior_id,expected_keys=keys,max_records=r.max_cells,reuse_rollout=self.reuse_rollout_anchor))
        _checked(c,'dual/anchor/records',lambda:anchor.validate_records(records))
        anchor.capture(records,device=self.device,predict=lambda sid,x,index:self._predict(ctx,sid,x,index),
            read_behavior_id=lambda:self._version(ctx,'dual/anchor/version'),gate=lambda event,**kw:c.gate('dual/'+event,**kw))
        c.gate('dual/anchor/complete')
        means=None
        if self.mode=='dual_flash':
            coefficients=_checked(c,'dual/coefficients/prepare',lambda:tuple(anchor.get(rec.sample_id,rec.index)[1] for rec in records))
            means=global_stream_means(coefficients,process_group=self.process_group,device=self.device)
        optimizer=self.prepared.optimizer
        def clear_grad():
            with memory.span('zero_grad',synchronize=True):optimizer.zero_grad(set_to_none=True)
        _checked(c,'dual/current/prepare',clear_grad)
        policy_steps=r.dynamics.schedule.num_steps if self.mode=='dual_flow' else 1
        global_count=len(ctx.plan.samples)*policy_steps
        scale=h3_slot_loss_scale(n_slot=1,global_count=global_count,reducer_divisor=self.prepared.reducer_divisor)
        total=0.
        for ordinal,rec in enumerate(records):
            state=_checked(c,f'dual/current/{ordinal}/prepare',lambda:rec.x.to(self.device))
            c.gate(f'dual/current/{ordinal}/forward')
            with memory.span('current_forward',synchronize=True,sample_id=rec.sample_id,selected_index=rec.index):
                prediction=self._predict(ctx,rec.sample_id,state,rec.index)
            def local_loss():
                old,_=anchor.get(rec.sample_id,rec.index)
                def audit(delta,ratio):
                    if delta>1e-4:raise ValueError('current/old logprob differs before the optimizer update')
                    print(json.dumps(dict(event='h3_initial_policy_ratio',sample_id=rec.sample_id,
                        optimizer_step=r.identity.optimizer_step,index=rec.index,max_abs_logprob_delta=delta,ratio=ratio)),flush=True)
                return policy_loss(dynamics=ctx.dual,record=rec,prediction=prediction,old_joint=old,
                    advantage=ctx.advantages[rec.sample_id],mode=('flow' if self.mode=='dual_flow' else 'flash'),
                    world_means=means,clip_range=.005,scale=scale,audit=audit)
            loss=_checked(c,f'dual/current/{ordinal}/backward',local_loss)
            total+=loss.detach().item()
            with memory.span('backward',synchronize=True,sample_id=rec.sample_id,selected_index=rec.index):loss.backward()
            prediction_ref=weakref.ref(prediction)
            del loss,prediction,state
            memory.event('prediction_released_after_backward',sample_id=rec.sample_id,selected_index=rec.index,
                         released=prediction_ref() is None)
        self._version(ctx,'dual/current/complete')
        with memory.span('gradient_clipping',synchronize=True):
            norm=c.clip_grad_norm(self.prepared,max_norm=1.)
        def update():
            with memory.span('optimizer_step',synchronize=True):
                memory.event('optimizer_step_about_to_execute',optimizer_index=r.identity.optimizer_step)
                optimizer.step()
                memory.event('optimizer_step_executed',optimizer_index=r.identity.optimizer_step,
                    counter_min=min(float(s['step'].item()) for s in optimizer.state.values()),
                    counter_max=max(float(s['step'].item()) for s in optimizer.state.values()))
            if self.prepared.scheduler is not None:
                with memory.span('scheduler_step',synchronize=True):self.prepared.scheduler.step()
            clear_grad()
        _checked(c,'dual/optimizer/complete',update)
        return StageValue(value.identity,H3UpdateResult(r.collection_id,r.behavior_id,r.next_behavior_id,len(records),total,norm))
