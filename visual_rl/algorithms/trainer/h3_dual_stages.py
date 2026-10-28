"""Experimental dual-SDE stages atop the established H3 lifecycle.

Flow and Flash use the same Gaussian dynamics and joint score; they differ in
policy step selection and per-stream Flash gradient rectification. This is an
H3 extension, not a claim of official Flash mathematical equivalence.
"""
from __future__ import annotations

import torch

from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.optimization.h3_dual_anchor import H3DualReplayAnchor
from visual_rl.algorithms.optimization.h3_dual_credit import global_stream_means, policy_loss
from visual_rl.algorithms.optimization.h3_world import h3_slot_loss_scale
from visual_rl.algorithms.trainer.h3_dual_rollout import collect
from visual_rl.algorithms.trainer.h3_stages import H3IterationStages, H3UpdateResult, _checked
from visual_rl.algorithms.trainer.interface import StageValue


class H3DualIterationStages(H3IterationStages):
    def __init__(self, *, mode, seed, **kwargs):
        if mode not in ('dual_flow','dual_flash') or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('dual mode and uint63 seed required')
        self.mode=mode
        self.seed=seed
        super().__init__(**kwargs)

    def build(self, optimizer_step):
        value=super().build(optimizer_step)
        ctx=self._context(value)
        # Prevent mismatched modes/seeds before any rank enters model forward.
        ctx.coordinator.gate(f'dual-contract/{self.mode}/{self.seed}')
        ctx.dual=H3DualFlowSDE(ctx.request.dynamics.layout,ctx.request.dynamics.schedule)
        ctx.dual_records=None
        ctx.dual_terminals=None
        return value

    def rollout(self, value):
        ctx=self._context(value);r=ctx.request;c=ctx.coordinator
        _checked(c,'dual/rollout/prepare',self.prepared.prepared_root.eval)
        records,terminals,selected=collect(dynamics=ctx.dual,
            samples=tuple((sid,gid) for sid,gid,_ in r.samples),initial_states=r.initial_states,
            generators=r.generators,mode=('flow' if self.mode=='dual_flow' else 'flash'),
            seed=self.seed,optimizer_step=r.identity.optimizer_step,device=self.device,
            predict=lambda sid,x,index:self._predict(ctx,sid,x,index),
            read_behavior_id=lambda:self._version(ctx,'dual/rollout/version'),
            behavior_id=r.behavior_id,gate=lambda event,**kw:c.gate('dual/'+event,**kw))
        ctx.dual_records,ctx.dual_terminals,ctx.dual_selection=records,terminals,selected
        c.gate('dual/rollout/complete')
        return StageValue(value.identity,ctx)

    def reward(self,value):
        ctx=self._context(value);r=ctx.request
        def work():
            import math
            rows=[]
            for sid,gid,member in r.samples:
                terminal=ctx.dual_terminals[sid].clone()
                with self.decode(terminal) as media:
                    score=self.score_one(sample_id=sid,prompt=r.prompts[sid],media=media)
                    if type(score) is not float or not math.isfinite(score):
                        raise ValueError('reward must be a finite Python float')
                rows.append((sid,gid,member,score,True))
            return tuple(rows)
        ctx.rewards=ctx.coordinator.aux_turns(work)
        return StageValue(value.identity,ctx)

    def credit(self,value):
        ctx=self._context(value)
        def validate():
            if not ctx.dual_records or not set(sid for sid,_,_ in ctx.request.samples).issubset(ctx.advantages):
                raise ValueError('dual records and complete advantages required')
        _checked(ctx.coordinator,'dual/credit/ready',validate)
        return StageValue(value.identity,ctx)

    def optimize(self,value):
        ctx=self._context(value);r=ctx.request;c=ctx.coordinator;records=ctx.dual_records
        keys=tuple((record.sample_id,record.index) for record in records)
        anchor=_checked(c,'dual/anchor/prepare',lambda:H3DualReplayAnchor(dynamics=ctx.dual,
            behavior_id=r.behavior_id,expected_keys=keys,max_records=r.max_cells))
        anchor.capture(records,device=self.device,predict=lambda sid,x,index:self._predict(ctx,sid,x,index),
            read_behavior_id=lambda:self._version(ctx,'dual/anchor/version'),gate=lambda event,**kw:c.gate('dual/'+event,**kw))
        c.gate('dual/anchor/complete')
        means=None
        if self.mode=='dual_flash':
            coefficients=_checked(c,'dual/coefficients/prepare',lambda:tuple(anchor.get(rec.sample_id,rec.index)[1] for rec in records))
            means=global_stream_means(coefficients,process_group=self.process_group,device=self.device)
        optimizer=self.prepared.optimizer
        _checked(c,'dual/current/prepare',lambda:optimizer.zero_grad(set_to_none=True))
        policy_steps=r.dynamics.schedule.num_steps if self.mode=='dual_flow' else 1
        global_count=len(ctx.plan.samples)*policy_steps
        scale=h3_slot_loss_scale(n_slot=1,global_count=global_count,reducer_divisor=self.prepared.reducer_divisor)
        total=0.
        for ordinal,rec in enumerate(records):
            c.gate(f'dual/current/{ordinal}/forward')
            prediction=self._predict(ctx,rec.sample_id,rec.x.to(self.device),rec.index)
            def local_loss():
                old,_=anchor.get(rec.sample_id,rec.index)
                return policy_loss(dynamics=ctx.dual,record=rec,prediction=prediction,old_joint=old,
                    advantage=ctx.advantages[rec.sample_id],mode=('flow' if self.mode=='dual_flow' else 'flash'),
                    world_means=means,clip_range=.005,scale=scale)
            loss=_checked(c,f'dual/current/{ordinal}/backward',local_loss)
            total+=loss.detach().item();loss.backward()
            del loss,prediction
        self._version(ctx,'dual/current/complete')
        norm=c.clip_grad_norm(self.prepared,max_norm=1.)
        def update():
            optimizer.step()
            if self.prepared.scheduler is not None:self.prepared.scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        _checked(c,'dual/optimizer/complete',update)
        return StageValue(value.identity,H3UpdateResult(r.collection_id,r.behavior_id,r.next_behavior_id,len(records),total,norm))
