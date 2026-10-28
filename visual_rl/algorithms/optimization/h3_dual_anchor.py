"""Frozen replay scores for experimental dual SDE; no optimizer ownership."""
from __future__ import annotations

import torch
from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.trainer.h3_dual_rollout import H3DualRecord


class H3DualReplayAnchor:
    def __init__(self, *, dynamics, behavior_id, expected_keys, max_records):
        if not isinstance(dynamics,H3DualFlowSDE) or not isinstance(behavior_id,str) or not behavior_id:
            raise ValueError('dual dynamics and behavior identity required')
        if type(expected_keys) is not tuple or not expected_keys or type(max_records) is not int or max_records<1 or len(expected_keys)>max_records:
            raise ValueError('bounded explicit expected keys required')
        if len(set(expected_keys))!=len(expected_keys):
            raise ValueError('duplicate expected key')
        for sid,index in expected_keys:
            if not isinstance(sid,str) or not sid:
                raise ValueError('invalid sample identity')
            dynamics.schedule.transition(index)
        self.dynamics=dynamics;self.behavior_id=behavior_id;self.expected=expected_keys
        self.attempted=False;self._scores=None

    def validate_records(self, records):
        if type(records) is not tuple or any(type(r) is not H3DualRecord for r in records):
            raise ValueError('owned tuple of dual records required')
        if tuple((r.sample_id,r.index) for r in records)!=self.expected:
            raise ValueError('record identity/order differs from expected')

    def capture(self, records, *, device, predict, read_behavior_id, gate):
        if self.attempted: raise RuntimeError('capture already attempted')
        self.attempted=True
        self.validate_records(records)
        if not all(callable(f) for f in (predict,read_behavior_id,gate)):
            raise TypeError('predict/version/gate are required')
        pending={};device=torch.device(device)
        with torch.no_grad():
            if read_behavior_id()!=self.behavior_id: raise ValueError('behavior changed before capture')
            for ordinal,record in enumerate(records):
                gate(f'anchor/{ordinal}/forward')
                try:
                    x=record.x.to(device);action=record.action.to(device)
                except Exception:
                    gate(f'anchor/{ordinal}/prepare',local_error=True)
                    raise
                gate(f'anchor/{ordinal}/prepare',local_error=False)
                prediction=predict(record.sample_id,x,record.index)
                try:
                    joint,_,coeff=self.dynamics.evaluate(x,action,prediction,record.index)
                    pending[(record.sample_id,record.index)]=(joint.detach().cpu().clone(),tuple(v.detach().cpu().clone() for v in coeff))
                except Exception:
                    gate(f'anchor/{ordinal}/result', local_error=True)
                    raise
                gate(f'anchor/{ordinal}/result', local_error=False)
                del prediction, joint, coeff, x, action
            if read_behavior_id()!=self.behavior_id: raise ValueError('behavior changed during capture')
        self._scores=pending

    def get(self,sample_id,index):
        if self._scores is None: raise RuntimeError('anchor capture incomplete')
        score,coeff=self._scores[(sample_id,index)]
        return score.clone(),tuple(c.clone() for c in coeff)
