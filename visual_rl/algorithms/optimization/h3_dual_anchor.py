"""Frozen replay scores for experimental dual SDE; no optimizer ownership."""
from __future__ import annotations

import torch
from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.trainer.h3_dual_rollout import H3DualRecord


# Observation-only S1 threshold: the unchanged B=1 bit-replay limit. This is
# deliberately not a run-identity field or a replacement for the declared gate.
ROLLOUT_REPLAY_WARN_THRESHOLD = 1e-4


class H3DualReplayAnchor:
    def __init__(self, *, dynamics, behavior_id, expected_keys, max_records, reuse_rollout=False,
                 batched_rollout_tolerance=None):
        if not isinstance(dynamics,H3DualFlowSDE) or not isinstance(behavior_id,str) or not behavior_id:
            raise ValueError('dual dynamics and behavior identity required')
        if type(reuse_rollout) is not bool: raise TypeError('reuse_rollout must be bool')
        # S1: records came from a batched rollout forward. The anchor stays a B=1
        # forward (the ratio's old score has current-forward numerics); the
        # rollout-vs-anchor replay delta is checked against the run's DECLARED
        # tolerance instead of the B=1 bit-replay limit 1e-4.
        if batched_rollout_tolerance is not None:
            import math
            if (type(batched_rollout_tolerance) is not float or not math.isfinite(batched_rollout_tolerance)
                    or not 0 < batched_rollout_tolerance <= 1e-2):
                raise ValueError('batched rollout replay tolerance must be a float in (0, 1e-2]')
            if reuse_rollout:
                raise ValueError('batched rollout requires the B=1 anchor forward; rollout anchor reuse is rejected')
        self.batched_rollout_tolerance=batched_rollout_tolerance
        if type(expected_keys) is not tuple or not expected_keys or type(max_records) is not int or max_records<1 or len(expected_keys)>max_records:
            raise ValueError('bounded explicit expected keys required')
        if len(set(expected_keys))!=len(expected_keys):
            raise ValueError('duplicate expected key')
        for sid,index in expected_keys:
            if not isinstance(sid,str) or not sid:
                raise ValueError('invalid sample identity')
            dynamics.schedule.transition(index)
        self.dynamics=dynamics;self.behavior_id=behavior_id;self.expected=expected_keys
        self.reuse_rollout=reuse_rollout
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
        summary=(None if self.batched_rollout_tolerance is None else dict(
            n_checks=0,n_warn=0,max_abs_delta=0.,warn_threshold=ROLLOUT_REPLAY_WARN_THRESHOLD,
            declared_tolerance=self.batched_rollout_tolerance))
        with torch.no_grad():
            if read_behavior_id()!=self.behavior_id: raise ValueError('behavior changed before capture')
            for ordinal,record in enumerate(records):
                if self.reuse_rollout:
                    # Opt-in: the rollout already scored this exact (x, action) under the
                    # same frozen behavior weights, and the Flash coefficient is a pure
                    # schedule function, so no second no-grad forward is needed.
                    try:
                        if record.old_joint is None:
                            raise ValueError('rollout-time old score missing; cannot reuse anchor')
                        coeff=self.dynamics.coefficients(record.index,device=device)
                        import json
                        print(json.dumps(dict(event='h3_rollout_anchor_reused',sample_id=record.sample_id,
                                              index=record.index)),flush=True)
                        pending[(record.sample_id,record.index)]=(record.old_joint.clone(),tuple(v.detach().cpu().clone() for v in coeff))
                    except Exception:
                        gate(f'anchor/{ordinal}/reuse',local_error=True)
                        raise
                    gate(f'anchor/{ordinal}/reuse',local_error=False)
                    continue
                gate(f'anchor/{ordinal}/forward')
                try:
                    x=record.x.to(device);action=record.action.to(device)
                except Exception:
                    gate(f'anchor/{ordinal}/prepare',local_error=True)
                    raise
                gate(f'anchor/{ordinal}/prepare',local_error=False)
                from visual_rl.runtime import h3_memory as memory
                with memory.span('anchor_forward',synchronize=True,sample_id=record.sample_id,selected_index=record.index):
                    prediction=predict(record.sample_id,x,record.index)
                try:
                    joint,_,coeff=self.dynamics.evaluate(x,action,prediction,record.index)
                    if record.old_joint is not None:
                        import json
                        delta=float((joint.detach().cpu()-record.old_joint).abs().max())
                        batched=self.batched_rollout_tolerance
                        if batched is None:
                            if delta>1e-4:raise ValueError('sampled/replayed old score mismatch')
                            print(json.dumps(dict(event='h3_rollout_replay_score_check',sample_id=record.sample_id,
                                                  index=record.index,max_abs_delta=delta)),flush=True)
                        else:
                            print(json.dumps(dict(event='h3_rollout_replay_score_check',sample_id=record.sample_id,
                                                  index=record.index,max_abs_delta=delta,rollout_batch=4,
                                                  declared_tolerance=batched,old_score_source='b1_anchor_forward',
                                                  within_tolerance=delta<=batched)),flush=True)
                            if delta>batched:
                                raise ValueError('sampled/replayed old score mismatch beyond the declared batched-rollout tolerance')
                            summary['n_checks']+=1
                            summary['max_abs_delta']=max(summary['max_abs_delta'],delta)
                            if delta>ROLLOUT_REPLAY_WARN_THRESHOLD:
                                summary['n_warn']+=1
                                print(json.dumps(dict(event='h3_rollout_replay_score_warning',sample_id=record.sample_id,
                                    index=record.index,max_abs_delta=delta,warn_threshold=ROLLOUT_REPLAY_WARN_THRESHOLD,
                                    declared_tolerance=batched)),flush=True)
                    pending[(record.sample_id,record.index)]=(joint.detach().cpu().clone(),tuple(v.detach().cpu().clone() for v in coeff))
                except Exception:
                    gate(f'anchor/{ordinal}/result', local_error=True)
                    raise
                gate(f'anchor/{ordinal}/result', local_error=False)
                del prediction, joint, coeff, x, action
            if read_behavior_id()!=self.behavior_id: raise ValueError('behavior changed during capture')
        self._scores=pending
        # Successful captures only; the lifecycle folds these local diagnostics
        # into its existing loss exchange, not a new model-stage collective.
        return summary

    def get(self,sample_id,index):
        if self._scores is None: raise RuntimeError('anchor capture incomplete')
        score,coeff=self._scores[(sample_id,index)]
        return score.clone(),tuple(c.clone() for c in coeff)
