"""Rank-local replay-anchor CPU evidence, not controller/world/GPU evidence."""

from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

import visual_rl
from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics, H3CPSRecord
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.optimization import h3_anchor
from visual_rl.algorithms.optimization.h3_anchor import H3ReplayAnchor
from visual_rl.algorithms.optimization.objective import ClippedSurrogateObjective, PolicyLossInputs
from visual_rl.models.h3_joint import H3JointLayout


class _TinyPrediction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.11, -.27, .32, -.41, .19]))

    def forward(self, record):
        return record.x.float() * .15 + self.weight


def _case():
    dynamics = H3CPSDynamics(H3JointLayout(1, 2, 1, 3), H3ScheduleSnapshot.build(3))
    model = _TinyPrediction().eval()
    # Distinct sample IDs remain separate even if supplied by the same prompt.
    ids = ("sample-a", "sample-b")
    entries = {}
    for row, sample_id in enumerate(ids):
        x = torch.tensor([[.2, -.1, .3, -.4, .5]], dtype=torch.float16) + row * .125
        for index in range(3):
            prediction = x.float() * .15 + model.weight
            record = dynamics.sample(
                x, prediction, index,
                video_generator=torch.Generator().manual_seed(10 + row * 3 + index),
                audio_generator=torch.Generator().manual_seed(20 + row * 3 + index),
            )
            # Deliberately false rollout diagnostic: capture must replay, not copy.
            entries[(sample_id, index)] = H3CPSRecord(
                dynamics, index, record.x, record.action, record.rollout_score + 37,
            )
            x = record.action
    order = (("sample-b", 2), ("sample-a", 0), ("sample-b", 0),
             ("sample-a", 2), ("sample-a", 1), ("sample-b", 1))
    return dynamics, model, ids, tuple((sid, entries[(sid, idx)]) for sid, idx in order)


def _owner(dynamics, ids, **overrides):
    options = dict(collection_id="collection-1", behavior_id="policy-0", sample_ids=ids,
                   dynamics=dynamics, max_cells=6,
                   max_trajectory_bytes=6 * (4 * dynamics.layout.packed_width + 4))
    options.update(overrides)
    return H3ReplayAnchor(**options)


def _capture(owner, records, predict, **overrides):
    options = dict(collection_id="collection-1", behavior_id="policy-0", device="cpu",
                   predict=predict, read_behavior_id=lambda: "policy-0")
    options.update(overrides)
    owner.capture(records, **options)


def _score(owner, sid, index, **overrides):
    options = dict(collection_id="collection-1", behavior_id="policy-0", sample_id=sid, index=index)
    options.update(overrides)
    return owner.score(**options)


def test_anchor_import_identity():
    root = Path(__file__).resolve().parents[1]
    assert Path(visual_rl.__file__).resolve().parent == root / "visual_rl"
    assert Path(h3_anchor.__file__).resolve() == root / "visual_rl/algorithms/optimization/h3_anchor.py"


def test_anchor_real_replay_once_full_coverage_then_existing_ppo_backward():
    dynamics, model, ids, records = _case()
    owner = _owner(dynamics, ids)
    with torch.no_grad():
        expected = {(sid, r.index): dynamics.evaluate(r, model(r)) for sid, r in records}
    calls, temporary_records = [], []

    def predict(record):
        assert not torch.is_grad_enabled()
        assert not model.training
        assert all(ref() is None for ref in temporary_records)
        temporary_records.append(weakref.ref(record))
        assert record.x.device.type == "cpu" and record.x.dtype == torch.float16
        assert not record.x.requires_grad and not record.action.requires_grad
        calls.append((record.index, record.x.clone()))
        return model(record)

    rng = torch.random.get_rng_state().clone()
    _capture(owner, records, predict)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert len(calls) == len(records) == len(ids) * dynamics.schedule.num_steps
    assert all(ref() is None for ref in temporary_records)
    assert [i for i, _ in calls] == [r.index for _, r in records]
    assert all(torch.equal(x, r.x) for (_, x), (_, r) in zip(calls, records))
    for sid, record in records:
        actual = _score(owner, sid, record.index)
        assert torch.equal(actual, expected[(sid, record.index)])
        assert not torch.equal(actual, record.rollout_score)
        assert actual.dtype == torch.float32 and actual.device.type == "cpu"
        assert not actual.requires_grad and actual.grad_fn is None
    with pytest.raises(RuntimeError, match="already attempted"):
        _capture(owner, records, predict)
    assert len(calls) == 6

    with torch.no_grad():
        model.weight.add_(.015)
    by_key = {(sid, r.index): r for sid, r in records}
    old = torch.stack([torch.cat([_score(owner, sid, i) for i in range(3)]) for sid in ids])
    current = torch.stack([torch.cat([
        dynamics.evaluate(by_key[(sid, i)], model(by_key[(sid, i)])) for i in range(3)
    ]) for sid in ids])
    assert not torch.equal(old, current.detach())
    assert all(torch.equal(_score(owner, sid, i), value) for (sid, i), value in expected.items())
    loss_inputs = PolicyLossInputs(
        base_advantage=torch.tensor([[1., .8, .6], [-.4, -.5, -.7]]),
        algorithm_weight=torch.ones(2, 3), active_mask=torch.ones(2, 3, dtype=torch.bool),
        clip_range=.2, reference_kl_weight=0.,
    )
    result = ClippedSurrogateObjective().compute(
        old_log_probs=old, policy_stats=SimpleNamespace(current_log_probs=current), loss_inputs=loss_inputs,
    )
    assert torch.isfinite(result.loss) and result.active_transition_count == 6
    assert result.reference_kl == 0
    result.loss.backward()
    assert torch.isfinite(model.weight.grad).all()
    assert bool((model.weight.grad[:2] != 0).any()) and bool((model.weight.grad[2:] != 0).any())


def test_anchor_preflight_coverage_contract_and_bounds_before_model():
    dynamics, model, ids, records = _case()
    sid, first = records[0]
    alternate = H3CPSDynamics(dynamics.layout, dynamics.schedule, eta=.5)
    conflicting = H3CPSRecord(alternate, first.index, first.x, first.action, first.rollout_score)
    batch2 = H3CPSRecord(dynamics, first.index, first.x.repeat(2, 1),
                         first.action.repeat(2, 1), first.rollout_score.repeat(2))
    cases = (
        ((), {}, "nonempty"),
        (list(records), {}, "tuple"),
        (records[:-1], {}, "coverage"),
        (tuple(item for item in records if item[0] != ids[1]), {}, "coverage"),
        (records[:-1] + (records[0],), {}, "duplicate"),
        ((("unknown", first),) + records[1:], {}, "sample_id"),
        (((sid, conflicting),) + records[1:], {}, "contract"),
        (((sid, batch2),) + records[1:], {}, "B1"),
        (records, {"max_cells": 5}, "max_cells"),
        (records, {"max_trajectory_bytes": 143}, "max_trajectory_bytes"),
    )
    calls = []
    for bad_records, options, error in cases:
        owner = _owner(dynamics, ids, **options)
        with pytest.raises(ValueError, match=error):
            _capture(owner, bad_records, lambda r: calls.append(r) or model(r))
        with pytest.raises(RuntimeError, match="no complete"):
            _score(owner, ids[0], 0)
        with pytest.raises(RuntimeError, match="already attempted"):
            _capture(owner, records, model)
    assert calls == []
    for options in ({"sample_ids": ()}, {"sample_ids": ("a", "a")}, {"sample_ids": ("",)},
                    {"max_cells": True}, {"max_trajectory_bytes": 0}, {"collection_id": ""}):
        with pytest.raises(ValueError):
            _owner(dynamics, ids, **options)


def test_anchor_failure_is_atomic_and_owner_is_not_retryable():
    dynamics, model, ids, records = _case()
    owner = _owner(dynamics, ids)
    calls = []

    def bad_predict(record):
        calls.append(record.index)
        if len(calls) == 2:
            return torch.full_like(record.x.float(), float("nan"))
        return model(record)

    with pytest.raises(ValueError, match="finite"):
        _capture(owner, records, bad_predict)
    assert len(calls) == 2
    for sid, record in records:
        with pytest.raises(RuntimeError, match="no complete"):
            _score(owner, sid, record.index)
    with pytest.raises(RuntimeError, match="already attempted"):
        _capture(owner, records, model)


def test_anchor_versions_identity_and_lookup_are_strict():
    dynamics, model, ids, records = _case()
    calls = []
    for override in ({"collection_id": "wrong"}, {"behavior_id": "wrong"},
                     {"read_behavior_id": lambda: "policy-1"}, {"device": "meta"}):
        owner = _owner(dynamics, ids)
        with pytest.raises(ValueError):
            _capture(owner, records, lambda r: calls.append(r) or model(r), **override)
        with pytest.raises(RuntimeError, match="no complete"):
            _score(owner, ids[0], 0)
    assert calls == []
    owner = _owner(dynamics, ids)
    versions = iter(("policy-0", "policy-1"))
    with pytest.raises(ValueError, match="during capture"):
        _capture(owner, records, lambda r: calls.append(r.index) or model(r),
                 read_behavior_id=lambda: next(versions))
    assert len(calls) == 6
    with pytest.raises(RuntimeError, match="no complete"):
        _score(owner, ids[0], 0)
    with pytest.raises(RuntimeError, match="already attempted"):
        _capture(owner, records, model)
    owner = _owner(dynamics, ids)
    _capture(owner, records, model)
    for override in ({"collection_id": "wrong"}, {"behavior_id": "policy-1"}):
        with pytest.raises(ValueError, match="identity"):
            _score(owner, ids[0], 0, **override)
    with pytest.raises(KeyError):
        _score(owner, "unknown", 0)
    for bad in (True, 0., -1, 3):
        with pytest.raises((IndexError, TypeError)):
            _score(owner, ids[0], bad)


def test_anchor_input_read_mutation_and_no_record_or_callback_retention():
    dynamics, model, ids, original = _case()
    caller_buffers, records = [], []
    for sid, record in original:
        x, action, diagnostic = record.x, record.action, record.rollout_score
        records.append((sid, H3CPSRecord(dynamics, record.index, x, action, diagnostic)))
        caller_buffers.extend((x, action, diagnostic))
    records = tuple(records)
    for buffer in caller_buffers:
        buffer.fill_(99)
    owner = _owner(dynamics, ids)
    _capture(owner, records, model)
    expected = {(sid, r.index): _score(owner, sid, r.index) for sid, r in records}
    for sid, record in records:
        record.x.fill_(99)
        record.action.fill_(99)
        record.rollout_score.fill_(99)
        _score(owner, sid, record.index).fill_(99)
    assert all(torch.equal(_score(owner, sid, i), score) for (sid, i), score in expected.items())
    refs = [weakref.ref(r) for _, r in records]
    model_ref = weakref.ref(model)
    del record, records, model
    assert all(ref() is None for ref in refs)
    assert model_ref() is None
    assert all(torch.equal(_score(owner, sid, i), score) for (sid, i), score in expected.items())
