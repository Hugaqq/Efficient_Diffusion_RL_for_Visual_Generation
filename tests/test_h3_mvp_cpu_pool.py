"""Actual local rollout/pool/anchor/objective control; only H3 weights are tiny."""

from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

import visual_rl
from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.optimization.h3_anchor import H3ReplayAnchor
from visual_rl.algorithms.optimization.objective import ClippedSurrogateObjective, PolicyLossInputs
from visual_rl.algorithms.trainer import h3_pool
from visual_rl.algorithms.trainer.h3_pool import H3RolloutPool
from visual_rl.models.h3_joint import H3JointLayout
from visual_rl.models.implementations.h3_forward import h3_t2va_forward
from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry


IDENTITY = dict(collection_id="collection-7", behavior_id="policy-3")
SAMPLES = (("s0", "g0", 0), ("s1", "g1", 1), ("s2", "g0", 1), ("s3", "g1", 0))


class _TinyH3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video = torch.nn.Parameter(torch.tensor(.12))
        self.audio = torch.nn.Parameter(torch.tensor(-.09))

    def forward(self, **kwargs):
        condition = kwargs["encoder_hidden_states"].mean()
        return (kwargs["hidden_states"].float() * .04 + self.video + condition,
                kwargs["audio_hidden_states"].float() * .07 + self.audio + condition)


def _dynamics(real=False):
    layout = H3T2VAGeometry(32, 32, 124).joint_layout if real else H3JointLayout(1, 2, 1, 3)
    return H3CPSDynamics(layout, H3ScheduleSnapshot.build(3))


def _inputs(dynamics, samples=SAMPLES, seed=40):
    initial, generators = {}, {}
    for row, (sid, _, _) in enumerate(samples):
        initial[sid] = torch.randn(1, dynamics.layout.packed_width,
                                  generator=torch.Generator().manual_seed(seed + row)).half()
        generators[sid] = (torch.Generator().manual_seed(seed + 10 + row),
                           torch.Generator().manual_seed(seed + 20 + row))
    return initial, generators


def _pool(dynamics, **overrides):
    width = dynamics.layout.packed_width
    options = dict(**IDENTITY, samples=SAMPLES, group_size=2, dynamics=dynamics,
                   max_samples=4, max_cells=12, max_trajectory_bytes=12*(4*width+4)+4*2*width)
    options.update(overrides)
    return H3RolloutPool(**options)


def _collect(pool, initial, generators, predict, **overrides):
    options = dict(**IDENTITY, device="cpu", predict=predict, read_behavior_id=lambda: "policy-3")
    options.update(overrides)
    pool.collect(initial, generators, **options)


def _bridge(model, dynamics):
    geometry = H3T2VAGeometry(32, 32, 124)
    conditions = {sid: torch.full((1, 2, 4), .01 * (row + 1)) for row, (sid, _, _) in enumerate(SAMPLES)}

    def predict(sid, x, index):
        vs, _, aus, _ = dynamics.schedule.transition(index)
        return h3_t2va_forward(model, x=x, geometry=geometry, text_embeddings=conditions[sid],
                               video_sigma=vs, audio_sigma=aus)
    return predict


def test_pool_imports_and_real_forward_cps_chain_order_cpu_storage():
    root = Path(__file__).resolve().parents[1]
    assert Path(visual_rl.__file__).resolve().parent == root / "visual_rl"
    assert Path(h3_pool.__file__).resolve() == root / "visual_rl/algorithms/trainer/h3_pool.py"
    dynamics = _dynamics(real=True)
    model = _TinyH3().eval()
    predict = _bridge(model, dynamics)
    initial, generators = _inputs(dynamics)
    pool = _pool(dynamics)
    calls, input_refs = [], []

    def checked(sid, x, index):
        assert not torch.is_grad_enabled() and not model.training
        assert all(ref() is None for ref in input_refs)
        input_refs.append(weakref.ref(x))
        calls.append((sid, index))
        return predict(sid, x, index)

    rng = torch.random.get_rng_state().clone()
    _collect(pool, initial, generators, checked)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(ref() is None for ref in input_refs)
    assert pool.state == "ready" and pool.samples(**IDENTITY) == SAMPLES
    records = pool.records(**IDENTITY)
    expected_keys = [(sid, i) for sid, _, _ in SAMPLES for i in range(3)]
    assert calls == expected_keys == [(sid, r.index) for sid, r in records]
    actual_bytes = 0
    for row, (sid, _, _) in enumerate(SAMPLES):
        trajectory = [r for _, r in records[row*3:(row+1)*3]]
        assert torch.equal(trajectory[0].x, initial[sid])
        for index, record in enumerate(trajectory):
            for tensor in (record.x, record.action, record.rollout_score):
                assert tensor.device.type == "cpu" and not tensor.requires_grad and tensor.grad_fn is None
                actual_bytes += tensor.numel() * tensor.element_size()
            assert record.x.dtype == record.action.dtype == torch.float16
            assert record.rollout_score.dtype == torch.float32
            if index:
                assert torch.equal(record.x, trajectory[index-1].action)
            v, a = dynamics.layout.unpack(record.action - record.x)
            assert bool((v != 0).any()) and bool((a != 0).any())
        terminal = pool.terminal(sid, **IDENTITY)
        assert torch.equal(terminal, trajectory[-1].action)
        actual_bytes += terminal.numel() * terminal.element_size()
    assert actual_bytes == pool.payload_bytes
    assert all(p.grad is None for p in model.parameters())


def test_pool_seed_reproduction_reference_and_terminal_rng_consumption():
    dynamics = _dynamics()
    predict = lambda sid, x, index: x.float() * .1 + .123
    outputs = []
    for seed in (40, 40, 41):
        pool = _pool(dynamics)
        initial, generators = _inputs(dynamics, seed=seed)
        _collect(pool, initial, generators, predict)
        outputs.append(pool.records(**IDENTITY))
        # Direct real CPS reference also confirms draws at the terminal step.
        reference_initial, reference_rng = _inputs(dynamics, seed=seed)
        with torch.no_grad():
            for row, (sid, _, _) in enumerate(SAMPLES):
                x = reference_initial[sid]
                for index in range(3):
                    expected = dynamics.sample(x, predict(sid, x, index), index,
                                               video_generator=reference_rng[sid][0],
                                               audio_generator=reference_rng[sid][1])
                    stored = outputs[-1][row*3+index][1]
                    assert torch.equal(stored.action, expected.action)
                    assert torch.equal(stored.rollout_score, expected.rollout_score)
                    x = expected.action
                assert all(torch.equal(g.get_state(), ref.get_state())
                           for g, ref in zip(generators[sid], reference_rng[sid]))
    assert all(torch.equal(a.action, b.action) for (_, a), (_, b) in zip(outputs[0], outputs[1]))
    assert any(not torch.equal(a.action, b.action) for (_, a), (_, b) in zip(outputs[0], outputs[2]))
    # Hold initial states and audio streams fixed: only video noise changes.
    base, changed = _pool(dynamics), _pool(dynamics)
    initial, streams = _inputs(dynamics)
    _, changed_streams = _inputs(dynamics)
    for video_rng, _ in changed_streams.values():
        video_rng.manual_seed(987)
    _collect(base, initial, streams, predict)
    _collect(changed, initial, changed_streams, predict)
    split = dynamics.layout.video_numel
    for (_, a), (_, b) in zip(base.records(**IDENTITY), changed.records(**IDENTITY)):
        assert torch.equal(a.action[:, split:], b.action[:, split:])
    assert any(not torch.equal(a.action[:, :split], b.action[:, :split])
               for (_, a), (_, b) in zip(base.records(**IDENTITY), changed.records(**IDENTITY)))


def test_pool_to_anchor_to_existing_objective_backward():
    dynamics = _dynamics(real=True)
    model = _TinyH3().eval()
    predict = _bridge(model, dynamics)
    pool = _pool(dynamics)
    _collect(pool, *_inputs(dynamics), predict)
    records = pool.records(**IDENTITY)
    ids = tuple(sid for sid, _, _ in pool.samples(**IDENTITY))
    anchor = H3ReplayAnchor(**IDENTITY, sample_ids=ids, dynamics=dynamics,
                            max_cells=12, max_trajectory_bytes=pool.payload_bytes)
    ordered = iter(records)
    anchor_keys = []

    def replay(record):
        sid, expected = next(ordered)
        assert record.index == expected.index
        assert torch.equal(record.x, expected.x) and torch.equal(record.action, expected.action)
        anchor_keys.append((sid, record.index))
        return predict(sid, record.x, record.index)

    anchor.capture(records, **IDENTITY, device="cpu", predict=replay, read_behavior_id=lambda: "policy-3")
    assert anchor_keys == [(sid, r.index) for sid, r in records]
    assert next(ordered, None) is None
    old = torch.stack([torch.cat([anchor.score(**IDENTITY, sample_id=sid, index=i) for i in range(3)]) for sid in ids])
    with torch.no_grad():
        model.video.add_(.01)
        model.audio.sub_(.015)
    current = torch.cat([dynamics.evaluate(r, predict(sid, r.x, r.index)) for sid, r in records]).reshape(4, 3)
    assert not torch.equal(old, current.detach())
    result = ClippedSurrogateObjective().compute(
        old_log_probs=old, policy_stats=SimpleNamespace(current_log_probs=current),
        loss_inputs=PolicyLossInputs(base_advantage=torch.ones_like(old), algorithm_weight=torch.ones_like(old),
                                    active_mask=torch.ones_like(old, dtype=torch.bool), clip_range=.2, reference_kl_weight=0),
    )
    assert result.active_transition_count == 12 and torch.isfinite(result.loss)
    result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad) and p.grad != 0 for p in model.parameters())
    assert torch.equal(old, torch.stack([torch.cat([
        anchor.score(**IDENTITY, sample_id=sid, index=i) for i in range(3)
    ]) for sid in ids]))
    # This test has no trainer tail: release only tests local ownership cleanup.
    pool.release()
    assert pool.state == "released"


def test_pool_group_budgets_and_all_initial_inputs_rejected_before_forward():
    dynamics = _dynamics()
    for overrides in (
        {"max_samples": 3}, {"max_cells": 11}, {"max_trajectory_bytes": 327},
        {"group_size": 1}, {"group_size": True}, {"max_samples": True},
        {"samples": ()}, {"samples": SAMPLES[:-1]},
        {"samples": (SAMPLES[0], SAMPLES[0])},
        {"samples": (("a", "g", 0), ("b", "g", 0))},
        {"samples": (("a", "g", -1), ("b", "g", 1))},
        {"samples": (("a", "g", False), ("b", "g", 1))},
        {"samples": (("a", "g", 0), ("b", "g", 2))},
        {"samples": (("", "g", 0), ("b", "g", 1))},
        {"samples": (("a", "", 0), ("b", "g", 1))},
    ):
        with pytest.raises(ValueError):
            _pool(dynamics, **overrides)
    calls = []
    initial, streams = _inputs(dynamics)
    invalid_states = (initial["s3"].float(), initial["s3"].repeat(2, 1), initial["s3"][:, :-1],
                      initial["s3"].clone().requires_grad_(), initial["s3"].to("meta"),
                      torch.full_like(initial["s3"], float("nan")), torch.full_like(initial["s3"], float("inf")))
    for bad in invalid_states:
        pool = _pool(dynamics)
        with pytest.raises(ValueError):
            _collect(pool, dict(initial, s3=bad), streams, lambda *args: calls.append(args))
        assert pool.state == "failed"
        with pytest.raises(RuntimeError, match="not ready"):
            pool.records(**IDENTITY)
        with pytest.raises(RuntimeError, match="only once"):
            _collect(pool, initial, streams, lambda *args: calls.append(args))
    bad_cases = (({sid: x for sid, x in initial.items() if sid != "s3"}, streams),
                 (dict(initial, extra=initial["s0"]), streams),
                 (initial, {sid: pair for sid, pair in streams.items() if sid != "s3"}),
                 (initial, dict(streams, s3=(streams["s0"][0], streams["s3"][1]))),
                 (initial, dict(streams, s3=(streams["s3"][0], streams["s3"][0]))),
                 (initial, dict(streams, s3=(None, streams["s3"][1]))))
    for states, generators in bad_cases:
        with pytest.raises(ValueError):
            _collect(_pool(dynamics), states, generators, lambda *args: calls.append(args))
    assert calls == []


def test_pool_failure_versions_and_no_partial_reads_or_retries():
    dynamics = _dynamics()
    initial, streams = _inputs(dynamics)
    calls = []
    pool = _pool(dynamics)

    def failing(sid, x, index):
        calls.append((sid, index))
        return torch.full_like(x.float(), float("nan")) if len(calls) == 4 else x.float() * .1

    with pytest.raises(ValueError, match="finite"):
        _collect(pool, initial, streams, failing)
    assert len(calls) == 4 and pool.state == "failed"
    for read in (lambda: pool.records(**IDENTITY), lambda: pool.samples(**IDENTITY),
                 lambda: pool.terminal("s0", **IDENTITY)):
        with pytest.raises(RuntimeError, match="not ready"):
            read()
    with pytest.raises(RuntimeError, match="only once"):
        _collect(pool, initial, streams, failing)
    for overrides in ({"collection_id": "wrong"}, {"behavior_id": "wrong"},
                      {"read_behavior_id": lambda: "policy-4"}, {"device": "meta"}):
        count = len(calls)
        with pytest.raises(ValueError):
            _collect(_pool(dynamics), initial, streams, failing, **overrides)
        assert len(calls) == count
    pool = _pool(dynamics)
    version = iter(("policy-3", "policy-4"))
    with pytest.raises(ValueError, match="during collection"):
        _collect(pool, initial, streams, lambda sid, x, i: x.float() * .1,
                 read_behavior_id=lambda: next(version))
    assert pool.state == "failed"
    with pytest.raises(RuntimeError, match="not ready"):
        pool.records(**IDENTITY)
    pool.release()
    pool.release()
    assert pool.state == "released"


def test_pool_input_read_ownership_identity_release_and_no_callback_retention():
    dynamics = _dynamics()
    initial, streams = _inputs(dynamics)
    originals = {sid: x.clone() for sid, x in initial.items()}
    pool = _pool(dynamics)
    model = torch.nn.Linear(5, 5, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(5) * .1)

    def predict(sid, x, index):
        # Mutating caller-owned states during collection cannot alter later samples.
        for value in initial.values():
            value.fill_(88)
        return model(x.float())

    model_ref, callback_ref = weakref.ref(model), weakref.ref(predict)
    _collect(pool, initial, streams, predict)
    records = pool.records(**IDENTITY)
    for row, (sid, _, _) in enumerate(SAMPLES):
        assert torch.equal(records[row*3][1].x, originals[sid])
        expected = pool.terminal(sid, **IDENTITY)
        pool.terminal(sid, **IDENTITY).fill_(99)
        for _, r in records[row*3:(row+1)*3]:
            r.x.fill_(99)
            r.action.fill_(99)
            r.rollout_score.fill_(99)
        assert torch.equal(pool.terminal(sid, **IDENTITY), expected)
        assert torch.equal(pool.terminal(sid, **IDENTITY), records[row*3+2][1].action)
    for bad_identity in (dict(IDENTITY, collection_id="other"), dict(IDENTITY, behavior_id="other")):
        with pytest.raises(ValueError, match="identity"):
            pool.records(**bad_identity)
        with pytest.raises(ValueError, match="identity"):
            pool.samples(**bad_identity)
        with pytest.raises(ValueError, match="identity"):
            pool.terminal("s0", **bad_identity)
    with pytest.raises(KeyError):
        pool.terminal("unknown", **IDENTITY)
    with pytest.raises(RuntimeError, match="only once"):
        _collect(pool, initial, streams, predict)
    refs = [weakref.ref(r) for _, r in records]
    del records, r, predict, model
    assert model_ref() is None and callback_ref() is None
    pool.release()
    pool.release()
    assert all(ref() is None for ref in refs)
    with pytest.raises(RuntimeError, match="not ready"):
        pool.records(**IDENTITY)
    with pytest.raises(RuntimeError, match="not ready"):
        pool.terminal("s0", **IDENTITY)
    with pytest.raises(RuntimeError, match="only once"):
        _collect(pool, initial, streams, lambda *args: None)
