from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.optimization.h3_dual_credit import policy_loss
from visual_rl.models.h3_joint import H3JointLayout
from tests.test_h3_training_extensions import legacy_module


def test_kl_two_stream_formula_zero_gradient_and_no_tgr():
    d = H3DualFlowSDE(H3JointLayout(2, 3, 1, 2), H3ScheduleSnapshot.build(10))
    x = torch.linspace(-1, 1, 8).reshape(1, -1).half()
    p = torch.linspace(-.1, .1, 8).reshape(1, -1).requires_grad_()
    reference = p.detach().clone()
    kl, streams = d.reference_kl(x, p, reference, 3)
    assert kl.item() == streams[0].item() == streams[1].item() == 0.
    assert torch.equal(torch.autograd.grad(kl.sum(), p)[0], torch.zeros_like(p))
    reference += .2
    kl, streams = d.reference_kl(x, p, reference, 3)
    current_stats = d._parts(x, p, 3)
    ref_stats = d._parts(x, reference, 3)
    for value, (mean, std, _), (ref, _, _) in zip(streams, current_stats, ref_stats):
        assert torch.equal(value, ((mean-ref).square()/(2*std.square())).mean().reshape(1))
    assert torch.equal(kl, streams[0]*.75 + streams[1]*.25)
    rec = SimpleNamespace(x=x, action=x, index=3)
    old = d.evaluate(x, x, p.detach(), 3)[0]
    gradients = []
    for means in ((torch.tensor(1.), torch.tensor(10.)), (torch.tensor(20.), torch.tensor(.1))):
        rows = []
        loss = policy_loss(dynamics=d, record=rec, prediction=p, old_joint=old, advantage=torch.tensor(0.),
            mode='flash', world_means=means, clip_range=.005, scale=.25,
            kl_beta=.3, reference_prediction=reference, kl_metrics=rows.append)
        gradients.append(torch.autograd.grad(loss, p)[0])
        assert rows[0]['kl'] == kl.item()
        assert rows[0]['policy_loss'] == 0.
        assert rows[0]['kl_loss'] == loss.item()
        assert torch.equal(loss, kl.mean()*.3*.25)
    assert torch.equal(*gradients)
    with pytest.raises(ValueError, match='detached'):
        d.reference_kl(x, p, p, 3)
    with pytest.raises(ValueError, match='finite'):
        d.reference_kl(x, p, reference*float('nan'), 3)
    legacy = legacy_module('visual_rl/algorithms/optimization/h3_dual_credit.py')
    args = dict(dynamics=d, record=rec, prediction=p, old_joint=old, advantage=torch.tensor(.4),
                mode='flash', world_means=(torch.tensor(1.), torch.tensor(2.)), clip_range=.005, scale=.25)
    a, b = policy_loss(**args), legacy.policy_loss(**args)
    assert torch.equal(a, b)
    assert torch.equal(torch.autograd.grad(a,p)[0], torch.autograd.grad(b,p)[0])


def test_real_peft_reference_binding_restores_rng_flags_grads_buffers_and_failure(tmp_path, monkeypatch):
    from tests.test_h3_mvp_cpu_binding import _fixture
    from visual_rl.composition.config.h3_profile import preflight_h3_profile
    from visual_rl.runtime.h3_binding import H3RuntimeBinding
    from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot
    from visual_rl.runtime import h3_file_phase
    from visual_rl.models.implementations import h3_loading
    monkeypatch.setattr(torch.distributed, 'broadcast', lambda *args, **kwargs: None)
    monkeypatch.setattr(h3_file_phase, 'file_phase', lambda work, **kwargs: work())
    monkeypatch.setattr(h3_loading, '_admit_host_memory', lambda budget: None)
    h = _fixture(tmp_path, monkeypatch, bind=False)
    profile = replace(h.profile, training=replace(h.profile.training, algorithm='dual_flow', kl_beta=.2))
    b = H3RuntimeBinding(**(h.kwargs | dict(profile=profile, facts=preflight_h3_profile(profile))))
    request = h.source.reserve(0)
    sid = request.samples[0][0]
    args = dict(sample_id=sid, x=request.initial_states[sid], index=0,
                condition=request.conditions[sid], dynamics=h.dynamics)
    params = tuple(h.adapter.meta_model.model.parameters())
    before = [p.detach().clone() for p in params]
    flags = [p.requires_grad for p in params]
    current = b.predict(h.prepared, **args)
    current.square().mean().backward()
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    rng = RankRNGSnapshot.capture_current(0)
    reference = b.predict_reference(h.prepared, **args)
    assert not reference.requires_grad and reference.grad_fn is None
    assert torch.equal(current.detach(), reference)  # zero-initialized B
    assert rng.state_identity == RankRNGSnapshot.capture_current(0).state_identity
    assert flags == [p.requires_grad for p in params]
    assert all(torch.equal(saved, p) for saved, p in zip(before, params))
    assert all((p.grad is None if g is None else torch.equal(p.grad,g)) for p,g in zip(params,grads))
    assert not h.prepared.optimizer.state
    predict = b.predict
    def fail(*args, **kwargs):
        assert b.reference.active and not torch.is_grad_enabled()
        torch.rand(3)
        raise RuntimeError('injected reference failure')
    monkeypatch.setattr(b, 'predict', fail)
    with pytest.raises(RuntimeError, match='injected'):
        b.predict_reference(h.prepared, **args)
    assert not b.reference.active
    assert rng.state_identity == RankRNGSnapshot.capture_current(0).state_identity
    assert flags == [p.requires_grad for p in params]
    monkeypatch.setattr(b, 'predict', predict)
    assert torch.equal(b.predict(h.prepared, **args), current)
    # Warm Adam moments and changed live LoRA must also remain untouched.
    from copy import deepcopy
    from tests.test_h3_mvp_cpu_checkpoint_state import _equal
    h.prepared.optimizer.step(); h.prepared.scheduler.step(); h.prepared.optimizer.zero_grad(set_to_none=True)
    warm = deepcopy(h.prepared.optimizer.state_dict())
    weights = [p.detach().clone() for p in params]
    assert torch.equal(b.predict_reference(h.prepared, **args), reference)
    _equal(warm, h.prepared.optimizer.state_dict())
    assert all(torch.equal(p,w) for p,w in zip(params,weights))
    assert all(p.grad is None for p in params)
    # 三个 cell 共用一次 RNG/buffer 快照与两次边界全模型检查。
    counts = dict(rng=0, buffers=0, observe=0)
    capture = RankRNGSnapshot.capture_current
    buffers = h.adapter.meta_model.model.buffers
    observe = b._observe_model
    def counted_capture(rank):
        counts['rng'] += 1
        return capture(rank)
    def counted_buffers(*args, **kwargs):
        counts['buffers'] += 1
        return buffers(*args, **kwargs)
    def counted_observe():
        counts['observe'] += 1
        return observe()
    monkeypatch.setattr(RankRNGSnapshot, 'capture_current', counted_capture)
    monkeypatch.setattr(h.adapter.meta_model.model, 'buffers', counted_buffers)
    monkeypatch.setattr(b, '_observe_model', counted_observe)
    with b.reference_update_scope():
        for _ in range(3):
            assert torch.equal(b.predict_reference(h.prepared, **args), reference)
    assert counts == dict(rng=1, buffers=1, observe=2)
    assert capture(0).state_identity == rng.state_identity
    buffer = next(buffers())
    saved_buffer = buffer.clone()
    def mutate(*args, **kwargs):
        torch.rand(3)
        buffer.add_(1)
        return reference
    monkeypatch.setattr(b, 'predict', mutate)
    with pytest.raises(RuntimeError, match='mutated a model buffer'):
        with b.reference_update_scope():
            b.predict_reference(h.prepared, **args)
    assert torch.equal(buffer, saved_buffer)
    assert not b.reference.active and b.reference._flags is None
    assert capture(0).state_identity == rng.state_identity
    assert flags == [p.requires_grad for p in params]
    monkeypatch.setattr(b, 'predict', fail)
    with pytest.raises(RuntimeError, match='injected'):
        with b.reference_update_scope():
            b.predict_reference(h.prepared, **args)
    assert capture(0).state_identity == rng.state_identity
    assert not b.reference.active and b.reference._flags is None
    assert flags == [p.requires_grad for p in params]
    b.close(); h.scorer.close(); h.adapter.close(); h.manager.close()


@pytest.mark.parametrize('beta', [0., .1])
def test_stage_reference_call_count_and_beta_zero_no_call(tmp_path, monkeypatch, beta):
    from tests.test_h3_mvp_cpu_stages import _setup
    from visual_rl.algorithms.trainer.h3_dual_stages import H3DualIterationStages
    from visual_rl.algorithms.trainer.interface import PrepareRunContext
    h = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda tensor, **kw: tensor.mul_(8))
    def commit(request):
        h.data.commits += 1; h.data.position += 2; h.data.behavior = request.next_behavior_id
    monkeypatch.setattr(h.data, 'commit', commit)
    from contextlib import contextmanager
    calls, scopes = [], []
    @contextmanager
    def update_scope():
        scopes.append('enter')
        try:
            yield
        finally:
            scopes.append('exit')
    def reference(*args, **kwargs):
        calls.append((kwargs['sample_id'], kwargs['index']))
        assert not torch.is_grad_enabled()
        return h.predict(*args, **kwargs).detach()
    stage = H3DualIterationStages(mode='dual_flow', seed=42, data=h.data, prepared=h.prepared,
        predict=h.predict, decode=h.decode, score_one=h.score_one, coordination_factory=h.coordination_factory,
        process_group=h.group, rank=0, device='cpu', read_behavior_id=lambda:h.data.behavior,
        kl_beta=beta, predict_reference=reference, reference_update_scope=update_scope)
    trainer = stage.trainer(); trainer.prepare_run(PrepareRunContext('kl-test', 'h3', 0))
    result = trainer.run_iteration(0)
    assert scopes == (['enter', 'exit'] if beta else [])
    assert len(calls) == (6 if beta else 0)
    if beta:
        assert stage.kl_metrics == dict(kl=0., kl_video=0., kl_audio=0., kl_loss=0.,
                                       policy_loss=result.value.payload.scaled_loss)
    else:
        assert stage.kl_metrics is None
