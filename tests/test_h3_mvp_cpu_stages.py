"""Real BaseTrainer stages/CPU pool/decode/world math/anchor/objective/Adam.

Only heavy H3/VAE weights, reward transport, and distributed/CUDA leaves are
substituted. No PG, network or GPU is created. W3 coordinator leaf is explicit.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from tests.test_h3_mvp_cpu_decode import _artifacts, _leaf_setup, _decoder
from tests.test_h3_mvp_cpu_pool import _TinyH3
from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.optimization import h3_world
from visual_rl.algorithms.trainer import h3_stages
from visual_rl.algorithms.trainer.grpo import BaseTrainer
from visual_rl.algorithms.trainer.interface import IterationIdentity, PrepareRunContext
from visual_rl.algorithms.trainer.h3_stages import H3IterationRequest, H3IterationStages
from visual_rl.models.implementations.h3_forward import h3_t2va_forward
from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry
from visual_rl.models.lifecycle.components import ComponentBinding, ComponentRole, ModelComponents
from visual_rl.models.lifecycle.h3_fsdp import H3PreparedModel
from visual_rl.models.lifecycle.prepared import PreparedModelBundle


def _world_leaf(monkeypatch, events):
    group = object()
    dist = h3_world.dist
    monkeypatch.setattr(dist, 'group', SimpleNamespace(WORLD=group))
    monkeypatch.setattr(dist, 'is_available', lambda: True)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_backend', lambda pg: 'nccl')
    monkeypatch.setattr(dist, 'get_world_size', lambda pg: 8)
    monkeypatch.setattr(dist, 'get_rank', lambda pg: 0)

    def gather(output, raw, *, group):
        receipt = json.loads(raw)
        events.append(('world', receipt[1], receipt[5]))
        for rank in range(8):
            peer = json.loads(raw)
            peer[2] = rank
            if peer[5] and rank:
                rows = peer[4][1] if peer[1] == 'admission' else peer[4][3]
                for row in rows:
                    row[0] += f'@rank{rank}'
                    row[1] += f'@rank{rank}'
            output[rank] = json.dumps(peer).encode()
    monkeypatch.setattr(dist, 'all_gather_object', gather)
    return group


class _CoordinationLeaf:
    """Single-rank CPU leaf for W3 gates/aux/global norm, not production fallback."""
    def __init__(self, plan, events):
        self.plan, self.events = plan, events
        self.gradients = []

    def gate(self, event, *, local_error=False):
        self.events.append(('gate', event, local_error))
        if local_error:
            raise ValueError('cooperative rejection: ' + event)

    def aux_turns(self, work):
        self.gate('aux/test/before')
        try:
            result = work()
        except Exception:
            self.gate('aux/test/after', local_error=True)
            raise
        self.gate('aux/test/after')
        return result

    def clip_grad_norm(self, prepared, *, max_norm):
        self.events.append(('norm', max_norm))
        params = [p for _, p in prepared.trainable_named_parameters()]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        self.gradients.append(tuple(p.grad.clone() for p in params))
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True))


def _setup(tmp_path, monkeypatch, *, fault=None, zero=False):
    events, coords, requests, pools, media_refs, prediction_refs = [], [], [], [], [], []
    group = _world_leaf(monkeypatch, events)
    geometry = H3T2VAGeometry(32, 32, 124)
    dynamics = H3CPSDynamics(geometry.joint_layout, H3ScheduleSnapshot.build(3))
    model = _TinyH3().eval()
    bundle = PreparedModelBundle(ModelComponents((ComponentBinding(
        'transformer', model, (ComponentRole.INFERENCE, ComponentRole.TRAINABLE)),)))
    optimizer = torch.optim.AdamW(bundle.trainable_parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    prepared = H3PreparedModel(source_bundle=bundle, prepared_root=bundle, optimizer=optimizer,
                               scheduler=scheduler, accelerator=object())
    optimizer.register_step_post_hook(lambda *args: events.append(('step',)))
    original_step = scheduler.step
    def scheduler_step():
        events.append(('scheduler',))
        return original_step()
    monkeypatch.setattr(scheduler, 'step', scheduler_step)
    if fault == 'step':
        original_adam = optimizer.step
        def partial_step(*args, **kwargs):
            original_adam(*args, **kwargs)
            raise RuntimeError('partial Adam failure')
        monkeypatch.setattr(optimizer, 'step', partial_step)

    class Data:
        behavior = 'policy-0'
        position = 0
        commits = 0
        aborts = 0

        def reserve(self, step):
            events.append(('reserve', step))
            samples = tuple((f's{step}-{i}', f'group-{step}', i) for i in range(2))
            ids = tuple(row[0] for row in samples)
            identity = IterationIdentity(step, 'tiny-source', 'train', ids,
                                         tuple(row[1] for row in samples), (0, 1))
            width = dynamics.layout.packed_width
            initial = {sid: torch.randn(1, width, generator=torch.Generator().manual_seed(42+step*2+i)).half()
                       for i, sid in enumerate(ids)}
            if fault == 'preflight':
                initial[ids[-1]][0, 0] = float('nan')
            request = H3IterationRequest(
                identity, f'collection-{step}', self.behavior, f'policy-{step+1}', samples,
                {sid: 'actual prompt ' + sid for sid in ids},
                {sid: torch.full((1, 2, 4), .02*(i+1)) for i, sid in enumerate(ids)},
                dynamics, '32x32x124@24:t2va', initial,
                {sid: (torch.Generator().manual_seed(10+step*2+i), torch.Generator().manual_seed(30+step*2+i))
                 for i, sid in enumerate(ids)}, 2, 2, 6, 6*(4*width+4)+4*width,
            )
            if fault == 'identity':
                request = replace(request, identity=replace(identity, optimizer_step=step+1))
            if fault == 'condition':
                request.conditions[ids[-1]].fill_(float('nan'))
            requests.append(request)
            return request

        def commit(self, request):
            assert stages._active.pool.state == 'ready'
            events.append(('data_commit', request.identity.optimizer_step))
            self.commits += 1
            self.position += 2
            self.behavior = request.next_behavior_id
            if fault == 'tail':
                raise RuntimeError('partial data tail failure')

        def abort(self, request):
            self.aborts += 1
            events.append(('abort', request.identity.optimizer_step))
    data = Data()
    _artifacts(tmp_path)
    _leaf_setup(monkeypatch, geometry)
    decoder = _decoder(tmp_path, geometry)

    @contextmanager
    def decode(terminal):
        events.append(('decode',))
        with decoder.decode_one(terminal, geometry) as media:
            media_refs.append(weakref.ref(media))
            yield media
        events.append(('media_release',))

    def score_one(*, sample_id, prompt, media):
        events.append(('score', sample_id))
        assert prompt.endswith(sample_id)
        assert media.metadata.frame_count == 124 and media.audio.shape[1] == 2
        if fault == 'score':
            return float('nan')
        return 1. if zero else float(media.video.mean())

    def coordination_factory(plan):
        events.append(('coordination', plan.plan_id))
        coord = _CoordinationLeaf(plan, events)
        coords.append(coord)
        return coord

    def predict(bound, *, sample_id, x, index, condition, dynamics):
        assert bound is prepared and not model.training
        assert all(ref() is None for ref in prediction_refs)
        phase = next(e[1].split('/')[0] for e in reversed(events) if e[0] == 'gate')
        events.append(('forward', phase, sample_id, index, torch.is_grad_enabled()))
        if fault == 'raw_forward':
            raise RuntimeError('raw forward failure')
        vs, _, aus, _ = dynamics.schedule.transition(index)
        result = h3_t2va_forward(prepared, x=x, geometry=geometry, text_embeddings=condition,
                                 video_sigma=vs, audio_sigma=aus)
        if fault == phase:
            result = result * float('nan')
        if fault == 'version_current' and phase == 'current':
            data.behavior = 'unexpected-policy'
        prediction_refs.append(weakref.ref(result))
        if torch.is_grad_enabled():
            def backward_hook(gradient):
                events.append(('backward', sample_id, index))
                if fault == 'raw_backward':
                    raise RuntimeError('raw backward failure')
                return gradient
            result.register_hook(backward_hook)
        return result

    stages = H3IterationStages(data=data, prepared=prepared, predict=predict, decode=decode,
                               score_one=score_one, coordination_factory=coordination_factory,
                               process_group=group, rank=0, device='cpu', read_behavior_id=lambda: data.behavior)
    trainer = stages.trainer()
    # Observe actual stages without replacing their implementation/control.
    identities = []
    for name, stage in tuple(trainer._stages.items()):
        def observed(value, stage=stage, name=name):
            events.append(('stage', name))
            identities.append(value.identity)
            if name == 'rollout':
                pools.append(value.payload.pool)
            result = stage(value)
            assert result.identity is value.identity
            return result
        trainer._stages[name] = observed
    trainer.prepare_run(PrepareRunContext('cpu-test', 'h3', 0))
    return SimpleNamespace(**locals())


def test_h3_stages_real_six_stage_two_updates_and_tail(tmp_path, monkeypatch):
    assert Path(h3_stages.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/algorithms/trainer/h3_stages.py'
    h = _setup(tmp_path, monkeypatch)
    assert isinstance(h.trainer, BaseTrainer)
    with pytest.raises(RuntimeError, match='already bound'):
        h.stages.trainer()
    before = [p.detach().clone() for p in h.model.parameters()]
    for step in range(2):
        start = len(h.events)
        result = h.trainer.run_iteration(step)
        events = h.events[start:]
        assert result.stage_order == ('prelude', 'rollout', 'reward', 'advantage', 'credit', 'optimize')
        assert [e[1] for e in events if e[0] == 'stage'] == list(result.stage_order[1:])
        assert all(identity is h.requests[-1].identity for identity in h.identities[step*5:(step+1)*5])
        forwards = [e for e in events if e[0] == 'forward']
        keys = [(row[0], i) for row in h.requests[-1].samples for i in range(3)]
        assert [(e[2], e[3]) for e in forwards] == keys * 3
        assert [e[1] for e in forwards] == ['rollout']*6 + ['anchor']*6 + ['current']*6
        assert [e[4] for e in forwards] == [False]*12 + [True]*6
        for phase in ('rollout', 'anchor', 'current'):
            cell_gates = [e[1] for e in events if e[0] == 'gate' and e[1].startswith(phase + '/cell/')]
            last = 'backward' if phase == 'current' else 'result'
            assert cell_gates == [f'{phase}/cell/{i}/{part}' for i in range(6) for part in ('forward', last)]
        assert [(e[1], e[2]) for e in events if e[0] == 'backward'] == keys
        assert max(i for i, e in enumerate(events) if e[:2] == ('forward', 'rollout')) < next(i for i, e in enumerate(events) if e[0] == 'decode')
        assert events.count(('step',)) == events.count(('scheduler',)) == 1
        assert next(i for i, e in enumerate(events) if e[0] == 'step') < next(i for i, e in enumerate(events) if e[0] == 'data_commit')
        assert h.pools[-1].state == 'released'
        assert h.data.commits == step+1 and h.data.position == 2*(step+1)
        assert h.data.behavior == f'policy-{step+1}' and h.trainer.next_optimizer_step == step+1
        assert all(ref() is None for ref in h.media_refs)
        assert all(ref() is None for ref in h.prediction_refs)
        assert result.value.payload.local_cells == 6
        assert all(torch.isfinite(g).all() and bool((g != 0).any()) for g in h.coords[-1].gradients[0])
        assert all(p.grad is None for p in h.model.parameters())
    assert all(not torch.equal(old, new) for old, new in zip(before, h.model.parameters()))
    assert all(state['step'].item() == 2 for state in h.optimizer.state.values())
    with pytest.raises(ValueError, match='expected optimizer step'):
        h.trainer.run_iteration(1)
    h.trainer.close()
    h.prepared.close()


def test_h3_stages_zero_credit_still_backwards_adam_and_tail(tmp_path, monkeypatch):
    h = _setup(tmp_path, monkeypatch, zero=True)
    before = [p.detach().clone() for p in h.model.parameters()]
    result = h.trainer.run_iteration(0)
    assert result.value.payload.grad_norm == 0.
    assert len([e for e in h.events if e[0] == 'backward']) == 6
    assert h.events.count(('step',)) == h.events.count(('scheduler',)) == 1
    assert h.data.commits == 1 and h.trainer.next_optimizer_step == 1
    assert all(torch.equal(a, b) for a, b in zip(before, h.model.parameters()))
    assert all(state['step'].item() == 1 for state in h.optimizer.state.values())


@pytest.mark.parametrize('fault', ['preflight', 'identity', 'condition', 'score', 'rollout', 'anchor', 'current', 'version_current', 'raw_forward', 'raw_backward', 'step', 'tail'])
def test_h3_stages_fail_closed_no_reupdate_or_fake_rollback(tmp_path, monkeypatch, fault):
    h = _setup(tmp_path, monkeypatch, fault=fault)
    with pytest.raises((ValueError, RuntimeError)):
        h.trainer.run_iteration(0)
    count = h.events.count(('step',))
    assert count == (1 if fault in ('step', 'tail') else 0)
    assert h.data.commits == (1 if fault == 'tail' else 0)
    assert h.trainer.next_optimizer_step == 0
    if fault in ('preflight', 'identity', 'condition'):
        assert not [e for e in h.events if e[0] == 'forward']
        assert ('world', 'admission', False) in h.events
    if fault in ('rollout', 'anchor', 'current'):
        assert any(e[0] == 'gate' and e[1].startswith(fault + '/cell/') and e[2] for e in h.events)
    if fault in ('raw_forward', 'raw_backward'):
        # No recovery collective after a possibly broken model collective.
        last_model = max(i for i, e in enumerate(h.events) if e[0] in ('forward', 'backward'))
        assert not any(e[0] == 'gate' for e in h.events[last_model+1:])
    if h.pools:
        assert h.pools[-1].state != 'released'  # no success-tail release on failed updates
    with pytest.raises(RuntimeError, match='poisoned'):
        h.trainer.run_iteration(0)
    assert h.events.count(('step',)) == count
    h.trainer.close()
    assert all(p.state == 'released' for p in h.pools)
