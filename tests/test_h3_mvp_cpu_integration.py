"""W3 real stages + HPS + coordination in eight bounded CPU worker threads.

No production object is replaced. Leaves: H3/VAE weights, HTTP transport,
WORLD/CUDA observation, in-memory collectives, toy weight gather/gradient
reduce-scatter, and DTensor local-shard observation. This is NOT CUDA FSDP.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch.distributed.tensor import Shard

from tests.test_h3_mvp_cpu_decode import _artifacts, _decoder, _leaf_setup
from tests.test_h3_mvp_cpu_hps import _request, _Transport, _fp32
from tests.test_h3_mvp_cpu_coordination import MeshObservation
import visual_rl
from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.rewards.clients import hpsv3
from visual_rl.algorithms.rewards.h3_hps import H3HPSReward
from visual_rl.algorithms.trainer.grpo import GRPOTrainer
from visual_rl.algorithms.trainer.h3_stages import H3IterationRequest, H3IterationStages
from visual_rl.algorithms.trainer.interface import IterationIdentity, PrepareRunContext
from visual_rl.core.protocols.flash_hpsv3 import VideoContract
from visual_rl.models.implementations.h3_forward import h3_t2va_forward
from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry
from visual_rl.models.lifecycle import h3_coordination as coordination
from visual_rl.models.lifecycle.components import ComponentBinding, ComponentRole, ModelComponents
from visual_rl.models.lifecycle.h3_fsdp import H3PreparedModel
from visual_rl.models.lifecycle.prepared import PreparedModelBundle


class _MemoryWorld:
    """Test communication leaf with exact per-rank ordering and finite waits."""
    def __init__(self, monkeypatch):
        self.group, self.tls = object(), threading.local()
        self.lock = threading.Condition()
        self.slots, self.trace, self.positions = {}, [], [0]*8
        self.cancelled = None
        dist = coordination.dist
        monkeypatch.setattr(dist, 'group', SimpleNamespace(WORLD=self.group))
        monkeypatch.setattr(dist, 'is_available', lambda: True)
        monkeypatch.setattr(dist, 'is_initialized', lambda: True)
        monkeypatch.setattr(dist, 'get_backend', lambda pg: 'nccl')
        monkeypatch.setattr(dist, 'get_world_size', lambda pg: 8)
        monkeypatch.setattr(dist, 'get_rank', lambda pg: self.tls.rank)
        monkeypatch.setattr(dist, 'all_gather_object', self.gather)
        monkeypatch.setattr(dist, 'all_reduce', self.reduce)
        def device(requested):
            assert requested == torch.device(f'cuda:{self.tls.rank}')
            return torch.device('cpu')
        monkeypatch.setattr(coordination, '_cuda_device', device)

    def exchange(self, rank, kind, value):
        with self.lock:
            index = self.positions[rank]
            self.positions[rank] += 1
            slot = self.slots.setdefault(index, dict(kind=kind, values={}))
            assert slot['kind'] == kind and rank not in slot['values'], (index, kind, slot['kind'])
            slot['values'][rank] = value
            if len(slot['values']) == 8:
                values = [slot['values'][r] for r in range(8)]
                self.trace.append((kind, values))
                if kind == 'receipt':
                    slot['result'] = values
                elif kind == 'weights':
                    slot['result'] = torch.stack(values)
                elif kind == 'backward':
                    slot['result'] = torch.stack(values).sum(0) / 8
                else:
                    assert kind == 'norm'
                    slot['result'] = torch.stack(values).sum()
                self.lock.notify_all()
            ready = self.lock.wait_for(lambda: 'result' in slot or self.cancelled is not None, timeout=20)
            if not ready:
                raise TimeoutError(f'eight CPU ranks did not join collective {index}/{kind}')
            if self.cancelled is not None:
                raise RuntimeError('CPU peer failed before collective') from self.cancelled
            return slot['result']

    def gather(self, outputs, raw, *, group):
        assert group is self.group and len(outputs) == 8
        assert type(raw) is bytes and len(raw) < 256*1024
        entry = json.loads(raw)  # Only small receipts, NEVER model or AV payloads.
        if entry[1] == 'coordination' and entry[5]:
            event = entry[4][2]
            if event.split('/')[0] in ('rollout', 'anchor', 'current'):
                self.tls.phase = event.split('/')[0]
        outputs[:] = self.exchange(self.tls.rank, 'receipt', raw)

    def reduce(self, total, *, op, group):
        assert group is self.group and op == coordination.dist.ReduceOp.SUM
        assert total.shape == () and total.dtype == torch.float64
        total.copy_(self.exchange(self.tls.rank, 'norm', total.detach().clone()))

    def run(self, work):
        def call(rank):
            self.tls.rank, self.tls.phase = rank, None
            try:
                return work(rank)
            except Exception as exc:
                if not isinstance(exc, coordination.H3CoordinationError):
                    with self.lock:
                        self.cancelled = exc
                        self.lock.notify_all()
                return exc
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(call, r) for r in range(8)]
            return [future.result(timeout=90) for future in futures]


class _ToyGather(torch.autograd.Function):
    """Only the sharded model communication leaf, not training/norm control.

    Eight unique [2] shards form a toy [8,2] weight. Backward SUM/8 followed
    by selection of the owning row mimics an averaged reduce-scatter for this
    toy model. This does not exercise real DTensor, FSDP hooks, AC, or NCCL.
    """
    @staticmethod
    def forward(ctx, local, world, rank, events):
        ctx.world, ctx.rank, ctx.events = world, rank, events
        return world.exchange(rank, 'weights', local.detach().clone()).clone()

    @staticmethod
    def backward(ctx, gradient):
        ctx.events.append(('backward',))
        reduced = ctx.world.exchange(ctx.rank, 'backward', gradient.detach().clone())
        return reduced[ctx.rank].clone(), None, None, None


class _ToyH3(torch.nn.Module):
    def __init__(self, world, rank, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.12, -.09]))
        self.world, self.rank, self.events = world, rank, events

    def forward(self, **kwargs):
        full = _ToyGather.apply(self.weight, self.world, self.rank, self.events)
        condition = kwargs['encoder_hidden_states'].mean()
        return (kwargs['hidden_states'].float() * .04 + full[:, 0].mean() + condition,
                kwargs['audio_hidden_states'].float() * .07 + full[:, 1].mean() + condition)


def _setup(tmp_path, monkeypatch, *, zero=False, fault=None):
    world = _MemoryWorld(monkeypatch)
    geometry = H3T2VAGeometry(32, 32, 124)
    dynamics = H3CPSDynamics(geometry.joint_layout, H3ScheduleSnapshot.build(2))
    _artifacts(tmp_path / 'vae')
    _leaf_setup(monkeypatch, geometry)  # Tiny VAE classes ONLY; real decoder below.
    owners, transports, before, aux_order = [], [], [], []
    aux_lock, active, peak = threading.Lock(), [0], [0]
    contract = VideoContract(124, 32, 32)
    for rank in range(8):
        request, health = _request(tmp_path / f'hps-{rank}', contract)
        transport = _Transport(health)
        if rank == 3 and fault == 'hps':
            transport.score_mutation = lambda payload: payload | {'sample_id': 'wrong'}
        transports.append(transport)
        scorer = H3HPSReward(request, video_contract=contract)
        assert not transport.calls
        events, coords, pools, credit_snapshots, media_refs, step_gradients = [], [], [], [], [], []
        model = _ToyH3(world, rank, events).eval()
        bundle = PreparedModelBundle(ModelComponents((ComponentBinding(
            'transformer', model, (ComponentRole.INFERENCE, ComponentRole.TRAINABLE)),)))
        optimizer = torch.optim.AdamW(bundle.trainable_parameters(), lr=1e-3, weight_decay=0)
        optimizer.register_step_post_hook(lambda *args, events=events: events.append(('step',)))
        optimizer.register_step_pre_hook(
            lambda *args, model=model, saved=step_gradients: saved.append(model.weight.grad.detach().clone()))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        original_step = scheduler.step
        def scheduler_step(*args, original=original_step, events=events, **kwargs):
            events.append(('scheduler',))
            return original(*args, **kwargs)
        monkeypatch.setattr(scheduler, 'step', scheduler_step)
        prepared = H3PreparedModel(source_bundle=bundle, prepared_root=bundle, optimizer=optimizer,
                                   scheduler=scheduler, accelerator=object())
        decoder = _decoder(tmp_path / 'vae', geometry)
        owners.append(SimpleNamespace(rank=rank, events=events, coords=coords, pools=pools, credits=credit_snapshots,
                                      model=model, prepared=prepared, optimizer=optimizer, scheduler=scheduler,
                                      decoder=decoder, scorer=scorer, media_refs=media_refs,
                                      step_gradients=step_gradients))
        before.append(model.weight.detach().clone())
    monkeypatch.setattr(hpsv3, 'requests_session', lambda: transports[world.tls.rank])

    def observe(value):
        # Explicit CPU observation leaf; real norm _trainables/_local_gradient
        # validation and all subsequent math remain untouched.
        parameter = owners[world.tls.rank].model.weight
        assert value is parameter or value is parameter.grad
        return MeshObservation(world.group), (Shard(0),), (16,), value.detach()
    monkeypatch.setattr(coordination, '_dtensor_parts', observe)

    def bind(owner):
        rank, events = owner.rank, owner.events
        class Data:
            behavior = 'policy-0'
            position = 0
            commits = 0
            aborts = 0
            def reserve(self, step):
                events.append(('reserve', step))
                samples = tuple((f'step{step}:rank{rank}:sample{i}', f'step{step}:rank{rank}:group', i) for i in range(2))
                ids = tuple(row[0] for row in samples)
                identity = IterationIdentity(step, 'source', 'train', ids, (samples[0][1],)*2, (0, 1))
                width = dynamics.layout.packed_width
                seeds = [42 if zero else 42+step*16+rank*2+i for i in range(2)]
                return H3IterationRequest(
                    identity, f'collection-{step}', self.behavior, f'policy-{step+1}', samples,
                    {sid: 'same prompt' for sid in ids},
                    {sid: torch.full((1, 2, 4), .02 if zero else .02*(rank*2+i+1)) for i, sid in enumerate(ids)},
                    dynamics, '32x32x124@24:t2va',
                    {sid: torch.randn(1, width, generator=torch.Generator().manual_seed(seed)).half()
                     for sid, seed in zip(ids, seeds)},
                    {sid: (torch.Generator().manual_seed(seed+100), torch.Generator().manual_seed(seed+200))
                     for sid, seed in zip(ids, seeds)}, 2, 2, 4, 4*(4*width+4)+4*width,
                )
            def commit(self, request):
                assert owner.stages._active.pool.state == 'ready'
                owner.pools.append(owner.stages._active.pool)
                events.append(('data_commit', request.identity.optimizer_step))
                self.position += 2
                self.commits += 1
                self.behavior = request.next_behavior_id
            def abort(self, request):
                events.append(('abort', request.identity.optimizer_step))
                self.aborts += 1
        data = Data()

        @contextmanager
        def decode(terminal):
            with aux_lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                aux_order.append(rank)
            events.append(('decode',))
            try:
                with owner.decoder.decode_one(terminal, geometry) as media:
                    owner.media_refs.append(weakref.ref(media))
                    yield media
            finally:
                events.append(('media_release',))
                with aux_lock:
                    active[0] -= 1

        def predict(prepared, *, sample_id, x, index, condition, dynamics):
            assert prepared is owner.prepared and not owner.model.training
            phase = world.tls.phase
            member = int(sample_id.rsplit('sample', 1)[1])
            assert torch.equal(condition, torch.full_like(condition, .02 if zero else .02*(rank*2+member+1)))
            events.append(('forward', phase, sample_id, index, torch.is_grad_enabled()))
            vs, _, aus, _ = dynamics.schedule.transition(index)
            output = h3_t2va_forward(prepared, x=x, geometry=geometry, text_embeddings=condition,
                                     video_sigma=vs, audio_sigma=aus)
            if fault == 'finite' and rank == 3 and phase == 'current' and index == 1:
                output = output * float('nan')
            return output

        def coordinator(plan):
            actual = coordination.H3Coordination(plan, process_group=world.group, device=f'cuda:{rank}')
            owner.coords.append(actual)
            return actual
        stages = H3IterationStages(data=data, prepared=owner.prepared, predict=predict, decode=decode,
                                   score_one=owner.scorer.score_one, coordination_factory=coordinator,
                                   process_group=world.group, rank=rank, device='cpu', read_behavior_id=lambda: data.behavior)
        trainer = stages.trainer()
        assert type(trainer) is GRPOTrainer
        for name, stage in tuple(trainer._stages.items()):
            def observed(value, name=name, stage=stage):
                events.append(('stage', name))
                result = stage(value)  # Observe, never substitute a production stage.
                assert result.identity is value.identity
                if name == 'credit':
                    owner.credits.append(result.payload.credit.base_advantage.clone())
                return result
            trainer._stages[name] = observed
        trainer.prepare_run(PrepareRunContext('integration', 'h3', 0))
        owner.data, owner.stages, owner.trainer = data, stages, trainer
    for owner in owners:
        bind(owner)
    return SimpleNamespace(world=world, owners=owners, transports=transports, before=before,
                           aux_order=aux_order, peak=peak, active=active)


def _run(h, iterations):
    def worker(rank):
        owner = h.owners[rank]
        owner.scorer.acquire()
        try:
            results = [owner.trainer.run_iteration(step) for step in range(iterations)]
            owner.results = results
            return results
        finally:
            owner.scorer.close()  # local only, never a recovery collective
    return h.world.run(worker)


def _receipts(trace):
    return [[json.loads(raw) for raw in values] for kind, values in trace if kind == 'receipt']


def _check_trace(trace, *, failed=False):
    sequences = {}
    events = []
    for entries in _receipts(trace):
        assert [entry[2] for entry in entries] == list(range(8))
        assert all(entry[1] == entries[0][1] for entry in entries)
        assert all(type(raw) is list for raw in entries)
        if entries[0][1] == 'admission':
            rows = [entry[4][1] for entry in entries]
            assert len({row[0] for table in rows for row in table}) == 16
        elif entries[0][1] == 'coordination':
            if all(entry[5] for entry in entries):
                payload = entries[0][4]
                assert all(entry[4] == payload for entry in entries)
                identity, sequence, event = payload
                key = tuple(identity)
                assert sequence == sequences.get(key, 0)
                sequences[key] = sequence+1
                events.append(event)
            else:
                assert failed
                assert sum(not entry[5] for entry in entries) == 1
                events.append('FAILED')
    return events


@pytest.mark.parametrize('zero', [False, True])
def test_h3_real_objects_eight_rank_two_iterations(tmp_path, monkeypatch, zero):
    root = Path(__file__).resolve().parents[1]
    assert Path(visual_rl.__file__).resolve().parent == root / 'visual_rl'
    for cls, relative in ((H3IterationStages, 'algorithms/trainer/h3_stages.py'),
                          (H3HPSReward, 'algorithms/rewards/h3_hps.py'),
                          (coordination.H3Coordination, 'models/lifecycle/h3_coordination.py')):
        import inspect
        assert Path(inspect.getfile(cls)).resolve() == root / 'visual_rl' / relative
    h = _setup(tmp_path, monkeypatch, zero=zero)
    results = _run(h, 2)
    assert all(isinstance(result, list) and len(result) == 2 for result in results), results
    events = _check_trace(h.world.trace)
    assert h.peak == [1] and h.active == [0]
    assert h.aux_order == [rank for _ in range(2) for rank in range(8) for _ in range(2)]
    assert [event for event in events if event.startswith('aux/')] == [
        f'aux/{rank}/{edge}' for _ in range(2) for rank in range(8) for edge in ('begin', 'end')]
    assert [kind for kind, _ in h.world.trace].count('backward') == 8  # 2 samples * 2 steps * 2 iterations
    assert [kind for kind, _ in h.world.trace].count('norm') == 2
    norms = [math.sqrt(sum(value.item() for value in values)) for kind, values in h.world.trace if kind == 'norm']
    starts = [i for i, event in enumerate(events) if event == 'coordination/init'] + [len(events)]
    assert len(starts) == 3
    for start, end in zip(starts, starts[1:]):
        iteration_events = events[start:end]
        assert (iteration_events.index('rollout/complete') < iteration_events.index('aux/0/begin')
                < iteration_events.index('aux/7/end') < iteration_events.index('anchor/cell/0/forward')
                < iteration_events.index('anchor/complete') < iteration_events.index('current/cell/0/forward')
                < iteration_events.index('current/cell/3/backward') < iteration_events.index('current/complete')
                < next(i for i, e in enumerate(iteration_events) if e.startswith('grad/preflight/'))
                < iteration_events.index('optimizer/complete') < iteration_events.index('tail/ready')
                < iteration_events.index('tail/complete'))
        for phase in ('rollout', 'anchor', 'current'):
            edge = 'backward' if phase == 'current' else 'result'
            assert [e for e in iteration_events if e.startswith(phase + '/cell/')] == [
                f'{phase}/cell/{i}/{part}' for i in range(4) for part in ('forward', edge)]
    # Independent toy reduce-scatter/accumulation reference from the actual
    # pre-reduction per-rank backward vectors, NOT from norm's input scalars.
    backwards = [values for kind, values in h.world.trace if kind == 'backward']
    for step in range(2):
        full_gradient = torch.zeros(8, 2)
        for values in backwards[step*4:(step+1)*4]:
            full_gradient += torch.stack(values).sum(0) / 8
        reference_norm = full_gradient.double().square().sum().sqrt().item()
        assert norms[step] == pytest.approx(reference_norm, rel=1e-7, abs=1e-12)
        coefficient = 1. if reference_norm == 0 else min(1., 1./reference_norm)
        for rank, owner in enumerate(h.owners):
            torch.testing.assert_close(owner.step_gradients[step], full_gradient[rank]*coefficient,
                                       rtol=1e-6, atol=1e-12)
    for rank, owner in enumerate(h.owners):
        assert owner.data.commits == 2 and owner.data.position == 4 and owner.data.behavior == 'policy-2'
        assert owner.trainer.next_optimizer_step == 2 and owner.data.aborts == 0
        assert all(pool.state == 'released' for pool in owner.pools)
        assert all(ref() is None for ref in owner.media_refs)
        assert owner.events.count(('step',)) == owner.events.count(('scheduler',)) == 2
        assert all(result.value.payload.grad_norm == pytest.approx(norms[i], abs=1e-12)
                   for i, result in enumerate(owner.results))
        assert owner.events.count(('backward',)) == 8
        assert owner.optimizer.state[owner.model.weight]['step'].item() == 2
        for step in range(2):
            start = owner.events.index(('reserve', step))
            end = owner.events.index(('data_commit', step))
            local_events = owner.events[start:end+1]
            assert [e[1] for e in local_events if e[0] == 'stage'] == ['rollout', 'reward', 'advantage', 'credit', 'optimize']
            assert max(i for i, e in enumerate(local_events) if e[0] == 'backward') < local_events.index(('step',))
            assert local_events.index(('step',)) < local_events.index(('scheduler',)) < len(local_events)-1
            forwards = [event for event in owner.events if event[0] == 'forward' and event[2].startswith(f'step{step}:')]
            keys = [(f'step{step}:rank{rank}:sample{i}', index) for i in range(2) for index in range(2)]
            assert [(e[2], e[3]) for e in forwards] == keys*3
            assert [e[1] for e in forwards] == ['rollout']*4 + ['anchor']*4 + ['current']*4
            assert [e[4] for e in forwards] == [False]*8 + [True]*4
        assert all(isinstance(c, coordination.H3Coordination) and not c._poisoned for c in owner.coords)
        transport = h.transports[rank]
        assert [call[0] for call in transport.calls] == ['GET', 'POST', 'POST', 'POST', 'POST']
        assert transport.close_count == 1 and all(response.closed for response in transport.responses)
        for index, call in enumerate(transport.calls[1:]):
            body = json.loads(call[2]['data'])
            assert len(body['jpeg_frames']) == len(transport.frame_scores[index]) == 124
            assert body['sample_id'] == f'step{index//2}:rank{rank}:sample{index%2}'
            assert body['prompt'] == 'same prompt' and 'audio' not in body
    # Independent credit reference from actual submitted JPEG-derived frame scores.
    for step in range(2):
        rewards = torch.tensor([_fp32(sum(sorted(h.transports[r].frame_scores[step*2+i], reverse=True)[:37])/37)
                                for r in range(8) for i in range(2)], dtype=torch.float64)
        expected = torch.zeros_like(rewards) if bool((rewards == rewards[0]).all()) else (
            (rewards.reshape(8, 2) - rewards.reshape(8, 2).mean(1, keepdim=True)) / (rewards.std(correction=0)+1e-4)).flatten()
        for rank, owner in enumerate(h.owners):
            torch.testing.assert_close(owner.credits[step], expected[rank*2:rank*2+2].float()[:, None].expand(2, 2))
        assert bool((expected == 0).all()) if zero else bool((expected != 0).any())
    if zero:
        assert norms == [0., 0.]
        assert all(torch.equal(before, owner.model.weight) for before, owner in zip(h.before, h.owners))
    else:
        assert all(torch.isfinite(owner.model.weight).all() and bool((before != owner.model.weight).all())
                   for before, owner in zip(h.before, h.owners))
    for owner in h.owners:
        owner.trainer.close()
        for coord in owner.coords:
            coord.close()
        owner.prepared.close()


@pytest.mark.parametrize('fault', ['hps', 'finite'])
def test_h3_real_objects_single_rank_failure_stops_every_rank(tmp_path, monkeypatch, fault):
    h = _setup(tmp_path, monkeypatch, fault=fault)
    results = _run(h, 1)
    assert all(isinstance(result, coordination.H3CoordinationError) for result in results), results
    events = _check_trace(h.world.trace, failed=True)
    assert events[-1] == 'FAILED'
    assert not any(event.startswith('optimizer/') or event.startswith('tail/') for event in events)
    assert not any(kind == 'norm' for kind, _ in h.world.trace)
    assert all(owner.data.commits == 0 and owner.trainer.next_optimizer_step == 0 for owner in h.owners)
    assert all(owner.data.aborts == 1 and owner.coords[-1]._poisoned for owner in h.owners)
    assert all(('step',) not in owner.events and not owner.optimizer.state for owner in h.owners)
    assert h.active == [0] and h.peak == [1]
    if fault == 'hps':
        assert h.aux_order == [0, 0, 1, 1, 2, 2, 3]
        assert all(not any(e[:2] == ('forward', 'anchor') for e in owner.events) for owner in h.owners)
        assert not any(kind == 'backward' for kind, _ in h.world.trace)
    else:
        assert h.aux_order == [r for r in range(8) for _ in range(2)]
        assert sum(kind == 'backward' for kind, _ in h.world.trace) == 1
    length = len(h.world.trace)
    for owner, transport in zip(h.owners, h.transports):
        assert transport.close_count == 1 and all(response.closed for response in transport.responses)
        with pytest.raises(RuntimeError, match='poisoned'):
            owner.trainer.run_iteration(0)
        with pytest.raises(coordination.H3CoordinationError, match='poisoned'):
            owner.coords[-1].gate('must-not-retry')
        owner.trainer.close()
        owner.coords[-1].close()
        owner.prepared.close()
    assert len(h.world.trace) == length  # no retry/finally rescue collective
