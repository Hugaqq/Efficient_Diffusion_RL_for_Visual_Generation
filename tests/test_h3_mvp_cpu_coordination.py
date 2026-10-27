"""In-memory eight-rank CPU simulation; no PG, sockets, CUDA, or fake DTensor.

Only WORLD/CUDA observations, transport, and DTensor local-shard observations
are replaced. Gates, turns, prepared parameter checks, norm math and clipping
are the actual production implementations.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import threading
import time

import pytest
import torch
from torch.distributed.tensor import Shard, Replicate, Partial

from visual_rl.algorithms.optimization.h3_world import H3WorldPlan
from visual_rl.models.lifecycle import h3_coordination as coordination
from visual_rl.models.lifecycle.components import ComponentBinding, ComponentRole, ModelComponents
from visual_rl.models.lifecycle.h3_fsdp import H3PreparedModel
from visual_rl.models.lifecycle.prepared import PreparedModelBundle


def _plan():
    return H3WorldPlan('collection', 'behavior', 2, 3, 'geometry', 'schedule', 'score',
                       tuple(tuple((f's{r}-{i}', f'occurrence-{r}', i) for i in range(2)) for r in range(8)))


class MemoryWorld:
    """A blocking in-memory collective leaf, not a distributed process group."""
    def __init__(self, monkeypatch):
        self.group = object()
        self.tls = threading.local()
        self.condition = threading.Condition()
        self.rounds = {}
        self.trace = []
        self.mutate = None
        monkeypatch.setattr(coordination.dist, 'group', type('Group', (), {'WORLD': self.group}))
        monkeypatch.setattr(coordination.dist, 'is_available', lambda: True)
        monkeypatch.setattr(coordination.dist, 'is_initialized', lambda: True)
        monkeypatch.setattr(coordination.dist, 'get_world_size', lambda group: 8)
        monkeypatch.setattr(coordination.dist, 'get_rank', lambda group: self.tls.rank)
        monkeypatch.setattr(coordination.dist, 'get_backend', lambda group: 'nccl')
        monkeypatch.setattr(coordination, '_cuda_device', self.device)
        monkeypatch.setattr(coordination.dist, 'all_gather_object', self.gather)
        monkeypatch.setattr(coordination.dist, 'all_reduce', self.reduce)

    def device(self, device):
        assert device == torch.device(f'cuda:{self.tls.rank}')
        return torch.device('cpu')

    def _exchange(self, kind, value):
        rank, index = self.tls.rank, self.tls.index
        self.tls.index += 1
        with self.condition:
            slot = self.rounds.setdefault(index, {'kind': kind, 'values': {}, 'ready': False})
            assert slot['kind'] == kind and rank not in slot['values']
            slot['values'][rank] = value
            if len(slot['values']) == 8:
                values = [slot['values'][r] for r in range(8)]
                self.trace.append((kind, values))
                try:
                    result = values if kind == 'gather' else torch.stack(values).sum()
                    slot['result'] = self.mutate(kind, result) if self.mutate else result
                except Exception as exc:
                    slot['error'] = exc
                slot['ready'] = True
                self.condition.notify_all()
            assert self.condition.wait_for(lambda: slot['ready'], timeout=8), 'rank did not join the same collective'
            if 'error' in slot:
                raise slot['error']
            return slot['result']

    def gather(self, outputs, payload, *, group):
        assert group is self.group and len(outputs) == 8
        assert type(payload) is bytes and len(payload) < 8192
        outputs[:] = self._exchange('gather', payload)

    def reduce(self, total, *, op, group):
        assert group is self.group and op == coordination.dist.ReduceOp.SUM
        assert total.shape == () and total.dtype == torch.float64 and not total.requires_grad
        total.copy_(self._exchange('sum', total.detach().clone()))

    def run(self, function):
        self.rounds, self.trace = {}, []
        def call(rank):
            self.tls.rank, self.tls.index = rank, 0
            try:
                return function(rank)
            except Exception as exc:
                return exc
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(call, rank) for rank in range(8)]
            return [future.result(timeout=15) for future in futures]

    def coordinators(self, plans=None):
        plans = plans or [_plan()] * 8
        result = self.run(lambda r: coordination.H3Coordination(
            plans[r], process_group=self.group, device=f'cuda:{r}'))
        assert all(isinstance(item, coordination.H3Coordination) for item in result), result
        return result


def _errors(results, match='poison'):
    assert all(isinstance(result, coordination.H3CoordinationError) for result in results), results
    assert all(match in str(result) for result in results)


def _events(sim):
    return [json.loads(values[0])[4][-1] for kind, values in sim.trace if kind == 'gather']


def test_h3_coordination_import_init_and_ordered_gate(monkeypatch):
    assert Path(coordination.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/models/lifecycle/h3_coordination.py'
    sim = MemoryWorld(monkeypatch)
    owners = sim.coordinators()
    assert _events(sim) == ['coordination/init']
    def work(r):
        owners[r].gate('rollout/cell/0/pre')
        owners[r].gate('rollout/cell/0/post')
        owners[r].gate('anchor/complete')
    assert sim.run(work) == [None] * 8
    assert _events(sim) == ['rollout/cell/0/pre', 'rollout/cell/0/post', 'anchor/complete']
    for _, receipts in sim.trace:
        entries = [json.loads(raw) for raw in receipts]
        assert [entry[2] for entry in entries] == list(range(8))
        assert all(entry[5] is True for entry in entries)
        assert all(entry[4] == entries[0][4] for entry in entries)
        assert entries[0][4][0] == [_plan().plan_id, 'collection', 'behavior']
    assert [json.loads(v[0])[4][1] for _, v in sim.trace] == [1, 2, 3]
    sim.run(lambda r: owners[r].close())
    assert sim.trace == []
    _errors(sim.run(lambda r: owners[r].gate('after-close')))
    assert sim.trace == []


def test_h3_coordination_local_errors_mismatches_poison_without_more_collectives(monkeypatch):
    sim = MemoryWorld(monkeypatch)
    for defect in ('event', 'sequence', 'local-error', 'status-type', 'event-type', 'event-size'):
        owners = sim.coordinators()
        if defect == 'sequence':
            owners[4]._sequence += 1
        def call(r):
            event, error = 'same-event', False
            if r == 4:
                if defect == 'event':
                    event = f's{r}-0'  # rank-specific sample key, not a local ordinal
                elif defect == 'local-error':
                    error = True
                elif defect == 'status-type':
                    error = 1
                elif defect == 'event-type':
                    event = torch.ones(2)
                elif defect == 'event-size':
                    event = 'x' * 257
            owners[r].gate(event, local_error=error)
        _errors(sim.run(call))
        assert len(sim.trace) == 1 and all(o._poisoned for o in owners)
        _errors(sim.run(lambda r: owners[r].gate('must-not-enter-model')))
        assert sim.trace == []
    plans = [_plan()] * 8
    plans[3] = replace(plans[3], behavior_id='other-policy')
    _errors(sim.run(lambda r: coordination.H3Coordination(plans[r], process_group=sim.group, device=f'cuda:{r}')))
    assert len(sim.trace) == 1
    _errors(sim.run(lambda r: coordination.H3Coordination(_plan(), process_group=sim.group,
                                                         device='cpu' if r == 2 else f'cuda:{r}')))
    assert len(sim.trace) == 1


def test_h3_coordination_aux_exact_rank_order_single_owner_and_local_result(monkeypatch):
    sim = MemoryWorld(monkeypatch)
    owners = sim.coordinators()
    lock, running, maximum, order = threading.Lock(), [0], [0], []
    def work(r):
        with lock:
            running[0] += 1
            maximum[0] = max(maximum[0], running[0])
            order.append(r)
        time.sleep(.002)  # expose overlap if the gates failed to serialize work
        with lock:
            running[0] -= 1
        return ('private-result', r)
    result = sim.run(lambda r: owners[r].aux_turns(lambda: work(r)))
    assert result == [('private-result', r) for r in range(8)]
    assert maximum == [1] and order == list(range(8)) and running == [0]
    assert _events(sim) == [f'aux/{r}/{edge}' for r in range(8) for edge in ('begin', 'end')]
    assert all(b'private-result' not in value for _, values in sim.trace for value in values)
    assert sim.run(lambda r: owners[r].gate('replay/pre')) == [None] * 8


def test_h3_coordination_aux_error_chain_and_transport_failure_no_cleanup_gate(monkeypatch):
    sim = MemoryWorld(monkeypatch)
    owners = sim.coordinators()
    order, original = [], ValueError('owned scorer failure')
    def work(r):
        order.append(r)
        if r == 3:
            raise original
        return float(r)
    errors = sim.run(lambda r: owners[r].aux_turns(lambda: work(r)))
    _errors(errors)
    assert errors[3].__cause__ is original
    assert order == [0, 1, 2, 3]
    assert _events(sim)[:-1] == [f'aux/{r}/{edge}' for r in range(3) for edge in ('begin', 'end')] + ['aux/3/begin']
    assert len(sim.trace) == 8  # exactly through failing turn's end; no retry/finally barrier
    _errors(sim.run(lambda r: owners[r].gate('next-model')))
    assert sim.trace == []
    owners = sim.coordinators()
    _errors(sim.run(lambda r: owners[r].aux_turns(None if r == 6 else lambda: order.append(r))))
    assert len(sim.trace) == 1 and order == [0, 1, 2, 3]
    owners = sim.coordinators()
    def broken(kind, result):
        raise OSError('collective leaf failed')
    sim.mutate = broken
    _errors(sim.run(lambda r: owners[r].gate('model/pre')))
    assert len(sim.trace) == 1 and all(o._poisoned for o in owners)
    _errors(sim.run(lambda r: owners[r].gate('no-rescue')))
    assert sim.trace == []


class MeshObservation:
    device_type, ndim = 'cuda', 1
    def __init__(self, group):
        self.group = group
        self.mesh = torch.arange(8)
    def size(self):
        return 8
    def get_group(self, dimension):
        assert dimension == 0
        return self.group


def _prepared_shards(monkeypatch, sim, *, zero=False):
    # Different actual local lengths, including empty shards. The DTensor
    # observation leaf alone supplies the global shapes and mesh metadata.
    vectors = [torch.tensor([3., -4., 12., 0., 2., -7., 1., 3., 8., -2., 0., 6., 1.]),
               torch.tensor([-2., 5., 7.])]
    if zero:
        vectors = [torch.zeros_like(v) for v in vectors]
    chunks = [list(v.split(math.ceil(len(v) / 8))) for v in vectors]
    for pieces in chunks:
        pieces.extend(torch.empty(0) for _ in range(8 - len(pieces)))
    observations, prepared = {}, []
    for rank in range(8):
        model = torch.nn.Module()
        for name, pieces in zip(('a', 'b'), chunks):
            model.register_parameter(name, torch.nn.Parameter(torch.zeros_like(pieces[rank])))
        # Real CPU backward produces the local gradients, including empty ones.
        (model.a.mul(chunks[0][rank]).sum() + model.b.mul(chunks[1][rank]).sum()).backward()
        for index, parameter in enumerate(model.parameters()):
            for value in (parameter, parameter.grad):
                observations[id(value)] = [MeshObservation(sim.group), (Shard(0),), tuple(vectors[index].shape), value.detach()]
        bundle = PreparedModelBundle(ModelComponents((ComponentBinding(
            'transformer', model, (ComponentRole.INFERENCE, ComponentRole.TRAINABLE)),)))
        optimizer = torch.optim.AdamW(bundle.trainable_parameters(), lr=1e-4)
        prepared.append(H3PreparedModel(source_bundle=bundle, prepared_root=bundle, optimizer=optimizer,
                                        scheduler=None, accelerator=object()))
    monkeypatch.setattr(coordination, '_dtensor_parts', lambda value: observations[id(value)])
    return prepared, observations, vectors


def _gradients(prepared):
    return [parameter.grad for _, parameter in prepared.trainable_named_parameters()]


def test_h3_coordination_unique_global_norm_clip_uneven_and_empty_shards(monkeypatch):
    sim = MemoryWorld(monkeypatch)
    owners = sim.coordinators()
    prepared, _, vectors = _prepared_shards(monkeypatch, sim)
    before = [[g.clone() for g in _gradients(p)] for p in prepared]
    assert len({g.numel() for grads in before for g in grads}) > 1
    assert any(g.numel() == 0 for grads in before for g in grads)
    global_vector = torch.cat(vectors).double()
    expected = global_vector.square().sum().sqrt().item()
    results = sim.run(lambda r: owners[r].clip_grad_norm(prepared[r], max_norm=2.5))
    assert results == pytest.approx([expected] * 8, abs=1e-12)
    assert [kind for kind, _ in sim.trace] == ['gather', 'sum']
    assert all(v.dtype == torch.float64 and v.shape == () for v in sim.trace[1][1])
    assert expected != pytest.approx(sum(torch.cat(g).double().norm().item() for g in before) / 8)
    for rank, p in enumerate(prepared):
        for current, old in zip(_gradients(p), before[rank]):
            torch.testing.assert_close(current, old * (2.5 / expected), rtol=1e-6, atol=0)
        assert p.optimizer.state == {}  # no step/zero_grad; weights remain unchanged
        assert all(not parameter.detach().any() for _, parameter in p.trainable_named_parameters())
        p.close()


def test_h3_coordination_zero_norm_is_legal_and_scale_one(monkeypatch):
    sim = MemoryWorld(monkeypatch)
    owners = sim.coordinators()
    prepared, _, _ = _prepared_shards(monkeypatch, sim, zero=True)
    results = sim.run(lambda r: owners[r].clip_grad_norm(prepared[r], max_norm=1.))
    assert results == [0.] * 8 and [kind for kind, _ in sim.trace] == ['gather', 'sum']
    for p in prepared:
        assert all(g is not None and torch.isfinite(g).all() and not g.any() for g in _gradients(p))
        p.close()


def test_h3_coordination_gradient_coverage_mesh_and_nonfinite_fail_together(monkeypatch):
    sim = MemoryWorld(monkeypatch)
    real_observer = coordination._dtensor_parts
    for defect in ('none', 'nan', 'inf', 'duplicate', 'missing', 'replicate', 'partial', 'mesh', 'mesh-order',
                   'mesh-dimension', 'shard-axis', 'shape', 'local-size', 'ordinary', 'max-zero', 'max-mismatch'):
        owners = sim.coordinators()
        prepared, observations, _ = _prepared_shards(monkeypatch, sim)
        named = prepared[2].trainable_named_parameters()
        parameter = named[0][1]
        if defect == 'none':
            parameter.grad = None
        elif defect in ('nan', 'inf'):
            parameter.grad.fill_(float(defect))
        elif defect == 'duplicate':
            prepared[2].trainable_named_parameters = lambda: (named[0], named[0])
        elif defect == 'missing':
            prepared[2].trainable_named_parameters = lambda: named[:1]
        elif defect in ('replicate', 'partial', 'shard-axis'):
            observations[id(parameter)][1] = (dict(replicate=Replicate(), partial=Partial(), **{'shard-axis': Shard(1)})[defect],)
        elif defect == 'mesh':
            observations[id(parameter)][0].group = object()
        elif defect == 'mesh-order':
            observations[id(parameter)][0].mesh = torch.arange(7, -1, -1)
        elif defect == 'mesh-dimension':
            observations[id(parameter)][0].ndim = 2
        elif defect == 'shape':
            observations[id(parameter.grad)][2] = (999,)
        elif defect == 'local-size':
            observations[id(parameter)][3] = torch.ones(99)
            observations[id(parameter.grad)][3] = torch.ones(99)
        elif defect == 'ordinary':
            bad_id = id(parameter)
            monkeypatch.setattr(coordination, '_dtensor_parts',
                                lambda value: real_observer(value) if id(value) == bad_id else observations[id(value)])
        def call(r):
            max_norm = (0. if defect == 'max-zero' else 2.) if r == 2 and defect.startswith('max-') else 1.
            return owners[r].clip_grad_norm(prepared[r], max_norm=max_norm)
        _errors(sim.run(call))
        assert [kind for kind, _ in sim.trace] == ['gather']
        assert all(owner._poisoned for owner in owners)
        _errors(sim.run(lambda r: owners[r].gate('optimizer/must-not-run')))
        assert sim.trace == []
        for p in prepared:
            p.close()


def test_h3_coordination_aggregate_failure_no_clip_and_real_dtensor_rejection(monkeypatch):
    # No CPU fallback in the actual production DTensor observer.
    with pytest.raises(ValueError, match='DTensor'):
        coordination._dtensor_parts(torch.ones(2))
    sim = MemoryWorld(monkeypatch)
    for invalid in (float('inf'), float('nan'), -1.):
        owners = sim.coordinators()
        prepared, _, _ = _prepared_shards(monkeypatch, sim)
        before = [[g.clone() for g in _gradients(p)] for p in prepared]
        sim.mutate = lambda kind, value: torch.tensor(invalid, dtype=torch.float64) if kind == 'sum' else value
        _errors(sim.run(lambda r: owners[r].clip_grad_norm(prepared[r], max_norm=1.)), match='aggregate')
        assert [kind for kind, _ in sim.trace] == ['gather', 'sum']
        for r, p in enumerate(prepared):
            assert all(torch.equal(g, old) for g, old in zip(_gradients(p), before[r]))
            p.close()
        sim.mutate = None
