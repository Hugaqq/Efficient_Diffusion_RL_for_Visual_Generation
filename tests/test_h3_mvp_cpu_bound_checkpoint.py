"""Real binding/source/stages -> published DCP -> fresh two-gate next iteration.

CPU replicas simulate only CUDA/FSDP/transport leaves. No fixture checkpoint
contract/identity/data replaces the binding or source's actual output. This is
NOT a RunController or real CUDA/NCCL validation.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import json
import math
import random
import threading

import numpy as np
import pytest
import torch
from torch.distributed.tensor import Shard

from tests.test_h3_mvp_cpu_binding import _fixture as _assets
from tests.test_h3_mvp_cpu_prepare import _cpu_leaves
from tests.test_h3_mvp_cpu_decode import _leaf_setup
from tests.test_h3_mvp_cpu_hps import _request, _Transport
from tests.test_h3_mvp_cpu_coordination import MeshObservation
from tests.test_h3_mvp_cpu_checkpoint_state import _snapshot, _equal
from visual_rl.algorithms.optimization import h3_world
from visual_rl.algorithms.rewards.clients import hpsv3
from visual_rl.algorithms.rewards.h3_hps import H3HPSReward
from visual_rl.algorithms.trainer.h3_stages import H3IterationStages
from visual_rl.algorithms.trainer.interface import PrepareRunContext
from visual_rl.artifacts.checkpoint import h3_state as codec, h3_transaction as transaction
from visual_rl.artifacts.checkpoint.manager import AtomicCheckpointManager, CommittedCheckpoint
from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot
from visual_rl.core.protocols.flash_hpsv3 import VideoContract
from visual_rl.data.h3_source import H3PromptSource
from visual_rl.models.implementations import h3_loading
from visual_rl.models.implementations.h3 import H3Adapter
from visual_rl.models.lifecycle import h3_fsdp, h3_coordination
from visual_rl.models.lifecycle.components import ComponentManager
from visual_rl.runtime import h3_binding
from visual_rl.runtime.h3_binding import H3RuntimeBinding


class _World:
    """Bounded in-memory communication, never a PG or a training runner."""
    def __init__(self, monkeypatch):
        self.group, self.tls = object(), threading.local()
        self.lock = threading.Condition()
        self.owners = []
        self.reset()
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)  # actual DCP is no-dist
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda group=None: self.tls.rank)
        monkeypatch.setattr(torch.distributed, 'all_gather_object', self.gather)
        monkeypatch.setattr(torch.distributed, 'all_reduce', self.reduce)
        for module in (h3_world, h3_binding, h3_coordination, codec, transaction):
            monkeypatch.setattr(module, '_world_rank', lambda group: self.tls.rank)
        for module in (h3_binding, h3_coordination):
            monkeypatch.setattr(module, '_cuda_device', lambda device: torch.device('cpu'))
        monkeypatch.setattr(h3_coordination, '_dtensor_parts', self.parts)
        monkeypatch.setattr(codec, '_local', lambda value, group, sharded=False: value.detach())
        def block_policy(block):
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
            assert isinstance(block, CheckpointWrapper)
            return dict(param_dtype='bfloat16', reduce_dtype='float32', cast_forward_inputs=False, divisor=8)
        monkeypatch.setattr(h3_binding, '_block_policy', block_policy)

    def reset(self):
        self.positions, self.slots, self.trace, self.error = [0]*8, {}, [], None

    def exchange(self, rank, kind, value):
        with self.lock:
            index = self.positions[rank]
            self.positions[rank] += 1
            slot = self.slots.setdefault(index, {'kind': kind, 'values': {}})
            assert slot['kind'] == kind and rank not in slot['values'], (index, kind, slot['kind'])
            slot['values'][rank] = value
            if len(slot['values']) == 8:
                values = [slot['values'][r] for r in range(8)]
                self.trace.append((kind, values))
                if kind.startswith('mean:'):
                    slot['result'] = torch.stack(values).sum(0) / 8
                elif kind.startswith('clip-shards:'):
                    slot['result'] = torch.cat(values, dim=0)
                elif kind == 'sum':
                    slot['result'] = torch.stack(values).sum()
                else:
                    slot['result'] = values
                self.lock.notify_all()
            if not self.lock.wait_for(lambda: 'result' in slot or self.error is not None, timeout=20):
                raise TimeoutError(f'CPU collective {index}/{kind} incomplete')
            if self.error is not None:
                raise RuntimeError('CPU peer failed') from self.error
            return slot['result']

    def gather(self, output, value, *, group):
        assert group is self.group and type(value) is bytes and len(value) <= 256*1024
        output[:] = self.exchange(self.tls.rank, 'receipt', value)

    def reduce(self, value, *, op, group):
        assert group is self.group and op == torch.distributed.ReduceOp.SUM and value.dtype == torch.float64
        value.copy_(self.exchange(self.tls.rank, 'sum', value.detach().clone()))

    @staticmethod
    def shard(value, rank):
        size = math.ceil(value.shape[0] / 8)
        start = min(rank * size, value.shape[0])
        return value.narrow(0, start, min(size, value.shape[0]-start))

    def parts(self, value):
        owner = self.owners[self.tls.rank]
        assert any(value is p or value is p.grad for _, p in owner.prepared.trainable_named_parameters())
        return MeshObservation(self.group), (Shard(0),), tuple(value.shape), self.shard(value, owner.rank).detach()

    def run(self, work):
        self.reset()
        def call(rank):
            self.tls.rank = rank
            try:
                return work(rank)
            except Exception as exc:
                with self.lock:
                    self.error = exc
                    self.lock.notify_all()
                return exc
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(call, r) for r in range(8)]
            return [future.result(timeout=90) for future in futures]


def _request_snapshot(request):
    return dict(identity=request.identity, collection=request.collection_id, behavior=request.behavior_id,
                samples=request.samples, prompts=dict(request.prompts),
                conditions={k: v.clone() for k, v in request.conditions.items()},
                initial={k: v.clone() for k, v in request.initial_states.items()},
                rng={k: tuple(g.get_state().clone() for g in pair) for k, pair in request.generators.items()})


def _envelope(binding):
    # Fresh restored source readback, NOT checkpoint_boundary on a trainer that
    # has never completed a tail. No private source/trainer counters are changed.
    source = binding.source
    state = source.state_dict()
    prefix = 'h3-batch:'
    assert source.next_global_batch_id.startswith(prefix)
    return dict(schema_version=1, rank=state['rank'], world_size=8,
                content_id=binding.validate_live()['data'], source_id=source.source_id,
                next_optimizer_step=state['next_optimizer_step'], next_global_batch_id=source.next_global_batch_id[len(prefix):],
                cursor=state['global_cursor'], behavior_id=source.read_behavior_id(), open_reservations=0, payload=state)


def _private_rng(rank):
    # Eight independent process RNG observations without sharing Python/Torch
    # global state between eight threads. Actual restore_current is exercised
    # separately, serially, below.
    seed = 1000 + rank
    n = np.random.RandomState(seed).get_state()
    return RankRNGSnapshot(rank, random.Random(seed).getstate(), n[0], tuple(int(x) for x in n[1]),
                           int(n[2]), int(n[3]), float(n[4]), torch.Generator().manual_seed(seed).get_state(), None)


def _case(tmp_path, monkeypatch):
    # Use the existing actual artifact/profile/cache setup with temporary leaf
    # patches only. No existing helper signature or production file is edited.
    with monkeypatch.context() as build:
        assets = _assets(tmp_path, build, bind=False)
    initial = {k: v.clone() for k, v in assets.adapter.meta_model.model.state_dict().items()}
    world = _World(monkeypatch)
    _leaf_setup(monkeypatch, assets.adapter.geometry)
    monkeypatch.setattr(transaction, '_capture_rng', _private_rng)
    calls, publishing = [], []
    real_publish = AtomicCheckpointManager._publish_staging
    def publish(self, *args, **kwargs):
        publishing.append(True)
        try:
            return real_publish(self, *args, **kwargs)
        finally:
            publishing.pop()
    monkeypatch.setattr(AtomicCheckpointManager, '_publish_staging', publish)
    real_api = codec._apis
    def api():
        dcp, options, gm, go, sm = real_api()
        def spy(label, fn):
            def wrapped(*args, **kwargs):
                assert not publishing, 'state API must stay outside single-writer publication'
                calls.append((label, world.tls.rank))
                return fn(*args, **kwargs)
            return wrapped
        return dcp, options, spy('get-model', gm), spy('get-adam', go), spy('set-model', sm)
    monkeypatch.setattr(codec, '_apis', api)
    def save(state, path, group):
        import torch.distributed.checkpoint as dcp
        rank = world.tls.rank
        assert not publishing
        calls.append(('save', rank))
        states = world.exchange(rank, 'dcp-inputs', state)
        if rank == 0:
            for peer in states[1:]:
                _equal(states[0], peer)  # no-dist file stands for IDENTICAL CPU replicas only
            dcp.save(state, checkpoint_id=path, no_dist=True)
        world.exchange(rank, 'dcp-written', None)
    def load(state, path, group):
        import torch.distributed.checkpoint as dcp
        assert not publishing
        calls.append(('load', world.tls.rank))
        dcp.load(state, checkpoint_id=path, no_dist=True)
    monkeypatch.setattr(codec, '_save_dcp', save)
    monkeypatch.setattr(codec, '_load_dcp', load)
    return SimpleNamespace(assets=assets, initial=initial, world=world, calls=calls, path=tmp_path, monkeypatch=monkeypatch)


def _cohort(case, label, *, use_initial=False):
    h, world, patch = case.assets, case.world, case.monkeypatch
    owners = []
    for rank in range(8):
        if use_initial and rank == 0:
            adapter, manager, prepared, scorer, transport = h.adapter, h.manager, h.prepared, h.scorer, h.transport
        else:
            with patch.context() as build:
                build.setattr(h3_loading, '_resolve_h3_class', lambda: h.HeavyLeaf)
                adapter = H3Adapter.from_config(h.profile.model.config, runtime_context=dict(
                    artifact_paths={'h3-model': h.profile.launch.model, 'h3-text': h.profile.launch.text_cache},
                    expected_cache_identity=h.profile.expected['cache_identity']))
                manager = ComponentManager(adapter, execution_device='cpu')
                manager.load()
                leaves = _cpu_leaves(adapter.meta_model, build, rank=rank,
                                     broadcast_state={k: v.clone() for k, v in case.initial.items()})
                prepared = h3_fsdp.prepare_h3_fsdp(adapter.meta_model, **leaves.args, constant_scheduler=True)
                request, health = _request(case.path / f'{label}-hps-{rank}', VideoContract(124, 32, 32))
                transport = _Transport(health)
                build.setattr(hpsv3, 'requests_session', lambda: transport)
                scorer = H3HPSReward(request, video_contract=VideoContract(124, 32, 32)).acquire()
        source = H3PromptSource(**(h.source_args | dict(adapter=adapter, rank=rank)))
        world.tls.rank = rank
        binding = H3RuntimeBinding(**(h.kwargs | dict(adapter=adapter, prepared=prepared, source=source, scorer=scorer,
            process_group=world.group, rank=rank, device=f'cuda:{rank}')))
        stages = H3IterationStages(data=source, prepared=prepared, predict=binding.predict, decode=binding.decode,
            score_one=binding.score_one, coordination_factory=binding.coordination_factory,
            process_group=world.group, rank=rank, device='cpu', read_behavior_id=source.read_behavior_id)
        trainer = stages.trainer()
        owner = SimpleNamespace(rank=rank, adapter=adapter, manager=manager, prepared=prepared, scorer=scorer,
                                transport=transport, source=source, binding=binding, stages=stages, trainer=trainer,
                                requests=[], before_tail=[], hooks=[])
        reserve = source.reserve
        def observe_reserve(step, reserve=reserve, owner=owner):
            request = reserve(step)
            owner.requests.append(_request_snapshot(request))
            return request
        patch.setattr(source, 'reserve', observe_reserve)  # observing spy delegates to REAL reserve
        for name, p in prepared.trainable_named_parameters():
            def average(gradient, rank=rank, name=name):
                return world.exchange(rank, 'mean:' + name, gradient.detach().clone()).clone()
            owner.hooks.append(p.register_hook(average))
        def assemble_gradients(optimizer, args, kwargs, owner=owner):
            # CPU replica transport leaf: actual norm clips only each owning
            # shard, then assemble those shards so every CPU replica can step.
            for name, p in owner.prepared.trainable_named_parameters():
                local = world.shard(p.grad, owner.rank).detach().clone()
                p.grad.copy_(world.exchange(owner.rank, 'clip-shards:' + name, local))
        owner.hooks.append(prepared.optimizer.register_step_pre_hook(assemble_gradients))
        def before_tail(optimizer, args, kwargs, owner=owner):
            with pytest.raises(RuntimeError, match='tail'):
                owner.binding.checkpoint_boundary(owner.stages, owner.trainer)
            owner.before_tail.append(True)
        owner.hooks.append(prepared.optimizer.register_step_post_hook(before_tail))
        owner.transaction = transaction.H3CheckpointTransaction(
            AtomicCheckpointManager(case.path / 'checkpoints'), process_group=world.group)
        owners.append(owner)
    world.owners = owners
    assert len({o.binding.contract.checkpoint_contract_id for o in owners}) == 1
    assert len({o.binding.state_identity for o in owners}) == 1
    return owners


def _success(results):
    assert not any(isinstance(item, Exception) for item in results), results
    return results


def _close(owners):
    for o in owners:
        for hook in o.hooks:
            hook.remove()
        o.trainer.close(); o.binding.close(); o.scorer.close(); o.adapter.close(); o.manager.close()


def test_bound_source_tail_publish_fresh_restore_and_real_next_iteration(tmp_path, monkeypatch):
    assert Path(h3_binding.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/runtime/h3_binding.py'
    case = _case(tmp_path, monkeypatch)
    original_rng = RankRNGSnapshot.capture_current(0)
    original = _cohort(case, 'original', use_initial=True)
    fresh = []
    try:
        for o in original:
            o.trainer.prepare_run(PrepareRunContext('bound-checkpoint', 'h3', 0))
        cold = case.world.run(lambda r: original[r].binding.checkpoint_boundary(original[r].stages, original[r].trainer))
        assert all(isinstance(error, RuntimeError) and 'tail' in str(error) for error in cold)
        assert not case.calls  # cold boundary did not reach codec
        first = _success(case.world.run(lambda r: original[r].trainer.run_iteration(0)))
        assert all(result.value.payload.grad_norm > 0 for result in first)
        assert all(any(s['exp_avg'].abs().sum() > 0 for s in o.prepared.optimizer.state.values()) for o in original)
        boundaries = _success(case.world.run(lambda r: original[r].binding.checkpoint_boundary(original[r].stages, original[r].trainer)))
        assert all(o.before_tail == [True] for o in original)
        assert all(o.trainer.next_optimizer_step == 1 and o.source.state_dict()['global_cursor'] == 8 for o in original)
        def save(r):
            o = original[r]
            progress, proof, data, _ = boundaries[r]
            assert data['payload'] == o.source.state_dict()
            return o.transaction.checkpoint(o.prepared, contract=o.binding.contract, progress=progress,
                safe_point=proof, identity=o.binding.state_identity, bindings=o.binding.validate_live(), data_state=data)
        committed = _success(case.world.run(save))
        assert all(isinstance(c, CommittedCheckpoint) for c in committed)
        assert len({c.progress_id for c in committed}) == 1
        manifest = json.loads((committed[0].path / 'h3_transaction.json').read_bytes())
        assert len({row['rng_id'] for row in manifest['ranks']}) == 8
        assert committed[0].progress.rng_state_id == transaction._rng_table_id(manifest['ranks'])
        original[0].transaction.manager.inspect_complete(committed[0].path, expected_contract=original[0].binding.contract)
        saved = [_snapshot(o.prepared) for o in original]
        for kind in ('get-model', 'get-adam', 'save'):
            assert sorted(r for label, r in case.calls if label == kind) == list(range(8))
        continuous = _success(case.world.run(lambda r: original[r].trainer.run_iteration(1)))
        expected = [_snapshot(o.prepared) for o in original]
        expected_data = [o.source.state_dict() for o in original]
        fresh = _cohort(case, 'fresh')
        assert all(not o.prepared.optimizer.state and o.source.state_dict()['next_optimizer_step'] == 0 for o in fresh)
        def prepared_restore(r):
            o = fresh[r]
            return o.transaction.restore_prepared(o.prepared, committed[r].path, expected_contract=o.binding.contract,
                identity=o.binding.state_identity, bindings=o.binding.validate_live())
        before_rng = RankRNGSnapshot.capture_current(0)
        pending = _success(case.world.run(prepared_restore))
        assert RankRNGSnapshot.capture_current(0) == before_rng
        assert all(o.before_tail == [] for o in fresh)  # no dummy optimizer.step during restore
        assert sum(label == 'get-adam' for label, _ in case.calls) == 8  # warm SAVE only
        for label in ('load', 'set-model'):
            assert sorted(r for name, r in case.calls if name == label) == list(range(8))
        for r, o in enumerate(fresh):
            _equal(_snapshot(o.prepared), saved[r])
            assert o.source.state_dict()['next_optimizer_step'] == 0  # first gate did not restore source
            assert pending[r]['data']['payload'] == boundaries[r][2]['payload']
            assert pending[r]['rng'].state_identity == _private_rng(r).state_identity
        assert len({p['rng'].torch_cpu.numpy().tobytes() for p in pending}) == 8
        restored = {}
        real_restore = RankRNGSnapshot.restore_current
        def private_restore(self):
            restored[self.rank] = self
        # Process-private RNG simulation for eight threads, with an actual
        # serial restore_current/draw reference check immediately afterwards.
        with monkeypatch.context() as rng_patch:
            rng_patch.setattr(RankRNGSnapshot, 'restore_current', private_restore)
            def bound_restore(r):
                o = fresh[r]
                return o.transaction.restore_bound(
                    validate_live=lambda: (o.binding.contract, o.binding.state_identity, o.binding.validate_live()),
                    restore_data=o.source.load_state_dict, read_data=lambda: _envelope(o.binding))
            progress = _success(case.world.run(bound_restore))
        assert set(restored) == set(range(8))
        for r, o in enumerate(fresh):
            assert o.source.state_dict() == boundaries[r][2]['payload']
            assert restored[r].state_identity == pending[r]['rng'].state_identity
            o.trainer.prepare_run(PrepareRunContext('bound-checkpoint-resume', 'h3', progress[r].next_optimizer_step))
        real_restore(restored[0])
        draws = random.random(), float(np.random.random()), torch.rand(4)
        real_restore(pending[0]['rng'])
        reference = random.random(), float(np.random.random()), torch.rand(4)
        assert draws[:2] == reference[:2] and torch.equal(draws[2], reference[2])
        resumed = _success(case.world.run(lambda r: fresh[r].trainer.run_iteration(1)))
        next_boundaries = _success(case.world.run(lambda r: fresh[r].binding.checkpoint_boundary(fresh[r].stages, fresh[r].trainer)))
        for r, o in enumerate(fresh):
            _equal(o.requests[0], original[r].requests[1])
            assert resumed[r] == continuous[r]
            _equal(_snapshot(o.prepared), expected[r])
            assert o.source.state_dict() == expected_data[r]
            assert o.source.read_behavior_id() == original[r].source.read_behavior_id()
            assert o.trainer.next_optimizer_step == 2
            assert next_boundaries[r][0].global_step == 2
        assert all(result.stage_order == ('prelude', 'rollout', 'reward', 'advantage', 'credit', 'optimize') for result in first + resumed)
    finally:
        original_rng.restore_current()
        _close(original + fresh)


def test_bound_checkpoint_real_rng_last_and_between_gate_drift(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    original = _cohort(case, 'original', use_initial=True)
    cohorts = [original]
    saved_rng = RankRNGSnapshot.capture_current(0)
    try:
        for o in original:
            o.trainer.prepare_run(PrepareRunContext('bound-save', 'h3', 0))
        _success(case.world.run(lambda r: original[r].trainer.run_iteration(0)))
        boundaries = _success(case.world.run(lambda r: original[r].binding.checkpoint_boundary(original[r].stages, original[r].trainer)))
        def save(r):
            o = original[r]
            progress, proof, data, _ = boundaries[r]
            return o.transaction.checkpoint(o.prepared, contract=o.binding.contract, progress=progress,
                safe_point=proof, identity=o.binding.state_identity, bindings=o.binding.validate_live(), data_state=data)
        committed = _success(case.world.run(save))
        for mode in ('serial-rng', 'cache-drift', 'checkpoint-drift'):
            fresh = _cohort(case, mode)
            cohorts.append(fresh)
            pending = _success(case.world.run(lambda r: fresh[r].transaction.restore_prepared(
                fresh[r].prepared, committed[r].path, expected_contract=fresh[r].binding.contract,
                identity=fresh[r].binding.state_identity, bindings=fresh[r].binding.validate_live())))
            if mode == 'serial-rng':
                # A real process has private globals. Exercise rank0 serially
                # with receipt transport alone replaced, avoiding thread races.
                case.world.tls.rank = 0
                o, trace = fresh[0], []
                real_restore = RankRNGSnapshot.restore_current
                def echo(output, raw, *, group):
                    row = json.loads(raw)
                    output[:] = [json.dumps(row[:2] + [r] + row[3:]).encode() for r in range(8)]
                def live():
                    trace.append('live')
                    return o.binding.contract, o.binding.state_identity, o.binding.validate_live()
                def data(payload):
                    trace.append('data')
                    o.source.load_state_dict(payload)
                    random.random(); np.random.random(); torch.rand(1)
                def read_data():
                    trace.append('read-data')
                    return _envelope(o.binding)
                def rng(snapshot):
                    trace.append('rng-last')
                    real_restore(snapshot)
                with monkeypatch.context() as leaf:
                    leaf.setattr(torch.distributed, 'all_gather_object', echo)
                    leaf.setattr(RankRNGSnapshot, 'restore_current', rng)
                    restored = o.transaction.restore_bound(validate_live=live, restore_data=data, read_data=read_data)
                assert restored.next_optimizer_step == 1 and trace == ['live', 'data', 'read-data', 'rng-last']
                assert o.source.state_dict() == boundaries[0][2]['payload']
                assert RankRNGSnapshot.capture_current(0).state_identity == pending[0]['rng'].state_identity
                draws = random.random(), float(np.random.random()), torch.rand(4)
                real_restore(pending[0]['rng'])
                reference = random.random(), float(np.random.random()), torch.rand(4)
                assert draws[:2] == reference[:2] and torch.equal(draws[2], reference[2])
                continue
            changed = (case.assets.profile.launch.text_cache / 'index.json' if mode == 'cache-drift'
                       else next((committed[0].path / 'h3-state/dcp').glob('*.distcp')))
            raw = changed.read_bytes()
            before_rng = RankRNGSnapshot.capture_current(0)
            codec_calls = list(case.calls)
            restored_data = []
            try:
                changed.write_bytes(raw + b' ')  # real bytes change between the two gates
                def restore(r):
                    o = fresh[r]
                    def data(payload):
                        restored_data.append(r)
                        o.source.load_state_dict(payload)
                    return o.transaction.restore_bound(
                        validate_live=lambda: (o.binding.contract, o.binding.state_identity, o.binding.validate_live()),
                        restore_data=data, read_data=lambda: _envelope(o.binding))
                errors = case.world.run(restore)
                assert all(isinstance(e, Exception) for e in errors), errors
                assert all(o.transaction._poisoned for o in fresh)
                assert not restored_data and case.calls == codec_calls
                assert RankRNGSnapshot.capture_current(0) == before_rng
            finally:
                changed.write_bytes(raw)
            assert all(o.source.state_dict()['next_optimizer_step'] == 0 for o in fresh)
    finally:
        saved_rng.restore_current()
        for owners in cohorts:
            _close(owners)
