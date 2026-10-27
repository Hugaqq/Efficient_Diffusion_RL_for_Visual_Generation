"""Real manager/DCP files + eight in-memory rank controls; no PG/network/GPU."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch

from tests.test_checkpoint_coordinator import _contract, _numerics
from tests.test_h3_mvp_cpu_checkpoint_state import _prepared, _step, _snapshot, _equal, _identity, _proof, _progress
from tests.test_h3_mvp_cpu_coordination import MemoryWorld
from visual_rl.artifacts.checkpoint import h3_state as codec
from visual_rl.artifacts.checkpoint import manager as manager_module
from visual_rl.artifacts.checkpoint import h3_transaction as tx
from visual_rl.artifacts.checkpoint.manager import AtomicCheckpointManager, CheckpointStaging, CommittedCheckpoint
from visual_rl.artifacts.checkpoint.protocol import ParameterContract, OptimizerGroupContract
from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot
from visual_rl.models.lifecycle.components import ModelComponents, ComponentBinding, ComponentRole
from visual_rl.models.state.parameters import ParameterStateManager


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _contract_for(prepared):
    # Existing typed CPU contract fixture, with this actual model's projection
    # and trainable FQNs. Not a claim that production H3 G3 binding is complete.
    components = ModelComponents((ComponentBinding('transformer', prepared.prepared_root.component('transformer'),
                                  (ComponentRole.INFERENCE, ComponentRole.TRAINABLE)),))
    projection = ParameterStateManager(components).state_projection
    numerics = _numerics(projection)
    parameters = tuple(ParameterContract(name, tuple(p.shape), str(p.dtype)) for name, p in prepared.trainable_named_parameters())
    return replace(_contract(world_size=8), model_state_projection=projection,
                   model_state_projection_id=projection.projection_id, model_execution_numerics=numerics,
                   model_execution_numerics_id=numerics.execution_numerics_id, trainable_parameters=parameters,
                   optimizer_groups=(OptimizerGroupContract('lora', tuple(p.name for p in parameters), _sha('lr-contract')),))


def _bindings(contract, cache='cache'):
    return dict(model=_identity().model_id, topology=_identity().topology_id, semantics=_identity().semantics_id,
                data=_sha('source-content'), cache=_sha(cache), score=_sha('hps124'), dynamics=_sha('dual-cps-grids'),
                resources=_sha('materialized-resource-set'), code=contract.code_identity)


def _data(rank, progress):
    return dict(schema_version=1, rank=rank, world_size=8, content_id=_sha('source-content'),
                source_id=progress.next_source_id, next_optimizer_step=progress.next_optimizer_step,
                next_global_batch_id=progress.next_prompt_batch_id, cursor=dict(progress.source_cursors)['train'],
                behavior_id=f'policy-{progress.global_step}', open_reservations=0,
                payload=dict(rank=rank, cursor=dict(progress.source_cursors)['train'], behavior=f'policy-{progress.global_step}'))


def _private_rng(rank):
    # Simulated private process RNGs without racing the ONE process's globals.
    seed = 200 + rank
    numpy = np.random.RandomState(seed).get_state()
    return RankRNGSnapshot(rank, random.Random(seed).getstate(), numpy[0], tuple(int(v) for v in numpy[1]),
                           int(numpy[2]), int(numpy[3]), float(numpy[4]),
                           torch.Generator().manual_seed(seed).get_state(), None)


def _setup(tmp_path, monkeypatch):
    models = [_prepared(tmp_path / f'model-{r}', monkeypatch) for r in range(8)]
    for model in models:
        _step(model)
        _step(model)
    contract = _contract_for(models[0])
    progress = replace(_progress(), execution_transform_plan_id=contract.execution_transform_plan_id)
    sim = MemoryWorld(monkeypatch)
    sim.tls.rank = 0  # subsequent explicitly single-process restore probes run on the main thread
    # Leave real DCP in explicit no-dist mode; fake only caller WORLD admission.
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    monkeypatch.setattr(tx, '_world_rank', lambda pg: sim.tls.rank)
    monkeypatch.setattr(codec, '_world_rank', lambda pg: sim.tls.rank)
    monkeypatch.setattr(codec, '_local', lambda value, pg, sharded=False: value.detach())
    monkeypatch.setattr(tx, '_capture_rng', _private_rng)
    managers = [AtomicCheckpointManager(tmp_path / 'checkpoints') for _ in range(8)]
    owners = sim.run(lambda r: tx.H3CheckpointTransaction(managers[r], process_group=sim.group))
    assert all(isinstance(owner, tx.H3CheckpointTransaction) for owner in owners)
    calls, publishing = [], []
    real_publish = AtomicCheckpointManager._publish_staging
    def publish(self, *args, **kwargs):
        publishing.append(True)
        try:
            return real_publish(self, *args, **kwargs)
        finally:
            publishing.pop()
    monkeypatch.setattr(AtomicCheckpointManager, '_publish_staging', publish)
    original_api = codec._apis
    def apis():
        dcp, options, get_model, get_optimizer, set_model = original_api()
        def observed(label, fn):
            def invoke(*args, **kwargs):
                assert not publishing, 'state API cannot run in manager publication'
                calls.append((label, sim.tls.rank))
                return fn(*args, **kwargs)
            return invoke
        return dcp, options, observed('model-get', get_model), observed('adam-get', get_optimizer), observed('model-set', set_model)
    monkeypatch.setattr(codec, '_apis', apis)
    def save(state, path, group):
        import torch.distributed.checkpoint as dcp
        assert not publishing and group is sim.group
        calls.append(('save', sim.tls.rank))
        sim._exchange('gather', b'dcp-save-enter')
        if sim.tls.rank == 0:
            # Eight identical CPU replicas stand in for distributed transport;
            # the actual DCP file format/planner/writer executes, not an IO mock.
            dcp.save(state, checkpoint_id=path, no_dist=True)
        sim._exchange('gather', b'dcp-save-written')
    def load(state, path, group):
        import torch.distributed.checkpoint as dcp
        assert not publishing and group is sim.group
        calls.append(('load', sim.tls.rank))
        dcp.load(state, checkpoint_id=path, no_dist=True)
    monkeypatch.setattr(codec, '_save_dcp', save)
    monkeypatch.setattr(codec, '_load_dcp', load)
    return sim, owners, managers, models, contract, progress, calls


def _save(sim, owners, models, contract, progress, **kwargs):
    return sim.run(lambda r: owners[r].checkpoint(models[r], contract=contract, progress=progress,
        safe_point=_proof(rank=r), identity=_identity(), bindings=_bindings(contract),
        data_state=_data(r, progress), **kwargs))


def _all_failed(results):
    assert all(isinstance(value, Exception) for value in results), results


def test_h3_shared_staging_manager_capability_and_legacy_fault_semantics(tmp_path):
    manager = AtomicCheckpointManager(tmp_path)
    contract = _contract(world_size=8)
    progress = replace(_progress(), execution_transform_plan_id=contract.execution_transform_plan_id)
    stage = manager.create_staging(2)
    (stage.path / 'weights').write_bytes(b'owned training state')
    with pytest.raises(FileExistsError):
        manager.create_staging(2)
    with pytest.raises(ValueError, match='unknown'):
        manager.commit_existing(CheckpointStaging(stage.path, 2), contract, progress=progress)
    foreign = AtomicCheckpointManager(tmp_path)
    with pytest.raises(ValueError, match='unknown'):
        foreign.commit_existing(stage, contract, progress=progress)
    with pytest.raises(ValueError, match='step'):
        manager.commit_existing(stage, contract, progress=replace(progress, global_step=3, next_optimizer_step=3))
    committed = manager.commit_existing(stage, contract, progress=progress)
    assert manager.inspect_complete(committed.path, expected_contract=contract).committed.state_tree_id == committed.state_tree_id
    with pytest.raises(ValueError, match='consumed'):
        manager.commit_existing(stage, contract, progress=progress)
    with pytest.raises(FileExistsError):
        manager.create_staging(2)
    unknown = tmp_path / 'step-99.tmp-unknown'
    unknown.mkdir()
    with pytest.raises(ValueError, match='unknown'):
        manager.commit_existing(CheckpointStaging(unknown, 99), contract, progress=progress)
    assert unknown.is_dir()  # unknown paths are not deleted or adopted
    replaced = manager.create_staging(6)
    replaced.path.rename(tmp_path / 'kept-original-staging')
    replaced.path.mkdir()
    with pytest.raises(ValueError, match='replaced'):
        manager.commit_existing(replaced, contract, progress=replace(progress, global_step=6, next_optimizer_step=6))
    linked = manager.create_staging(7)
    linked.path.rmdir()
    linked.path.symlink_to(unknown, target_is_directory=True)
    with pytest.raises(ValueError, match='real directory'):
        manager.commit_existing(linked, contract, progress=replace(progress, global_step=7, next_optimizer_step=7))
    # Legacy writer failure and before/after rename semantics still execute the
    # SAME publication implementation. All outputs are real tmp_path files.
    for step, where in ((3, 'after_writer'), (4, 'after_complete_marker'), (5, 'after_checkpoint_rename')):
        p = replace(progress, global_step=step, next_optimizer_step=step)
        def fail(label):
            if label == where:
                raise OSError(where)
        with pytest.raises(OSError, match=where):
            manager.commit(step, contract, lambda path: (path / 'state').write_bytes(b'real'), progress=p, fault_injector=fail)
        if where == 'after_checkpoint_rename':
            assert manager.inspect_complete(tmp_path / f'step-{step}').committed.step == step
        else:
            assert not (tmp_path / f'step-{step}').exists()
            assert not list(tmp_path.glob(f'step-{step}.tmp-*'))


def test_h3_allrank_publish_private_rng_full_tree_and_prepared_restore(tmp_path, monkeypatch):
    assert Path(tx.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/artifacts/checkpoint/h3_transaction.py'
    sim, owners, managers, models, contract, progress, calls = _setup(tmp_path, monkeypatch)
    before = [_snapshot(model) for model in models]
    global_rng = RankRNGSnapshot.capture_current(0)
    results = _save(sim, owners, models, contract, progress)
    assert all(isinstance(result, CommittedCheckpoint) for result in results), results
    assert len({result.progress_id for result in results}) == 1
    committed = results[0]
    assert committed.progress.rng_state_id != progress.rng_state_id
    manifest = json.loads((committed.path / 'h3_transaction.json').read_bytes())
    assert len({row['rng_id'] for row in manifest['ranks']}) == 8
    assert committed.progress.rng_state_id == tx._rng_table_id(manifest['ranks'])
    assert (managers[0].root / 'latest.json').is_file() and (committed.path / 'complete.json').is_file()
    assert len(list((committed.path / 'h3-ranks').glob('rank-*.pt'))) == 8
    assert list((committed.path / 'h3-state/dcp').glob('*.distcp'))
    managers[0].inspect_complete(committed.path, expected_contract=contract)
    assert RankRNGSnapshot.capture_current(0) == global_rng
    for r, model in enumerate(models):
        _equal(_snapshot(model), before[r])
    for label in ('model-get', 'adam-get', 'save'):
        assert sorted(r for name, r in calls if name == label) == list(range(8))
    fresh = [_prepared(tmp_path / f'fresh-{r}', monkeypatch) for r in range(8)]
    assert all(model.optimizer.state == {} for model in fresh)
    def forbidden(*args, **kwargs):
        raise AssertionError('restore must not take a dummy optimizer step')
    for model in fresh:
        monkeypatch.setattr(model.optimizer, 'step', forbidden)
    resumed = sim.run(lambda r: tx.H3CheckpointTransaction(managers[r], process_group=sim.group))
    saved_rng = RankRNGSnapshot.capture_current(0)
    pending = sim.run(lambda r: resumed[r].restore_prepared(fresh[r], committed.path,
        expected_contract=contract, identity=_identity(), bindings=_bindings(contract)))
    assert all(isinstance(value, dict) for value in pending), pending
    assert RankRNGSnapshot.capture_current(0) == saved_rng
    for r, value in enumerate(pending):
        _equal(_snapshot(fresh[r]), before[r])
        assert value['rng'].state_identity == _private_rng(r).state_identity
        assert value['data'] == _data(r, progress)
    for label in ('load', 'model-set'):
        assert sorted(r for name, r in calls if name == label) == list(range(8))
    # Global RNG is process-wide in this CPU simulation. For the eight-worker
    # bound gate replace ONLY its final restore observation with private-process
    # state storage; a separate case below calls real restore_current sequentially.
    restored_rng, data_objects = {}, [{} for _ in range(8)]
    monkeypatch.setattr(RankRNGSnapshot, 'restore_current', lambda self: restored_rng.__setitem__(self.rank, self.state_identity))
    def bound(r):
        return resumed[r].restore_bound(validate_live=lambda: (contract, _identity(), _bindings(contract)),
            restore_data=lambda payload: data_objects[r].update(payload),
            read_data=lambda: _data(r, progress) | {'payload': dict(data_objects[r]),
                'cursor': data_objects[r]['cursor'], 'behavior_id': data_objects[r]['behavior']})
    bound_results = sim.run(bound)
    assert all(value == committed.progress for value in bound_results), bound_results
    assert restored_rng == {r: _private_rng(r).state_identity for r in range(8)}
    _all_failed(sim.run(lambda r: resumed[r].restore_bound(validate_live=lambda: None, restore_data=lambda _: None, read_data=lambda: None)))
    assert sim.trace == []
    for model in models + fresh:
        model.close()


def _single_transport(monkeypatch):
    import torch.distributed.checkpoint as dcp
    group = object()
    monkeypatch.setattr(tx, '_world_rank', lambda pg: 0)
    monkeypatch.setattr(codec, '_world_rank', lambda pg: 0)
    monkeypatch.setattr(codec, '_local', lambda value, pg, sharded=False: value.detach())
    def gather(output, value, *, group):
        row = json.loads(value)
        output[:] = [json.dumps(row[:2] + [r] + row[3:]).encode() for r in range(8)]
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    monkeypatch.setattr(codec, '_load_dcp', lambda state, path, pg: dcp.load(state, checkpoint_id=path, no_dist=True))
    return group


def test_h3_bound_refresh_rng_last_fresh_data_and_next_update(tmp_path, monkeypatch):
    sim, owners, managers, models, contract, progress, _ = _setup(tmp_path, monkeypatch)
    committed = _save(sim, owners, models, contract, progress)[0]
    assert isinstance(committed, CommittedCheckpoint), committed
    group = _single_transport(monkeypatch)
    fresh = _prepared(tmp_path / 'sequential-fresh', monkeypatch)
    resumed = tx.H3CheckpointTransaction(managers[0], process_group=group)
    original_rng = RankRNGSnapshot.capture_current(0)
    try:
        pending = resumed.restore_prepared(fresh, committed.path, expected_contract=contract,
                                           identity=_identity(), bindings=_bindings(contract))
        assert RankRNGSnapshot.capture_current(0) == original_rng
        trace, new_data = [], {}
        restore_rng = RankRNGSnapshot.restore_current
        def observed_restore(self):
            trace.append('rng-last')
            restore_rng(self)
        monkeypatch.setattr(RankRNGSnapshot, 'restore_current', observed_restore)
        def live():
            trace.append('live-refresh')
            return contract, _identity(), _bindings(contract)
        def restore(payload):
            trace.append('data')
            new_data.update(payload)
            random.random(); np.random.random(); torch.rand(1)  # deliberately consume RNG before final restore
        def read():
            trace.append('read-data')
            return _data(0, progress) | {'payload': dict(new_data), 'cursor': new_data['cursor'], 'behavior_id': new_data['behavior']}
        resumed.restore_bound(validate_live=live, restore_data=restore, read_data=read)
        assert trace == ['live-refresh', 'data', 'read-data', 'rng-last']
        assert RankRNGSnapshot.capture_current(0).state_identity == pending['rng'].state_identity
        draws = random.random(), float(np.random.random()), torch.rand(5)
        restore_rng(pending['rng'])
        reference = random.random(), float(np.random.random()), torch.rand(5)
        assert draws[:2] == reference[:2] and torch.equal(draws[2], reference[2])
        assert new_data == _data(0, progress)['payload']
        _step(models[0]); _step(fresh)
        _equal(_snapshot(fresh), _snapshot(models[0]))
    finally:
        original_rng.restore_current()
        for model in models + [fresh]:
            model.close()


def test_h3_publish_io_failures_no_partial_complete_and_latest_failure_preserves_complete(tmp_path, monkeypatch):
    for defect in ('attachment', 'dcp', 'latest'):
        case = tmp_path / defect
        case.mkdir()
        with monkeypatch.context() as patch:
            sim, owners, managers, models, contract, progress, _ = _setup(case, patch)
            if defect == 'attachment':
                real_write = tx._write_rank
                def broken(path, payload):
                    if payload['rank'] == 3:
                        raise OSError('attachment failed')
                    return real_write(path, payload)
                patch.setattr(tx, '_write_rank', broken)
            elif defect == 'dcp':
                def broken(*args):
                    raise OSError('DCP transport failed')
                patch.setattr(codec, '_save_dcp', broken)
            else:
                real_json = manager_module._atomic_json
                def fail_latest(path, value):
                    if path.name == 'latest.json':
                        raise OSError('latest IO failed after rename')
                    return real_json(path, value)
                patch.setattr(manager_module, '_atomic_json', fail_latest)
            results = _save(sim, owners, models, contract, progress)
            _all_failed(results)
            assert all(owner._poisoned for owner in owners)
            if defect == 'latest':
                assert managers[0].inspect_complete(managers[0].root / 'step-2').progress.global_step == 2
                assert not (managers[0].root / 'latest.json').exists()
                assert managers[0].latest_complete(expected_contract=contract).step == 2
            else:
                assert managers[0].latest_complete() is None
                assert not list(managers[0].root.glob('step-*/complete.json'))
            _all_failed(_save(sim, owners, models, contract, progress))
            assert sim.trace == []
            for model in models:
                model.close()


def test_h3_restore_rechecks_complete_live_and_private_rank_identity(tmp_path, monkeypatch):
    sim, owners, managers, models, contract, progress, _ = _setup(tmp_path, monkeypatch)
    committed = _save(sim, owners, models, contract, progress)[0]
    assert isinstance(committed, CommittedCheckpoint), committed
    group = _single_transport(monkeypatch)
    # Live binding changes between gates must be re-evaluated, not ignored in
    # favor of the cached expected binding from restore_prepared.
    fresh = _prepared(tmp_path / 'fresh-live', monkeypatch)
    resumed = tx.H3CheckpointTransaction(managers[0], process_group=group)
    resumed.restore_prepared(fresh, committed.path, expected_contract=contract, identity=_identity(), bindings=_bindings(contract))
    calls = []
    with pytest.raises(ValueError, match='rejected local input'):
        resumed.restore_bound(validate_live=lambda: (contract, _identity(), _bindings(contract, cache='changed-cache')),
                              restore_data=lambda payload: calls.append(payload), read_data=lambda: {})
    assert calls == [] and resumed._poisoned
    fresh.close()
    # Real manager publishes newly hashed malformed semantic fixtures, so these
    # fail PRIVATE rank/schema checks, not merely a stale outer file hash.
    for defect in ('rank', 'topology', 'data'):
        bad_manager = AtomicCheckpointManager(tmp_path / f'bad-{defect}')
        def writer(path):
            for name in ('h3-state', 'h3-ranks'):
                shutil.copytree(committed.path / name, path / name)
            manifest = json.loads((committed.path / 'h3_transaction.json').read_bytes())
            file = path / 'h3-ranks/rank-3.pt'
            payload = torch.load(file, weights_only=True)
            if defect == 'rank':
                payload['rank'] = 2
            elif defect == 'topology':
                payload['world_size'] = 4
            else:
                payload['data']['content_id'] = '0' * 64
            torch.save(payload, file)
            manifest['ranks'][3]['size'], manifest['ranks'][3]['sha256'] = tx._file_identity(file)
            manifest['ranks'][3]['data_id'] = tx._payload_digest(payload['data'])
            (path / 'h3_transaction.json').write_text(json.dumps(manifest))
        bad = bad_manager.commit(2, contract, writer, progress=committed.progress)
        other = _prepared(tmp_path / f'fresh-{defect}', monkeypatch)
        owner = tx.H3CheckpointTransaction(bad_manager, process_group=group)
        with pytest.raises(ValueError, match='rejected local input'):
            owner.restore_prepared(other, bad.path, expected_contract=contract, identity=_identity(), bindings=_bindings(contract))
        assert other.optimizer.state == {} and owner._poisoned
        other.close()
    # A partial data callback failure poisons the attempt without restoring RNG
    # or rerunning the prepared codec; no fictitious rollback of the model/data.
    other = _prepared(tmp_path / 'fresh-data-failure', monkeypatch)
    failed = tx.H3CheckpointTransaction(managers[0], process_group=group)
    failed.restore_prepared(other, committed.path, expected_contract=contract, identity=_identity(), bindings=_bindings(contract))
    rng_before = RankRNGSnapshot.capture_current(0)
    partial = {}
    def fail_data(payload):
        partial['cursor'] = payload['cursor']
        raise OSError('partial source restore')
    with pytest.raises(ValueError, match='rejected local input'):
        failed.restore_bound(validate_live=lambda: (contract, _identity(), _bindings(contract)),
                             restore_data=fail_data, read_data=lambda: {})
    assert partial == {'cursor': 32} and failed._poisoned
    assert RankRNGSnapshot.capture_current(0) == rng_before
    with pytest.raises(RuntimeError, match='poisoned'):
        failed.restore_prepared(other, committed.path, expected_contract=contract, identity=_identity(), bindings=_bindings(contract))
    other.close()
    # Tampering AFTER prepared gate is detected by the second actual tree read.
    fresh = _prepared(tmp_path / 'fresh-tamper', monkeypatch)
    owner = tx.H3CheckpointTransaction(managers[0], process_group=group)
    owner.restore_prepared(fresh, committed.path, expected_contract=contract, identity=_identity(), bindings=_bindings(contract))
    blob = next((committed.path / 'h3-state/dcp').glob('*.distcp'))
    raw = blob.read_bytes()
    blob.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match='rejected local input'):
        owner.restore_bound(validate_live=lambda: (contract, _identity(), _bindings(contract)),
                            restore_data=lambda payload: calls.append(payload), read_data=lambda: {})
    assert calls == [] and owner._poisoned
    blob.write_bytes(raw)
    (committed.path / 'h3-ranks/rank-7.pt').unlink()
    with pytest.raises(ValueError):
        managers[0].inspect_complete(committed.path, expected_contract=contract)
    for model in models + [fresh]:
        model.close()


def test_h3_publish_unsafe_data_and_midcollection_fail_before_codec(tmp_path, monkeypatch):
    sim, _, managers, models, contract, progress, calls = _setup(tmp_path, monkeypatch)
    for defect in ('reservation', 'tensor', 'cursor', 'content', 'behavior', 'cold-proof'):
        owners = sim.run(lambda r: tx.H3CheckpointTransaction(managers[r], process_group=sim.group))
        def call(r):
            data, proof = _data(r, progress), _proof(rank=r)
            if r == 4:
                if defect == 'reservation':
                    proof = replace(proof, open_data_reservations=1)
                elif defect == 'tensor':
                    data['payload']['tensor'] = torch.ones(1)
                elif defect == 'cursor':
                    data['cursor'] += 1
                elif defect == 'content':
                    data['content_id'] = '0' * 64
                elif defect == 'behavior':
                    data['behavior_id'] = 'different-boundary'
                else:
                    proof = replace(proof, committed_optimizer_step=0)
            return owners[r].checkpoint(models[r], contract=contract, progress=progress, safe_point=proof,
                                         identity=_identity(), bindings=_bindings(contract), data_state=data)
        _all_failed(sim.run(call))
        assert not calls and not list(managers[0].root.iterdir())
    for model in models:
        model.close()
