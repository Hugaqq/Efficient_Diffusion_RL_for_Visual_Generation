"""Real disk DCP and cold/fresh Adam CPU evidence; no PG/network/GPU."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from tests.test_h3_mvp_cpu import _loading_fixture, _load_cpu
from visual_rl.algorithms.dynamics.selection import DynamicsSelectionPolicyState
from visual_rl.artifacts.checkpoint import h3_state as codec
from visual_rl.artifacts.checkpoint.coordination import CheckpointSafePoint
from visual_rl.artifacts.checkpoint.protocol import CheckpointProgress
from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot
from visual_rl.models.lifecycle.components import ComponentBinding, ComponentRole, ModelComponents
from visual_rl.models.lifecycle.h3_fsdp import H3PreparedModel
from visual_rl.models.lifecycle.prepared import PreparedModelBundle, PreparedBundleError


def _identity():
    return codec.H3StateIdentity('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64)


def _progress(step=2):
    return CheckpointProgress(global_step=step, iteration=step, next_optimizer_step=step,
        next_source_id='train', next_prompt_batch_id='1' * 64, next_phase_id='train',
        active_reward_ids=('hps',), source_cursors=(('train', step * 16),),
        dynamics_selection_policy=DynamicsSelectionPolicyState(base_seed=42),
        gradient_accumulation_position=0, ema_state_saved=False, reference_state_saved=False,
        execution_transform_plan_id='2' * 64, rng_state_id='3' * 64)


def _proof(step=2, rank=0):
    # Explicit CPU caller-proof fixture, not evidence of a real trainer tail.
    return CheckpointSafePoint(rank, 8, 'committed', step, 0, 0, 0, True, 0, False, 'd' * 64)


def _prepared(path, monkeypatch, *, scheduler=True, lr=1e-4):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
    path.mkdir()
    meta, _, _ = _loading_fixture(path, monkeypatch)
    model = meta.model
    model.load_state_dict(_load_cpu(meta).state, strict=True, assign=True)
    for index, block in enumerate(model.transformer_blocks):
        model.transformer_blocks[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    model.eval()
    bundle = PreparedModelBundle(ModelComponents((ComponentBinding(
        'transformer', model, (ComponentRole.INFERENCE, ComponentRole.TRAINABLE)),)))
    optimizer = torch.optim.AdamW(bundle.trainable_parameters(), lr=lr, eps=1e-12, weight_decay=0.)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.) if scheduler else None
    handle = H3PreparedModel(source_bundle=bundle, prepared_root=bundle, optimizer=optimizer,
                             scheduler=schedule, accelerator=object())
    return handle


def _step(prepared):
    video = torch.linspace(-.2, .4, 192).reshape(1, 2, 96)
    audio = torch.linspace(-.3, .6, 96).reshape(1, 3, 32)
    v, a = prepared(video, audio)
    (v.square().mean() + a.square().mean()).backward()
    prepared.optimizer.step()
    if prepared.scheduler is not None:
        prepared.scheduler.step()
    prepared.optimizer.zero_grad(set_to_none=True)


def _snapshot(prepared):
    model = prepared.prepared_root.component('transformer')
    return deepcopy(dict(model=model.state_dict(), buffers=dict(model.named_buffers()),
                         optimizer=prepared.optimizer.state_dict(),
                         scheduler=None if prepared.scheduler is None else prepared.scheduler.state_dict(),
                         gradients={n: p.grad for n, p in model.named_parameters()}))


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _cpu_transport(monkeypatch, *, rank=0):
    # Only admission/device/transport leaves differ. Actual DCP save/load run
    # on the real filesystem in explicit no-dist mode, never an in-memory dict.
    import torch.distributed.checkpoint as dcp
    group, events = object(), []
    monkeypatch.setattr(codec, '_world_rank', lambda pg: rank)
    monkeypatch.setattr(codec, '_local', lambda value, pg, sharded=False: value.detach())
    def gather(output, payload, *, group):
        row = json.loads(payload)
        assert row[2] == rank and type(payload) is bytes
        events.append(row[1])
        output[:] = [json.dumps(row[:2] + [r] + row[3:]).encode() for r in range(8)]
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    def save(state, path, pg):
        assert pg is group
        events.append('dcp.save')
        dcp.save(state, checkpoint_id=path, no_dist=True)
    def load(state, path, pg):
        assert pg is group
        events.append('dcp.load')
        dcp.load(state, checkpoint_id=path, no_dist=True)
    monkeypatch.setattr(codec, '_save_dcp', save)
    monkeypatch.setattr(codec, '_load_dcp', load)
    return group, events


def _save(prepared, path, group, **changes):
    return codec.save_h3_state(prepared, path, process_group=group, **(dict(
        identity=_identity(), safe_point=_proof(), progress=_progress()) | changes))


def _load(prepared, path, group, **changes):
    return codec.load_h3_state(prepared, path, process_group=group,
                               **(dict(identity=_identity(), progress=_progress()) | changes))


def test_h3_codec_real_disk_save_purity_fresh_adam_restore_and_next_step(tmp_path, monkeypatch):
    assert Path(codec.__file__).resolve() == Path(__file__).resolve().parents[1] / 'visual_rl/artifacts/checkpoint/h3_state.py'
    source = _prepared(tmp_path / 'source', monkeypatch)
    _step(source)
    _step(source)
    assert source.optimizer.state and any(s['exp_avg'].abs().sum() > 0 for s in source.optimizer.state.values())
    model = source.prepared_root.component('transformer')
    model.rope.inv_freq.mul_(.9375)  # restore saved nonpersistent state, not constructor defaults
    before, rng = _snapshot(source), RankRNGSnapshot.capture_current(0)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('step')
        raise AssertionError('codec must not call optimizer.step')
    original_step = source.optimizer.step
    monkeypatch.setattr(source.optimizer, 'step', forbidden)
    group, events = _cpu_transport(monkeypatch)
    receipt = _save(source, tmp_path, group)
    assert calls == [] and receipt['step'] == 2 and receipt['relative_path'] == 'h3-state'
    _equal(_snapshot(source), before)
    assert RankRNGSnapshot.capture_current(0) == rng
    assert (tmp_path / 'h3-state/dcp/.metadata').is_file()
    assert list((tmp_path / 'h3-state/dcp').glob('*.distcp'))
    assert not (tmp_path / 'complete.json').exists() and not (tmp_path / 'latest.json').exists()
    assert events == ['h3-state/save/preflight', 'h3-state/save/captured', 'dcp.save', 'h3-state/save/written']
    fresh = _prepared(tmp_path / 'fresh', monkeypatch, lr=9e-4)
    assert fresh.optimizer.state == {} and fresh.scheduler.last_epoch == 0
    with torch.no_grad():
        fresh_model = fresh.prepared_root.component('transformer')
        fresh_model.context_embedder.weight.add_(123.)  # frozen base must be restored too
        fresh_model.time_embedder.scale.add_(17.)  # persistent buffer is part of full model
    original_fresh_step = fresh.optimizer.step
    monkeypatch.setattr(fresh.optimizer, 'step', forbidden)
    # Also prove non-owner enters actual disk load/getter/model/Adam setters.
    group, events = _cpu_transport(monkeypatch, rank=1)
    before_load_rng = RankRNGSnapshot.capture_current(0)
    loaded = _load(fresh, tmp_path, group)
    assert loaded == receipt and calls == []
    assert RankRNGSnapshot.capture_current(0) == before_load_rng
    assert events == ['h3-state/load/preflight', 'dcp.load', 'h3-state/load/validated', 'h3-state/load/applied']
    _equal(_snapshot(fresh), before)
    assert fresh.optimizer.param_groups[0]['lr'] == 1e-4
    assert all(state['step'].item() == 2 for state in fresh.optimizer.state.values())
    with pytest.raises(PreparedBundleError, match='bypassed'):
        fresh.prepared_root.component('transformer')(torch.ones(1, 2, 96), torch.ones(1, 3, 32))
    monkeypatch.setattr(source.optimizer, 'step', original_step)
    monkeypatch.setattr(fresh.optimizer, 'step', original_fresh_step)
    _step(source)
    _step(fresh)
    _equal(_snapshot(fresh), _snapshot(source))
    source.close()
    fresh.close()


def test_h3_codec_cold_partial_grad_and_unsafe_proofs_precede_helpers(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / 'model', monkeypatch)
    group, events = _cpu_transport(monkeypatch)
    calls = []
    original_api = codec._apis
    def forbidden_api():
        calls.append('api')
        raise AssertionError('unsafe/cold snapshot must precede any state helper')
    monkeypatch.setattr(codec, '_apis', forbidden_api)
    original = _snapshot(prepared)
    for change in ({}, dict(progress=_progress(0), safe_point=_proof(0))):
        with pytest.raises(ValueError, match='rejected local input'):
            _save(prepared, tmp_path, group, **change)
    assert calls == [] and not (tmp_path / 'h3-state').exists()
    _equal(_snapshot(prepared), original)
    _step(prepared)
    _step(prepared)
    for change in (dict(open_data_reservations=1), dict(active_reward_futures=1),
                   dict(active_dynamics_sessions=1), dict(poisoned=True),
                   dict(gradient_accumulation_position=1), dict(gradients_synchronized=False),
                   dict(committed_optimizer_step=1), dict(world_size=4), dict(group_geometry_id='0'*64)):
        with pytest.raises(ValueError, match='rejected local input'):
            _save(prepared, tmp_path, group, safe_point=replace(_proof(), **change))
    parameter = next(iter(prepared.optimizer.state))
    state = prepared.optimizer.state[parameter]
    for defect in ('missing', 'field', 'step', 'shape', 'nonfinite', 'stale', 'grad'):
        saved = deepcopy(prepared.optimizer.state_dict())
        if defect == 'missing':
            del prepared.optimizer.state[parameter]
        elif defect == 'field':
            del state['exp_avg_sq']
        elif defect == 'step':
            state['step'].zero_()
        elif defect == 'shape':
            state['exp_avg'] = torch.ones(1)
        elif defect == 'nonfinite':
            state['exp_avg_sq'].fill_(float('nan'))
        elif defect == 'stale':
            prepared.optimizer.state[torch.nn.Parameter(torch.ones(1))] = {}
        else:
            parameter.grad = torch.zeros_like(parameter)
        with pytest.raises(ValueError, match='rejected local input'):
            _save(prepared, tmp_path, group)
        parameter.grad = None
        prepared.optimizer.load_state_dict(saved)
        state = prepared.optimizer.state[parameter]
    assert calls == [] and all(event == 'h3-state/save/preflight' for event in events)
    monkeypatch.setattr(codec, '_apis', original_api)
    _save(prepared, tmp_path, group)
    prepared.close()


def test_h3_codec_none_scheduler_schema_and_existing_output_refusal(tmp_path, monkeypatch):
    source = _prepared(tmp_path / 'source', monkeypatch, scheduler=False)
    _step(source)
    _step(source)
    group, _ = _cpu_transport(monkeypatch)
    _save(source, tmp_path, group)
    with pytest.raises(ValueError, match='rejected local input'):
        _save(source, tmp_path, group)
    fresh = _prepared(tmp_path / 'fresh', monkeypatch, scheduler=False)
    _load(fresh, tmp_path, group)
    _equal(_snapshot(fresh), _snapshot(source))
    with pytest.raises(ValueError, match='rejected local input'):
        _load(fresh, tmp_path, group)  # warm load is not the fresh restore gate
    wrong = _prepared(tmp_path / 'scheduler-present', monkeypatch)
    with pytest.raises(ValueError, match='rejected local input'):
        _load(wrong, tmp_path, group)
    for prepared in (source, fresh, wrong):
        prepared.close()


def test_h3_codec_identity_manifest_shape_and_missing_storage_preflight(tmp_path, monkeypatch):
    source = _prepared(tmp_path / 'source', monkeypatch)
    _step(source)
    _step(source)
    group, events = _cpu_transport(monkeypatch)
    _save(source, tmp_path, group)
    fresh = _prepared(tmp_path / 'fresh', monkeypatch)
    baseline = _snapshot(fresh)
    file = tmp_path / 'h3-state/manifest.json'
    original = file.read_bytes()
    data = json.loads(original)
    defects = [dict(schema='wrong'), dict(identity=data['identity'] | {'topology_id': '0'*64}),
               dict(identity=data['identity'] | {'model_id': '0'*64}),
               dict(identity=data['identity'] | {'semantics_id': '0'*64}), dict(step=3),
               dict(trainable=['unknown']), dict(buffers={}),
               dict(scheduler={'kind': 'none', 'state': None}),
               dict(scheduler=data['scheduler'] | {'state': data['scheduler']['state'] | {'last_epoch': 'bad'}}),
               dict(model=data['model'] | {next(iter(data['model'])): [[1], 'torch.float32']}),
               dict(groups=[data['groups'][0] | {'amsgrad': True}])]
    load_count = events.count('dcp.load')
    for defect in defects:
        file.write_text(json.dumps(data | defect))
        with pytest.raises(ValueError, match='rejected local input'):
            _load(fresh, tmp_path, group)
        _equal(_snapshot(fresh), baseline)
    file.write_text('{"schema":"first","schema":"duplicate"}')
    with pytest.raises(ValueError, match='rejected local input'):
        _load(fresh, tmp_path, group)
    file.write_bytes(original)
    storage = next((tmp_path / 'h3-state/dcp').glob('*.distcp'))
    contents = storage.read_bytes()
    storage.write_bytes(contents[:12])
    with pytest.raises(ValueError, match='rejected local input'):
        _load(fresh, tmp_path, group)
    storage.write_bytes(contents)
    meta = tmp_path / 'h3-state/dcp/.metadata'
    metadata = meta.read_bytes()
    meta.unlink()
    with pytest.raises(ValueError, match='rejected local input'):
        _load(fresh, tmp_path, group)
    meta.write_bytes(b'corrupt pickle')
    with pytest.raises(ValueError, match='rejected local input'):
        _load(fresh, tmp_path, group)
    meta.write_bytes(metadata)
    assert events.count('dcp.load') == load_count and fresh.optimizer.state == {}
    _load(fresh, tmp_path, group)
    _equal(_snapshot(fresh), _snapshot(source))
    source.close()
    fresh.close()


def test_h3_codec_real_dcp_missing_state_and_wrong_shape_rejected(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp
    source = _prepared(tmp_path / 'source', monkeypatch)
    _step(source)
    _step(source)
    group, events = _cpu_transport(monkeypatch)
    _save(source, tmp_path, group)
    fresh = _prepared(tmp_path / 'fresh', monkeypatch)
    model = source.prepared_root.component('transformer')
    _, options, get_model, get_optimizer, _ = codec._apis()
    state = dict(model=get_model(model, options=options),
                 optimizer=get_optimizer(model, source.optimizer, options=options),
                 buffers={'rope.inv_freq': model.rope.inv_freq})
    original = deepcopy(state)
    for defect in ('missing-moment', 'wrong-shape'):
        state = deepcopy(original)
        if defect == 'missing-moment':
            del state['optimizer']['state'][next(iter(state['optimizer']['state']))]['exp_avg_sq']
        else:
            state['model'][next(iter(state['model']))] = torch.zeros(1)
        dcp.save(state, checkpoint_id=tmp_path / 'h3-state/dcp', no_dist=True)
        with pytest.raises(ValueError, match='rejected local input'):
            _load(fresh, tmp_path, group)
    assert 'dcp.load' not in events and fresh.optimizer.state == {}
    source.close()
    fresh.close()


def test_h3_public_optimizer_fqn_setter_fresh_without_step():
    with pytest.raises(ValueError, match='genuinely sharded'):
        codec._local(torch.ones(2), object(), sharded=True)
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=False, strict=True))
    fresh_model = torch.nn.Linear(2, 1)
    fresh = torch.optim.AdamW(fresh_model.parameters())
    weights = deepcopy(fresh_model.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError('fresh load must not call dummy step')
    fresh.step = forbidden
    # Current DCP set_optimizer_state_dict calls _init_optim_state -> step on
    # empty Adam (reproduced during API investigation). The public optimizer
    # setter accepts checked FQN keys without that helper.
    fresh.load_state_dict(state)
    assert len(fresh.state) == 2 and all(s['step'].item() == 1 for s in fresh.state.values())
    assert any(s['exp_avg'].abs().sum() > 0 for s in fresh.state.values())
    _equal(weights, fresh_model.state_dict())


def test_h3_codec_public_dcp_leaf_options(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp
    state, group, calls = {'x': torch.ones(1)}, object(), []
    def save(value, *, checkpoint_id, process_group, storage_writer):
        assert value is state and checkpoint_id == tmp_path and process_group is group
        assert storage_writer.overwrite is False and storage_writer.sync_files is True
        calls.append('save')
    def load(value, *, checkpoint_id, process_group):
        assert value is state and checkpoint_id == tmp_path and process_group is group
        calls.append('load')
    monkeypatch.setattr(dcp, 'save', save)
    monkeypatch.setattr(dcp, 'load', load)
    codec._save_dcp(state, tmp_path, group)
    codec._load_dcp(state, tmp_path, group)
    assert calls == ['save', 'load']
    _, options, _, _, _ = codec._apis()
    assert options.strict and not options.full_state_dict and not options.cpu_offload and not options.ignore_frozen_params
