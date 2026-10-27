"""H3 full-model/sharded Adam DCP codec inside caller-owned shared staging.

This is NOT a checkpoint manager: no complete/latest publication, data/RNG
restore, rollback, or mid-collection recovery. The caller supplies the existing
safe-point proof after the trainer's entire commit tail has returned.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from visual_rl.algorithms.optimization.h3_world import _exchange, _world_rank
from visual_rl.artifacts.checkpoint.coordination import CheckpointSafePoint
from visual_rl.artifacts.checkpoint.protocol import CheckpointProgress
from visual_rl.models.lifecycle.h3_coordination import _trainables
from visual_rl.models.lifecycle.h3_fsdp import _canonical_parameter_names

_SCHEMA = 'visual_rl.h3-state.v1'
_DIRECTORY = 'h3-state'
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate manifest JSON key')
        result[key] = value
    return result


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode('ascii')


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('identity must be an explicit lowercase SHA256')


@dataclass(frozen=True, slots=True)
class H3StateIdentity:
    topology_id: str
    model_id: str
    semantics_id: str
    group_geometry_id: str
    world_size: int = 8

    def __post_init__(self):
        for value in (self.topology_id, self.model_id, self.semantics_id, self.group_geometry_id):
            _sha(value)
        if type(self.world_size) is not int or self.world_size != 8:
            raise ValueError('H3 state codec only supports the same WORLD8 topology')

    def payload(self):
        return dict(topology_id=self.topology_id, model_id=self.model_id, semantics_id=self.semantics_id,
                    group_geometry_id=self.group_geometry_id, world_size=self.world_size)


def _apis():
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, get_optimizer_state_dict, set_model_state_dict,
    )
    # Keep DTensors sharded, include frozen parameters, never broadcast/full-gather.
    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True,
                               ignore_frozen_params=False)
    return dcp, options, get_model_state_dict, get_optimizer_state_dict, set_model_state_dict


def _local(value, process_group, *, sharded=False):
    """Only CUDA/WORLD/DTensor observation; replaced by a CPU leaf in tests."""
    from torch.distributed.tensor import DTensor, Shard
    if not isinstance(value, torch.Tensor) or value.is_meta:
        raise ValueError('state must contain materialized tensors')
    if isinstance(value, DTensor):
        mesh = value.device_mesh
        if (mesh.device_type != 'cuda' or mesh.ndim != 1 or mesh.mesh.tolist() != list(range(8))
                or mesh.get_group(0) is not process_group or value.placements != (Shard(0),)):
            raise ValueError('state DTensor must use the same ordered WORLD8 Shard(0)')
        local = value.to_local().detach()
    else:
        if sharded:
            raise ValueError('trainable/moment state must remain genuinely sharded')
        local = value.detach()
    if local.device != torch.device('cuda', torch.cuda.current_device()) or local.layout != torch.strided:
        raise ValueError('state tensor is on the wrong device/layout')
    return local


def _tensor_spec(value):
    return [list(value.shape), str(value.dtype)]


def _group_schema(group):
    if any(group.get(k, False) for k in ('amsgrad', 'capturable', 'differentiable', 'fused')):
        raise ValueError('unsupported AdamW state layout/device policy')
    for key in ('lr', 'eps', 'weight_decay', 'initial_lr'):
        if key not in group:
            continue
        value = group[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == 'eps' and value == 0):
            raise ValueError('invalid AdamW scalar hyperparameter')
    betas = group['betas']
    if (type(betas) not in (tuple, list) or len(betas) != 2
            or any(type(v) not in (int, float) or not 0 <= v < 1 for v in betas)):
        raise ValueError('invalid AdamW betas')


def _inventory(prepared, process_group):
    _trainables(prepared)  # final objects/optimizer coverage, without invoking any optimizer helper
    optimizer = prepared.optimizer
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError('codec supports the H3 single-group AdamW only')
    group = optimizer.param_groups[0]
    _group_schema(group)
    model = prepared.prepared_root.component('transformer')
    blocks = tuple(f'transformer_blocks.{i}' for i in range(len(model.transformer_blocks)))
    params = _canonical_parameter_names(model, blocks)
    state = dict(params)
    extras = {}
    for name, module in model.named_modules():
        for leaf, buffer in module._buffers.items():
            if buffer is None:
                continue
            fqn = f'{name}.{leaf}' if name else leaf
            for block in blocks:
                fqn = fqn.replace(block + '._checkpoint_wrapped_module.', block + '.', 1)
            (extras if leaf in module._non_persistent_buffers_set else state)[fqn] = buffer
    if (set(extras) != {'rope.inv_freq'} or type(extras['rope.inv_freq']) is not torch.Tensor
            or extras['rope.inv_freq'].dtype != torch.float32):
        raise ValueError('missing/unsupported nonpersistent H3 buffer')
    if any(p.dtype != torch.float32 for p in params.values() if p.requires_grad):
        raise ValueError('H3 trainable masters must be FP32')
    for name, value in (state | extras).items():
        if isinstance(value, torch.nn.Parameter) and value.grad is not None:
            raise ValueError('checkpoint requires gradients cleared to None')
        local = _local(value, process_group, sharded=name in params and value.requires_grad)
        if not bool(torch.isfinite(local).all()):
            raise ValueError(f'nonfinite model/buffer state: {name}')
    trainable = {name: p for name, p in params.items() if p.requires_grad}
    names = {id(p): name for name, p in trainable.items()}
    groups = deepcopy([{k: v for k, v in group.items() if k != 'params'}])
    groups[0]['params'] = [names[id(p)] for p in group['params']]
    # Canonical JSON metadata stores only small primitive hyperparameters.
    groups = json.loads(_json_bytes(groups))
    scheduler = prepared.scheduler
    if scheduler is None:
        schedule = {'kind': 'none', 'state': None}
    else:
        if type(scheduler) is not torch.optim.lr_scheduler.LambdaLR or scheduler.optimizer is not optimizer:
            raise ValueError('codec only supports the prepared constant LambdaLR scheduler')
        scheduler_state = scheduler.state_dict()
        if any(value is not None for value in scheduler_state['lr_lambdas']):
            raise ValueError('stateful LR callables are unsupported')
        schedule = {'kind': 'constant-lambda.v1', 'state': json.loads(_json_bytes(scheduler_state))}
    return model, optimizer, params, trainable, state, extras, groups, schedule


def _warm_state(states, trainable, process_group, step):
    if set(states) != set(trainable.values()):
        raise ValueError('cold/partial/stale Adam state; no optimizer getter may initialize it')
    for name, parameter in trainable.items():
        state = states[parameter]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError(f'incomplete Adam fields: {name}')
        counter = state['step']
        if (type(counter) is not torch.Tensor or counter.shape != () or counter.device.type != 'cpu'
                or counter.dtype != torch.float32 or not bool(torch.isfinite(counter))
                or counter.item() != step or step <= 0):
            raise ValueError('Adam step must be finite, positive and equal committed progress')
        for field in ('exp_avg', 'exp_avg_sq'):
            value = state[field]
            if _tensor_spec(value) != _tensor_spec(parameter):
                raise ValueError(f'Adam moment shape/dtype mismatch: {name}')
            local = _local(value, process_group, sharded=True)
            if not bool(torch.isfinite(local).all()) or (field == 'exp_avg_sq' and bool((local < 0).any())):
                raise ValueError(f'invalid Adam moment: {name}')


def _schedule_schema(schedule, groups, step):
    if set(schedule) != {'kind', 'state'}:
        raise ValueError('scheduler schema fields mismatch')
    state = schedule['state']
    if schedule['kind'] == 'none':
        if state is not None:
            raise ValueError('None scheduler must have None state')
        return
    if schedule['kind'] != 'constant-lambda.v1' or not isinstance(state, dict):
        raise ValueError('unsupported scheduler schema')
    if (type(state['last_epoch']) is not int or state['last_epoch'] != step
            or type(state['_step_count']) is not int or state['_step_count'] != step + 1
            or state['lr_lambdas'] != [None]
            or state['base_lrs'] != [groups[0]['initial_lr']]
            or state['_last_lr'] != [groups[0]['lr']] or state['base_lrs'] != state['_last_lr']
            or any(state.get(key, False) is not False for key in ('_is_initial', '_get_lr_called_within_step'))):
        raise ValueError('constant scheduler counters/lr do not match committed progress')


def _progress(identity, progress):
    if not isinstance(identity, H3StateIdentity) or not isinstance(progress, CheckpointProgress):
        raise TypeError('explicit H3 identity and CheckpointProgress are required')
    if progress.global_step <= 0:
        raise ValueError('cold checkpoint progress is forbidden')


def _root(staging, *, loading):
    staging = Path(staging)
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError('caller must provide a real existing staging/checkpoint directory')
    target = staging.resolve() / _DIRECTORY
    if target.is_symlink():
        raise ValueError('codec directory cannot be a symlink')
    if loading:
        if not target.is_dir():
            raise ValueError('missing codec directory')
    elif target.exists() or any((staging / n).exists() for n in ('complete.json', 'latest.json')):
        raise ValueError('refuse existing codec state or already-published output')
    return target


def _health(rank, group, phase, payload, error):
    try:
        values = _exchange('h3-state/' + phase, rank, group, payload if error is None else None, error is None)
        if any(value != payload for value in values):
            raise ValueError('state codec ranks disagree on path/identity/schema/progress')
    except Exception as exc:
        if error is not None:
            raise exc from error
        raise


def _buffer_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _manifest(identity, progress, inventory):
    _, _, _, trainable, state, extras, groups, schedule = inventory
    return dict(schema=_SCHEMA, identity=identity.payload(), progress_id=progress.progress_id,
                step=progress.global_step, model={n: _tensor_spec(v) for n, v in state.items()},
                trainable=list(trainable), buffers={n: _tensor_spec(v) for n, v in extras.items()},
                buffer_hashes={n: _buffer_hash(v) for n, v in extras.items()},
                groups=groups, scheduler=schedule)


def _template(inventory, manifest):
    _, _, _, trainable, state, extras, _, _ = inventory
    # These are load destinations, NOT installed optimizer moments and NOT a
    # zero-gradient/dummy step. DCP fills them with saved nonzero state first.
    moments = {name: dict(step=torch.empty((), dtype=torch.float32, device='cpu'),
                         exp_avg=torch.empty_like(p), exp_avg_sq=torch.empty_like(p))
               for name, p in trainable.items()}
    return dict(model={n: v.detach() for n, v in state.items()},
                optimizer={'state': moments, 'param_groups': deepcopy(manifest['groups'])},
                buffers={n: torch.empty_like(v) for n, v in extras.items()})


def _disk_schema(path, template):
    from torch.distributed.checkpoint import FileSystemReader, DefaultSavePlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata, BytesStorageMetadata
    metadata_file = path / '.metadata'
    if (path.is_symlink() or not path.is_dir() or metadata_file.is_symlink() or not metadata_file.is_file()
            or not 0 < metadata_file.stat().st_size <= _MAX_MANIFEST_BYTES):
        raise ValueError('missing/escaping/oversized DCP metadata')
    reader = FileSystemReader(path)
    metadata = reader.read_metadata()
    planner = DefaultSavePlanner()
    planner.set_up_planner(template, is_coordinator=True)
    expected = {}
    for item in planner.create_local_plan().items:
        data = item.tensor_data
        expected[item.index.fqn] = None if data is None else [list(data.size), str(data.properties.dtype)]
    if any(not isinstance(v, (TensorStorageMetadata, BytesStorageMetadata)) for v in metadata.state_dict_metadata.values()):
        raise ValueError('unsupported DCP metadata kind')
    actual = {n: [list(v.size), str(v.properties.dtype)] if isinstance(v, TensorStorageMetadata) else None
              for n, v in metadata.state_dict_metadata.items()}
    if expected != actual:
        raise ValueError('DCP state keys/shape/dtype do not exactly match target schema')
    for entry in metadata.storage_data.values():
        relative = Path(entry.relative_path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('DCP storage path escape')
        file = path / relative
        if (type(entry.offset) is not int or type(entry.length) is not int or file.is_symlink() or not file.resolve().is_relative_to(path.resolve()) or not file.is_file()
                or entry.offset < 0 or entry.length <= 0 or file.stat().st_size < entry.offset + entry.length):
            raise ValueError('missing/truncated/escaping DCP storage file')


def _save_dcp(state, path, process_group):
    import torch.distributed.checkpoint as dcp
    dcp.save(state, checkpoint_id=path, process_group=process_group,
             storage_writer=dcp.FileSystemWriter(path, overwrite=False, sync_files=True))


def _load_dcp(state, path, process_group):
    import torch.distributed.checkpoint as dcp
    dcp.load(state, checkpoint_id=path, process_group=process_group)


def _receipt(manifest):
    return dict(relative_path=_DIRECTORY, schema=_SCHEMA, manifest_id=_digest(manifest),
                progress_id=manifest['progress_id'], step=manifest['step'])


def save_h3_state(prepared, staging, *, process_group, identity: H3StateIdentity,
                  safe_point: CheckpointSafePoint, progress: CheckpointProgress):
    """ALL ranks save to the same new child of caller-owned staging; no publish."""
    rank = _world_rank(process_group)
    error = None
    payload = None
    try:
        _progress(identity, progress)
        if not isinstance(safe_point, CheckpointSafePoint):
            raise TypeError('existing CheckpointSafePoint proof is required')
        safe_point.assert_ready(progress)
        if (safe_point.rank != rank or safe_point.world_size != 8
                or safe_point.group_geometry_id != identity.group_geometry_id):
            raise ValueError('safe-point topology/geometry mismatch')
        target = _root(staging, loading=False)
        inventory = _inventory(prepared, process_group)
        model, optimizer, _, trainable, _, extras, _, _ = inventory
        _warm_state(optimizer.state, trainable, process_group, progress.global_step)
        manifest = _manifest(identity, progress, inventory)
        _schedule_schema(manifest['scheduler'], manifest['groups'], progress.global_step)
        if len(_json_bytes(manifest)) > _MAX_MANIFEST_BYTES:
            raise ValueError('state manifest exceeds budget')
        api = _apis()
        payload = [str(target), _digest(manifest)]
    except Exception as exc:
        error = exc
    _health(rank, process_group, 'save/preflight', payload, error)
    _, options, get_model, get_optimizer, _ = api
    # Warm-state admission above precedes the potentially side-effecting helper.
    state = dict(model=get_model(model, options=options),
                 optimizer=get_optimizer(model, optimizer, options=options),
                 buffers={n: v.detach() for n, v in extras.items()})
    error = None
    try:
        if ({n: _tensor_spec(v) for n, v in state['model'].items()} != manifest['model']
                or set(state['optimizer']['state']) != set(trainable)
                or json.loads(_json_bytes(state['optimizer']['param_groups'])) != manifest['groups']):
            raise ValueError('public state getters changed full FQN/shape/dtype/group coverage')
    except Exception as exc:
        error = exc
    _health(rank, process_group, 'save/captured', payload, error)
    _save_dcp(state, target / 'dcp', process_group)
    error = None
    try:
        if rank == 0:
            with (target / 'manifest.json').open('xb') as stream:
                stream.write(_json_bytes(manifest))
                stream.flush()
                os.fsync(stream.fileno())
    except Exception as exc:
        error = exc
    _health(rank, process_group, 'save/written', payload, error)
    return _receipt(manifest)


def load_h3_state(prepared, staging, *, process_group, identity: H3StateIdentity,
                  progress: CheckpointProgress):
    """ALL ranks load into fresh empty Adam; outer manager must validate trust.

    DCP metadata is pickle-based. Call only for trusted, outer-manager-validated
    artifacts. Raw staging is not a complete/recoverable run checkpoint. No RNG
    is restored here; the enclosing resume gate restores RankRNGSnapshot last.
    """
    rank = _world_rank(process_group)
    error = None
    payload = None
    try:
        _progress(identity, progress)
        target = _root(staging, loading=True)
        file = target / 'manifest.json'
        if file.is_symlink() or not file.is_file() or file.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ValueError('missing/oversized state manifest')
        manifest = json.loads(file.read_bytes(), object_pairs_hook=_unique_object)
        inventory = _inventory(prepared, process_group)
        model, optimizer, _, trainable, _, extras, _, schedule = inventory
        if optimizer.state:
            raise ValueError('load requires genuinely fresh empty Adam state')
        expected = _manifest(identity, progress, inventory)
        # Current lr/step are fresh defaults: saved groups/scheduler values must
        # be restored, while their schema and FQN ordering must remain exact.
        if (set(manifest) != set(expected) or manifest['scheduler']['kind'] != schedule['kind']
                or len(manifest['groups']) != 1 or set(manifest['groups'][0]) != set(expected['groups'][0])
                or manifest['groups'][0]['params'] != expected['groups'][0]['params']):
            raise ValueError('checkpoint optimizer/scheduler schema mismatch')
        if set(manifest['scheduler']) != {'kind', 'state'}:
            raise ValueError('checkpoint scheduler fields mismatch')
        if schedule['kind'] == 'none':
            if manifest['scheduler']['state'] is not None:
                raise ValueError('None scheduler must have None state')
        elif set(manifest['scheduler']['state']) != set(schedule['state']):
            raise ValueError('checkpoint scheduler state schema mismatch')
        _group_schema(manifest['groups'][0])
        _schedule_schema(manifest['scheduler'], manifest['groups'], progress.global_step)
        expected['groups'] = manifest['groups']
        expected['scheduler'] = manifest['scheduler']
        if set(manifest['buffer_hashes']) != set(extras):
            raise ValueError('nonpersistent buffer identity coverage mismatch')
        for value in manifest['buffer_hashes'].values():
            _sha(value)
        expected['buffer_hashes'] = manifest['buffer_hashes']
        if manifest != expected:
            raise ValueError('checkpoint topology/model/semantics/progress/state schema mismatch')
        state = _template(inventory, manifest)
        _disk_schema(target / 'dcp', state)
        api = _apis()
        payload = [str(target), _digest(manifest)]
    except Exception as exc:
        error = exc
    _health(rank, process_group, 'load/preflight', payload, error)
    _, options, get_model, _, set_model = api
    state['model'] = get_model(model, options=options)
    _load_dcp(state, target / 'dcp', process_group)
    error = None
    try:
        osd = state['optimizer']
        if (set(osd['state']) != set(trainable)
                or json.loads(_json_bytes(osd['param_groups'])) != manifest['groups']):
            raise ValueError('loaded Adam FQN/groups mismatch')
        # Validate loaded moments without installing them into the fresh Adam.
        by_parameter = {trainable[n]: s for n, s in osd['state'].items()}
        _warm_state(by_parameter, trainable, process_group, progress.global_step)
        if {n: _buffer_hash(v) for n, v in state['buffers'].items()} != manifest['buffer_hashes']:
            raise ValueError('loaded nonpersistent buffer identity mismatch')
        for name, value in (state['model'] | state['buffers']).items():
            if not bool(torch.isfinite(_local(value, process_group)).all()):
                raise ValueError(f'nonfinite loaded model/buffer: {name}')
    except Exception as exc:
        error = exc
    _health(rank, process_group, 'load/validated', payload, error)
    incompatible = set_model(model, model_state_dict=state['model'], options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError('strict model setter returned incompatible keys')
    # Current DCP set_optimizer_state_dict calls _init_optim_state -> dummy step
    # on fresh Adam. The public Optimizer setter accepts these checked FQN keys
    # via its saved-ID -> current-Parameter mapping, without any step/getter.
    error = None
    try:
        optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            for name, value in extras.items():
                value.copy_(state['buffers'][name])
        if prepared.scheduler is not None:
            prepared.scheduler.load_state_dict(manifest['scheduler']['state'])
        _warm_state(optimizer.state, trainable, process_group, progress.global_step)
        restored = _manifest(identity, progress, _inventory(prepared, process_group))
        if restored != manifest:
            raise ValueError('applied model/optimizer/scheduler/buffer schema mismatch')
    except Exception as exc:
        error = exc
    _health(rank, process_group, 'load/applied', payload, error)
    return _receipt(manifest)
