"""All-rank H3 state transaction; publication belongs ONLY to the existing manager.

Caller supplies real post-tail proof and serialized boundary data. No trainer,
PG ownership, source cursor invention, or default no-op live/data callbacks.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re

import torch

from visual_rl.algorithms.optimization.h3_world import _exchange, _world_rank
from visual_rl.artifacts.checkpoint.coordination import CheckpointSafePoint
from visual_rl.artifacts.checkpoint.h3_state import H3StateIdentity, save_h3_state, load_h3_state, _receipt as _codec_receipt
from visual_rl.artifacts.checkpoint.manager import (
    AtomicCheckpointManager, CommittedCheckpoint, _atomic_json, _direct_checkpoint_path,
    _payload_digest, _read_exact_json, _fsync_directory,
)
from visual_rl.artifacts.checkpoint.protocol import CheckpointContract, CheckpointProgress
from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot

_SCHEMA = 'h3-transaction.v1'
_MANIFEST = 'h3_transaction.json'
_RANKS = 'h3-ranks'
_MAX_DATA = 64 * 1024
_MAX_RANK_FILE = 1024 * 1024
_BINDINGS = {'model', 'topology', 'semantics', 'data', 'cache', 'score', 'dynamics', 'resources', 'code'}
_DATA_KEYS = {'schema_version', 'rank', 'world_size', 'content_id', 'source_id', 'next_optimizer_step',
              'next_global_batch_id', 'cursor', 'behavior_id', 'open_reservations', 'payload'}


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected lowercase SHA256 identity')


def _json_data(value, depth=0):
    if depth > 16:
        raise ValueError('data payload nesting exceeds bound')
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_data(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_data(item, depth + 1)
        return
    raise ValueError('rank data must be finite JSON only, never tensor/generator/model state')


def _data(value, rank, progress):
    _json_data(value)
    if (type(value) is not dict or set(value) != _DATA_KEYS
            or type(value['schema_version']) is not int or value['schema_version'] != 1):
        raise ValueError('rank data envelope schema mismatch')
    for name in ('rank', 'world_size', 'next_optimizer_step', 'cursor', 'open_reservations'):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError('rank data counts must be nonnegative integers')
    if (value['rank'] != rank or value['world_size'] != 8 or value['open_reservations'] != 0
            or value['next_optimizer_step'] != progress.next_optimizer_step
            or value['next_global_batch_id'] != progress.next_prompt_batch_id
            or value['source_id'] != progress.next_source_id
            or dict(progress.source_cursors).get(value['source_id']) != value['cursor']):
        raise ValueError('rank data topology/cursor/progress/reservation mismatch')
    _sha(value['content_id'])
    _sha(value['next_global_batch_id'])
    behavior = value['behavior_id']
    if type(behavior) is not str or not behavior or behavior.strip() != behavior or len(behavior.encode()) > 256:
        raise ValueError('invalid behavior identity')
    if type(value['payload']) is not dict:
        raise ValueError('data source state payload must be a JSON object')
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > _MAX_DATA:
        raise ValueError('rank data exceeds byte budget')
    return json.loads(encoded)  # owned snapshot, no reference to live mutable source state


def _binding(contract, identity, bindings):
    if not isinstance(contract, CheckpointContract) or not isinstance(identity, H3StateIdentity):
        raise TypeError('actual CheckpointContract and H3StateIdentity required')
    if contract.world_size != 8 or type(bindings) is not dict or set(bindings) != _BINDINGS:
        raise ValueError('same WORLD8 and complete live binding identity table required')
    for value in bindings.values():
        _sha(value)
    if (bindings['model'] != identity.model_id or bindings['topology'] != identity.topology_id
            or bindings['semantics'] != identity.semantics_id or bindings['code'] != contract.code_identity):
        raise ValueError('live identities disagree with model/topology/semantics/code contract')
    return dict(contract_id=contract.checkpoint_contract_id, identity=identity.payload(), live=dict(bindings))


def _rng_table_id(rows):
    if len(rows) != 8 or [row['rank'] for row in rows] != list(range(8)):
        raise ValueError('RNG table must contain ordered ranks 0..7 exactly once')
    for row in rows:
        _sha(row['rng_id'])
    return _payload_digest({'schema_version': 1, 'rank_rng_ids': [[row['rank'], row['rng_id']] for row in rows]})


def _capture_rng(rank):
    return RankRNGSnapshot.capture_current(rank)


def _file_identity(path):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= _MAX_RANK_FILE:
        raise ValueError('missing/oversized/nonregular rank attachment')
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _write_rank(path, payload):
    root = path / _RANKS
    root.mkdir(exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('rank attachment directory must be real')
    file = root / f"rank-{payload['rank']}.pt"
    with file.open('xb') as stream:
        torch.save(payload, stream)  # ONLY existing RNG snapshot + JSON data/proof, never DTensor weights
        stream.flush()
        os.fsync(stream.fileno())
    size, digest = _file_identity(file)
    return dict(rank=payload['rank'], file=file.relative_to(path).as_posix(), size=size, sha256=digest,
                rng_id=payload['rng']['state_identity'], data_id=_payload_digest(payload['data']),
                safe_point_id=CheckpointSafePoint.from_payload(payload['safe_point']).safe_point_id)


def _read_rank(path, row, identity, progress, codec_id):
    rank = row['rank']
    if row['file'] != f'{_RANKS}/rank-{rank}.pt' or (path / _RANKS).is_symlink():
        raise ValueError('rank attachment path mismatch')
    file = path / row['file']
    if _file_identity(file) != (row['size'], row['sha256']):
        raise ValueError('rank attachment hash mismatch')
    payload = torch.load(file, map_location='cpu', weights_only=True)
    if (set(payload) != {'schema', 'rank', 'world_size', 'identity_id', 'progress_id', 'codec_id', 'safe_point', 'rng', 'data'}
            or payload['schema'] != _SCHEMA or type(payload['rank']) is not int or payload['rank'] != rank
            or payload['world_size'] != 8 or payload['identity_id'] != _payload_digest(identity.payload())
            or payload['progress_id'] != progress.progress_id or payload['codec_id'] != codec_id):
        raise ValueError('rank attachment identity/topology/schema mismatch')
    proof = CheckpointSafePoint.from_payload(payload['safe_point'])
    proof.assert_ready(progress)
    if (proof.rank != rank or proof.world_size != 8 or proof.group_geometry_id != identity.group_geometry_id
            or proof.safe_point_id != row['safe_point_id']):
        raise ValueError('rank proof identity mismatch')
    data = _data(payload['data'], rank, progress)
    rng = RankRNGSnapshot.from_checkpoint_payload(payload['rng'])
    if rng.rank != rank or rng.state_identity != row['rng_id'] or _payload_digest(data) != row['data_id']:
        raise ValueError('rank RNG/data identity mismatch')
    if rng.torch_cuda is not None and not torch.cuda.is_available():
        raise ValueError('CUDA RNG cannot restore on a CPU runtime')
    return data, rng


class H3CheckpointTransaction:
    """All-rank orchestration around the ONE AtomicCheckpointManager publisher.

    One instance per rank/attempt. Restore-prepared and restore-bound are each
    once-only; any failure poisons the instance. No retry/rollback/finally gate.
    """
    def __init__(self, manager: AtomicCheckpointManager, *, process_group):
        if not isinstance(manager, AtomicCheckpointManager):
            raise TypeError('existing AtomicCheckpointManager required')
        self.manager, self.group = manager, process_group
        self.rank = _world_rank(process_group)
        self._poisoned = False
        self._sequence = 0
        self._prepared_attempted = self._bound_attempted = False
        self._pending = None

    @contextmanager
    def _attempt(self):
        if self._poisoned:
            raise RuntimeError('H3 checkpoint attempt is poisoned')
        try:
            yield
        except BaseException:
            self._poisoned = True
            raise

    def _phase(self, label, action):
        payload, error = None, None
        try:
            payload = action()  # local metadata/file IO ONLY, never a state/model collective
        except Exception as exc:
            error = exc
        try:
            rows = _exchange(f'h3-checkpoint/{self._sequence}/{label}', self.rank, self.group,
                             payload, error is None)
        except Exception as exc:
            if error is not None:
                raise exc from error
            raise
        self._sequence += 1
        return rows

    @staticmethod
    def _same(rows):
        if any(row != rows[0] for row in rows):
            raise ValueError('checkpoint ranks disagree')
        return rows[0]

    def checkpoint(self, prepared, *, contract, progress, safe_point, identity, bindings, data_state,
                   fault_injector=None) -> CommittedCheckpoint:
        with self._attempt():
            data = rng = binding = None
            def preflight():
                nonlocal data, rng, binding
                if self._prepared_attempted and not self._bound_attempted:
                    raise ValueError('cannot save between restore gates')
                binding = _binding(contract, identity, bindings)
                if not isinstance(progress, CheckpointProgress) or progress.global_step <= 0:
                    raise ValueError('positive complete progress required')
                if not isinstance(safe_point, CheckpointSafePoint):
                    raise TypeError('actual tail safe point required')
                safe_point.assert_ready(progress)
                if (safe_point.rank != self.rank or safe_point.world_size != 8
                        or safe_point.group_geometry_id != identity.group_geometry_id
                        or progress.execution_transform_plan_id != contract.execution_transform_plan_id):
                    raise ValueError('tail proof/contract topology mismatch')
                data = _data(data_state, self.rank, progress)
                if data['content_id'] != bindings['data']:
                    raise ValueError('data content binding mismatch')
                rng = _capture_rng(self.rank)
                if not isinstance(rng, RankRNGSnapshot) or rng.rank != self.rank:
                    raise ValueError('rank RNG capture identity mismatch')
                common = progress.to_payload()
                common.pop('rng_state_id')  # replaced below by ORDERED world table digest, not discarded
                return dict(rank=self.rank, rng_id=rng.state_identity,
                            common=_payload_digest(dict(root=str(self.manager.root), binding=binding,
                                progress=common, data_content=data['content_id'], cursor=data['cursor'],
                                next_batch=data['next_global_batch_id'], behavior=data['behavior_id'])))
            candidates = self._phase('preflight', preflight)
            self._same([row['common'] for row in candidates])
            progress = replace(progress, rng_state_id=_rng_table_id(candidates))
            staging = None
            def create():
                nonlocal staging
                if self.rank != 0:
                    return None
                staging = self.manager.create_staging(progress.global_step)
                return dict(name=staging.path.name, step=staging.step, binding=_payload_digest(binding),
                            progress=progress.progress_id)
            created = self._phase('create-staging', create)
            def resolve():
                root = created[0]
                if any(row is not None for row in created[1:]) or not isinstance(root, dict):
                    raise ValueError('only main writer may create staging')
                if (root['step'] != progress.global_step or root['binding'] != _payload_digest(binding)
                        or root['progress'] != progress.progress_id
                        or re.fullmatch(rf"step-{progress.global_step}\.tmp-[0-9a-f]{{32}}", root['name']) is None):
                    raise ValueError('staging broadcast identity mismatch')
                return str(_direct_checkpoint_path(self.manager.root, self.manager.root / root['name']))
            path = Path(self._same(self._phase('resolve-staging', resolve)))
            # ALL ranks call the real codec outside ANY manager writer callback.
            codec = save_h3_state(prepared, path, process_group=self.group, identity=identity,
                                  safe_point=safe_point, progress=progress)
            self._same(self._phase('codec-complete', lambda: codec))
            payload = dict(schema=_SCHEMA, rank=self.rank, world_size=8,
                           identity_id=_payload_digest(identity.payload()), progress_id=progress.progress_id,
                           codec_id=_payload_digest(codec), safe_point=safe_point.to_payload(),
                           rng=rng.to_checkpoint_payload(), data=data)
            descriptors = self._phase('rank-written', lambda: _write_rank(path, payload))
            manifest = dict(schema=_SCHEMA, binding=binding, progress_id=progress.progress_id,
                            codec=codec, ranks=descriptors)
            def verify_files():
                self._validate_ranks(path, manifest, identity, progress)
                return _payload_digest(manifest)
            self._same(self._phase('verify-ranks', verify_files))
            def publish():
                if self.rank != 0:
                    return None
                # File fsync does not persist the eight new directory entries.
                # Run once after all-rank validation, inside the failure phase.
                _fsync_directory(path / _RANKS)
                _atomic_json(path / _MANIFEST, manifest)
                committed = self.manager.commit_existing(staging, contract, progress=progress,
                                                          fault_injector=fault_injector)
                return dict(name=committed.path.name, step=committed.step,
                            contract=committed.checkpoint_contract_id, progress=committed.progress_id,
                            tree=committed.state_tree_id)
            published = self._phase('publish', publish)
            receipt = published[0]
            if (any(row is not None for row in published[1:]) or not isinstance(receipt, dict)
                    or receipt['name'] != f'step-{progress.global_step}' or receipt['step'] != progress.global_step
                    or receipt['contract'] != contract.checkpoint_contract_id or receipt['progress'] != progress.progress_id):
                raise ValueError('invalid main-writer commit receipt')
            return CommittedCheckpoint(self.manager.root / receipt['name'], receipt['step'], receipt['contract'],
                                       progress, receipt['tree'])

    def _validate_ranks(self, path, manifest, identity, progress):
        rows = manifest['ranks']
        if (type(rows) is not list or any(type(row) is not dict or set(row) !=
                {'rank', 'file', 'size', 'sha256', 'rng_id', 'data_id', 'safe_point_id'} for row in rows)):
            raise ValueError('rank descriptor schema mismatch')
        if _rng_table_id(rows) != progress.rng_state_id:
            raise ValueError('global ordered RNG table identity mismatch')
        root = path / _RANKS
        if root.is_symlink() or not root.is_dir() or sorted(p.name for p in root.iterdir()) != [f'rank-{r}.pt' for r in range(8)]:
            raise ValueError('rank attachment coverage mismatch')
        local = None
        common = []
        for row in rows:
            data, rng = _read_rank(path, row, identity, progress, _payload_digest(manifest['codec']))
            if data['content_id'] != manifest['binding']['live']['data']:
                raise ValueError('rank data content binding mismatch')
            common.append((data['content_id'], data['source_id'], data['cursor'], data['next_global_batch_id'], data['behavior_id']))
            if row['rank'] == self.rank:
                local = data, rng
        self._same(common)
        return local

    def _inspect(self, path, contract, identity, bindings):
        inspection = self.manager.inspect_complete(path, expected_contract=contract)  # hashes BEFORE torch/DCP reads
        file = inspection.committed.path / _MANIFEST
        if file.stat().st_size > 256 * 1024:
            raise ValueError('H3 manifest exceeds budget')
        manifest = _read_exact_json(file, {'schema', 'binding', 'progress_id', 'codec', 'ranks'})
        if (manifest['schema'] != _SCHEMA or manifest['binding'] != _binding(contract, identity, bindings)
                or manifest['progress_id'] != inspection.progress.progress_id):
            raise ValueError('complete H3 checkpoint binding mismatch')
        codec_file = inspection.committed.path / 'h3-state' / 'manifest.json'
        if codec_file.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('codec manifest exceeds budget')
        codec_manifest = json.loads(codec_file.read_bytes())
        if _codec_receipt(codec_manifest) != manifest['codec']:
            raise ValueError('complete codec manifest/receipt mismatch')
        data, rng = self._validate_ranks(inspection.committed.path, manifest, identity, inspection.progress)
        return inspection, data, rng, manifest

    def restore_prepared(self, prepared, path, *, expected_contract, identity, bindings):
        """Refresh complete/tree/contract, load fresh model/Adam, but NOT data/RNG."""
        with self._attempt():
            if self._prepared_attempted:
                raise RuntimeError('restore_prepared is once-only')
            self._prepared_attempted = True
            inspected = None
            def inspect():
                nonlocal inspected
                inspected = self._inspect(path, expected_contract, identity, bindings)
                receipt = inspected[0].committed
                return dict(path=str(receipt.path), tree=receipt.state_tree_id, progress=receipt.progress_id)
            self._same(self._phase('restore-prepared/inspect', inspect))
            inspection, data, rng, manifest = inspected
            codec = load_h3_state(prepared, inspection.committed.path, process_group=self.group,
                                  identity=identity, progress=inspection.progress)
            def validate():
                if codec != manifest['codec']:
                    raise ValueError('restored codec receipt mismatch')
                return codec
            self._same(self._phase('restore-prepared/loaded', validate))
            self._pending = inspection.committed
            return dict(checkpoint=inspection.committed, data=deepcopy(data), rng=rng)

    def restore_bound(self, *, validate_live, restore_data, read_data) -> CheckpointProgress:
        """Refresh disk AND live bindings, restore/check data, then RNG LAST.

        validate_live() must freshly inspect actual live model/resources/cache/
        score/dual-schedule/topology/data and return (contract, identity, bindings).
        restore_data(payload) applies source state; read_data() returns the full
        checked data envelope, including actual cursor and behavior afterwards.
        """
        with self._attempt():
            if self._pending is None or self._bound_attempted:
                raise RuntimeError('restore_bound requires one successful prepared gate and is once-only')
            self._bound_attempted = True
            inspected = None
            def refresh():
                nonlocal inspected
                if not all(callable(fn) for fn in (validate_live, restore_data, read_data)):
                    raise TypeError('live validation and data restore/read callbacks are mandatory')
                contract, identity, bindings = validate_live()  # never just reuse prepared-gate hashes
                inspected = self._inspect(self._pending.path, contract, identity, bindings)
                current = inspected[0].committed
                if (current.state_tree_id != self._pending.state_tree_id or current.progress_id != self._pending.progress_id
                        or current.checkpoint_contract_id != self._pending.checkpoint_contract_id):
                    raise ValueError('complete checkpoint changed between restore gates')
                return dict(path=str(current.path), tree=current.state_tree_id, progress=current.progress_id)
            self._same(self._phase('restore-bound/refresh', refresh))
            inspection, data, rng, _ = inspected
            def apply_data():
                restore_data(deepcopy(data['payload']))
                if _data(read_data(), self.rank, inspection.progress) != data:
                    raise ValueError('restored data/behavior/cursor differs from saved boundary')
                return inspection.progress.progress_id
            self._same(self._phase('restore-bound/data', apply_data))
            rng.restore_current()  # no IO, helper, model callback or collective after this
            return inspection.progress
