"""Distributed two-phase safe-point checkpoint coordination.

Each rank writes its own non-resumable staging shard.  Rank zero validates the
complete descriptor set and delegates the only resumable filesystem mutation
to ``AtomicCheckpointManager``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any
import uuid

from visual_rl.artifacts.checkpoint.coordination import (
    CheckpointCollectiveBackend,
    CheckpointConsensusError,
    CheckpointSafePoint,
    CheckpointSafetyError,
    SingleProcessCheckpointBackend,
    StrategyCheckpointBackend,
    _digest,
    _rank_and_world_size,
    _validate_backend,
)
from visual_rl.artifacts.checkpoint.manager import AtomicCheckpointManager, CommittedCheckpoint
from visual_rl.artifacts.checkpoint.protocol import CheckpointContract, CheckpointProgress
from visual_rl.artifacts.checkpoint.state import (
    CheckpointStateCollector,
    RankCheckpointSnapshot,
    RankRNGSnapshot,
)

_MANIFEST_FILE = "coordinator_manifest.json"
_SHARD_DIRECTORY = "rank_shards"
_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class _RankShardDescriptor:
    rank: int
    world_size: int
    staging_name: str
    staging_file: str
    shard_sha256: str
    shard_size: int
    safe_point_id: str
    rng_state_id: str
    dynamics_selection_policy_id: str
    group_geometry_id: str
    component_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _rank_and_world_size(self.rank, self.world_size)
        if not isinstance(self.staging_name, str) or not re.fullmatch(
            r"step-[0-9]+-[0-9a-f]{32}", self.staging_name
        ):
            raise ValueError("rank shard staging_name is invalid")
        if self.staging_file != f"rank-{self.rank}.pt":
            raise ValueError("rank shard filename does not match rank")
        for name in (
            "shard_sha256",
            "safe_point_id",
            "rng_state_id",
            "dynamics_selection_policy_id",
            "group_geometry_id",
        ):
            _digest(name, getattr(self, name))
        if type(self.shard_size) is not int or self.shard_size < 1:
            raise ValueError("rank shard size must be a positive integer")
        if type(self.component_names) is not tuple or not self.component_names:
            raise ValueError("rank shard component_names must be non-empty")
        if tuple(sorted(set(self.component_names))) != self.component_names:
            raise ValueError("rank shard component_names must be sorted and unique")


@dataclass(frozen=True, slots=True)
class _CheckpointCandidate:
    checkpoint_contract_id: str
    progress_id: str
    safe_point: CheckpointSafePoint
    descriptor: _RankShardDescriptor

    def __post_init__(self) -> None:
        _digest("checkpoint_contract_id", self.checkpoint_contract_id)
        _digest("progress_id", self.progress_id)
        if not isinstance(self.safe_point, CheckpointSafePoint):
            raise TypeError("candidate safe_point must be CheckpointSafePoint")
        if not isinstance(self.descriptor, _RankShardDescriptor):
            raise TypeError("candidate descriptor must be _RankShardDescriptor")
        if (self.descriptor.rank, self.descriptor.world_size) != (
            self.safe_point.rank,
            self.safe_point.world_size,
        ):
            raise ValueError("candidate descriptor topology disagrees")
        if self.descriptor.safe_point_id != self.safe_point.safe_point_id:
            raise ValueError("candidate descriptor safe point disagrees")
        if self.descriptor.group_geometry_id != self.safe_point.group_geometry_id:
            raise ValueError("candidate descriptor group geometry disagrees")


@dataclass(frozen=True, slots=True)
class _StagingReceipt:
    staging_name: str
    step: int

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("checkpoint staging step must be non-negative")
        if not isinstance(self.staging_name, str) or not re.fullmatch(
            rf"step-{self.step}-[0-9a-f]{{32}}", self.staging_name
        ):
            raise ValueError("checkpoint staging receipt is invalid")


@dataclass(frozen=True, slots=True)
class _CommitReceipt:
    path_name: str
    step: int
    checkpoint_contract_id: str
    progress_id: str
    state_tree_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.path_name, str) or self.path_name != f"step-{self.step}":
            raise ValueError("commit receipt path/step identity is invalid")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("commit receipt step must be non-negative")
        for name in (
            "checkpoint_contract_id",
            "progress_id",
            "state_tree_id",
        ):
            _digest(name, getattr(self, name))


class CheckpointCoordinator:
    """Coordinate one all-rank safe point into one atomic checkpoint tree."""

    def __init__(
        self,
        *,
        manager: AtomicCheckpointManager,
        backend: CheckpointCollectiveBackend,
        collector: CheckpointStateCollector,
    ) -> None:
        if not isinstance(manager, AtomicCheckpointManager):
            raise TypeError("manager must be AtomicCheckpointManager")
        if not isinstance(collector, CheckpointStateCollector):
            raise TypeError("collector must be CheckpointStateCollector")
        _validate_backend(backend)
        self.manager = manager
        self.backend = backend
        self.collector = collector

    def checkpoint(
        self,
        *,
        contract: CheckpointContract,
        progress: CheckpointProgress,
        safe_point: CheckpointSafePoint,
        fault_injector: Callable[[str], None] | None = None,
    ) -> CommittedCheckpoint:
        """Commit only after every rank presents a complete safe-point shard."""

        self._synchronized_local_phase(
            "checkpoint.prepare",
            lambda: self._validate_local_request(contract, progress, safe_point),
        )
        self.backend.barrier("checkpoint.safe_point")
        snapshot = self._synchronized_local_phase(
            "checkpoint.capture_state",
            lambda: self._capture_snapshot(safe_point, progress),
        )
        root_staging = self._synchronized_root_phase(
            "checkpoint.prepare_staging",
            lambda: _prepare_staging(
                self.manager.root,
                progress.global_step,
                fault_injector=fault_injector,
            ),
        )
        staging_receipt: _StagingReceipt | None = None
        received: object = None
        try:
            staging_receipt = self._synchronized_local_phase(
                "checkpoint.broadcast_staging",
                lambda: self.backend.broadcast_object(root_staging, src=0),
            )
            if not isinstance(staging_receipt, _StagingReceipt):
                raise CheckpointConsensusError("checkpoint staging receipt is invalid")
            if staging_receipt.step != progress.global_step:
                raise CheckpointConsensusError(
                    "checkpoint staging receipt step disagrees with progress"
                )
            staging_path = _resolve_staging(
                self.manager.root,
                staging_receipt,
            )
            descriptor = self._synchronized_local_phase(
                "checkpoint.write_rank_shard",
                lambda: _write_local_rank_shard(
                    staging_path,
                    snapshot,
                    staging_name=staging_receipt.staging_name,
                    fault_injector=fault_injector,
                ),
            )
            self.backend.barrier("checkpoint.rank_shards_written")
            candidate = _CheckpointCandidate(
                checkpoint_contract_id=contract.checkpoint_contract_id,
                progress_id=progress.progress_id,
                safe_point=safe_point,
                descriptor=descriptor,
            )
            gathered = self._synchronized_local_phase(
                "checkpoint.gather_shards",
                lambda: self.backend.gather_object(candidate, dst=0),
            )
            candidates = self._synchronized_root_phase(
                "checkpoint.validate_consensus",
                lambda: self._validate_consensus(
                    gathered,
                    contract=contract,
                    progress=progress,
                    staging_path=staging_path,
                ),
            )
            committed = self._synchronized_root_phase(
                "checkpoint.atomic_commit",
                lambda: self.manager.commit(
                    progress.global_step,
                    contract,
                    lambda path: _write_rank_state_tree(
                        path,
                        candidates,
                        staging_path=staging_path,
                        contract=contract,
                        progress=progress,
                        fault_injector=fault_injector,
                    ),
                    progress=progress,
                    fault_injector=fault_injector,
                ),
            )
            receipt = (
                None
                if committed is None
                else _CommitReceipt(
                    path_name=committed.path.name,
                    step=committed.step,
                    checkpoint_contract_id=committed.checkpoint_contract_id,
                    progress_id=committed.progress_id,
                    state_tree_id=committed.state_tree_id,
                )
            )
            received = self._synchronized_local_phase(
                "checkpoint.broadcast_commit",
                lambda: self.backend.broadcast_object(receipt, src=0),
            )
        finally:
            if self.backend.is_main_process and isinstance(
                root_staging, _StagingReceipt
            ):
                _cleanup_staging(self.manager.root, root_staging)
        if not isinstance(received, _CommitReceipt):
            raise CheckpointConsensusError("checkpoint commit receipt is invalid")
        if (
            received.step != progress.global_step
            or received.checkpoint_contract_id != contract.checkpoint_contract_id
            or received.progress_id != progress.progress_id
        ):
            raise CheckpointConsensusError(
                "checkpoint commit receipt disagrees with frozen request"
            )
        self.backend.barrier("checkpoint.committed")
        return CommittedCheckpoint(
            self.manager.root / received.path_name,
            received.step,
            received.checkpoint_contract_id,
            progress,
            received.state_tree_id,
        )

    def load_rank_snapshot(
        self,
        checkpoint: CommittedCheckpoint,
        *,
        rank: int | None = None,
    ) -> RankCheckpointSnapshot:
        """Load one hash-validated shard from a complete coordinated checkpoint."""

        if not isinstance(checkpoint, CommittedCheckpoint):
            raise TypeError("checkpoint must be CommittedCheckpoint")
        resolved = self.manager.latest_complete(explicit_path=checkpoint.path)
        if resolved is None:
            raise ValueError("checkpoint is not complete")
        if (
            resolved.checkpoint_contract_id != checkpoint.checkpoint_contract_id
            or resolved.progress_id != checkpoint.progress_id
            or resolved.state_tree_id != checkpoint.state_tree_id
        ):
            raise ValueError("checkpoint receipt disagrees with validated directory")
        selected_rank = self.backend.rank if rank is None else rank
        _rank_and_world_size(selected_rank, self.backend.world_size)
        manifest = _read_manifest(resolved.path / _MANIFEST_FILE)
        _validate_manifest_header(manifest, resolved)
        if manifest["world_size"] != self.backend.world_size:
            raise ValueError(
                "coordinator manifest world_size disagrees with runtime backend"
            )
        entries = manifest["shards"]
        entry = next(
            (item for item in entries if item["rank"] == selected_rank),
            None,
        )
        if entry is None:
            raise ValueError(f"checkpoint has no shard for rank {selected_rank}")
        expected_path = f"{_SHARD_DIRECTORY}/rank-{selected_rank}.pt"
        if entry["path"] != expected_path:
            raise ValueError("rank shard manifest path is invalid")
        shard_path = resolved.path / entry["path"]
        shard_size, shard_digest = _file_identity(shard_path)
        if shard_size != entry["shard_size"] or shard_digest != entry["shard_sha256"]:
            raise ValueError("rank shard digest disagrees with coordinator manifest")
        payload = _load_torch_payload(shard_path)
        snapshot = RankCheckpointSnapshot.from_checkpoint_payload(payload)
        _validate_snapshot_manifest_entry(snapshot, entry)
        return snapshot

    def _validate_local_request(
        self,
        contract: CheckpointContract,
        progress: CheckpointProgress,
        safe_point: CheckpointSafePoint,
    ) -> None:
        if not isinstance(contract, CheckpointContract):
            raise TypeError("contract must be CheckpointContract")
        if not isinstance(progress, CheckpointProgress):
            raise TypeError("progress must be CheckpointProgress")
        if not isinstance(safe_point, CheckpointSafePoint):
            raise TypeError("safe_point must be CheckpointSafePoint")
        if (safe_point.rank, safe_point.world_size) != (
            self.backend.rank,
            self.backend.world_size,
        ):
            raise CheckpointConsensusError(
                "safe-point topology disagrees with collective backend"
            )
        if contract.world_size != self.backend.world_size:
            raise CheckpointConsensusError(
                "checkpoint contract world_size disagrees with backend"
            )
        if progress.execution_transform_plan_id != contract.execution_transform_plan_id:
            raise CheckpointConsensusError(
                "checkpoint progress transform plan disagrees with contract"
            )
        safe_point.assert_ready(progress)

    def _capture_snapshot(
        self,
        safe_point: CheckpointSafePoint,
        progress: CheckpointProgress,
    ) -> RankCheckpointSnapshot:
        snapshot = self.collector.capture(safe_point)
        if snapshot.dynamics_selection_policy != progress.dynamics_selection_policy:
            raise CheckpointConsensusError(
                "captured Dynamics selection policy disagrees with progress"
            )
        return snapshot

    def _validate_consensus(
        self,
        gathered: list[object] | None,
        *,
        contract: CheckpointContract,
        progress: CheckpointProgress,
        staging_path: Path,
    ) -> tuple[_CheckpointCandidate, ...]:
        if not self.backend.is_main_process:
            if gathered is not None:
                raise CheckpointConsensusError(
                    "non-main rank unexpectedly received gathered shards"
                )
            return ()
        if not isinstance(gathered, list):
            raise CheckpointConsensusError("main rank did not receive rank shards")
        if len(gathered) != self.backend.world_size:
            raise CheckpointConsensusError(
                "checkpoint shard count does not equal world_size"
            )
        if any(not isinstance(item, _CheckpointCandidate) for item in gathered):
            raise CheckpointConsensusError(
                "checkpoint gather contains an invalid shard"
            )
        candidates = tuple(gathered)
        component_names: tuple[str, ...] | None = None
        geometry_id: str | None = None
        for expected_rank, candidate in enumerate(candidates):
            safe_point = candidate.safe_point
            descriptor = candidate.descriptor
            if (safe_point.rank, safe_point.world_size) != (
                expected_rank,
                self.backend.world_size,
            ):
                raise CheckpointConsensusError(
                    "checkpoint shards are missing, duplicated, or out of rank order"
                )
            safe_point.assert_ready(progress)
            if candidate.checkpoint_contract_id != contract.checkpoint_contract_id:
                raise CheckpointConsensusError(
                    "checkpoint contract differs across ranks"
                )
            if candidate.progress_id != progress.progress_id:
                raise CheckpointConsensusError(
                    "checkpoint progress differs across ranks"
                )
            if (descriptor.rank, descriptor.world_size) != (
                expected_rank,
                self.backend.world_size,
            ):
                raise CheckpointConsensusError("rank shard topology is invalid")
            if descriptor.staging_name != staging_path.name:
                raise CheckpointConsensusError(
                    "rank shard was written into a different staging transaction"
                )
            if component_names is None:
                component_names = descriptor.component_names
            elif descriptor.component_names != component_names:
                raise CheckpointConsensusError(
                    "checkpoint component state set differs across ranks"
                )
            if geometry_id is None:
                geometry_id = safe_point.group_geometry_id
            elif safe_point.group_geometry_id != geometry_id:
                raise CheckpointConsensusError(
                    "group placement geometry differs across ranks"
                )
        for candidate in candidates:
            _verify_staged_rank_shard(
                staging_path,
                candidate.descriptor,
                validate_payload=True,
            )
        return candidates

    def _synchronized_local_phase(
        self,
        phase: str,
        operation: Callable[[], Any],
    ) -> Any:
        value: Any = None
        failure: BaseException | None = None
        try:
            value = operation()
        except BaseException as exc:
            failure = exc
        self.backend.failure_gate(phase, failure)
        if failure is not None:
            raise AssertionError(
                "failure_gate returned after local failure"
            ) from failure
        return value

    def _synchronized_root_phase(
        self,
        phase: str,
        operation: Callable[[], Any],
    ) -> Any:
        value: Any = None
        failure: BaseException | None = None
        if self.backend.is_main_process:
            try:
                value = operation()
            except BaseException as exc:
                failure = exc
        self.backend.failure_gate(phase, failure)
        if failure is not None:
            raise AssertionError(
                "failure_gate returned after root failure"
            ) from failure
        return value


def _write_rank_state_tree(
    root: Path,
    candidates: tuple[_CheckpointCandidate, ...],
    *,
    staging_path: Path,
    contract: CheckpointContract,
    progress: CheckpointProgress,
    fault_injector: Callable[[str], None] | None,
) -> None:
    if not candidates:
        raise CheckpointConsensusError("cannot write a checkpoint without rank shards")
    shard_root = root / _SHARD_DIRECTORY
    shard_root.mkdir(mode=0o700)
    manifest_entries: list[dict[str, object]] = []
    for candidate in candidates:
        descriptor = candidate.descriptor
        relative = f"{_SHARD_DIRECTORY}/rank-{descriptor.rank}.pt"
        source = staging_path / descriptor.staging_file
        _verify_staged_rank_shard(staging_path, descriptor)
        _fault(fault_injector, f"before_rank_state_copy.rank-{descriptor.rank}")
        shutil.copyfile(source, root / relative)
        _fault(fault_injector, f"after_rank_state_copy.rank-{descriptor.rank}")
        manifest_entries.append(
            {
                "rank": descriptor.rank,
                "path": relative,
                "shard_sha256": descriptor.shard_sha256,
                "shard_size": descriptor.shard_size,
                "safe_point_id": descriptor.safe_point_id,
                "rng_state_id": descriptor.rng_state_id,
                "dynamics_selection_policy_id": (
                    descriptor.dynamics_selection_policy_id
                ),
                "group_geometry_id": descriptor.group_geometry_id,
                "component_names": list(descriptor.component_names),
            }
        )
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "visual_rl_coordinated_checkpoint",
        "step": progress.global_step,
        "world_size": len(candidates),
        "checkpoint_contract_id": contract.checkpoint_contract_id,
        "progress_id": progress.progress_id,
        "shards": manifest_entries,
    }
    (root / _MANIFEST_FILE).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    _fault(fault_injector, "after_coordinator_manifest")


def _prepare_staging(
    checkpoint_root: Path,
    step: int,
    *,
    fault_injector: Callable[[str], None] | None,
) -> _StagingReceipt:
    if type(step) is not int or step < 0:
        raise ValueError("checkpoint staging step must be non-negative")
    staging_root = checkpoint_root / ".coordinator-staging"
    if staging_root.is_symlink():
        raise ValueError("checkpoint staging root must not be a symlink")
    staging_root.mkdir(mode=0o700, exist_ok=True)
    staging_name = f"step-{step}-{uuid.uuid4().hex}"
    staging_path = staging_root / staging_name
    staging_path.mkdir(mode=0o700)
    receipt = _StagingReceipt(staging_name=staging_name, step=step)
    try:
        _fault(fault_injector, "after_staging_prepare")
    except BaseException:
        shutil.rmtree(staging_path)
        try:
            staging_root.rmdir()
        except OSError:
            pass
        raise
    return receipt


def _resolve_staging(
    checkpoint_root: Path,
    receipt: _StagingReceipt,
) -> Path:
    if not isinstance(receipt, _StagingReceipt):
        raise TypeError("receipt must be _StagingReceipt")
    staging_root = checkpoint_root / ".coordinator-staging"
    staging_path = staging_root / receipt.staging_name
    if (
        staging_root.is_symlink()
        or staging_path.is_symlink()
        or not staging_path.is_dir()
        or staging_path.parent != staging_root
    ):
        raise ValueError("checkpoint staging directory is missing or unsafe")
    return staging_path


def _cleanup_staging(checkpoint_root: Path, receipt: _StagingReceipt) -> None:
    try:
        staging_path = _resolve_staging(checkpoint_root, receipt)
    except (OSError, ValueError):
        return
    try:
        shutil.rmtree(staging_path)
    except OSError:
        return
    staging_root = staging_path.parent
    try:
        staging_root.rmdir()
    except OSError:
        pass


def _write_local_rank_shard(
    staging_path: Path,
    snapshot: RankCheckpointSnapshot,
    *,
    staging_name: str,
    fault_injector: Callable[[str], None] | None,
) -> _RankShardDescriptor:
    if not isinstance(snapshot, RankCheckpointSnapshot):
        raise TypeError("snapshot must be RankCheckpointSnapshot")
    if staging_path.name != staging_name:
        raise ValueError("rank shard staging identity disagrees")
    filename = f"rank-{snapshot.rank}.pt"
    destination = staging_path / filename
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"rank shard already exists: {destination}")
    _fault(fault_injector, f"before_rank_state_write.rank-{snapshot.rank}")
    _save_torch_payload(destination, snapshot.to_checkpoint_payload())
    _fault(fault_injector, f"after_rank_state_write.rank-{snapshot.rank}")
    size, digest = _file_identity(destination)
    return _RankShardDescriptor(
        rank=snapshot.rank,
        world_size=snapshot.world_size,
        staging_name=staging_name,
        staging_file=filename,
        shard_sha256=digest,
        shard_size=size,
        safe_point_id=snapshot.safe_point.safe_point_id,
        rng_state_id=snapshot.rng_state.state_identity,
        dynamics_selection_policy_id=(
            snapshot.dynamics_selection_policy.policy_identity
        ),
        group_geometry_id=snapshot.safe_point.group_geometry_id,
        component_names=snapshot.component_names,
    )


def _verify_staged_rank_shard(
    staging_path: Path,
    descriptor: _RankShardDescriptor,
    *,
    validate_payload: bool = False,
) -> None:
    if not isinstance(descriptor, _RankShardDescriptor):
        raise TypeError("descriptor must be _RankShardDescriptor")
    if descriptor.staging_name != staging_path.name:
        raise ValueError("rank shard staging transaction is invalid")
    path = staging_path / descriptor.staging_file
    if path.is_symlink() or not path.is_file() or path.parent != staging_path:
        raise ValueError("rank shard is missing or not a regular staged file")
    size, digest = _file_identity(path)
    if size != descriptor.shard_size or digest != descriptor.shard_sha256:
        raise ValueError("rank shard digest or size changed before commit")
    if not validate_payload:
        return
    snapshot = RankCheckpointSnapshot.from_checkpoint_payload(_load_torch_payload(path))
    if (snapshot.rank, snapshot.world_size) != (
        descriptor.rank,
        descriptor.world_size,
    ):
        raise ValueError("rank shard payload topology disagrees with descriptor")
    if snapshot.safe_point.safe_point_id != descriptor.safe_point_id:
        raise ValueError("rank shard payload safe point disagrees with descriptor")
    if snapshot.rng_state.state_identity != descriptor.rng_state_id:
        raise ValueError("rank shard payload RNG state disagrees with descriptor")
    if (
        snapshot.dynamics_selection_policy.policy_identity
        != descriptor.dynamics_selection_policy_id
    ):
        raise ValueError(
            "rank shard payload Dynamics selection policy disagrees with descriptor"
        )
    if snapshot.safe_point.group_geometry_id != descriptor.group_geometry_id:
        raise ValueError("rank shard payload group geometry disagrees with descriptor")
    if snapshot.component_names != descriptor.component_names:
        raise ValueError("rank shard payload component set disagrees with descriptor")


def _read_manifest(path: Path) -> dict[str, object]:
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate coordinator manifest key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite coordinator manifest value {item}")
        ),
    )
    expected = {
        "schema_version",
        "kind",
        "step",
        "world_size",
        "checkpoint_contract_id",
        "progress_id",
        "shards",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("coordinator manifest has an invalid exact key set")
    if value["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("coordinator manifest schema version is unsupported")
    if value["kind"] != "visual_rl_coordinated_checkpoint":
        raise ValueError("coordinator manifest kind is invalid")
    raw_shards = value["shards"]
    shard_keys = {
        "rank",
        "path",
        "shard_sha256",
        "shard_size",
        "safe_point_id",
        "rng_state_id",
        "dynamics_selection_policy_id",
        "group_geometry_id",
        "component_names",
    }
    if not isinstance(raw_shards, list) or any(
        not isinstance(item, dict) or set(item) != shard_keys for item in raw_shards
    ):
        raise ValueError("coordinator manifest shard entries are invalid")
    return value


def _validate_manifest_header(
    manifest: Mapping[str, object],
    checkpoint: CommittedCheckpoint,
) -> None:
    if (
        manifest["step"] != checkpoint.step
        or manifest["checkpoint_contract_id"] != checkpoint.checkpoint_contract_id
        or manifest["progress_id"] != checkpoint.progress_id
    ):
        raise ValueError("coordinator manifest disagrees with checkpoint marker")
    world_size = manifest["world_size"]
    shards = manifest["shards"]
    if type(world_size) is not int or world_size < 1 or len(shards) != world_size:
        raise ValueError("coordinator manifest world_size/shard count is invalid")
    ranks = tuple(item["rank"] for item in shards)
    if ranks != tuple(range(world_size)):
        raise ValueError("coordinator manifest rank shard set is incomplete")


def _validate_snapshot_manifest_entry(
    snapshot: RankCheckpointSnapshot,
    entry: Mapping[str, object],
) -> None:
    expected_path = f"{_SHARD_DIRECTORY}/rank-{snapshot.rank}.pt"
    if entry["path"] != expected_path:
        raise ValueError("rank shard path is invalid")
    _digest("rank shard manifest digest", entry["shard_sha256"])
    if type(entry["shard_size"]) is not int or entry["shard_size"] < 1:
        raise ValueError("rank shard manifest size is invalid")
    if entry["safe_point_id"] != snapshot.safe_point.safe_point_id:
        raise ValueError("rank shard safe-point identity disagrees with manifest")
    if entry["rng_state_id"] != snapshot.rng_state.state_identity:
        raise ValueError("rank shard RNG identity disagrees with manifest")
    if (
        entry["dynamics_selection_policy_id"]
        != snapshot.dynamics_selection_policy.policy_identity
    ):
        raise ValueError("rank shard Dynamics selection policy disagrees with manifest")
    if entry["group_geometry_id"] != snapshot.safe_point.group_geometry_id:
        raise ValueError("rank shard group geometry disagrees with manifest")
    if entry["component_names"] != list(snapshot.component_names):
        raise ValueError("rank shard component set disagrees with manifest")


def _save_torch_payload(path: Path, payload: Mapping[str, object]) -> None:
    import torch

    torch.save(dict(payload), path)


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _load_torch_payload(path: Path) -> object:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _fault(callback: Callable[[str], None] | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


__all__ = (
    "CheckpointCollectiveBackend",
    "CheckpointConsensusError",
    "CheckpointCoordinator",
    "CheckpointSafePoint",
    "CheckpointSafetyError",
    "CheckpointStateCollector",
    "RankCheckpointSnapshot",
    "RankRNGSnapshot",
    "SingleProcessCheckpointBackend",
    "StrategyCheckpointBackend",
)
