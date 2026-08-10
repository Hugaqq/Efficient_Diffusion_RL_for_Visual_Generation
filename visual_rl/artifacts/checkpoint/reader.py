"""Read one validated rank shard without constructing checkpoint writers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import visual_rl.artifacts.checkpoint.transaction as _coordinator_io
from visual_rl.artifacts.checkpoint.manager import (
    AtomicCheckpointManager,
    CheckpointInspection,
    CommittedCheckpoint,
)
from visual_rl.artifacts.checkpoint.state import RankCheckpointSnapshot

__all__ = ("RankCheckpointReader",)


@dataclass(frozen=True, slots=True)
class RankCheckpointReader:
    """Restore-side reader for one already inspected coordinated checkpoint."""

    manager: AtomicCheckpointManager

    def __post_init__(self) -> None:
        if not isinstance(self.manager, AtomicCheckpointManager):
            raise TypeError("manager must be an AtomicCheckpointManager")

    def read_rank_snapshot(
        self,
        inspection: CheckpointInspection | CommittedCheckpoint,
        *,
        expected_world_size: int,
        expected_rank: int,
    ) -> RankCheckpointSnapshot:
        """Revalidate the receipt, then load one manifest-bound rank snapshot."""

        _rank_and_world_size(expected_rank, expected_world_size)
        refreshed = _refresh_inspection(self.manager, inspection)
        if refreshed.contract.world_size != expected_world_size:
            raise ValueError(
                "checkpoint contract world_size disagrees with expected runtime"
            )

        checkpoint = refreshed.committed
        manifest = _coordinator_io._read_manifest(
            checkpoint.path / _coordinator_io._MANIFEST_FILE
        )
        _coordinator_io._validate_manifest_header(manifest, checkpoint)
        if manifest["world_size"] != expected_world_size:
            raise ValueError(
                "coordinator manifest world_size disagrees with expected runtime"
            )

        entries = manifest["shards"]
        entry = next(
            (item for item in entries if item["rank"] == expected_rank),
            None,
        )
        if entry is None:
            raise ValueError(f"checkpoint has no shard for rank {expected_rank}")
        expected_path = f"{_coordinator_io._SHARD_DIRECTORY}/rank-{expected_rank}.pt"
        if entry["path"] != expected_path:
            raise ValueError("rank shard manifest path is invalid")

        shard_root = checkpoint.path / _coordinator_io._SHARD_DIRECTORY
        shard_path = checkpoint.path / expected_path
        _validate_shard_path(checkpoint.path, shard_root, shard_path)
        shard_size, shard_digest = _coordinator_io._file_identity(shard_path)
        if shard_size != entry["shard_size"] or shard_digest != entry["shard_sha256"]:
            raise ValueError("rank shard digest disagrees with coordinator manifest")

        snapshot = RankCheckpointSnapshot.from_checkpoint_payload(
            _coordinator_io._load_torch_payload(shard_path)
        )
        _coordinator_io._validate_snapshot_manifest_entry(snapshot, entry)
        if (snapshot.rank, snapshot.world_size) != (
            expected_rank,
            expected_world_size,
        ):
            raise ValueError("rank shard topology disagrees with expected runtime")
        snapshot.safe_point.assert_ready(refreshed.progress)
        if snapshot.dynamics_selection_policy != (
            refreshed.progress.dynamics_selection_policy
        ):
            raise ValueError(
                "rank shard Dynamics selection policy disagrees with progress"
            )
        return snapshot

    def read(
        self,
        inspection: CheckpointInspection | CommittedCheckpoint,
        *,
        expected_world_size: int,
        expected_rank: int,
    ) -> RankCheckpointSnapshot:
        """Short alias for :meth:`read_rank_snapshot`."""

        return self.read_rank_snapshot(
            inspection,
            expected_world_size=expected_world_size,
            expected_rank=expected_rank,
        )

    def load_rank_snapshot(
        self,
        inspection: CheckpointInspection | CommittedCheckpoint,
        *,
        expected_world_size: int,
        expected_rank: int,
    ) -> RankCheckpointSnapshot:
        """Coordinator-compatible naming without requiring a coordinator."""

        return self.read_rank_snapshot(
            inspection,
            expected_world_size=expected_world_size,
            expected_rank=expected_rank,
        )


def _rank_and_world_size(rank: object, world_size: object) -> None:
    if type(world_size) is not int or world_size < 1:
        raise ValueError("expected_world_size must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("expected_rank must satisfy 0 <= rank < expected_world_size")


def _validate_same_inspection(
    expected: CheckpointInspection,
    found: CheckpointInspection,
) -> None:
    if expected.committed.path != found.committed.path:
        raise ValueError("checkpoint inspection path changed")
    if expected.committed.checkpoint_contract_id != (
        found.committed.checkpoint_contract_id
    ):
        raise ValueError("checkpoint inspection contract identity changed")
    if expected.committed.progress_id != found.committed.progress_id:
        raise ValueError("checkpoint inspection progress identity changed")
    if expected.committed.state_tree_id != found.committed.state_tree_id:
        raise ValueError("checkpoint inspection state tree identity changed")


def _refresh_inspection(
    manager: AtomicCheckpointManager,
    value: CheckpointInspection | CommittedCheckpoint,
) -> CheckpointInspection:
    if isinstance(value, CheckpointInspection):
        refreshed = manager.inspect_complete(
            value.committed.path,
            expected_contract=value.contract,
        )
        _validate_same_inspection(value, refreshed)
        return refreshed
    if isinstance(value, CommittedCheckpoint):
        refreshed = manager.inspect_complete(value.path)
        found = refreshed.committed
        if (
            value.path != found.path
            or value.checkpoint_contract_id != found.checkpoint_contract_id
            or value.progress_id != found.progress_id
            or value.state_tree_id != found.state_tree_id
        ):
            raise ValueError(
                "committed checkpoint receipt disagrees with inspected directory"
            )
        return refreshed
    raise TypeError("inspection must be CheckpointInspection or CommittedCheckpoint")


def _validate_shard_path(
    checkpoint_path: Path,
    shard_root: Path,
    shard_path: Path,
) -> None:
    if (
        shard_root.parent != checkpoint_path
        or shard_root.is_symlink()
        or not shard_root.is_dir()
    ):
        raise ValueError("rank shard directory is missing or unsafe")
    if (
        shard_path.parent != shard_root
        or shard_path.is_symlink()
        or not shard_path.is_file()
    ):
        raise ValueError("rank shard is missing or not a regular file")
    try:
        resolved = shard_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("rank shard path cannot be resolved safely") from exc
    if resolved != shard_path:
        raise ValueError("rank shard path must not traverse symlinks")
