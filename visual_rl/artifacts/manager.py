"""Authoritative artifact transactions and deterministic projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import errno
import fcntl
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Protocol
import uuid

from visual_rl.artifacts.manifest import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleRecord,
)
from visual_rl.artifacts.preview import PreviewWriteResult, PreviewWriter
from visual_rl.artifacts.serialization import (
    canonical_json_text,
    redact_artifact_config,
    strict_json_load,
)
from visual_rl.core.types import RolloutBatch, to_plain_dict
from visual_rl.errors import ArtifactError, ResumeError


CORE_METRIC_SCHEMA_VERSION = "3"
COMMIT_SCHEMA_VERSION = "2"

_PROJECTION_NAMES = ("sample_manifest.json", "metrics.jsonl")
_MARKER_FIELDS = {
    "schema_version",
    "kind",
    "run_id",
    "transaction_id",
    "completed_steps",
    "staged_steps",
    "checkpoint",
    "steps",
}
_JOURNAL_FIELDS = {
    "schema_version",
    "kind",
    "state",
    "run_id",
    "transaction_id",
    "completed_steps",
    "staged_steps",
    "checkpoint",
}
_STEP_FIELDS = {"artifact_step", "manifest_records", "core_metric_row"}
_CHECKPOINT_FIELDS = {"completed_steps", "path", "tree_sha256"}
_STAGED_CHECKPOINT_FIELDS = {
    "completed_steps",
    "staged_path",
    "final_path",
    "tree_sha256",
}
_METRIC_BASE_FIELDS = (
    "schema_version",
    "step",
    "sample_count",
    "active_transition_count",
)


class StepMetricsLike(Protocol):
    values: Mapping[str, Any]
    sample_count: int
    active_transition_count: int


@dataclass(slots=True)
class StepArtifactTransaction:
    """One open checkpoint cycle owned by exactly one manager."""

    transaction_id: str
    run_id: str
    staging_dir: Path
    completed_steps: int | None = None
    staged_steps: list[int] = field(default_factory=list)
    state: str = "open"


class ArtifactManager:
    """The sole writer for one run directory."""

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        *,
        config: Any,
    ) -> None:
        self._initialize(output_dir, run_id, create=True)
        try:
            if read_authoritative_commit_chain(self.output_dir):
                raise FileExistsError(
                    f"Artifact directory already contains commits: {self.output_dir}"
                )
            if any(self.staging_root.iterdir()):
                raise FileExistsError(
                    f"Artifact directory already contains staging: {self.output_dir}"
                )
            if any((self.output_dir / name).exists() for name in _PROJECTION_NAMES):
                raise FileExistsError(
                    f"Artifact directory already contains projections: "
                    f"{self.output_dir}"
                )
            self.write_resolved_config(config)
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        output_dir: str | Path,
        run_id: str,
        *,
        create: bool,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        self.output_dir = _validated_run_root(output_dir, create=create)
        self.run_id = run_id
        self.config_path = self.output_dir / "config.resolved.json"
        self.manifest_path = self.output_dir / "sample_manifest.json"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.commits_dir = self.output_dir / "commits"
        self.staging_root = self.output_dir / ".staging"
        self._lock_path = self.output_dir / ".artifact.lock"
        self._lock_fd: int | None = None
        self._closed = False
        self._open_transaction: StepArtifactTransaction | None = None
        _safe_directory(self.commits_dir, create=True)
        _safe_directory(self.staging_root, create=True)
        self._acquire_lock()

    @classmethod
    def open_resume(cls, output_dir: str | Path) -> ArtifactManager:
        """Open one run under a single writer lock without guessing a locator."""

        root = _validated_run_root(output_dir, create=False)
        provisional = cls.__new__(cls)
        provisional.output_dir = root
        provisional.config_path = root / "config.resolved.json"
        provisional.manifest_path = root / "sample_manifest.json"
        provisional.metrics_path = root / "metrics.jsonl"
        provisional.commits_dir = root / "commits"
        provisional.staging_root = root / ".staging"
        provisional._lock_path = root / ".artifact.lock"
        provisional._lock_fd = None
        provisional._closed = False
        provisional._open_transaction = None
        _safe_directory(provisional.commits_dir, create=False)
        _safe_directory(provisional.staging_root, create=False)
        provisional._acquire_lock()
        try:
            run_id = provisional._discover_resume_run_id()
            provisional.run_id = run_id
            return provisional
        except BaseException:
            provisional.close()
            raise

    def __enter__(self) -> ArtifactManager:
        self._require_open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @property
    def head(self) -> Mapping[str, Any] | None:
        chain = read_authoritative_commit_chain(self.output_dir)
        return None if not chain else chain[-1]

    @property
    def start_step(self) -> int:
        head = self.head
        return 0 if head is None else int(head["completed_steps"])

    @property
    def checkpoint_path(self) -> Path | None:
        head = self.head
        if head is None:
            return None
        return self.output_dir / str(head["checkpoint"]["path"])

    def write_resolved_config(self, config: Any) -> None:
        """Atomically replace the sole redacted resolved-config projection."""

        self._require_open()
        _atomic_write_json(
            self.config_path,
            redact_artifact_config(config),
            root=self.output_dir,
        )

    def begin_transaction(self) -> StepArtifactTransaction:
        self._require_open()
        if self._open_transaction is not None:
            raise RuntimeError("ArtifactManager already owns an open transaction")
        transaction_id = uuid.uuid4().hex
        staging_dir = self.staging_root / f"txn_{transaction_id}"
        _safe_directory(staging_dir, create=True)
        transaction = StepArtifactTransaction(
            transaction_id=transaction_id,
            run_id=self.run_id,
            staging_dir=staging_dir,
        )
        self._open_transaction = transaction
        self._write_journal(transaction, checkpoint=None)
        return transaction

    def stage_records(
        self,
        transaction: StepArtifactTransaction,
        *,
        step: int,
        records: Sequence[SampleRecord],
        metrics: StepMetricsLike,
    ) -> None:
        """Durably stage the only manifest rows and core metric row for a step."""

        self._validate_transaction(transaction, expected_state="open")
        if type(step) is not int or step < 0:
            raise ValueError("artifact step must be a non-negative integer")
        expected_step = self.start_step + len(transaction.staged_steps)
        if step != expected_step:
            raise ValueError(
                f"artifact steps must be contiguous: expected {expected_step}, "
                f"got {step}"
            )
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of SampleRecord values")
        rows = tuple(records)
        if not rows:
            raise ValueError("artifact records must not be empty")
        if any(not isinstance(record, SampleRecord) for record in rows):
            raise TypeError("records must contain only SampleRecord values")
        sample_ids: set[str] = set()
        media_paths: set[str] = set()
        for record in rows:
            if record.run_id != self.run_id:
                raise ValueError("record run_id does not match manager run_id")
            if record.step != step:
                raise ValueError("record step does not match staged step")
            if record.sample_id in sample_ids:
                raise ValueError("staged records contain duplicate sample_id")
            sample_ids.add(record.sample_id)
            self._validate_record_paths(record)
            if record.media_path is not None:
                if record.media_path in media_paths:
                    raise ValueError("staged records contain duplicate media_path")
                media_paths.add(record.media_path)
                self._validate_staged_preview(transaction, record)
        committed_ids = {
            row["sample_id"]
            for marker in read_authoritative_commit_chain(self.output_dir)
            for payload in marker["steps"]
            for row in payload["manifest_records"]
        }
        if committed_ids.intersection(sample_ids):
            raise ValueError("staged sample_id is already authoritative")
        for prior_step in transaction.staged_steps:
            wrapper = self._load_staged_manifest(transaction, prior_step)
            prior_ids = {row["sample_id"] for row in wrapper["records"]}
            if prior_ids.intersection(sample_ids):
                raise ValueError("staged sample_id is duplicated within transaction")

        metric_row = _core_metric_row(step, metrics)
        wrapper = {
            "schema_version": SAMPLE_MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "artifact_step": step,
            "records": [record.to_plain_dict() for record in rows],
        }
        step_dir = transaction.staging_dir / f"step_{step:06d}"
        _safe_directory(step_dir, create=True)
        _atomic_write_json(
            step_dir / "manifest_records.json",
            wrapper,
            root=transaction.staging_dir,
        )
        _atomic_write_json(
            step_dir / "metric.json",
            metric_row,
            root=transaction.staging_dir,
        )
        _fsync_directory(step_dir)
        transaction.staged_steps.append(step)
        self._write_journal(transaction, checkpoint=None)

    def stage_previews(
        self,
        transaction: StepArtifactTransaction,
        batch: RolloutBatch,
        *,
        max_samples: int,
    ) -> PreviewWriteResult:
        """Best-effort encode selected media inside the open transaction."""

        self._validate_transaction(transaction, expected_state="open")
        writer = PreviewWriter(transaction.staging_dir)
        return writer.write_batch(batch, max_samples=max_samples)

    def commit(
        self,
        transaction: StepArtifactTransaction,
        *,
        checkpoint_path: Path,
    ) -> Mapping[str, Any]:
        """Publish checkpoint then exact v2 marker; projections remain separate."""

        self._validate_transaction(transaction, expected_state="open")
        if not transaction.staged_steps:
            raise ValueError("cannot commit an empty artifact transaction")
        completed_steps = transaction.staged_steps[-1] + 1
        expected_checkpoint = (
            transaction.staging_dir / f"checkpoint_{completed_steps:06d}"
        )
        if checkpoint_path != expected_checkpoint:
            raise ValueError(
                "checkpoint_path must be the current transaction checkpoint staging "
                f"path: {expected_checkpoint}"
            )
        _validated_directory_within(
            checkpoint_path,
            transaction.staging_dir,
            label="checkpoint staging path",
        )
        from visual_rl.artifacts.checkpoint import checkpoint_tree_sha256

        tree_sha256 = checkpoint_tree_sha256(
            checkpoint_path,
            trusted_root=transaction.staging_dir,
        )
        transaction.completed_steps = completed_steps
        transaction.state = "ready"
        checkpoint = {
            "completed_steps": completed_steps,
            "staged_path": checkpoint_path.relative_to(
                self.output_dir
            ).as_posix(),
            "final_path": f"checkpoint_{completed_steps:06d}",
            "tree_sha256": tree_sha256,
        }
        _fsync_tree(transaction.staging_dir)
        self._write_journal(transaction, checkpoint=checkpoint)
        _fsync_tree(transaction.staging_dir)
        return self._publish_ready(transaction, checkpoint)

    def abort(self, transaction: StepArtifactTransaction) -> None:
        self._validate_transaction(transaction, expected_state="open")
        _safe_rmtree(transaction.staging_dir, root=self.staging_root)
        transaction.state = "aborted"
        self._open_transaction = None

    def recover(self) -> None:
        """Finish ready journals, reject unsafe state, and rebuild projections."""

        self._require_open()
        transactions = self._load_transactions()
        ready = [item for item in transactions if item[1]["state"] == "ready"]
        opened = [item for item in transactions if item[1]["state"] == "open"]
        chain = read_authoritative_commit_chain(self.output_dir)
        if not chain and not ready and opened:
            raise ResumeError(
                "Cannot resume a run with only an open transaction and no "
                "authoritative marker",
                path=str(self.output_dir),
            )

        for transaction, journal in ready:
            marker_path = self._commit_path(int(journal["completed_steps"]))
            if marker_path.exists():
                marker = _load_artifact_json(marker_path)
                if marker.get("transaction_id") != transaction.transaction_id:
                    self._quarantine(
                        transaction,
                        "ready transaction conflicts with authoritative marker",
                    )
            else:
                self._publish_ready(
                    transaction,
                    journal["checkpoint"],
                )

        chain = read_authoritative_commit_chain(
            self.output_dir,
            verify_checkpoint_trees=True,
        )
        head_step = 0 if not chain else int(chain[-1]["completed_steps"])
        committed_transactions = {
            marker["transaction_id"] for marker in chain
        }
        for transaction, journal in opened:
            if transaction.transaction_id in committed_transactions:
                self._quarantine(
                    transaction,
                    "open transaction already has an authoritative marker",
                )
            if any(step < head_step for step in journal["staged_steps"]):
                self._quarantine(
                    transaction,
                    "open transaction overlaps authoritative history",
                )
            checkpoint_candidates = tuple(
                transaction.staging_dir.glob("checkpoint_*")
            )
            if checkpoint_candidates:
                self._quarantine(
                    transaction,
                    "open transaction unexpectedly contains a checkpoint",
                )
            _safe_rmtree(transaction.staging_dir, root=self.staging_root)

        self._open_transaction = None
        self.rebuild_projections()
        for transaction, _journal in ready:
            self.cleanup_published_staging(transaction)

    def rebuild_projections(self) -> None:
        """Rebuild manifest/metrics using marker-embedded rows only."""

        self._require_open()
        chain = read_authoritative_commit_chain(self.output_dir)
        records = tuple(
            SampleRecord.from_dict(row)
            for marker in chain
            for step in marker["steps"]
            for row in step["manifest_records"]
        )
        manifest = SampleManifest(run_id=self.run_id, records=records)
        metric_rows = [
            step["core_metric_row"]
            for marker in chain
            for step in marker["steps"]
        ]
        _atomic_write_json(
            self.manifest_path,
            manifest.to_dict(),
            root=self.output_dir,
        )
        text = "".join(
            canonical_json_text(row) + "\n" for row in metric_rows
        )
        _atomic_write_text(
            self.metrics_path,
            text,
            root=self.output_dir,
        )

    def cleanup_published_staging(
        self,
        transaction: StepArtifactTransaction,
    ) -> None:
        self._require_open()
        if not isinstance(transaction, StepArtifactTransaction):
            raise TypeError("transaction must be a StepArtifactTransaction")
        marker = next(
            (
                item
                for item in read_authoritative_commit_chain(self.output_dir)
                if item["transaction_id"] == transaction.transaction_id
            ),
            None,
        )
        if marker is None:
            raise ValueError("cannot clean staging for an unpublished transaction")
        if transaction.staging_dir.exists():
            _safe_rmtree(transaction.staging_dir, root=self.staging_root)

    def apply_checkpoint_retention(self, *, keep_last: int | None) -> None:
        self._require_open()
        if keep_last is None:
            return
        if type(keep_last) is not int or keep_last <= 0:
            raise ValueError("keep_last must be a positive integer or None")
        chain = read_authoritative_commit_chain(self.output_dir)
        retained = {
            marker["checkpoint"]["path"] for marker in chain[-keep_last:]
        }
        if chain:
            retained.add(chain[-1]["checkpoint"]["path"])
        for marker in chain:
            relative = marker["checkpoint"]["path"]
            if relative in retained:
                continue
            path = self.output_dir / relative
            if not path.exists():
                continue
            _safe_rmtree(path, root=self.output_dir)
        _fsync_directory(self.output_dir)

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        fd = self._lock_fd
        self._lock_fd = None
        self._closed = True
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _publish_ready(
        self,
        transaction: StepArtifactTransaction,
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        _validate_staged_checkpoint(checkpoint)
        completed_steps = int(checkpoint["completed_steps"])
        if transaction.completed_steps not in (None, completed_steps):
            raise ArtifactError("transaction completed_steps drifted")
        transaction.completed_steps = completed_steps
        transaction.state = "ready"
        if transaction.staged_steps[-1] + 1 != completed_steps:
            raise ArtifactError("checkpoint step does not match staged steps")

        chain = read_authoritative_commit_chain(self.output_dir)
        expected_start = 0 if not chain else int(chain[-1]["completed_steps"])
        expected_steps = list(range(expected_start, completed_steps))
        if transaction.staged_steps != expected_steps:
            raise ArtifactError(
                "transaction staged steps are not contiguous with authoritative head"
            )
        steps = [
            self._load_staged_step(transaction, step)
            for step in transaction.staged_steps
        ]
        staged_path = self.output_dir / str(checkpoint["staged_path"])
        final_path = self.output_dir / str(checkpoint["final_path"])
        _validate_relative_checkpoint_path(
            str(checkpoint["final_path"]),
            completed_steps,
        )
        _validate_record_checkpoint_paths(
            steps,
            completed_steps=completed_steps,
            checkpoint_path=str(checkpoint["final_path"]),
        )
        for relative in _preview_paths_from_steps(steps):
            self._publish_preview(transaction, relative)
        if staged_path.exists():
            _validated_directory_within(
                staged_path,
                transaction.staging_dir,
                label="ready checkpoint staging path",
            )
            if final_path.exists():
                raise ArtifactError(
                    "ready transaction has both staged and final checkpoints",
                    path=str(final_path),
                )
            os.replace(staged_path, final_path)
            _fsync_directory(self.output_dir)
        elif not final_path.exists():
            raise ArtifactError(
                "ready transaction checkpoint is missing",
                path=str(staged_path),
            )

        from visual_rl.artifacts.checkpoint import checkpoint_tree_sha256

        actual_digest = checkpoint_tree_sha256(
            final_path,
            trusted_root=self.output_dir,
        )
        if actual_digest != checkpoint["tree_sha256"]:
            raise ArtifactError(
                "ready checkpoint tree SHA256 mismatch",
                path=str(final_path),
            )
        marker = {
            "schema_version": COMMIT_SCHEMA_VERSION,
            "kind": "artifact_commit",
            "run_id": self.run_id,
            "transaction_id": transaction.transaction_id,
            "completed_steps": completed_steps,
            "staged_steps": list(transaction.staged_steps),
            "checkpoint": {
                "completed_steps": completed_steps,
                "path": str(checkpoint["final_path"]),
                "tree_sha256": str(checkpoint["tree_sha256"]),
            },
            "steps": steps,
        }
        marker_path = self._commit_path(completed_steps)
        if marker_path.exists():
            existing = _load_artifact_json(marker_path)
            if existing != marker:
                raise ArtifactError(
                    "authoritative marker already exists with different content",
                    path=str(marker_path),
                )
        else:
            _atomic_write_json(marker_path, marker, root=self.commits_dir)
            _fsync_directory(self.commits_dir)
        transaction.state = "committed"
        self._open_transaction = None
        return marker

    def _load_staged_step(
        self,
        transaction: StepArtifactTransaction,
        step: int,
    ) -> dict[str, Any]:
        wrapper = self._load_staged_manifest(transaction, step)
        metric = _load_artifact_json(
            transaction.staging_dir
            / f"step_{step:06d}"
            / "metric.json"
        )
        _validate_metric_row(metric, expected_step=step)
        return {
            "artifact_step": step,
            "manifest_records": wrapper["records"],
            "core_metric_row": metric,
        }

    def _load_staged_manifest(
        self,
        transaction: StepArtifactTransaction,
        step: int,
    ) -> dict[str, Any]:
        path = (
            transaction.staging_dir
            / f"step_{step:06d}"
            / "manifest_records.json"
        )
        wrapper = _load_artifact_json(path)
        if set(wrapper) != {
            "schema_version",
            "run_id",
            "artifact_step",
            "records",
        }:
            raise ArtifactError(
                "staged manifest wrapper fields do not match v3 schema",
                path=str(path),
            )
        if wrapper["schema_version"] != SAMPLE_MANIFEST_SCHEMA_VERSION:
            raise ArtifactError(
                "staged manifest wrapper has unsupported schema_version",
                path=str(path),
            )
        if wrapper["run_id"] != self.run_id or wrapper["artifact_step"] != step:
            raise ArtifactError(
                "staged manifest wrapper identity mismatch",
                path=str(path),
            )
        if not isinstance(wrapper["records"], list) or not wrapper["records"]:
            raise ArtifactError(
                "staged manifest records must be a non-empty list",
                path=str(path),
            )
        records = [
            SampleRecord.from_dict(row) for row in wrapper["records"]
        ]
        if any(record.run_id != self.run_id or record.step != step for record in records):
            raise ArtifactError(
                "staged SampleRecord identity mismatch",
                path=str(path),
            )
        return wrapper

    def _write_journal(
        self,
        transaction: StepArtifactTransaction,
        *,
        checkpoint: Mapping[str, Any] | None,
    ) -> None:
        journal = {
            "schema_version": COMMIT_SCHEMA_VERSION,
            "kind": "artifact_transaction",
            "state": transaction.state,
            "run_id": transaction.run_id,
            "transaction_id": transaction.transaction_id,
            "completed_steps": transaction.completed_steps,
            "staged_steps": list(transaction.staged_steps),
            "checkpoint": (
                None if checkpoint is None else dict(checkpoint)
            ),
        }
        _validate_journal(journal)
        _atomic_write_json(
            transaction.staging_dir / "pending.json",
            journal,
            root=transaction.staging_dir,
        )
        _fsync_directory(transaction.staging_dir)

    def _load_transactions(
        self,
    ) -> list[tuple[StepArtifactTransaction, dict[str, Any]]]:
        result = []
        for path in sorted(self.staging_root.iterdir()):
            if path.name.startswith("quarantine_"):
                continue
            if not path.name.startswith("txn_"):
                raise ArtifactError(
                    "artifact staging contains an unknown entry",
                    path=str(path),
                )
            _validated_directory_within(
                path,
                self.staging_root,
                label="transaction directory",
            )
            journal = _load_artifact_json(path / "pending.json")
            _validate_journal(journal)
            transaction_id = path.name.removeprefix("txn_")
            if (
                journal["transaction_id"] != transaction_id
                or journal["run_id"] != self.run_id
            ):
                raise ArtifactError(
                    "transaction directory identity mismatch",
                    path=str(path),
                )
            result.append(
                (
                    StepArtifactTransaction(
                        transaction_id=transaction_id,
                        run_id=self.run_id,
                        staging_dir=path,
                        completed_steps=journal["completed_steps"],
                        staged_steps=list(journal["staged_steps"]),
                        state=journal["state"],
                    ),
                    journal,
                )
            )
        return result

    def _discover_resume_run_id(self) -> str:
        chain = read_authoritative_commit_chain(self.output_dir)
        run_ids = {marker["run_id"] for marker in chain}
        ready_ids: set[str] = set()
        open_ids: set[str] = set()
        for path in sorted(self.staging_root.iterdir()):
            if not path.name.startswith("txn_"):
                continue
            journal = _load_artifact_json(path / "pending.json")
            _validate_journal(journal)
            if journal["state"] == "ready":
                ready_ids.add(journal["run_id"])
            else:
                open_ids.add(journal["run_id"])
        candidates = run_ids | ready_ids
        if len(candidates) != 1:
            if not candidates and open_ids:
                raise ResumeError(
                    "Cannot infer run_id from an open-only transaction",
                    path=str(self.output_dir),
                )
            raise ResumeError(
                "Resume directory must contain exactly one authoritative run_id",
                path=str(self.output_dir),
            )
        run_id = next(iter(candidates))
        if any(item != run_id for item in open_ids | ready_ids):
            raise ResumeError(
                "Artifact transactions disagree on run_id",
                path=str(self.output_dir),
            )
        return run_id

    def _validate_transaction(
        self,
        transaction: StepArtifactTransaction,
        *,
        expected_state: str,
    ) -> None:
        self._require_open()
        if not isinstance(transaction, StepArtifactTransaction):
            raise TypeError("transaction must be a StepArtifactTransaction")
        if transaction is not self._open_transaction:
            raise ValueError("transaction is not the manager's current transaction")
        if transaction.run_id != self.run_id:
            raise ValueError("transaction run_id does not match manager")
        if transaction.state != expected_state:
            raise ValueError(
                f"transaction state must be {expected_state!r}, "
                f"got {transaction.state!r}"
            )
        expected_dir = self.staging_root / f"txn_{transaction.transaction_id}"
        if transaction.staging_dir != expected_dir:
            raise ValueError("transaction staging path is not canonical")

    def _validate_record_paths(self, record: SampleRecord) -> None:
        for name in (
            "media_path",
            "rollout_cache_path",
            "checkpoint_path",
        ):
            value = getattr(record, name)
            if value is None:
                continue
            path = self.output_dir / value
            _require_lexically_within(path, self.output_dir, label=name)
            if path.exists():
                if path.is_symlink():
                    raise ArtifactError(
                        f"{name} cannot be a symlink",
                        path=str(path),
                    )
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(self.output_dir):
                    raise ArtifactError(
                        f"{name} escapes the output directory",
                        path=str(path),
                    )

    def _validate_staged_preview(
        self,
        transaction: StepArtifactTransaction,
        record: SampleRecord,
    ) -> None:
        relative = _expected_preview_path(record)
        if record.media_path != relative:
            raise ValueError(
                "media_path must match the canonical preview path for its record"
            )
        staged = transaction.staging_dir / relative
        _require_lexically_within(
            staged,
            transaction.staging_dir,
            label="staged preview",
        )
        if staged.is_symlink() or not staged.is_file():
            raise ArtifactError(
                "record media_path has no staged preview file",
                path=str(staged),
            )
        final = self.output_dir / relative
        _require_lexically_within(final, self.output_dir, label="preview")
        if final.exists():
            raise ArtifactError(
                "preview destination already exists before commit",
                path=str(final),
            )

    def _publish_preview(
        self,
        transaction: StepArtifactTransaction,
        relative: str,
    ) -> None:
        staged = transaction.staging_dir / relative
        final = self.output_dir / relative
        _require_lexically_within(
            staged,
            transaction.staging_dir,
            label="ready preview staging path",
        )
        _require_lexically_within(final, self.output_dir, label="ready preview")
        staged_exists = staged.exists()
        final_exists = final.exists()
        if staged_exists and final_exists:
            raise ArtifactError(
                "ready transaction has both staged and final previews",
                path=str(final),
            )
        if not staged_exists and not final_exists:
            raise ArtifactError(
                "ready transaction preview is missing",
                path=str(staged),
            )
        if staged_exists:
            if staged.is_symlink() or not staged.is_file():
                raise ArtifactError(
                    "ready staged preview must be a regular file",
                    path=str(staged),
                )
            _safe_directory(final.parent, create=True)
            os.replace(staged, final)
            _fsync_directory(final.parent)
        elif final.is_symlink() or not final.is_file():
            raise ArtifactError(
                "published preview must be a regular file",
                path=str(final),
            )

    def _commit_path(self, completed_steps: int) -> Path:
        return self.commits_dir / f"commit_{completed_steps:06d}.json"

    def _quarantine(
        self,
        transaction: StepArtifactTransaction,
        reason: str,
    ) -> None:
        destination = self.staging_root / (
            f"quarantine_{transaction.transaction_id}_{uuid.uuid4().hex}"
        )
        os.replace(transaction.staging_dir, destination)
        _fsync_directory(self.staging_root)
        raise ArtifactError(reason, path=str(destination))

    def _acquire_lock(self) -> None:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self._lock_path, flags, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise RuntimeError(
                        f"Artifact directory has an active writer: {self.output_dir}"
                    ) from error
                raise
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.fsync(fd)
        except BaseException:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            raise
        self._lock_fd = fd

    def _require_open(self) -> None:
        if self._closed or self._lock_fd is None:
            raise RuntimeError("ArtifactManager is closed")


def read_authoritative_commit_chain(
    run_root: str | Path,
    *,
    verify_checkpoint_trees: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    """Read and fully validate the exact v2 authoritative marker chain."""

    if not isinstance(verify_checkpoint_trees, bool):
        raise TypeError("verify_checkpoint_trees must be a bool")
    root = _validated_run_root(run_root, create=False)
    commits_dir = root / "commits"
    if not commits_dir.exists():
        return ()
    _safe_directory(commits_dir, create=False)
    marker_paths = []
    for path in sorted(commits_dir.iterdir()):
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.name.startswith("commit_")
            or not path.name.endswith(".json")
        ):
            raise ArtifactError(
                "authoritative commit directory contains an invalid entry",
                path=str(path),
            )
        marker_paths.append(path)

    markers: list[dict[str, Any]] = []
    expected_start = 0
    run_id: str | None = None
    transactions: set[str] = set()
    sample_ids: set[str] = set()
    for path in marker_paths:
        marker = _load_artifact_json(path)
        _validate_marker(marker, path=path)
        completed_steps = int(marker["completed_steps"])
        expected_name = f"commit_{completed_steps:06d}.json"
        if path.name != expected_name:
            raise ArtifactError(
                "authoritative marker filename does not match completed_steps",
                path=str(path),
            )
        if run_id is None:
            run_id = marker["run_id"]
        elif marker["run_id"] != run_id:
            raise ArtifactError("authoritative markers disagree on run_id")
        transaction_id = marker["transaction_id"]
        if transaction_id in transactions:
            raise ArtifactError("authoritative transaction_id is duplicated")
        transactions.add(transaction_id)
        expected_steps = list(range(expected_start, completed_steps))
        if marker["staged_steps"] != expected_steps:
            raise ArtifactError(
                "authoritative staged_steps are not contiguous"
            )
        step_indices = [item["artifact_step"] for item in marker["steps"]]
        if step_indices != expected_steps:
            raise ArtifactError("authoritative step payload order is invalid")
        for payload in marker["steps"]:
            _validate_step_payload(
                payload,
                run_id=marker["run_id"],
                sample_ids=sample_ids,
            )
        checkpoint = marker["checkpoint"]
        if checkpoint["completed_steps"] != completed_steps:
            raise ArtifactError(
                "marker checkpoint completed_steps mismatch",
                path=str(path),
            )
        _validate_relative_checkpoint_path(
            checkpoint["path"],
            completed_steps,
        )
        _validate_record_checkpoint_paths(
            marker["steps"],
            completed_steps=completed_steps,
            checkpoint_path=checkpoint["path"],
        )
        expected_start = completed_steps
        markers.append(marker)

    if verify_checkpoint_trees and markers:
        from visual_rl.artifacts.checkpoint import checkpoint_tree_sha256

        for index, marker in enumerate(markers):
            checkpoint = marker["checkpoint"]
            path = root / checkpoint["path"]
            _require_lexically_within(
                path,
                root,
                label="authoritative checkpoint",
            )
            if not path.exists():
                if index == len(markers) - 1:
                    raise ArtifactError(
                        "latest authoritative checkpoint is missing",
                        path=str(path),
                    )
                continue
            digest = checkpoint_tree_sha256(path, trusted_root=root)
            if digest != checkpoint["tree_sha256"]:
                raise ArtifactError(
                    "authoritative checkpoint tree SHA256 mismatch",
                    path=str(path),
                )
    return tuple(markers)


def _validate_marker(marker: Any, *, path: Path) -> None:
    if not isinstance(marker, dict) or set(marker) != _MARKER_FIELDS:
        raise ArtifactError(
            "authoritative marker fields do not match commit schema v2",
            path=str(path),
        )
    if (
        marker["schema_version"] != COMMIT_SCHEMA_VERSION
        or marker["kind"] != "artifact_commit"
    ):
        raise ArtifactError(
            "unsupported authoritative commit marker",
            path=str(path),
        )
    _non_empty_string("run_id", marker["run_id"])
    _transaction_id(marker["transaction_id"])
    completed_steps = _positive_int(
        "completed_steps",
        marker["completed_steps"],
    )
    if not isinstance(marker["staged_steps"], list):
        raise ArtifactError("staged_steps must be a list", path=str(path))
    _ordered_steps(marker["staged_steps"])
    if not marker["staged_steps"]:
        raise ArtifactError("staged_steps must not be empty", path=str(path))
    if marker["staged_steps"][-1] + 1 != completed_steps:
        raise ArtifactError(
            "completed_steps must equal max(staged_steps)+1",
            path=str(path),
        )
    _validate_checkpoint(marker["checkpoint"])
    if not isinstance(marker["steps"], list) or not marker["steps"]:
        raise ArtifactError("marker steps must be a non-empty list", path=str(path))


def _validate_step_payload(
    payload: Any,
    *,
    run_id: str,
    sample_ids: set[str],
) -> None:
    if not isinstance(payload, dict) or set(payload) != _STEP_FIELDS:
        raise ArtifactError("marker step payload fields are invalid")
    step = _non_negative_int("artifact_step", payload["artifact_step"])
    records = payload["manifest_records"]
    if not isinstance(records, list) or not records:
        raise ArtifactError("manifest_records must be a non-empty list")
    local_ids: set[str] = set()
    for row in records:
        record = SampleRecord.from_dict(row)
        if record.run_id != run_id or record.step != step:
            raise ArtifactError("marker SampleRecord identity mismatch")
        if record.sample_id in local_ids or record.sample_id in sample_ids:
            raise ArtifactError("marker SampleRecord sample_id is duplicated")
        local_ids.add(record.sample_id)
        sample_ids.add(record.sample_id)
    _validate_metric_row(payload["core_metric_row"], expected_step=step)


def _validate_record_checkpoint_paths(
    steps: Sequence[Mapping[str, Any]],
    *,
    completed_steps: int,
    checkpoint_path: str,
) -> None:
    boundary_step = completed_steps - 1
    for payload in steps:
        expected = (
            checkpoint_path
            if payload["artifact_step"] == boundary_step
            else None
        )
        if any(
            row["checkpoint_path"] != expected
            for row in payload["manifest_records"]
        ):
            raise ArtifactError(
                "SampleRecord checkpoint_path does not match commit boundary"
            )


def _expected_preview_path(record: SampleRecord) -> str:
    extension = "jpg" if record.media_type == "image" else "mp4"
    return (
        f"previews/step_{record.step:06d}/"
        f"rank_{record.rank}/sample_{record.sample_index:06d}.{extension}"
    )


def _preview_paths_from_steps(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    for payload in steps:
        for row in payload["manifest_records"]:
            record = SampleRecord.from_dict(row)
            relative = record.media_path
            if relative is None:
                continue
            expected = _expected_preview_path(record)
            if relative != expected:
                raise ArtifactError(
                    "SampleRecord media_path is not a canonical preview path"
                )
            if relative in seen:
                raise ArtifactError("SampleRecord media_path is duplicated")
            seen.add(relative)
            paths.append(relative)
    return tuple(paths)


def _validate_checkpoint(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_FIELDS:
        raise ArtifactError("checkpoint promise fields are invalid")
    _positive_int("checkpoint.completed_steps", value["completed_steps"])
    if not isinstance(value["path"], str):
        raise ArtifactError("checkpoint.path must be a string")
    _sha256(value["tree_sha256"])


def _validate_staged_checkpoint(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _STAGED_CHECKPOINT_FIELDS:
        raise ArtifactError("staged checkpoint promise fields are invalid")
    completed = _positive_int(
        "checkpoint.completed_steps",
        value["completed_steps"],
    )
    _relative_posix("checkpoint.staged_path", value["staged_path"])
    _validate_relative_checkpoint_path(value["final_path"], completed)
    _sha256(value["tree_sha256"])


def _validate_journal(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _JOURNAL_FIELDS:
        raise ArtifactError("transaction journal fields do not match schema v2")
    if (
        value["schema_version"] != COMMIT_SCHEMA_VERSION
        or value["kind"] != "artifact_transaction"
        or value["state"] not in {"open", "ready"}
    ):
        raise ArtifactError("unsupported artifact transaction journal")
    _non_empty_string("run_id", value["run_id"])
    _transaction_id(value["transaction_id"])
    if not isinstance(value["staged_steps"], list):
        raise ArtifactError("journal staged_steps must be a list")
    _ordered_steps(value["staged_steps"])
    if value["state"] == "open":
        if value["completed_steps"] is not None or value["checkpoint"] is not None:
            raise ArtifactError("open journal must not contain a checkpoint")
    else:
        completed = _positive_int("completed_steps", value["completed_steps"])
        if not value["staged_steps"] or value["staged_steps"][-1] + 1 != completed:
            raise ArtifactError("ready journal completed_steps mismatch")
        _validate_staged_checkpoint(value["checkpoint"])


def _core_metric_row(step: int, metrics: StepMetricsLike) -> dict[str, Any]:
    try:
        values = to_plain_dict(metrics.values)
        sample_count = metrics.sample_count
        active_count = metrics.active_transition_count
    except AttributeError as error:
        raise TypeError(
            "metrics must define values/sample_count/active_transition_count"
        ) from error
    if not isinstance(values, dict):
        raise TypeError("metrics.values must be a mapping")
    if any(name in values for name in _METRIC_BASE_FIELDS):
        raise ValueError("metrics.values contains a reserved core metric field")
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("metric values must be finite Python floats")
    _positive_int("sample_count", sample_count)
    _positive_int("active_transition_count", active_count)
    return {
        "schema_version": CORE_METRIC_SCHEMA_VERSION,
        "step": step,
        "sample_count": sample_count,
        "active_transition_count": active_count,
        **values,
    }


def _validate_metric_row(row: Any, *, expected_step: int) -> None:
    if not isinstance(row, dict):
        raise ArtifactError("core metric row must be an object")
    if any(name not in row for name in _METRIC_BASE_FIELDS):
        raise ArtifactError("core metric row is missing required fields")
    if row["schema_version"] != CORE_METRIC_SCHEMA_VERSION:
        raise ArtifactError("unsupported core metric schema_version")
    if row["step"] != expected_step:
        raise ArtifactError("core metric step mismatch")
    _positive_int("sample_count", row["sample_count"])
    _positive_int(
        "active_transition_count",
        row["active_transition_count"],
    )
    for name, value in row.items():
        if name in _METRIC_BASE_FIELDS:
            continue
        if not isinstance(name, str) or not name:
            raise ArtifactError("metric names must be non-empty strings")
        if type(value) is not float or not math.isfinite(value):
            raise ArtifactError("metric values must be finite Python floats")


def _validated_run_root(
    value: str | Path,
    *,
    create: bool,
) -> Path:
    requested = Path(value).absolute()
    if requested.is_symlink():
        raise ValueError(f"artifact output directory cannot be a symlink: {value}")
    if create:
        requested.mkdir(parents=True, exist_ok=True)
    if not requested.is_dir():
        raise ValueError(f"artifact output directory is not a directory: {value}")
    return requested.resolve(strict=True)


def _safe_directory(path: Path, *, create: bool) -> None:
    if path.is_symlink():
        raise ArtifactError("artifact directory cannot be a symlink", path=str(path))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ArtifactError("artifact path is not a directory", path=str(path))


def _validated_directory_within(
    path: Path,
    root: Path,
    *,
    label: str,
) -> Path:
    _require_lexically_within(path, root, label=label)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactError(f"{label} is not a safe directory", path=str(path))
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ArtifactError(f"{label} escapes its trusted root", path=str(path))
    return resolved


def _require_lexically_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise ArtifactError(f"{label} escapes its trusted root", path=str(path)) from error


def _relative_posix(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactError(f"{name} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactError(f"{name} must be a normalized relative POSIX path")
    if path.as_posix() != value:
        raise ArtifactError(f"{name} must be a normalized relative POSIX path")
    return value


def _validate_relative_checkpoint_path(value: Any, completed_steps: int) -> str:
    path = _relative_posix("checkpoint.path", value)
    expected = f"checkpoint_{completed_steps:06d}"
    if path != expected:
        raise ArtifactError(
            f"checkpoint.path must be {expected!r}, got {path!r}"
        )
    return path


def _transaction_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError("transaction_id must be 32 lowercase hex characters")
    return value


def _sha256(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError("tree_sha256 must be 64 lowercase hex characters")
    return value


def _non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{name} must be a non-empty string")
    return value


def _non_negative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ArtifactError(f"{name} must be a positive integer")
    return value


def _ordered_steps(values: list[Any]) -> None:
    if any(type(value) is not int or value < 0 for value in values):
        raise ArtifactError("staged_steps must contain non-negative integers")
    if values != sorted(set(values)):
        raise ArtifactError("staged_steps must be sorted and unique")


def _atomic_write_json(path: Path, value: Any, *, root: Path) -> None:
    _atomic_write_text(
        path,
        canonical_json_text(value) + "\n",
        root=root,
    )


def _load_artifact_json(path: Path) -> Any:
    try:
        return strict_json_load(path)
    except ValueError as error:
        raise ArtifactError(
            f"Invalid artifact JSON: {error}",
            path=str(path),
        ) from error


def _atomic_write_text(path: Path, text: str, *, root: Path) -> None:
    _require_lexically_within(path, root, label="artifact write")
    if path.exists() and path.is_symlink():
        raise ArtifactError("artifact destination cannot be a symlink", path=str(path))
    _safe_directory(path.parent, create=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _safe_rmtree(path: Path, *, root: Path) -> None:
    _require_lexically_within(path, root, label="artifact delete")
    if path == root or path.is_symlink() or not path.is_dir():
        raise ArtifactError("artifact delete target is unsafe", path=str(path))
    shutil.rmtree(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    _validated_directory_within(root, root, label="fsync root")
    directories = []
    for current, dir_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.append(current_path)
        for name in dir_names:
            child = current_path / name
            if child.is_symlink():
                raise ArtifactError("artifact tree contains a symlink", path=str(child))
        for name in file_names:
            child = current_path / name
            child_stat = child.lstat()
            if not stat.S_ISREG(child_stat.st_mode):
                raise ArtifactError(
                    "artifact tree contains a non-regular file",
                    path=str(child),
                )
            fd = os.open(child, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    for directory in reversed(directories):
        _fsync_directory(directory)
