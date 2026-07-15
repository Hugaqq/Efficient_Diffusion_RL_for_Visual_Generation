"""Run-scoped artifact persistence and checkpoint-cycle transactions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import errno
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any
import uuid

from visual_rl.artifacts.builder import ManifestBuilder
from visual_rl.artifacts.checkpoint import (
    checkpoint_tree_sha256,
    load_json,
    strict_json_loads,
)
from visual_rl.artifacts.manifest import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleRecord,
)
from visual_rl.artifacts.serialization import redact_artifact_config, to_jsonable
from visual_rl.core.types import RewardBatch, RolloutBatch


ARTIFACT_SCHEMA_VERSION = SAMPLE_MANIFEST_SCHEMA_VERSION
COMMIT_SCHEMA_VERSION = "1"
COMMIT_RUNTIME_SCHEMA_VERSION = "1"
_RUNTIME_METRIC_NAMES = frozenset(
    {
        "artifact_commit_time_s",
        "artifact_cycle_samples_per_second",
        "artifact_cycle_steps",
        "artifact_cycle_time_s",
        "artifact_stage_time_s",
        "checkpoint_time_s",
        "checkpoint_write_time_s",
        "post_commit_bookkeeping_time_s",
        "samples_per_second",
        "step_time_s",
    }
)
_RUNTIME_MEASUREMENT_BOUNDARY = (
    "after_authoritative_commit_latest_and_retention_before_"
    "runtime_sidecar_persist_and_projection_refresh"
)
_PROJECTION_NAMES = (
    "sample_manifest.json",
    "reward_table.json",
    "metrics.jsonl",
    "prompt_set.json",
    "visual_report.md",
)


@dataclass
class StepArtifactTransaction:
    """One checkpoint cycle containing one or more zero-based artifact steps."""

    transaction_id: str
    run_id: str
    staging_dir: Path
    completed_steps: int | None = None
    staged_steps: list[int] = field(default_factory=list)
    state: str = "open"


class ArtifactManager:
    """Persist legacy projections and transactional checkpoint-cycle artifacts."""

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        *,
        config: Any | None = None,
        resume: bool = False,
        hold_writer_lock: bool | None = None,
    ):
        requested_output_dir = Path(output_dir).absolute()
        if requested_output_dir.is_symlink():
            raise ValueError(
                f"Artifact output directory cannot be a symlink: {output_dir}"
            )
        requested_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = requested_output_dir.resolve(strict=True)
        self._output_root = self.output_dir
        self.run_id = run_id
        self.builder = ManifestBuilder(run_id)
        self.manifest_path = self.output_dir / "sample_manifest.json"
        self.metric_path = self.output_dir / "metrics.jsonl"
        self.commits_dir = self.output_dir / "commits"
        self.staging_root = self.output_dir / ".staging"
        self._lock_path = self.output_dir / ".artifact.lock"
        self._lock_fd: int | None = None
        self._closed = False
        self.post_commit_errors: list[dict[str, str]] = []
        # The old runner has no close hook yet. Its record path briefly owns the lock
        # per call, while the explicit transaction API retains it until close().
        self._legacy_lock_per_call = not (
            config is None if hold_writer_lock is None else bool(hold_writer_lock)
        )
        self._acquire_lock()
        try:
            has_commits = bool(
                self._load_commit_markers(verify_checkpoints=bool(resume))
            )
            if not resume and (self.manifest_path.exists() or has_commits):
                raise FileExistsError(
                    f"Artifact directory already contains run data: {self.output_dir}. "
                    "Use resume=True to continue it."
                )
            if has_commits:
                self.rebuild_projections()
            self.manifest = self._load_manifest(resume or has_commits)
            self._metrics_by_step = self._load_metrics(resume or has_commits)
            if config is not None and not resume:
                self._write_json(
                    self.output_dir / "config.resolved.json",
                    redact_artifact_config(config),
                )
            if self._legacy_lock_per_call:
                self._release_lock()
        except BaseException:
            try:
                self._release_lock()
            except BaseException:
                pass
            self._closed = True
            raise

    def __enter__(self) -> ArtifactManager:
        self._ensure_writer_lock()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release the run-directory writer lock."""

        if self._closed:
            return
        self._release_lock()
        self._closed = True

    @classmethod
    def recover_run(cls, output_dir: str | Path) -> list[dict[str, Any]]:
        """Recover pending transactions after deriving their persisted run identity."""

        root = Path(output_dir).absolute()
        if root.is_symlink() or not root.is_dir():
            raise ValueError(
                f"Artifact recovery root is not a safe directory: {output_dir}"
            )
        staging_root = root / ".staging"
        if staging_root.is_symlink():
            raise ValueError(
                f"artifact staging root cannot be a symlink: {staging_root}"
            )
        if not staging_root.exists():
            return []
        if not staging_root.is_dir():
            raise ValueError(
                f"artifact staging root is not a directory: {staging_root}"
            )
        transaction_dirs = sorted(staging_root.glob("txn_*"))
        if not transaction_dirs:
            return []
        if any(path.is_symlink() or not path.is_dir() for path in transaction_dirs):
            raise ValueError(
                f"artifact staging root contains an unsafe transaction: {staging_root}"
            )
        run_id = cls._discover_recovery_run_id(root, transaction_dirs)
        if run_id is None:
            raise ValueError(
                "Cannot recover artifact transactions without a valid persisted run_id"
            )
        with cls(root, run_id, resume=True) as manager:
            return manager.recover()

    @classmethod
    def _discover_recovery_run_id(
        cls,
        root: Path,
        transaction_dirs: list[Path],
    ) -> str | None:
        commits_dir = root / "commits"
        if commits_dir.is_symlink():
            raise ValueError(
                f"authoritative commit directory cannot be a symlink: {commits_dir}"
            )
        marker_ids: set[str] = set()
        if commits_dir.exists():
            if not commits_dir.is_dir():
                raise ValueError(
                    f"authoritative commit path is not a directory: {commits_dir}"
                )
            for marker_path in sorted(commits_dir.glob("commit_*.json")):
                if marker_path.is_symlink() or not marker_path.is_file():
                    raise ValueError(
                        "authoritative commit marker is not a regular file: "
                        f"{marker_path}"
                    )
                marker = cls._load_json_if_object(marker_path)
                if marker is None:
                    raise ValueError(
                        f"authoritative commit marker is invalid JSON: {marker_path}"
                    )
                run_id = marker.get("run_id")
                if not isinstance(run_id, str) or not run_id:
                    raise ValueError(
                        f"authoritative commit marker has no valid run_id: {marker_path}"
                    )
                marker_ids.add(run_id)
        if len(marker_ids) > 1:
            raise ValueError("authoritative commit markers disagree on run_id")

        journal_ids: set[str] = set()
        for transaction_dir in transaction_dirs:
            journal = cls._load_json_if_object(transaction_dir / "pending.json")
            transaction_id = transaction_dir.name.removeprefix("txn_")
            if (
                journal is None
                or journal.get("schema_version") != COMMIT_SCHEMA_VERSION
                or journal.get("kind") != "artifact_transaction"
                or journal.get("transaction_id") != transaction_id
            ):
                continue
            run_id = journal.get("run_id")
            if isinstance(run_id, str) and run_id:
                journal_ids.add(run_id)
        if marker_ids:
            return next(iter(marker_ids))
        if len(journal_ids) > 1:
            raise ValueError("pending artifact transactions disagree on run_id")
        return next(iter(journal_ids), None)

    def acquire_writer_lock(self) -> None:
        """Acquire the writer lock without changing the manager lifecycle."""

        self._ensure_writer_lock()

    def release_writer_lock(self) -> None:
        """Release the writer lock while leaving the manager reusable."""

        self._release_lock()

    def begin_transaction(
        self, *, completed_steps: int | None = None
    ) -> StepArtifactTransaction:
        """Begin one checkpoint cycle without making any step visible."""

        self._ensure_writer_lock()
        if completed_steps is not None and completed_steps <= 0:
            raise ValueError(
                "completed_steps is a one-based count and must be positive"
            )
        transaction_id = uuid.uuid4().hex
        staging_dir = self.staging_root / f"txn_{transaction_id}"
        self._safe_mkdir(staging_dir)
        transaction = StepArtifactTransaction(
            transaction_id=transaction_id,
            run_id=self.run_id,
            staging_dir=staging_dir,
            completed_steps=completed_steps,
        )
        self._write_journal(transaction)
        return transaction

    def stage_step(
        self,
        transaction: StepArtifactTransaction,
        *,
        step: int,
        batch: RolloutBatch,
        rewards: RewardBatch,
        metrics: dict[str, Any],
        media_type: str,
        rollout_type: str | None = None,
        media_paths: list[str | Path | None] | str | Path | None = None,
        rollout_cache_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[SampleRecord]:
        """Stage one zero-based step as three independently durable rows."""

        records = self.builder.build_records(
            step=step,
            batch=batch,
            rewards=rewards,
            media_type=media_type,
            rollout_type=rollout_type,
            media_paths=media_paths,
            rollout_cache_path=rollout_cache_path,
            checkpoint_path=checkpoint_path,
        )
        return self.stage_records(
            transaction,
            step=step,
            records=records,
            metrics=metrics,
        )

    def stage_records(
        self,
        transaction: StepArtifactTransaction,
        *,
        step: int,
        records: list[SampleRecord],
        metrics: dict[str, Any],
    ) -> list[SampleRecord]:
        """Stage rank-merged records through the canonical transaction format."""

        self._validate_transaction(transaction, expected_state="open")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("artifact step must be a non-negative integer")
        if step in transaction.staged_steps:
            raise ValueError(f"artifact step {step} is already staged")
        if self._committed_steps().intersection({step}):
            raise FileExistsError(f"artifact step {step} is already committed")
        if not isinstance(records, list):
            raise TypeError("records must be a list of SampleRecord values")
        if not records:
            raise ValueError("artifact records must not be empty")
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary")

        if any(not isinstance(record, SampleRecord) for record in records):
            raise TypeError("records must contain only SampleRecord values")
        sample_ids = [record.sample_id for record in records]
        if any(
            not isinstance(sample_id, str) or not sample_id.strip()
            for sample_id in sample_ids
        ):
            raise ValueError("artifact sample_id values must be non-empty strings")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("artifact records contain duplicate sample_id values")
        self._validate_records(records, expected_step=step)
        overlap = self._known_sample_ids(transaction).intersection(sample_ids)
        if overlap:
            raise ValueError(
                f"artifact sample_id values are already staged or committed: "
                f"{sorted(overlap)}"
            )

        metric_values = to_jsonable(dict(metrics))
        self._validate_json_value(metric_values, label="artifact metrics")
        step_dir = transaction.staging_dir / f"step_{step:06d}"
        self._safe_mkdir(step_dir)
        manifest_rows = self._manifest_rows(records)
        reward_rows = [self._reward_row(record) for record in records]
        metric_row = {
            **metric_values,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "step": step,
        }
        self._write_json(
            step_dir / "manifest_records.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "artifact_step": step,
                "records": manifest_rows,
            },
        )
        self._write_json(
            step_dir / "reward_rows.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "artifact_step": step,
                "records": reward_rows,
            },
        )
        self._write_json(step_dir / "metric.json", metric_row)
        transaction.staged_steps.append(step)
        transaction.staged_steps.sort()
        self._write_journal(transaction)
        return list(records)

    def commit(
        self,
        transaction: StepArtifactTransaction,
        *,
        completed_steps: int | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Atomically publish a staged checkpoint cycle via its commit marker."""

        self._ensure_writer_lock()
        marker = self._existing_transaction_marker(transaction.transaction_id)
        if marker is not None:
            transaction.state = "committed"
            return marker
        self._validate_transaction(transaction, expected_state={"open", "ready"})
        recovering_ready_transaction = transaction.state == "ready"
        resolved_completed = completed_steps or transaction.completed_steps
        if resolved_completed is None:
            if not transaction.staged_steps:
                raise ValueError("cannot infer completed_steps without staged steps")
            resolved_completed = max(transaction.staged_steps) + 1
        self._validate_step_semantics(transaction.staged_steps, resolved_completed)
        transaction.completed_steps = resolved_completed

        marker_path = self._commit_path(resolved_completed)
        if marker_path.exists():
            raise FileExistsError(
                f"commit {resolved_completed} already belongs to another transaction"
            )
        overlap = self._committed_steps().intersection(transaction.staged_steps)
        if overlap:
            raise FileExistsError(
                f"artifact steps are already committed: {sorted(overlap)}"
            )

        expected_checkpoint = (
            self._ready_checkpoint_expectation(transaction)
            if recovering_ready_transaction
            else None
        )
        checkpoint = self._prepare_checkpoint(
            transaction,
            resolved_completed,
            checkpoint_path,
        )
        if recovering_ready_transaction:
            self._validate_recovery_checkpoint(expected_checkpoint, checkpoint)
        transaction.state = "ready"
        self._write_journal(transaction, checkpoint=checkpoint)
        steps = [
            self._load_staged_step(transaction, step)
            for step in transaction.staged_steps
        ]
        if checkpoint is not None and checkpoint.get("staged_path") is None:
            self._fsync_tree(self.output_dir / checkpoint["final_path"])
        self._fsync_tree(transaction.staging_dir)
        checkpoint = self._publish_checkpoint(checkpoint)

        marker = {
            "schema_version": COMMIT_SCHEMA_VERSION,
            "kind": "artifact_commit",
            "run_id": self.run_id,
            "transaction_id": transaction.transaction_id,
            "commit_id": resolved_completed,
            "completed_steps": resolved_completed,
            "artifact_step_semantics": "zero_based",
            "completed_steps_semantics": "one_based_count",
            "staged_steps": list(transaction.staged_steps),
            "checkpoint": checkpoint,
            "steps": steps,
        }
        self._validate_marker(marker, marker_path)
        self._write_json(marker_path, marker)
        transaction.state = "committed"
        self._recoverable_post_commit(
            "projection_refresh",
            self.rebuild_projections,
        )
        self._recoverable_post_commit(
            "staging_cleanup",
            lambda: self._safe_rmtree(transaction.staging_dir),
        )
        return marker

    def _recoverable_post_commit(self, operation: str, callback: Any) -> Any:
        """Attempt derived work without weakening a persisted commit marker."""

        try:
            return callback()
        except Exception as exc:
            self.post_commit_errors.append(
                {
                    "operation": operation,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return None

    def record_commit_runtime(
        self,
        marker: dict[str, Any],
        metrics_by_step: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """Persist post-commit timing overlays without rewriting the commit marker.

        The measurement boundary is immediately before this sidecar is written, so
        the sidecar and the projection refresh cannot be included in their own
        reported durations.
        """

        self._ensure_writer_lock()
        completed_steps = int(marker["completed_steps"])
        marker_path = self._commit_path(completed_steps)
        self._validate_marker(marker, marker_path)
        persisted_marker = self._load_json_if_object(marker_path)
        if persisted_marker is None or persisted_marker.get(
            "transaction_id"
        ) != marker.get("transaction_id"):
            raise ValueError("runtime metrics require the persisted commit marker")
        staged_steps = [int(step) for step in marker["staged_steps"]]
        if sorted(metrics_by_step) != staged_steps:
            raise ValueError("runtime metrics must cover exactly the committed steps")

        steps = []
        for step in staged_steps:
            metrics = dict(metrics_by_step[step])
            unknown = set(metrics).difference(_RUNTIME_METRIC_NAMES)
            if unknown:
                raise ValueError(
                    f"unsupported commit runtime metric names: {sorted(unknown)}"
                )
            if not self._valid_runtime_metrics(metrics):
                raise ValueError(
                    "commit runtime metrics must be finite and non-negative"
                )
            steps.append(
                {
                    "artifact_step": step,
                    "metrics": to_jsonable(metrics),
                }
            )
        runtime = {
            "schema_version": COMMIT_RUNTIME_SCHEMA_VERSION,
            "kind": "artifact_commit_runtime",
            "run_id": self.run_id,
            "transaction_id": marker["transaction_id"],
            "commit_id": completed_steps,
            "completed_steps": completed_steps,
            "measurement_boundary": _RUNTIME_MEASUREMENT_BOUNDARY,
            "steps": steps,
        }
        self._write_json(self._runtime_path(completed_steps), runtime)
        self.rebuild_projections()
        return runtime

    def abort(self, transaction: StepArtifactTransaction) -> None:
        """Discard an unpublished transaction without touching committed objects."""

        self._validate_transaction(transaction, expected_state={"open", "ready"})
        if self._existing_transaction_marker(transaction.transaction_id) is not None:
            raise RuntimeError("cannot abort a committed transaction")
        journal = self._load_json_if_object(transaction.staging_dir / "pending.json")
        checkpoint = journal.get("checkpoint") if journal else None
        if isinstance(checkpoint, dict):
            final_name = checkpoint.get("final_path")
            final_path = self.output_dir / str(final_name)
            if final_name and final_path.exists():
                raise RuntimeError(
                    "checkpoint was already published; recover or quarantine the transaction"
                )
        self._safe_rmtree(transaction.staging_dir)
        transaction.state = "aborted"

    def recover(self) -> list[dict[str, Any]]:
        """Finish ready journals and isolate conflicts; open journals stay invisible."""

        self._ensure_writer_lock()
        audit: list[dict[str, Any]] = []
        if self.staging_root.is_symlink():
            raise ValueError(
                f"artifact staging root cannot be a symlink: {self.staging_root}"
            )
        markers = self._load_commit_markers(verify_checkpoints=True)
        if not self.staging_root.exists():
            if markers:
                self.rebuild_projections()
            return audit
        if not self.staging_root.is_dir():
            raise ValueError(
                f"artifact staging root is not a directory: {self.staging_root}"
            )
        for staging_dir in sorted(self.staging_root.glob("txn_*")):
            if staging_dir.is_symlink() or not staging_dir.is_dir():
                continue
            journal = self._load_json_if_object(staging_dir / "pending.json")
            if journal is None or journal.get("run_id") != self.run_id:
                audit.append(self._quarantine(staging_dir, "invalid_journal"))
                continue
            transaction_id = str(journal.get("transaction_id", ""))
            existing = self._existing_transaction_marker(transaction_id)
            if existing is not None:
                self._safe_rmtree(staging_dir)
                audit.append(
                    {
                        "action": "cleanup",
                        "transaction_id": transaction_id,
                        "reason": "commit_marker_exists",
                    }
                )
                continue
            state = journal.get("state")
            if state != "ready":
                audit.append(
                    {
                        "action": "ignored",
                        "transaction_id": transaction_id,
                        "reason": "transaction_not_ready",
                    }
                )
                continue
            try:
                transaction = StepArtifactTransaction(
                    transaction_id=transaction_id,
                    run_id=self.run_id,
                    staging_dir=staging_dir,
                    completed_steps=int(journal["completed_steps"]),
                    staged_steps=[
                        int(step) for step in journal.get("staged_steps", [])
                    ],
                    state="ready",
                )
                checkpoint = self._ready_checkpoint_expectation(transaction)
                checkpoint_path: Path | None = None
                if checkpoint is not None:
                    staged_name = checkpoint.get("staged_path")
                    final_name = checkpoint["final_path"]
                    if staged_name and (self.output_dir / staged_name).exists():
                        checkpoint_path = self.output_dir / staged_name
                    else:
                        checkpoint_path = self.output_dir / final_name
                self.commit(
                    transaction,
                    completed_steps=transaction.completed_steps,
                    checkpoint_path=checkpoint_path,
                )
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                audit.append(
                    self._quarantine(
                        staging_dir,
                        f"recovery_conflict:{type(exc).__name__}:{exc}",
                    )
                )
            else:
                audit.append(
                    {
                        "action": "recovered",
                        "transaction_id": transaction_id,
                        "completed_steps": transaction.completed_steps,
                    }
                )
        if self._load_commit_markers(verify_checkpoints=True):
            self.rebuild_projections()
        return audit

    def rebuild_projections(self) -> dict[str, int]:
        """Rebuild caches from markers plus validated runtime sidecars."""

        self._ensure_writer_lock()
        markers = self._load_commit_markers()
        records: list[SampleRecord] = []
        reward_rows: list[dict[str, Any]] = []
        metrics: dict[int, dict[str, Any]] = {}
        for marker in markers:
            runtime_metrics = self._load_commit_runtime(marker)
            for step_payload in marker["steps"]:
                step = int(step_payload["artifact_step"])
                for row in step_payload["manifest_records"]:
                    record_data = dict(row)
                    record_data.pop("schema_version", None)
                    records.append(SampleRecord(**record_data))
                reward_rows.extend(step_payload["reward_rows"])
                metrics[step] = {
                    **dict(step_payload["metric_row"]),
                    **runtime_metrics.get(step, {}),
                }
        manifest = SampleManifest(
            run_id=self.run_id,
            schema_version=ARTIFACT_SCHEMA_VERSION,
            records=records,
        )
        manifest.validate()
        self._write_json(self.manifest_path, manifest.to_dict())
        self._write_json(
            self.output_dir / "reward_table.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "records": reward_rows,
            },
        )
        self._write_projection_extras(manifest, metrics)
        self.manifest = manifest
        self._metrics_by_step = metrics
        return {
            "commits": len(markers),
            "steps": len(metrics),
            "records": len(records),
        }

    def apply_retention(
        self,
        *,
        checkpoint_keep_last: int | None = None,
        rollout_cache_keep_last: int | None = None,
        rollout_cache_max_bytes: int | None = None,
        artifact_max_bytes: int | None = None,
        rollout_root: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Delete only committed checkpoints, known rollout triplets, or projections."""

        self._ensure_writer_lock()
        for name, value in (
            ("checkpoint_keep_last", checkpoint_keep_last),
            ("rollout_cache_keep_last", rollout_cache_keep_last),
            ("rollout_cache_max_bytes", rollout_cache_max_bytes),
            ("artifact_max_bytes", artifact_max_bytes),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        markers = self._load_commit_markers()
        audit: list[dict[str, Any]] = []
        checkpoints = self._retained_checkpoints(markers)
        checkpoint_delete: set[int] = set()
        if checkpoint_keep_last is not None:
            keep_count = max(1, checkpoint_keep_last)
            kept = checkpoints[-keep_count:]
            kept_steps = {step for step, _path in kept}
            checkpoint_delete.update(
                step for step, _path in checkpoints if step not in kept_steps
            )
        for step, path in checkpoints:
            if step in checkpoint_delete:
                audit.append(
                    self._delete_checkpoint(path, step, "checkpoint_keep_last")
                )

        requested_rollout_dir = (
            Path(rollout_root) if rollout_root else self.output_dir / "rollouts"
        )
        rollout_dir = self._validated_rollout_root(requested_rollout_dir)
        rollout_groups = (
            self._rollout_groups(markers, rollout_dir)
            if rollout_dir is not None
            else []
        )
        rollout_delete: set[int] = set()
        if rollout_cache_keep_last is not None:
            kept = (
                rollout_groups[-rollout_cache_keep_last:]
                if rollout_cache_keep_last
                else []
            )
            kept_steps = {step for step, _paths in kept}
            rollout_delete.update(
                step for step, _paths in rollout_groups if step not in kept_steps
            )
        if rollout_cache_max_bytes is not None:
            remaining_bytes = sum(
                self._paths_size(paths)
                for step, paths in rollout_groups
                if step not in rollout_delete
            )
            for step, paths in rollout_groups:
                if remaining_bytes <= rollout_cache_max_bytes:
                    break
                if step not in rollout_delete:
                    rollout_delete.add(step)
                    remaining_bytes -= self._paths_size(paths)
        for step, paths in rollout_groups:
            if step in rollout_delete:
                if rollout_dir is None:
                    raise RuntimeError("validated rollout root disappeared")
                audit.append(
                    self._delete_rollout_group(
                        paths,
                        step,
                        "rollout_cache_policy",
                        rollout_root=rollout_dir,
                    )
                )

        if artifact_max_bytes is not None:
            audit.extend(self._enforce_artifact_budget(markers, artifact_max_bytes))
        return audit

    def record(
        self,
        *,
        step: int,
        batch: RolloutBatch,
        rewards: RewardBatch,
        metrics: dict[str, Any],
        media_type: str,
        rollout_type: str | None = None,
        media_paths: list[str | Path | None] | str | Path | None = None,
        rollout_cache_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[SampleRecord]:
        """Compatibility path for the runner until transaction integration."""

        acquired_for_call = self._lock_fd is None
        self._ensure_writer_lock()
        try:
            if self._load_commit_markers():
                raise RuntimeError(
                    "record() cannot append to a commit-log run; use transactions"
                )
            records = self.builder.build_records(
                step=step,
                batch=batch,
                rewards=rewards,
                media_type=media_type,
                rollout_type=rollout_type,
                media_paths=media_paths,
                rollout_cache_path=rollout_cache_path,
                checkpoint_path=checkpoint_path,
            )
            self.manifest.records = [
                record for record in self.manifest.records if record.step != step
            ]
            for record in records:
                self.manifest.add(record)
            self._metrics_by_step[step] = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                **to_jsonable(dict(metrics)),
                "step": step,
            }
            self._flush()
            return records
        finally:
            if self._legacy_lock_per_call and acquired_for_call:
                self._release_lock()

    def truncate_from_step(self, start_step: int) -> None:
        """Discard legacy projections newer than a resumed checkpoint."""

        acquired_for_call = self._lock_fd is None
        self._ensure_writer_lock()
        try:
            if start_step < 0:
                raise ValueError("start_step must be non-negative")
            if self._load_commit_markers():
                raise RuntimeError(
                    "truncate_from_step() cannot rewrite committed transaction history"
                )
            self.manifest.records = [
                record for record in self.manifest.records if record.step < start_step
            ]
            self._metrics_by_step = {
                step: metrics
                for step, metrics in self._metrics_by_step.items()
                if step < start_step
            }
            self._flush()
        finally:
            if self._legacy_lock_per_call and acquired_for_call:
                self._release_lock()

    def _load_manifest(self, resume: bool) -> SampleManifest:
        if resume and self.manifest_path.exists():
            manifest = SampleManifest.load(self.manifest_path)
            if manifest.run_id != self.run_id:
                raise ValueError(
                    "Existing manifest run_id does not match ArtifactManager run_id"
                )
            return manifest
        return SampleManifest(
            run_id=self.run_id,
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )

    def _load_metrics(self, resume: bool) -> dict[int, dict[str, Any]]:
        if not resume or not self.metric_path.exists():
            return {}
        metrics: dict[int, dict[str, Any]] = {}
        for line in self.metric_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = strict_json_loads(line)
            if not isinstance(row, dict):
                raise ValueError("artifact metric row must be a JSON object")
            metrics[int(row["step"])] = row
        return metrics

    def _flush(self) -> None:
        self.manifest.schema_version = ARTIFACT_SCHEMA_VERSION
        self._write_json(self.manifest_path, self.manifest.to_dict())
        reward_rows = [self._reward_row(record) for record in self.manifest.records]
        self._write_json(
            self.output_dir / "reward_table.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "records": reward_rows,
            },
        )
        self._write_projection_extras(self.manifest, self._metrics_by_step)

    def _write_projection_extras(
        self,
        manifest: SampleManifest,
        metrics: dict[int, dict[str, Any]],
    ) -> None:
        self._write_json(
            self.output_dir / "prompt_set.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "prompts": self._prompt_rows(manifest.records),
            },
        )
        metric_lines = [
            json.dumps(metrics[step], sort_keys=True, ensure_ascii=False)
            for step in sorted(metrics)
        ]
        self._write_text(
            self.metric_path,
            "\n".join(metric_lines) + ("\n" if metric_lines else ""),
        )
        self._write_text(
            self.output_dir / "visual_report.md",
            self._visual_report(manifest.records),
        )

    @staticmethod
    def _reward_row(record: SampleRecord) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "sample_id": record.sample_id,
            "step": record.step,
            "reward_values": record.reward_values,
        }

    def _manifest_rows(
        self, records: list[SampleRecord]
    ) -> list[dict[str, Any]]:
        return [
            {
                **to_jsonable(asdict(record)),
                "schema_version": ARTIFACT_SCHEMA_VERSION,
            }
            for record in records
        ]

    def _validate_records(
        self,
        records: list[SampleRecord],
        *,
        expected_step: int,
    ) -> None:
        for record in records:
            if not isinstance(record, SampleRecord):
                raise TypeError("records must contain only SampleRecord values")
            if record.run_id != self.run_id:
                raise ValueError("record run_id does not match ArtifactManager run_id")
            if (
                isinstance(record.step, bool)
                or not isinstance(record.step, int)
                or record.step != expected_step
            ):
                raise ValueError(
                    "record step does not match staged artifact step: "
                    f"{record.step!r} != {expected_step}"
                )
            if isinstance(record.sample_index, bool) or not isinstance(
                record.sample_index, int
            ):
                raise ValueError("record sample_index must be a non-negative integer")
            if not isinstance(record.prompt, str):
                raise ValueError("record prompt must be a string")
            if not isinstance(record.media_type, str) or record.media_type not in {
                "image",
                "video",
            }:
                raise ValueError("record media_type must be 'image' or 'video'")
            for field_name in (
                "prompt_metadata",
                "timestep_summary",
                "reward_values",
                "model_metadata",
            ):
                if not isinstance(getattr(record, field_name), dict):
                    raise ValueError(f"record {field_name} must be a dictionary")
            if record.seed is not None and (
                isinstance(record.seed, bool) or not isinstance(record.seed, int)
            ):
                raise ValueError("record seed must be an integer or None")
            for field_name in (
                "rollout_type",
                "prompt_id",
                "group_id",
            ):
                value = getattr(record, field_name)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"record {field_name} must be a string or None")
            for field_name in (
                "media_path",
                "rollout_cache_path",
                "checkpoint_path",
            ):
                self._validate_record_path(
                    getattr(record, field_name),
                    field_name=field_name,
                )
            payload = to_jsonable(asdict(record))
            self._validate_json_value(
                payload,
                label=f"artifact record {record.sample_id!r}",
            )

        manifest = SampleManifest(
            run_id=self.run_id,
            schema_version=ARTIFACT_SCHEMA_VERSION,
            records=list(records),
        )
        manifest.validate()

    def _validate_record_path(
        self,
        value: str | None,
        *,
        field_name: str,
    ) -> None:
        if value is None:
            return
        if not isinstance(value, str) or not value:
            raise ValueError(f"record {field_name} must be a non-empty path or None")
        candidate = Path(value)
        absolute = candidate if candidate.is_absolute() else self.output_dir / candidate
        try:
            resolved = absolute.resolve(strict=False)
            resolved.relative_to(self._output_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"record {field_name} escapes artifact output directory: {value}"
            ) from exc

    @staticmethod
    def _validate_json_value(value: Any, *, label: str) -> None:
        try:
            json.dumps(value, allow_nan=False, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} is not strict JSON data") from exc

    def _known_sample_ids(
        self, transaction: StepArtifactTransaction
    ) -> set[str]:
        sample_ids = {record.sample_id for record in self.manifest.records}
        if not self.staging_root.is_dir() or self.staging_root.is_symlink():
            return sample_ids
        for staging_dir in sorted(self.staging_root.glob("txn_*")):
            if staging_dir.is_symlink() or not staging_dir.is_dir():
                continue
            journal = self._load_json_if_object(staging_dir / "pending.json")
            if journal is None or journal.get("run_id") != transaction.run_id:
                continue
            for raw_step in journal.get("staged_steps", []):
                step = int(raw_step)
                path = staging_dir / f"step_{step:06d}" / "manifest_records.json"
                payload = self._require_json_object(path)
                rows = payload.get("records")
                staged_records = self._records_from_rows(rows, expected_step=step)
                sample_ids.update(record.sample_id for record in staged_records)
        return sample_ids

    def _records_from_rows(
        self,
        rows: Any,
        *,
        expected_step: int,
    ) -> list[SampleRecord]:
        if not isinstance(rows, list) or not rows:
            raise ValueError("staged manifest records must be a non-empty list")
        records: list[SampleRecord] = []
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            ):
                raise ValueError("staged manifest record schema_version mismatch")
            record_data = dict(row)
            record_data.pop("schema_version")
            try:
                records.append(SampleRecord(**record_data))
            except TypeError as exc:
                raise ValueError("staged manifest record schema is invalid") from exc
        self._validate_records(records, expected_step=expected_step)
        if len({record.sample_id for record in records}) != len(records):
            raise ValueError("staged manifest records contain duplicate sample_id values")
        return records

    @staticmethod
    def _prompt_rows(records: list[SampleRecord]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            row = {"prompt": record.prompt, "metadata": record.prompt_metadata}
            key = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                rows.append(row)
        return rows

    def _visual_report(self, records: list[SampleRecord]) -> str:
        image_count = sum(record.media_type == "image" for record in records)
        video_count = sum(record.media_type == "video" for record in records)
        lines = [
            f"# VisualRL Run Report: {self.run_id}",
            "",
            f"- Samples: {len(records)}",
            f"- Images: {image_count}",
            f"- Videos: {video_count}",
            "",
            "## Samples",
            "",
            "| sample_id | step | media_type | prompt | weighted_total |",
            "|---|---:|---|---|---:|",
        ]
        for record in records:
            reward = record.reward_values.get("weighted_total", "")
            prompt = record.prompt.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {record.sample_id} | {record.step} | {record.media_type} | "
                f"{prompt} | {reward} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _write_journal(
        self,
        transaction: StepArtifactTransaction,
        *,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        existing = self._load_json_if_object(transaction.staging_dir / "pending.json")
        self._write_json(
            transaction.staging_dir / "pending.json",
            {
                "schema_version": COMMIT_SCHEMA_VERSION,
                "kind": "artifact_transaction",
                "run_id": self.run_id,
                "transaction_id": transaction.transaction_id,
                "state": transaction.state,
                "completed_steps": transaction.completed_steps,
                "artifact_step_semantics": "zero_based",
                "completed_steps_semantics": "one_based_count",
                "staged_steps": list(transaction.staged_steps),
                "checkpoint": checkpoint
                if checkpoint is not None
                else (existing or {}).get("checkpoint"),
            },
        )

    def _ready_checkpoint_expectation(
        self,
        transaction: StepArtifactTransaction,
    ) -> dict[str, Any] | None:
        """Read and validate the immutable checkpoint promise in a ready journal."""

        journal = self._require_json_object(transaction.staging_dir / "pending.json")
        completed_steps = transaction.completed_steps
        if completed_steps is None:
            raise ValueError("ready transaction has no completed_steps")
        try:
            journal_steps = [int(step) for step in journal["staged_steps"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ready transaction journal has invalid staged_steps") from exc
        if (
            journal.get("schema_version") != COMMIT_SCHEMA_VERSION
            or journal.get("kind") != "artifact_transaction"
            or journal.get("run_id") != self.run_id
            or journal.get("transaction_id") != transaction.transaction_id
            or journal.get("state") != "ready"
            or isinstance(journal.get("completed_steps"), bool)
            or journal.get("completed_steps") != completed_steps
            or journal_steps != transaction.staged_steps
            or journal.get("artifact_step_semantics") != "zero_based"
            or journal.get("completed_steps_semantics") != "one_based_count"
        ):
            raise ValueError("ready transaction journal identity mismatch")

        checkpoint = journal.get("checkpoint")
        if checkpoint is None:
            return None
        if not isinstance(checkpoint, dict):
            raise ValueError("ready transaction checkpoint metadata must be an object")
        expected_name = f"checkpoint_{completed_steps:06d}"
        expected_staged = str(
            (transaction.staging_dir / expected_name).relative_to(self.output_dir)
        )
        digest = checkpoint.get("sha256")
        if (
            isinstance(checkpoint.get("completed_steps"), bool)
            or checkpoint.get("completed_steps") != completed_steps
            or checkpoint.get("path") != expected_name
            or checkpoint.get("final_path") != expected_name
            or checkpoint.get("staged_path") not in {None, expected_staged}
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("ready transaction checkpoint metadata is invalid")
        return dict(checkpoint)

    @staticmethod
    def _validate_recovery_checkpoint(
        expected: dict[str, Any] | None,
        actual: dict[str, Any] | None,
    ) -> None:
        """Require recovered checkpoint bytes to match the ready journal promise."""

        if expected is None or actual is None:
            if expected is actual:
                return
            raise ValueError("recovered checkpoint presence does not match ready journal")
        for key in ("completed_steps", "path", "final_path", "sha256"):
            if actual.get(key) != expected.get(key):
                raise ValueError(
                    f"recovered checkpoint {key} does not match ready journal"
                )

    def _load_staged_step(
        self, transaction: StepArtifactTransaction, step: int
    ) -> dict[str, Any]:
        step_dir = transaction.staging_dir / f"step_{step:06d}"
        manifest = self._require_json_object(step_dir / "manifest_records.json")
        rewards = self._require_json_object(step_dir / "reward_rows.json")
        metric = self._require_json_object(step_dir / "metric.json")
        if (
            manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or rewards.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or manifest.get("run_id") != self.run_id
            or rewards.get("run_id") != self.run_id
            or int(manifest.get("artifact_step", -1)) != step
            or int(rewards.get("artifact_step", -1)) != step
            or int(metric.get("step", -1)) != step
        ):
            raise ValueError(f"staged payload step mismatch for step {step}")
        records = self._records_from_rows(
            manifest.get("records"),
            expected_step=step,
        )
        reward_rows = rewards.get("records")
        if (
            not isinstance(reward_rows, list)
            or len(reward_rows) != len(records)
            or any(
                not isinstance(row, dict)
                or row.get("schema_version") != ARTIFACT_SCHEMA_VERSION
                or row.get("sample_id") != record.sample_id
                or int(row.get("step", -1)) != step
                for row, record in zip(reward_rows, records, strict=False)
            )
        ):
            raise ValueError(f"staged reward rows are invalid for step {step}")
        if metric.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"staged metric schema is invalid for step {step}")
        self._validate_json_value(metric, label=f"staged metric for step {step}")
        return {
            "artifact_step": step,
            "manifest_records": self._manifest_rows(records),
            "reward_rows": reward_rows,
            "metric_row": metric,
        }

    def _prepare_checkpoint(
        self,
        transaction: StepArtifactTransaction,
        completed_steps: int,
        checkpoint_path: str | Path | None,
    ) -> dict[str, Any] | None:
        expected_name = f"checkpoint_{completed_steps:06d}"
        default_staged = transaction.staging_dir / expected_name
        source = Path(checkpoint_path).absolute() if checkpoint_path else None
        if source is None and default_staged.exists():
            source = default_staged
        if source is None:
            return None
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {source}")
        self._reject_tree_symlinks(source)
        final_path = self.output_dir / expected_name
        if source != final_path and source != default_staged:
            if default_staged.exists():
                raise FileExistsError(
                    f"staged checkpoint already exists: {default_staged}"
                )
            self._safe_destination(default_staged)
            shutil.copytree(source, default_staged)
            self._fsync_tree(default_staged)
            source = default_staged
        digest = self._tree_digest(source)
        return {
            "completed_steps": completed_steps,
            "path": expected_name,
            "final_path": expected_name,
            "staged_path": (
                str(source.relative_to(self.output_dir))
                if source != final_path
                else None
            ),
            "sha256": digest,
        }

    def _publish_checkpoint(
        self, checkpoint: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if checkpoint is None:
            return None
        final_path = self.output_dir / checkpoint["final_path"]
        staged_name = checkpoint.get("staged_path")
        staged_path = self.output_dir / staged_name if staged_name else None
        expected_digest = checkpoint["sha256"]
        if final_path.exists():
            if (
                final_path.is_symlink()
                or self._tree_digest(final_path) != expected_digest
            ):
                raise FileExistsError(
                    f"checkpoint destination conflicts with pending transaction: {final_path}"
                )
            if staged_path is not None and staged_path.exists():
                raise FileExistsError(
                    f"both staged and final checkpoint exist: {final_path}"
                )
        else:
            if staged_path is None or not staged_path.exists():
                raise FileNotFoundError("pending checkpoint is missing from staging")
            self._safe_destination(final_path)
            os.replace(staged_path, final_path)
            self._fsync_directory(final_path.parent)
        published = dict(checkpoint)
        published.pop("staged_path", None)
        return published

    def _load_commit_markers(
        self,
        *,
        verify_checkpoints: bool = False,
    ) -> list[dict[str, Any]]:
        if self.commits_dir.is_symlink():
            raise ValueError(
                f"authoritative commit directory cannot be a symlink: {self.commits_dir}"
            )
        if not self.commits_dir.exists():
            return []
        if not self.commits_dir.is_dir():
            raise ValueError(
                f"authoritative commit path is not a directory: {self.commits_dir}"
            )
        valid: list[dict[str, Any]] = []
        seen_steps: set[int] = set()
        for path in sorted(self.commits_dir.glob("commit_*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"authoritative commit marker is not a regular file: {path}"
                )
            marker = self._load_json_if_object(path)
            if marker is None:
                raise ValueError(f"authoritative commit marker is invalid JSON: {path}")
            try:
                self._validate_marker(marker, path)
                if verify_checkpoints:
                    self._validate_marker_checkpoint_tree(marker)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"authoritative commit marker is invalid: {path}: {exc}"
                ) from exc
            steps = {int(step) for step in marker["staged_steps"]}
            if seen_steps.intersection(steps):
                raise ValueError(
                    f"authoritative commit markers overlap artifact steps: {path}"
                )
            seen_steps.update(steps)
            valid.append(marker)
        valid.sort(key=lambda marker: int(marker["completed_steps"]))
        return valid

    def _load_commit_runtime(self, marker: dict[str, Any]) -> dict[int, dict[str, Any]]:
        completed_steps = int(marker["completed_steps"])
        runtime = self._load_json_if_object(self._runtime_path(completed_steps))
        if runtime is None:
            return {}
        try:
            identity_matches = (
                runtime.get("schema_version") == COMMIT_RUNTIME_SCHEMA_VERSION
                and runtime.get("kind") == "artifact_commit_runtime"
                and runtime.get("run_id") == self.run_id
                and runtime.get("transaction_id") == marker.get("transaction_id")
                and int(runtime.get("commit_id", -1)) == completed_steps
                and int(runtime.get("completed_steps", -1)) == completed_steps
                and runtime.get("measurement_boundary") == _RUNTIME_MEASUREMENT_BOUNDARY
            )
        except (TypeError, ValueError):
            return {}
        if not identity_matches:
            return {}
        expected_steps = [int(step) for step in marker["staged_steps"]]
        steps = runtime.get("steps")
        if not isinstance(steps, list) or len(steps) != len(expected_steps):
            return {}
        overlays: dict[int, dict[str, Any]] = {}
        for expected, payload in zip(expected_steps, steps, strict=True):
            if not isinstance(payload, dict):
                return {}
            if int(payload.get("artifact_step", -1)) != expected:
                return {}
            values = payload.get("metrics")
            if (
                not isinstance(values, dict)
                or not set(values) <= _RUNTIME_METRIC_NAMES
                or not self._valid_runtime_metrics(values)
            ):
                return {}
            overlays[expected] = dict(values)
        return overlays

    @staticmethod
    def _valid_runtime_metrics(metrics: dict[str, Any]) -> bool:
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)) or value < 0:
                return False
            if name == "artifact_cycle_steps" and (
                not isinstance(value, int) or value < 1
            ):
                return False
        return True

    def _validate_marker(self, marker: dict[str, Any], path: Path) -> None:
        if marker.get("schema_version") != COMMIT_SCHEMA_VERSION:
            raise ValueError("unsupported commit schema_version")
        if (
            marker.get("kind") != "artifact_commit"
            or marker.get("run_id") != self.run_id
        ):
            raise ValueError("commit marker identity mismatch")
        completed = int(marker["completed_steps"])
        if int(marker["commit_id"]) != completed or path != self._commit_path(
            completed
        ):
            raise ValueError("commit marker path does not match completed_steps")
        if marker.get("artifact_step_semantics") != "zero_based":
            raise ValueError("commit marker must declare zero-based artifact steps")
        if marker.get("completed_steps_semantics") != "one_based_count":
            raise ValueError("commit marker must declare one-based completed_steps")
        staged_steps = [int(step) for step in marker["staged_steps"]]
        self._validate_step_semantics(staged_steps, completed)
        steps = marker.get("steps")
        if not isinstance(steps, list) or len(steps) != len(staged_steps):
            raise ValueError("commit marker step payload count mismatch")
        for expected, payload in zip(staged_steps, steps, strict=True):
            if int(payload.get("artifact_step", -1)) != expected:
                raise ValueError("commit marker step payload order mismatch")
            metric = payload.get("metric_row")
            if not isinstance(metric, dict) or int(metric.get("step", -1)) != expected:
                raise ValueError("commit marker metric step mismatch")
            if metric.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
                raise ValueError("commit marker metric schema_version mismatch")
            self._validate_json_value(
                metric,
                label=f"commit marker metric for step {expected}",
            )
            records = self._records_from_rows(
                payload.get("manifest_records"),
                expected_step=expected,
            )
            reward_rows = payload.get("reward_rows")
            if (
                not isinstance(reward_rows, list)
                or len(reward_rows) != len(records)
                or any(
                    not isinstance(row, dict)
                    or row.get("schema_version") != ARTIFACT_SCHEMA_VERSION
                    or row.get("sample_id") != record.sample_id
                    or int(row.get("step", -1)) != expected
                    for row, record in zip(reward_rows, records, strict=False)
                )
            ):
                raise ValueError("commit marker reward_rows are invalid")
        checkpoint = marker.get("checkpoint")
        if checkpoint is not None:
            expected_name = f"checkpoint_{completed:06d}"
            digest = checkpoint.get("sha256")
            if (
                int(checkpoint.get("completed_steps", -1)) != completed
                or checkpoint.get("path") != expected_name
                or checkpoint.get("final_path") != expected_name
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("checkpoint metadata does not match commit")

    def _validate_marker_checkpoint_tree(self, marker: dict[str, Any]) -> None:
        checkpoint = marker.get("checkpoint")
        if checkpoint is None:
            return
        checkpoint_path = self.output_dir / str(checkpoint["final_path"])
        if checkpoint_path.is_symlink():
            raise ValueError(
                f"committed checkpoint path cannot be a symlink: {checkpoint_path}"
            )
        if not checkpoint_path.exists():
            return
        if not checkpoint_path.is_dir():
            raise ValueError(
                f"committed checkpoint path is not a directory: {checkpoint_path}"
            )
        try:
            actual_digest = self._tree_digest(checkpoint_path)
        except RuntimeError as exc:
            raise ValueError(
                f"committed checkpoint tree is unsafe: {checkpoint_path}: {exc}"
            ) from exc
        if actual_digest != checkpoint["sha256"]:
            raise ValueError(
                f"committed checkpoint tree SHA256 mismatch: {checkpoint_path}"
            )

    @staticmethod
    def _validate_step_semantics(steps: list[int], completed_steps: int) -> None:
        if completed_steps <= 0:
            raise ValueError(
                "completed_steps is a one-based count and must be positive"
            )
        if not steps:
            raise ValueError("a transaction must contain at least one artifact step")
        if steps != sorted(set(steps)) or min(steps) < 0:
            raise ValueError("artifact steps must be unique sorted zero-based values")
        if max(steps) + 1 != completed_steps:
            raise ValueError(
                "completed_steps must equal the final zero-based artifact step plus one"
            )

    def _validate_transaction(
        self,
        transaction: StepArtifactTransaction,
        *,
        expected_state: str | set[str],
    ) -> None:
        self._ensure_writer_lock()
        expected = (
            {expected_state} if isinstance(expected_state, str) else expected_state
        )
        if transaction.run_id != self.run_id:
            raise ValueError("transaction belongs to another run")
        if transaction.state not in expected:
            raise RuntimeError(
                f"transaction state is {transaction.state!r}, expected {sorted(expected)}"
            )
        expected_dir = self.staging_root / f"txn_{transaction.transaction_id}"
        if transaction.staging_dir != expected_dir or not expected_dir.is_dir():
            raise ValueError("transaction staging directory is invalid")

    def _existing_transaction_marker(
        self, transaction_id: str
    ) -> dict[str, Any] | None:
        for marker in self._load_commit_markers():
            if marker.get("transaction_id") == transaction_id:
                return marker
        return None

    def _committed_steps(self) -> set[int]:
        return {
            int(step)
            for marker in self._load_commit_markers()
            for step in marker["staged_steps"]
        }

    def _commit_path(self, completed_steps: int) -> Path:
        return self.commits_dir / f"commit_{completed_steps:06d}.json"

    def _runtime_path(self, completed_steps: int) -> Path:
        return self.commits_dir / f"runtime_{completed_steps:06d}.json"

    def _quarantine(self, staging_dir: Path, reason: str) -> dict[str, Any]:
        quarantine = self.staging_root / "quarantine"
        self._safe_mkdir(quarantine)
        destination = quarantine / f"{staging_dir.name}_{uuid.uuid4().hex[:8]}"
        self._safe_destination(destination)
        os.replace(staging_dir, destination)
        self._fsync_directory(quarantine)
        return {
            "action": "quarantine",
            "transaction_id": staging_dir.name.removeprefix("txn_"),
            "reason": reason,
            "path": str(destination.relative_to(self.output_dir)),
        }

    def _retained_checkpoints(
        self, markers: list[dict[str, Any]]
    ) -> list[tuple[int, Path]]:
        checkpoints: list[tuple[int, Path]] = []
        for marker in markers:
            checkpoint = marker.get("checkpoint")
            if not isinstance(checkpoint, dict):
                continue
            step = int(marker["completed_steps"])
            path = self.output_dir / f"checkpoint_{step:06d}"
            if path.is_dir() and not path.is_symlink():
                checkpoints.append((step, path))
        return sorted(checkpoints)

    def _rollout_groups(
        self,
        markers: list[dict[str, Any]],
        rollout_root: Path,
    ) -> list[tuple[int, tuple[Path, ...]]]:
        committed = sorted(
            {int(step) for marker in markers for step in marker["staged_steps"]}
        )
        rank_dirs = sorted(
            path
            for path in rollout_root.iterdir()
            if self._is_rank_rollout_dir(path)
        )
        directories = [rollout_root, *rank_dirs]
        groups: list[tuple[int, tuple[Path, ...]]] = []
        for step in committed:
            paths = tuple(
                path
                for directory in directories
                for path in self._validated_rollout_triplet(
                    rollout_root,
                    directory,
                    step,
                )
            )
            if paths:
                groups.append((step, paths))
        return groups

    @staticmethod
    def _validated_rollout_root(path: Path) -> Path | None:
        absolute = path.absolute()
        if absolute.is_symlink() or not absolute.is_dir():
            return None
        try:
            return absolute.resolve(strict=True)
        except OSError:
            return None

    @staticmethod
    def _is_rank_rollout_dir(path: Path) -> bool:
        prefix = "rank_"
        suffix = path.name.removeprefix(prefix)
        return (
            path.name.startswith(prefix)
            and len(suffix) == 4
            and suffix.isascii()
            and suffix.isdigit()
            and path.is_dir()
            and not path.is_symlink()
        )

    @classmethod
    def _validated_rollout_triplet(
        cls,
        rollout_root: Path,
        directory: Path,
        step: int,
    ) -> tuple[Path, ...]:
        if directory.is_symlink() or not directory.is_dir():
            return ()
        try:
            directory.resolve(strict=True).relative_to(rollout_root)
        except (OSError, ValueError):
            return ()
        base = directory / f"batch_{step:06d}"
        paths = (
            base.with_suffix(".pt"),
            base.with_suffix(".media.pt"),
            base.with_suffix(".json"),
        )
        if all(cls._is_regular_file_in_root(rollout_root, path) for path in paths):
            return paths
        return ()

    @staticmethod
    def _is_regular_file_in_root(root: Path, path: Path) -> bool:
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            return False
        return stat.S_ISREG(metadata.st_mode)

    def _delete_checkpoint(
        self, path: Path, completed_steps: int, reason: str
    ) -> dict[str, Any]:
        size = self._path_size(path)
        self._safe_rmtree(path)
        return {
            "action": "delete",
            "category": "checkpoint",
            "completed_steps": completed_steps,
            "paths": [str(path)],
            "bytes": size,
            "reason": reason,
        }

    def _delete_rollout_group(
        self,
        paths: tuple[Path, ...],
        step: int,
        reason: str,
        *,
        rollout_root: Path,
    ) -> dict[str, Any]:
        if not paths or any(
            not self._is_regular_file_in_root(rollout_root, path) for path in paths
        ):
            raise ValueError("refusing to delete an invalid rollout cache group")
        size = self._paths_size(paths)
        for path in paths:
            path.unlink()
        return {
            "action": "delete",
            "category": "rollout_cache",
            "artifact_step": step,
            "paths": [str(path) for path in paths],
            "bytes": size,
            "reason": reason,
        }

    def _enforce_artifact_budget(
        self, markers: list[dict[str, Any]], max_bytes: int
    ) -> list[dict[str, Any]]:
        audit: list[dict[str, Any]] = []
        checkpoints = self._retained_checkpoints(markers)
        while len(checkpoints) > 1 and self._owned_artifact_bytes(markers) > max_bytes:
            step, path = checkpoints.pop(0)
            audit.append(self._delete_checkpoint(path, step, "artifact_max_bytes"))
        if self._owned_artifact_bytes(markers) > max_bytes:
            for name in _PROJECTION_NAMES:
                path = self.output_dir / name
                if path.is_file() and not path.is_symlink():
                    size = path.stat().st_size
                    path.unlink()
                    audit.append(
                        {
                            "action": "delete",
                            "category": "projection",
                            "paths": [str(path)],
                            "bytes": size,
                            "reason": "artifact_max_bytes",
                        }
                    )
                    if self._owned_artifact_bytes(markers) <= max_bytes:
                        break
        remaining = self._owned_artifact_bytes(markers)
        if remaining > max_bytes:
            audit.append(
                {
                    "action": "budget_unsatisfied",
                    "category": "artifact",
                    "bytes": remaining,
                    "max_bytes": max_bytes,
                    "reason": "authoritative_commits_and_latest_checkpoint_are_not_deletable",
                }
            )
        return audit

    def _owned_artifact_bytes(self, markers: list[dict[str, Any]]) -> int:
        paths: list[Path] = []
        if self.commits_dir.is_dir() and not self.commits_dir.is_symlink():
            paths.extend(
                path
                for path in self.commits_dir.glob("commit_*.json")
                if path.is_file() and not path.is_symlink()
            )
        paths.extend(path for _step, path in self._retained_checkpoints(markers))
        paths.extend(
            path
            for name in _PROJECTION_NAMES
            if (path := self.output_dir / name).is_file() and not path.is_symlink()
        )
        return sum(self._path_size(path) for path in paths)

    @staticmethod
    def _paths_size(paths: tuple[Path, ...]) -> int:
        return sum(path.stat().st_size for path in paths)

    def _path_size(self, path: Path) -> int:
        if path.is_symlink():
            raise ValueError(f"refusing to size symlink: {path}")
        if path.is_file():
            return path.stat().st_size
        return sum(
            child.stat().st_size
            for child in path.rglob("*")
            if child.is_file() and not child.is_symlink()
        )

    def _acquire_lock(self) -> None:
        self._safe_destination(self._lock_path)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self._lock_path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError(
                    f"artifact lock cannot be a symlink: {self._lock_path}"
                ) from exc
            raise
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError(
                f"Artifact directory already has an active writer: {self.output_dir}"
            ) from exc
        except BaseException:
            os.close(fd)
            raise
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()}\n".encode())
            os.fsync(fd)
        except BaseException:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            raise
        self._lock_fd = fd

    def _ensure_writer_lock(self) -> None:
        if self._closed:
            raise RuntimeError("ArtifactManager is closed")
        if self._lock_fd is None:
            self._acquire_lock()

    def _release_lock(self) -> None:
        if self._lock_fd is None:
            return
        fd = self._lock_fd
        self._lock_fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _safe_destination(self, path: Path) -> None:
        absolute = path.absolute()
        try:
            absolute.relative_to(self._output_root)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes output directory: {path}") from exc
        if absolute.is_symlink():
            raise ValueError(f"artifact target cannot be a symlink: {path}")
        if absolute == self._output_root:
            return
        current = absolute.parent
        while current != self._output_root:
            if current.is_symlink():
                raise ValueError(f"artifact parent cannot be a symlink: {current}")
            current = current.parent

    def _safe_mkdir(self, path: Path) -> None:
        self._safe_destination(path)
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError(f"artifact directory cannot be a symlink: {path}")

    def _safe_rmtree(self, path: Path) -> None:
        self._safe_destination(path)
        if not path.exists():
            return
        self._reject_tree_symlinks(path)
        shutil.rmtree(path)

    def _write_json(self, path: Path, data: Any) -> None:
        text = json.dumps(
            to_jsonable(data),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        self._write_text(path, text + "\n")

    def _write_text(self, path: Path, text: str) -> None:
        self._safe_destination(path)
        self._safe_mkdir(path.parent)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            self._safe_destination(path)
            os.replace(tmp_path, path)
            self._fsync_directory(path.parent)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _load_json_if_object(path: Path) -> dict[str, Any] | None:
        try:
            if path.is_symlink() or not path.is_file():
                return None
            data = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _require_json_object(self, path: Path) -> dict[str, Any]:
        data = self._load_json_if_object(path)
        if data is None:
            raise ValueError(f"expected JSON object: {path}")
        return data

    @staticmethod
    def _reject_tree_symlinks(path: Path) -> None:
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in artifact tree: {path}")
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_symlink():
                    raise ValueError(
                        f"symlink is not allowed in artifact tree: {child}"
                    )

    @classmethod
    def _tree_digest(cls, path: Path) -> str:
        return checkpoint_tree_sha256(path, trusted_root=path)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @classmethod
    def _fsync_tree(cls, path: Path) -> None:
        cls._reject_tree_symlinks(path)
        children = list(path.rglob("*"))
        for child in children:
            if child.is_file():
                flags = os.O_RDONLY
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(child, flags)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        for directory in sorted(
            (child for child in children if child.is_dir()),
            reverse=True,
        ):
            cls._fsync_directory(directory)
        cls._fsync_directory(path)
