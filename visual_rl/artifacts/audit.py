"""Read-only, fail-closed audit of authoritative run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any

from visual_rl.artifacts.checkpoint import (
    checkpoint_tree_sha256,
    load_json,
    read_and_validate_training_state,
)
from visual_rl.artifacts.manifest import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleRecord,
)


_MARKER_PATTERN = re.compile(r"commit_(\d{6})\.json")


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_run_root(run_dir: str | Path) -> Path:
    candidate = Path(run_dir)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise RuntimeError(f"Run artifact directory does not exist: {candidate}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(
            f"Run artifact root must be a real directory, not a symlink: {candidate}"
        )
    return candidate.resolve(strict=True)


def _trusted_file(root: Path, path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes trusted run root: {candidate}") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"{label} contains an unsafe relative component: {candidate}")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise FileNotFoundError(f"Missing {label}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} path must not contain symlinks: {current}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular file: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes trusted run root: {candidate}") from exc
    if resolved.parent != candidate.parent.resolve(strict=True):
        raise RuntimeError(f"{label} escapes its parent directory: {candidate}")
    return resolved


def _read_authoritative_markers(root: Path) -> list[dict[str, Any]]:
    commits_dir = root / "commits"
    if commits_dir.is_symlink() or not commits_dir.is_dir():
        raise RuntimeError(
            f"Authoritative commit directory is not a safe directory: {commits_dir}"
        )
    markers: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    run_id: str | None = None
    for path in sorted(commits_dir.glob("commit_*.json")):
        match = _MARKER_PATTERN.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"Authoritative commit marker is not a regular marker file: {path}"
            )
        try:
            marker = load_json(path)
            completed = marker["completed_steps"]
            if (
                isinstance(completed, bool)
                or not isinstance(completed, int)
                or completed <= 0
                or int(match.group(1)) != completed
                or marker.get("commit_id") != completed
                or marker.get("schema_version") != "1"
                or marker.get("kind") != "artifact_commit"
            ):
                raise ValueError("marker identity mismatch")
            marker_run_id = marker.get("run_id")
            if not isinstance(marker_run_id, str) or not marker_run_id:
                raise ValueError("marker run_id is invalid")
            if run_id is None:
                run_id = marker_run_id
            elif marker_run_id != run_id:
                raise ValueError("marker run_id values disagree")
            staged_steps = marker.get("staged_steps")
            if (
                not isinstance(staged_steps, list)
                or any(
                    isinstance(step, bool) or not isinstance(step, int) or step < 0
                    for step in staged_steps
                )
                or staged_steps != sorted(set(staged_steps))
                or not staged_steps
                or max(staged_steps) + 1 != completed
                or seen_steps.intersection(staged_steps)
            ):
                raise ValueError("marker staged_steps are invalid or overlapping")
            steps = marker.get("steps")
            if not isinstance(steps, list) or len(steps) != len(staged_steps):
                raise ValueError("marker step payload count mismatch")
            checkpoint = marker.get("checkpoint")
            expected_name = f"checkpoint_{completed:06d}"
            digest = checkpoint.get("sha256") if isinstance(checkpoint, dict) else None
            if (
                not isinstance(checkpoint, dict)
                or checkpoint.get("completed_steps") != completed
                or checkpoint.get("path") != expected_name
                or checkpoint.get("final_path") != expected_name
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("marker checkpoint promise is invalid")
        except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Authoritative commit marker is invalid: {path}: {exc}"
            ) from exc
        seen_steps.update(staged_steps)
        markers.append(marker)
    if not markers:
        raise RuntimeError(f"Run has no authoritative commit markers: {commits_dir}")
    markers.sort(key=lambda marker: int(marker["completed_steps"]))
    return markers


def _records_from_step(
    payload: dict[str, Any],
    *,
    run_id: str,
    step: int,
) -> list[SampleRecord]:
    if payload.get("artifact_step") != step:
        raise RuntimeError(f"Authoritative payload step mismatch for step {step}")
    rows = payload.get("manifest_records")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Authoritative records are missing for step {step}")
    records: list[SampleRecord] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != SAMPLE_MANIFEST_SCHEMA_VERSION
        ):
            raise RuntimeError(f"Authoritative record schema is invalid for step {step}")
        values = dict(row)
        values.pop("schema_version", None)
        records.append(SampleRecord(**values))
    SampleManifest(
        run_id=run_id,
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        records=records,
    ).validate()
    if any(record.step != step for record in records):
        raise RuntimeError(f"Authoritative record step mismatch for step {step}")
    return records


def audit_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Audit marker-owned rows and safely validate every retained checkpoint.

    Commit markers are the only authority for step rows and checkpoint promises.
    Derived projections such as ``latest.json`` and ``sample_manifest.json`` are
    deliberately not used to decide validity.
    """

    root = _safe_run_root(run_dir)
    resolved_config = load_json(
        _trusted_file(root, root / "config.resolved.json", label="resolved config")
    )
    markers = _read_authoritative_markers(root)
    run_id = str(markers[0]["run_id"])
    errors: list[str] = []
    warnings: list[str] = []
    metrics: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    all_records: list[SampleRecord] = []

    for marker_index, marker in enumerate(markers):
        completed = int(marker["completed_steps"])
        checkpoint_name = f"checkpoint_{completed:06d}"
        checkpoint = root / checkpoint_name
        checkpoint_exists = checkpoint.exists() or checkpoint.is_symlink()
        checkpoint_metadata: dict[str, Any] | None = None
        if checkpoint_exists:
            expected_digest = marker["checkpoint"]["sha256"]
            actual_digest = checkpoint_tree_sha256(
                checkpoint,
                trusted_root=root,
            )
            if actual_digest != expected_digest:
                raise RuntimeError(
                    f"Committed checkpoint tree SHA256 mismatch: {checkpoint}"
                )
            checkpoint_metadata = load_json(checkpoint / "checkpoint.json")
            validated = read_and_validate_training_state(
                checkpoint,
                config=resolved_config,
                trusted_root=root,
                use_checkpoint_implementation_identity=True,
            )
            if validated.step != completed:
                errors.append(
                    f"checkpoint {checkpoint_name} completed-step disagrees with marker"
                )
        elif marker_index == len(markers) - 1:
            errors.append(f"newest committed checkpoint is missing: {checkpoint_name}")
        else:
            warnings.append(f"retained audit skipped pruned checkpoint: {checkpoint_name}")

        for expected_step, payload in zip(
            marker["staged_steps"], marker["steps"], strict=True
        ):
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Authoritative step payload is invalid for step {expected_step}"
                )
            step = int(expected_step)
            records = _records_from_step(payload, run_id=run_id, step=step)
            all_records.extend(records)
            metric = payload.get("metric_row")
            if (
                not isinstance(metric, dict)
                or metric.get("schema_version") != SAMPLE_MANIFEST_SCHEMA_VERSION
                or metric.get("step") != step
            ):
                raise RuntimeError(f"Authoritative metric row is invalid for step {step}")
            metrics.append(metric)
            reward_rows = payload.get("reward_rows")
            if not isinstance(reward_rows, list) or len(reward_rows) != len(records):
                raise RuntimeError(f"Authoritative reward rows are invalid for step {step}")
            if [row.get("sample_id") for row in reward_rows] != [
                record.sample_id for record in records
            ]:
                errors.append(f"step {step} reward rows disagree with manifest records")

            reward_values = [
                float(record.reward_values["weighted_total"]) for record in records
            ]
            reward_mean = sum(reward_values) / len(reward_values)
            if not math.isclose(
                float(metric.get("reward_mean", math.nan)),
                reward_mean,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                errors.append(f"step {step} metric reward_mean disagrees with records")

            sample_ids = [record.sample_id for record in records]
            if len(sample_ids) != len(set(sample_ids)):
                errors.append(f"step {step} contains duplicate sample IDs")
            prompts = [record.prompt for record in records]
            prompt_metadata = [record.prompt_metadata for record in records]
            model_metadata = records[0].model_metadata
            if any(record.model_metadata != model_metadata for record in records):
                errors.append(f"step {step} model metadata are inconsistent")

            cache_paths = {record.rollout_cache_path for record in records}
            cache_seed: int | None = None
            if None not in cache_paths and len(cache_paths) == 1:
                cache_tensor = Path(next(iter(cache_paths)))
                cache_metadata_path = cache_tensor.with_suffix(".json")
                try:
                    trusted_cache = _trusted_file(
                        root,
                        cache_metadata_path,
                        label=f"step {step} rollout cache metadata",
                    )
                except FileNotFoundError:
                    warnings.append(f"step {step} rollout cache was pruned")
                else:
                    cache = load_json(trusted_cache)
                    if cache.get("prompts") != prompts:
                        errors.append(f"step {step} cache prompts disagree with records")
                    if cache.get("metadata") != prompt_metadata:
                        errors.append(
                            f"step {step} cache prompt metadata disagree with records"
                        )
                    if cache.get("model_metadata") != model_metadata:
                        errors.append(
                            f"step {step} cache model metadata disagree with records"
                        )
                    if cache.get("weighted_total") != reward_values:
                        errors.append(f"step {step} cache rewards disagree with records")
                    context = cache.get("context")
                    cache_seed = context.get("seed") if isinstance(context, dict) else None
                    if any(record.seed != cache_seed for record in records):
                        errors.append(f"step {step} cache seed disagrees with records")
            elif None not in cache_paths:
                errors.append(f"step {step} does not have one rollout cache path")

            is_final_cycle_step = step == completed - 1
            checkpoint_paths = {record.checkpoint_path for record in records}
            expected_checkpoint_value = str(checkpoint) if is_final_cycle_step else None
            if checkpoint_paths != {expected_checkpoint_value}:
                errors.append(
                    f"step {step} checkpoint references disagree with commit cycle"
                )
            configured_adapter = (resolved_config.get("model") or {}).get("name")
            recorded_adapter_key = model_metadata.get(
                "adapter_key", model_metadata.get("adapter")
            )
            if configured_adapter != recorded_adapter_key:
                errors.append(f"step {step} configured adapter disagrees with rollout")

            checkpoint_dataset_hash = None
            config_fingerprint = None
            if checkpoint_metadata is not None:
                checkpoint_dataset_hash = (
                    (checkpoint_metadata.get("data_identity") or {}).get("train") or {}
                ).get("content_sha256")
                config_fingerprint = checkpoint_metadata.get("config_fingerprint")
                dataset_hash = (resolved_config.get("dataset") or {}).get(
                    "content_sha256"
                )
                if dataset_hash != checkpoint_dataset_hash:
                    errors.append(f"step {step} dataset content identity disagrees")
            identity_rows.append(
                {
                    "step": step,
                    "sample_ids_sha256": _hash_json(sample_ids),
                    "prompts_sha256": _hash_json(prompts),
                    "prompt_metadata_sha256": _hash_json(prompt_metadata),
                    "seed": cache_seed,
                    "model_metadata_sha256": _hash_json(model_metadata),
                    "config_fingerprint": config_fingerprint,
                    "data_content_sha256": checkpoint_dataset_hash,
                    "checkpoint": checkpoint_name,
                    "checkpoint_retained": checkpoint_exists,
                }
            )

    metric_steps = [int(row["step"]) for row in metrics]
    if metric_steps != sorted(metric_steps) or len(metric_steps) != len(set(metric_steps)):
        errors.append("authoritative metric steps are duplicate or unsorted")
    manifest_payload = SampleManifest(
        run_id=run_id,
        schema_version=SAMPLE_MANIFEST_SCHEMA_VERSION,
        records=all_records,
    ).to_dict()
    return {
        "schema_version": "2",
        "run_dir": str(root),
        "run_id": run_id,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "commit_markers": len(markers),
        "metric_rows": len(metrics),
        "manifest_records": len(all_records),
        "steps": metric_steps,
        "identity_rows": identity_rows,
        "manifest_sha256": _hash_json(manifest_payload),
        "metrics_sha256": _hash_json(metrics),
        "resolved_config_sha256": _hash_json(resolved_config),
    }
