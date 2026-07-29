"""Deep, read-only audit of authoritative v2/v3/v5 run artifacts."""

from __future__ import annotations

from pathlib import Path
import stat
from typing import Any

from visual_rl.artifacts.checkpoint import (
    _audit_checkpoint_artifacts,
    checkpoint_tree_sha256,
    strict_json_loads,
)
from visual_rl.artifacts.manager import read_authoritative_commit_chain
from visual_rl.artifacts.manifest import SampleManifest
from visual_rl.core.types import ValidationCheck
from visual_rl.errors import ArtifactError, ResumeError


def audit_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Validate the commit chain, retained checkpoints and both projections."""

    root = _run_root(run_dir)
    checks: list[ValidationCheck] = []
    checked_paths: list[str] = []
    try:
        chain = read_authoritative_commit_chain(
            root,
            verify_checkpoint_trees=False,
        )
    except ArtifactError as exc:
        return _projection(
            checks=(
                _check(
                    "error",
                    "audit.commit_chain",
                    exc.path or str(root / "commits"),
                    str(exc),
                ),
            ),
        )

    if not chain:
        return _projection(
            checks=(
                _check(
                    "warning",
                    "audit.no_commits",
                    str(root / "commits"),
                    "Run has no authoritative commit marker.",
                ),
            ),
        )

    run_id = str(chain[-1]["run_id"])
    committed_steps = int(chain[-1]["completed_steps"])
    for index, marker in enumerate(chain):
        completed = int(marker["completed_steps"])
        marker_relative = f"commits/commit_{completed:06d}.json"
        checked_paths.append(marker_relative)
        checkpoint_relative = str(marker["checkpoint"]["path"])
        checkpoint = root / checkpoint_relative
        if not checkpoint.exists():
            if index == len(chain) - 1:
                checks.append(
                    _check(
                        "error",
                        "audit.head_checkpoint_missing",
                        str(checkpoint),
                        "Newest authoritative checkpoint is missing.",
                    )
                )
            else:
                checks.append(
                    _check(
                        "warning",
                        "audit.pruned_checkpoint",
                        str(checkpoint),
                        "Historical checkpoint was removed by retention.",
                    )
                )
            continue
        try:
            actual_digest = checkpoint_tree_sha256(
                checkpoint,
                trusted_root=root,
            )
            if actual_digest != marker["checkpoint"]["tree_sha256"]:
                raise ArtifactError(
                    "authoritative checkpoint tree SHA256 mismatch",
                    path=str(checkpoint),
                )
            _audit_checkpoint_artifacts(
                checkpoint,
                expected_global_step=completed,
            )
        except (ArtifactError, ResumeError, OSError, RuntimeError, ValueError) as exc:
            checks.append(
                _check(
                    "error",
                    "audit.checkpoint",
                    getattr(exc, "path", None) or str(checkpoint),
                    str(exc),
                )
            )
            continue
        checked_paths.append(checkpoint_relative)

    expected_records = [
        row
        for marker in chain
        for step in marker["steps"]
        for row in step["manifest_records"]
    ]
    expected_metrics = [
        step["core_metric_row"]
        for marker in chain
        for step in marker["steps"]
    ]
    _audit_manifest_projection(
        root,
        run_id=run_id,
        expected_records=expected_records,
        checked_paths=checked_paths,
        checks=checks,
    )
    _audit_metrics_projection(
        root,
        expected_metrics=expected_metrics,
        checked_paths=checked_paths,
        checks=checks,
    )
    _audit_record_references(
        root,
        records=expected_records,
        checked_paths=checked_paths,
        checks=checks,
    )
    return {
        "run_id": run_id,
        "committed_steps": committed_steps,
        "checked_commit_count": len(chain),
        "checked_artifact_paths": tuple(checked_paths),
        "checks": tuple(checks),
    }


def _audit_manifest_projection(
    root: Path,
    *,
    run_id: str,
    expected_records: list[dict[str, Any]],
    checked_paths: list[str],
    checks: list[ValidationCheck],
) -> None:
    path = root / "sample_manifest.json"
    try:
        _require_regular_child(root, path, label="sample manifest")
        manifest = SampleManifest.load(path)
        if manifest.run_id != run_id:
            raise ValueError("sample manifest run_id disagrees with commit chain")
        if manifest.to_dict()["records"] != expected_records:
            raise ValueError("sample manifest is not the exact chain projection")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        checks.append(
            _check(
                "error",
                "audit.manifest_projection",
                str(path),
                str(exc),
            )
        )
        return
    checked_paths.append("sample_manifest.json")


def _audit_metrics_projection(
    root: Path,
    *,
    expected_metrics: list[dict[str, Any]],
    checked_paths: list[str],
    checks: list[ValidationCheck],
) -> None:
    path = root / "metrics.jsonl"
    try:
        _require_regular_child(root, path, label="metrics projection")
        lines = path.read_text(encoding="utf-8").splitlines()
        metrics = [strict_json_loads(line) for line in lines]
        if metrics != expected_metrics:
            raise ValueError("metrics.jsonl is not the exact chain projection")
    except (OSError, UnicodeError, RuntimeError, TypeError, ValueError) as exc:
        checks.append(
            _check(
                "error",
                "audit.metrics_projection",
                str(path),
                str(exc),
            )
        )
        return
    checked_paths.append("metrics.jsonl")


def _audit_record_references(
    root: Path,
    *,
    records: list[dict[str, Any]],
    checked_paths: list[str],
    checks: list[ValidationCheck],
) -> None:
    seen = set(checked_paths)
    for record in records:
        for field in ("media_path", "rollout_cache_path"):
            relative = record[field]
            if relative is None or relative in seen:
                continue
            path = root / relative
            try:
                _require_existing_child(root, path, label=field)
            except (OSError, RuntimeError, ValueError) as exc:
                checks.append(
                    _check(
                        "error",
                        f"audit.{field}",
                        str(path),
                        str(exc),
                    )
                )
                continue
            checked_paths.append(relative)
            seen.add(relative)


def _require_regular_child(root: Path, path: Path, *, label: str) -> None:
    _require_existing_child(root, path, label=label)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError(f"{label} must be a regular file")


def _require_existing_child(root: Path, path: Path, *, label: str) -> None:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the run directory") from exc
    current = root
    relative = path.absolute().relative_to(root.absolute())
    for part in relative.parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} path contains a symlink")
    if not path.resolve(strict=True).is_relative_to(root):
        raise RuntimeError(f"{label} escapes the run directory")


def _projection(
    *,
    checks: tuple[ValidationCheck, ...],
) -> dict[str, Any]:
    return {
        "run_id": None,
        "committed_steps": 0,
        "checked_commit_count": 0,
        "checked_artifact_paths": (),
        "checks": checks,
    }


def _run_root(value: str | Path) -> Path:
    requested = Path(value).absolute()
    try:
        metadata = requested.lstat()
    except OSError as exc:
        raise ArtifactError(
            "run directory does not exist",
            path=str(requested),
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactError(
            "run directory must be a real directory",
            path=str(requested),
        )
    return requested.resolve(strict=True)


def _check(
    level: str,
    code: str,
    path: str,
    message: str,
) -> ValidationCheck:
    return ValidationCheck(
        level=level,
        code=code,
        path=path,
        message=message,
        volatile=False,
    )


__all__ = ("audit_run_artifacts",)
