"""Fast, read-only status derived from the authoritative commit chain."""

from __future__ import annotations

from pathlib import Path
import re
import stat
from typing import Any

from visual_rl.artifacts.checkpoint import checkpoint_tree_sha256
from visual_rl.artifacts.manager import read_authoritative_commit_chain
from visual_rl.core.types import ValidationCheck
from visual_rl.errors import ArtifactError


_TRANSACTION_DIRECTORY = re.compile(r"txn_[0-9a-f]{32}")


def inspect_run_status(run_dir: str | Path) -> dict[str, Any]:
    """Project resumability without consulting a competing lifecycle file."""

    root = _run_root(run_dir)
    checks: list[ValidationCheck] = []
    pending_count = _pending_transaction_count(root, checks=checks)
    try:
        chain = read_authoritative_commit_chain(
            root,
            verify_checkpoint_trees=False,
        )
    except ArtifactError as exc:
        checks.append(
            _check(
                "error",
                "status.commit_chain",
                exc.path or str(root / "commits"),
                str(exc),
            )
        )
        return _projection(
            run_id=None,
            committed_steps=0,
            checkpoint=None,
            resumable=False,
            pending_count=pending_count,
            checks=checks,
        )

    if not chain:
        checks.append(
            _check(
                "warning",
                "status.no_commits",
                str(root / "commits"),
                "Run has no authoritative commit marker.",
            )
        )
        return _projection(
            run_id=None,
            committed_steps=0,
            checkpoint=None,
            resumable=False,
            pending_count=pending_count,
            checks=checks,
        )

    head = chain[-1]
    checkpoint_relative = str(head["checkpoint"]["path"])
    checkpoint = root / checkpoint_relative
    resumable = False
    try:
        if not checkpoint.exists():
            raise ArtifactError(
                "latest authoritative checkpoint is missing",
                path=str(checkpoint),
            )
        actual_digest = checkpoint_tree_sha256(
            checkpoint,
            trusted_root=root,
        )
        if actual_digest != head["checkpoint"]["tree_sha256"]:
            raise ArtifactError(
                "latest authoritative checkpoint tree SHA256 mismatch",
                path=str(checkpoint),
            )
        resumable = True
    except (ArtifactError, OSError, RuntimeError, ValueError) as exc:
        checks.append(
            _check(
                "error",
                "status.checkpoint",
                getattr(exc, "path", None) or str(checkpoint),
                str(exc),
            )
        )

    return _projection(
        run_id=str(head["run_id"]),
        committed_steps=int(head["completed_steps"]),
        checkpoint=checkpoint_relative,
        resumable=resumable,
        pending_count=pending_count,
        checks=checks,
    )


def _projection(
    *,
    run_id: str | None,
    committed_steps: int,
    checkpoint: str | None,
    resumable: bool,
    pending_count: int,
    checks: list[ValidationCheck],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "committed_steps": committed_steps,
        "authoritative_checkpoint": checkpoint,
        "resumable": resumable,
        "pending_transaction_count": pending_count,
        "checks": tuple(checks),
    }


def _pending_transaction_count(
    root: Path,
    *,
    checks: list[ValidationCheck],
) -> int:
    staging = root / ".staging"
    if not staging.exists() and not staging.is_symlink():
        return 0
    metadata = staging.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        checks.append(
            _check(
                "error",
                "status.staging",
                str(staging),
                "The staging path must be a real directory.",
            )
        )
        return 0
    count = 0
    for path in sorted(staging.iterdir()):
        if not _TRANSACTION_DIRECTORY.fullmatch(path.name):
            continue
        item = path.lstat()
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
            checks.append(
                _check(
                    "error",
                    "status.pending_transaction",
                    str(path),
                    "A pending transaction path is not a real directory.",
                )
            )
            continue
        count += 1
    if count:
        checks.append(
            _check(
                "warning",
                "status.pending_transactions",
                str(staging),
                f"Run has {count} unfinished transaction director"
                f"{'y' if count == 1 else 'ies'}.",
            )
        )
    return count


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


__all__ = ("inspect_run_status",)
