"""Atomic, rank-zero run lifecycle status with marker-aware inspection."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

from visual_rl.artifacts.checkpoint import load_json, save_json
from visual_rl.artifacts.serialization import redact_artifact_config


RUN_STATUS_SCHEMA_VERSION = "1"
_RUN_STATES = {"running", "failed", "completed"}
_GENERIC_FAILURE_MESSAGE = "Run failed; inspect trusted process logs for details."


def _authoritative_step(run_root: Path) -> int:
    from visual_rl.preflight import latest_committed_step

    return int(latest_committed_step(run_root) or 0)


def _safe_status_path(path: str | Path, *, must_exist: bool) -> Path:
    status_path = Path(path)
    if must_exist:
        try:
            metadata = status_path.lstat()
        except OSError as exc:
            raise RuntimeError(f"Cannot read run status: {status_path}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"run_status.json must be a regular file, not a symlink: {status_path}"
            )
    elif status_path.is_symlink():
        raise ValueError(f"Run status target cannot be a symlink: {status_path}")
    return status_path


def _sanitized_failure(exception: BaseException | None) -> dict[str, str]:
    error_type = type(exception).__name__ if exception is not None else "RunError"
    return {
        "type": error_type,
        "message": _GENERIC_FAILURE_MESSAGE,
    }


def write_run_status(
    path: str | Path,
    payload: dict[str, Any],
    *,
    rank: int = 0,
    exception: BaseException | None = None,
) -> bool:
    """Persist one lifecycle state on rank zero and return whether it was written."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("Run status rank must be a non-negative integer")
    if rank != 0:
        return False
    if not isinstance(payload, dict):
        raise TypeError("Run status payload must be a dictionary")
    status_path = _safe_status_path(path, must_exist=False)
    state = payload.get("state")
    if state not in _RUN_STATES:
        raise ValueError(f"Unsupported run status state: {state!r}")

    # Never persist caller-provided exception strings or tracebacks.  Other
    # structured fields still pass through the standard recursive redactor.
    public_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"error", "exception", "traceback"}
    }
    status = redact_artifact_config(public_payload)
    committed_steps = _authoritative_step(status_path.parent)
    declared_steps = status.get("completed_steps", committed_steps)
    if (
        isinstance(declared_steps, bool)
        or not isinstance(declared_steps, int)
        or declared_steps < 0
    ):
        raise ValueError("Run status completed_steps must be a non-negative integer")
    if declared_steps > committed_steps:
        raise ValueError(
            "Run status cannot claim steps beyond the authoritative commit log"
        )
    if state == "completed" and declared_steps != committed_steps:
        raise ValueError(
            "Completed run status must match the authoritative committed step"
        )
    if state == "completed":
        target_steps = status.get("target_steps")
        if (
            committed_steps == 0
            and isinstance(target_steps, int)
            and not isinstance(target_steps, bool)
            and target_steps > 0
        ):
            raise ValueError(
                "Completed run status requires an authoritative commit marker"
            )
        if committed_steps > 0:
            from visual_rl.preflight import resolve_resume_checkpoint

            _checkpoint, resolved_step = resolve_resume_checkpoint(status_path.parent)
            if resolved_step != committed_steps:
                raise ValueError(
                    "Completed run status does not match the authoritative head"
                )
    if state == "running":
        pid = status.get("pid", os.getpid())
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("Running status requires a positive integer pid")
        status["pid"] = pid
    if state == "failed":
        status["error"] = _sanitized_failure(exception)
    status.update(
        {
            "schema_version": RUN_STATUS_SCHEMA_VERSION,
            "rank": 0,
            "state": state,
            "completed_steps": declared_steps,
            "authoritative_completed_steps": committed_steps,
            "valid": state == "completed",
        }
    )
    save_json(status_path, status)
    return True


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inspect_run_status(path: str | Path) -> dict[str, Any]:
    status_path = _safe_status_path(path, must_exist=True)
    try:
        value = load_json(status_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"Cannot read run status: {status_path}") from exc
    if str(value.get("schema_version")) != RUN_STATUS_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported run status schema_version: {value.get('schema_version')}"
        )
    if value.get("rank", 0) != 0:
        raise RuntimeError("Only rank zero may publish run_status.json")
    state = value.get("state")
    if state not in _RUN_STATES:
        raise RuntimeError(f"Unknown run status state: {state!r}")
    if bool(value.get("valid")) != (state == "completed"):
        raise RuntimeError("run status valid flag contradicts state")
    current_authority = _authoritative_step(status_path.parent)
    completed_steps = value.get("completed_steps", current_authority)
    recorded_authority = value.get(
        "authoritative_completed_steps",
        completed_steps,
    )
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
        or isinstance(recorded_authority, bool)
        or not isinstance(recorded_authority, int)
        or recorded_authority < 0
    ):
        raise RuntimeError("run status contains an invalid committed step")
    if recorded_authority > current_authority or completed_steps > current_authority:
        raise RuntimeError("run status claims a nonexistent authoritative commit")

    marker_valid = False
    if state == "completed":
        if completed_steps != current_authority:
            raise RuntimeError(
                "completed run status is stale relative to authoritative commits"
            )
        if current_authority == 0:
            target_steps = value.get("target_steps")
            if (
                isinstance(target_steps, int)
                and not isinstance(target_steps, bool)
                and target_steps > 0
            ):
                raise RuntimeError(
                    "completed run status has no authoritative commit marker"
                )
            marker_valid = True
        else:
            from visual_rl.preflight import resolve_resume_checkpoint

            try:
                _checkpoint, resolved_step = resolve_resume_checkpoint(
                    status_path.parent
                )
            except Exception as exc:
                raise RuntimeError(
                    f"completed run status has an invalid authoritative head: {exc}"
                ) from exc
            if resolved_step != current_authority:
                raise RuntimeError(
                    "completed run status does not have a valid authoritative head"
                )
            marker_valid = True

    observed_state = state
    pid_alive = None
    if state == "running":
        pid = value.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeError("running status is missing a valid pid")
        pid_alive = _pid_alive(pid)
        if not pid_alive:
            observed_state = "stale_running"
    return {
        **value,
        "rank": 0,
        "completed_steps": completed_steps,
        "authoritative_completed_steps": current_authority,
        "observed_state": observed_state,
        "pid_alive": pid_alive,
        "marker_valid": marker_valid,
        "ready_for_aggregation": state == "completed" and marker_valid,
    }


def require_completed_runs(paths: list[str | Path]) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("At least one run status path is required")
    statuses = [inspect_run_status(path) for path in paths]
    rejected = [
        {
            "path": str(path),
            "observed_state": status["observed_state"],
            "valid": status["valid"],
        }
        for path, status in zip(paths, statuses, strict=True)
        if not status["ready_for_aggregation"]
    ]
    if rejected:
        raise RuntimeError(f"Aggregation rejected incomplete runs: {rejected}")
    return statuses
