"""Service-owned liveness checks for strict native reward managers.

The native reward sources are revision-pinned.  HTTP readiness therefore
adds process supervision here, without modifying their scoring or lifecycle
implementations.  Injected strict-manager doubles without the native worker
surface retain their own ``is_ready()`` semantics.
"""

from __future__ import annotations

from typing import Any

_NATIVE_WORKER_ATTRIBUTES = ("_closed", "_ready", "_workers", "num_gpus")


def _manager_ready(manager: Any) -> tuple[bool, str | None]:
    try:
        return bool(manager.is_ready()), None
    except Exception as exc:  # noqa: BLE001 - readiness must fail closed
        return False, type(exc).__name__


def _worker_status(worker: Any, *, slot: int) -> dict[str, Any]:
    errors = []
    try:
        alive = bool(worker.is_alive())
    except Exception as exc:  # noqa: BLE001 - process inspection must fail closed
        alive = False
        errors.append(f"is_alive:{type(exc).__name__}")
    try:
        exitcode = getattr(worker, "exitcode", None)
    except Exception as exc:  # noqa: BLE001 - process inspection must fail closed
        exitcode = None
        errors.append(f"exitcode:{type(exc).__name__}")
    try:
        pid = getattr(worker, "pid", None)
    except Exception as exc:  # noqa: BLE001 - process inspection must fail closed
        pid = None
        errors.append(f"pid:{type(exc).__name__}")

    healthy = alive and exitcode is None and not errors
    if errors:
        reason = "worker_state_unavailable"
    elif exitcode is not None:
        reason = "worker_exited"
    elif not alive:
        reason = "worker_not_alive"
    else:
        reason = None
    return {
        "slot": slot,
        "pid": pid,
        "alive": alive,
        "exitcode": exitcode,
        "healthy": healthy,
        "reason": reason,
        "errors": tuple(errors),
    }


def readiness_status(manager: Any) -> dict[str, Any]:
    """Return a structured, fail-closed manager/worker readiness snapshot."""

    manager_ready, manager_error = _manager_ready(manager)
    has_worker_surface = all(
        hasattr(manager, attribute) for attribute in _NATIVE_WORKER_ATTRIBUTES
    )
    if not has_worker_surface:
        return {
            "ready": manager_ready,
            "reason": None if manager_ready else "manager_not_ready",
            "manager_error": manager_error,
            "worker_supervision": False,
            "expected_workers": None,
            "actual_workers": None,
            "workers": (),
        }

    try:
        workers = tuple(manager._workers)
    except (AttributeError, TypeError):
        workers = ()
        worker_surface_error = "workers_unavailable"
    else:
        worker_surface_error = None
    expected_workers = manager.num_gpus
    worker_status = tuple(
        _worker_status(worker, slot=slot)
        for slot, worker in enumerate(workers)
    )
    worker_count_matches = (
        type(expected_workers) is int
        and expected_workers > 0
        and len(workers) == expected_workers
    )
    workers_healthy = (
        worker_surface_error is None
        and worker_count_matches
        and all(status["healthy"] for status in worker_status)
    )
    ready = manager_ready and workers_healthy
    if manager_error is not None:
        reason = "manager_state_unavailable"
    elif not manager_ready:
        reason = "manager_not_ready"
    elif worker_surface_error is not None:
        reason = worker_surface_error
    elif not worker_count_matches:
        reason = "worker_count_mismatch"
    elif not workers_healthy:
        reason = "worker_unhealthy"
    else:
        reason = None
    return {
        "ready": ready,
        "reason": reason,
        "manager_error": manager_error,
        "manager_ready_flag": manager._ready,
        "manager_closed": manager._closed,
        "worker_supervision": True,
        "expected_workers": expected_workers,
        "actual_workers": len(workers),
        "workers": worker_status,
    }


def is_ready(manager: Any) -> bool:
    """Return whether both the manager and every native worker are ready."""

    return bool(readiness_status(manager)["ready"])


__all__ = ("is_ready", "readiness_status")
