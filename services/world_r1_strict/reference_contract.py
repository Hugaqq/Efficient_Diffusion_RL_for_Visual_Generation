"""Deployment-time contract for the patched native World-R1 reward managers.

The companion service only accepts the fail-closed native managers produced by
``reference_patches/world_r1_fail_closed_v1.patch``.  This module is the unique
owner of

- :func:`require_strict_manager` — class/instance marker and surface checks;
- :func:`require_service_runtime` — the local-import Torch/CUDA/Qwen gate that
  must pass before any manager class is imported or constructed;
- :func:`run_native_fault_injection_gate` — the no-checkpoint, no-CUDA-memory
  fault-injection gate executed against the real imported manager class;
- the process-local lifecycle registry used by ``gunicorn_conf.worker_exit``
  and ``atexit`` to guarantee call-once manager shutdown.

No Torch, World-R1 or Flask imports exist at module top level.
"""

from __future__ import annotations

import atexit
import importlib
import io
import multiprocessing
import threading
from typing import Any

from visual_rl.world_r1_protocol import (
    MANAGER_CONTRACT,
    REWARD_3D,
    REWARD_GENERAL,
    validate_reward_kind,
)

_REQUIRED_METHODS = ("is_ready", "compute_batch_scores", "shutdown")


class ServiceRuntimeError(RuntimeError):
    """The deployment host does not provide the frozen service runtime."""


class ManagerContractError(RuntimeError):
    """A manager does not implement the world_r1_fail_closed_v1 contract."""


class NativeFaultGateError(RuntimeError):
    """The real manager class misbehaved under fault injection."""


def require_strict_manager(manager: Any, *, reward: str) -> Any:
    """Validate the fail-closed manager markers and surface, or refuse startup."""

    reward = validate_reward_kind(reward)
    protocol = getattr(manager, "STRICT_MANAGER_PROTOCOL", None)
    if protocol != MANAGER_CONTRACT:
        raise ManagerContractError(
            f"{reward} manager must declare STRICT_MANAGER_PROTOCOL="
            f"{MANAGER_CONTRACT!r} (apply reference_patches/"
            f"world_r1_fail_closed_v1.patch), got {protocol!r}."
        )
    kind = getattr(manager, "STRICT_REWARD_KIND", None)
    if kind != reward:
        raise ManagerContractError(
            f"manager STRICT_REWARD_KIND must be {reward!r}, got {kind!r}."
        )
    for method in _REQUIRED_METHODS:
        if not callable(getattr(manager, method, None)):
            raise ManagerContractError(
                f"strict {reward} manager is missing required method {method!r}."
            )
    return manager


def require_service_runtime() -> None:
    """Check the frozen service runtime before any manager import/construction.

    Only local imports and version/CUDA capability checks happen here; no
    checkpoint is loaded.  requirements-service.txt remains the single owner of
    package versions — this gate verifies, it does not re-declare them.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ServiceRuntimeError(
            "strict World-R1 service requires torch==2.7.1+cu128."
        ) from exc
    if torch.__version__ != "2.7.1+cu128":
        raise ServiceRuntimeError(
            f"strict World-R1 service requires torch==2.7.1+cu128, got {torch.__version__}."
        )
    try:
        import torchvision
    except ModuleNotFoundError as exc:
        raise ServiceRuntimeError(
            "strict World-R1 service requires torchvision==0.22.1+cu128."
        ) from exc
    if torchvision.__version__ != "0.22.1+cu128":
        raise ServiceRuntimeError(
            "strict World-R1 service requires torchvision==0.22.1+cu128, "
            f"got {torchvision.__version__}."
        )
    if torch.version.cuda != "12.8":
        raise ServiceRuntimeError(
            f"strict World-R1 service requires a CUDA 12.8 Torch build, got {torch.version.cuda}."
        )
    if not torch.cuda.is_available():
        raise ServiceRuntimeError("strict World-R1 service requires an available CUDA device.")
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: F401
    except (ModuleNotFoundError, ImportError) as exc:
        raise ServiceRuntimeError(
            "strict World-R1 service requires transformers with "
            "Qwen3VLForConditionalGeneration/AutoProcessor."
        ) from exc
    for module_name in ("qwen_vl_utils", "lpips", "gsplat", "pycolmap", "hpsv2"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ServiceRuntimeError(
                f"strict World-R1 service requires the {module_name!r} package."
            ) from exc


# ---------------------------------------------------------------------------
# Native fault-injection gate (deployment only; never runs in ordinary tests)
# ---------------------------------------------------------------------------


class _GateAliveWorker:
    """Controlled worker stand-in that looks alive until terminated."""

    def __init__(self) -> None:
        self._alive = True
        self.pid = None

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False

    def kill(self) -> None:
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self._alive = False

    def close(self) -> None:
        self._alive = False


class _GateDeadWorker(_GateAliveWorker):
    """Controlled worker stand-in that is already dead."""

    def __init__(self) -> None:
        super().__init__()
        self._alive = False


class _GateRaisingScorer:
    """Scorer double injected through the patched worker's factory seam."""

    def __init__(self, logical_index: int) -> None:
        self.logical_index = logical_index

    def load_model(self) -> None:
        return None

    def compute_score(self, images: Any, prompts: Any) -> list[float]:
        del images, prompts
        raise RuntimeError("strict gate injected scorer failure")


def _gate_raising_factory(logical_index: int) -> _GateRaisingScorer:
    return _GateRaisingScorer(logical_index)


def _gate_instance(manager_class: type, *, reward: str) -> Any:
    instance = manager_class.__new__(manager_class)
    if reward == REWARD_3D:
        manager_class.__init__(instance, scorer_type="qwen", use_lpips=True)
    else:
        manager_class.__init__(instance)
    instance._score_timeout_s = 2.0  # controlled dependency: bounded gate deadline
    return instance


def _gate_jpeg() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (128, 64, 32)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _gate_inputs(reward: str, *, corrupt: bool) -> tuple[Any, Any, dict[str, Any]]:
    blob = b"strict-gate-corrupt-bytes" if corrupt else _gate_jpeg()
    if reward == REWARD_GENERAL:
        return [blob], ["a strict gate prompt"], {}
    identity = [[[1.0, 0.0, 0.0, 0.5]] + [[0.0, 1.0, 0.0, -0.25]] + [[0.0, 0.0, 1.0, 2.0]] + [[0.0, 0.0, 0.0, 1.0]]]
    return [[blob]], ["a strict gate prompt"], {"camera_trajectories": identity}


def _manager_error_base(manager_class: type) -> type[BaseException]:
    module = importlib.import_module(manager_class.__module__)
    base = getattr(module, "StrictManagerError", None)
    if not isinstance(base, type) or not issubclass(base, RuntimeError):
        raise NativeFaultGateError(
            f"patched manager module {module.__name__} must define StrictManagerError(RuntimeError)."
        )
    return base


def _ready_with_workers(instance: Any, workers: list[Any]) -> None:
    context = instance._mp_context
    instance.num_gpus = len(workers)
    instance._task_queues = [context.Queue() for _ in workers]
    instance._result_queue = context.Queue()
    instance._workers = list(workers)
    instance._ready = True


def _expect_batch_failure(instance: Any, manager_class: type, *, reward: str, corrupt: bool) -> BaseException:
    base = _manager_error_base(manager_class)
    items, prompts, kwargs = _gate_inputs(reward, corrupt=corrupt)
    try:
        if reward == REWARD_GENERAL:
            instance.compute_batch_scores(items, prompts)
        else:
            instance.compute_batch_scores(items, prompts, **kwargs)
    except base as exc:
        return exc
    raise NativeFaultGateError(
        f"strict gate case returned finite default scores instead of raising {base.__name__}."
    )


def _assert_poisoned(instance: Any, *, case: str) -> None:
    if instance.is_ready():
        raise NativeFaultGateError(f"strict gate case {case!r} left the manager ready.")
    leftover = [w for w in getattr(instance, "_workers", []) if w.is_alive()]
    if getattr(instance, "_workers", []) or leftover:
        raise NativeFaultGateError(f"strict gate case {case!r} left residual workers.")


def _run_gate_case(manager_class: type, *, reward: str, case: str) -> None:
    instance = _gate_instance(manager_class, reward=reward)
    if case == "uninitialized":
        _expect_batch_failure(instance, manager_class, reward=reward, corrupt=False)
        _assert_poisoned(instance, case=case)
        return
    if case == "init_error":
        def failing_start(logical_index, task_queue, result_queue, init_queue, **kwargs):
            init_queue.put(("INIT_ERROR", logical_index, "STRICT_GATE_NO_CUDA"))
            return _GateAliveWorker()

        instance._start_worker = failing_start
        base = _manager_error_base(manager_class)
        try:
            instance.initialize()
        except base:
            pass
        else:
            raise NativeFaultGateError(
                "strict gate init case did not raise on INIT_ERROR."
            ) from None
        _assert_poisoned(instance, case=case)
        return
    if case in {"bad_jpeg", "bad_frame_jpeg"}:
        _ready_with_workers(instance, [_GateAliveWorker()])
        _expect_batch_failure(instance, manager_class, reward=reward, corrupt=True)
        _assert_poisoned(instance, case=case)
        return
    if case == "scorer_raise":
        worker = instance._start_worker(
            0,
            instance._mp_context.Queue(),
            instance._mp_context.Queue(),
            instance._mp_context.Queue(),
            instance_factory=_gate_raising_factory,
        )
        instance._score_timeout_s = 30.0
        _ready_with_workers(instance, [worker])
        _expect_batch_failure(instance, manager_class, reward=reward, corrupt=False)
        _assert_poisoned(instance, case=case)
        if worker.is_alive():
            raise NativeFaultGateError("strict gate scorer case left a live worker process.")
        return
    if case == "row_error":
        _ready_with_workers(instance, [_GateAliveWorker()])
        instance._result_queue.put(("ROW_ERROR", 0, "SCORER_EXCEPTION"))
        _expect_batch_failure(instance, manager_class, reward=reward, corrupt=False)
        _assert_poisoned(instance, case=case)
        return
    if case == "worker_death":
        _ready_with_workers(instance, [_GateDeadWorker()])
        _expect_batch_failure(instance, manager_class, reward=reward, corrupt=False)
        _assert_poisoned(instance, case=case)
        return
    if case == "result_timeout":
        instance._score_timeout_s = 0.3
        _ready_with_workers(instance, [_GateAliveWorker()])
        _expect_batch_failure(instance, manager_class, reward=reward, corrupt=False)
        _assert_poisoned(instance, case=case)
        return
    raise NativeFaultGateError(f"unknown strict gate case {case!r}.")


def run_native_fault_injection_gate(manager_class: type, *, reward: str) -> None:
    """Prove the real imported manager class fails closed, then return.

    Every case runs on ``__new__``/controlled-dependency instances with test
    queues only: no checkpoint, no CUDA memory, no network.  Any case that
    returns finite default scores, stays ready or leaks workers refuses WSGI
    app startup by raising NativeFaultGateError.
    """

    reward = validate_reward_kind(reward)
    require_strict_manager(manager_class, reward=reward)
    if reward == REWARD_GENERAL:
        cases = (
            "uninitialized",
            "bad_jpeg",
            "scorer_raise",
            "row_error",
            "worker_death",
            "result_timeout",
        )
    else:
        cases = (
            "init_error",
            "bad_frame_jpeg",
            "row_error",
            "worker_death",
            "result_timeout",
        )
    for case in cases:
        _run_gate_case(manager_class, reward=reward, case=case)


# ---------------------------------------------------------------------------
# Process-local lifecycle registry (call-once manager shutdown)
# ---------------------------------------------------------------------------


class _RegistryEntry:
    __slots__ = ("manager", "close", "closed")

    def __init__(self, manager: Any, close: Any) -> None:
        self.manager = manager
        self.close = close
        self.closed = False


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: dict[int, list[_RegistryEntry]] = {}
_ATEXIT_REGISTERED = False


def register_manager(*, manager: Any, pid: int, close: Any = None) -> None:
    """Register ``(pid, call-once close closure)`` and arm the atexit hook."""

    global _ATEXIT_REGISTERED
    close_fn = close if close is not None else getattr(manager, "shutdown", None)
    if not callable(close_fn):
        raise ManagerContractError("registered managers must provide a shutdown() method.")
    with _REGISTRY_LOCK:
        _REGISTRY.setdefault(int(pid), []).append(_RegistryEntry(manager, close_fn))
        if not _ATEXIT_REGISTERED:
            atexit.register(_close_at_exit)
            _ATEXIT_REGISTERED = True


def close_registered_manager(*, expected_pid: int) -> int:
    """Close every manager registered for ``expected_pid`` at most once.

    Wrong PID, never-registered and already-closed entries are idempotent
    no-ops and return 0.
    """

    with _REGISTRY_LOCK:
        entries = _REGISTRY.pop(int(expected_pid), [])
    closed = 0
    for entry in entries:
        if entry.closed:
            continue
        entry.closed = True
        try:
            entry.close()
        except Exception:  # noqa: BLE001 - shutdown hooks must not mask process exit
            pass
        closed += 1
    return closed


def _close_at_exit() -> None:
    close_registered_manager(expected_pid=multiprocessing.current_process().pid)


def registered_manager_count(*, pid: int) -> int:
    """Number of live registry entries for ``pid`` (test/introspection use)."""

    with _REGISTRY_LOCK:
        return len(_REGISTRY.get(int(pid), []))


__all__ = (
    "ManagerContractError",
    "NativeFaultGateError",
    "ServiceRuntimeError",
    "close_registered_manager",
    "register_manager",
    "registered_manager_count",
    "require_service_runtime",
    "require_strict_manager",
    "run_native_fault_injection_gate",
)
