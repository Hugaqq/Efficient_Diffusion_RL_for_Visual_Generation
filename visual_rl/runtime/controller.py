"""Single composition-root lifecycle for the v0.8 training runtime.

The controller owns ordering and rollback only.  Model math, rollout policy,
reward computation, and optimizer semantics remain behind typed runtime ports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, Protocol, TypeVar, runtime_checkable

__all__ = (
    "ControllerStage",
    "ControllerState",
    "RestoreBoundOutcome",
    "RunController",
    "RuntimeLifecycleBackend",
)


CompiledT = TypeVar("CompiledT")
PreflightT = TypeVar("PreflightT")
RuntimeT = TypeVar("RuntimeT")
GraphT = TypeVar("GraphT")
PreparedT = TypeVar("PreparedT")
PreparedRestoreT = TypeVar("PreparedRestoreT")
TransformedT = TypeVar("TransformedT")
BoundT = TypeVar("BoundT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class RestoreBoundOutcome(Generic[ResultT]):
    """Typed completion signal produced by the full restore gate.

    ``completed_result`` is populated only when the restored checkpoint already
    represents the requested terminal state.  An empty outcome continues into
    normal runtime activation and execution.
    """

    completed_result: ResultT | None = None


class ControllerStage(str, Enum):
    """Canonical acquisition/execution order owned by ``RunController``."""

    COMPILE = "compile"
    PREFLIGHT = "preflight"
    CREATE_RUNTIME = "create_runtime"
    CONSTRUCT_GRAPH = "construct_graph"
    PREPARE = "prepare"
    RESTORE_PREPARED = "restore_prepared"
    TRANSFORM = "transform"
    BIND = "bind"
    RESTORE_BOUND = "restore_bound"
    PREPARE_RUN = "prepare_run"
    RUN = "run"
    CHECKPOINT = "checkpoint"


class ControllerState(str, Enum):
    """One-shot controller state."""

    NEW = "new"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CLOSED = "closed"


@runtime_checkable
class RuntimeLifecycleBackend(
    Protocol[
        CompiledT,
        PreflightT,
        RuntimeT,
        GraphT,
        PreparedT,
        PreparedRestoreT,
        TransformedT,
        BoundT,
        ResultT,
    ]
):
    """Ports implemented by the concrete v0.8 runtime assembler.

    The protocol deliberately mirrors the irreversible lifecycle boundaries.
    It is internal: public callers receive only the thin training entry.
    """

    def compile(self, config_path: Path) -> CompiledT: ...

    def preflight(self, compiled: CompiledT) -> PreflightT: ...

    def create_runtime(self, preflight: PreflightT) -> RuntimeT: ...

    def construct_graph(
        self,
        preflight: PreflightT,
        runtime: RuntimeT,
    ) -> GraphT: ...

    def prepare(
        self,
        graph: GraphT,
        runtime: RuntimeT,
    ) -> PreparedT: ...

    def restore_prepared(
        self,
        preflight: PreflightT,
        runtime: RuntimeT,
        graph: GraphT,
        prepared: PreparedT,
    ) -> PreparedRestoreT: ...

    def transform(
        self,
        preflight: PreflightT,
        runtime: RuntimeT,
        graph: GraphT,
        prepared: PreparedT,
    ) -> TransformedT: ...

    def bind(
        self,
        preflight: PreflightT,
        runtime: RuntimeT,
        graph: GraphT,
        prepared: PreparedT,
        transformed: TransformedT,
    ) -> BoundT: ...

    def restore_bound(
        self,
        preflight: PreflightT,
        runtime: RuntimeT,
        graph: GraphT,
        prepared: PreparedT,
        bound: BoundT,
        prepared_restore: PreparedRestoreT,
    ) -> RestoreBoundOutcome[ResultT] | None: ...

    def prepare_run(self, bound: BoundT) -> None: ...

    def run(self, bound: BoundT) -> ResultT: ...

    def checkpoint(self, bound: BoundT, result: ResultT) -> ResultT | None: ...


class _ResourceLedger:
    """Close acquired resources once, in strict reverse acquisition order."""

    def __init__(self) -> None:
        self._resources: list[tuple[str, object, Callable[[], object]]] = []
        self._seen: set[int] = set()

    def acquire(self, stage: ControllerStage, value: object) -> None:
        if value is None or id(value) in self._seen:
            return
        close = getattr(value, "close", None)
        if not callable(close):
            return
        self._seen.add(id(value))
        self._resources.append((stage.value, value, close))

    def close(self, primary: BaseException | None = None) -> None:
        errors: list[tuple[str, BaseException]] = []
        while self._resources:
            stage, _value, close = self._resources.pop()
            try:
                close()
            except BaseException as exc:  # noqa: BLE001
                errors.append((stage, exc))
        if primary is not None:
            for stage, error in errors:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "runtime rollback close failed after "
                        f"{stage}: {type(error).__name__}: {error}"
                    )
            return
        if errors:
            stage, error = errors[0]
            for later_stage, later_error in errors[1:]:
                if hasattr(error, "add_note"):
                    error.add_note(
                        "additional runtime close failure after "
                        f"{later_stage}: {type(later_error).__name__}: {later_error}"
                    )
            if hasattr(error, "add_note"):
                error.add_note(f"resource was acquired during {stage}")
            raise error


class RunController(
    Generic[
        CompiledT,
        PreflightT,
        RuntimeT,
        GraphT,
        PreparedT,
        PreparedRestoreT,
        TransformedT,
        BoundT,
        ResultT,
    ]
):
    """Execute exactly one training run through the canonical lifecycle."""

    def __init__(
        self,
        backend: RuntimeLifecycleBackend[
            CompiledT,
            PreflightT,
            RuntimeT,
            GraphT,
            PreparedT,
            PreparedRestoreT,
            TransformedT,
            BoundT,
            ResultT,
        ],
    ) -> None:
        _validate_backend(backend)
        self._backend = backend
        self._state = ControllerState.NEW
        self._attempted: list[ControllerStage] = []
        self._completed: list[ControllerStage] = []

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def attempted_stages(self) -> tuple[ControllerStage, ...]:
        return tuple(self._attempted)

    @property
    def completed_stages(self) -> tuple[ControllerStage, ...]:
        return tuple(self._completed)

    def run(self, config_path: str | Path) -> ResultT:
        """Run once and always close acquired runtime resources."""

        if self._state is not ControllerState.NEW:
            raise RuntimeError("RunController is single-use")
        if not isinstance(config_path, (str, Path)) or isinstance(config_path, bool):
            raise TypeError("config_path must be str or Path")
        path = Path(config_path).expanduser().resolve(strict=False)
        self._state = ControllerState.RUNNING
        ledger = _ResourceLedger()
        primary: BaseException | None = None
        try:
            compiled = self._invoke(
                ControllerStage.COMPILE, self._backend.compile, path
            )
            preflight = self._invoke(
                ControllerStage.PREFLIGHT,
                self._backend.preflight,
                compiled,
            )
            runtime = self._invoke(
                ControllerStage.CREATE_RUNTIME,
                self._backend.create_runtime,
                preflight,
            )
            ledger.acquire(ControllerStage.CREATE_RUNTIME, runtime)
            graph = self._invoke(
                ControllerStage.CONSTRUCT_GRAPH,
                self._backend.construct_graph,
                preflight,
                runtime,
            )
            ledger.acquire(ControllerStage.CONSTRUCT_GRAPH, graph)
            prepared = self._invoke(
                ControllerStage.PREPARE,
                self._backend.prepare,
                graph,
                runtime,
            )
            ledger.acquire(ControllerStage.PREPARE, prepared)
            prepared_restore = self._invoke(
                ControllerStage.RESTORE_PREPARED,
                self._backend.restore_prepared,
                preflight,
                runtime,
                graph,
                prepared,
            )
            ledger.acquire(ControllerStage.RESTORE_PREPARED, prepared_restore)
            transformed = self._invoke(
                ControllerStage.TRANSFORM,
                self._backend.transform,
                preflight,
                runtime,
                graph,
                prepared,
            )
            ledger.acquire(ControllerStage.TRANSFORM, transformed)
            bound = self._invoke(
                ControllerStage.BIND,
                self._backend.bind,
                preflight,
                runtime,
                graph,
                prepared,
                transformed,
            )
            ledger.acquire(ControllerStage.BIND, bound)
            restore_outcome = self._invoke(
                ControllerStage.RESTORE_BOUND,
                self._backend.restore_bound,
                preflight,
                runtime,
                graph,
                prepared,
                bound,
                prepared_restore,
            )
            if restore_outcome is not None and not isinstance(
                restore_outcome, RestoreBoundOutcome
            ):
                raise TypeError("restore_bound must return RestoreBoundOutcome or None")
            if (
                restore_outcome is not None
                and restore_outcome.completed_result is not None
            ):
                self._state = ControllerState.SUCCEEDED
                return restore_outcome.completed_result
            self._invoke(
                ControllerStage.PREPARE_RUN,
                self._backend.prepare_run,
                bound,
            )
            result = self._invoke(ControllerStage.RUN, self._backend.run, bound)
            finalized = self._invoke(
                ControllerStage.CHECKPOINT,
                self._backend.checkpoint,
                bound,
                result,
            )
            if finalized is not None:
                result = finalized
            self._state = ControllerState.SUCCEEDED
            return result
        except BaseException as exc:
            primary = exc
            self._state = ControllerState.FAILED
            raise
        finally:
            if primary is not None:
                ledger.close(primary)
            else:
                try:
                    ledger.close()
                except BaseException:
                    self._state = ControllerState.FAILED
                    raise
                self._state = ControllerState.CLOSED

    def _invoke(self, stage: ControllerStage, function: Callable, *args: object):
        self._attempted.append(stage)
        result = function(*args)
        self._completed.append(stage)
        return result


def _validate_backend(backend: object) -> None:
    missing = tuple(
        name
        for name in (
            "compile",
            "preflight",
            "create_runtime",
            "construct_graph",
            "prepare",
            "restore_prepared",
            "transform",
            "bind",
            "restore_bound",
            "prepare_run",
            "run",
            "checkpoint",
        )
        if not callable(getattr(backend, name, None))
    )
    if missing:
        raise TypeError(
            "runtime backend is missing callable lifecycle methods: "
            + ", ".join(missing)
        )
