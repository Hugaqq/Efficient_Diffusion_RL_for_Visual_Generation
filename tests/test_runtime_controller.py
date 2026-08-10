"""Lifecycle and rollback tests for the sole v0.8 composition root."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from visual_rl.runtime.controller import (
    ControllerStage,
    ControllerState,
    RestoreBoundOutcome,
    RunController,
)


@dataclass
class _Resource:
    name: str
    events: list[str]
    fail_close: bool = False

    def close(self) -> None:
        self.events.append(f"close:{self.name}")
        if self.fail_close:
            raise RuntimeError(f"close {self.name} failed")


class _Backend:
    def __init__(
        self,
        *,
        fail_at: str | None = None,
        fail_close_at: str | None = None,
        alias_prepared: bool = False,
        alias_prepared_restore: bool = False,
        fresh_run: bool = False,
        completed_result: str | None = None,
        empty_restore_outcome: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.fail_at = fail_at
        self.fail_close_at = fail_close_at
        self.alias_prepared = alias_prepared
        self.alias_prepared_restore = alias_prepared_restore
        self.fresh_run = fresh_run
        self.completed_result = completed_result
        self.empty_restore_outcome = empty_restore_outcome
        self.prepared_restore = None

    def _step(self, name: str) -> None:
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} failed")

    def compile(self, config_path):
        self._step("compile")
        return {"path": config_path}

    def preflight(self, compiled):
        self._step("preflight")
        return {"compiled": compiled}

    def create_runtime(self, preflight):
        self._step("create_runtime")
        return _Resource(
            "runtime",
            self.events,
            fail_close=self.fail_close_at == "runtime",
        )

    def construct_graph(self, preflight, runtime):
        self._step("construct_graph")
        return _Resource(
            "graph",
            self.events,
            fail_close=self.fail_close_at == "graph",
        )

    def prepare(self, graph, runtime):
        self._step("prepare")
        if self.alias_prepared:
            return graph
        return _Resource(
            "prepared",
            self.events,
            fail_close=self.fail_close_at == "prepared",
        )

    def restore_prepared(self, preflight, runtime, graph, prepared):
        self._step("restore_prepared")
        if self.fresh_run:
            self.prepared_restore = None
        elif self.alias_prepared_restore:
            self.prepared_restore = prepared
        else:
            self.prepared_restore = _Resource(
                "prepared_restore",
                self.events,
                fail_close=self.fail_close_at == "prepared_restore",
            )
        return self.prepared_restore

    def transform(self, preflight, runtime, graph, prepared):
        self._step("transform")
        return prepared

    def bind(self, preflight, runtime, graph, prepared, transformed):
        self._step("bind")
        return _Resource(
            "bound",
            self.events,
            fail_close=self.fail_close_at == "bound",
        )

    def restore_bound(
        self,
        preflight,
        runtime,
        graph,
        prepared,
        bound,
        prepared_restore,
    ):
        assert prepared_restore is self.prepared_restore
        self._step("restore_bound")
        if self.completed_result is not None:
            return RestoreBoundOutcome(completed_result=self.completed_result)
        if self.empty_restore_outcome:
            return RestoreBoundOutcome()
        return None

    def prepare_run(self, bound):
        self._step("prepare_run")

    def run(self, bound):
        self._step("run")
        return "result"

    def checkpoint(self, bound, result):
        self._step("checkpoint")


class _FinalizingBackend(_Backend):
    def checkpoint(self, bound, result):
        super().checkpoint(bound, result)
        return f"{result}:checkpointed"


@dataclass(frozen=True)
class _DuckRestoreOutcome:
    completed_result: str


class _DuckOutcomeBackend(_Backend):
    def restore_bound(
        self,
        preflight,
        runtime,
        graph,
        prepared,
        bound,
        prepared_restore,
    ):
        super().restore_bound(
            preflight,
            runtime,
            graph,
            prepared,
            bound,
            prepared_restore,
        )
        return _DuckRestoreOutcome(completed_result="must-not-be-used")


def test_controller_owns_the_only_canonical_runtime_order() -> None:
    backend = _Backend()
    controller = RunController(backend)

    assert controller.run("experiment.yaml") == "result"
    assert backend.events == [
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
        "close:bound",
        "close:prepared_restore",
        "close:prepared",
        "close:graph",
        "close:runtime",
    ]
    assert controller.attempted_stages == tuple(ControllerStage)
    assert controller.completed_stages == tuple(ControllerStage)
    assert controller.state is ControllerState.CLOSED
    with pytest.raises(RuntimeError, match="single-use"):
        controller.run("again.yaml")


def test_checkpoint_stage_may_finalize_the_returned_run_result() -> None:
    backend = _FinalizingBackend()

    assert RunController(backend).run("experiment.yaml") == "result:checkpointed"


@pytest.mark.parametrize(
    ("fail_at", "closed"),
    (
        ("compile", ()),
        ("preflight", ()),
        ("create_runtime", ()),
        ("construct_graph", ("runtime",)),
        ("prepare", ("graph", "runtime")),
        ("restore_prepared", ("prepared", "graph", "runtime")),
        (
            "transform",
            ("prepared_restore", "prepared", "graph", "runtime"),
        ),
        (
            "bind",
            ("prepared_restore", "prepared", "graph", "runtime"),
        ),
        (
            "restore_bound",
            ("bound", "prepared_restore", "prepared", "graph", "runtime"),
        ),
        (
            "prepare_run",
            ("bound", "prepared_restore", "prepared", "graph", "runtime"),
        ),
        (
            "run",
            ("bound", "prepared_restore", "prepared", "graph", "runtime"),
        ),
        (
            "checkpoint",
            ("bound", "prepared_restore", "prepared", "graph", "runtime"),
        ),
    ),
)
def test_every_failure_boundary_rolls_back_in_strict_reverse_order(
    fail_at: str,
    closed: tuple[str, ...],
) -> None:
    backend = _Backend(fail_at=fail_at)
    controller = RunController(backend)

    with pytest.raises(RuntimeError, match=f"{fail_at} failed"):
        controller.run("experiment.yaml")

    close_events = tuple(
        item.removeprefix("close:")
        for item in backend.events
        if item.startswith("close:")
    )
    assert close_events == closed
    assert controller.state is ControllerState.FAILED
    with pytest.raises(RuntimeError, match="single-use"):
        controller.run("again.yaml")


def test_g3_failure_cannot_start_preprocess_or_training() -> None:
    backend = _Backend(fail_at="bind")
    controller = RunController(backend)

    with pytest.raises(RuntimeError, match="bind failed"):
        controller.run("experiment.yaml")

    assert "prepare_run" not in backend.events
    assert "run" not in backend.events
    assert "checkpoint" not in backend.events
    assert controller.attempted_stages[-1] is ControllerStage.BIND


def test_aliasing_lifecycle_values_are_closed_exactly_once() -> None:
    backend = _Backend(alias_prepared=True, alias_prepared_restore=True)
    RunController(backend).run("experiment.yaml")
    assert backend.events.count("close:graph") == 1
    assert "close:prepared" not in backend.events
    assert "close:prepared_restore" not in backend.events


def test_fresh_run_may_return_no_prepared_restore_token() -> None:
    backend = _Backend(fresh_run=True)

    RunController(backend).run("experiment.yaml")

    assert "restore_bound" in backend.events
    assert "close:prepared_restore" not in backend.events


def test_empty_restore_bound_outcome_continues_normal_execution() -> None:
    backend = _Backend(empty_restore_outcome=True)

    assert RunController(backend).run("experiment.yaml") == "result"

    assert "prepare_run" in backend.events
    assert "run" in backend.events
    assert "checkpoint" in backend.events


def test_completed_resume_returns_without_activating_or_executing_run() -> None:
    backend = _Backend(completed_result="restored-result")
    controller = RunController(backend)

    assert controller.run("experiment.yaml") == "restored-result"

    assert controller.attempted_stages[-1] is ControllerStage.RESTORE_BOUND
    assert controller.completed_stages[-1] is ControllerStage.RESTORE_BOUND
    assert ControllerStage.PREPARE_RUN not in controller.attempted_stages
    assert ControllerStage.RUN not in controller.attempted_stages
    assert ControllerStage.CHECKPOINT not in controller.attempted_stages
    assert backend.events == [
        "compile",
        "preflight",
        "create_runtime",
        "construct_graph",
        "prepare",
        "restore_prepared",
        "transform",
        "bind",
        "restore_bound",
        "close:bound",
        "close:prepared_restore",
        "close:prepared",
        "close:graph",
        "close:runtime",
    ]
    assert controller.state is ControllerState.CLOSED


def test_restore_bound_rejects_duck_typed_completion_outcome() -> None:
    backend = _DuckOutcomeBackend()
    controller = RunController(backend)

    with pytest.raises(
        TypeError,
        match="restore_bound must return RestoreBoundOutcome or None",
    ):
        controller.run("experiment.yaml")

    assert controller.attempted_stages[-1] is ControllerStage.RESTORE_BOUND
    assert "prepare_run" not in backend.events
    assert "run" not in backend.events
    assert "checkpoint" not in backend.events


def test_backend_contract_is_checked_before_any_work() -> None:
    with pytest.raises(
        TypeError,
        match=r"restore_prepared.*restore_bound.*checkpoint",
    ):
        RunController(object())


def test_successful_work_with_teardown_failure_is_not_reported_as_success() -> None:
    backend = _Backend(fail_close_at="prepared")
    controller = RunController(backend)

    with pytest.raises(RuntimeError, match="close prepared failed"):
        controller.run("experiment.yaml")

    assert controller.state is ControllerState.FAILED
    assert backend.events[-5:] == [
        "close:bound",
        "close:prepared_restore",
        "close:prepared",
        "close:graph",
        "close:runtime",
    ]
