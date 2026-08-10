"""Exact call-order and commit tests for the v0.8 update transaction."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from visual_rl.algorithms.optimization.execution import (
    PreparedLoss,
    UpdateDisposition,
    UpdateExecutionPlan,
    UpdateTransactionPoisonedError,
)

ROOT = Path(__file__).resolve().parents[1]


def test_update_execution_owner_is_canonical_and_training_namespace_is_absent() -> None:
    assert not (ROOT / "visual_rl" / "training").exists()
    assert UpdateExecutionPlan.__module__ == (
        "visual_rl.algorithms.optimization.execution"
    )


class _TracingSGD(torch.optim.SGD):
    def __init__(self, parameters, *, events):
        self.events = events
        self.step_calls = 0
        self.zero_grad_calls = 0
        self.fail_zero_grad = False
        super().__init__(parameters, lr=0.1)

    def step(self, closure=None):
        self.events.append("optimizer.step")
        self.step_calls += 1
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.events.append("optimizer.zero_grad")
        self.zero_grad_calls += 1
        if self.fail_zero_grad:
            raise RuntimeError("zero_grad failed")
        return super().zero_grad(set_to_none=set_to_none)


class _Accelerator:
    def __init__(self, events, *, sync_gradients=True):
        self.events = events
        self.sync_gradients = sync_gradients
        self.roots = []

    @contextmanager
    def accumulate(self, root):
        self.roots.append(root)
        self.events.append("accumulate.enter")
        try:
            yield
        finally:
            self.events.append("accumulate.exit")

    def backward(self, loss):
        self.events.append("accelerator.backward")
        loss.backward()

    def unscale_gradients(self, optimizer):
        assert optimizer is not None
        self.events.append("accelerator.unscale")


class _StepOwner:
    def __init__(self, events, name, *, fail=False):
        self.events = events
        self.name = name
        self.fail = fail
        self.calls = 0

    def step(self):
        self.events.append(self.name)
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} failed")


class _Scale:
    def __init__(self, events, *, skip):
        self.events = events
        self.skip = skip
        self.scale_value = 16.0
        self.step_calls = 0
        self.update_calls = 0

    def unscale_(self, optimizer):
        assert optimizer is not None
        self.events.append("scaler.unscale")

    def step(self, optimizer):
        self.events.append("scaler.step")
        self.step_calls += 1
        if not self.skip:
            optimizer.step()

    def update(self):
        self.events.append("scaler.update")
        self.update_calls += 1
        if self.skip:
            self.scale_value /= 2.0

    def get_scale(self):
        return self.scale_value


def _closure(parameter, events, *, multiplier=1.0):
    def compute():
        events.append("objective.compute")
        loss = multiplier * (parameter - 2.0).square()
        return PreparedLoss(loss=loss, payload={"loss": float(loss.detach())})

    return compute


def test_plan_is_frozen_and_has_stable_canonical_identity() -> None:
    first = UpdateExecutionPlan(max_grad_norm=0.5)
    second = UpdateExecutionPlan(max_grad_norm=0.5)

    assert first.plan_id == second.plan_id
    assert first.to_payload()["execution_order"] == list(first.EXECUTION_ORDER)
    with pytest.raises(FrozenInstanceError):
        first.max_grad_norm = 1.0


def test_deterministic_execution_is_global_torch_rng_neutral() -> None:
    events = []
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TracingSGD((parameter,), events=events)
    accelerator = _Accelerator(events)
    state_before = torch.random.get_rng_state().clone()

    result = UpdateExecutionPlan().execute(
        loss_closure=_closure(parameter, events),
        accelerator=accelerator,
        prepared_root=object(),
        optimizer=optimizer,
        parameters=(parameter,),
        optimizer_step=0,
    )

    assert result.committed
    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_success_uses_prepared_root_accelerator_backward_and_fixed_order() -> None:
    events = []
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TracingSGD((parameter,), events=events)
    accelerator = _Accelerator(events)
    scheduler = _StepOwner(events, "scheduler.step")
    root = object()
    ema_calls = 0
    reference_calls = 0
    committed_steps = []

    def ema_update():
        nonlocal ema_calls
        events.append("ema.update")
        ema_calls += 1

    def reference_update():
        nonlocal reference_calls
        events.append("reference.update")
        reference_calls += 1

    result = UpdateExecutionPlan(max_grad_norm=0.25).execute(
        loss_closure=_closure(parameter, events),
        accelerator=accelerator,
        prepared_root=root,
        optimizer=optimizer,
        parameters=(parameter,),
        optimizer_step=7,
        lr_scheduler=scheduler,
        ema_update=ema_update,
        reference_update=reference_update,
        logical_commit=committed_steps.append,
    )

    assert result.committed
    assert not result.skipped
    assert result.disposition is UpdateDisposition.COMMITTED
    assert result.next_optimizer_step == 8
    assert result.trace == UpdateExecutionPlan.EXECUTION_ORDER
    assert result.gradient_norm_pre_clip == pytest.approx(4.0)
    assert result.gradient_norm_post_clip == pytest.approx(0.25)
    assert accelerator.roots == [root]
    assert optimizer.step_calls == scheduler.calls == ema_calls == reference_calls == 1
    assert committed_steps == [8]
    assert optimizer.zero_grad_calls == 1
    assert parameter.grad is None
    assert events == [
        "accumulate.enter",
        "objective.compute",
        "accelerator.backward",
        "accumulate.exit",
        "accelerator.unscale",
        "optimizer.step",
        "scheduler.step",
        "ema.update",
        "reference.update",
        "optimizer.zero_grad",
    ]


def test_accumulation_result_retains_gradients_and_never_advances_step() -> None:
    events = []
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TracingSGD((parameter,), events=events)
    accelerator = _Accelerator(events, sync_gradients=False)
    scheduler = _StepOwner(events, "scheduler.step")

    result = UpdateExecutionPlan().execute(
        loss_closure=_closure(parameter, events),
        accelerator=accelerator,
        prepared_root=object(),
        optimizer=optimizer,
        parameters=(parameter,),
        optimizer_step=3,
        lr_scheduler=scheduler,
    )

    assert not result.committed
    assert not result.skipped
    assert result.disposition is UpdateDisposition.ACCUMULATING
    assert result.next_optimizer_step == 3
    assert result.trace == ("accumulate", "objective", "backward")
    assert optimizer.step_calls == scheduler.calls == optimizer.zero_grad_calls == 0
    assert parameter.grad is not None


def test_scaler_overflow_is_terminal_but_does_not_commit_lr_ema_or_step() -> None:
    events = []
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TracingSGD((parameter,), events=events)
    accelerator = _Accelerator(events)
    scaler = _Scale(events, skip=True)
    scheduler = _StepOwner(events, "scheduler.step")
    ema_calls = 0
    committed_steps = []

    def ema_update():
        nonlocal ema_calls
        ema_calls += 1

    result = UpdateExecutionPlan().execute(
        loss_closure=_closure(parameter, events),
        accelerator=accelerator,
        prepared_root=object(),
        optimizer=optimizer,
        parameters=(parameter,),
        optimizer_step=11,
        scaler=scaler,
        lr_scheduler=scheduler,
        ema_update=ema_update,
        logical_commit=committed_steps.append,
    )

    assert not result.committed
    assert result.skipped
    assert result.disposition is UpdateDisposition.SCALER_SKIPPED
    assert result.next_optimizer_step == 11
    assert optimizer.step_calls == scheduler.calls == ema_calls == 0
    assert committed_steps == []
    assert scaler.step_calls == scaler.update_calls == 1
    assert optimizer.zero_grad_calls == 1
    assert parameter.grad is None
    assert "logical_commit" not in result.trace
    assert "lr_scheduler" not in result.trace
    assert "ema" not in result.trace


def test_nonfinite_gradient_never_reaches_optimizer() -> None:
    events = []
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    parameter.register_hook(lambda gradient: torch.full_like(gradient, torch.nan))
    optimizer = _TracingSGD((parameter,), events=events)
    accelerator = _Accelerator(events)

    with pytest.raises(RuntimeError, match="non-finite gradient"):
        UpdateExecutionPlan().execute(
            loss_closure=_closure(parameter, events),
            accelerator=accelerator,
            prepared_root=object(),
            optimizer=optimizer,
            parameters=(parameter,),
            optimizer_step=4,
        )

    assert optimizer.step_calls == 0
    assert optimizer.zero_grad_calls == 1
    assert parameter.grad is None


@pytest.mark.parametrize(
    ("failed_phase", "message"),
    (
        ("lr_scheduler", "scheduler.step failed"),
        ("ema", "ema failed"),
        ("reference", "reference failed"),
        ("zero_grad", "zero_grad failed"),
    ),
)
def test_post_optimizer_failure_is_typed_fatal_and_never_logical_commit(
    failed_phase,
    message,
) -> None:
    events = []
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = _TracingSGD((parameter,), events=events)
    accelerator = _Accelerator(events)
    scheduler = _StepOwner(
        events,
        "scheduler.step",
        fail=failed_phase == "lr_scheduler",
    )
    committed_steps = []

    def ema_update():
        if failed_phase == "ema":
            raise RuntimeError("ema failed")

    def reference_update():
        if failed_phase == "reference":
            raise RuntimeError("reference failed")

    optimizer.fail_zero_grad = failed_phase == "zero_grad"

    with pytest.raises(UpdateTransactionPoisonedError, match=message) as raised:
        UpdateExecutionPlan().execute(
            loss_closure=_closure(parameter, events),
            accelerator=accelerator,
            prepared_root=object(),
            optimizer=optimizer,
            parameters=(parameter,),
            optimizer_step=4,
            lr_scheduler=scheduler,
            ema_update=ema_update,
            reference_update=reference_update,
            logical_commit=committed_steps.append,
        )

    error = raised.value
    assert error.optimizer_step == 4
    assert error.optimizer_step_applied is True
    assert error.failed_phase == failed_phase
    assert error.fatal
    assert not error.retryable
    assert optimizer.step_calls == 1
    assert committed_steps == []
    assert "logical_commit" not in error.trace
    if failed_phase == "zero_grad":
        assert optimizer.zero_grad_calls == 2
        assert error.cleanup_error is not None
        assert parameter.grad is not None
    else:
        assert optimizer.zero_grad_calls == 1
        assert error.cleanup_error is None
        assert parameter.grad is None
