"""Algorithm-owned optimizer construction without model/runtime dependencies."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from visual_rl.algorithms.optimization.execution import (
    OptimizerExecutionSpec,
    build_adamw,
    build_lr_scheduler,
)


def _spec(**changes: object) -> OptimizerExecutionSpec:
    values = {
        "learning_rate": 2e-4,
        "beta1": 0.8,
        "beta2": 0.95,
        "epsilon": 1e-7,
        "weight_decay": 0.02,
        "amsgrad": True,
        "schedule_kind": "constant",
        "warmup_steps": 0,
        "min_lr_ratio": 0.1,
        "max_optimizer_steps": 10,
    }
    values.update(changes)
    return OptimizerExecutionSpec(**values)  # type: ignore[arg-type]


def test_adamw_factory_consumes_only_parameters_and_algorithm_spec() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = build_adamw((parameter,), _spec())

    group = optimizer.param_groups[0]
    assert group["params"] == [parameter]
    assert group["lr"] == pytest.approx(2e-4)
    assert group["betas"] == pytest.approx((0.8, 0.95))
    assert group["eps"] == pytest.approx(1e-7)
    assert group["weight_decay"] == pytest.approx(0.02)
    assert group["amsgrad"] is True


@pytest.mark.parametrize(
    ("kind", "step", "expected"),
    (
        ("constant", 7, 1.0),
        ("linear", 5, 0.55),
        ("cosine", 5, 0.55),
    ),
)
def test_lr_scheduler_formula_is_algorithm_owned(
    kind: str,
    step: int,
    expected: float,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    spec = _spec(schedule_kind=kind)
    scheduler = build_lr_scheduler(build_adamw((parameter,), spec), spec)

    assert scheduler.lr_lambdas[0](step) == pytest.approx(expected)


def test_lr_scheduler_warmup_is_exact() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    spec = replace(_spec(), warmup_steps=2)
    scheduler = build_lr_scheduler(build_adamw((parameter,), spec), spec)

    assert scheduler.lr_lambdas[0](0) == pytest.approx(0.5)
    assert scheduler.lr_lambdas[0](1) == pytest.approx(1.0)
    assert scheduler.lr_lambdas[0](2) == pytest.approx(1.0)
