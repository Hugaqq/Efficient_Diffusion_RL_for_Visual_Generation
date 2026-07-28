"""The final single-process/DDP Strategy surface."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from visual_rl.core.types import (
    FrozenMapping,
    MetricContribution,
    RewardBatch,
    ValidatedRuntimeEnv,
)
from visual_rl.distributed import SingleProcessStrategy, build_strategy


def _strategy() -> SingleProcessStrategy:
    strategy = build_strategy(
        SimpleNamespace(
            mode="single",
            device="cpu",
            timeout_s=5.0,
            max_snapshot_tensor_bytes=None,
        ),
        ValidatedRuntimeEnv(
            mode="single",
            rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            group_rank=None,
            group_world_size=None,
            master_addr=None,
            master_port=None,
            visible_gpu_count=0,
            raw_launch_env=FrozenMapping(),
        ),
    )
    assert isinstance(strategy, SingleProcessStrategy)
    return strategy


def _rewards() -> RewardBatch:
    values = torch.tensor([1.0, 3.0], dtype=torch.float32)
    return RewardBatch(
        sample_id=("sample-0", "sample-1"),
        raw={"score": values},
        weighted={"score": values},
        weighted_total=values,
        valid_mask=torch.ones(2, dtype=torch.bool),
        shared_metadata={"score": {}},
        sample_metadata={"score": ({}, {})},
    )


def test_strategy_exposes_only_the_frozen_training_and_phase_surface() -> None:
    strategy = _strategy()
    try:
        expected = {
            "rank",
            "local_rank",
            "world_size",
            "device",
            "backend",
            "is_main_process",
            "run_phase",
            "dataset_start",
            "prepare",
            "recompute_policy_stats",
            "gradient_sync_context",
            "sum_active_transition_count",
            "reduce_tensor_weighted_mean",
            "reduce_metric_contributions",
            "reduce_reward_metrics",
            "atomic_optimizer_step",
            "gather_object",
            "broadcast_object",
            "failure_gate",
            "close",
        }
        public = {
            name
            for name in dir(strategy)
            if not name.startswith("_")
        }
        assert public == expected
        assert not hasattr(strategy, "synchronize_failure")
        assert not hasattr(strategy, "reduce_weighted_mean")
        assert not hasattr(strategy, "last_atomic_snapshot_metrics")
    finally:
        strategy.close()


def test_single_strategy_reduces_the_three_explicit_contracts() -> None:
    strategy = _strategy()
    try:
        assert strategy.dataset_start(3, 4) == 12
        assert strategy.sum_active_transition_count(7) == 7
        mean = strategy.reduce_tensor_weighted_mean(
            torch.tensor(2.5),
            3,
        )
        torch.testing.assert_close(mean, torch.tensor(2.5))
        metrics = strategy.reduce_metric_contributions(
            {
                "mean": MetricContribution(torch.tensor(9.0), 3),
                "sum": MetricContribution(torch.tensor(4.0), None),
            }
        )
        assert metrics == {"mean": 3.0, "sum": 4.0}
        assert strategy.reduce_reward_metrics(_rewards()) == {
            "reward_mean": 2.0,
            "reward_std": 1.0,
        }
    finally:
        strategy.close()


def test_single_atomic_boundary_validates_identity_then_steps_once() -> None:
    strategy = _strategy()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        parameter.grad = torch.tensor(1.0)
        optimizer.step()

    try:
        strategy.atomic_optimizer_step(
            operation,
            parameters=(parameter,),
            optimizer=optimizer,
            scaler=None,
        )
        assert calls == 1
        assert float(parameter.detach()) < 1.0
        with pytest.raises(ValueError, match="identity/order"):
            strategy.atomic_optimizer_step(
                operation,
                parameters=(torch.nn.Parameter(torch.tensor(2.0)),),
                optimizer=optimizer,
                scaler=None,
            )
        assert calls == 1
    finally:
        strategy.close()


def test_strategy_methods_keep_explicit_keyword_roots() -> None:
    assert tuple(
        inspect.signature(SingleProcessStrategy.gather_object).parameters
    ) == ("self", "value", "dst")
    assert tuple(
        inspect.signature(SingleProcessStrategy.broadcast_object).parameters
    ) == ("self", "value", "src")
    assert tuple(
        inspect.signature(
            SingleProcessStrategy.atomic_optimizer_step
        ).parameters
    ) == ("self", "operation", "parameters", "optimizer", "scaler")
