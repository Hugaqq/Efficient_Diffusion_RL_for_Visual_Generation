"""Numerical parity gates for memory-bounded slot policy updates."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from tests.support.policy_recompute_oracle import compute_full_policy_stats_oracle
from tests.support.policy_update_oracle import (
    LocalTestAccelerator,
    run_monolithic_policy_update_oracle,
)
from tests.test_policy_recompute import _Dynamics, _rollout, _TrainableAdapter
from visual_rl.algorithms.optimization.execution import UpdateExecutionPlan
from visual_rl.algorithms.optimization.kernel import PolicyUpdateKernel
from visual_rl.algorithms.optimization.objective import PolicyLossInputs
from visual_rl.algorithms.optimization.recompute import PolicyRecomputeRequest
from visual_rl.models import BatchRowProjection

_ACTIVE_MASK = torch.tensor(
    (
        (True, False, True),
        (False, True, False),
        (True, True, False),
        (False, False, True),
    ),
    dtype=torch.bool,
)


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, *, lr: float) -> None:
        self.step_calls = 0
        super().__init__(parameters, lr=lr)

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _CountingScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1


@dataclass(frozen=True)
class _SelectableConditioning:
    condition_identity: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return len(self.condition_identity)

    def project_rows(
        self,
        projection: BatchRowProjection,
    ) -> _SelectableConditioning:
        if projection.source_batch_size != self.batch_size:
            raise ValueError("projection source batch differs from conditioning")
        return _SelectableConditioning(
            projection.project_tuple(self.condition_identity)
        )


def _irregular_rollout():
    rollout_adapter = _TrainableAdapter()
    dynamics = _Dynamics()
    execution, latent_spec = _rollout(rollout_adapter, dynamics)
    conditioning = _SelectableConditioning(execution.model_condition_identity)
    return (
        replace(
            execution,
            encoded_conditioning=conditioning,
        ),
        latent_spec,
        dynamics,
    )


def _loss_inputs(*, reference_kl_weight: float) -> PolicyLossInputs:
    return PolicyLossInputs(
        base_advantage=torch.tensor(
            (
                (1.00, -0.25, 0.50),
                (-0.75, 0.80, -0.30),
                (1.20, -0.40, 0.90),
                (-0.20, 0.60, -1.10),
            ),
            dtype=torch.float32,
        ),
        algorithm_weight=torch.tensor(
            (
                (1.00, 1.05, 1.10),
                (1.15, 1.20, 1.25),
                (1.30, 1.35, 1.40),
                (1.45, 1.50, 1.55),
            ),
            dtype=torch.float32,
        ),
        active_mask=_ACTIVE_MASK.clone(),
        clip_range=0.2,
        reference_kl_weight=reference_kl_weight,
    )


def _adapter_at_current_policy() -> _TrainableAdapter:
    adapter = _TrainableAdapter()
    with torch.no_grad():
        adapter.weight.fill_(0.30)
    return adapter


@pytest.mark.parametrize("reference_kl_weight", (0.0, 0.35))
def test_step_slots_matches_monolithic_update_with_irregular_active_mask(
    reference_kl_weight: float,
) -> None:
    rollout, latent_spec, dynamics = _irregular_rollout()
    trajectory = rollout.trajectory
    loss_inputs = _loss_inputs(reference_kl_weight=reference_kl_weight)
    require_reference = reference_kl_weight > 0.0

    monolithic_adapter = _adapter_at_current_policy()
    streaming_adapter = _adapter_at_current_policy()
    monolithic_before = monolithic_adapter.weight.detach().clone()
    streaming_before = streaming_adapter.weight.detach().clone()
    monolithic_request = PolicyRecomputeRequest(
        adapter=monolithic_adapter,
        dynamics=dynamics,
        rollout=rollout,
        latent_spec=latent_spec,
        require_reference_statistics=require_reference,
    )
    streaming_request = PolicyRecomputeRequest(
        adapter=streaming_adapter,
        dynamics=dynamics,
        rollout=rollout,
        latent_spec=latent_spec,
        require_reference_statistics=require_reference,
    )
    monolithic_stats = compute_full_policy_stats_oracle(monolithic_request)
    monolithic_optimizer = _CountingSGD((monolithic_adapter.weight,), lr=0.01)
    streaming_optimizer = _CountingSGD((streaming_adapter.weight,), lr=0.01)
    kernel = PolicyUpdateKernel()

    monolithic = run_monolithic_policy_update_oracle(
        trajectory=trajectory,
        policy_stats=monolithic_stats,
        loss_inputs=loss_inputs,
        optimizer=monolithic_optimizer,
        scaler=None,
        optimizer_step=1,
        kernel=kernel,
    )
    streaming = kernel.step_slots(
        trajectory=trajectory,
        loss_inputs=loss_inputs,
        recompute_request=streaming_request,
        optimizer=streaming_optimizer,
        scaler=None,
        optimizer_step=1,
        accelerator=LocalTestAccelerator(None),
        prepared_root=(streaming_adapter.weight,),
        execution_plan=UpdateExecutionPlan(
            row_microbatch_size=2,
            transition_window_size=1,
        ),
    )

    torch.testing.assert_close(
        streaming_adapter.weight.detach() - streaming_before,
        monolithic_adapter.weight.detach() - monolithic_before,
        rtol=2e-5,
        atol=2e-7,
    )
    assert streaming.gradient_norm_pre_clip == pytest.approx(
        monolithic.gradient_norm_pre_clip,
        rel=2e-5,
        abs=2e-7,
    )
    assert streaming.gradient_norm_post_clip == pytest.approx(
        monolithic.gradient_norm_post_clip,
        rel=2e-5,
        abs=2e-7,
    )
    for field in (
        "loss",
        "policy_loss",
        "reference_kl",
        "approx_kl",
        "clipfrac",
        "logprob_delta_max",
    ):
        assert getattr(streaming, field) == pytest.approx(
            getattr(monolithic, field),
            rel=2e-6,
            abs=2e-7,
        )
    if require_reference:
        assert streaming.reference_kl > 0.0
    else:
        assert streaming.reference_kl == 0.0

    summary = streaming.policy_summary
    assert summary is not None
    assert summary.new_log_probs.shape == trajectory.old_log_probs.shape
    assert summary.new_log_probs.device.type == "cpu"
    assert not summary.new_log_probs.requires_grad
    assert summary.new_log_probs.grad_fn is None
    assert summary.materialized_mask.shape == trajectory.old_log_probs.shape
    assert summary.materialized_mask.device.type == "cpu"
    assert summary.materialized_mask.dtype == torch.bool
    assert not summary.materialized_mask.requires_grad
    active = loss_inputs.active_mask
    assert bool(summary.materialized_mask.logical_and(active).equal(active))
    assert summary.slot_count == 6
    monolithic_ratio = torch.exp(
        monolithic_stats.current_log_probs.detach() - trajectory.old_log_probs
    )
    streaming_ratio = torch.exp(summary.new_log_probs - trajectory.old_log_probs)
    torch.testing.assert_close(
        streaming_ratio.masked_select(active),
        monolithic_ratio.masked_select(active),
        rtol=2e-6,
        atol=2e-7,
    )
    assert monolithic_optimizer.step_calls == streaming_optimizer.step_calls == 1


def test_multiple_slots_commit_optimizer_and_scheduler_exactly_once() -> None:
    rollout, latent_spec, dynamics = _irregular_rollout()
    adapter = _adapter_at_current_policy()
    optimizer = _CountingSGD((adapter.weight,), lr=0.01)
    scheduler = _CountingScheduler()
    commits: list[int] = []

    result = PolicyUpdateKernel().step_slots(
        trajectory=rollout.trajectory,
        loss_inputs=_loss_inputs(reference_kl_weight=0.0),
        recompute_request=PolicyRecomputeRequest(
            adapter=adapter,
            dynamics=dynamics,
            rollout=rollout,
            latent_spec=latent_spec,
        ),
        optimizer=optimizer,
        scaler=None,
        optimizer_step=7,
        accelerator=LocalTestAccelerator(None),
        prepared_root=(adapter.weight,),
        lr_scheduler=scheduler,
        logical_commit=commits.append,
        execution_plan=UpdateExecutionPlan(
            row_microbatch_size=2,
            transition_window_size=1,
        ),
    )

    assert result.policy_summary is not None
    assert result.policy_summary.slot_count == 6
    assert result.transaction.trace.count("objective") == 6
    assert result.transaction.trace.count("backward") == 6
    assert result.transaction.trace.count("optimizer") == 1
    assert result.transaction.trace.count("lr_scheduler") == 1
    assert result.transaction.trace.count("logical_commit") == 1
    assert optimizer.step_calls == 1
    assert scheduler.calls == 1
    assert commits == [8]
