"""Test-only monolithic policy-update oracle.

Production deliberately exposes only slot streaming.  Numerical parity tests
may use this helper to materialize a complete ``[B,T]`` policy graph, but no
runtime module may import it.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from visual_rl.algorithms.optimization.execution import (
    PreparedLoss,
    UpdateExecutionPlan,
    UpdateNotCommittedError,
)
from visual_rl.algorithms.optimization.kernel import (
    PolicyUpdateKernel,
    PolicyUpdateResult,
)
from visual_rl.algorithms.optimization.objective import (
    ClippedSurrogateObjective,
    LossOutput,
    PolicyLossInputs,
)
from visual_rl.algorithms.optimization.recompute import PolicyStats
from visual_rl.data.samples.trajectory import TrajectoryBatch


@dataclass(slots=True)
class LocalTestAccelerator:
    """Minimal explicit backward backend for CPU-only unit tests."""

    scaler: object | None
    sync_gradients: bool = True

    def accumulate(self, prepared_root: object) -> object:
        if prepared_root is None:
            raise TypeError("prepared_root must not be None")
        return nullcontext()

    def backward(self, loss: Any) -> None:
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()


def run_monolithic_policy_update_oracle(
    *,
    trajectory: TrajectoryBatch,
    policy_stats: PolicyStats,
    loss_inputs: PolicyLossInputs,
    optimizer: Any,
    scaler: Any | None,
    optimizer_step: int,
    kernel: PolicyUpdateKernel | None = None,
    strategy: Any | None = None,
    accelerator: Any | None = None,
    prepared_root: Any | None = None,
    lr_scheduler: Any | None = None,
    ema_update: Any | None = None,
    reference_update: Any | None = None,
    logical_commit: Any | None = None,
    execution_plan: UpdateExecutionPlan | None = None,
) -> PolicyUpdateResult:
    """Run one full-grid reference update outside the production package."""

    import torch

    if not isinstance(trajectory, TrajectoryBatch):
        raise TypeError("trajectory must be a TrajectoryBatch")
    if not isinstance(policy_stats, PolicyStats):
        raise TypeError("policy_stats must be PolicyStats")
    if not isinstance(loss_inputs, PolicyLossInputs):
        raise TypeError("loss_inputs must be PolicyLossInputs")
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("optimizer_step must be a non-negative integer")
    kernel = PolicyUpdateKernel() if kernel is None else kernel
    if not isinstance(kernel, PolicyUpdateKernel):
        raise TypeError("kernel must be PolicyUpdateKernel")
    policy_stats.validate_against_trajectory(trajectory)
    loss_inputs.validate_shape(trajectory.old_log_probs)
    parameters = PolicyUpdateKernel._optimizer_parameters(
        optimizer,
        device=policy_stats.current_log_probs.device,
    )
    if execution_plan is None:
        execution_plan = UpdateExecutionPlan(
            require_finite_gradients=kernel.require_finite_gradients,
            require_nonzero_gradients=kernel.require_nonzero_gradients,
            max_grad_norm=kernel.max_grad_norm,
        )
    elif not isinstance(execution_plan, UpdateExecutionPlan):
        raise TypeError("execution_plan must be UpdateExecutionPlan")
    resolved_accelerator = (
        LocalTestAccelerator(scaler) if accelerator is None else accelerator
    )
    resolved_root = parameters if prepared_root is None else prepared_root
    optimizer.zero_grad(set_to_none=execution_plan.zero_grad_set_to_none)
    objective_state: dict[str, Any] = {}

    def loss_closure() -> PreparedLoss:
        output = ClippedSurrogateObjective().compute(
            old_log_probs=trajectory.old_log_probs,
            policy_stats=policy_stats,
            loss_inputs=loss_inputs,
        )
        active_mask = loss_inputs.active_mask.to(
            device=policy_stats.current_log_probs.device,
            dtype=torch.bool,
        )
        active_delta = (
            (policy_stats.current_log_probs.detach() - trajectory.old_log_probs)
            .abs()
            .masked_select(active_mask)
        )
        delta_max = float(active_delta.max().detach().cpu())
        if (
            kernel.max_initial_logprob_delta is not None
            and delta_max > kernel.max_initial_logprob_delta
        ):
            raise RuntimeError(
                "on-policy old/current ratio parity gate failed: "
                f"log-prob delta {delta_max:.6g} exceeds "
                f"{kernel.max_initial_logprob_delta:.6g}"
            )
        if (
            kernel.require_initial_clipfrac_zero
            and float(output.clipfrac.detach().cpu()) != 0.0
        ):
            raise RuntimeError(
                "on-policy old/current ratio parity gate failed: clipfrac is not zero"
            )
        objective_state["output"] = output
        objective_state["delta_max"] = delta_max
        return PreparedLoss(loss=output.loss, payload=output)

    transaction = execution_plan.execute(
        loss_closure=loss_closure,
        accelerator=resolved_accelerator,
        prepared_root=resolved_root,
        optimizer=optimizer,
        parameters=parameters,
        optimizer_step=optimizer_step,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        ema_update=ema_update,
        reference_update=reference_update,
        logical_commit=logical_commit,
        strategy=strategy,
    )
    if not transaction.committed:
        raise UpdateNotCommittedError(transaction)
    output = objective_state.get("output")
    delta_max = objective_state.get("delta_max")
    if not isinstance(output, LossOutput) or not isinstance(delta_max, float):
        raise TypeError("oracle update lost its objective result")
    pre_clip = transaction.gradient_norm_pre_clip
    post_clip = transaction.gradient_norm_post_clip
    if pre_clip is None or post_clip is None:
        raise RuntimeError("committed oracle update lost gradient diagnostics")
    return PolicyUpdateResult(
        optimizer_step=optimizer_step,
        loss=float(output.loss.detach().cpu()),
        policy_loss=float(output.policy_loss.detach().cpu()),
        reference_kl=float(output.reference_kl.detach().cpu()),
        approx_kl=float(output.approx_kl.detach().cpu()),
        clipfrac=float(output.clipfrac.detach().cpu()),
        active_transition_count=output.active_transition_count,
        logprob_delta_max=delta_max,
        gradient_norm_pre_clip=float(pre_clip),
        gradient_norm_post_clip=float(post_clip),
        transaction=transaction,
    )


__all__ = (
    "LocalTestAccelerator",
    "run_monolithic_policy_update_oracle",
)
