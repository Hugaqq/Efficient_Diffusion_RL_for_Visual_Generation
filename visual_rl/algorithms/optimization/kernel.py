"""Memory-bounded policy-update kernel for canonical slot streaming.

The kernel is the sole owner of slot objective accumulation, graph-lifetime
boundaries, update-result materialization, and failure traceback sanitation.
It has no monolithic full-grid update path: every current-policy graph is
created for one slot and backpropagated before the next graph is created.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

from visual_rl.algorithms.optimization.execution import (
    PreparedLoss,
    UpdateExecutionPlan,
    UpdateNotCommittedError,
    UpdateTransactionPoisonedError,
    UpdateTransactionResult,
)
from visual_rl.algorithms.optimization.objective import (
    ClippedSurrogateObjective,
    LossOutput,
    PolicyLossInputs,
)
from visual_rl.algorithms.optimization.recompute import (
    PolicyRecomputer,
    PolicyRecomputeRequest,
    ReferencePolicyStats,
)
from visual_rl.algorithms.optimization.slots import UpdateSlot, UpdateSlotPlan
from visual_rl.data.samples.trajectory import TrajectoryBatch


def _clear_exception_traceback_chain(error: BaseException) -> None:
    """Sever graph-bearing frames while retaining typed exception evidence."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        current.__traceback__ = None
        related = (
            current.__cause__,
            current.__context__,
            getattr(current, "cause", None),
            getattr(current, "cleanup_error", None),
        )
        pending.extend(item for item in related if isinstance(item, BaseException))


def _raise_sanitized(error: BaseException) -> None:
    """Raise from a frame that owns only an already-sanitized exception."""

    error.__traceback__ = None
    raise error from None


@dataclass(frozen=True, slots=True)
class DetachedPolicySummary:
    """Materialized ``new_log_probs[B,T]`` with no retained autograd graph."""

    new_log_probs: Any
    materialized_mask: Any
    slot_count: int
    slot_plan_id: str

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.new_log_probs, torch.Tensor):
            raise TypeError("new_log_probs must be a torch.Tensor")
        if self.new_log_probs.ndim != 2 or not self.new_log_probs.is_floating_point():
            raise ValueError("new_log_probs must be floating point [B,T]")
        if self.new_log_probs.device.type != "cpu":
            raise ValueError("detached policy summary must be stored on CPU")
        if self.new_log_probs.requires_grad or self.new_log_probs.grad_fn is not None:
            raise ValueError("detached policy summary must not retain autograd state")
        if not bool(torch.isfinite(self.new_log_probs).all()):
            raise ValueError("new_log_probs must be finite")
        if not isinstance(self.materialized_mask, torch.Tensor):
            raise TypeError("materialized_mask must be a torch.Tensor")
        if self.materialized_mask.dtype != torch.bool or tuple(
            self.materialized_mask.shape
        ) != tuple(self.new_log_probs.shape):
            raise ValueError("materialized_mask must be bool with shape [B,T]")
        if self.materialized_mask.device.type != "cpu":
            raise ValueError("materialized_mask must be stored on CPU")
        if (
            self.materialized_mask.requires_grad
            or self.materialized_mask.grad_fn is not None
        ):
            raise ValueError("materialized_mask must be detached")
        if not bool(self.materialized_mask.any()):
            raise ValueError("materialized_mask must contain a computed cell")
        if bool((self.new_log_probs.masked_select(~self.materialized_mask) != 0).any()):
            raise ValueError(
                "unmaterialized new_log_probs must use canonical zero fill"
            )
        if type(self.slot_count) is not int or self.slot_count < 1:
            raise ValueError("slot_count must be a positive integer")
        if not isinstance(self.slot_plan_id, str) or len(self.slot_plan_id) != 64:
            raise ValueError("slot_plan_id must be a SHA-256 identity")


def _threshold(
    name: str,
    value: object,
    *,
    allow_none: bool,
    allow_zero: bool,
) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a finite number")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number or None")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if resolved < 0.0 or (resolved == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return resolved


@dataclass(frozen=True, slots=True)
class PolicyUpdateResult:
    """Detached metrics for one successfully committed optimizer update."""

    optimizer_step: int
    loss: float
    policy_loss: float
    reference_kl: float
    approx_kl: float
    clipfrac: float
    active_transition_count: int
    logprob_delta_max: float
    gradient_norm_pre_clip: float
    gradient_norm_post_clip: float
    transaction: UpdateTransactionResult = field(compare=False, repr=False)
    policy_summary: DetachedPolicySummary | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        if (
            type(self.active_transition_count) is not int
            or self.active_transition_count < 1
        ):
            raise ValueError("active_transition_count must be positive")
        for name in (
            "loss",
            "policy_loss",
            "reference_kl",
            "approx_kl",
            "clipfrac",
            "logprob_delta_max",
            "gradient_norm_pre_clip",
            "gradient_norm_post_clip",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite Python number")
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, resolved)
        if not isinstance(self.transaction, UpdateTransactionResult):
            raise TypeError("transaction must be an UpdateTransactionResult")
        if not self.transaction.committed:
            raise ValueError("PolicyUpdateResult requires a committed transaction")
        if self.transaction.optimizer_step != self.optimizer_step:
            raise ValueError("transaction optimizer_step does not match metrics")
        if (
            self.transaction.gradient_norm_pre_clip != self.gradient_norm_pre_clip
            or self.transaction.gradient_norm_post_clip != self.gradient_norm_post_clip
        ):
            raise ValueError("transaction gradient diagnostics do not match metrics")
        if self.policy_summary is not None and not isinstance(
            self.policy_summary,
            DetachedPolicySummary,
        ):
            raise TypeError("policy_summary must be DetachedPolicySummary or None")


class _StreamingObjectiveAccumulator:
    """Detach each slot immediately and retain only fixed-size summaries."""

    _FIELDS = ("loss", "policy_loss", "reference_kl", "approx_kl", "clipfrac")

    def __init__(self, trajectory: TrajectoryBatch, plan: UpdateSlotPlan) -> None:
        import torch

        self.plan = plan
        self.numerators = {
            name: torch.zeros(
                (),
                dtype=torch.float64,
                device=trajectory.old_log_probs.device,
            )
            for name in self._FIELDS
        }
        self.delta_max = torch.zeros(
            (),
            dtype=torch.float64,
            device=trajectory.old_log_probs.device,
        )
        self.new_log_probs = torch.zeros_like(trajectory.old_log_probs).detach()
        self.seen = torch.zeros_like(trajectory.transition_mask, dtype=torch.bool)

    def add(
        self,
        *,
        slot: UpdateSlot,
        output: LossOutput,
        old_log_probs: Any,
        current_log_probs: Any,
        active_mask: Any,
    ) -> float:
        import torch

        if output.active_transition_count != slot.active_count:
            raise ValueError("objective active transition count drifted from slot plan")
        for name in self._FIELDS:
            self.numerators[name].add_(
                getattr(output, name)
                .detach()
                .to(
                    device=self.numerators[name].device,
                    dtype=torch.float64,
                )
                * slot.active_count
            )
        active_delta = (
            (current_log_probs.detach() - old_log_probs)
            .abs()
            .masked_select(active_mask)
        )
        if active_delta.numel() != slot.active_count:
            raise ValueError("slot log-prob delta count drifted")
        slot_delta = active_delta.max().to(
            device=self.delta_max.device,
            dtype=torch.float64,
        )
        self.delta_max.copy_(torch.maximum(self.delta_max, slot_delta))
        for local_row, global_row in enumerate(slot.row_indices):
            target = self.new_log_probs[
                global_row,
                slot.transition_start : slot.transition_stop,
            ]
            target.copy_(current_log_probs.detach()[local_row].to(device=target.device))
            self.seen[
                global_row,
                slot.transition_start : slot.transition_stop,
            ] = True
        return float(slot_delta.detach().cpu())

    def finalize(self) -> tuple[dict[str, float], DetachedPolicySummary]:
        expected_active = self._active_mask_on_seen_device()
        if bool((expected_active & ~self.seen).any()):
            raise RuntimeError("streaming update did not materialize every active cell")
        metrics = {
            name: float((value / self.plan.global_active_count).detach().cpu())
            for name, value in self.numerators.items()
        }
        metrics["logprob_delta_max"] = float(self.delta_max.detach().cpu())
        summary = DetachedPolicySummary(
            new_log_probs=self.new_log_probs.detach().to(device="cpu"),
            materialized_mask=self.seen.detach().to(device="cpu"),
            slot_count=len(self.plan.slots),
            slot_plan_id=self.plan.plan_id,
        )
        return metrics, summary

    def _active_mask_on_seen_device(self) -> Any:
        import torch

        expected = torch.zeros_like(self.seen, dtype=torch.bool)
        for row, transition in self.plan.active_cells:
            expected[row, transition] = True
        return expected


class PolicyUpdateKernel:
    """Recompute/backward one slot at a time, then commit exactly once."""

    def __init__(
        self,
        *,
        max_initial_logprob_delta: float | None = None,
        require_initial_clipfrac_zero: bool = False,
        require_finite_gradients: bool = True,
        require_nonzero_gradients: bool = True,
        max_grad_norm: float | None = None,
    ) -> None:
        self.max_initial_logprob_delta = _threshold(
            "max_initial_logprob_delta",
            max_initial_logprob_delta,
            allow_none=True,
            allow_zero=True,
        )
        if type(require_initial_clipfrac_zero) is not bool:
            raise TypeError("require_initial_clipfrac_zero must be bool")
        if type(require_finite_gradients) is not bool:
            raise TypeError("require_finite_gradients must be bool")
        if type(require_nonzero_gradients) is not bool:
            raise TypeError("require_nonzero_gradients must be bool")
        self.require_initial_clipfrac_zero = require_initial_clipfrac_zero
        self.require_finite_gradients = require_finite_gradients
        self.require_nonzero_gradients = require_nonzero_gradients
        self.max_grad_norm = _threshold(
            "max_grad_norm",
            max_grad_norm,
            allow_none=True,
            allow_zero=False,
        )

    @staticmethod
    def _optimizer_parameters(optimizer: Any, *, device: Any) -> tuple[Any, ...]:
        import torch

        param_groups = getattr(optimizer, "param_groups", None)
        if not isinstance(param_groups, (tuple, list)) or not param_groups:
            raise TypeError("optimizer must expose non-empty param_groups")
        if any(
            not isinstance(group, dict) or "params" not in group
            for group in param_groups
        ):
            raise TypeError("optimizer param_groups are malformed")
        parameters = tuple(
            parameter for group in param_groups for parameter in group["params"]
        )
        if not parameters:
            raise ValueError("optimizer must contain trainable parameters")
        if any(not isinstance(item, torch.nn.Parameter) for item in parameters):
            raise TypeError("optimizer params must be torch.nn.Parameter values")
        identities = tuple(id(item) for item in parameters)
        if len(identities) != len(set(identities)):
            raise ValueError("optimizer parameter identities must be unique")
        if any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("optimizer parameters must require gradients")
        if any(parameter.device != device for parameter in parameters):
            raise ValueError("optimizer parameters must match recompute device")
        return parameters

    def step_slots(
        self,
        *,
        trajectory: TrajectoryBatch,
        loss_inputs: PolicyLossInputs,
        recompute_request: PolicyRecomputeRequest,
        optimizer: Any,
        scaler: Any | None,
        optimizer_step: int,
        accelerator: Any,
        prepared_root: Any,
        recomputer: PolicyRecomputer | None = None,
        strategy: Any | None = None,
        lr_scheduler: Any | None = None,
        ema_update: Any | None = None,
        reference_update: Any | None = None,
        logical_commit: Any | None = None,
        execution_plan: UpdateExecutionPlan | None = None,
    ) -> PolicyUpdateResult:
        """Execute one bounded slot stream behind a traceback cleanup boundary."""

        try:
            return self._step_slots_impl(
                trajectory=trajectory,
                loss_inputs=loss_inputs,
                recompute_request=recompute_request,
                optimizer=optimizer,
                scaler=scaler,
                optimizer_step=optimizer_step,
                accelerator=accelerator,
                prepared_root=prepared_root,
                recomputer=recomputer,
                strategy=strategy,
                lr_scheduler=lr_scheduler,
                ema_update=ema_update,
                reference_update=reference_update,
                logical_commit=logical_commit,
                execution_plan=execution_plan,
            )
        except BaseException as error:  # noqa: BLE001 - cleanup covers interrupts
            _clear_exception_traceback_chain(error)
            del (
                trajectory,
                loss_inputs,
                recompute_request,
                optimizer,
                scaler,
                accelerator,
                prepared_root,
                recomputer,
                strategy,
                lr_scheduler,
                ema_update,
                reference_update,
                logical_commit,
                execution_plan,
            )
            _raise_sanitized(error)

    def _step_slots_impl(
        self,
        *,
        trajectory: TrajectoryBatch,
        loss_inputs: PolicyLossInputs,
        recompute_request: PolicyRecomputeRequest,
        optimizer: Any,
        scaler: Any | None,
        optimizer_step: int,
        accelerator: Any,
        prepared_root: Any,
        recomputer: PolicyRecomputer | None,
        strategy: Any | None,
        lr_scheduler: Any | None,
        ema_update: Any | None,
        reference_update: Any | None,
        logical_commit: Any | None,
        execution_plan: UpdateExecutionPlan | None,
    ) -> PolicyUpdateResult:
        """Heavy slot implementation isolated from retained failures."""

        import torch

        if not isinstance(trajectory, TrajectoryBatch):
            raise TypeError("trajectory must be a TrajectoryBatch")
        if not isinstance(loss_inputs, PolicyLossInputs):
            raise TypeError("loss_inputs must be PolicyLossInputs")
        if not isinstance(recompute_request, PolicyRecomputeRequest):
            raise TypeError("recompute_request must be PolicyRecomputeRequest")
        if recompute_request.rollout.trajectory is not trajectory:
            raise ValueError("recompute request must retain the exact trajectory")
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        if accelerator is None:
            raise TypeError("accelerator must be injected")
        if prepared_root is None:
            raise TypeError("prepared_root must be injected")
        trajectory.validate()
        loss_inputs.validate_shape(trajectory.old_log_probs)
        if loss_inputs.base_advantage.device != trajectory.old_log_probs.device:
            raise ValueError("loss inputs and replay log-probs must share a device")
        if loss_inputs.base_advantage.dtype != trajectory.old_log_probs.dtype:
            raise TypeError("loss inputs and replay log-probs must share a dtype")
        objective_mask = loss_inputs.active_mask.to(
            device=trajectory.transition_mask.device,
            dtype=torch.bool,
        )
        if bool((objective_mask & ~trajectory.transition_mask).any()):
            raise ValueError(
                "objective active_mask must be a subset of replay transitions"
            )
        require_reference = loss_inputs.reference_kl_weight > 0.0
        if recompute_request.require_reference_statistics != require_reference:
            raise ValueError(
                "reference recompute requirement differs from the objective"
            )
        if execution_plan is None:
            execution_plan = UpdateExecutionPlan(
                require_finite_gradients=self.require_finite_gradients,
                require_nonzero_gradients=self.require_nonzero_gradients,
                max_grad_norm=self.max_grad_norm,
            )
        elif not isinstance(execution_plan, UpdateExecutionPlan):
            raise TypeError("execution_plan must be an UpdateExecutionPlan")
        parameters = self._optimizer_parameters(
            optimizer,
            device=recompute_request.latent_spec.device,
        )
        if scaler is not None:
            for method in ("scale", "unscale_", "step", "update"):
                if not callable(getattr(scaler, method, None)):
                    raise TypeError(f"scaler must implement {method}()")
        if strategy is not None and not callable(
            getattr(strategy, "atomic_optimizer_step", None)
        ):
            raise TypeError("strategy must implement atomic_optimizer_step()")
        recomputer = recomputer or PolicyRecomputer()
        if not isinstance(recomputer, PolicyRecomputer):
            raise TypeError("recomputer must be PolicyRecomputer")

        slot_plan = UpdateSlotPlan.from_active_mask(
            loss_inputs.active_mask,
            row_microbatch_size=execution_plan.row_microbatch_size,
            transition_window_size=execution_plan.transition_window_size,
        )
        slot_plan.validate_against(loss_inputs.active_mask)
        accumulator = _StreamingObjectiveAccumulator(trajectory, slot_plan)
        optimizer.zero_grad(set_to_none=execution_plan.zero_grad_set_to_none)

        reference_by_slot: dict[str, ReferencePolicyStats] = {}
        if require_reference:
            context = (
                nullcontext()
                if recompute_request.reference_context is None
                else recompute_request.reference_context()
            )
            if not callable(getattr(context, "__enter__", None)) or not callable(
                getattr(context, "__exit__", None)
            ):
                raise TypeError("reference_context must return a context manager")
            with context, torch.no_grad():
                for slot in slot_plan.slots:
                    reference_by_slot[slot.slot_id] = recomputer.compute_reference_slot(
                        recompute_request,
                        slot,
                    )

        def slice_tensor(value: Any, slot: UpdateSlot) -> Any:
            rows = torch.tensor(
                slot.row_indices,
                dtype=torch.int64,
                device=value.device,
            )
            return value.index_select(0, rows)[
                :, slot.transition_start : slot.transition_stop
            ]

        closures = []
        finalized_state: dict[str, Any] = {}
        for slot in slot_plan.slots:

            def loss_closure(slot: UpdateSlot = slot) -> PreparedLoss:
                stats = recomputer.compute_current_slot(
                    recompute_request,
                    slot,
                    reference_stats=reference_by_slot.pop(slot.slot_id, None),
                )
                slot_inputs = loss_inputs.slice(slot.row_indices).slice_transitions(
                    slot.transition_start,
                    slot.transition_stop,
                )
                slot_inputs = replace(
                    slot_inputs,
                    base_advantage=slot_inputs.base_advantage.to(
                        device=stats.current_log_probs.device,
                        dtype=stats.current_log_probs.dtype,
                    ),
                    algorithm_weight=slot_inputs.algorithm_weight.to(
                        device=stats.current_log_probs.device,
                        dtype=stats.current_log_probs.dtype,
                    ),
                    active_mask=slot_inputs.active_mask.to(
                        device=stats.current_log_probs.device,
                        dtype=torch.bool,
                    ),
                )
                old_log_probs = slice_tensor(trajectory.old_log_probs, slot).to(
                    device=stats.current_log_probs.device,
                    dtype=stats.current_log_probs.dtype,
                )
                output = ClippedSurrogateObjective().compute(
                    old_log_probs=old_log_probs,
                    policy_stats=stats,
                    loss_inputs=slot_inputs,
                )
                active_mask = slot_inputs.active_mask.to(
                    device=stats.current_log_probs.device,
                    dtype=torch.bool,
                )
                delta_max = accumulator.add(
                    slot=slot,
                    output=output,
                    old_log_probs=old_log_probs,
                    current_log_probs=stats.current_log_probs,
                    active_mask=active_mask,
                )
                if (
                    self.max_initial_logprob_delta is not None
                    and delta_max > self.max_initial_logprob_delta
                ):
                    raise RuntimeError(
                        "on-policy old/current ratio parity gate failed: "
                        f"log-prob delta {delta_max:.6g} exceeds "
                        f"{self.max_initial_logprob_delta:.6g}"
                    )
                if (
                    self.require_initial_clipfrac_zero
                    and float(output.clipfrac.detach().cpu()) != 0.0
                ):
                    raise RuntimeError(
                        "on-policy old/current ratio parity gate failed: "
                        "clipfrac is not zero"
                    )
                if slot.slot_index == len(slot_plan.slots) - 1:
                    metrics, summary = accumulator.finalize()
                    finalized_state["metrics"] = metrics
                    finalized_state["summary"] = summary
                scaled_loss = output.loss * slot.active_count / slot.global_active_count
                return PreparedLoss(
                    loss=scaled_loss,
                    payload={
                        "slot_id": slot.slot_id,
                        "active_transition_count": slot.active_count,
                    },
                )

            closures.append(loss_closure)

        transaction = execution_plan.execute(
            loss_closures=tuple(closures),
            backward_context=recompute_request.current_context,
            accelerator=accelerator,
            prepared_root=prepared_root,
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
        try:
            metrics = finalized_state.get("metrics")
            summary = finalized_state.get("summary")
            if not isinstance(metrics, dict) or not isinstance(
                summary,
                DetachedPolicySummary,
            ):
                raise TypeError("streaming update lost its pre-commit summary")
            pre_clip = transaction.gradient_norm_pre_clip
            post_clip = transaction.gradient_norm_post_clip
            if pre_clip is None or post_clip is None:
                raise RuntimeError("committed update lost gradient diagnostics")
            return PolicyUpdateResult(
                optimizer_step=optimizer_step,
                loss=metrics["loss"],
                policy_loss=metrics["policy_loss"],
                reference_kl=metrics["reference_kl"],
                approx_kl=metrics["approx_kl"],
                clipfrac=metrics["clipfrac"],
                active_transition_count=slot_plan.global_active_count,
                logprob_delta_max=metrics["logprob_delta_max"],
                gradient_norm_pre_clip=float(pre_clip),
                gradient_norm_post_clip=float(post_clip),
                transaction=transaction,
                policy_summary=summary,
            )
        except BaseException as error:
            raise UpdateTransactionPoisonedError(
                optimizer_step=optimizer_step,
                optimizer_step_applied=True,
                failed_phase="result_materialization",
                trace=transaction.trace,
                cause=error,
            ) from error


__all__ = (
    "DetachedPolicySummary",
    "PolicyUpdateKernel",
    "PolicyUpdateResult",
)
