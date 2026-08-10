"""Exact, memory-bounded policy replay for one immutable update slot.

Recompute is the only owner of differentiable policy statistics.  It bridges
the model runtime and dynamics ports, but owns neither rollout metadata,
credit planning, the numerical objective, nor optimizer execution.  The
production API deliberately has no full-trajectory ``compute`` method: each
current-policy graph is built for one slot and must be backpropagated before
the next slot is materialized.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Any

from visual_rl.algorithms.dynamics.interface import Dynamics, TransitionInput
from visual_rl.algorithms.dynamics.session import DynamicsSession
from visual_rl.algorithms.optimization.advantage import AdvantageGrouping
from visual_rl.algorithms.optimization.slots import UpdateSlot
from visual_rl.algorithms.rollout.interface import (
    RolloutExecution,
    project_model_payload_rows,
)
from visual_rl.core.contracts.runtime import (
    PolicyRuntimePort,
    PolicyTransitionRequest,
)
from visual_rl.data.samples.trajectory import TrajectoryBatch
from visual_rl.models.interface import ModelInput, ModelLatentSpec

__all__ = (
    "PolicyRecomputeError",
    "PolicyRecomputeRequest",
    "PolicyRecomputer",
    "PolicyStats",
    "ReferencePolicyStats",
)


class PolicyRecomputeError(ValueError):
    """Stored rollout state cannot be replayed through the bound policy."""


RecomputeContextFactory = Callable[[], AbstractContextManager[Any]]


def _require_floating_tensor(name: str, value: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    return value


@dataclass(frozen=True, slots=True)
class PolicyStats:
    """One slot's differentiable policy output and optional reference inputs.

    Transition noise and rectification metadata are rollout facts stored on
    :class:`TrajectoryBatch`; duplicating them here would give recompute a
    second, potentially divergent owner.
    """

    grouping: AdvantageGrouping
    current_log_probs: Any
    current_transition_mean: Any | None = None
    transition_std: Any | None = None
    reference_transition_mean: Any | None = None

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.grouping, AdvantageGrouping):
            raise TypeError("grouping must be an AdvantageGrouping")
        current = _require_floating_tensor(
            "current_log_probs",
            self.current_log_probs,
        )
        if current.ndim != 2 or current.shape[0] != self.grouping.batch_size:
            raise ValueError("current_log_probs must have shape [B,T]")
        if not current.requires_grad:
            raise ValueError("current_log_probs must require gradients")
        if not bool(torch.isfinite(current).all()):
            raise ValueError("current_log_probs must be finite")

        reference_fields = (
            self.current_transition_mean,
            self.transition_std,
            self.reference_transition_mean,
        )
        if any(value is None for value in reference_fields) and any(
            value is not None for value in reference_fields
        ):
            raise ValueError(
                "reference policy stats must provide current mean, reference "
                "mean and transition std together"
            )
        if all(value is not None for value in reference_fields):
            current_mean = _require_floating_tensor(
                "current_transition_mean",
                self.current_transition_mean,
            )
            transition_std = _require_floating_tensor(
                "transition_std",
                self.transition_std,
            )
            reference_mean = _require_floating_tensor(
                "reference_transition_mean",
                self.reference_transition_mean,
            )
            expected_prefix = tuple(current.shape)
            if tuple(current_mean.shape[:2]) != expected_prefix:
                raise ValueError("current_transition_mean must start with [B,T]")
            if tuple(reference_mean.shape) != tuple(current_mean.shape):
                raise ValueError("current/reference transition means must match")
            if tuple(transition_std.shape[:2]) != expected_prefix:
                raise ValueError("transition_std must start with [B,T]")
            for name, value in (
                ("current_transition_mean", current_mean),
                ("transition_std", transition_std),
                ("reference_transition_mean", reference_mean),
            ):
                if value.device != current.device:
                    raise ValueError(f"{name} must be on the policy stats device")
                if value.dtype != current.dtype:
                    raise TypeError(f"{name} must use the policy stats dtype")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{name} must be finite")
            if not current_mean.requires_grad:
                raise ValueError("current_transition_mean must require gradients")
            for name, value in (
                ("transition_std", transition_std),
                ("reference_transition_mean", reference_mean),
            ):
                if value.requires_grad or value.grad_fn is not None:
                    raise ValueError(f"{name} must be detached")

    @property
    def batch_size(self) -> int:
        return self.grouping.batch_size

    @property
    def transition_count(self) -> int:
        return int(self.current_log_probs.shape[1])

    def validate_against_trajectory(self, trajectory: TrajectoryBatch) -> None:
        """Validate a test-oracle full-grid value against its source rollout."""

        import torch

        if not isinstance(trajectory, TrajectoryBatch):
            raise TypeError("trajectory must be a TrajectoryBatch")
        self.grouping.validate_against_trajectory(trajectory)
        if tuple(self.current_log_probs.shape) != tuple(trajectory.old_log_probs.shape):
            raise ValueError("policy stats must match trajectory [B,T]")
        if self.current_log_probs.dtype != trajectory.old_log_probs.dtype:
            raise TypeError("old/current log-probs must share one dtype")
        active = trajectory.transition_mask.to(
            device=self.current_log_probs.device,
            dtype=torch.bool,
        )
        if not bool(torch.isfinite(self.current_log_probs.masked_select(active)).all()):
            raise ValueError("active current_log_probs must be finite")


@dataclass(frozen=True, slots=True)
class ReferencePolicyStats:
    """Detached CPU reference-policy means for one immutable update slot."""

    slot_id: str
    transition_mean: Any

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.slot_id, str) or not self.slot_id:
            raise ValueError("slot_id must be non-empty")
        mean = _require_floating_tensor("transition_mean", self.transition_mean)
        if mean.ndim < 3:
            raise ValueError("transition_mean must have shape [B,T,...]")
        if mean.device.type != "cpu":
            raise ValueError("reference transition_mean must be stored on CPU")
        if mean.requires_grad or mean.grad_fn is not None:
            raise ValueError("reference transition_mean must be detached")
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("reference transition_mean must be finite")


@dataclass(frozen=True, slots=True)
class PolicyRecomputeRequest:
    """Live ports and frozen replay state required for exact action replay."""

    adapter: PolicyRuntimePort
    dynamics: Dynamics
    rollout: RolloutExecution
    latent_spec: ModelLatentSpec
    guidance: object | None = None
    require_reference_statistics: bool = False
    current_context: RecomputeContextFactory | None = None
    reference_context: RecomputeContextFactory | None = None

    def __post_init__(self) -> None:
        required = ["predict"]
        if self.require_reference_statistics:
            required.append("predict_reference")
        missing = tuple(
            name for name in required if not callable(getattr(self.adapter, name, None))
        )
        if missing:
            raise TypeError(
                "adapter must implement PolicyRuntimePort operations: "
                + ", ".join(missing)
            )
        if not isinstance(self.dynamics, Dynamics):
            raise TypeError("dynamics must be a Dynamics")
        if not isinstance(self.rollout, RolloutExecution):
            raise TypeError("rollout must be a RolloutExecution")
        if not isinstance(self.latent_spec, ModelLatentSpec):
            raise TypeError("latent_spec must be a ModelLatentSpec")
        if type(self.require_reference_statistics) is not bool:
            raise TypeError("require_reference_statistics must be bool")
        for name, factory in (
            ("current_context", self.current_context),
            ("reference_context", self.reference_context),
        ):
            if factory is not None and not callable(factory):
                raise TypeError(f"{name} must be callable or None")
        if self.require_reference_statistics and (
            (self.current_context is None) != (self.reference_context is None)
        ):
            raise PolicyRecomputeError(
                "reference recompute requires current_context and "
                "reference_context together"
            )
        if not self.require_reference_statistics and self.reference_context is not None:
            raise PolicyRecomputeError(
                "reference_context is invalid when reference statistics are disabled"
            )

        trajectory = self.rollout.trajectory
        trajectory.validate()
        expected_shape = (
            trajectory.batch_size,
            *tuple(trajectory.x_t.shape[2:]),
        )
        if self.latent_spec.shape != expected_shape:
            raise PolicyRecomputeError(
                "latent_spec does not match stored trajectory latent geometry"
            )
        if self.latent_spec.dtype != trajectory.x_t.dtype:
            raise PolicyRecomputeError(
                "latent_spec and stored trajectory must share dtype"
            )
        if len(self.rollout.model_condition_identity) != trajectory.batch_size:
            raise PolicyRecomputeError(
                "model conditioning identity does not match trajectory rows"
            )
        if trajectory.old_log_probs.dtype != trajectory.x_t.dtype:
            raise PolicyRecomputeError(
                "stored policy log-probs and latents must share one dtype"
            )


class PolicyRecomputer:
    """Recompute differentiable policy statistics one bounded slot at a time."""

    def compute_reference_slot(
        self,
        request: PolicyRecomputeRequest,
        slot: UpdateSlot,
    ) -> ReferencePolicyStats:
        """Replay one frozen-reference slot without retaining an autograd graph.

        Model execution-mode/autocast contexts remain caller-owned so the
        kernel can execute every reference slot before opening the current
        policy TRAIN context.
        """

        if not isinstance(request, PolicyRecomputeRequest):
            raise TypeError("request must be a PolicyRecomputeRequest")
        if not isinstance(slot, UpdateSlot):
            raise TypeError("slot must be an UpdateSlot")
        if not request.require_reference_statistics:
            raise PolicyRecomputeError(
                "reference slot recompute requires reference statistics"
            )
        import torch

        _validate_slot(request, slot)
        trajectory = request.rollout.trajectory
        session = DynamicsSession.from_snapshot(
            request.dynamics,
            request.rollout.schedule_snapshot,
        )
        means: list[Any] = []
        with torch.no_grad():
            for step in range(slot.transition_start, slot.transition_stop):
                prediction = _replay_prediction_for_rows(
                    request,
                    slot.row_indices,
                    step,
                    reference=True,
                )
                transition = _transition_for_rows(
                    trajectory,
                    slot.row_indices,
                    prediction,
                    step,
                )
                means.append(session.transition_mean_std(transition).mean.detach())
        return ReferencePolicyStats(
            slot_id=slot.slot_id,
            transition_mean=torch.stack(means, dim=1)
            .detach()
            .to(device="cpu")
            .contiguous(),
        )

    def compute_current_slot(
        self,
        request: PolicyRecomputeRequest,
        slot: UpdateSlot,
        *,
        reference_stats: ReferencePolicyStats | None = None,
    ) -> PolicyStats:
        """Build exactly one differentiable slot graph.

        The caller keeps the current-policy execution context open through the
        matching backward call.  This method intentionally never enters or
        leaves ``request.current_context``.
        """

        if not isinstance(request, PolicyRecomputeRequest):
            raise TypeError("request must be a PolicyRecomputeRequest")
        if not isinstance(slot, UpdateSlot):
            raise TypeError("slot must be an UpdateSlot")
        if request.require_reference_statistics:
            if not isinstance(reference_stats, ReferencePolicyStats):
                raise PolicyRecomputeError(
                    "current slot requires matching detached reference statistics"
                )
            if reference_stats.slot_id != slot.slot_id:
                raise PolicyRecomputeError(
                    "reference statistics belong to another slot"
                )
        elif reference_stats is not None:
            raise PolicyRecomputeError(
                "reference statistics are invalid when reference KL is disabled"
            )
        import torch

        _validate_slot(request, slot)
        trajectory = request.rollout.trajectory
        session = DynamicsSession.from_snapshot(
            request.dynamics,
            request.rollout.schedule_snapshot,
        )
        log_probs: list[Any] = []
        current_means: list[Any] = []
        transition_stds: list[Any] = []

        for step in range(slot.transition_start, slot.transition_stop):
            prediction = _replay_prediction_for_rows(
                request,
                slot.row_indices,
                step,
                reference=False,
            )
            transition = _transition_for_rows(
                trajectory,
                slot.row_indices,
                prediction,
                step,
            )
            action = _select_rows(
                trajectory.scoring_target[:, step],
                slot.row_indices,
            ).to(device=prediction.device, dtype=prediction.dtype)
            transition_port = getattr(request.adapter, "transition", None)
            if callable(transition_port):
                evaluation = transition_port(
                    PolicyTransitionRequest(
                        mode="evaluate",
                        transition_input=transition,
                        transition_session=session,
                        action_latent=action,
                    )
                ).transition_output
            else:
                evaluation = session.evaluate_transition(transition, action)
            log_probs.append(evaluation.log_prob)
            if request.require_reference_statistics:
                current_means.append(evaluation.stats.mean)
                transition_stds.append(evaluation.stats.std.detach())

        current_log_probs = torch.stack(log_probs, dim=1)
        reference_mean = (
            None if reference_stats is None else reference_stats.transition_mean
        )
        if reference_mean is not None:
            reference_mean = reference_mean.to(
                device=current_log_probs.device,
                dtype=current_log_probs.dtype,
            )
        result = PolicyStats(
            grouping=AdvantageGrouping.from_trajectory(trajectory).select_rows(
                slot.row_indices
            ),
            current_log_probs=current_log_probs,
            current_transition_mean=(
                torch.stack(current_means, dim=1)
                if request.require_reference_statistics
                else None
            ),
            transition_std=(
                torch.stack(transition_stds, dim=1).detach()
                if request.require_reference_statistics
                else None
            ),
            reference_transition_mean=reference_mean,
        )
        _validate_policy_stats_against_slot(result, request, slot)
        return result


def _select_rows(value: Any, row_indices: tuple[int, ...]) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        index = torch.tensor(
            row_indices,
            dtype=torch.int64,
            device=value.device,
        )
        return value.index_select(0, index)
    if type(value) is tuple:
        return tuple(value[index] for index in row_indices)
    raise TypeError("slot row selection requires a tensor or tuple")


def _conditioning_for_rows(
    request: PolicyRecomputeRequest,
    row_indices: tuple[int, ...],
) -> object:
    trajectory = request.rollout.trajectory
    canonical_rows = tuple(range(trajectory.batch_size))
    if row_indices == canonical_rows:
        return request.rollout.encoded_conditioning
    expected_identity = tuple(
        request.rollout.model_condition_identity[index] for index in row_indices
    )
    conditioning = project_model_payload_rows(
        request.rollout.encoded_conditioning,
        row_indices,
        label="conditioning",
        identity_attribute="condition_identity",
        expected_identity=expected_identity,
        require_projection=True,
    )
    if conditioning is None:
        raise PolicyRecomputeError("replay conditioning projection returned None")
    return conditioning


def _guidance_for_rows(
    request: PolicyRecomputeRequest,
    row_indices: tuple[int, ...],
    step: int,
) -> object | None:
    trajectory = request.rollout.trajectory
    return project_model_payload_rows(
        request.guidance,
        row_indices,
        label="guidance",
        identity_attribute="guidance_identity",
        expected_identity=tuple(
            trajectory.guidance_identity[index][step] for index in row_indices
        ),
        require_projection=False,
    )


def _model_input_for_rows(
    request: PolicyRecomputeRequest,
    row_indices: tuple[int, ...],
    step: int,
) -> ModelInput:
    trajectory = request.rollout.trajectory
    latents = _select_rows(trajectory.x_t[:, step], row_indices).to(
        device=request.latent_spec.device,
        dtype=request.latent_spec.dtype,
    )
    timestep = _select_rows(trajectory.timesteps[:, step], row_indices).to(
        device=request.latent_spec.device,
    )
    return ModelInput(
        latents=latents,
        timestep=timestep,
        conditioning=_conditioning_for_rows(request, row_indices),
        guidance=_guidance_for_rows(request, row_indices, step),
        latent_spec=replace(
            request.latent_spec,
            shape=(len(row_indices), *request.latent_spec.shape[1:]),
        ),
        condition_identity=tuple(
            request.rollout.model_condition_identity[index] for index in row_indices
        ),
        guidance_identity=tuple(
            trajectory.guidance_identity[index][step] for index in row_indices
        ),
    )


def _replay_prediction_for_rows(
    request: PolicyRecomputeRequest,
    target_row_indices: tuple[int, ...],
    step: int,
    *,
    reference: bool,
) -> Any:
    """Replay complete original forward partitions, then expand target rows.

    The original batch partition and its ordering are numerical inputs for
    bf16/fp16 execution.  Repacking only the slot rows would change rounding
    and break old/new policy parity even when the mathematical model is the
    same.
    """

    import torch

    if type(reference) is not bool:
        raise TypeError("reference must be bool")
    replay = request.rollout.model_forward_replay
    if replay is None:
        raise PolicyRecomputeError("rollout lost its model-forward replay plan")
    target_forward_positions = tuple(
        replay.row_to_forward_position[row] for row in target_row_indices
    )
    target_leaders = tuple(
        replay.forward_row_indices[position] for position in target_forward_positions
    )
    required_leaders = set(target_leaders)
    predict = (
        request.adapter.predict_reference if reference else request.adapter.predict
    )
    prediction_by_leader: dict[int, Any] = {}
    for partition in replay.forward_partitions or ():
        if required_leaders.isdisjoint(partition):
            continue
        model_input = _model_input_for_rows(request, partition, step)
        prediction = predict(model_input)
        prediction.validate_against(model_input)
        for position, leader in enumerate(partition):
            if leader in required_leaders:
                prediction_by_leader[leader] = prediction.value[position : position + 1]
    missing = required_leaders.difference(prediction_by_leader)
    if missing:
        raise PolicyRecomputeError(
            f"model-forward replay partitions lost leader rows: {sorted(missing)}"
        )
    selected = tuple(prediction_by_leader[leader] for leader in target_leaders)
    expanded = selected[0] if len(selected) == 1 else torch.cat(selected, dim=0)
    expected_shape = (
        len(target_row_indices),
        *tuple(request.rollout.trajectory.x_t.shape[2:]),
    )
    if tuple(expanded.shape) != expected_shape:
        raise PolicyRecomputeError(
            "expanded replay prediction does not match target row geometry"
        )
    return expanded


def _validate_slot(request: PolicyRecomputeRequest, slot: UpdateSlot) -> None:
    trajectory = request.rollout.trajectory
    if any(row >= trajectory.batch_size for row in slot.row_indices):
        raise PolicyRecomputeError("update slot row is outside the trajectory")
    if slot.transition_stop > trajectory.transition_count:
        raise PolicyRecomputeError("update slot transition is outside the trajectory")
    import torch

    rows = torch.tensor(
        slot.row_indices,
        dtype=torch.int64,
        device=trajectory.transition_mask.device,
    )
    window = trajectory.transition_mask.index_select(0, rows)[
        :, slot.transition_start : slot.transition_stop
    ]
    if int(window.sum().item()) < slot.active_count:
        raise PolicyRecomputeError(
            "update slot selects more objective cells than replay transitions"
        )
    if int(trajectory.transition_mask.sum().item()) < slot.global_active_count:
        raise PolicyRecomputeError(
            "objective active count exceeds the replay transition count"
        )


def _transition_for_rows(
    trajectory: TrajectoryBatch,
    row_indices: tuple[int, ...],
    prediction: Any,
    step: int,
) -> TransitionInput:
    device = prediction.device

    def tensor_rows(value: Any) -> Any:
        return _select_rows(value, row_indices).to(device=device)

    return TransitionInput(
        x_t=tensor_rows(trajectory.x_t[:, step]),
        model_prediction=prediction,
        t=tensor_rows(trajectory.timesteps[:, step]),
        t_next=tensor_rows(trajectory.next_timesteps[:, step]),
        mask=tensor_rows(trajectory.transition_mask[:, step]),
        transition_index=tensor_rows(trajectory.transition_index[:, step]),
        condition_identity=tuple(
            trajectory.condition_identity[index][step] for index in row_indices
        ),
        guidance_identity=tuple(
            trajectory.guidance_identity[index][step] for index in row_indices
        ),
        storage_dtype_identity=tuple(
            trajectory.storage_dtype_identity[index][step] for index in row_indices
        ),
        quantization_identity=tuple(
            trajectory.quantization_identity[index][step] for index in row_indices
        ),
    )


def _validate_policy_stats_against_slot(
    stats: PolicyStats,
    request: PolicyRecomputeRequest,
    slot: UpdateSlot,
) -> None:
    trajectory = request.rollout.trajectory
    expected_shape = (len(slot.row_indices), slot.transition_count)
    if tuple(stats.current_log_probs.shape) != expected_shape:
        raise PolicyRecomputeError("slot policy statistics have the wrong [B,T] shape")
    old = _select_rows(trajectory.old_log_probs, slot.row_indices)[
        :, slot.transition_start : slot.transition_stop
    ].to(device=stats.current_log_probs.device)
    if stats.current_log_probs.dtype != old.dtype:
        raise PolicyRecomputeError("slot old/current log-probs must share dtype")
    expected_grouping = AdvantageGrouping.from_trajectory(trajectory).select_rows(
        slot.row_indices
    )
    if stats.grouping != expected_grouping:
        raise PolicyRecomputeError("slot policy grouping identity drifted")
