"""Canonical detached credit plans for the built-in GRPO family.

Credit consumes only immutable rollout/reward state and produces detached
``PolicyLossInputs``.  Differentiable policy statistics belong to recompute;
this module never owns or retains a current-policy graph.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from visual_rl.algorithms.optimization.advantage import (
    AdvantageGrouping,
    NormalizedAdvantage,
)
from visual_rl.algorithms.optimization.config import (
    FlashCreditConfig,
    GRPOCreditConfig,
    TempFlowCreditConfig,
)
from visual_rl.algorithms.optimization.interface import (
    CreditComponent,
    CreditPlanningPort,
)
from visual_rl.algorithms.optimization.objective import PolicyLossInputs
from visual_rl.data.samples.trajectory import TrajectoryBatch


class CoefficientMeanReducer(Protocol):
    """Reduce Flash coefficients over the runtime's declared process domain."""

    def reduce_mean(self, values: Any, active_mask: Any) -> Any: ...


class CoefficientReductionError(ValueError):
    """A reducer cannot produce a valid coefficient normalization scalar."""


@dataclass(frozen=True, slots=True)
class LocalCoefficientMeanReducer:
    """Single-process masked mean; distributed runtimes inject their reducer."""

    def reduce_mean(self, values: Any, active_mask: Any) -> Any:
        import torch

        if not isinstance(values, torch.Tensor):
            raise TypeError("coefficient values must be a torch.Tensor")
        if not isinstance(active_mask, torch.Tensor):
            raise TypeError("coefficient active_mask must be a torch.Tensor")
        if tuple(values.shape) != tuple(active_mask.shape):
            raise CoefficientReductionError(
                "coefficient values and active_mask must share shape [B,T]"
            )
        if active_mask.dtype != torch.bool or active_mask.device != values.device:
            raise CoefficientReductionError(
                "coefficient active_mask must be bool on the values device"
            )
        selected = values.detach().masked_select(active_mask)
        if selected.numel() < 1:
            raise CoefficientReductionError(
                "coefficient normalization requires an active transition"
            )
        result = selected.mean().detach()
        if not bool(torch.isfinite(result)) or not bool(result > 0):
            raise CoefficientReductionError(
                "coefficient normalization mean must be finite and positive"
            )
        return result


def _finite_parameter(name: str, value: object, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if positive and resolved <= 0.0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _validate_credit_plan_inputs(
    trajectory: TrajectoryBatch,
    advantage: NormalizedAdvantage,
) -> tuple[Any, Any, Any]:
    """Align detached advantage with the replay policy dtype and device."""

    import torch

    if not isinstance(trajectory, TrajectoryBatch):
        raise TypeError("trajectory must be a TrajectoryBatch")
    if not isinstance(advantage, NormalizedAdvantage):
        raise TypeError("advantage must be a NormalizedAdvantage")
    trajectory.validate()
    advantage.validate_against_trajectory(trajectory)
    prototype = trajectory.old_log_probs
    base = advantage.values.to(
        device=prototype.device,
        dtype=prototype.dtype,
    ).detach()
    valid = advantage.valid_mask.to(device=prototype.device, dtype=torch.bool)
    transition_mask = trajectory.transition_mask.to(
        device=prototype.device,
        dtype=torch.bool,
    )
    return base, valid, transition_mask


class CreditStrategy(CreditPlanningPort, ABC):
    """Map detached rollout credit state to model-independent loss inputs."""

    @staticmethod
    def grouping_spec(trajectory: TrajectoryBatch) -> AdvantageGrouping:
        return AdvantageGrouping.from_trajectory(trajectory)

    @abstractmethod
    def plan(
        self,
        *,
        trajectory: TrajectoryBatch,
        advantage: NormalizedAdvantage,
        coefficient_mean_reducer: CoefficientMeanReducer | None = None,
    ) -> PolicyLossInputs:
        """Build detached full-group inputs without a current-policy graph."""

        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class GRPOCreditStrategy(CreditStrategy):
    """Full-trajectory GRPO credit with optional Flow-GRPO reference KL."""

    advantage_epsilon: float = 1.0e-4
    advantage_std_domain: str = "group"
    clip_range: float = 1.0e-4
    advantage_clip: float = 5.0
    reference_kl_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.advantage_std_domain not in {"group", "batch"}:
            raise ValueError("advantage_std_domain must be group or batch")
        epsilon = _finite_parameter(
            "advantage_epsilon",
            self.advantage_epsilon,
            positive=True,
        )
        clip = _finite_parameter("clip_range", self.clip_range, positive=True)
        if clip >= 1.0:
            raise ValueError("clip_range must be smaller than one")
        advantage_clip = _finite_parameter(
            "advantage_clip",
            self.advantage_clip,
            positive=True,
        )
        beta = _finite_parameter(
            "reference_kl_weight",
            self.reference_kl_weight,
            positive=False,
        )
        if beta < 0.0:
            raise ValueError("reference_kl_weight must be non-negative")
        object.__setattr__(self, "advantage_epsilon", epsilon)
        object.__setattr__(self, "clip_range", clip)
        object.__setattr__(self, "advantage_clip", advantage_clip)
        object.__setattr__(self, "reference_kl_weight", beta)

    def plan(
        self,
        *,
        trajectory: TrajectoryBatch,
        advantage: NormalizedAdvantage,
        coefficient_mean_reducer: CoefficientMeanReducer | None = None,
    ) -> PolicyLossInputs:
        del coefficient_mean_reducer
        import torch

        base, row_valid, transition_mask = _validate_credit_plan_inputs(
            trajectory,
            advantage,
        )
        if advantage.score_axis_names:
            raise ValueError("GRPO credit requires row-only advantage")
        expanded = base[:, None].expand_as(trajectory.old_log_probs)
        expanded = expanded.clamp(
            -self.advantage_clip,
            self.advantage_clip,
        ).detach()
        return PolicyLossInputs(
            base_advantage=expanded,
            algorithm_weight=torch.ones_like(expanded),
            active_mask=transition_mask & row_valid[:, None],
            clip_range=self.clip_range,
            reference_kl_weight=self.reference_kl_weight,
        )


@dataclass(frozen=True, slots=True)
class TempFlowGRPOCreditStrategy(CreditStrategy):
    """TempFlow branch credit with an explicitly configured noise scale."""

    advantage_epsilon: float = 1.0e-4
    advantage_std_domain: str = "group"
    clip_range: float = 1.0e-4
    advantage_clip: float = 5.0
    transition_noise_scale: float = 2.25

    def __post_init__(self) -> None:
        if self.advantage_std_domain != "group":
            raise ValueError("TempFlow advantage_std_domain must be group")
        epsilon = _finite_parameter(
            "advantage_epsilon",
            self.advantage_epsilon,
            positive=True,
        )
        clip = _finite_parameter("clip_range", self.clip_range, positive=True)
        if clip >= 1.0:
            raise ValueError("clip_range must be smaller than one")
        advantage_clip = _finite_parameter(
            "advantage_clip",
            self.advantage_clip,
            positive=True,
        )
        transition_noise_scale = _finite_parameter(
            "transition_noise_scale",
            self.transition_noise_scale,
            positive=True,
        )
        object.__setattr__(self, "advantage_epsilon", epsilon)
        object.__setattr__(self, "clip_range", clip)
        object.__setattr__(self, "advantage_clip", advantage_clip)
        object.__setattr__(
            self,
            "transition_noise_scale",
            transition_noise_scale,
        )

    def plan(
        self,
        *,
        trajectory: TrajectoryBatch,
        advantage: NormalizedAdvantage,
        coefficient_mean_reducer: CoefficientMeanReducer | None = None,
    ) -> PolicyLossInputs:
        del coefficient_mean_reducer
        base, valid, transition_mask = _validate_credit_plan_inputs(
            trajectory,
            advantage,
        )
        if trajectory.kind != "branching" or trajectory.branch_topology is None:
            raise ValueError("TempFlow credit requires a branching trajectory")
        std_dev = trajectory.transition_std_dev
        if std_dev is None:
            raise ValueError(
                "TempFlow credit requires rollout-captured transition_std_dev"
            )
        topology = trajectory.branch_topology
        if topology.kind == "every_policy_timestep":
            if trajectory.branch_group_completeness != "complete":
                raise ValueError(
                    "TempFlow paper credit requires complete K-member groups"
                )
            if advantage.score_axis_names != ("branch_timestep",):
                raise ValueError(
                    "TempFlow paper topology requires [B,T] branch_timestep advantage"
                )
            if tuple(base.shape) != tuple(trajectory.old_log_probs.shape):
                raise ValueError("TempFlow paper advantage must match policy [B,T]")
            active_mask = transition_mask & valid
            if not bool(active_mask.all()):
                raise ValueError(
                    "TempFlow paper credit requires all [B,T] rewards active"
                )
            expanded = base
        else:
            if advantage.score_axis_names:
                raise ValueError(
                    "single_point_branch_ablation requires row-only advantage"
                )
            if trajectory.branch_step_index is None:
                raise ValueError(
                    "single_point_branch_ablation requires branch_step_index"
                )
            branch_steps = trajectory.branch_step_index.to(
                device=base.device,
                dtype=trajectory.transition_index.dtype,
            )
            transition_index = trajectory.transition_index.to(device=base.device)
            selected = transition_index == branch_steps[:, None]
            active_mask = selected & transition_mask & valid[:, None]
            if not bool((active_mask.sum(dim=1)[valid] == 1).all()):
                raise ValueError(
                    "each valid TempFlow ablation row must select exactly one "
                    "active transition"
                )
            expanded = base[:, None].expand_as(trajectory.old_log_probs)
        expanded = expanded.clamp(
            -self.advantage_clip,
            self.advantage_clip,
        ).detach()
        weight = std_dev.to(device=base.device, dtype=base.dtype).detach()
        return PolicyLossInputs(
            base_advantage=expanded,
            algorithm_weight=weight * self.transition_noise_scale,
            active_mask=active_mask,
            clip_range=self.clip_range,
            reference_kl_weight=0.0,
        )


@dataclass(frozen=True, slots=True)
class FlashGRPOCreditStrategy(CreditStrategy):
    """Single-step Flash rectification using an externally reduced mean."""

    advantage_epsilon: float = 1.0e-4
    advantage_std_domain: str = "batch"
    clip_range: float = 0.001
    advantage_clip: float = 5.0

    def __post_init__(self) -> None:
        if self.advantage_std_domain != "batch":
            raise ValueError("Flash advantage_std_domain must be batch")
        epsilon = _finite_parameter(
            "advantage_epsilon",
            self.advantage_epsilon,
            positive=True,
        )
        clip = _finite_parameter("clip_range", self.clip_range, positive=True)
        if clip >= 1.0:
            raise ValueError("clip_range must be smaller than one")
        advantage_clip = _finite_parameter(
            "advantage_clip",
            self.advantage_clip,
            positive=True,
        )
        object.__setattr__(self, "advantage_epsilon", epsilon)
        object.__setattr__(self, "clip_range", clip)
        object.__setattr__(self, "advantage_clip", advantage_clip)

    def plan(
        self,
        *,
        trajectory: TrajectoryBatch,
        advantage: NormalizedAdvantage,
        coefficient_mean_reducer: CoefficientMeanReducer | None = None,
    ) -> PolicyLossInputs:
        base, row_valid, transition_mask = _validate_credit_plan_inputs(
            trajectory,
            advantage,
        )
        if advantage.score_axis_names:
            raise ValueError("Flash credit requires row-only advantage")
        if trajectory.kind != "single_step" or trajectory.transition_count != 1:
            raise ValueError("Flash credit requires a physical single-step trajectory")
        coefficient = trajectory.rectification_coefficient
        if coefficient is None:
            raise ValueError(
                "Flash credit requires rollout-captured rectification_coefficient"
            )
        reduce_mean = getattr(coefficient_mean_reducer, "reduce_mean", None)
        if not callable(reduce_mean):
            raise TypeError(
                "Flash credit requires an explicit coefficient mean reducer"
            )
        active_mask = transition_mask & row_valid[:, None]
        normalization = reduce_mean(coefficient, active_mask)
        expanded = (
            base[:, None].clamp(-self.advantage_clip, self.advantage_clip).detach()
        )
        weight = coefficient.to(device=base.device, dtype=base.dtype)
        weight = (weight / normalization.to(dtype=base.dtype)).detach()
        return PolicyLossInputs(
            base_advantage=expanded,
            algorithm_weight=weight,
            active_mask=active_mask,
            clip_range=self.clip_range,
            reference_kl_weight=0.0,
        )


def _require_runtime_context(runtime_context: object) -> Mapping[str, Any]:
    if not isinstance(runtime_context, Mapping):
        raise TypeError("runtime_context must be a mapping")
    return runtime_context


class RegisteredGRPOCredit(GRPOCreditStrategy, CreditComponent):
    """Registry wrapper materialized exclusively from ``GRPOCreditConfig``."""

    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = "visual_rl.algorithms.optimization.config:GRPOCreditConfig"

    @classmethod
    def describe(cls, config: object) -> object:
        del cls
        if type(config) is not GRPOCreditConfig:
            raise TypeError("config must be exactly GRPOCreditConfig")
        return config.describe_contract()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> RegisteredGRPOCredit:
        if type(config) is not GRPOCreditConfig:
            raise TypeError("config must be exactly GRPOCreditConfig")
        context = _require_runtime_context(runtime_context)
        beta = context.get("beta", 0.0)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError("runtime beta must be numeric")
        return cls(
            advantage_epsilon=config.advantage_epsilon,
            advantage_std_domain=config.advantage_std_domain,
            clip_range=config.clip_range,
            advantage_clip=config.advantage_clip,
            reference_kl_weight=float(beta),
        )


class RegisteredTempFlowCredit(TempFlowGRPOCreditStrategy, CreditComponent):
    """Registry wrapper materialized exclusively from ``TempFlowCreditConfig``."""

    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = "visual_rl.algorithms.optimization.config:TempFlowCreditConfig"

    @classmethod
    def describe(cls, config: object) -> object:
        del cls
        if type(config) is not TempFlowCreditConfig:
            raise TypeError("config must be exactly TempFlowCreditConfig")
        return config.describe_contract()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> RegisteredTempFlowCredit:
        if type(config) is not TempFlowCreditConfig:
            raise TypeError("config must be exactly TempFlowCreditConfig")
        _require_runtime_context(runtime_context)
        return cls(
            advantage_epsilon=config.advantage_epsilon,
            advantage_std_domain=config.advantage_std_domain,
            clip_range=config.clip_range,
            advantage_clip=config.advantage_clip,
            transition_noise_scale=config.transition_noise_scale,
        )


class RegisteredFlashCredit(FlashGRPOCreditStrategy, CreditComponent):
    """Registry wrapper materialized exclusively from ``FlashCreditConfig``."""

    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = "visual_rl.algorithms.optimization.config:FlashCreditConfig"

    @classmethod
    def describe(cls, config: object) -> object:
        del cls
        if type(config) is not FlashCreditConfig:
            raise TypeError("config must be exactly FlashCreditConfig")
        return config.describe_contract()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> RegisteredFlashCredit:
        if type(config) is not FlashCreditConfig:
            raise TypeError("config must be exactly FlashCreditConfig")
        _require_runtime_context(runtime_context)
        return cls(
            advantage_epsilon=config.advantage_epsilon,
            advantage_std_domain=config.advantage_std_domain,
            clip_range=config.clip_range,
            advantage_clip=config.advantage_clip,
        )


__all__ = (
    "CoefficientReductionError",
    "CoefficientMeanReducer",
    "CreditStrategy",
    "FlashGRPOCreditStrategy",
    "GRPOCreditStrategy",
    "LocalCoefficientMeanReducer",
    "RegisteredFlashCredit",
    "RegisteredGRPOCredit",
    "RegisteredTempFlowCredit",
    "TempFlowGRPOCreditStrategy",
)
