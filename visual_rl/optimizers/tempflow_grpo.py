"""TempFlow-GRPO typed branch-credit preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import ClassVar, TYPE_CHECKING

from visual_rl.core.types import (
    FrozenMapping,
    MetricContribution,
    ResolutionContext,
    RolloutBatch,
    RuntimeBuildContext,
)
from visual_rl.optimizers.advantages import AdvantageResult
from visual_rl.optimizers.base import (
    PolicyAlgorithm,
    _resolve_algorithm_params,
)
from visual_rl.optimizers.objective import PolicyLossInputs

if TYPE_CHECKING:
    import torch


@dataclass
class TempFlowGRPOAlgorithm(PolicyAlgorithm):
    """Policy-identity TempFlow preparation with one credited transition."""

    TRAINING_CONTRACT_VERSION = 2
    ADVANTAGE_DTYPE = "float64"
    MIN_GROUP_SIZE = 2
    NOISE_SCALE: ClassVar[float] = 2.25

    clip_range: float = 0.001
    adv_clip_max: float = 5.0

    def __post_init__(self) -> None:
        for name, value in (
            ("clip_range", self.clip_range),
            ("adv_clip_max", self.adv_clip_max),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < float(self.clip_range) < 1.0:
            raise ValueError("clip_range must satisfy 0 < clip_range < 1")
        if float(self.adv_clip_max) <= 0.0:
            raise ValueError("adv_clip_max must be positive")

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> FrozenMapping:
        return _resolve_algorithm_params(raw, context, allow_beta=False)

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> TempFlowGRPOAlgorithm:
        del context
        return cls(
            clip_range=float(resolved["clip_range"]),
            adv_clip_max=float(resolved["adv_clip_max"]),
        )

    def weight_normalization_request(
        self,
        batch: RolloutBatch,
        advantages: AdvantageResult,
    ) -> tuple["torch.Tensor", int] | None:
        _validate_inputs(batch, advantages)
        return None

    def prepare_loss_inputs(
        self,
        batch: RolloutBatch,
        advantages: AdvantageResult,
        *,
        normalization_mean: "torch.Tensor | None",
    ) -> PolicyLossInputs:
        import torch

        if normalization_mean is not None:
            raise ValueError("TempFlow-GRPO does not use weight normalization")
        base = _validate_inputs(batch, advantages)
        branch_steps = batch.branch_step_index.to(device=base.device)
        trajectory_steps = batch.trajectory_step_index.to(device=base.device)
        branch_credit_mask = (
            trajectory_steps.unsqueeze(0) == branch_steps.unsqueeze(1)
        )
        if not bool((branch_credit_mask.sum(dim=1) == 1).all()):
            raise ValueError(
                "each TempFlow branch_step_index must map to exactly one "
                "trajectory_step_index"
            )
        batch_size, transition_count = batch.old_log_probs.shape
        expanded = (
            base[:, None]
            .expand(batch_size, transition_count)
            .clamp(-self.adv_clip_max, self.adv_clip_max)
            .detach()
        )
        transition_mask = batch.transition_mask.to(
            device=expanded.device,
            dtype=torch.bool,
        )
        active_mask = transition_mask & branch_credit_mask
        weight = (
            batch.transition_std_dev.to(
                device=expanded.device,
                dtype=expanded.dtype,
            )
            * self.NOISE_SCALE
        ).detach()
        inputs = PolicyLossInputs(
            base_advantage=expanded,
            algorithm_weight=weight,
            active_mask=active_mask,
            clip_range=float(self.clip_range),
            reference_kl_weight=0.0,
        )
        inputs.validate_against(batch)
        return inputs

    def diagnostics(
        self,
        batch: RolloutBatch,
        inputs: PolicyLossInputs,
    ) -> Mapping[str, MetricContribution]:
        import torch

        inputs.validate_against(batch)
        active_count = int(inputs.active_mask.sum().item())
        return {
            "algorithm/tempflow_noise_weight_mean": MetricContribution(
                numerator=inputs.algorithm_weight.masked_select(
                    inputs.active_mask
                )
                .to(torch.float64)
                .sum()
                .detach(),
                denominator=active_count,
            ),
            "algorithm/tempflow_active_timestep_frac": MetricContribution(
                numerator=inputs.active_mask.to(torch.float64)
                .sum()
                .detach(),
                denominator=batch.batch_size * batch.transition_count,
            ),
        }


def _validate_inputs(
    batch: RolloutBatch,
    advantages: AdvantageResult,
):
    import torch

    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    if not isinstance(advantages, AdvantageResult):
        raise TypeError("advantages must be an AdvantageResult")
    if batch.branch_step_index is None:
        raise ValueError("TempFlow requires branch_step_index")
    if batch.trajectory_step_index is None:
        raise ValueError("TempFlow requires trajectory_step_index")
    if batch.transition_std_dev is None:
        raise ValueError("TempFlow requires transition_std_dev")
    values = advantages.base_advantage
    if values.dtype != torch.float64:
        raise TypeError("TempFlow base_advantage must use torch.float64")
    if tuple(values.shape) != (batch.batch_size,):
        raise ValueError("TempFlow base_advantage must have shape [B]")
    return values
