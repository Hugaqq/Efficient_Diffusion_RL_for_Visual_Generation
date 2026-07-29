"""Flash-GRPO typed rectification preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

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
class FlashGRPOAlgorithm(PolicyAlgorithm):
    """Reference-coefficient Flash preparation with one global mean."""

    TRAINING_CONTRACT_VERSION = 2
    ADVANTAGE_DTYPE = "float32"
    MIN_GROUP_SIZE = 2

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
    ) -> FlashGRPOAlgorithm:
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
        coefficient, base = _validate_inputs(batch, advantages)
        coefficient = coefficient.to(
            device=base.device,
            dtype=base.dtype,
        )
        return coefficient.mean().detach(), batch.batch_size

    def prepare_loss_inputs(
        self,
        batch: RolloutBatch,
        advantages: AdvantageResult,
        *,
        normalization_mean: "torch.Tensor | None",
    ) -> PolicyLossInputs:
        import torch

        coefficient, base = _validate_inputs(batch, advantages)
        if normalization_mean is None:
            raise ValueError("Flash-GRPO requires a global coefficient mean")
        mean = torch.as_tensor(
            normalization_mean,
            device=base.device,
            dtype=base.dtype,
        )
        if (
            mean.ndim != 0
            or not bool(torch.isfinite(mean))
            or not bool(mean > 0)
        ):
            raise ValueError(
                "Flash-GRPO normalization mean must be finite and positive"
            )
        expanded = (
            base[:, None]
            .clamp(-self.adv_clip_max, self.adv_clip_max)
            .detach()
        )
        weight = coefficient.to(
            device=expanded.device,
            dtype=expanded.dtype,
        ) / mean
        active_mask = batch.transition_mask.to(
            device=expanded.device,
            dtype=torch.bool,
        )
        inputs = PolicyLossInputs(
            base_advantage=expanded,
            algorithm_weight=weight.detach(),
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
        selected = batch.selected_timestep_index
        if selected is None:
            raise ValueError(
                "Flash-GRPO requires selected_timestep_index diagnostics"
            )
        active_mask = inputs.active_mask
        active_count = int(active_mask.sum().item())
        effective_advantage = (
            inputs.base_advantage * inputs.algorithm_weight
        )
        return {
            "algorithm/flash_rectification_weight_mean": MetricContribution(
                numerator=inputs.algorithm_weight.masked_select(active_mask)
                .to(torch.float64)
                .sum()
                .detach(),
                denominator=active_count,
            ),
            "algorithm/flash_selected_timestep_mean": MetricContribution(
                numerator=selected.to(
                    device=inputs.base_advantage.device,
                    dtype=torch.float64,
                )
                .sum()
                .detach(),
                denominator=batch.batch_size,
            ),
            "algorithm/flash_active_timestep_frac": MetricContribution(
                numerator=(
                    (effective_advantage != 0) & active_mask
                )
                .to(torch.float64)
                .sum()
                .detach(),
                denominator=active_count,
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
    if batch.transition_count != 1:
        raise ValueError("Flash-GRPO requires a physical T=1 rollout")
    coefficient = batch.flash_coefficient
    if coefficient is None:
        raise ValueError("Flash-GRPO requires batch.flash_coefficient")
    values = advantages.base_advantage
    if values.dtype != torch.float32:
        raise TypeError("Flash-GRPO base_advantage must use torch.float32")
    if tuple(values.shape) != (batch.batch_size,):
        raise ValueError("Flash-GRPO base_advantage must have shape [B]")
    return coefficient, values
