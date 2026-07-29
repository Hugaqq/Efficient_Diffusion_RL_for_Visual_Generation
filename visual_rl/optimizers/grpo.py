"""GRPO preparation for the sole shared policy objective."""

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
class GRPOAlgorithm(PolicyAlgorithm):
    """Group-relative advantage with optional Flow-GRPO reference KL."""

    TRAINING_CONTRACT_VERSION = 2
    ADVANTAGE_DTYPE = "float32"
    MIN_GROUP_SIZE = 2

    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("clip_range", self.clip_range),
            ("adv_clip_max", self.adv_clip_max),
            ("beta", self.beta),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < float(self.clip_range) < 1.0:
            raise ValueError("clip_range must satisfy 0 < clip_range < 1")
        if float(self.adv_clip_max) <= 0.0:
            raise ValueError("adv_clip_max must be positive")
        if float(self.beta) < 0.0:
            raise ValueError("beta must be non-negative")

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> FrozenMapping:
        return _resolve_algorithm_params(raw, context, allow_beta=True)

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> GRPOAlgorithm:
        del context
        return cls(
            clip_range=float(resolved["clip_range"]),
            adv_clip_max=float(resolved["adv_clip_max"]),
            beta=float(resolved["beta"]),
        )

    @classmethod
    def required_capabilities(
        cls,
        resolved_params: Mapping[str, object],
    ) -> frozenset[str]:
        del cls
        return (
            frozenset({"policy.reference_stats"})
            if float(resolved_params["beta"]) > 0.0
            else frozenset()
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
            raise ValueError("GRPO does not use weight normalization")
        base = _validate_inputs(batch, advantages)
        batch_size, transition_count = batch.old_log_probs.shape
        expanded = (
            base[:, None]
            .expand(batch_size, transition_count)
            .clamp(-self.adv_clip_max, self.adv_clip_max)
            .detach()
        )
        active_mask = batch.transition_mask.to(
            device=expanded.device,
            dtype=torch.bool,
        )
        inputs = PolicyLossInputs(
            base_advantage=expanded,
            algorithm_weight=torch.ones_like(expanded),
            active_mask=active_mask,
            clip_range=float(self.clip_range),
            reference_kl_weight=float(self.beta),
        )
        inputs.validate_against(batch)
        return inputs

    def diagnostics(
        self,
        batch: RolloutBatch,
        inputs: PolicyLossInputs,
    ) -> Mapping[str, MetricContribution]:
        inputs.validate_against(batch)
        return {}


def _validate_inputs(
    batch: RolloutBatch,
    advantages: AdvantageResult,
):
    import torch

    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    if not isinstance(advantages, AdvantageResult):
        raise TypeError("advantages must be an AdvantageResult")
    values = advantages.base_advantage
    if values.dtype != torch.float32:
        raise TypeError("GRPO base_advantage must use torch.float32")
    if tuple(values.shape) != (batch.batch_size,):
        raise ValueError("GRPO base_advantage must have shape [B]")
    return values
