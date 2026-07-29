"""The sole reward-to-group-advantage conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

from visual_rl.core.types import (
    MetricContribution,
    RewardBatch,
    RolloutBatch,
)


@dataclass(frozen=True)
class AdvantageResult:
    """Detached group-relative advantage plus reducible diagnostics."""

    base_advantage: Any
    metrics: Mapping[str, MetricContribution]

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.base_advantage, torch.Tensor):
            raise TypeError("base_advantage must be a torch.Tensor")
        if self.base_advantage.ndim != 1:
            raise ValueError("base_advantage must have shape [B]")
        if not self.base_advantage.is_floating_point():
            raise TypeError("base_advantage must be floating point")
        if (
            self.base_advantage.requires_grad
            or self.base_advantage.grad_fn is not None
        ):
            raise ValueError("base_advantage must be detached without grad_fn")
        if not bool(torch.isfinite(self.base_advantage).all()):
            raise ValueError("base_advantage must be finite")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        frozen_metrics: dict[str, MetricContribution] = {}
        for key, contribution in self.metrics.items():
            if not isinstance(key, str) or not key.startswith("advantage/"):
                raise ValueError(
                    "advantage metric keys must use the advantage/ namespace"
                )
            if not isinstance(contribution, MetricContribution):
                raise TypeError(
                    "advantage metrics must contain MetricContribution values"
                )
            frozen_metrics[key] = contribution
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(frozen_metrics),
        )


class AdvantageComputer:
    """Normalize the already-weighted reward once by typed occurrence group."""

    def __init__(
        self,
        *,
        epsilon: float,
        output_dtype: Literal["float32", "float64"],
    ) -> None:
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise TypeError("advantage epsilon must be a number, not bool")
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("advantage epsilon must be finite and positive")
        if output_dtype not in {"float32", "float64"}:
            raise ValueError("output_dtype must be float32 or float64")
        self.output_dtype = output_dtype

    def __call__(
        self,
        batch: RolloutBatch,
        rewards: RewardBatch,
    ) -> AdvantageResult:
        import torch

        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if not isinstance(rewards, RewardBatch):
            raise TypeError("rewards must be a RewardBatch")
        rewards.validate_against(batch)
        values = rewards.weighted_total.detach().to(dtype=torch.float64)
        groups: dict[str, list[int]] = {}
        for row, group_id in enumerate(batch.group_id):
            groups.setdefault(group_id, []).append(row)
        singleton = tuple(
            group_id for group_id, rows in groups.items() if len(rows) < 2
        )
        if singleton:
            raise ValueError(
                "group-relative advantage requires at least two rows per "
                f"group: {singleton}"
            )

        advantages = torch.empty_like(values)
        zero_std_groups = 0
        for rows in groups.values():
            index = torch.tensor(rows, dtype=torch.long, device=values.device)
            group_values = values.index_select(0, index)
            std = group_values.std(correction=0)
            if float(std.detach().cpu()) <= self.epsilon:
                zero_std_groups += 1
            normalized = (
                group_values - group_values.mean()
            ) / (std + self.epsilon)
            advantages.index_copy_(0, index, normalized)

        output_dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
        }[self.output_dtype]
        group_count = len(groups)
        metric_device = values.device
        return AdvantageResult(
            base_advantage=advantages.to(dtype=output_dtype).detach(),
            metrics={
                "advantage/zero_std_ratio": MetricContribution(
                    numerator=torch.tensor(
                        float(zero_std_groups),
                        dtype=torch.float64,
                        device=metric_device,
                    ),
                    denominator=group_count,
                ),
                "advantage/group_size_mean": MetricContribution(
                    numerator=torch.tensor(
                        float(sum(map(len, groups.values()))),
                        dtype=torch.float64,
                        device=metric_device,
                    ),
                    denominator=group_count,
                ),
                "advantage/trained_prompt_num": MetricContribution(
                    numerator=torch.tensor(
                        float(group_count),
                        dtype=torch.float64,
                        device=metric_device,
                    ),
                    denominator=None,
                ),
            },
        )
