"""The sole reward-to-group-advantage conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

from visual_rl.core.types import RewardBatch, RolloutBatch


@dataclass(frozen=True)
class AdvantageResult:
    advantages: Any
    metrics: dict[str, float]


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
        values = rewards.weighted_total.detach().to(
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[self.output_dtype]
        )
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

        return AdvantageResult(
            advantages=advantages.detach(),
            metrics={
                "zero_std_ratio": zero_std_groups / len(groups),
                "group_size": sum(map(len, groups.values())) / len(groups),
                "trained_prompt_num": float(len(groups)),
            },
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("AdvantageComputer has no persistent state")
