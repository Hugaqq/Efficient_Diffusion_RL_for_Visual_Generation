"""Full-trajectory rollout engine."""

from __future__ import annotations

from typing import Any

from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.rollout.base import RolloutEngine


class FullTrajectoryRollout(RolloutEngine):
    def sample(self, adapter: ModelAdapter, prompts: list[str], metadata: list[dict[str, Any]]) -> RolloutBatch:
        return adapter.sample(prompts, metadata, self.config)


def build_rollout_engine(config: dict[str, Any]) -> RolloutEngine:
    name = config.get("name", "full_trajectory")
    if name != "full_trajectory":
        raise ValueError(f"v0.1 supports full_trajectory rollout, got {name!r}")
    return FullTrajectoryRollout(config)
