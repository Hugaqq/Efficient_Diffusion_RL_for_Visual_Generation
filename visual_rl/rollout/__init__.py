"""Public rollout contract and config-driven factory."""

from visual_rl.rollout.base import RolloutEngine
from visual_rl.rollout.full_trajectory import build_rollout_engine

__all__ = ["RolloutEngine", "build_rollout_engine"]
