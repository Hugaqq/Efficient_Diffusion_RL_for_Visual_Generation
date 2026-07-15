"""Policy algorithms, advantage computation, and update plugins."""

from visual_rl.optimizers.base import OptimizerPlugin
from visual_rl.optimizers.advantages import AdvantageComputer, AdvantageFunction, AdvantageResult
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin
from visual_rl.optimizers.factory import build_algorithm, build_optimizer_plugin
from visual_rl.optimizers.objective import (
    AlgorithmPolicyObjective,
    ObjectiveOutput,
    PolicyObjective,
)
from visual_rl.optimizers.update_engine import UpdateEngine

__all__ = [
    "OptimizerPlugin",
    "AdvantageFunction",
    "AdvantageComputer",
    "AdvantageResult",
    "PolicyObjective",
    "AlgorithmPolicyObjective",
    "ObjectiveOutput",
    "UpdateEngine",
    "AlgorithmOptimizerPlugin",
    "build_algorithm",
    "build_optimizer_plugin",
]
