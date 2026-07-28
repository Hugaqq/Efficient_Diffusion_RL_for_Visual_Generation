"""Policy algorithms, advantage computation, and update plugins."""

from visual_rl.optimizers.base import OptimizerPlugin
from visual_rl.optimizers.advantages import (
    AdvantageComputer,
    AdvantageResult,
)
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin
from visual_rl.optimizers.objective import (
    ObjectiveOutput,
    PolicyLossInputs,
    PolicyObjective,
)
from visual_rl.optimizers.update_engine import UpdateEngine, UpdateResult

__all__ = [
    "OptimizerPlugin",
    "AdvantageComputer",
    "AdvantageResult",
    "PolicyLossInputs",
    "PolicyObjective",
    "ObjectiveOutput",
    "UpdateEngine",
    "UpdateResult",
    "AlgorithmOptimizerPlugin",
]
