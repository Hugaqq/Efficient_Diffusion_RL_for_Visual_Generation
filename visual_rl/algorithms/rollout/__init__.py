"""Import-safe rollout declaration surface."""

from visual_rl.algorithms.rollout.config import (
    ROLLOUT_CATALOG_FRAGMENT,
    BranchingRolloutConfig,
    BranchingRolloutDeclarationProvider,
    FullTrajectoryRolloutConfig,
    FullTrajectoryRolloutDeclarationProvider,
    SingleStepRolloutConfig,
    SingleStepRolloutDeclarationProvider,
    rollout_catalog_fragment,
)

__all__ = (
    "ROLLOUT_CATALOG_FRAGMENT",
    "BranchingRolloutConfig",
    "BranchingRolloutDeclarationProvider",
    "FullTrajectoryRolloutConfig",
    "FullTrajectoryRolloutDeclarationProvider",
    "SingleStepRolloutConfig",
    "SingleStepRolloutDeclarationProvider",
    "rollout_catalog_fragment",
)
