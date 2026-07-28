"""Core VisualRL data contracts.

``visual_rl.core.types`` and ``visual_rl.core.components`` are lightweight
leaf modules.  The concrete manifest and its only lookup live in
``visual_rl.builtins`` and are intentionally not re-exported here.
"""

from visual_rl.core.types import (
    UINT32_MAX,
    FrozenMapping,
    MetricContribution,
    PolicyRecomputeStats,
    ResolutionContext,
    RewardBatch,
    RewardVector,
    RolloutBatch,
    RolloutRequest,
    RuntimeBuildContext,
    StepContext,
    ValidatedRuntimeEnv,
    ValidationCheck,
    ValidationContext,
    to_plain_dict,
    validate_step_seed_budget,
)
from visual_rl.core.components import (
    CAPABILITY_OWNER,
    CAPABILITY_VOCABULARY,
    COMPONENT_KINDS,
    ComponentKind,
    ComponentSpec,
)

__all__ = [
    "CAPABILITY_OWNER",
    "CAPABILITY_VOCABULARY",
    "COMPONENT_KINDS",
    "UINT32_MAX",
    "ComponentKind",
    "ComponentSpec",
    "FrozenMapping",
    "MetricContribution",
    "PolicyRecomputeStats",
    "ResolutionContext",
    "RewardBatch",
    "RewardVector",
    "RolloutBatch",
    "RolloutRequest",
    "RuntimeBuildContext",
    "StepContext",
    "ValidatedRuntimeEnv",
    "ValidationCheck",
    "ValidationContext",
    "to_plain_dict",
    "validate_step_seed_budget",
]
