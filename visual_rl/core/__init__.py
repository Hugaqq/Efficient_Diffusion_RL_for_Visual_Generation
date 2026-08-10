"""Small, dependency-light data contracts shared by VisualRL."""

from visual_rl.core.types import (
    UINT32_MAX,
    FrozenMapping,
    ResolutionContext,
    RuntimeBuildContext,
    StepContext,
    ValidatedRuntimeEnv,
    ValidationCheck,
    ValidationContext,
    to_plain_dict,
    validate_step_seed_budget,
)

__all__ = [
    "UINT32_MAX",
    "FrozenMapping",
    "ResolutionContext",
    "RuntimeBuildContext",
    "StepContext",
    "ValidatedRuntimeEnv",
    "ValidationCheck",
    "ValidationContext",
    "to_plain_dict",
    "validate_step_seed_budget",
]
