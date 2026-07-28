"""Core VisualRL data contracts.

``visual_rl.core.types`` is imported eagerly (it is a leaf module with no
VisualRL imports). The component manifest lives in
``visual_rl.core.components`` and is exported lazily so that importing base
classes (e.g. ``visual_rl.model_adapters.base``) never triggers a circular
import through the manifest's factory imports.
"""

from visual_rl.core.types import (
    UINT32_MAX,
    AdvantageResult,
    FrozenMapping,
    MetricContribution,
    ObjectiveOutput,
    PolicyLossInputs,
    PolicyRecomputeStats,
    ResolutionContext,
    RewardBatch,
    RewardVector,
    RolloutBatch,
    RolloutRequest,
    RuntimeBuildContext,
    SampleRecord,
    StepArtifacts,
    StepContext,
    StepMetrics,
    StepResult,
    UpdateResult,
    ValidatedRuntimeEnv,
    ValidationCheck,
    ValidationContext,
    to_plain_dict,
    validate_step_seed_budget,
)

_LAZY_COMPONENT_EXPORTS = frozenset(
    {
        "CAPABILITY_OWNER",
        "CAPABILITY_VOCABULARY",
        "COMPONENT_KINDS",
        "ComponentKind",
        "ComponentSpec",
        "builtin_components",
        "get_builtin_component",
    }
)

__all__ = [
    "CAPABILITY_OWNER",
    "CAPABILITY_VOCABULARY",
    "COMPONENT_KINDS",
    "UINT32_MAX",
    "AdvantageResult",
    "ComponentKind",
    "ComponentSpec",
    "FrozenMapping",
    "MetricContribution",
    "ObjectiveOutput",
    "PolicyLossInputs",
    "PolicyRecomputeStats",
    "ResolutionContext",
    "RewardBatch",
    "RewardVector",
    "RolloutBatch",
    "RolloutRequest",
    "RuntimeBuildContext",
    "SampleRecord",
    "StepArtifacts",
    "StepContext",
    "StepMetrics",
    "StepResult",
    "UpdateResult",
    "ValidatedRuntimeEnv",
    "ValidationCheck",
    "ValidationContext",
    "builtin_components",
    "get_builtin_component",
    "to_plain_dict",
    "validate_step_seed_budget",
]


def __getattr__(name: str):
    if name in _LAZY_COMPONENT_EXPORTS:
        from visual_rl.core import components

        value = getattr(components, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
