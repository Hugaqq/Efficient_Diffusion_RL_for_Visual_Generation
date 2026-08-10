"""Canonical, import-safe schema-v2 configuration front-end."""

from visual_rl.composition.config.bootstrap import (
    RecipeBootstrapV2,
    bootstrap_recipe_v2,
)
from visual_rl.composition.config.errors import (
    ConfigMigrationError,
    ConfigSourceError,
)
from visual_rl.composition.config.integration import (
    DynamicsConditioningMode,
    DynamicsIntegrationSpec,
    DynamicsProjectionRegistry,
    ModelBoundDynamicsProjection,
    ModelBoundDynamicsProjectionError,
    ModelBoundDynamicsProjector,
    bind_model_bound_dynamics_declaration,
    default_dynamics_projection_registry,
    project_model_bound_dynamics,
)
from visual_rl.composition.config.source import SourceRecipe, load_source_recipe
from visual_rl.composition.config.specs import (
    AdamWSpec,
    ArtifactLocations,
    ExecutionPolicySpec,
    LaunchSpec,
    LearningRateScheduleSpec,
    PolicyRecomputeSpec,
    RewardRuntimeBindingSpec,
    RolloutExecutionPolicySpec,
    SpecValidationError,
    TrainingSpec,
    UpdateSafetySpec,
)

__all__ = (
    "AdamWSpec",
    "ArtifactLocations",
    "ConfigMigrationError",
    "ConfigSourceError",
    "DynamicsConditioningMode",
    "DynamicsIntegrationSpec",
    "DynamicsProjectionRegistry",
    "ExecutionPolicySpec",
    "LaunchSpec",
    "LearningRateScheduleSpec",
    "ModelBoundDynamicsProjection",
    "ModelBoundDynamicsProjectionError",
    "ModelBoundDynamicsProjector",
    "PolicyRecomputeSpec",
    "RecipeBootstrapV2",
    "RewardRuntimeBindingSpec",
    "RolloutExecutionPolicySpec",
    "SourceRecipe",
    "SpecValidationError",
    "TrainingSpec",
    "UpdateSafetySpec",
    "bind_model_bound_dynamics_declaration",
    "default_dynamics_projection_registry",
    "bootstrap_recipe_v2",
    "compile_recipe_v2",
    "default_catalog",
    "load_source_recipe",
    "project_model_bound_dynamics",
)


def __getattr__(name: str):
    """Load compiler symbols only when the composition root requests them."""

    if name in {"compile_recipe_v2", "default_catalog"}:
        from visual_rl.composition.config import compiler

        return getattr(compiler, name)
    raise AttributeError(name)
