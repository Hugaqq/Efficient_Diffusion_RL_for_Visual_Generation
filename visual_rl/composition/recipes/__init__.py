"""Canonical typed recipe definitions and resolved/materialized identities."""

from visual_rl.composition.recipes.builtins import (
    BUILTIN_RECIPE_DEFINITIONS,
    LogicalRewardDefinition,
    RecipeDefinition,
    RequestedComponent,
    apply_recipe_overrides,
    builtin_recipe_definitions,
    get_recipe_definition,
)
from visual_rl.composition.recipes.schema import (
    FidelityTarget,
    MaterializedRecipe,
    ResolvedRecipe,
    ResolvedSlotDeclaration,
)

__all__ = (
    "BUILTIN_RECIPE_DEFINITIONS",
    "FidelityTarget",
    "LogicalRewardDefinition",
    "MaterializedRecipe",
    "RecipeDefinition",
    "RequestedComponent",
    "ResolvedRecipe",
    "ResolvedSlotDeclaration",
    "apply_recipe_overrides",
    "builtin_recipe_definitions",
    "get_recipe_definition",
)
