"""Static schema/component/compatibility preflight with no artifact access."""

from __future__ import annotations

from visual_rl.composition.registry import (
    AlgorithmDeclarationResolver,
    Catalog,
    DeclarationResolver,
)
from visual_rl.composition.config.integration import DynamicsProjectionRegistry
from visual_rl.composition.config.compiler import compile_recipe_v2
from visual_rl.composition.config.source import SourceRecipe
from visual_rl.composition.preflight.types import StaticPreflightResult

__all__ = ("run_static_preflight",)


def run_static_preflight(
    source: SourceRecipe,
    catalog: Catalog | None = None,
    *,
    declaration_resolver: DeclarationResolver | None = None,
    algorithm_declaration_resolver: AlgorithmDeclarationResolver | None = None,
    dynamics_projection_registry: DynamicsProjectionRegistry | None = None,
) -> StaticPreflightResult:
    """Compile schema v2 and its typed port graph without environment access."""

    return StaticPreflightResult(
        resolved=compile_recipe_v2(
            source,
            catalog,
            declaration_resolver=declaration_resolver,
            algorithm_declaration_resolver=algorithm_declaration_resolver,
            dynamics_projection_registry=dynamics_projection_registry,
        )
    )
