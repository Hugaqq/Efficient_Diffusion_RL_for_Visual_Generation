"""Strict schema-v2 bootstrap for the production composition root."""

from __future__ import annotations

from dataclasses import dataclass

from visual_rl.composition.config.errors import ConfigMigrationError
from visual_rl.composition.config.source import SourceRecipe
from visual_rl.composition.config.specs import LaunchSpec, SpecValidationError
from visual_rl.core.types import FrozenMapping
from visual_rl.errors import ConfigError

__all__ = (
    "RecipeBootstrapV2",
    "bootstrap_recipe_v2",
)


@dataclass(frozen=True, slots=True)
class RecipeBootstrapV2:
    """Minimal explicit v0.8 recipe selection parsed from schema version 2."""

    schema_version: int
    recipe_id: str
    overrides: FrozenMapping
    source: SourceRecipe
    launch: LaunchSpec | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 2 or type(self.schema_version) is not int:
            raise ValueError("RecipeBootstrapV2.schema_version must equal 2")
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ValueError("RecipeBootstrapV2.recipe_id must be non-empty")
        if not isinstance(self.overrides, FrozenMapping):
            raise TypeError("RecipeBootstrapV2.overrides must be a FrozenMapping")
        if not isinstance(self.source, SourceRecipe):
            raise TypeError("RecipeBootstrapV2.source must be a SourceRecipe")
        if self.launch is not None and not isinstance(self.launch, LaunchSpec):
            raise TypeError("RecipeBootstrapV2.launch must be a LaunchSpec or None")

    def require_launch(self) -> LaunchSpec:
        """Return production launch locations or fail before runtime setup."""

        if self.launch is None:
            raise ConfigError(
                "production execution requires an explicit launch section",
                key="launch",
                path=str(self.source.path),
            )
        return self.launch


def bootstrap_recipe_v2(source: SourceRecipe) -> RecipeBootstrapV2:
    """Parse the exact schema-v2 bootstrap without resolving components."""

    if not isinstance(source, SourceRecipe):
        raise TypeError("source must be a SourceRecipe")
    values = source.values
    schema_version = values.get("schema_version")
    if type(schema_version) is int and schema_version == 1:
        raise ConfigMigrationError(
            source_schema_version=1,
            required_schema_version=2,
            path=str(source.path),
        )
    expected = {"schema_version", "recipe", "overrides", "launch"}
    required = {"schema_version", "recipe"}
    unknown = set(values).difference(expected)
    missing = required.difference(values)
    if unknown or missing:
        raise ConfigError(
            "schema version 2 has an invalid exact key set: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}",
            key="<root>",
            path=str(source.path),
        )
    schema_version = values["schema_version"]
    if type(schema_version) is not int or schema_version != 2:
        raise ConfigError(
            "schema_version must equal integer 2",
            key="schema_version",
            path=str(source.path),
        )
    recipe_id = values["recipe"]
    if not isinstance(recipe_id, str) or not recipe_id:
        raise ConfigError(
            "recipe must be a non-empty versioned recipe id",
            key="recipe",
            path=str(source.path),
        )
    raw_overrides = values.get("overrides", FrozenMapping())
    if not isinstance(raw_overrides, FrozenMapping):
        raise ConfigError(
            "overrides must be a mapping",
            key="overrides",
            path=str(source.path),
        )
    try:
        launch = (
            None
            if "launch" not in values
            else LaunchSpec.from_mapping(values["launch"], context=source.context)
        )
    except SpecValidationError as exc:
        raise ConfigError(
            f"invalid launch specification: {exc}",
            key=exc.path,
            path=str(source.path),
        ) from exc
    return RecipeBootstrapV2(
        schema_version=2,
        recipe_id=recipe_id,
        overrides=raw_overrides,
        source=source,
        launch=launch,
    )
