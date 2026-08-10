"""Source loading and the strict schema-v2 bootstrap boundary."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from visual_rl.composition.config.bootstrap import RecipeBootstrapV2, bootstrap_recipe_v2
from visual_rl.composition.config.errors import ConfigMigrationError, ConfigSourceError
from visual_rl.composition.config.source import SourceRecipe, load_source_recipe
from visual_rl.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs" / "v2" / "flow_grpo_sd3.yaml"


def test_source_recipe_hashes_and_freezes_the_exact_utf8_snapshot(tmp_path: Path):
    path = tmp_path / "config.yaml"
    original = V2.read_bytes()
    path.write_bytes(original)

    source = load_source_recipe(path)
    path.write_bytes(original.replace(b"seed: 41", b"seed: 99"))

    assert isinstance(source, SourceRecipe)
    assert source.path == path.resolve()
    assert source.config_source_id == hashlib.sha256(original).hexdigest()
    assert source.values["schema_version"] == 2
    assert source.context.config_path == source.path
    assert source.context.config_dir == source.path.parent


def test_duplicate_key_error_keeps_structured_source_location(tmp_path: Path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "schema_version: 2\nrecipe: flow_grpo_v1\nrecipe: flash_grpo_v1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigSourceError) as excinfo:
        load_source_recipe(path)

    error = excinfo.value
    assert error.code == "config.duplicate_key"
    assert error.key == "recipe"
    assert error.path == str(path.resolve())
    assert (error.line, error.column) == (3, 1)


def test_bootstrap_accepts_only_the_exact_schema_v2_surface(tmp_path: Path):
    source = load_source_recipe(V2)
    bootstrap = bootstrap_recipe_v2(source)

    assert isinstance(bootstrap, RecipeBootstrapV2)
    assert bootstrap.recipe_id == "flow_grpo_v1"
    assert bootstrap.source is source
    assert bootstrap.require_launch().output_dir.is_absolute()

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigMigrationError) as excinfo:
        bootstrap_recipe_v2(load_source_recipe(legacy))
    migration = excinfo.value
    assert migration.code == "config.schema_v1_migration_required"
    assert migration.key == "schema_version"
    assert migration.path == str(legacy.resolve())
    assert migration.source_schema_version == 1
    assert migration.required_schema_version == 2
    assert migration.migration_examples == "configs/v2/"
    assert migration.migration_mode == "offline_only"
    assert "configs/v2/" in str(migration)
    assert "offline" in str(migration)

    inferred = tmp_path / "inferred.yaml"
    inferred.write_text(
        "schema_version: 2\nrecipe: flow_grpo_v1\nmodel: sd3\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"unknown=\['model'\]"):
        bootstrap_recipe_v2(load_source_recipe(inferred))


def test_v08_config_frontend_has_no_legacy_projection_type_owner() -> None:
    class_owners: dict[str, list[Path]] = {
        "LegacyConfigProjection": [],
        "RunResult": [],
    }
    for path in (ROOT / "visual_rl").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in class_owners:
                class_owners[node.name].append(path)

    assert class_owners["LegacyConfigProjection"] == []
    assert class_owners["RunResult"] == [ROOT / "visual_rl" / "runtime" / "types.py"]
