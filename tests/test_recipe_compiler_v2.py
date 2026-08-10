"""Canonical M2.5 compiler, config, and six-route release-cut tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from visual_rl.composition.config.bootstrap import (
    RecipeBootstrapV2,
    bootstrap_recipe_v2,
)
from visual_rl.composition.config.compiler import compile_recipe_v2, default_catalog
from visual_rl.composition.config.errors import ConfigMigrationError
from visual_rl.composition.config.source import load_source_recipe
from visual_rl.composition.recipes.schema import ResolvedRecipe
from visual_rl.composition.registry import Catalog
from visual_rl.core.contracts import AlgorithmComponentRole
from visual_rl.errors import ConfigError

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _ROOT / "configs" / "v2"

_EXPECTED = {
    "flow_grpo_sd3.yaml": {
        "definition": "flow_grpo_v1",
        "algorithm": "flow-grpo",
        "model": "sd3",
        "dynamics": "flow-sde",
        "rollout": "full-trajectory",
        "credit": "grpo",
        "conditioner": None,
        "group_size": 8,
        "beta": 0.004,
        "sources": (("main", "prompt-image"),),
        "rewards": (("reward_quality", "image-quality"),),
    },
    "flow_grpo_wan.yaml": {
        "definition": "flow_grpo_v1",
        "algorithm": "flow-grpo",
        "model": "wan-t2v",
        "dynamics": "wan-flow-sde",
        "rollout": "full-trajectory",
        "credit": "grpo",
        "conditioner": None,
        "group_size": 4,
        "beta": 0.0,
        "sources": (("main", "prompt-video"),),
        "rewards": (("reward_quality", "video-general"),),
    },
    "tempflow_sd3.yaml": {
        "definition": "tempflow_grpo_v1",
        "algorithm": "tempflow-grpo",
        "model": "sd3",
        "dynamics": "flow-sde",
        "rollout": "branching",
        "credit": "tempflow",
        "conditioner": None,
        "group_size": 6,
        "beta": 0.0,
        "sources": (("main", "prompt-image"),),
        "rewards": (("reward_quality", "image-quality"),),
    },
    "flash_wan.yaml": {
        "definition": "flash_grpo_v1",
        "algorithm": "flash-grpo",
        "model": "wan-t2v",
        "dynamics": "wan-flow-sde",
        "rollout": "single-step",
        "credit": "flash",
        "conditioner": None,
        "group_size": 4,
        "beta": 0.0,
        "sources": (("main", "prompt-video"),),
        "rewards": (("reward_general", "video-general"),),
    },
    "world_r1_core_wan.yaml": {
        "definition": "world_r1_core_v1",
        "algorithm": "flow-grpo",
        "model": "wan-t2v",
        "dynamics": "wan-flow-sde",
        "rollout": "full-trajectory",
        "credit": "grpo",
        "conditioner": "world-r1-camera",
        "group_size": 4,
        "beta": 0.0,
        "sources": (("main", "world-r1-prompts"),),
        "rewards": (
            ("reward_3d", "world-r1-3d"),
            ("reward_general", "world-r1-general"),
        ),
    },
    "world_r1_release_surrogate_wan.yaml": {
        "definition": "world_r1_release_surrogate_v1",
        "algorithm": "flow-grpo",
        "model": "wan-t2v",
        "dynamics": "wan-flow-sde",
        "rollout": "full-trajectory",
        "credit": "grpo",
        "conditioner": "world-r1-camera",
        "group_size": 4,
        "beta": 0.0,
        "sources": (
            ("dynamic", "world-r1-dynamic-prompts"),
            ("main", "world-r1-prompts"),
        ),
        "rewards": (
            ("reward_3d", "world-r1-3d"),
            ("reward_general", "world-r1-general"),
        ),
    },
}


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _component_aliases(resolved: ResolvedRecipe) -> dict[str, str]:
    return {item.slot: item.declaration.alias for item in resolved.internal_components}


def test_schema_v2_bootstrap_is_exact_and_v1_requires_offline_migration(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "schema_version: 2\nrecipe: flow_grpo_v1\n")
    bootstrap = bootstrap_recipe_v2(load_source_recipe(path))

    assert isinstance(bootstrap, RecipeBootstrapV2)
    assert bootstrap.recipe_id == "flow_grpo_v1"
    assert bootstrap.overrides == {}

    path.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigMigrationError) as excinfo:
        bootstrap_recipe_v2(load_source_recipe(path))
    assert excinfo.value.code == "config.schema_v1_migration_required"
    assert excinfo.value.migration_mode == "offline_only"

    path.write_text(
        "schema_version: 2\nrecipe: flow_grpo_v1\nmodel: forbidden\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"unknown=\['model'\]"):
        bootstrap_recipe_v2(load_source_recipe(path))


def test_default_catalog_is_the_single_eight_kind_catalog() -> None:
    first = default_catalog()
    second = default_catalog()

    assert isinstance(first, Catalog)
    assert first is second
    assert first.kinds == (
        "model",
        "algorithm",
        "trainer",
        "dynamics",
        "rollout",
        "reward",
        "conditioner",
        "credit",
    )


@pytest.mark.parametrize("filename", tuple(_EXPECTED))
def test_six_official_routes_compile_to_one_canonical_typed_graph(
    filename: str,
) -> None:
    expected = _EXPECTED[filename]
    resolved = compile_recipe_v2(load_source_recipe(_CONFIG_ROOT / filename))
    aliases = _component_aliases(resolved)

    assert isinstance(resolved, ResolvedRecipe)
    assert resolved.definition_id == expected["definition"]
    assert resolved.algorithm.alias == expected["algorithm"]
    assert resolved.model.declaration.alias == expected["model"]
    assert aliases["dynamics"] == expected["dynamics"]
    assert aliases["rollout"] == expected["rollout"]
    assert aliases["credit"] == expected["credit"]
    assert aliases.get("conditioner") == expected["conditioner"]
    assert resolved.execution_policy.group_size == expected["group_size"]
    assert resolved.algorithm_spec.beta == pytest.approx(expected["beta"])
    assert resolved.algorithm_spec.execution_policy_id == (
        resolved.execution_policy.policy_id
    )
    assert resolved.compatibility.status == "compatible"
    assert (
        tuple((item.source_id, item.selector) for item in resolved.source_plan.sources)
        == expected["sources"]
    )
    assert (
        tuple(
            (
                item.logical_reward_id,
                resolved.component(
                    f"rewards.{item.logical_reward_id}"
                ).declaration.alias,
            )
            for item in resolved.reward_plan.logical_rewards
        )
        == expected["rewards"]
    )
    assert resolved.reward_plan.provisional
    assert not hasattr(resolved, "semantic_config")

    selections = {item.role: item for item in resolved.algorithm_spec.components}
    for item in resolved.internal_components:
        role = AlgorithmComponentRole(item.slot)
        assert selections[role].component_declaration_id == (
            item.declaration.declaration_id
        )


def test_world_exact_integration_is_a_separate_typed_definition(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "schema_version: 2\nrecipe: world_r1_exact_env_hook_v1\n",
    )
    resolved = compile_recipe_v2(load_source_recipe(path))

    assert resolved.dynamics_integration.to_payload() == {
        "conditioning": "conditioned",
        "likelihood_semantics": "exact_env_action",
        "replay_target": "sampled_action",
    }
    assert resolved.dynamics_projection.params["profile"] == "conditioned"
    assert resolved.component("conditioner").declaration.alias == "world-r1-camera"


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ("components: {}", "unsupported override roots"),
        ("data: {}", "unsupported override roots"),
        ("rollout: {}", "unsupported override roots"),
        ("algorithm: {id: flow-grpo}", "invalid exact key set"),
        ("execution: {beta: 0.0}", "invalid typed execution override"),
    ),
)
def test_legacy_internal_and_duplicate_owner_fields_fail_during_source_compile(
    tmp_path: Path,
    override: str,
    match: str,
) -> None:
    path = _write(
        tmp_path,
        f"schema_version: 2\nrecipe: flow_grpo_v1\noverrides:\n  {override}\n",
    )

    with pytest.raises(ConfigError, match=match):
        compile_recipe_v2(load_source_recipe(path))


def test_tempflow_group_size_must_equal_blueprint_branch_count(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
schema_version: 2
recipe: tempflow_grpo_v1
overrides:
  execution:
    group_size: 5
""",
    )

    with pytest.raises(
        ConfigError,
        match="branching execution group_size must equal blueprint branch_count",
    ):
        compile_recipe_v2(load_source_recipe(path))


@pytest.mark.parametrize(
    ("recipe", "model_id", "params", "match"),
    (
        (
            "tempflow_grpo_v1",
            "wan-t2v",
            "{artifact_ref: main}",
            "Wan Dynamics binding is not branchable",
        ),
        (
            "flash_grpo_v1",
            "sd3",
            "{artifact_ref: main}",
            "does not implement single-step rectification",
        ),
    ),
)
def test_unsupported_model_algorithm_pairs_fail_before_runtime_import(
    tmp_path: Path,
    recipe: str,
    model_id: str,
    params: str,
    match: str,
) -> None:
    path = _write(
        tmp_path,
        "schema_version: 2\n"
        f"recipe: {recipe}\n"
        "overrides:\n"
        "  model:\n"
        f"    id: {model_id}\n"
        f"    params: {params}\n",
    )

    with pytest.raises(ConfigError, match=match):
        compile_recipe_v2(load_source_recipe(path))


def test_positive_beta_flow_wan_fails_without_reference_policy(tmp_path: Path) -> None:
    text = (_CONFIG_ROOT / "flow_grpo_wan.yaml").read_text(encoding="utf-8")
    text = text.replace("beta: 0.0", "beta: 0.004", 1)
    path = _write(tmp_path, text)

    with pytest.raises(ConfigError, match="reference policy"):
        compile_recipe_v2(load_source_recipe(path))


def test_recipe_identity_contains_no_source_or_launch_audit_data() -> None:
    source = load_source_recipe(_CONFIG_ROOT / "world_r1_release_surrogate_wan.yaml")
    resolved = compile_recipe_v2(source)
    payload = resolved.canonical_semantic_payload()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, (tuple, list)):
            return set().union(*(keys(item) for item in value))
        return set()

    assert {
        "semantic_config",
        "source_path",
        "config_source_id",
        "raw_yaml",
        "override_paths",
        "output_dir",
        "endpoint",
    }.isdisjoint(keys(payload))
    assert str(source.path) not in str(payload)


def test_fresh_process_six_route_compile_does_not_import_heavy_or_runtime_modules() -> (
    None
):
    script = """
import pathlib
import sys
from visual_rl.composition.config.compiler import compile_recipe_v2
from visual_rl.composition.config.source import load_source_recipe
root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob('*.yaml')):
    compile_recipe_v2(load_source_recipe(path))
forbidden = (
    'torch', 'diffusers', 'transformers', 'peft', 'accelerate', 'requests',
    'visual_rl.models.implementations',
    'visual_rl.algorithms.trainer.grpo',
    'visual_rl.algorithms.dynamics.sd3_flow_sde',
    'visual_rl.algorithms.dynamics.wan_flow_sde',
    'visual_rl.algorithms.rewards.components',
)
loaded = sorted(name for name in sys.modules if name == forbidden[0] or any(
    name == prefix or name.startswith(prefix + '.') for prefix in forbidden
))
if loaded:
    raise SystemExit('forbidden imports: ' + ', '.join(loaded))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(_CONFIG_ROOT)],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
