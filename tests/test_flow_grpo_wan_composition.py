"""A3 cross-composition contract for existing Flow-GRPO and Wan plugins."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

from visual_rl.composition.config.compiler import compile_recipe_v2, default_catalog
from visual_rl.composition.config.source import load_source_recipe
from visual_rl.composition.recipes.schema import ResolvedRecipe
from visual_rl.composition.registry.resolver import ResolvedComponentDeclaration

ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "configs" / "v2"


@pytest.fixture(autouse=True)
def _available_descriptor_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Static descriptor tests must not require heavyweight train extras."""

    original_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: object, **kwargs: object):
        if name in {
            "diffusers",
            "imageio_ffmpeg",
            "peft",
            "torch",
            "transformers",
        }:
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _compile(filename: str) -> ResolvedRecipe:
    return compile_recipe_v2(load_source_recipe(CONFIG_ROOT / filename))


def _component(
    recipe: ResolvedRecipe,
    slot: str,
) -> ResolvedComponentDeclaration:
    return recipe.component(slot).declaration


def _registry_descriptor(
    declaration: ResolvedComponentDeclaration,
) -> tuple[object, ...]:
    """Project only the registry/class identity, not per-recipe typed params."""

    return (
        declaration.kind,
        declaration.alias,
        declaration.implementation_class_path,
        declaration.declaration_provider_path,
        declaration.config_type_path,
        declaration.descriptor.interface_version,
        declaration.descriptor.optional_dependencies,
    )


def _class_source_sha256(declaration: ResolvedComponentDeclaration) -> str:
    module_name, class_name = declaration.implementation_class_path.split(":", 1)
    component_class = getattr(importlib.import_module(module_name), class_name)
    source = inspect.getsource(component_class).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def test_flow_wan_reuses_flow_algorithm_and_the_one_wan_descriptor() -> None:
    flow_sd3 = _compile("flow_grpo_sd3.yaml")
    flow_wan = _compile("flow_grpo_wan.yaml")
    flash_wan = _compile("flash_wan.yaml")
    world_wan = _compile("world_r1_core_wan.yaml")

    flow_sd3_algorithm = flow_sd3.algorithm.component
    flow_wan_algorithm = flow_wan.algorithm.component
    assert _registry_descriptor(flow_wan_algorithm) == _registry_descriptor(
        flow_sd3_algorithm
    )
    assert _class_source_sha256(flow_wan_algorithm) == _class_source_sha256(
        flow_sd3_algorithm
    )

    registered_flow = default_catalog().for_kind("algorithm").lookup("flow-grpo")
    assert registered_flow is not None
    assert (
        flow_wan_algorithm.implementation_class_path
        == registered_flow.implementation_class_path
    )
    assert (
        flow_wan_algorithm.declaration_provider_path
        == registered_flow.declaration_provider_path
    )
    # Wan does not expose a reference-policy view.  This composition therefore
    # changes only the typed beta config, not the selected algorithm plugin.
    assert flow_sd3_algorithm.config.beta == pytest.approx(0.004)
    assert flow_wan_algorithm.config.beta == pytest.approx(0.0)

    flow_wan_model = flow_wan.model.declaration
    assert flow_wan_model.declaration_id == flash_wan.model.declaration.declaration_id
    assert flow_wan_model.declaration_id == world_wan.model.declaration.declaration_id


def test_cross_selection_replaces_old_typed_params_and_compiles_compatible() -> None:
    flow_wan = _compile("flow_grpo_wan.yaml")

    assert flow_wan.compatibility.status == "compatible"
    dynamics = _component(flow_wan, "dynamics")
    assert dynamics.config.profile.value == "standard"
    assert dynamics.config.likelihood_semantics.value == "exact_env_action"
    assert dynamics.config.replay_target.value == "sampled_action"
    assert not hasattr(dynamics.config, "noise_level")

    reward = _component(flow_wan, "rewards.reward_quality")
    resource = reward.config.resource
    assert resource.factory_class == "reward_general"
    assert resource.artifact_ref == "reward_general"
    assert "default_color" not in resource.semantic_factory_config

    source = next(
        item for item in flow_wan.source_plan.sources if item.source_id == "main"
    )
    assert source.selector == "prompt-video"
    assert source.artifact_ref == "main"
    assert source.artifact_kind == "file"
    assert source.format == "text"


def test_flow_wan_config_adds_no_recipe_or_model_name_branch() -> None:
    package_sources = tuple((ROOT / "visual_rl").rglob("*.py"))
    forbidden = ("flow_grpo_wan", "flow-grpo-wan")
    offenders = {
        path.relative_to(ROOT).as_posix(): token
        for path in package_sources
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }
    assert offenders == {}
