"""Golden tests for the one immutable builtin component manifest."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from visual_rl.builtins import builtin_components, get_builtin_component
from visual_rl.core.components import (
    CAPABILITY_OWNER,
    CAPABILITY_VOCABULARY,
    COMPONENT_KINDS,
    ComponentSpec,
)
from visual_rl.errors import ComponentError, UnknownComponentError

_MODEL_DEPS = (
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "sentencepiece",
    "google.protobuf",
)

GOLDEN_MATRIX = {
    ("model", "tiny_diffusion"): (
        {
            "media.image",
            "policy.reference_stats",
            "sampling.full_trajectory",
            "sampling.single_step",
            "sampling.branching",
        },
        set(),
        ("torch",),
    ),
    ("model", "sd3_tempflow"): (
        {
            "media.image",
            "policy.reference_stats",
            "sampling.full_trajectory",
            "sampling.branching",
        },
        set(),
        _MODEL_DEPS,
    ),
    ("model", "wan_flash"): (
        {"media.video", "sampling.single_step"},
        set(),
        _MODEL_DEPS,
    ),
    ("model", "wan_world_r1"): (
        {"media.video", "sampling.full_trajectory", "conditioning.camera"},
        set(),
        _MODEL_DEPS,
    ),
    ("rollout", "full_trajectory"): (
        {"rollout.full_trajectory"},
        {"sampling.full_trajectory"},
        ("torch",),
    ),
    ("rollout", "single_step"): (
        {"rollout.single_step"},
        {"sampling.single_step"},
        ("torch",),
    ),
    ("rollout", "branching"): (
        {"rollout.branching"},
        {"sampling.branching"},
        ("torch",),
    ),
    ("reward", "mock"): (set(), set(), ("numpy",)),
    ("reward", "prompt_color"): (set(), {"media.image"}, ("numpy",)),
    ("reward", "prompt_color_margin"): (set(), {"media.image"}, ("numpy",)),
    ("reward", "prompt_color_guarded"): (set(), {"media.image"}, ("numpy",)),
    ("reward", "reward_general"): (
        set(),
        {"media.video"},
        ("numpy", "PIL", "requests"),
    ),
    ("reward", "reward_3d"): (
        set(),
        {"media.video", "conditioning.camera"},
        ("numpy", "PIL", "requests"),
    ),
    ("algorithm", "grpo"): (
        set(),
        {"rollout.full_trajectory"},
        ("torch",),
    ),
    ("algorithm", "flash_grpo"): (
        set(),
        {"media.video", "rollout.single_step"},
        ("torch",),
    ),
    ("algorithm", "tempflow_grpo"): (
        set(),
        {"media.image", "rollout.branching"},
        ("torch",),
    ),
}

EXPECTED_CAPABILITY_OWNER = {
    "media.image": "model",
    "media.video": "model",
    "sampling.full_trajectory": "model",
    "sampling.single_step": "model",
    "sampling.branching": "model",
    "rollout.full_trajectory": "rollout",
    "rollout.single_step": "rollout",
    "rollout.branching": "rollout",
    "policy.reference_stats": "model",
    "conditioning.camera": "model",
}

RETIRED_ALIASES = (
    ("model", "tempflow_sd3_legacy"),
    ("model", "world_r1_wan_legacy"),
    ("model", "mock_wan"),
    ("model", "wan"),
    ("rollout", "flash_single_step"),
    ("reward", "remote_pickle"),
    ("reward", "pickscore"),
    ("reward", "video_hpsv3"),
    ("reward", "reward_router"),
)


def _matrix():
    return {
        (spec.kind, spec.name): (
            set(spec.provides),
            set(spec.requires),
            spec.dependencies,
        )
        for spec in builtin_components()
    }


def test_manifest_matches_the_complete_plan_matrix():
    assert _matrix() == GOLDEN_MATRIX
    assert COMPONENT_KINDS == ("model", "rollout", "reward", "algorithm")
    assert {spec.kind for spec in builtin_components()} == set(COMPONENT_KINDS)


def test_manifest_is_one_fixed_tuple_with_unique_factories():
    manifest = builtin_components()
    assert isinstance(manifest, tuple)
    assert builtin_components() is manifest
    pairs = [(spec.kind, spec.name) for spec in manifest]
    assert len(pairs) == len(set(pairs))
    for kind in COMPONENT_KINDS:
        factories = [spec.factory for spec in manifest if spec.kind == kind]
        assert len(factories) == len(set(factories))


def test_component_spec_is_only_the_frozen_description():
    assert dataclasses.is_dataclass(ComponentSpec)
    assert ComponentSpec.__dataclass_params__.frozen
    assert tuple(field.name for field in dataclasses.fields(ComponentSpec)) == (
        "kind",
        "name",
        "factory",
        "provides",
        "requires",
        "dependencies",
    )
    for spec in builtin_components():
        assert isinstance(spec, ComponentSpec)
        assert inspect.isclass(spec.factory)
        assert spec.provides <= CAPABILITY_VOCABULARY
        assert spec.requires <= CAPABILITY_VOCABULARY
        assert all(CAPABILITY_OWNER[item] == spec.kind for item in spec.provides)


def test_capability_vocabulary_and_owners_are_exact():
    assert CAPABILITY_OWNER == EXPECTED_CAPABILITY_OWNER
    assert CAPABILITY_VOCABULARY == frozenset(EXPECTED_CAPABILITY_OWNER)
    providers = {
        spec.name
        for spec in builtin_components()
        if "policy.reference_stats" in spec.provides
    }
    assert providers == {"tiny_diffusion", "sd3_tempflow"}


def test_only_builtins_module_owns_lookup_and_manifest():
    import visual_rl.builtins as builtins_module
    import visual_rl.core.components as descriptions

    assert callable(builtins_module.builtin_components)
    assert callable(builtins_module.get_builtin_component)
    assert not hasattr(descriptions, "builtin_components")
    assert not hasattr(descriptions, "get_builtin_component")
    assert builtins_module.__all__ == [
        "builtin_components",
        "get_builtin_component",
    ]


def test_lookup_round_trips_and_rejects_retired_names():
    for spec in builtin_components():
        assert get_builtin_component(spec.kind, spec.name) is spec
    for kind, name in RETIRED_ALIASES:
        with pytest.raises(UnknownComponentError):
            get_builtin_component(kind, name)
    with pytest.raises(ComponentError, match="Unknown component kind"):
        get_builtin_component("provider", "anything")


def test_unknown_error_lists_only_same_kind_canonical_names():
    with pytest.raises(UnknownComponentError) as excinfo:
        get_builtin_component("model", "missing")
    assert excinfo.value.available == tuple(
        spec.name for spec in builtin_components() if spec.kind == "model"
    )


def test_dependencies_are_bare_import_names():
    for spec in builtin_components():
        assert spec.dependencies
        assert all(
            dependency
            and not set(dependency).intersection("><=~![]; ")
            for dependency in spec.dependencies
        )
