"""Golden tests for the frozen builtin component manifest (v0.7 W02).

The capability/dependency matrices below are verbatim copies of the master
plan stage 2.5 (capability matrix) and stage 2.2 (dependency table). The
manifest must equal them exactly, modulo the explicitly deferred
``wan_flash`` entry, which only joins when the Wan split lands in the atomic
cutover (no ``WanFlashAdapter`` class exists yet).
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from visual_rl.core.components import (
    CAPABILITY_OWNER,
    CAPABILITY_VOCABULARY,
    COMPONENT_KINDS,
    ComponentSpec,
    builtin_components,
    get_builtin_component,
)
from visual_rl.errors import ComponentError, UnknownComponentError

# Master plan stage 2.5 capability matrix + stage 2.2 dependency table,
# transcribed row by row. Order of each dependencies tuple is significant.
GOLDEN_MATRIX = {
    ("model", "tiny_diffusion"): {
        "provides": {
            "media.image",
            "sampling.full_trajectory",
            "sampling.single_step",
            "sampling.branching",
        },
        "requires": set(),
        "dependencies": ("torch",),
    },
    ("model", "sd3_tempflow"): {
        "provides": {
            "media.image",
            "sampling.full_trajectory",
            "sampling.branching",
        },
        "requires": set(),
        "dependencies": (
            "torch",
            "diffusers",
            "transformers",
            "peft",
            "sentencepiece",
            "google.protobuf",
        ),
    },
    ("model", "wan_world_r1"): {
        "provides": {
            "media.video",
            "sampling.full_trajectory",
            "conditioning.camera",
        },
        "requires": set(),
        "dependencies": (
            "torch",
            "diffusers",
            "transformers",
            "peft",
            "sentencepiece",
            "google.protobuf",
            "einops",
            "rp",
        ),
    },
    ("rollout", "full_trajectory"): {
        "provides": {"rollout.full_trajectory"},
        "requires": {"sampling.full_trajectory"},
        "dependencies": ("torch",),
    },
    ("rollout", "single_step"): {
        "provides": {"rollout.single_step"},
        "requires": {"sampling.single_step"},
        "dependencies": ("torch",),
    },
    ("rollout", "branching"): {
        "provides": {"rollout.branching"},
        "requires": {"sampling.branching"},
        "dependencies": ("torch",),
    },
    ("reward", "mock"): {
        "provides": set(),
        "requires": set(),
        "dependencies": ("numpy",),
    },
    ("reward", "prompt_color"): {
        "provides": set(),
        "requires": {"media.image"},
        "dependencies": ("numpy",),
    },
    ("reward", "prompt_color_margin"): {
        "provides": set(),
        "requires": {"media.image"},
        "dependencies": ("numpy",),
    },
    ("reward", "prompt_color_guarded"): {
        "provides": set(),
        "requires": {"media.image"},
        "dependencies": ("numpy",),
    },
    ("reward", "reward_general"): {
        "provides": set(),
        "requires": {"media.video"},
        "dependencies": ("numpy", "PIL", "requests"),
    },
    ("reward", "reward_3d"): {
        "provides": set(),
        "requires": {"media.video", "conditioning.camera"},
        "dependencies": ("numpy", "PIL", "requests"),
    },
    ("algorithm", "grpo"): {
        "provides": set(),
        "requires": {"rollout.full_trajectory"},
        "dependencies": ("torch",),
    },
    ("algorithm", "flash_grpo"): {
        "provides": set(),
        "requires": {"media.video", "rollout.single_step"},
        "dependencies": ("torch",),
    },
    ("algorithm", "tempflow_grpo"): {
        "provides": set(),
        "requires": {"media.image", "rollout.branching"},
        "dependencies": ("torch",),
    },
}

# Plan matrix rows deferred to the Wan-split cutover. ``wan_flash`` currently
# has no factory class, so it must resolve as unknown in this phase; this set
# documents exactly which golden rows are pending and may never grow silently.
DEFERRED_MATRIX = {
    ("model", "wan_flash"): {
        "provides": {"media.video", "sampling.single_step"},
        "requires": set(),
        "dependencies": (
            "torch",
            "diffusers",
            "transformers",
            "peft",
            "sentencepiece",
            "google.protobuf",
        ),
    },
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
    ("model", "wan_flash"),  # deferred, not yet a selectable builtin
    ("rollout", "flash_single_step"),
    ("reward", "remote_pickle"),
    ("reward", "pickscore"),
    ("reward", "video_hpsv3"),
    ("reward", "reward_router"),
)

ILLEGAL_KINDS = (
    "optimizer",
    "feedback_provider",
    "provider",
    "plugin",
    "objective",
    "runner",
    "artifact",
)


def _manifest_matrix():
    return {
        (spec.kind, spec.name): {
            "provides": set(spec.provides),
            "requires": set(spec.requires),
            "dependencies": spec.dependencies,
        }
        for spec in builtin_components()
    }


def test_manifest_kinds_are_exactly_the_four_legal_kinds():
    kinds = {spec.kind for spec in builtin_components()}
    assert kinds == {"model", "rollout", "reward", "algorithm"}
    assert COMPONENT_KINDS == ("model", "rollout", "reward", "algorithm")


def test_manifest_is_a_fixed_order_immutable_tuple():
    first = builtin_components()
    second = builtin_components()
    assert isinstance(first, tuple)
    assert first is second
    assert all(isinstance(spec, ComponentSpec) for spec in first)


def test_kind_name_pairs_are_unique():
    pairs = [(spec.kind, spec.name) for spec in builtin_components()]
    assert len(pairs) == len(set(pairs))


def test_factory_is_unique_within_each_kind():
    by_kind: dict[str, dict[type, str]] = {}
    for spec in builtin_components():
        seen = by_kind.setdefault(spec.kind, {})
        assert spec.factory not in seen, (
            f"factory {spec.factory.__name__} is registered twice in kind "
            f"{spec.kind}: {seen[spec.factory]!r} and {spec.name!r}"
        )
        seen[spec.factory] = spec.name


def test_component_spec_is_frozen_and_has_no_parallel_fields():
    field_names = {item.name for item in dataclasses.fields(ComponentSpec)}
    assert field_names == {
        "kind",
        "name",
        "factory",
        "provides",
        "requires",
        "dependencies",
    }
    spec = builtin_components()[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"
    for spec in builtin_components():
        assert isinstance(spec.provides, frozenset)
        assert isinstance(spec.requires, frozenset)
        assert isinstance(spec.dependencies, tuple)
        assert inspect.isclass(spec.factory)


def test_capability_and_dependency_golden_matrix_matches_plan_verbatim():
    assert _manifest_matrix() == GOLDEN_MATRIX
    assert set(GOLDEN_MATRIX).isdisjoint(DEFERRED_MATRIX)
    # The union with the deferred rows is exactly the plan's full matrix.
    assert set(GOLDEN_MATRIX) | set(DEFERRED_MATRIX) == set(GOLDEN_MATRIX) | {
        ("model", "wan_flash")
    }


def test_deferred_entries_resolve_as_unknown_for_now():
    for kind, name in DEFERRED_MATRIX:
        with pytest.raises(ComponentError):
            get_builtin_component(kind, name)


def test_capability_vocabulary_and_owner_table_match_plan():
    assert CAPABILITY_VOCABULARY == frozenset(EXPECTED_CAPABILITY_OWNER)
    assert CAPABILITY_OWNER == EXPECTED_CAPABILITY_OWNER


def test_provides_and_requires_stay_inside_the_closed_vocabulary():
    for spec in builtin_components():
        assert spec.provides <= CAPABILITY_VOCABULARY
        assert spec.requires <= CAPABILITY_VOCABULARY


def test_provides_only_come_from_the_owner_kind():
    for spec in builtin_components():
        for capability in spec.provides:
            assert CAPABILITY_OWNER[capability] == spec.kind


def test_each_builtin_model_declares_exactly_one_media_capability():
    for spec in builtin_components():
        if spec.kind != "model":
            continue
        media = {item for item in spec.provides if item.startswith("media.")}
        assert len(media) == 1, f"{spec.name} declares {sorted(media)}"


def test_no_component_prematurely_provides_reference_stats():
    # policy.reference_stats may only be declared after stage 4 implements
    # and tests it (master plan stage 2.1).
    for spec in builtin_components():
        assert "policy.reference_stats" not in spec.provides


def test_dependencies_are_bare_import_names_without_versions():
    for spec in builtin_components():
        assert spec.dependencies, f"{spec.name} has empty dependencies"
        for dependency in spec.dependencies:
            assert isinstance(dependency, str)
            assert dependency
            assert not set(dependency) & set("><=~![]; "), dependency


def test_get_builtin_component_round_trips_every_spec():
    for spec in builtin_components():
        assert get_builtin_component(spec.kind, spec.name) is spec


def test_unknown_name_raises_component_error_listing_same_kind_names():
    with pytest.raises(UnknownComponentError) as excinfo:
        get_builtin_component("model", "does_not_exist")
    error = excinfo.value
    assert isinstance(error, ComponentError)
    assert error.kind == "model"
    assert error.name == "does_not_exist"
    assert error.available == tuple(
        spec.name for spec in builtin_components() if spec.kind == "model"
    )
    assert "tiny_diffusion" in str(error)


@pytest.mark.parametrize(("kind", "name"), RETIRED_ALIASES)
def test_retired_aliases_resolve_as_unknown(kind, name):
    with pytest.raises(ComponentError) as excinfo:
        get_builtin_component(kind, name)
    assert name not in getattr(excinfo.value, "available", ())


@pytest.mark.parametrize("kind", ILLEGAL_KINDS)
def test_illegal_component_kinds_are_rejected(kind):
    with pytest.raises(ComponentError, match="Unknown component kind"):
        get_builtin_component(kind, "anything")


def test_module_exposes_no_registration_surface():
    import visual_rl.core.components as components

    for banned in (
        "register",
        "add",
        "freeze",
        "snapshot",
        "override",
        "register_builtin_plugins",
    ):
        assert not hasattr(components, banned), banned
