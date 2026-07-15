"""Contract tests for layered VisualRL configuration resolution."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import visual_rl.configs.resolver as resolver_module
from visual_rl.configs import (
    ConfigDocument,
    ExperimentSpec,
    KeyOverride,
    SourceRef,
    VisualRLConfig,
    config_from_dict,
    load_config,
    read_experiment_spec,
    resolve_experiment,
    validate_config,
)


def _document(
    kind: str, values: dict, base_dir: Path, name: str | None = None
) -> ConfigDocument:
    return ConfigDocument(
        values,
        SourceRef(kind=kind, name=name or kind, base_dir=base_dir),
    )


def _winning_tracking_paths(value, prefix=""):
    if isinstance(value, Mapping) and value:
        paths = set()
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.update(_winning_tracking_paths(child, child_prefix))
        return paths
    return {prefix} if prefix else set()


def _assert_tracking_invariant(values, provenance, base_dirs=None):
    winning_paths = _winning_tracking_paths(values)
    assert set(provenance).issubset(winning_paths)
    if base_dirs is not None:
        assert set(base_dirs) == set(provenance)


def test_layer_order_and_leaf_provenance(tmp_path):
    preset = _document(
        "preset",
        {
            "run_name": "preset",
            "seed": 1,
            "rollout": {"winner": "preset", "preset_only": True},
        },
        tmp_path,
    )
    recipe = _document(
        "recipe",
        {"seed": 2, "rollout": {"winner": "recipe", "recipe_only": True}},
        tmp_path,
    )
    profile = _document(
        "profile",
        {"seed": 3, "rollout": {"winner": "profile", "profile_only": True}},
        tmp_path,
    )
    user = _document(
        "user",
        {"seed": 4, "rollout": {"winner": "user", "user_only": True}},
        tmp_path,
    )
    set_source = SourceRef("set", "second set", tmp_path)
    explicit = _document(
        "explicit",
        {
            "seed": 7,
            "rollout": {"winner": "explicit", "explicit_only": True},
            "paths": {"output_dir": "explicit-runs"},
        },
        tmp_path,
    )

    resolved = resolve_experiment(
        ExperimentSpec(
            preset=preset,
            recipe=recipe,
            profile=profile,
            user=user,
            set_overrides=(
                KeyOverride("seed", 5),
                KeyOverride("seed", 6, source=set_source),
                KeyOverride("train.max_steps", 8, source=set_source),
            ),
            explicit=explicit,
            context_dir=tmp_path,
        )
    )

    assert resolved.config.seed == 7
    assert resolved.config.train.max_steps == 8
    assert resolved.config.rollout == {
        "winner": "explicit",
        "preset_only": True,
        "recipe_only": True,
        "profile_only": True,
        "user_only": True,
        "explicit_only": True,
    }
    assert resolved.config.paths.output_dir == str(tmp_path / "explicit-runs")
    assert resolved.provenance["seed"] == explicit.source
    assert resolved.provenance["train.max_steps"] == set_source
    assert resolved.provenance["rollout.preset_only"] == preset.source
    assert resolved.provenance["rollout.recipe_only"] == recipe.source
    assert resolved.provenance["rollout.profile_only"] == profile.source
    assert resolved.provenance["rollout.user_only"] == user.source
    assert resolved.provenance["sample.batch_size"].kind == "schema"
    assert not hasattr(resolved.config, "provenance")


def test_replace_defaults_prunes_schema_reward_leaves_individually(tmp_path):
    preset = _document(
        "preset",
        {
            "run_name": "replace-defaults",
            "rewards": {
                "replace_defaults": True,
                "weights": {"quality": 1.0},
                "clients": {"mock": {"version": "preset-v1"}},
            },
        },
        tmp_path,
    )
    user = _document(
        "user",
        {"rewards": {"clients": {"mock": {"mode": "user"}}}},
        tmp_path,
    )

    resolved = resolve_experiment(
        ExperimentSpec(preset=preset, user=user, context_dir=tmp_path)
    )

    assert resolved.config.rewards.weights == {"quality": 1.0}
    assert resolved.config.rewards.clients == {
        "mock": {"version": "preset-v1", "mode": "user"}
    }
    assert "rewards.clients.mock.name" not in resolved.provenance


def test_replace_defaults_handles_schema_empty_mappings_without_stale_tracking(
    tmp_path, monkeypatch
):
    defaults = resolver_module._schema_defaults()
    defaults["rewards"]["clients"].update(
        {
            "schema_only_empty": {},
            "schema_nested_empty": {"params": {}},
            "explicit_schema_empty": {},
        }
    )
    monkeypatch.setattr(resolver_module, "_schema_defaults", lambda: deepcopy(defaults))

    resolved = resolve_experiment(
        ExperimentSpec(
            user={
                "run_name": "schema-empty-rewards",
                "rewards": {
                    "replace_defaults": True,
                    "clients": {
                        "explicit_schema_empty": {},
                        "user_empty": {},
                    },
                },
            },
            context_dir=tmp_path,
        )
    )

    assert resolved.config.rewards.clients == {
        "explicit_schema_empty": {},
        "user_empty": {},
    }
    assert resolved.provenance["rewards.clients.explicit_schema_empty"].kind == "user"
    assert resolved.provenance["rewards.clients.user_empty"].kind == "user"
    assert not any(
        path.startswith(
            (
                "rewards.clients.mock",
                "rewards.clients.schema_only_empty",
                "rewards.clients.schema_nested_empty",
            )
        )
        for path in resolved.provenance
    )


def test_reward_prune_synchronizes_provenance_and_base_dir_tracking(tmp_path):
    schema_source = SourceRef("schema", "defaults", tmp_path)
    user_source = SourceRef("user", "explicit", tmp_path)
    values = {
        "rewards": {
            "replace_defaults": True,
            "weights": {"schema_score": 1.0, "user_score": 2.0},
            "clients": {
                "schema_empty": {},
                "schema_nested": {"params": {}},
                "user_empty": {},
            },
        }
    }
    provenance = {
        "rewards.weights.schema_score": schema_source,
        "rewards.weights.user_score": user_source,
        "rewards.clients.schema_empty": schema_source,
        "rewards.clients.schema_nested.params": schema_source,
        "rewards.clients.user_empty": user_source,
    }
    base_dirs = {path: tmp_path for path in provenance}

    resolver_module._drop_schema_reward_defaults(
        values,
        provenance=provenance,
        base_dirs=base_dirs,
    )

    assert values["rewards"]["weights"] == {"user_score": 2.0}
    assert values["rewards"]["clients"] == {"user_empty": {}}
    expected_paths = {
        "rewards.weights.user_score",
        "rewards.clients.user_empty",
    }
    assert set(provenance) == expected_paths
    assert set(base_dirs) == expected_paths


@pytest.mark.parametrize("layer", ["preset", "user"])
def test_schema_fill_clears_explicit_empty_nonleaf_tracking_when_not_replacing(
    tmp_path, monkeypatch, layer
):
    captured_base_dirs = {}
    normalize_paths = resolver_module._normalize_paths

    def capture_normalized_tracking(values, *, provenance, base_dirs):
        captured_base_dirs.update(base_dirs)
        normalize_paths(values, provenance=provenance, base_dirs=base_dirs)

    monkeypatch.setattr(
        resolver_module,
        "_normalize_paths",
        capture_normalized_tracking,
    )
    document = _document(
        layer,
        {
            "run_name": f"empty-{layer}-mapping",
            "rewards": {
                "replace_defaults": False,
                "clients": {"mock": {}},
            },
        },
        tmp_path,
    )
    resolved = resolve_experiment(
        ExperimentSpec(**{layer: document}, context_dir=tmp_path)
    )

    assert resolved.config.rewards.clients["mock"] == {"name": "mock"}
    assert "rewards.clients.mock" not in resolved.provenance
    assert resolved.provenance["rewards.clients.mock.name"].kind == "schema"
    assert "rewards.clients.mock" not in captured_base_dirs
    assert captured_base_dirs["rewards.clients.mock.name"] == tmp_path
    _assert_tracking_invariant(
        resolved.values,
        resolved.provenance,
        captured_base_dirs,
    )


def test_tracking_normalization_keeps_only_winning_leaves_and_empty_mappings(tmp_path):
    source = SourceRef("user", "tracking", tmp_path)
    values = {
        "leaf": 1,
        "empty": {},
        "branch": {"child": 2},
    }
    provenance = {
        "leaf": source,
        "empty": source,
        "branch": source,
        "branch.child": source,
        "missing": source,
    }
    base_dirs = {path: tmp_path for path in provenance}
    base_dirs["orphan"] = tmp_path

    resolver_module._normalize_tracking(
        values,
        provenance=provenance,
        base_dirs=base_dirs,
    )

    assert set(provenance) == {"leaf", "empty", "branch.child"}
    _assert_tracking_invariant(values, provenance, base_dirs)


def test_deep_merge_list_replacement_and_explicit_none(tmp_path):
    preset = _document(
        "preset",
        {
            "run_name": "merge",
            "model": {"extra": {"nested": {"left": 1, "right": 2}}},
            "dataset": {"prompts": ["one", "two"]},
            "train": {"lora_path": "weights/adapter"},
        },
        tmp_path / "preset",
    )
    user = _document(
        "user",
        {
            "model": {"extra": {"nested": {"right": 9}}},
            "dataset": {"prompts": []},
            "train": {"lora_path": None},
        },
        tmp_path / "user",
    )

    resolved = resolve_experiment(
        ExperimentSpec(preset=preset, user=user, context_dir=tmp_path)
    )

    assert resolved.config.model.extra["nested"] == {"left": 1, "right": 9}
    assert resolved.config.dataset.prompts == []
    assert resolved.config.train.lora_path is None
    assert resolved.provenance["model.extra.nested.left"] == preset.source
    assert resolved.provenance["model.extra.nested.right"] == user.source
    assert resolved.provenance["dataset.prompts"] == user.source
    assert resolved.provenance["train.lora_path"] == user.source


def test_ordered_set_overrides_and_non_mapping_traversal(tmp_path):
    set_source = SourceRef("set", "ordered", tmp_path)
    spec = ExperimentSpec(
        user={"run_name": "ordered", "model": {"extra": {"left": 1}}},
        set_overrides=(
            KeyOverride("model.extra", {"middle": 2}),
            KeyOverride("model.extra", {"right": 3}),
            KeyOverride("train.max_steps", 4),
            KeyOverride("train.max_steps", 5, source=set_source),
        ),
        context_dir=tmp_path,
    )

    resolved = resolve_experiment(spec)

    assert resolved.config.model.extra == {"left": 1, "middle": 2, "right": 3}
    assert resolved.config.train.max_steps == 5
    assert resolved.provenance["train.max_steps"] == set_source

    invalid = ExperimentSpec(
        user={"run_name": "invalid-set"},
        set_overrides=(KeyOverride("model.name.child", "bad"),),
        context_dir=tmp_path,
    )
    with pytest.raises(TypeError, match="is not a mapping"):
        resolve_experiment(invalid)


def test_override_and_explicit_paths_use_winning_source_bases(tmp_path):
    context_dir = tmp_path / "context"
    override_dir = tmp_path / "override-source"
    explicit_dir = tmp_path / "explicit-source"
    override_source = SourceRef("set", "path override", override_dir)
    explicit = _document(
        "explicit",
        {"runner": {"rollout_cache_dir": "cache/rollouts"}},
        explicit_dir,
    )

    resolved = resolve_experiment(
        ExperimentSpec(
            user={"run_name": "source-bases"},
            set_overrides=(
                KeyOverride("dataset.path", "data/train.jsonl", override_source),
                KeyOverride("paths.output_dir", "runs/default-source"),
            ),
            explicit=explicit,
            context_dir=context_dir,
        )
    )

    assert resolved.config.dataset.path == str(override_dir / "data/train.jsonl")
    assert resolved.config.paths.output_dir == str(context_dir / "runs/default-source")
    assert resolved.config.runner.rollout_cache_dir == str(
        explicit_dir / "cache/rollouts"
    )
    assert resolved.provenance["dataset.path"] == override_source
    assert resolved.provenance["runner.rollout_cache_dir"] == explicit.source


def test_override_with_source_but_no_base_fails_closed(tmp_path):
    spec = ExperimentSpec(
        user={"run_name": "missing-override-base"},
        set_overrides=(
            KeyOverride(
                "dataset.path",
                "relative.jsonl",
                SourceRef("set", "unanchored override"),
            ),
        ),
        context_dir=tmp_path / "context",
    )

    with pytest.raises(ValueError, match="has no base_dir"):
        resolve_experiment(spec)


def test_descriptor_base_dir_is_relative_to_envelope_not_cwd(tmp_path, monkeypatch):
    envelope_dir = tmp_path / "experiment"
    other_dir = tmp_path / "other-cwd"
    envelope_dir.mkdir()
    other_dir.mkdir()
    envelope = envelope_dir / "descriptor.yaml"
    envelope.write_text(
        yaml.safe_dump(
            {
                "context_dir": "../context",
                "config": {
                    "values": {
                        "run_name": "descriptor-base",
                        "dataset": {"path": "data/train.jsonl"},
                    },
                    "name": "inline user descriptor",
                    "base_dir": "source-assets",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(other_dir)
    resolved = resolve_experiment(read_experiment_spec(envelope))

    expected_base = envelope_dir / "source-assets"
    assert resolved.config.dataset.path == str(expected_base / "data/train.jsonl")
    assert resolved.provenance["dataset.path"].name == "inline user descriptor"
    assert resolved.provenance["dataset.path"].base_dir == expected_base


def test_file_sources_keep_distinct_bases_and_ignore_later_cwd(tmp_path, monkeypatch):
    preset_dir = tmp_path / "preset"
    profile_dir = tmp_path / "profile"
    envelope_dir = tmp_path / "experiment"
    other_dir = tmp_path / "other"
    for directory in (preset_dir, profile_dir, envelope_dir, other_dir):
        directory.mkdir()

    (preset_dir / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": "data/train.jsonl"},
                "paths": {"output_dir": "preset-runs"},
            }
        ),
        encoding="utf-8",
    )
    (profile_dir / "gpu.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_path": "checkpoints/model",
                    "extra": {"repo_root": "repos/world-r1"},
                }
            }
        ),
        encoding="utf-8",
    )
    envelope = envelope_dir / "experiment.yaml"
    envelope.write_text(
        yaml.safe_dump(
            {
                "preset": "../preset/base.yaml",
                "profile": "../profile/gpu.yaml",
                "config": {"run_name": "paths"},
                "set_overrides": [{"path": "paths.output_dir", "value": "set-runs"}],
                "explicit": {
                    "runner": {"rollout_cache_dir": "rollout-cache"},
                    "model": {"extra": {"lora_path": "https://example/lora"}},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    spec = read_experiment_spec(envelope)
    monkeypatch.chdir(other_dir)
    resolved = resolve_experiment(spec)

    assert resolved.config.dataset.path == str(preset_dir / "data/train.jsonl")
    assert resolved.config.model.model_path == str(profile_dir / "checkpoints/model")
    assert resolved.config.model.extra["repo_root"] == str(
        profile_dir / "repos/world-r1"
    )
    assert resolved.config.paths.output_dir == str(envelope_dir / "set-runs")
    assert resolved.config.runner.rollout_cache_dir == str(
        envelope_dir / "rollout-cache"
    )
    assert resolved.config.model.extra["lora_path"] == "https://example/lora"
    assert resolved.provenance["dataset.path"].kind == "preset"
    assert resolved.provenance["model.model_path"].kind == "profile"
    assert resolved.provenance["paths.output_dir"].kind == "set"


def test_legacy_yaml_unknown_fields_and_final_pair_validation(tmp_path):
    legacy = tmp_path / "legacy_config.yaml"
    legacy.write_text(
        yaml.safe_dump({"seed": 17, "paths": {"output_dir": "runs/legacy"}}),
        encoding="utf-8",
    )

    loaded = load_config(legacy)
    assert loaded.run_name == "legacy_config"
    assert loaded.seed == 17
    assert loaded.paths.output_dir == str(tmp_path / "runs/legacy")

    with pytest.raises(ValueError, match="Unknown fields"):
        config_from_dict({"run_name": "unknown", "not_a_field": True})

    repaired = resolve_experiment(
        ExperimentSpec(
            preset={
                "run_name": "repaired",
                "sample": {"name": "single_step"},
                "algorithm": "temporarily-invalid",
            },
            explicit={"algorithm": {"name": "flash_grpo"}},
            context_dir=tmp_path,
        )
    )
    assert repaired.config.algorithm.name == "flash_grpo"
    assert repaired.config.algorithm.clip_range == pytest.approx(0.001)

    invalid_pair = ExperimentSpec(
        user={
            "run_name": "invalid-pair",
            "sample": {"name": "single_step"},
            "algorithm": {"name": "grpo"},
        },
        context_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="Incompatible config"):
        resolve_experiment(invalid_pair)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train": {"precision": "tf32"}}, "train.precision"),
        (
            {"train": {"update_microbatch_size": 0}},
            "train.update_microbatch_size",
        ),
        (
            {"runner": {"checkpoint_keep_last": -1}},
            "runner.checkpoint_keep_last",
        ),
        (
            {"runner": {"rollout_cache_max_bytes": -1}},
            "runner.rollout_cache_max_bytes",
        ),
    ],
)
def test_runtime_controls_fail_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        config_from_dict({"run_name": "invalid-runtime", **overrides})


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("sample", "batch_size", 1),
        ("sample", "num_steps", 1),
        ("sample", "guidance_scale", -2.0),
        ("sample", "guidance_scale", 0.0),
        ("train", "learning_rate", 0.0),
        ("train", "max_steps", 1),
        ("train", "adam_beta1", 0.0),
        ("train", "adam_beta1", 0.999999),
        ("train", "adam_beta2", 0.0),
        ("train", "adam_beta2", 0.999999),
        ("train", "adam_weight_decay", 0.0),
        ("train", "adam_epsilon", 1e-12),
        ("train", "max_grad_norm", None),
        ("train", "max_grad_norm", 0.5),
        ("algorithm", "clip_range", 0.0),
        ("algorithm", "clip_range", 1.0),
        ("algorithm", "adv_clip_max", 0.0),
        ("algorithm", "beta", 0.0),
        ("algorithm", "advantage_epsilon", 1e-12),
    ],
)
def test_core_numeric_controls_accept_supported_boundaries(section, field, value):
    config = config_from_dict(
        {"run_name": "valid-numeric-control", section: {field: value}}
    )

    assert getattr(getattr(config, section), field) == value


def test_public_validator_accepts_a_legal_direct_config_without_mutation() -> None:
    config = VisualRLConfig(run_name="direct-valid")
    before = deepcopy(config)

    assert validate_config(config) is None
    assert config == before


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("sample", "batch_size", 0, "sample.batch_size"),
        ("algorithm", "clip_range", 1.01, "algorithm.clip_range"),
        ("train", "learning_rate", float("nan"), "train.learning_rate"),
        ("train", "adam_beta2", 1.0, "train.adam_beta2"),
    ],
)
def test_public_validator_rechecks_mutated_direct_config(
    section, field, value, message
) -> None:
    config = VisualRLConfig(run_name="direct-mutated")
    validate_config(config)
    setattr(getattr(config, section), field, value)

    with pytest.raises((TypeError, ValueError), match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("sample", "batch_size", 0),
        ("sample", "batch_size", -1),
        ("sample", "num_steps", 0),
        ("sample", "num_steps", -1),
        ("sample", "guidance_scale", float("nan")),
        ("sample", "guidance_scale", float("inf")),
        ("sample", "guidance_scale", float("-inf")),
        ("train", "learning_rate", -1e-4),
        ("train", "learning_rate", float("nan")),
        ("train", "learning_rate", float("inf")),
        ("train", "max_steps", 0),
        ("train", "max_steps", -1),
        ("train", "adam_beta1", -1e-4),
        ("train", "adam_beta1", 1.0),
        ("train", "adam_beta1", float("nan")),
        ("train", "adam_beta2", -1e-4),
        ("train", "adam_beta2", 1.0),
        ("train", "adam_beta2", float("inf")),
        ("train", "adam_weight_decay", -1e-4),
        ("train", "adam_weight_decay", float("nan")),
        ("train", "adam_epsilon", 0.0),
        ("train", "adam_epsilon", -1e-8),
        ("train", "adam_epsilon", float("inf")),
        ("train", "max_grad_norm", 0.0),
        ("train", "max_grad_norm", -1.0),
        ("train", "max_grad_norm", float("nan")),
        ("train", "max_grad_norm", float("inf")),
        ("algorithm", "clip_range", -1e-4),
        ("algorithm", "clip_range", 1.0001),
        ("algorithm", "clip_range", float("nan")),
        ("algorithm", "clip_range", float("inf")),
        ("algorithm", "adv_clip_max", -1e-4),
        ("algorithm", "adv_clip_max", float("nan")),
        ("algorithm", "adv_clip_max", float("inf")),
        ("algorithm", "beta", -1e-4),
        ("algorithm", "beta", float("nan")),
        ("algorithm", "beta", float("inf")),
        ("algorithm", "advantage_epsilon", 0.0),
        ("algorithm", "advantage_epsilon", -1e-6),
        ("algorithm", "advantage_epsilon", float("nan")),
    ],
)
def test_core_numeric_controls_reject_invalid_values(section, field, value):
    path = f"{section}.{field}"

    with pytest.raises(ValueError, match=path):
        config_from_dict(
            {"run_name": "invalid-numeric-control", section: {field: value}}
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("sample", "batch_size"),
        ("sample", "num_steps"),
        ("sample", "guidance_scale"),
        ("train", "learning_rate"),
        ("train", "max_steps"),
        ("train", "adam_beta1"),
        ("train", "adam_beta2"),
        ("train", "adam_weight_decay"),
        ("train", "adam_epsilon"),
        ("train", "max_grad_norm"),
        ("algorithm", "clip_range"),
        ("algorithm", "adv_clip_max"),
        ("algorithm", "beta"),
        ("algorithm", "advantage_epsilon"),
    ],
)
def test_core_numeric_controls_reject_bool(section, field):
    path = f"{section}.{field}"

    with pytest.raises(TypeError, match=path):
        config_from_dict(
            {"run_name": "invalid-bool-control", section: {field: True}}
        )


def test_package_preset_and_missing_path_base(tmp_path):
    envelope = tmp_path / "package.yaml"
    envelope.write_text(
        yaml.safe_dump({"preset": "flash_tiny_single_step"}),
        encoding="utf-8",
    )

    resolved = resolve_experiment(read_experiment_spec(envelope))
    assert resolved.config.sample.name == "single_step"
    assert resolved.config.paths.output_dir == str(tmp_path / "runs/default")
    assert resolved.provenance["sample.name"].kind == "preset"
    assert resolved.provenance["paths.output_dir"].kind == "schema"

    no_base = ConfigDocument(
        {
            "run_name": "no-base",
            "dataset": {"path": "relative.jsonl"},
            "paths": {"output_dir": str(tmp_path / "absolute-run")},
        },
        SourceRef("user", "memory"),
    )
    with pytest.raises(ValueError, match="has no base_dir"):
        resolve_experiment(ExperimentSpec(user=no_base))


def test_resolver_is_pure_and_does_not_mutate_inputs(tmp_path, monkeypatch):
    values = {
        "run_name": "pure",
        "model": {"extra": {"repo_root": "repos/reference"}},
        "paths": {"output_dir": "runs/pure"},
    }
    original = deepcopy(values)
    document = _document("user", values, tmp_path)
    spec = ExperimentSpec(user=document, context_dir=tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("resolver attempted filesystem mutation or probing")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)

    first = resolve_experiment(spec)
    second = resolve_experiment(spec)

    assert values == original
    assert document.values == original
    assert first.values == second.values
    assert first.provenance == second.provenance
    assert first.config is not second.config
    assert first.config.model.extra["repo_root"] == str(tmp_path / "repos/reference")


def test_resolved_experiment_config_is_the_single_truth(tmp_path):
    resolved = resolve_experiment(
        ExperimentSpec(user={"run_name": "single-truth"}, context_dir=tmp_path)
    )

    first_values = resolved.values
    first_resolved = resolved.resolved
    first_values["seed"] = -1
    first_resolved["model"]["extra"]["detached"] = True

    assert resolved.config.seed == 42
    assert "detached" not in resolved.config.model.extra

    resolved.config.seed = 73
    resolved.config.model.extra["live"] = True
    assert resolved.values["seed"] == 73
    assert resolved.resolved["model"]["extra"]["live"] is True
    assert resolved.values is not resolved.values


def test_reward_replace_defaults_only_removes_schema_entries(tmp_path):
    resolved = resolve_experiment(
        ExperimentSpec(
            user={
                "run_name": "replace-schema-reward",
                "rewards": {
                    "replace_defaults": True,
                    "weights": {"quality": 0.5},
                    "clients": {},
                },
            },
            context_dir=tmp_path,
        )
    )

    assert resolved.config.rewards.weights == {"quality": 0.5}
    assert resolved.config.rewards.clients == {}
    assert "rewards.weights.mock" not in resolved.provenance
    assert not any(
        key.startswith("rewards.clients.mock") for key in resolved.provenance
    )

    layered = resolve_experiment(
        ExperimentSpec(
            preset={
                "run_name": "keep-non-schema-reward",
                "rewards": {
                    "weights": {"preset_reward": 1.0},
                    "clients": {"preset_reward": {"name": "preset_reward"}},
                },
            },
            explicit={"rewards": {"replace_defaults": True}},
            context_dir=tmp_path,
        )
    )

    assert layered.config.rewards.weights == {"preset_reward": 1.0}
    assert set(layered.config.rewards.clients) == {"preset_reward"}


def test_importing_configs_in_fresh_process_has_no_runtime_side_effects():
    script = """
import sys
import visual_rl.configs
from visual_rl.core.registry import (
    ALGORITHMS,
    FEEDBACK_PROVIDERS,
    MODEL_ADAPTERS,
    OPTIMIZER_PLUGINS,
    REWARD_CLIENTS,
    ROLLOUT_ENGINES,
)

for prefix in (
    "visual_rl.feedback",
    "visual_rl.optimizers",
    "visual_rl.rollout",
    "visual_rl.runner",
):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix

for registry in (
    ALGORITHMS,
    FEEDBACK_PROVIDERS,
    MODEL_ADAPTERS,
    OPTIMIZER_PLUGINS,
    REWARD_CLIENTS,
    ROLLOUT_ENGINES,
):
    assert registry.keys() == [], (registry.name, registry.keys())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )


def test_lazy_top_level_public_exports_remain_compatible():
    from visual_rl import (
        ExperimentRunner,
        FeedbackProvider,
        ModelAdapter,
        OptimizerPlugin,
        RolloutEngine,
    )

    assert ExperimentRunner.__name__ == "ExperimentRunner"
    assert FeedbackProvider.__name__ == "FeedbackProvider"
    assert ModelAdapter.__name__ == "ModelAdapter"
    assert OptimizerPlugin.__name__ == "OptimizerPlugin"
    assert RolloutEngine.__name__ == "RolloutEngine"


def test_package_presets_contain_only_portable_semantics():
    preset_dir = Path(__file__).resolve().parents[1] / "visual_rl/configs/presets"
    forbidden_keys = {
        "api_key",
        "auth_token",
        "checkpoint",
        "lora_path",
        "model_path",
        "output_dir",
        "pretrained_model",
        "repo_root",
        "resume_from",
        "token",
        "url",
        "world_r1_root",
    }

    published_local_urls = {
        "world_r1_wan_bounded.yaml.rewards.clients.reward_general.url": (
            "http://127.0.0.1:8090/"
        ),
        "world_r1_wan_bounded.yaml.rewards.clients.reward_3d.url": (
            "http://127.0.0.1:8089/"
        ),
    }

    def walk(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                dotted = f"{path}.{key}" if path else str(key)
                if dotted in published_local_urls:
                    assert child == published_local_urls[dotted]
                    continue
                assert key not in forbidden_keys, dotted
                yield from walk(child, dotted)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, f"{path}[{index}]")

    presets = sorted(preset_dir.glob("*.yaml"))
    assert presets
    for preset in presets:
        values = yaml.safe_load(preset.read_text(encoding="utf-8")) or {}
        list(walk(values, preset.name))
