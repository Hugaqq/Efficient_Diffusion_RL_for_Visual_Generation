"""Focused contracts for the sole v0.7 YAML/config resolution path."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from visual_rl.api import load
from visual_rl.configs.schema import VisualRLConfig
from visual_rl.core.types import FrozenMapping, to_plain_dict
from visual_rl.errors import ComponentError, ConfigError


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "configs" / "tiny_grpo.yaml"


def _tiny_mapping() -> dict:
    return yaml.safe_load(TINY.read_text(encoding="utf-8"))


def _write(tmp_path: Path, value: dict, *, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_tiny_fixture_resolves_to_one_frozen_canonical_config():
    config = load(TINY).resolve()

    assert isinstance(config, VisualRLConfig)
    assert config.schema_version == 1
    assert config.run.seed == 42
    assert config.dataset.prompts == ("a red cube", "a blue cube")
    assert config.dataset.path is None
    assert config.model.params == {"image_size": 16}
    assert config.rollout.params == {"num_steps": 2, "samples_per_prompt": 2}
    assert config.reward.components[0].params == {"mode": "prompt_media"}
    assert config.algorithm.params == {
        "adv_clip_max": 5.0,
        "beta": 0.0,
        "clip_range": 0.001,
    }
    assert config.artifacts.preview_samples_per_event == 0
    assert not hasattr(config.runtime, "rollout_cache")
    assert config.artifacts.output_dir == (
        TINY.parent / "runs" / "tiny-grpo"
    ).resolve()
    assert isinstance(config.model.params, FrozenMapping)
    with pytest.raises(FrozenInstanceError):
        config.run.seed = 7
    with pytest.raises(TypeError):
        config.model.params["image_size"] = 8


def test_to_plain_dict_uses_yaml_resume_key_and_plain_containers(tmp_path):
    values = _tiny_mapping()
    values["artifacts"]["output_dir"] = str(tmp_path / "run")
    values["resume"]["from"] = str(tmp_path / "run")
    config = load(_write(tmp_path, values)).resolve()

    plain = to_plain_dict(config)

    assert "from" in plain["resume"]
    assert "from_" not in plain["resume"]
    assert plain["resume"]["from"] == str((tmp_path / "run").resolve())
    assert isinstance(plain["reward"]["components"], list)
    assert isinstance(plain["model"]["params"], dict)
    assert "rollout_cache" not in plain["runtime"]


@pytest.mark.parametrize("value", [0, 1, 2])
def test_preview_samples_per_event_accepts_only_bounded_integer_values(
    tmp_path: Path,
    value: int,
) -> None:
    values = _tiny_mapping()
    values["artifacts"]["preview_samples_per_event"] = value

    config = load(_write(tmp_path, values)).resolve()

    assert config.artifacts.preview_samples_per_event == value


@pytest.mark.parametrize("value", [True, -1, 3])
def test_preview_samples_per_event_rejects_invalid_values(
    tmp_path: Path,
    value: object,
) -> None:
    values = _tiny_mapping()
    values["artifacts"]["preview_samples_per_event"] = value

    with pytest.raises(ConfigError, match="preview_samples_per_event"):
        load(_write(tmp_path, values)).resolve()


def test_removed_runtime_rollout_cache_is_rejected_as_unknown_config(
    tmp_path: Path,
) -> None:
    values = _tiny_mapping()
    values["runtime"]["rollout_cache"] = {
        "enabled": True,
        "keep_last": 1,
    }

    with pytest.raises(ConfigError, match="unknown keys.*rollout_cache"):
        load(_write(tmp_path, values)).resolve()


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "train"),
        (None, "sample"),
        (None, "runner"),
        (None, "paths"),
        (None, "plugins"),
        ("run", "name"),
        ("dataset", "split_name"),
        ("optimizer", "name"),
        ("runtime", "backend"),
        ("runtime", "world_size"),
    ],
)
def test_unknown_legacy_or_duplicate_owner_fields_fail_fast(
    tmp_path, section, key
):
    values = _tiny_mapping()
    target = values if section is None else values[section]
    target[key] = "legacy"

    with pytest.raises(ConfigError, match="unknown keys"):
        load(_write(tmp_path, values)).resolve()


def test_nested_duplicate_yaml_key_is_rejected_at_load(tmp_path):
    path = tmp_path / "duplicate.yaml"
    text = TINY.read_text(encoding="utf-8").replace(
        "  seed: 42\n",
        "  seed: 42\n  seed: 43\n",
        1,
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate mapping key.*seed"):
        load(path)


def test_relative_global_and_component_paths_share_yaml_directory(tmp_path):
    values = _tiny_mapping()
    values["model"] = {
        "name": "sd3_tempflow",
        "adapter_checkpoint": "adapters/warm-start",
        "params": {
            "checkpoint": "models/sd3",
            "reference_repo": "references/tempflow",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["to_q", "to_k"],
            "gradient_checkpointing": True,
            "guidance_scale": 4.5,
            "resolution": 64,
            "max_sequence_length": 32,
        },
    }
    values["rollout"] = {
        "name": "branching",
        "params": {
            "num_steps": 3,
            "branch_count": 2,
            "branch_timesteps": "auto",
        },
    }
    values["algorithm"] = {
        "name": "tempflow_grpo",
        "params": {"clip_range": 0.001, "adv_clip_max": 5.0},
        "advantage": {"epsilon": 1e-6},
    }
    path = _write(tmp_path, values)

    config = load(path).resolve()

    assert config.model.adapter_checkpoint == (
        tmp_path / "adapters" / "warm-start"
    ).resolve()
    assert config.model.params["checkpoint"] == (tmp_path / "models" / "sd3").resolve()
    assert config.model.params["reference_repo"] == (
        tmp_path / "references" / "tempflow"
    ).resolve()


@pytest.mark.parametrize("relationship", ["same", "inside", "contains"])
def test_reward_cache_and_output_directory_must_not_overlap(
    tmp_path, relationship
):
    values = _tiny_mapping()
    output = tmp_path / "run"
    cache = {
        "same": output,
        "inside": output / "reward-cache",
        "contains": tmp_path,
    }[relationship]
    values["artifacts"]["output_dir"] = str(output)
    values["reward"]["cache_dir"] = str(cache)

    with pytest.raises(ConfigError, match="must not overlap"):
        load(_write(tmp_path, values)).resolve()


def test_resume_is_in_place_and_mutually_exclusive_with_adapter_warm_start(tmp_path):
    values = _tiny_mapping()
    values["artifacts"]["output_dir"] = str(tmp_path / "run")
    values["resume"]["from"] = str(tmp_path / "other")
    with pytest.raises(ConfigError, match="same path"):
        load(_write(tmp_path, values)).resolve()

    values["resume"]["from"] = str(tmp_path / "run")
    values["model"]["adapter_checkpoint"] = str(tmp_path / "adapter")
    with pytest.raises(ConfigError, match="mutually exclusive"):
        load(_write(tmp_path, values)).resolve()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("run.seed", True),
        ("dataset.repeat_per_prompt", 0),
        ("runtime.max_steps", 0),
        ("runtime.update_microbatch_size", False),
        ("optimizer.learning_rate", 0.0),
        ("optimizer.adam_beta1", 1.0),
        ("algorithm.advantage.epsilon", float("inf")),
        ("reward.execution.max_retries", 11),
    ],
)
def test_global_numeric_contract_rejects_bool_boundaries_and_nonfinite(
    tmp_path, path, value
):
    values = _tiny_mapping()
    cursor = values
    segments = path.split(".")
    for segment in segments[:-1]:
        cursor = cursor[segment]
    cursor[segments[-1]] = value

    with pytest.raises(ConfigError):
        load(_write(tmp_path, values)).resolve()


def test_inline_prompt_canonicalization_happens_once_before_repeat(tmp_path):
    values = _tiny_mapping()
    values["dataset"]["prompts"] = ["  a red cube  ", " ", "a blue cube"]
    values["dataset"]["empty_prompt_policy"] = "skip"
    config = load(_write(tmp_path, values)).resolve()
    assert config.dataset.prompts == ("a red cube", "a blue cube")

    values["dataset"]["prompts"] = ["a red cube", " a red cube "]
    with pytest.raises(ConfigError, match="unique"):
        load(_write(tmp_path, values)).resolve()


def test_unknown_component_never_falls_back_to_registry(tmp_path):
    values = _tiny_mapping()
    values["model"]["name"] = "not_a_builtin"

    with pytest.raises(ComponentError, match="not_a_builtin"):
        load(_write(tmp_path, values)).resolve()


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "configs" / "tempflow_sd3.yaml",
        ROOT / "configs" / "flash_wan.yaml",
        ROOT / "configs" / "world_r1_wan.yaml",
    ],
)
def test_public_baselines_are_complete_and_resolve_without_resource_access(path):
    config = load(path).resolve()
    assert config.schema_version == 1
    assert config.runtime.max_steps == 20
    assert config.artifacts.output_dir.is_absolute()
    assert config.dataset.path is not None and config.dataset.path.is_absolute()
