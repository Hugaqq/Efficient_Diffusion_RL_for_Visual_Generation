"""Contracts for the frozen Pick-a-Pic SFW Flow-GRPO prompt subsets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import visual_rl as vr
from experiments.v0_7.prepare_pickapic_sfw import (
    EVAL_BIN_COUNTS,
    MAX_HPS_CLIP_TOKENS,
    MAX_T5_TOKENS,
    OUTPUT_VERSION,
    SELECTION_SEED,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_SHA256,
    TRAIN_BIN_COUNTS,
    _length_bin,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "data" / "prompts"
TRAIN_PATH = PROMPT_ROOT / "pickapic_sfw_q100_train_v1.txt"
HELDOUT_PATH = PROMPT_ROOT / "pickapic_sfw_heldout_eval_v1.txt"
PROVENANCE_PATH = PROMPT_ROOT / "pickapic_sfw_provenance_v1.json"
TRAIN_V2_PATH = PROMPT_ROOT / "pickapic_sfw_q100_train_v2.txt"
HELDOUT_V2_PATH = PROMPT_ROOT / "pickapic_sfw_heldout_eval_v2.txt"
PROVENANCE_V2_PATH = PROMPT_ROOT / "pickapic_sfw_provenance_v2.json"
CONFIG_ROOT = ROOT / "experiments" / "flow_pickapic_20260801" / "configs"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompts(path: Path) -> tuple[str, ...]:
    return tuple(path.read_text(encoding="utf-8").splitlines())


def _bin_counts(prompts: tuple[str, ...]) -> dict[str, int]:
    return {
        name: sum(_length_bin(prompt) == name for prompt in prompts)
        for name in ("medium", "long", "very_long")
    }


def test_pickapic_prompt_subsets_match_frozen_provenance() -> None:
    manifest = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    train = _prompts(TRAIN_PATH)
    heldout = _prompts(HELDOUT_PATH)

    assert len(train) == len(set(train)) == 100
    assert len(heldout) == len(set(heldout)) == 64
    assert not set(train).intersection(heldout)
    assert all(prompt and len(prompt.split()) >= 6 for prompt in train + heldout)
    assert _bin_counts(train) == TRAIN_BIN_COUNTS
    assert _bin_counts(heldout) == EVAL_BIN_COUNTS

    assert manifest["schema_version"] == 1
    assert manifest["source"] == {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "train_rows": 15486,
        "train_sha256": SOURCE_SHA256["train"],
        "test_rows": 1024,
        "test_sha256": SOURCE_SHA256["test"],
    }
    assert manifest["selection"]["seed"] == SELECTION_SEED
    assert manifest["selection"]["maximum_t5_tokens"] == MAX_T5_TOKENS
    assert manifest["outputs"]["q100_train"] == {
        "count": 100,
        "length_bins": TRAIN_BIN_COUNTS,
        "max_t5_tokens": 123,
        "min_t5_tokens": 8,
        "sha256": _sha256(TRAIN_PATH),
    }
    assert manifest["outputs"]["heldout_eval"] == {
        "count": 64,
        "length_bins": EVAL_BIN_COUNTS,
        "max_t5_tokens": 125,
        "min_t5_tokens": 9,
        "sha256": _sha256(HELDOUT_PATH),
    }


def test_pickapic_v2_subsets_fit_sd3_and_hps_token_budgets() -> None:
    manifest = json.loads(PROVENANCE_V2_PATH.read_text(encoding="utf-8"))
    train = _prompts(TRAIN_V2_PATH)
    heldout = _prompts(HELDOUT_V2_PATH)

    assert len(train) == len(set(train)) == 100
    assert len(heldout) == len(set(heldout)) == 64
    assert not set(train).intersection(heldout)
    assert all(prompt and len(prompt.split()) >= 6 for prompt in train + heldout)
    assert _bin_counts(train) == TRAIN_BIN_COUNTS
    assert _bin_counts(heldout) == EVAL_BIN_COUNTS

    assert manifest["schema_version"] == OUTPUT_VERSION == 2
    assert manifest["source"] == {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "train_rows": 15486,
        "train_sha256": SOURCE_SHA256["train"],
        "test_rows": 1024,
        "test_sha256": SOURCE_SHA256["test"],
    }
    selection = manifest["selection"]
    assert selection["seed"] == SELECTION_SEED
    assert selection["maximum_t5_tokens"] == MAX_T5_TOKENS
    assert selection["maximum_hps_clip_tokens"] == MAX_HPS_CLIP_TOKENS
    assert selection["t5_tokenizer_class"] == "T5Tokenizer"
    assert selection["hps_clip_tokenizer_class"] == "CLIPTokenizer"
    assert manifest["outputs"]["q100_train"] == {
        "count": 100,
        "length_bins": TRAIN_BIN_COUNTS,
        "max_t5_tokens": 89,
        "min_t5_tokens": 8,
        "max_hps_clip_tokens": 72,
        "min_hps_clip_tokens": 8,
        "sha256": _sha256(TRAIN_V2_PATH),
    }
    assert manifest["outputs"]["heldout_eval"] == {
        "count": 64,
        "length_bins": EVAL_BIN_COUNTS,
        "max_t5_tokens": 100,
        "min_t5_tokens": 9,
        "max_hps_clip_tokens": 75,
        "min_hps_clip_tokens": 8,
        "sha256": _sha256(HELDOUT_V2_PATH),
    }


def test_pickapic_flow_configs_resolve_to_one_training_contract() -> None:
    expected = {
        "flow_pickapic_c20_seed17.yaml": (
            17,
            20,
            TRAIN_PATH,
            1,
            8,
            3.0e-4,
        ),
        "flow_pickapic_c20_stable_v2_seed17.yaml": (
            17,
            20,
            TRAIN_V2_PATH,
            2,
            4,
            1.0e-4,
        ),
        "flow_pickapic_q100_seed17.yaml": (
            17,
            100,
            TRAIN_V2_PATH,
            1,
            8,
            3.0e-4,
        ),
        "flow_pickapic_q100_seed29.yaml": (
            29,
            100,
            TRAIN_V2_PATH,
            1,
            8,
            3.0e-4,
        ),
        "flow_pickapic_q100_seed43.yaml": (
            43,
            100,
            TRAIN_V2_PATH,
            1,
            8,
            3.0e-4,
        ),
    }
    assert {path.name for path in CONFIG_ROOT.glob("*.yaml")} == set(expected)

    for name, values in expected.items():
        seed, steps, dataset_path, prompt_batch_size, group_size, learning_rate = (
            values
        )
        config = vr.load(CONFIG_ROOT / name).resolve()
        assert config.run.seed == seed
        assert config.runtime.max_steps == steps
        assert config.runtime.batch_size == prompt_batch_size
        assert config.runtime.precision == "bf16"
        assert config.runtime.distributed.mode == "single"
        assert config.model.name == "sd3_tempflow"
        assert config.model.params["offload_frozen_modules_during_update"] is True
        assert config.dataset.path == dataset_path.resolve()
        assert config.dataset.sampling_seed == seed
        assert config.rollout.name == "full_trajectory"
        assert config.rollout.params["num_steps"] == 20
        assert config.rollout.params["samples_per_prompt"] == group_size
        assert prompt_batch_size * group_size == 8
        assert [item.name for item in config.reward.components] == ["reward_general"]
        assert config.reward.components[0].params["server_revision"] == (
            "world-r1-e156b02bc171"
        )
        assert config.algorithm.name == "grpo"
        assert config.algorithm.params["beta"] == 0.004
        assert config.optimizer.learning_rate == learning_rate
        targets = config.model.params["lora_target_modules"]
        assert len(targets) == len(set(targets)) == 8
        assert config.artifacts.output_dir.is_relative_to(CONFIG_ROOT.parent / "runs")
        assert config.reward.cache_dir.is_relative_to(
            CONFIG_ROOT.parent / "reward_cache"
        )
