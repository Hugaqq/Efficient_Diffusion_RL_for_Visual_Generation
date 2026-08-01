"""Contracts for the frozen Pick-a-Pic SFW Flow-GRPO prompt subsets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import visual_rl as vr
from experiments.v0_7.prepare_pickapic_sfw import (
    EVAL_BIN_COUNTS,
    FINAL_TEST_ACCESS_POLICY,
    FINAL_TEST_BIN_COUNTS,
    FINAL_TEST_FILENAME,
    MAX_HPS_CLIP_TOKENS,
    MAX_T5_TOKENS,
    OUTPUT_VERSION,
    PROVENANCE_FILENAME,
    SELECTION_SEED,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_SHA256,
    TRAIN_BIN_COUNTS,
    V2_FROZEN_SHA256,
    V2_OUTPUT_VERSION,
    V3_FROZEN_SHA256,
    _length_bin,
    _rank,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "data" / "prompts"
TRAIN_PATH = PROMPT_ROOT / "pickapic_sfw_q100_train_v1.txt"
HELDOUT_PATH = PROMPT_ROOT / "pickapic_sfw_heldout_eval_v1.txt"
PROVENANCE_PATH = PROMPT_ROOT / "pickapic_sfw_provenance_v1.json"
TRAIN_V2_PATH = PROMPT_ROOT / "pickapic_sfw_q100_train_v2.txt"
HELDOUT_V2_PATH = PROMPT_ROOT / "pickapic_sfw_heldout_eval_v2.txt"
PROVENANCE_V2_PATH = PROMPT_ROOT / "pickapic_sfw_provenance_v2.json"
FINAL_TEST_V3_PATH = PROMPT_ROOT / FINAL_TEST_FILENAME
PROVENANCE_V3_PATH = PROMPT_ROOT / PROVENANCE_FILENAME
CONFIG_ROOT = ROOT / "experiments" / "flow_pickapic_20260801" / "configs"
EXPECTED_V2_FROZEN_SHA256 = {
    "q100_train": (
        "bda5208d4f90465063861d52c401fca8b4adcf22f273b892efb5cb848279c3d7"
    ),
    "validation": (
        "26cd082a5677d5de1bfeefa8ff0da2be3e7d21d9ea35091dc7df10aae788f68d"
    ),
    "provenance": (
        "57f37eea53c2420a4355c88e2d3f86205bceb4c68d1553ed2be59dd3e6e0316b"
    ),
}
EXPECTED_FINAL_TEST_V3_SHA256 = (
    "fe5def8d78ff1233a371cdb029cdc14cd30d4a20baa62714b4594973f6ad58d2"
)
EXPECTED_PROVENANCE_V3_SHA256 = (
    "4b6b22afc88a5758373a130ee8fa2b3a6e5604a6b2219905cb9cb81ea4d061db"
)


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

    assert manifest["schema_version"] == V2_OUTPUT_VERSION == 2
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


def test_pickapic_v3_final_test_is_frozen_disjoint_and_sealed() -> None:
    manifest = json.loads(PROVENANCE_V3_PATH.read_text(encoding="utf-8"))
    train = _prompts(TRAIN_V2_PATH)
    validation = _prompts(HELDOUT_V2_PATH)
    final_test = _prompts(FINAL_TEST_V3_PATH)

    assert V2_FROZEN_SHA256 == EXPECTED_V2_FROZEN_SHA256
    assert V3_FROZEN_SHA256 == {
        "final_test": EXPECTED_FINAL_TEST_V3_SHA256,
        "provenance": EXPECTED_PROVENANCE_V3_SHA256,
    }
    assert _sha256(TRAIN_V2_PATH) == EXPECTED_V2_FROZEN_SHA256["q100_train"]
    assert _sha256(HELDOUT_V2_PATH) == EXPECTED_V2_FROZEN_SHA256["validation"]
    assert _sha256(PROVENANCE_V2_PATH) == (
        EXPECTED_V2_FROZEN_SHA256["provenance"]
    )
    assert _sha256(FINAL_TEST_V3_PATH) == EXPECTED_FINAL_TEST_V3_SHA256
    assert _sha256(PROVENANCE_V3_PATH) == EXPECTED_PROVENANCE_V3_SHA256

    assert len(train) == len(set(train)) == 100
    assert len(validation) == len(set(validation)) == 64
    assert len(final_test) == len(set(final_test)) == 64
    assert not set(train).intersection(validation)
    assert not set(train).intersection(final_test)
    assert not set(validation).intersection(final_test)
    assert all(
        prompt and len(prompt.split()) >= 6
        for prompt in train + validation + final_test
    )
    assert _bin_counts(train) == TRAIN_BIN_COUNTS
    assert _bin_counts(validation) == EVAL_BIN_COUNTS
    assert _bin_counts(final_test) == FINAL_TEST_BIN_COUNTS
    assert final_test == tuple(
        sorted(final_test, key=lambda prompt: _rank(prompt, split="final_test"))
    )

    assert manifest["schema_version"] == OUTPUT_VERSION == 3
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
    assert selection["minimum_words"] == 6
    assert selection["maximum_t5_tokens"] == MAX_T5_TOKENS
    assert selection["maximum_hps_clip_tokens"] == MAX_HPS_CLIP_TOKENS
    assert selection["t5_tokenizer_class"] == "T5Tokenizer"
    assert selection["t5_tokenizer_vocab_sha256"] == (
        "3e4dac4c480136f1337a43a23a00b56e92b10e1e65483bb718ceb971eef2aa14"
    )
    assert selection["hps_clip_tokenizer_class"] == "CLIPTokenizer"
    assert selection["hps_clip_tokenizer_vocab_sha256"] == (
        "f9190aead40a965a09db1ba2e556dccac0eda8423cb4926bcd994722899ec68e"
    )
    assert selection["final_test_candidate_count"] == 909
    assert selection["final_test_candidate_length_bins"] == {
        "medium": 613,
        "long": 222,
        "very_long": 74,
    }
    assert selection["final_test_candidate_pool"] == (
        "eligible raw test prompts after excluding train-source duplicates "
        "and the frozen v2 validation prompts"
    )
    assert selection["final_test_rank_domain"] == "final_test"
    assert selection["rank_function"] == "sha256('<seed>:<split>:<prompt>')"
    assert selection["uses_model_outputs"] is False
    assert selection["uses_reward_outputs"] is False

    splits = manifest["splits"]
    assert splits["q100_train"] == {
        "artifact_version": 2,
        "count": 100,
        "length_bins": TRAIN_BIN_COUNTS,
        "max_hps_clip_tokens": 72,
        "max_t5_tokens": 89,
        "min_hps_clip_tokens": 8,
        "min_t5_tokens": 8,
        "path": "pickapic_sfw_q100_train_v2.txt",
        "role": "training",
        "sha256": EXPECTED_V2_FROZEN_SHA256["q100_train"],
    }
    assert splits["validation"] == {
        "artifact_version": 2,
        "count": 64,
        "former_role": "heldout_eval",
        "length_bins": EVAL_BIN_COUNTS,
        "max_hps_clip_tokens": 75,
        "max_t5_tokens": 100,
        "min_hps_clip_tokens": 8,
        "min_t5_tokens": 9,
        "path": "pickapic_sfw_heldout_eval_v2.txt",
        "role": "validation",
        "sha256": EXPECTED_V2_FROZEN_SHA256["validation"],
        "status": "used_for_c20_tuning",
    }
    assert splits["final_test"] == {
        "access_policy": FINAL_TEST_ACCESS_POLICY,
        "allowed_use": (
            "open once for the final multi-seed conclusion after all training "
            "and model selection are frozen"
        ),
        "artifact_version": 3,
        "count": 64,
        "length_bins": FINAL_TEST_BIN_COUNTS,
        "max_hps_clip_tokens": 73,
        "max_t5_tokens": 106,
        "min_hps_clip_tokens": 8,
        "min_t5_tokens": 8,
        "path": FINAL_TEST_FILENAME,
        "role": "final_test",
        "sha256": EXPECTED_FINAL_TEST_V3_SHA256,
        "status": "not_used_for_c20_tuning",
    }
    assert splits["final_test"]["max_t5_tokens"] <= MAX_T5_TOKENS
    assert (
        splits["final_test"]["max_hps_clip_tokens"]
        <= MAX_HPS_CLIP_TOKENS
    )
    assert manifest["v2_provenance"] == {
        "path": "pickapic_sfw_provenance_v2.json",
        "sha256": EXPECTED_V2_FROZEN_SHA256["provenance"],
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
