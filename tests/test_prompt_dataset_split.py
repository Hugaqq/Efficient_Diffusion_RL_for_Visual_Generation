from __future__ import annotations


def test_prompt_dataset_records_stable_ids_split_and_content_hash(tmp_path):
    from visual_rl.datasets.prompt_dataset import (
        PromptDataset,
        prompt_content_sha256,
        prompt_id,
    )

    path = tmp_path / "train.txt"
    path.write_text("a red cube\na green bus\n", encoding="utf-8")
    expected_hash = prompt_content_sha256(["a red cube", "a green bus"])

    dataset = PromptDataset.from_config(
        {
            "path": str(path),
            "split_name": "train",
            "content_sha256": expected_hash,
            "require_unique": True,
            "repeat_per_prompt": 1,
        }
    )

    assert dataset.content_sha256 == expected_hash
    assert dataset.split_name == "train"
    assert dataset[0].metadata == {
        "prompt_id": prompt_id("a red cube"),
        "split": "train",
    }


def test_prompt_dataset_rejects_changed_content_and_duplicates(tmp_path):
    import pytest

    from visual_rl.datasets.prompt_dataset import PromptDataset

    path = tmp_path / "train.txt"
    path.write_text("same prompt\nsame prompt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        PromptDataset.from_config(
            {"path": str(path), "require_unique": True}
        )

    path.write_text("different prompt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        PromptDataset.from_config(
            {"path": str(path), "content_sha256": "0" * 64}
        )


def test_prompt_split_validation_rejects_overlap_and_snapshots_hashes():
    import pytest

    from visual_rl.datasets.prompt_dataset import validate_prompt_splits

    snapshot = validate_prompt_splits(
        ["a red cube", "a green bus"],
        ["a blue vase"],
        train_path="train.txt",
        heldout_path="heldout.txt",
    )
    assert snapshot["overlap_count"] == 0
    assert snapshot["train"]["count"] == 2
    assert snapshot["heldout"]["count"] == 1
    assert len(snapshot["train"]["content_sha256"]) == 64

    with pytest.raises(ValueError, match="overlap"):
        validate_prompt_splits(["same"], ["same"])


def test_checkpoint_fingerprint_includes_prompt_and_evaluation_hashes():
    from visual_rl.artifacts.checkpoint import config_fingerprint
    from visual_rl.configs.schema import VisualRLConfig, config_to_dict

    config = VisualRLConfig(run_name="split-fingerprint")
    config.dataset.content_sha256 = "a" * 64
    config.evaluation.content_sha256 = "b" * 64
    original = config_fingerprint(config_to_dict(config))

    config.evaluation.content_sha256 = "c" * 64
    assert config_fingerprint(config_to_dict(config)) != original

