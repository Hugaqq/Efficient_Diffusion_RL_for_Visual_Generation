from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _bounded_args(tmp_path, train_path, heldout_path, *, resume_from=None, steps=5):
    return SimpleNamespace(
        adapter="sd3_tempflow",
        model_path="/models/sd35",
        repo_root="/ref/tempflow",
        prompt="a red square",
        train_prompts_file=str(train_path),
        heldout_prompts_file=str(heldout_path),
        baseline_eval=None,
        eval_seeds=[1701, 1702],
        eval_max_prompts=None,
        resolution=16,
        num_steps=2,
        guidance_scale=3.5,
        seed=99,
        device="cpu",
        dtype="float32",
        lora_rank=8,
        lora_alpha=16,
        max_sequence_length=77,
        logprob_atol=1e-5,
        steps=steps,
        output_dir=str(tmp_path / "run"),
        resume_from=resume_from,
        allow_long_run=steps > 5,
        disable_lora=False,
        disable_rollout_cache=True,
    )


def test_sd3_dataset_config_fingerprints_train_heldout_and_resumes_cleanly(
    tmp_path,
):
    from scripts import legacy_cli as cli
    from visual_rl.artifacts.checkpoint import config_fingerprint
    from visual_rl.configs.schema import config_to_dict
    from visual_rl.datasets.prompt_dataset import prompt_content_sha256

    train_path = tmp_path / "train.txt"
    heldout_path = tmp_path / "heldout.txt"
    train_path.write_text("a red cube\na green bus\n", encoding="utf-8")
    heldout_path.write_text("a blue vase\n", encoding="utf-8")
    fresh_args = _bounded_args(tmp_path, train_path, heldout_path)
    fresh = cli._sd3_bounded_trainer_config(fresh_args)  # noqa: SLF001

    checkpoint = tmp_path / "checkpoint_000005"
    checkpoint.mkdir()
    resume_args = _bounded_args(
        tmp_path,
        train_path,
        heldout_path,
        resume_from=str(checkpoint),
        steps=6,
    )
    resumed = cli._sd3_bounded_trainer_config(resume_args)  # noqa: SLF001

    assert fresh.dataset.path == str(train_path)
    assert fresh.dataset.prompts == []
    assert fresh.dataset.require_unique is True
    assert fresh.dataset.content_sha256 == prompt_content_sha256(
        ["a red cube", "a green bus"]
    )
    assert fresh.evaluation.path == str(heldout_path)
    assert fresh.evaluation.content_sha256 == prompt_content_sha256(
        ["a blue vase"]
    )
    assert fresh.evaluation.seeds == [1701, 1702]
    assert fresh.rewards.weights == {"prompt_color_margin": 1.0}
    assert fresh.rewards.clients["prompt_color_margin"]["name"] == (
        "prompt_color_margin"
    )
    assert fresh.train.lora_path is None
    assert resumed.train.lora_path is None
    assert config_fingerprint(config_to_dict(fresh)) == config_fingerprint(
        config_to_dict(resumed)
    )


def test_prompt_color_margin_preserves_branch_signal_when_clipped_score_saturates():
    import numpy as np

    from visual_rl.feedback.image_rewards import (
        PromptColorMarginRewardClient,
        PromptColorRewardClient,
    )

    media = np.zeros((2, 3, 2, 2), dtype=np.float32)
    media[0, 0] = 0.9
    media[1, 0] = 0.8
    prompts = ["a red object", "a red object"]
    metadata = [{}, {}]

    clipped, _ = PromptColorRewardClient().score(media, prompts, metadata)
    margin, details = PromptColorMarginRewardClient().score(
        media,
        prompts,
        metadata,
    )

    assert clipped.tolist() == [1.0, 1.0]
    assert margin.tolist() == pytest.approx([0.9, 0.8])
    assert details["score_kind"] == "unclipped_target_channel_margin"


def test_sd3_heldout_panel_is_paired_and_reports_reward_and_guardrails(tmp_path):
    import torch

    from scripts import legacy_cli as cli
    from visual_rl.core.types import RewardBatch, RolloutBatch

    train_path = tmp_path / "train.txt"
    heldout_path = tmp_path / "heldout.txt"
    train_path.write_text("a red cube\n", encoding="utf-8")
    heldout_path.write_text("a red vase\na blue bus\n", encoding="utf-8")
    args = _bounded_args(tmp_path, train_path, heldout_path)

    class FakeAdapter:
        def __init__(self):
            self.offset = 0.0

        def sample(self, prompts, metadata, rollout_config):
            generator = torch.Generator().manual_seed(int(rollout_config["seed"]))
            media = torch.rand(1, 3, 8, 8, generator=generator)
            media[:, 0] += self.offset
            return RolloutBatch(
                prompts=prompts,
                metadata=metadata,
                media=media.clamp(0.0, 1.0),
                latents=torch.zeros(1, 1, 1, 2, 2),
                next_latents=torch.ones(1, 1, 1, 2, 2),
                timesteps=torch.ones(1, 1),
                old_log_probs=torch.zeros(1, 1),
                kl=torch.zeros(1, 1),
                model_metadata={"contract": "fake"},
            )

    class FakeFeedback:
        def score(self, batch):
            is_red = "red" in batch.prompts[0]
            base = 0.6 if is_red else 0.5
            reward = base + float(trainer.adapter.offset)
            target = "red" if is_red else "blue"
            return RewardBatch(
                raw={"prompt_color": torch.tensor([reward])},
                weighted={"prompt_color": torch.tensor([reward])},
                weighted_total=torch.tensor([reward]),
                valid_mask=torch.tensor([True]),
                metadata={"prompt_color": {"targets": [target]}},
            )

    trainer = SimpleNamespace(
        adapter=FakeAdapter(),
        feedback_provider=FakeFeedback(),
    )
    output_dir = tmp_path / "eval"
    before = cli._bounded_heldout_summary(  # noqa: SLF001
        trainer,
        args,
        "before",
        output_dir,
        milestone_step=0,
    )
    trainer.adapter.offset = 0.1
    after = cli._bounded_heldout_summary(  # noqa: SLF001
        trainer,
        args,
        "after",
        output_dir,
        milestone_step=5,
    )
    delta = cli._paired_heldout_delta(before, after)  # noqa: SLF001

    assert before["prompt_count"] == 2
    assert before["sample_count"] == 4
    assert before["seeds"] == [1701, 1702]
    assert set(before["per_color"]) == {"blue", "red"}
    assert len(before["png_paths"]) == 4
    assert all(
        (tmp_path / "eval") in path.parents
        for path in map(Path, before["png_paths"])
    )
    assert delta["paired_count"] == 4
    assert delta["eval_seed_cluster_count"] == 2
    assert delta["ci_method"] == "bootstrap_over_eval_seed_cluster_means"
    assert delta["reward_delta_mean"] == pytest.approx(0.1)
    assert delta["reward_delta_ci95_low"] > 0.0
    assert "spatial_std" in delta["guardrail_after_over_before"]


def test_sd3_heldout_ci_bootstraps_eval_seed_clusters_not_grid_cells():
    from scripts import legacy_cli as cli

    records_before = []
    records_after = []
    for seed, delta in ((1, 0.10), (2, 0.10), (3, -0.05)):
        for prompt_index in range(9):
            prompt_id = f"prompt-{prompt_index}"
            records_before.append(
                {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "reward": 0.5,
                }
            )
            records_after.append(
                {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "reward": 0.5 + delta,
                }
            )
    before = {"records": records_before, "image_guardrail": {}}
    after = {"records": records_after, "image_guardrail": {}}

    summary = cli._paired_heldout_delta(before, after)  # noqa: SLF001

    assert summary["reward_delta_mean"] == pytest.approx(0.05)
    assert summary["eval_seed_cluster_count"] == 3
    assert summary["reward_delta_ci95_low"] == pytest.approx(-0.05)
    assert summary["reward_delta_ci95_high"] == pytest.approx(0.10)
