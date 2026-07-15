"""CPU-only tests for canonical rollout and reward data contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import pytest
import torch

from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.datasets.prompt_dataset import (
    PromptDataset,
    prompt_split_leakage_report,
    validate_prompt_splits,
)
from visual_rl.rollout.branching import BranchingRollout
from visual_rl.rollout.cache import RolloutCache
from visual_rl.rollout.full_trajectory import FullTrajectoryRollout
from visual_rl.rollout.single_step import SingleStepRollout


def _rollout_batch(
    prompts: list[str],
    metadata: list[dict],
    *,
    steps: int = 3,
    video: bool = False,
) -> RolloutBatch:
    batch_size = len(prompts)
    media = (
        torch.zeros(batch_size, 2, 3, 4, 4)
        if video
        else torch.zeros(batch_size, 3, 4, 4)
    )
    return RolloutBatch(
        prompts=prompts,
        metadata=metadata,
        media=media,
        latents=torch.zeros(batch_size, steps, 2, 2, 2),
        next_latents=torch.ones(batch_size, steps, 2, 2, 2),
        timesteps=torch.arange(steps).repeat(batch_size, 1),
        old_log_probs=torch.zeros(batch_size, steps, requires_grad=True),
        kl=torch.zeros(batch_size, steps),
        model_tensors={
            "nested": {
                "float": torch.ones(batch_size, 2, requires_grad=True),
                "integer": torch.ones(batch_size, dtype=torch.long),
            }
        },
    )


def test_step_context_is_frozen() -> None:
    context = StepContext(step=2, seed=7, epoch_tag=3)
    with pytest.raises(FrozenInstanceError):
        context.step = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("video", "layout", "media_shape"),
    [
        (False, "BCHW", (2, 3, 4, 4)),
        (True, "BFCHW", (2, 2, 3, 4, 4)),
    ],
)
def test_rollout_batch_layout_shapes_to_and_detach(
    video: bool,
    layout: str,
    media_shape: tuple[int, ...],
) -> None:
    batch = _rollout_batch(
        ["alpha", "beta"],
        [
            {"prompt_id": "p0", "group_id": "g0"},
            {"prompt_id": "p1", "group_id": "g1"},
        ],
        video=video,
    )
    batch.validate_lightweight(strict=True)

    assert batch.batch_size == 2
    assert batch.media_layout == layout
    assert batch.shapes["media"] == media_shape
    assert batch.shapes["transition_mask"] == (2, 3)
    assert batch.branch_ids is batch.branch_id

    moved = batch.to("cpu", dtype=torch.float64)
    assert moved is not batch
    assert moved.media.dtype == torch.float64
    assert moved.timesteps.dtype == torch.int64
    assert moved.transition_mask.dtype == torch.bool
    assert moved.model_tensors["nested"]["float"].dtype == torch.float64
    assert moved.model_tensors["nested"]["integer"].dtype == torch.int64

    detached = moved.detach()
    assert detached is not moved
    assert not detached.old_log_probs.requires_grad
    assert not detached.model_tensors["nested"]["float"].requires_grad


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"sample_id": ["only-one"]}, "sample_id length"),
        ({"media_layout": "BHWC"}, "media_layout"),
        (
            {"transition_mask": torch.ones(2, 3)},
            "bool dtype",
        ),
        (
            {"timesteps": torch.zeros(2, 2)},
            "transition tensors must share",
        ),
    ],
)
def test_rollout_batch_rejects_bad_identity_layout_and_transition_shapes(
    updates: dict,
    match: str,
) -> None:
    batch = _rollout_batch(
        ["alpha", "beta"],
        [{"group_id": "g0"}, {"group_id": "g1"}],
    ).replace(**updates)
    with pytest.raises(ValueError, match=match):
        batch.validate_lightweight(strict=True)


def test_reward_batch_validates_order_vectors_masks_finite_and_total() -> None:
    batch = _rollout_batch(
        ["alpha", "beta"],
        [{"group_id": "g0"}, {"group_id": "g1"}],
    )
    rewards = RewardBatch(
        sample_id=list(batch.sample_id),
        raw={"quality": torch.tensor([1.0, 2.0])},
        weighted={"quality": torch.tensor([0.5, 1.0])},
        weighted_total=torch.tensor([0.5, 1.0]),
        valid_mask=torch.tensor([True, False]),
    )
    rewards.validate_against(batch)
    assert rewards.batch_size == 2
    assert rewards.shapes["weighted_total"] == (2,)
    assert rewards.to("cpu", dtype=torch.float64).valid_mask.dtype == torch.bool
    assert rewards.detach() is not rewards

    legacy = RewardBatch(
        raw={},
        weighted={},
        weighted_total=torch.zeros(2),
        valid_mask=torch.ones(2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="sample_id is required"):
        legacy.validate_against(batch)

    bad_cases = [
        ({"sample_id": list(reversed(batch.sample_id))}, "order"),
        ({"weighted_total": torch.zeros(2, 1)}, "shape"),
        ({"valid_mask": torch.ones(2)}, "bool dtype"),
        ({"raw": {"other": torch.ones(2)}}, "keys"),
        ({"weighted_total": torch.tensor([0.5, float("nan")])}, "finite"),
        ({"weighted_total": torch.zeros(2)}, "sum of weighted"),
    ]
    for updates, match in bad_cases:
        values = {
            "sample_id": list(rewards.sample_id),
            "raw": dict(rewards.raw),
            "weighted": dict(rewards.weighted),
            "weighted_total": rewards.weighted_total,
            "valid_mask": rewards.valid_mask,
        }
        values.update(updates)
        with pytest.raises(ValueError, match=match):
            RewardBatch(**values).validate_against(batch)


def test_prompt_dataset_assigns_stable_occurrence_groups() -> None:
    dataset = PromptDataset.from_config(
        {
            "prompts": ["alpha", "beta"],
            "repeat_per_prompt": 2,
        }
    )
    _, first, _ = dataset.batch(0, 4, epoch_tag=0)
    _, repeated, _ = dataset.batch(0, 4, epoch_tag=0)
    _, next_epoch, _ = dataset.batch(4, 4, epoch_tag=1)

    assert [item["group_id"] for item in first] == [
        item["group_id"] for item in repeated
    ]
    assert len({item["group_id"] for item in first}) == 4
    assert {item["group_id"] for item in first}.isdisjoint(
        {item["group_id"] for item in next_epoch}
    )
    assert first[0]["prompt_id"] == first[1]["prompt_id"]
    assert first[0]["group_id"] != first[1]["group_id"]


@pytest.mark.parametrize(
    "metadata",
    [
        [{}, {}],
        [{"group_id": "occurrence-0"}, {}],
    ],
)
def test_formal_batch_rejects_repeated_prompt_ids_without_occurrence_groups(
    metadata: list[dict],
) -> None:
    with pytest.raises(ValueError, match="explicit group_id for each prompt occurrence"):
        RolloutBatch(
            prompts=["same prompt", "same prompt"],
            metadata=metadata,
            context=StepContext(step=0, seed=7, epoch_tag=0),
        )


def test_formal_batch_accepts_explicit_occurrence_and_branch_groups() -> None:
    context = StepContext(step=0, seed=7, epoch_tag=0)
    independent = RolloutBatch(
        prompts=["same prompt", "same prompt"],
        metadata=[{}, {}],
        group_id=["occurrence-0", "occurrence-1"],
        context=context,
    )
    independent.validate_lightweight()
    assert independent.group_id == ["occurrence-0", "occurrence-1"]

    branches = RolloutBatch(
        prompts=["same prompt", "same prompt"],
        metadata=[
            {"group_id": "shared-occurrence", "branch_id": 0},
            {"group_id": "shared-occurrence", "branch_id": 1},
        ],
        context=context,
    )
    branches.validate_lightweight()
    assert branches.group_id == ["shared-occurrence", "shared-occurrence"]


def test_legacy_seed_context_cannot_bypass_formal_occurrence_groups() -> None:
    with pytest.raises(ValueError, match="explicit group_id for each prompt occurrence"):
        RolloutBatch(
            prompts=["same prompt", "same prompt"],
            metadata=[{}, {}],
            seed=7,
            epoch_tag=0,
        )


def test_lightweight_validation_rechecks_context_mutation() -> None:
    batch = RolloutBatch(
        prompts=["same prompt", "same prompt"],
        metadata=[{}, {}],
    )
    batch.context = StepContext(step=0, seed=7, epoch_tag=0)

    with pytest.raises(ValueError, match="explicit group_id for each prompt occurrence"):
        batch.validate_lightweight()


def test_legacy_batch_keeps_repeated_prompt_id_fallback() -> None:
    batch = RolloutBatch(
        prompts=["same prompt", "same prompt"],
        metadata=[{}, {}],
    )
    batch.validate_lightweight()
    assert batch.context is None
    assert batch.group_id[0] == batch.group_id[1] == batch.prompt_id[0]


def test_prompt_dataset_occurrence_groups_survive_rollout_finalization() -> None:
    dataset = PromptDataset.from_config(
        {"prompts": ["same prompt"], "repeat_per_prompt": 2}
    )
    prompts, metadata, _ = dataset.batch(0, 2, epoch_tag=0)
    batch = FullTrajectoryRollout(
        {"samples_per_prompt": 2, "nested": {"value": "original"}}
    ).sample(
        _OfflineAdapter(),
        prompts,
        metadata,
        StepContext(step=0, seed=7, epoch_tag=0),
    )

    assert batch.context is not None
    assert len(set(batch.group_id)) == 2
    assert sorted(batch.group_id.count(group) for group in set(batch.group_id)) == [
        2,
        2,
    ]


def test_prompt_empty_rows_are_explicit_and_split_audits_are_bounded(
    tmp_path,
) -> None:
    source = tmp_path / "prompts.txt"
    source.write_text("first\n\nsecond\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty prompt row at line 2"):
        PromptDataset.from_config({"path": str(source)})
    dataset = PromptDataset.from_config(
        {"path": str(source), "empty_prompt_policy": "skip"}
    )
    assert dataset.source_prompts == ["first", "second"]
    assert dataset.empty_prompt_policy == "skip"

    with pytest.raises(ValueError, match="normalized=1"):
        validate_prompt_splits(["A red square!"], ["a red square"])
    with pytest.raises(ValueError, match="near_duplicate=1"):
        validate_prompt_splits(
            ["a red square with one small tree"],
            ["a red square with a small tree"],
        )
    with pytest.raises(ValueError, match="exceeds max_comparisons"):
        prompt_split_leakage_report(
            ["one", "two"],
            ["three", "four"],
            max_comparisons=3,
        )


class _NativeSingleStepAdapter:
    name = "native-single-step"
    media_type = "video"

    def sample_single_step(self, prompts, metadata, rollout_config):
        selected = rollout_config["selected_timestep_indices"]
        actual_timesteps = torch.tensor(
            [[999 if index == 0 else 982] for index in selected]
        )
        batch_size = len(prompts)
        return RolloutBatch(
            prompts=list(prompts),
            metadata=[dict(item) for item in metadata],
            media=torch.zeros(batch_size, 1, 3, 4, 4),
            latents=torch.zeros(batch_size, 1, 2),
            next_latents=torch.ones(batch_size, 1, 2),
            timesteps=actual_timesteps,
            old_log_probs=torch.zeros(batch_size, 1),
            kl=torch.zeros(batch_size, 1),
        )


def test_single_step_rectification_uses_native_scheduler_timestep_values() -> None:
    batch = SingleStepRollout(
        {
            "samples_per_prompt": 1,
            "num_steps": 2,
            "selected_step_strategy": "cycle",
            "rectification_mode": "flash_reference_table",
        }
    ).sample(
        _NativeSingleStepAdapter(),
        ["alpha", "beta"],
        [
            {"prompt_id": "p0", "group_id": "g0"},
            {"prompt_id": "p1", "group_id": "g1"},
        ],
        StepContext(step=0, seed=7, epoch_tag=0),
    )

    expected = [7.4770, 7.0414]
    expected = [value / (sum(expected) / len(expected)) for value in expected]
    assert batch.model_metadata["selected_timestep_values"] == [999, 982]
    assert [row[0] for row in batch.model_metadata["flash_rectification_weights"]] == pytest.approx(
        expected
    )


class _OfflineAdapter:
    name = "offline"
    media_type = "image"

    def __init__(self) -> None:
        self.runtime_configs: list[dict] = []

    def sample(self, prompts, metadata, rollout_config):
        self.runtime_configs.append(deepcopy(rollout_config))
        rollout_config["nested"]["value"] = "adapter-mutated"
        return _rollout_batch(list(prompts), list(metadata), steps=3)

    def branch_transition_count(self, rollout_config):
        rollout_config["nested"]["counter"] = "adapter-mutated"
        return 3

    def sample_branching(self, prompts, metadata, rollout_config):
        self.runtime_configs.append(deepcopy(rollout_config))
        expanded_prompts = []
        expanded_metadata = []
        for parent_index, (prompt, item) in enumerate(zip(prompts, metadata)):
            for branch_id in range(rollout_config["branch_count"]):
                expanded_prompts.append(prompt)
                expanded_metadata.append(
                    {
                        **item,
                        "parent_prompt_index": parent_index,
                        "branch_id": branch_id,
                        "branch_step_index": rollout_config["branch_step_index"],
                        "branch_timestep_value": 100,
                    }
                )
        return _rollout_batch(expanded_prompts, expanded_metadata, steps=1)


@pytest.mark.parametrize(
    "engine",
    [
        FullTrajectoryRollout(
            {"samples_per_prompt": 2, "nested": {"value": "original"}}
        ),
        SingleStepRollout(
            {
                "samples_per_prompt": 2,
                "num_steps": 3,
                "nested": {"value": "original"},
            }
        ),
        BranchingRollout(
            {
                "branch_count": 2,
                "num_steps": 3,
                "nested": {"value": "original"},
            }
        ),
    ],
)
def test_rollouts_use_context_and_never_mutate_config(engine) -> None:
    adapter = _OfflineAdapter()
    context = StepContext(
        step=4,
        seed=19,
        epoch_tag=2,
        rank=1,
        world_size=2,
        policy_version=3,
    )
    original_config = deepcopy(engine.config)
    batch = engine.sample(
        adapter,
        ["alpha"],
        [{"prompt_id": "prompt-alpha", "group_id": "group-alpha"}],
        context,
    )

    assert engine.config == original_config
    assert batch.context == context
    assert batch.seed == 19
    assert batch.epoch_tag == 2
    assert len(set(batch.sample_id)) == batch.batch_size
    assert set(batch.group_id) == {"group-alpha"}
    assert adapter.runtime_configs[-1]["step"] == 4
    assert adapter.runtime_configs[-1]["rank"] == 1


def test_rollout_cache_round_trip_and_legacy_branch_ids(tmp_path) -> None:
    context = StepContext(step=5, seed=11, epoch_tag=5)
    batch = FullTrajectoryRollout(
        {"samples_per_prompt": 2, "nested": {"value": "original"}}
    ).sample(
        _OfflineAdapter(),
        ["alpha"],
        [{"prompt_id": "p0", "group_id": "g0"}],
        context,
    )
    cache = RolloutCache(tmp_path)
    cache.save(5, batch)
    loaded = cache.load(5)
    audit = cache.validate_step(5)

    assert loaded.sample_id == batch.sample_id
    assert loaded.group_id == batch.group_id
    assert loaded.branch_id == batch.branch_id
    assert loaded.context == context
    assert loaded.media_layout == "BCHW"
    assert torch.equal(loaded.transition_mask, batch.transition_mask)
    assert audit["valid"] is True
    assert audit["version"] == 2
    assert audit["prompt_count"] == batch.batch_size
    assert audit["generation"]
    assert set(audit["files"]) == {"tensor", "media", "metadata"}

    old_base = tmp_path / "batch_000006"
    torch.save(
        {
            "latents": batch.latents,
            "next_latents": batch.next_latents,
            "timesteps": batch.timesteps,
            "old_log_probs": batch.old_log_probs,
            "kl": batch.kl,
            "branch_ids": torch.tensor([7, 8]),
            "model_tensors": {},
        },
        old_base.with_suffix(".pt"),
    )
    torch.save(batch.media, old_base.with_suffix(".media.pt"))
    old_base.with_suffix(".json").write_text(
        json.dumps(
            {
                "prompts": batch.prompts,
                "metadata": batch.metadata,
                "model_metadata": {},
                "media_path": old_base.with_suffix(".media.pt").name,
            }
        ),
        encoding="utf-8",
    )
    legacy = cache.load(6)
    assert legacy.branch_id.tolist() == [7, 8]
    legacy.validate_lightweight(strict=True)


def test_rollout_cache_validate_step_rejects_corrupt_v2_media(tmp_path) -> None:
    batch = _rollout_batch(
        ["alpha"],
        [{"prompt_id": "p0", "group_id": "g0"}],
        steps=1,
    )
    cache = RolloutCache(tmp_path)
    cache.save(0, batch)
    (tmp_path / "batch_000000.media.pt").write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="media payload is corrupt"):
        cache.validate_step(0)
