"""CPU-only coverage for the formal rollout/reward/update dataflow."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from visual_rl import ExperimentRunner, load_config
from visual_rl.artifacts.manifest import SampleManifest
from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.feedback.provider import RewardRouterFeedbackProvider
from visual_rl.feedback.router import RewardRouter
from visual_rl.optimizers.algorithm_plugin import AlgorithmOptimizerPlugin


ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "visual_rl/configs/presets/flash_tiny_single_step.yaml"


def _config(output_dir: Path, *, steps: int = 1):
    config = load_config(PRESET)
    config.paths.output_dir = str(output_dir)
    config.train.max_steps = steps
    config.train.save_every = max(1, steps)
    config.runner.show_progress = False
    config.runner.strict_rollout_validation = True
    return config


def test_runner_threads_context_identity_and_preserves_rollout_config(tmp_path):
    config = _config(tmp_path / "context-run", steps=2)
    runner = ExperimentRunner(config)
    rollout_config = deepcopy(runner.rollout.config)
    contexts = []
    batches = []
    optimizer_contexts = []
    advantage_groups = []

    original_sample = runner.rollout.sample

    def recording_sample(adapter, prompts, metadata, context):
        assert runner.rollout.config == rollout_config
        batch = original_sample(adapter, prompts, metadata, context)
        for index, item in enumerate(batch.metadata):
            item["parent_prompt_index"] = index
        contexts.append(context)
        batches.append(batch)
        return batch

    runner.rollout.sample = recording_sample
    original_compute = runner.optimizer_plugin.advantage_computer.compute

    def recording_compute(*args, **kwargs):
        advantage_groups.append(list(kwargs["group_ids"]))
        return original_compute(*args, **kwargs)

    runner.optimizer_plugin.advantage_computer.compute = recording_compute
    original_step = runner.optimizer_plugin.step

    def recording_step(*args, **kwargs):
        optimizer_contexts.append(kwargs["context"])
        return original_step(*args, **kwargs)

    runner.optimizer_plugin.step = recording_step

    metrics = runner.run()

    assert [row["step"] for row in metrics] == [0, 1]
    assert runner.rollout.config == rollout_config
    assert contexts == optimizer_contexts
    assert contexts == [
        StepContext(0, config.seed, 0, 0, 1, 0),
        StepContext(1, config.seed + 1, 1, 0, 1, 1),
    ]
    assert advantage_groups == [list(batch.group_id) for batch in batches]

    records = json.loads(
        (runner.output_dir / "sample_manifest.json").read_text(encoding="utf-8")
    )["records"]
    expected = [
        (
            sample_id,
            prompt_id,
            group_id,
            branch_id,
            batch.context.seed,
        )
        for batch in batches
        for sample_id, prompt_id, group_id, branch_id in zip(
            batch.sample_id,
            batch.prompt_id,
            batch.group_id,
            batch.branch_id,
            strict=True,
        )
    ]
    actual = [
        (
            record["sample_id"],
            record["prompt_id"],
            record["group_id"],
            record["branch_id"],
            record["seed"],
        )
        for record in records
    ]
    assert actual == expected


def test_reward_order_fails_before_cache_or_optimizer(tmp_path):
    runner = ExperimentRunner(_config(tmp_path / "bad-reward-order"))
    original_score = runner.feedback_provider.score
    calls = {"cache": 0, "optimizer": 0}

    def reversed_score(batch):
        rewards = original_score(batch)
        rewards.sample_id = list(reversed(rewards.sample_id))
        return rewards

    def cache_save(*args, **kwargs):
        calls["cache"] += 1
        return {}

    def optimizer_step(*args, **kwargs):
        calls["optimizer"] += 1
        return {}

    runner.feedback_provider.score = reversed_score
    runner.rollout_cache.save = cache_save
    runner.optimizer_plugin.step = optimizer_step

    with pytest.raises(ValueError, match="sample_id order"):
        runner.run()

    assert calls == {"cache": 0, "optimizer": 0}


def test_runner_canonicalizes_list_and_numpy_reward_fields_before_use(tmp_path):
    import torch

    runner = ExperimentRunner(_config(tmp_path / "canonical-rewards"))
    original_score = runner.feedback_provider.score
    original_step = runner.optimizer_plugin.step
    optimizer_calls = 0

    def list_score(batch):
        rewards = original_score(batch)
        return RewardBatch(
            raw={name: values.tolist() for name, values in rewards.raw.items()},
            weighted={
                name: values.detach().cpu().numpy()
                for name, values in rewards.weighted.items()
            },
            weighted_total=rewards.weighted_total.tolist(),
            valid_mask=rewards.valid_mask.tolist(),
            metadata=dict(rewards.metadata),
            sample_id=tuple(rewards.sample_id),
        )

    def checked_step(*args, **kwargs):
        nonlocal optimizer_calls
        optimizer_calls += 1
        rewards = kwargs["rewards"]
        assert all(
            isinstance(values, torch.Tensor) and values.device.type == "cpu"
            for values in rewards.raw.values()
        )
        assert rewards.weighted_total.is_floating_point()
        assert rewards.weighted_total.requires_grad is False
        assert rewards.valid_mask.dtype == torch.bool
        assert rewards.valid_mask.device.type == "cpu"
        assert isinstance(rewards.sample_id, list)
        return original_step(*args, **kwargs)

    runner.feedback_provider.score = list_score
    runner.optimizer_plugin.step = checked_step

    metrics = runner.run()

    assert len(metrics) == 1
    assert optimizer_calls == 1


def test_runner_rejects_stale_rollout_context_before_feedback(tmp_path):
    runner = ExperimentRunner(_config(tmp_path / "stale-context"))
    original_sample = runner.rollout.sample
    feedback_calls = 0

    def stale_sample(adapter, prompts, metadata, context):
        batch = original_sample(adapter, prompts, metadata, context)
        return batch.replace(
            context=StepContext(
                step=context.step,
                seed=context.seed,
                epoch_tag=context.epoch_tag,
                rank=context.rank,
                world_size=context.world_size,
                policy_version=context.policy_version + 1,
            )
        )

    def feedback_score(batch):
        nonlocal feedback_calls
        feedback_calls += 1
        raise AssertionError("feedback must not run for stale rollout context")

    runner.rollout.sample = stale_sample
    runner.feedback_provider.score = feedback_score

    with pytest.raises(ValueError, match="context must match"):
        runner.run()

    assert feedback_calls == 0


def test_feedback_requires_router_verified_ids_without_blind_fill():
    import torch

    batch = RolloutBatch(
        prompts=["prompt"],
        metadata=[{"sample_id": "sample-from-rollout"}],
    )

    class StubRouter:
        def __init__(self, sample_id):
            self.sample_id = sample_id

        def score(self, media, prompts, metadata, sample_id=None):
            return RewardBatch(
                raw={"reward": torch.tensor([1.0])},
                weighted={"reward": torch.tensor([1.0])},
                weighted_total=torch.tensor([1.0]),
                valid_mask=torch.tensor([True]),
                sample_id=self.sample_id,
            )

    provider = RewardRouterFeedbackProvider.__new__(RewardRouterFeedbackProvider)
    provider.reward_router = StubRouter(None)
    with pytest.raises(ValueError, match="returned no sample_id"):
        provider.score(batch)

    provider.reward_router = StubRouter(["declared-by-provider"])
    with pytest.raises(ValueError, match="sample_id order"):
        provider.score(batch)

    provider.reward_router = StubRouter(list(batch.sample_id))
    assert provider.score(batch).sample_id == batch.sample_id


def test_reward_router_identity_cache_round_trip_and_legacy_marker(tmp_path):
    class LegacyInputOrderClient:
        calls = 0

        def score(self, media, prompts, metadata):
            del media, metadata
            type(self).calls += 1
            return np.arange(len(prompts), dtype=np.float32), {"source": "test"}

    name = "c4_legacy_input_order"
    REWARD_CLIENTS.register(name, LegacyInputOrderClient)
    router = RewardRouter(
        {
            "weights": {name: 1.0},
            "clients": {name: {"name": name}},
        },
        cache_dir=tmp_path,
    )
    sample_id = ["sample-a", "sample-b"]

    first = router.score(None, ["a", "b"], [{}, {}], sample_id=sample_id)
    second = router.score(None, ["a", "b"], [{}, {}], sample_id=sample_id)

    assert first.sample_id == sample_id
    assert second.sample_id == sample_id
    assert LegacyInputOrderClient.calls == 1
    assert first.metadata[name]["sample_id_mode"] == "trusted_input_order_legacy"
    payload = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["sample_id"] == sample_id

    direct_legacy = router.score(None, ["a", "b"], [{}, {}])
    assert direct_legacy.sample_id is None
    assert (
        direct_legacy.metadata[name]["sample_id_mode"]
        == "trusted_input_order_legacy"
    )


def test_reward_cache_identity_isolated_and_mismatch_rejected(tmp_path):
    class CountingClient:
        calls = 0

        def score(self, media, prompts, metadata):
            del media, metadata
            type(self).calls += 1
            return np.ones(len(prompts), dtype=np.float32), {}

    name = "c4_cache_identity"
    REWARD_CLIENTS.register(name, CountingClient)
    router = RewardRouter(
        {
            "weights": {name: 1.0},
            "clients": {name: {"name": name}},
        },
        cache_dir=tmp_path,
    )
    prompts = ["same", "positions"]
    metadata = [{}, {}]

    router.score(None, prompts, metadata, sample_id=["a", "b"])
    router.score(None, prompts, metadata, sample_id=["c", "d"])

    assert CountingClient.calls == 2
    assert len(list(tmp_path.glob("*.json"))) == 2

    cache_key = router._cache_key(
        name,
        "v1",
        prompts,
        metadata,
        None,
        ["a", "b"],
        router.client_fingerprints[name],
    )
    cache_path = tmp_path / f"{cache_key}.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["sample_id"] = ["b", "a"]
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_id identity mismatch"):
        router.score(None, prompts, metadata, sample_id=["a", "b"])
    assert CountingClient.calls == 2


def test_explicit_reward_client_sample_id_order_fails_before_cache(tmp_path):
    class ReorderedClient:
        def score(self, media, prompts, metadata):
            del media, metadata
            return np.ones(len(prompts), dtype=np.float32), {
                "sample_id": ["sample-b", "sample-a"]
            }

    name = "c4_explicit_reordered"
    REWARD_CLIENTS.register(name, ReorderedClient)
    router = RewardRouter(
        {
            "weights": {name: 1.0},
            "clients": {name: {"name": name}},
        },
        cache_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="sample_id order"):
        router.score(
            None,
            ["a", "b"],
            [{}, {}],
            sample_id=["sample-a", "sample-b"],
        )
    assert list(tmp_path.glob("*.json")) == []


def test_formal_group_ids_and_legacy_manifest_compatibility():
    context = StepContext(3, 17, 2, policy_version=3)
    formal = RolloutBatch(
        prompts=["same", "same"],
        metadata=[
            {"parent_prompt_index": 10},
            {"parent_prompt_index": 11},
        ],
        group_id=["formal-group", "formal-group"],
        context=context,
    )
    assert AlgorithmOptimizerPlugin._advantage_group_ids(formal) == [
        "formal-group",
        "formal-group",
    ]

    legacy = RolloutBatch(
        prompts=["same", "same"],
        metadata=[
            {"parent_prompt_index": 4},
            {"parent_prompt_index": 4},
        ],
    )
    assert AlgorithmOptimizerPlugin._advantage_group_ids(legacy) == [4, 4]

    manifest = SampleManifest.migrate_legacy_to_v2(
        {
            "run_id": "legacy-run",
            "records": [
                {
                    "run_id": "legacy-run",
                    "sample_id": "legacy-sample",
                    "sample_index": 0,
                    "step": 0,
                    "prompt": "old prompt",
                    "media_type": "image",
                    "prompt_metadata": {},
                }
            ],
        }
    )
    record = manifest.records[0]
    assert record.prompt_id is None
    assert record.group_id is None
    assert record.branch_id is None
