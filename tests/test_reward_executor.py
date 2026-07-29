"""Focused contracts for the one synchronous reward facade."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import pytest
import torch

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RewardVector,
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.feedback.base import RewardClient
from visual_rl.feedback.cache import RewardCache, reward_cache_key
from visual_rl.feedback.clients import MockRewardClient
from visual_rl.feedback.executor import RewardExecutionError, RewardExecutor
from visual_rl.feedback.image_rewards import (
    PromptColorGuardedRewardClient,
    PromptColorMarginRewardClient,
    PromptColorRewardClient,
)
from visual_rl.feedback.provider import (
    RewardClientBinding,
    RewardFeedbackProvider,
)


def _context() -> StepContext:
    return StepContext(step=3, seed=17, rank=0, world_size=1)


def _runtime_context() -> RuntimeBuildContext:
    return RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cpu"),
        precision="fp32",
    )


def _batch(
    context: StepContext,
    *,
    camera: torch.Tensor | None = None,
) -> RolloutBatch:
    batch_size = 4
    transitions = 2
    media = (
        torch.arange(
            batch_size * 3 * 3 * 2 * 2,
            dtype=torch.float32,
        ).reshape(batch_size, 3, 3, 2, 2)
        if camera is not None
        else torch.arange(
            batch_size * 3 * 2 * 2,
            dtype=torch.float32,
        ).reshape(batch_size, 3, 2, 2)
    )
    return RolloutBatch(
        prompts=("prompt-0", "prompt-0", "prompt-1", "prompt-1"),
        metadata=(
            {"source": 0},
            {"source": 0},
            {"source": 1},
            {"source": 1},
        ),
        media=media,
        latents=torch.zeros(batch_size, transitions, 1),
        next_latents=torch.ones(batch_size, transitions, 1),
        timesteps=torch.arange(transitions).repeat(batch_size, 1),
        old_log_probs=torch.zeros(batch_size, transitions),
        transition_mask=torch.ones(batch_size, transitions, dtype=torch.bool),
        sample_id=tuple(f"sample-{row}" for row in range(batch_size)),
        prompt_id=("prompt-id-0", "prompt-id-0", "prompt-id-1", "prompt-id-1"),
        group_id=("group-0", "group-0", "group-1", "group-1"),
        branch_id=(0, 1, 0, 1),
        media_layout="BFCHW" if camera is not None else "BCHW",
        camera_trajectory=camera,
        context=context,
        selected_timestep_index=None,
        flash_coefficient=None,
        branch_step_index=None,
        trajectory_step_index=None,
        transition_std_dev=None,
        recompute_payload={},
        artifact_metadata={},
    )


class _Client(RewardClient):
    def __init__(
        self,
        name: str,
        *,
        offset: float = 0.0,
        dtype: torch.dtype = torch.float64,
        shared_by_shard: bool = False,
        fail_sample: str | None = None,
    ) -> None:
        self.name = name
        self.offset = offset
        self.dtype = dtype
        self.shared_by_shard = shared_by_shard
        self.fail_sample = fail_sample
        self.calls: list[tuple[str, ...]] = []
        self.failures = 0
        self.close_calls = 0

    def score(self, batch: RolloutBatch, context: StepContext) -> RewardVector:
        assert batch.context is context
        self.calls.append(batch.sample_id)
        if (
            self.fail_sample is not None
            and self.fail_sample in batch.sample_id
            and self.failures == 0
        ):
            self.failures += 1
            raise RuntimeError("transient reward failure")
        values = torch.tensor(
            [
                float(sample.rsplit("-", 1)[-1]) + 1.0 + self.offset
                for sample in batch.sample_id
            ],
            dtype=self.dtype,
        )
        shared = {
            "revision": (
                batch.sample_id[0] if self.shared_by_shard else "fixed-revision"
            )
        }
        return RewardVector(
            sample_id=batch.sample_id,
            values=values,
            shared_metadata=shared,
            sample_metadata=tuple(
                {"evidence": sample} for sample in batch.sample_id
            ),
        )

    def close(self) -> None:
        self.close_calls += 1


class _ReorderedClient(_Client):
    def score(self, batch: RolloutBatch, context: StepContext) -> RewardVector:
        vector = super().score(batch, context)
        return RewardVector(
            sample_id=tuple(reversed(vector.sample_id)),
            values=vector.values.flip(0).contiguous(),
            shared_metadata=vector.shared_metadata,
            sample_metadata=tuple(reversed(vector.sample_metadata)),
        )


def _binding(
    name: str,
    client: RewardClient,
    *,
    weight: float,
    params: dict | None = None,
) -> RewardClientBinding:
    return RewardClientBinding(
        name=name,
        client=client,
        weight=weight,
        resolved_params=FrozenMapping(params or {"revision": "v1"}),
    )


def test_executor_serially_shards_normalizes_weights_and_preserves_order() -> None:
    context = _context()
    batch = _batch(context)
    quality = _Client("quality", offset=0.0, dtype=torch.float64)
    safety = _Client("safety", offset=10.0, dtype=torch.float16)
    provider = RewardFeedbackProvider(
        clients=(
            _binding("quality", quality, weight=0.5),
            _binding("safety", safety, weight=-0.25),
        ),
        cache=None,
    )
    executor = RewardExecutor(
        provider=provider,
        microbatch_size=2,
        max_retries=0,
    )

    rewards = executor.score(batch, context)

    assert quality.calls == [
        ("sample-0", "sample-1"),
        ("sample-2", "sample-3"),
    ]
    assert safety.calls == quality.calls
    assert tuple(rewards.raw) == ("quality", "safety")
    assert rewards.sample_id == batch.sample_id
    assert rewards.raw["quality"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert rewards.raw["safety"].tolist() == [11.0, 12.0, 13.0, 14.0]
    torch.testing.assert_close(
        rewards.weighted_total,
        rewards.raw["quality"] * 0.5 + rewards.raw["safety"] * -0.25,
    )
    for mapping in (rewards.raw, rewards.weighted):
        for value in mapping.values():
            assert value.device.type == "cpu"
            assert value.dtype == torch.float32
            assert value.is_contiguous()
            assert value.requires_grad is False
            assert value.grad_fn is None
    assert rewards.sample_metadata["quality"][2]["evidence"] == "sample-2"


def test_second_shard_retry_is_serial_and_reuses_identical_input() -> None:
    context = _context()
    batch = _batch(context)
    client = _Client("quality", fail_sample="sample-2")
    provider = RewardFeedbackProvider(
        clients=(_binding("quality", client, weight=1.0),),
        cache=None,
    )
    executor = RewardExecutor(
        provider=provider,
        microbatch_size=2,
        max_retries=1,
    )

    rewards = executor.score(batch, context)

    assert rewards.sample_id == batch.sample_id
    assert client.calls == [
        ("sample-0", "sample-1"),
        ("sample-2", "sample-3"),
        ("sample-2", "sample-3"),
    ]


def test_final_retry_failure_is_wrapped_before_any_reward_batch() -> None:
    context = _context()
    batch = _batch(context)

    class AlwaysFail(_Client):
        def score(self, batch, context):
            self.calls.append(batch.sample_id)
            raise RuntimeError("still unavailable")

    client = AlwaysFail("quality")
    executor = RewardExecutor(
        provider=RewardFeedbackProvider(
            clients=(_binding("quality", client, weight=1.0),),
            cache=None,
        ),
        microbatch_size=2,
        max_retries=2,
    )

    with pytest.raises(RewardExecutionError) as caught:
        executor.score(batch, context)

    assert caught.value.shard_index == 0
    assert caught.value.attempts == 3
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert len(client.calls) == 3


def test_reordered_client_result_fails_before_assembly() -> None:
    context = _context()
    batch = _batch(context)
    client = _ReorderedClient("quality")
    executor = RewardExecutor(
        provider=RewardFeedbackProvider(
            clients=(_binding("quality", client, weight=1.0),),
            cache=None,
        ),
        microbatch_size=None,
        max_retries=0,
    )

    with pytest.raises(RewardExecutionError) as caught:
        executor.score(batch, context)

    assert isinstance(caught.value.__cause__, ValueError)
    assert "sample_id order" in str(caught.value.__cause__)


def test_cross_shard_shared_metadata_conflict_is_rejected() -> None:
    context = _context()
    batch = _batch(context)
    client = _Client("quality", shared_by_shard=True)
    executor = RewardExecutor(
        provider=RewardFeedbackProvider(
            clients=(_binding("quality", client, weight=1.0),),
            cache=None,
        ),
        microbatch_size=2,
        max_retries=0,
    )

    with pytest.raises(ValueError, match="shared_metadata is inconsistent"):
        executor.score(batch, context)


def test_context_must_be_the_identical_object() -> None:
    context = _context()
    batch = _batch(context)
    equal_but_distinct = StepContext(
        step=context.step,
        seed=context.seed,
        rank=context.rank,
        world_size=context.world_size,
    )
    executor = RewardExecutor(
        provider=RewardFeedbackProvider(
            clients=(_binding("quality", _Client("quality"), weight=1.0),),
            cache=None,
        ),
        microbatch_size=None,
        max_retries=0,
    )

    with pytest.raises(ValueError, match="identical StepContext"):
        executor.score(batch, equal_but_distinct)


def test_cache_hit_reuses_raw_vector_but_reapplies_current_weight(
    tmp_path: Path,
) -> None:
    context = _context()
    batch = _batch(context)
    cache = RewardCache(tmp_path / "reward-cache")
    first_client = _Client("quality")
    first = RewardFeedbackProvider(
        clients=(_binding("quality", first_client, weight=1.0),),
        cache=cache,
    ).score(batch, context)
    assert first_client.calls == [batch.sample_id]

    class MustNotRun(_Client):
        def score(self, batch, context):
            raise AssertionError("cache hit called the client")

    second = RewardFeedbackProvider(
        clients=(_binding("quality", MustNotRun("quality"), weight=-2.0),),
        cache=cache,
    ).score(batch, context)

    torch.testing.assert_close(second.raw["quality"], first.raw["quality"])
    torch.testing.assert_close(
        second.weighted_total,
        first.raw["quality"] * -2.0,
    )


def test_cache_key_covers_params_and_only_3d_camera() -> None:
    context = _context()
    camera = torch.eye(4, dtype=torch.float64).repeat(4, 3, 1, 1)
    batch = _batch(context, camera=camera)
    changed = replace(
        batch,
        camera_trajectory=camera.add(1.0),
    )
    params = FrozenMapping({"revision": "v1"})
    changed_params = FrozenMapping({"revision": "v2"})

    general = reward_cache_key(
        component_name="reward_general",
        resolved_params=params,
        batch=batch,
        context=context,
        row=0,
    )
    general_camera_changed = reward_cache_key(
        component_name="reward_general",
        resolved_params=params,
        batch=changed,
        context=context,
        row=0,
    )
    general_params_changed = reward_cache_key(
        component_name="reward_general",
        resolved_params=changed_params,
        batch=batch,
        context=context,
        row=0,
    )
    reward_3d = reward_cache_key(
        component_name="reward_3d",
        resolved_params=params,
        batch=batch,
        context=context,
        row=0,
    )
    reward_3d_camera_changed = reward_cache_key(
        component_name="reward_3d",
        resolved_params=params,
        batch=changed,
        context=context,
        row=0,
    )

    assert general == general_camera_changed
    assert general != general_params_changed
    assert reward_3d != reward_3d_camera_changed


def test_close_is_idempotent_and_never_recurses_into_injected_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client("quality")
    cache = RewardCache(tmp_path / "cache")
    cache_close_calls = 0

    def count_cache_close() -> None:
        nonlocal cache_close_calls
        cache_close_calls += 1

    monkeypatch.setattr(cache, "close", count_cache_close)
    provider = RewardFeedbackProvider(
        clients=(_binding("quality", client, weight=1.0),),
        cache=cache,
    )
    executor = RewardExecutor(
        provider=provider,
        microbatch_size=None,
        max_retries=0,
    )

    executor.close()
    executor.close()
    provider.close()
    provider.close()

    assert client.close_calls == 0
    assert cache_close_calls == 0
    with pytest.raises(RewardExecutionError, match="closed"):
        executor.score(_batch(_context()), _context())


def test_executor_source_contains_no_async_or_split_lifecycle() -> None:
    import visual_rl.feedback.executor as module

    source = inspect.getsource(module)
    for forbidden in (
        "AsyncRewardExecutor",
        "SyncRewardExecutor",
        "RewardHandle",
        "ThreadPoolExecutor",
        "def submit(",
        "def collect(",
        "Future",
        "Semaphore",
    ):
        assert forbidden not in source


def test_all_cache_keys_are_validated_before_the_first_client_runs(
    tmp_path: Path,
) -> None:
    context = _context()
    batch = _batch(context)
    first = _Client("quality")
    provider = RewardFeedbackProvider(
        clients=(
            _binding("quality", first, weight=1.0),
            _binding("reward_3d", _Client("reward_3d"), weight=1.0),
        ),
        cache=RewardCache(tmp_path / "cache"),
    )

    with pytest.raises(ValueError, match="requires camera_trajectory"):
        provider.score(batch, context)

    assert first.calls == []


def test_builtin_mock_uses_exact_final_contract(tmp_path: Path) -> None:
    context = _context()
    batch = _batch(context)
    resolution = ResolutionContext(
        config_path=(tmp_path / "config.yaml").resolve(),
        config_dir=tmp_path.resolve(),
    )
    resolved = MockRewardClient.resolve_params(
        {"mode": "prompt_media"},
        resolution,
    )
    client = MockRewardClient.from_config(resolved, _runtime_context())

    vector = client.score(batch, context)

    assert isinstance(client, RewardClient)
    assert vector.sample_id == batch.sample_id
    assert vector.values.shape == (batch.batch_size,)
    assert vector.values.dtype == torch.float32
    assert vector.shared_metadata == {"mode": "prompt_media"}
    with pytest.raises(ValueError, match="exactly"):
        MockRewardClient.resolve_params(
            {"mode": "constant", "legacy": True},
            resolution,
        )


def test_image_reward_params_and_scores_use_one_batch_contract(
    tmp_path: Path,
) -> None:
    context = _context()
    batch = _batch(context)
    resolution = ResolutionContext(
        config_path=(tmp_path / "config.yaml").resolve(),
        config_dir=tmp_path.resolve(),
    )
    for client_type in (
        PromptColorRewardClient,
        PromptColorMarginRewardClient,
    ):
        resolved = client_type.resolve_params(
            {"default_color": "red"},
            resolution,
        )
        client = client_type.from_config(resolved, _runtime_context())
        vector = client.score(batch, context)
        assert isinstance(client, RewardClient)
        assert vector.sample_id == batch.sample_id
        assert vector.values.dtype == torch.float32
        assert len(vector.sample_metadata) == batch.batch_size

    guarded_params = {
        "default_color": "red",
        "margin_clip": 0.5,
        "saturation_max": 0.9,
        "luminance_min": 0.1,
        "luminance_max": 0.9,
        "spatial_std_min": 0.05,
        "spatial_std_max": 0.5,
        "saturation_penalty_weight": 1.0,
        "luminance_penalty_weight": 1.0,
        "spatial_penalty_weight": 1.0,
    }
    resolved = PromptColorGuardedRewardClient.resolve_params(
        guarded_params,
        resolution,
    )
    guarded = PromptColorGuardedRewardClient.from_config(
        resolved,
        _runtime_context(),
    )
    vector = guarded.score(batch, context)
    assert vector.values.shape == (batch.batch_size,)
    assert vector.shared_metadata["margin_clip"] == 0.5
