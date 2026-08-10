"""Focused contracts for native local reward clients."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest
import torch

from visual_rl.algorithms.rewards.clients.image import (
    PromptColorGuardedRewardClient,
    PromptColorMarginRewardClient,
    PromptColorRewardClient,
)
from visual_rl.algorithms.rewards.clients.mock import MockRewardClient
from visual_rl.core.types import ResolutionContext, RuntimeBuildContext, StepContext
from visual_rl.algorithms.rewards import (
    PointwiseRewardOutput,
    RewardBatchIdentity,
    RewardBatchView,
    RewardRuntimeContext,
)
from visual_rl.data.samples import (
    BatchRowContext,
    NoConditionBatchState,
    SourceItemContext,
    StackedSampleBatch,
    TrajectoryBatch,
    TrajectoryContext,
)


def _runtime_context() -> RuntimeBuildContext:
    return RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cpu"),
        precision="fp32",
    )


def _resolution(tmp_path: Path) -> ResolutionContext:
    return ResolutionContext(
        config_path=(tmp_path / "config.yaml").resolve(),
        config_dir=tmp_path.resolve(),
    )


def _image_batch(media: torch.Tensor | None = None) -> RewardBatchView:
    context = StepContext(step=3, seed=17)
    batch_size = 4
    if media is None:
        media = torch.stack(
            (
                torch.tensor([0.8, 0.1, 0.1])[:, None, None].expand(3, 2, 2),
                torch.tensor([0.1, 0.8, 0.1])[:, None, None].expand(3, 2, 2),
                torch.tensor([0.1, 0.1, 0.8])[:, None, None].expand(3, 2, 2),
                torch.full((3, 2, 2), 0.5),
            )
        ).contiguous()
    rows = tuple(
        BatchRowContext(
            occurrence_id=f"occurrence-{row}",
            group_id=f"group-{row // 2}",
            member_id=row % 2,
            phase="main",
            optimizer_step=context.step,
            source_item_id=f"source-{row // 2}",
        )
        for row in range(batch_size)
    )
    sources = tuple(
        SourceItemContext(
            source_item_id=row.source_item_id,
            dataset_source_id="main",
            dataset_index=index // 2,
            dataset_revision="test-v1",
        )
        for index, row in enumerate(rows)
    )
    condition_state = NoConditionBatchState(batch_size)
    prompts = ("a red cube", "a green cube", "a blue cube", "plain object")
    samples = StackedSampleBatch(
        task_type="t2i",
        prompts=prompts,
        sources=sources,
        rows=rows,
        metadata=({}, {}, {}, {"target_color": "red"}),
        condition_state=condition_state,
    )
    contexts = tuple(
        TrajectoryContext(
            sample_id=f"sample-{row}",
            trajectory_id=f"trajectory-{row}",
            batch_row=rows[row],
        )
        for row in range(batch_size)
    )
    trajectory = TrajectoryBatch(
        kind="full_trajectory",
        contexts=contexts,
        x_t=torch.zeros(batch_size, 1, 1),
        sampled_action=torch.ones(batch_size, 1, 1),
        conditioned_next=torch.ones(batch_size, 1, 1),
        timesteps=torch.zeros(batch_size, 1),
        next_timesteps=torch.ones(batch_size, 1),
        old_log_probs=torch.zeros(batch_size, 1),
        transition_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        transition_index=torch.zeros(batch_size, 1, dtype=torch.int64),
        likelihood_semantics="exact_env_action",
        condition_identity=(("none",),) * batch_size,
        guidance_identity=(("cfg",),) * batch_size,
        storage_dtype_identity=(("torch.float32",),) * batch_size,
        quantization_identity=(("none",),) * batch_size,
        media=media,
        media_layout="BCHW",
        condition_state=condition_state,
    )
    return RewardBatchView(
        identity=RewardBatchIdentity(
            source_id="main",
            phase_id="main",
            batch_row_ids=tuple(row.identity for row in rows),
            sample_ids=tuple(item.sample_id for item in contexts),
            trajectory_ids=tuple(item.trajectory_id for item in contexts),
            condition_payload_ids=("none",) * batch_size,
            group_ids=tuple(row.group_id for row in rows),
        ),
        active_reward_ids=("quality",),
        payload={
            "trajectory": trajectory,
            "samples": samples,
            "reward_runtime_context": RewardRuntimeContext(context),
            "media": media,
            "condition_state": condition_state,
        },
    )


def test_builtin_mock_returns_canonical_pointwise_output(tmp_path: Path) -> None:
    resolved = MockRewardClient.resolve_params(
        {"mode": "prompt_media"},
        _resolution(tmp_path),
    )
    client = MockRewardClient.from_config(resolved, _runtime_context())
    batch = _image_batch()

    output = client.score(batch=batch)

    assert isinstance(output, PointwiseRewardOutput)
    assert output.identity is batch.identity
    assert output.values.shape == (batch.batch_size,)
    assert output.values.dtype == np.float64
    assert output.execution_provenance["shared_metadata"] == {
        "mode": "prompt_media"
    }
    with pytest.raises(ValueError, match="exactly"):
        MockRewardClient.resolve_params(
            {"mode": "constant", "legacy": True},
            _resolution(tmp_path),
        )


def test_prompt_color_rewards_use_the_same_typed_batch(tmp_path: Path) -> None:
    batch = _image_batch()
    for client_type in (PromptColorRewardClient, PromptColorMarginRewardClient):
        resolved = client_type.resolve_params(
            {"default_color": "red"},
            _resolution(tmp_path),
        )
        client = client_type.from_config(resolved, _runtime_context())
        output = client.score(batch=batch)
        assert output.identity is batch.identity
        assert output.values.shape == (batch.batch_size,)
        assert len(output.execution_provenance["sample_metadata"]) == batch.batch_size
        assert bool(np.isfinite(output.values).all())


def test_guarded_image_reward_is_finite_without_division_warnings(
    tmp_path: Path,
) -> None:
    resolved = PromptColorGuardedRewardClient.resolve_params(
        {
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
        },
        _resolution(tmp_path),
    )
    client = PromptColorGuardedRewardClient.from_config(
        resolved,
        _runtime_context(),
    )
    media = torch.stack(
        (
            torch.zeros(3, 2, 2),
            torch.ones(3, 2, 2),
            torch.tensor([1.0, 0.0, 0.0])[:, None, None].expand(3, 2, 2),
            torch.full((3, 2, 2), 1.0e-8),
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        output = client.score(batch=_image_batch(media))
    assert bool(np.isfinite(output.values).all())
    assert output.execution_provenance["shared_metadata"]["margin_clip"] == 0.5
