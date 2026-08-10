"""Lossless projection from a routed reward batch to physical client inputs.

The projection deliberately exposes only reward-facing values.  Policy
latents, likelihood tensors, and optimizer facts never cross this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from visual_rl.core.types import FrozenMapping, StepContext, to_plain_dict
from visual_rl.algorithms.rewards.types import (
    PointwiseRewardOutput,
    RewardBatchView,
    RewardRuntimeContext,
)
from visual_rl.data.samples import (
    CameraConditionBatchState,
    StackedSampleBatch,
    TrajectoryBatch,
)

__all__ = (
    "PointwiseRewardInput",
    "pointwise_reward_output",
    "resolve_pointwise_reward_input",
)


@dataclass(frozen=True, slots=True)
class PointwiseRewardInput:
    """Reward-only row projection, flattened in batch-row-major order."""

    batch: RewardBatchView
    sample_ids: tuple[str, ...]
    prompts: tuple[str, ...]
    metadata: tuple[Mapping[str, object], ...]
    media: Any
    media_layout: str
    context: StepContext
    camera_trajectory: Any | None = None

    @property
    def flat_size(self) -> int:
        return math.prod(self.batch.score_shape)


def resolve_pointwise_reward_input(batch: RewardBatchView) -> PointwiseRewardInput:
    """Validate identities and expose the exact physical scoring rows."""

    if not isinstance(batch, RewardBatchView):
        raise TypeError("batch must be a RewardBatchView")
    trajectory = batch.payload.get("trajectory")
    samples = batch.payload.get("samples")
    runtime_context = batch.payload.get("reward_runtime_context")
    if not isinstance(trajectory, TrajectoryBatch):
        raise TypeError("reward payload requires a TrajectoryBatch")
    if not isinstance(samples, StackedSampleBatch):
        raise TypeError("reward payload requires a StackedSampleBatch")
    if not isinstance(runtime_context, RewardRuntimeContext):
        raise TypeError("reward payload requires RewardRuntimeContext")
    if trajectory.batch_size != batch.batch_size or samples.batch_size != batch.batch_size:
        raise ValueError("reward trajectory and samples must have batch size B")

    sample_ids = tuple(item.sample_id for item in trajectory.contexts)
    if sample_ids != batch.identity.sample_ids:
        raise ValueError("reward trajectory sample identity does not match rows")
    if tuple(row.identity for row in samples.rows) != batch.identity.batch_row_ids:
        raise ValueError("reward samples do not match batch row identities")
    if tuple(item.trajectory_id for item in trajectory.contexts) != (
        batch.identity.trajectory_ids
    ):
        raise ValueError("reward trajectory identities do not match batch identity")
    if tuple(item.batch_row.group_id for item in trajectory.contexts) != (
        batch.identity.group_ids
    ):
        raise ValueError("reward trajectory group identities do not match batch identity")
    if tuple(source.source_item_id for source in samples.sources) != tuple(
        item.batch_row.source_item_id for item in trajectory.contexts
    ):
        raise ValueError("reward trajectory changed source item identities")
    camera = None
    if isinstance(trajectory.condition_state, CameraConditionBatchState):
        camera = trajectory.condition_state.camera_trajectory

    if not batch.score_axis_names:
        return PointwiseRewardInput(
            batch=batch,
            sample_ids=sample_ids,
            prompts=samples.prompts,
            metadata=samples.metadata,
            media=trajectory.media,
            media_layout=trajectory.media_layout,
            context=runtime_context.step_context,
            camera_trajectory=camera,
        )

    if batch.score_axis_names != ("branch_timestep",):
        raise NotImplementedError("unsupported pointwise reward score axes")
    topology = trajectory.branch_topology
    if (
        trajectory.kind != "branching"
        or topology is None
        or topology.kind != "every_policy_timestep"
    ):
        raise ValueError("branch_timestep reward axis requires TempFlow paper topology")
    terminal = trajectory.transition_terminal_media
    if terminal is None:
        raise TypeError("branch_timestep reward requires terminal media")
    batch_size, transition_count = batch.score_shape
    if tuple(terminal.shape[:2]) != (batch_size, transition_count):
        raise ValueError("terminal reward media does not match score axes")
    media_layout = {
        "BTCHW": "BCHW",
        "BTFCHW": "BFCHW",
        "BTFHWC": "BFHWC",
    }.get(trajectory.transition_terminal_media_layout)
    if media_layout is None:
        raise ValueError("unknown transition terminal media layout")
    flat_size = batch_size * transition_count

    def repeat_rows(values: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            values[row]
            for row in range(batch_size)
            for _ in range(transition_count)
        )

    return PointwiseRewardInput(
        batch=batch,
        sample_ids=tuple(
            f"{sample_ids[row]}::branch-timestep-{step}"
            for row in range(batch_size)
            for step in range(transition_count)
        ),
        prompts=repeat_rows(samples.prompts),
        metadata=repeat_rows(samples.metadata),
        media=terminal.reshape(flat_size, *terminal.shape[2:]),
        media_layout=media_layout,
        context=runtime_context.step_context,
        camera_trajectory=None,
    )


def pointwise_reward_output(
    resolved: PointwiseRewardInput,
    scores: Sequence[float] | np.ndarray,
    *,
    shared_metadata: Mapping[str, object],
    sample_metadata: Sequence[Mapping[str, object]],
) -> PointwiseRewardOutput:
    """Build the canonical typed output and record the exact row mapping."""

    if not isinstance(resolved, PointwiseRewardInput):
        raise TypeError("resolved must be a PointwiseRewardInput")
    flat_values = np.asarray(scores, dtype=np.float64)
    if flat_values.shape != (resolved.flat_size,):
        raise ValueError("reward resource returned an invalid score shape")
    records = tuple(sample_metadata)
    if len(records) != resolved.flat_size or any(
        not isinstance(item, Mapping) for item in records
    ):
        raise ValueError("sample_metadata must contain one mapping per score cell")
    batch = resolved.batch
    axis_coordinates = (
        tuple(np.ndindex(batch.score_axis_sizes))
        if batch.score_axis_sizes
        else ((),)
    )
    score_cells: list[dict[str, object]] = []
    flat_index = 0
    for batch_row_index in range(batch.batch_size):
        for coordinate in axis_coordinates:
            score_cells.append(
                {
                    "flat_index": flat_index,
                    "batch_row_index": batch_row_index,
                    "batch_row_id": batch.identity.batch_row_ids[batch_row_index],
                    "base_sample_id": batch.identity.sample_ids[batch_row_index],
                    "flattened_sample_id": resolved.sample_ids[flat_index],
                    "score_axis_indices": {
                        name: int(index)
                        for name, index in zip(
                            batch.score_axis_names,
                            coordinate,
                            strict=True,
                        )
                    },
                }
            )
            flat_index += 1
    return PointwiseRewardOutput(
        identity=batch.identity,
        values=flat_values.reshape(batch.score_shape),
        valid_mask=np.ones(batch.score_shape, dtype=np.bool_),
        score_axis_names=batch.score_axis_names,
        execution_provenance=FrozenMapping(
            {
                "shared_metadata": to_plain_dict(shared_metadata),
                "sample_metadata": [to_plain_dict(item) for item in records],
                "flattened_sample_ids": resolved.sample_ids,
                "score_axis_names": batch.score_axis_names,
                "score_shape": batch.score_shape,
                "flattening_order": "batch_row_major_c_order",
                "score_cells": score_cells,
            }
        ),
    )
