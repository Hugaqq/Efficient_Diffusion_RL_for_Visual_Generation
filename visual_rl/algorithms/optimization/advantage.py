"""Canonical reward-to-advantage normalization for policy optimization.

This module owns the typed grouping identity and detached advantage tensor.
It intentionally has no dependency on an algorithm implementation, policy
recompute graph, or optimizer runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from visual_rl.algorithms.rewards.types import RewardResult
from visual_rl.data.samples.trajectory import TrajectoryBatch


def _identity_tuple(
    value: object,
    *,
    field_name: str,
    size: int | None = None,
    unique: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if size is not None and len(value) != size:
        raise ValueError(f"{field_name} must have shape [{size}]")
    if unique and len(set(value)) != len(value):
        raise ValueError(f"{field_name} must contain unique identities")
    return value


def _trajectory_identities(
    trajectory: TrajectoryBatch,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(trajectory, TrajectoryBatch):
        raise TypeError("trajectory must be a TrajectoryBatch")
    return (
        tuple(context.batch_row_identity for context in trajectory.contexts),
        tuple(context.sample_id for context in trajectory.contexts),
        tuple(context.trajectory_id for context in trajectory.contexts),
        tuple(context.batch_row.group_id for context in trajectory.contexts),
    )


@dataclass(frozen=True, slots=True)
class AdvantageGrouping:
    """Immutable row identities and the exact normalization group key."""

    batch_row_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = _identity_tuple(
            self.batch_row_ids,
            field_name="batch_row_ids",
            unique=True,
        )
        size = len(rows)
        _identity_tuple(
            self.sample_ids,
            field_name="sample_ids",
            size=size,
            unique=True,
        )
        _identity_tuple(
            self.trajectory_ids,
            field_name="trajectory_ids",
            size=size,
            unique=True,
        )
        _identity_tuple(
            self.group_ids,
            field_name="group_ids",
            size=size,
        )

    @property
    def batch_size(self) -> int:
        return len(self.batch_row_ids)

    @classmethod
    def from_trajectory(cls, trajectory: TrajectoryBatch) -> AdvantageGrouping:
        rows, samples, trajectories, groups = _trajectory_identities(trajectory)
        return cls(
            batch_row_ids=rows,
            sample_ids=samples,
            trajectory_ids=trajectories,
            group_ids=groups,
        )

    def validate_against_trajectory(self, trajectory: TrajectoryBatch) -> None:
        observed = _trajectory_identities(trajectory)
        expected = (
            self.batch_row_ids,
            self.sample_ids,
            self.trajectory_ids,
            self.group_ids,
        )
        if observed != expected:
            raise ValueError("advantage grouping identity does not match trajectory")

    def validate_against_reward(self, reward: RewardResult) -> None:
        if not isinstance(reward, RewardResult):
            raise TypeError("reward must be a RewardResult")
        identity = reward.identity
        observed = (
            identity.batch_row_ids,
            identity.sample_ids,
            identity.trajectory_ids,
            identity.group_ids,
        )
        expected = (
            self.batch_row_ids,
            self.sample_ids,
            self.trajectory_ids,
            self.group_ids,
        )
        if observed != expected:
            raise ValueError("advantage grouping identity does not match reward")

    def select_rows(self, row_indices: tuple[int, ...]) -> AdvantageGrouping:
        """Select execution rows after full-group advantage normalization."""

        if type(row_indices) is not tuple or not row_indices:
            raise ValueError("row_indices must be a non-empty tuple")
        if any(type(index) is not int for index in row_indices):
            raise TypeError("row_indices must contain integers")
        if len(set(row_indices)) != len(row_indices):
            raise ValueError("row_indices must not contain duplicates")
        if any(index < 0 or index >= self.batch_size for index in row_indices):
            raise IndexError("advantage grouping row index is out of range")
        return AdvantageGrouping(
            batch_row_ids=tuple(self.batch_row_ids[index] for index in row_indices),
            sample_ids=tuple(self.sample_ids[index] for index in row_indices),
            trajectory_ids=tuple(self.trajectory_ids[index] for index in row_indices),
            group_ids=tuple(self.group_ids[index] for index in row_indices),
        )


@dataclass(frozen=True, slots=True)
class NormalizedAdvantage:
    """Detached ``[B, *score_axes]`` advantage and reward validity."""

    grouping: AdvantageGrouping
    values: Any
    valid_mask: Any
    score_axis_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.grouping, AdvantageGrouping):
            raise TypeError("grouping must be an AdvantageGrouping")
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if not isinstance(self.valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a torch.Tensor")
        if type(self.score_axis_names) is not tuple or any(
            item != "branch_timestep" for item in self.score_axis_names
        ):
            raise ValueError("normalized advantage has unknown score axes")
        if len(self.score_axis_names) != len(set(self.score_axis_names)):
            raise ValueError("normalized advantage score axes must be unique")
        if self.values.ndim != 1 + len(self.score_axis_names):
            raise ValueError("values rank must match score_axis_names")
        shape = tuple(self.values.shape)
        if shape[0] != self.grouping.batch_size:
            raise ValueError("values first dimension must equal B")
        if not self.values.is_floating_point():
            raise ValueError("values must be floating point")
        if tuple(self.valid_mask.shape) != shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with the advantage shape")
        if self.values.device != self.valid_mask.device:
            raise ValueError("values and valid_mask must share one device")
        if self.values.requires_grad or self.values.grad_fn is not None:
            raise ValueError("normalized advantage must be detached")
        safe_values = torch.where(self.valid_mask, self.values, 0.0)
        if not bool(torch.isfinite(safe_values).all()):
            raise ValueError("normalized advantage must be finite at valid rows")
        if not bool(self.valid_mask.any()):
            raise ValueError("normalized advantage requires at least one valid row")

    @property
    def batch_size(self) -> int:
        return self.grouping.batch_size

    def validate_against_trajectory(self, trajectory: TrajectoryBatch) -> None:
        self.grouping.validate_against_trajectory(trajectory)
        if self.score_axis_names == ():
            if tuple(self.values.shape) != (trajectory.batch_size,):
                raise ValueError("row advantage must have shape [B]")
            return
        if self.score_axis_names != ("branch_timestep",):
            raise ValueError("unsupported advantage score axes")
        topology = trajectory.branch_topology
        if (
            trajectory.kind != "branching"
            or topology is None
            or topology.kind != "every_policy_timestep"
        ):
            raise ValueError(
                "branch_timestep advantage requires TempFlow paper topology"
            )
        if tuple(self.values.shape) != (
            trajectory.batch_size,
            trajectory.transition_count,
        ):
            raise ValueError("branch_timestep advantage must have shape [B,T]")


class GroupZScoreAdvantageProcessor:
    """Normalize weighted reward once within each explicit occurrence group."""

    def __init__(
        self,
        *,
        epsilon: float = 1e-8,
        std_domain: Literal["group", "batch"] = "group",
        output_dtype: Literal["float32", "float64"] = "float32",
    ) -> None:
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a finite positive number")
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if std_domain not in {"group", "batch"}:
            raise ValueError("std_domain must be group or batch")
        self.std_domain = std_domain
        if output_dtype not in {"float32", "float64"}:
            raise ValueError("output_dtype must be float32 or float64")
        self.output_dtype = output_dtype

    def normalize(
        self,
        reward: RewardResult,
        grouping: AdvantageGrouping,
        *,
        device: object = "cpu",
    ) -> NormalizedAdvantage:
        """Return one detached tensor; invalid score cells remain masked at zero."""

        import torch

        if not isinstance(reward, RewardResult):
            raise TypeError("reward must be a RewardResult")
        if not isinstance(grouping, AdvantageGrouping):
            raise TypeError("grouping must be an AdvantageGrouping")
        grouping.validate_against_reward(reward)
        if reward.weighted_total.shape[0] != grouping.batch_size:
            raise ValueError("reward weighted_total first dimension must equal B")

        # Reward execution is NumPy-only. Normalize in float64 before the one
        # explicit conversion to the credit strategy's declared output dtype.
        values = np.array(reward.weighted_total, dtype=np.float64, copy=True)
        valid = np.array(reward.valid_mask, dtype=np.bool_, copy=True)
        if not bool(valid.any()):
            raise ValueError("advantage normalization requires a valid reward row")
        normalized = np.zeros_like(values, dtype=np.float64)
        group_rows: dict[str, list[int]] = {}
        for row, group_id in enumerate(grouping.group_ids):
            if bool(np.any(valid[row])):
                group_rows.setdefault(group_id, []).append(row)

        declared_groups = tuple(dict.fromkeys(grouping.group_ids))
        missing_groups = tuple(
            group_id for group_id in declared_groups if group_id not in group_rows
        )
        if missing_groups:
            raise ValueError(
                f"every advantage group requires valid rewards: {missing_groups}"
            )
        score_cell_shape = values.shape[1:]
        batch_std: dict[tuple[int, ...], float] = {}
        if self.std_domain == "batch":
            for score_cell in np.ndindex(score_cell_shape or (1,)):
                cell = () if not score_cell_shape else score_cell
                valid_rows = np.flatnonzero(valid[(slice(None), *cell)])
                if valid_rows.size < 2:
                    raise ValueError(
                        "batch-standard-deviation advantage normalization requires "
                        f"at least two valid rows per score cell: cell={cell!r}"
                    )
                batch_std[cell] = float(
                    values[(valid_rows.astype(np.int64), *cell)].std(ddof=0)
                )
        for group_id, declared_rows in group_rows.items():
            for score_cell in np.ndindex(score_cell_shape or (1,)):
                cell = () if not score_cell_shape else score_cell
                valid_rows = [row for row in declared_rows if bool(valid[(row, *cell)])]
                if len(valid_rows) < 2:
                    raise ValueError(
                        "group z-score requires at least two valid rows per "
                        f"score cell: group={group_id!r}, cell={cell!r}"
                    )
                row_index = np.asarray(valid_rows, dtype=np.int64)
                group_values = values[(row_index, *cell)]
                mean = float(group_values.mean())
                std = (
                    batch_std[cell]
                    if self.std_domain == "batch"
                    else float(group_values.std(ddof=0))
                )
                normalized[(row_index, *cell)] = (group_values - mean) / (
                    std + self.epsilon
                )
        if not bool(np.isfinite(normalized).all()):
            raise ValueError("normalized advantage contains a non-finite value")

        dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
        }[self.output_dtype]
        value_tensor = torch.as_tensor(
            normalized,
            dtype=dtype,
            device=device,
        ).detach()
        valid_tensor = torch.as_tensor(
            valid,
            dtype=torch.bool,
            device=device,
        ).detach()
        return NormalizedAdvantage(
            grouping=grouping,
            values=value_tensor,
            valid_mask=valid_tensor,
            score_axis_names=reward.score_axis_names,
        )

    __call__ = normalize


__all__ = (
    "AdvantageGrouping",
    "GroupZScoreAdvantageProcessor",
    "NormalizedAdvantage",
)
