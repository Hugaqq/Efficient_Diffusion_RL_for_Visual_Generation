"""Mutable rollout collection and the single immutable trajectory build boundary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from visual_rl.algorithms.dynamics.interface import TransitionRecord
from visual_rl.algorithms.rollout.interface import (
    ModelForwardReplayPlan,
    RolloutContractError,
    RolloutRequest,
)
from visual_rl.data.media import DecodedMediaBatch
from visual_rl.data.samples import (
    BranchingTrajectoryItem,
    BranchTopology,
    ConditionPayload,
    ExplicitCollator,
    FullTrajectoryItem,
    SingleStepTrajectoryItem,
    TrajectoryBatch,
    TrajectoryContext,
    TrajectoryStep,
)

__all__ = (
    "RolloutStrategyResult",
    "TrajectoryRowBuilder",
    "build_trajectory_batch",
    "resolve_item_media_layout",
)

RolloutStrategyKind = Literal["full-trajectory", "branching", "single-step"]


@dataclass(frozen=True, slots=True)
class RolloutStrategyResult:
    """Detached control result consumed exactly once by the trajectory builder."""

    strategy: RolloutStrategyKind
    steps_by_row: tuple[tuple[TrajectoryStep, ...], ...]
    final_latents: Any
    strategy_identity: str | None = None
    branch_topology: BranchTopology | None = None
    transition_terminal_media: Any | None = None
    transition_terminal_media_layout: str | None = None
    branch_step_index: int | None = None
    selected_timestep_index: tuple[int, ...] | None = None
    shared_prefix_id: tuple[str, ...] | None = None
    branch_step_identity: tuple[str, ...] | None = None
    selection_policy_identity: str | None = None
    selection_mapping_identity: str | None = None
    schedule_snapshot_identity: str | None = None
    model_forward_replay: ModelForwardReplayPlan | None = None

    def __post_init__(self) -> None:
        if self.strategy not in {"full-trajectory", "branching", "single-step"}:
            raise ValueError("unsupported rollout strategy result")
        if type(self.steps_by_row) is not tuple or not self.steps_by_row:
            raise ValueError("steps_by_row must be a non-empty tuple")
        if any(type(row) is not tuple or not row for row in self.steps_by_row):
            raise ValueError("every strategy row must contain a policy transition")
        if any(
            not isinstance(step, TrajectoryStep)
            for row in self.steps_by_row
            for step in row
        ):
            raise TypeError("steps_by_row must contain TrajectoryStep values")
        if self.final_latents is None:
            raise ValueError("final_latents must not be None")


class TrajectoryRowBuilder:
    """Own mutable row writes before freezing data-owned trajectory steps."""

    __slots__ = ("_rows", "_sealed")

    def __init__(self, batch_size: int) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._rows: list[list[TrajectoryStep]] = [[] for _ in range(batch_size)]
        self._sealed = False

    @property
    def batch_size(self) -> int:
        return len(self._rows)

    def append_record(
        self,
        record: TransitionRecord,
        *,
        row_indices: tuple[int, ...] | None = None,
        storage_device: str = "model",
    ) -> None:
        if self._sealed:
            raise RuntimeError("trajectory row builder is already sealed")
        if not isinstance(record, TransitionRecord):
            raise TypeError("record must be a TransitionRecord")
        if self.batch_size != record.batch_size:
            raise RolloutContractError("compact step rows must match record batch size")
        selected = (
            tuple(range(record.batch_size)) if row_indices is None else row_indices
        )
        if type(selected) is not tuple or not selected:
            raise ValueError("compact record projection requires at least one row")
        if len(set(selected)) != len(selected):
            raise ValueError("compact record row indices must be unique")
        for row_index in selected:
            self._rows[row_index].append(
                _trajectory_step(
                    record,
                    row_index,
                    storage_device=storage_device,
                )
            )

    def freeze(
        self,
        *,
        expected_steps_per_row: int | None = None,
    ) -> tuple[tuple[TrajectoryStep, ...], ...]:
        if self._sealed:
            raise RuntimeError("trajectory row builder is already sealed")
        if expected_steps_per_row is not None and (
            type(expected_steps_per_row) is not int or expected_steps_per_row < 1
        ):
            raise ValueError("expected_steps_per_row must be positive or None")
        if any(not row for row in self._rows):
            raise RolloutContractError(
                "every trajectory row must contain at least one policy step"
            )
        if expected_steps_per_row is not None and any(
            len(row) != expected_steps_per_row for row in self._rows
        ):
            raise RolloutContractError(
                "trajectory rows do not match the expected stored step count"
            )
        self._sealed = True
        return tuple(tuple(row) for row in self._rows)


def _trajectory_step(
    record: TransitionRecord,
    row_index: int,
    *,
    storage_device: str,
) -> TrajectoryStep:
    """Project one dynamics record row onto the compact data-owned DTO."""

    if type(row_index) is not int or not 0 <= row_index < record.batch_size:
        raise IndexError("trajectory record row is out of range")
    if storage_device not in {"model", "cpu"}:
        raise ValueError("trajectory storage device must be model or cpu")
    tensor_cache: dict[int, Any] = {}

    def stored_row(value: Any) -> Any:
        key = id(value)
        owned = tensor_cache.get(key)
        if owned is None:
            owned = value[row_index].detach()
            if storage_device == "cpu":
                owned = owned.to(device="cpu").contiguous()
            tensor_cache[key] = owned
        return owned

    return TrajectoryStep(
        x_t=stored_row(record.x_t),
        sampled_action=stored_row(record.sampled_action),
        conditioned_next=stored_row(record.conditioned_next),
        t=stored_row(record.t),
        t_next=stored_row(record.t_next),
        old_log_prob=stored_row(record.old_log_prob),
        likelihood_semantics=record.likelihood_semantics,
        condition_identity=record.condition_identity[row_index],
        guidance_identity=record.guidance_identity[row_index],
        transition_index=int(record.transition_index[row_index].item()),
        storage_dtype_identity=record.storage_dtype_identity[row_index],
        quantization_identity=record.quantization_identity[row_index],
        active=bool(record.mask[row_index].item()),
        transition_std_dev=(
            None
            if record.policy_metadata.transition_std_dev is None
            else stored_row(record.policy_metadata.transition_std_dev)
        ),
        rectification_coefficient=(
            None
            if record.policy_metadata.rectification_coefficient is None
            else stored_row(record.policy_metadata.rectification_coefficient)
        ),
    )


def resolve_item_media_layout(
    task_type: str,
    media: DecodedMediaBatch,
) -> str:
    if not isinstance(media, DecodedMediaBatch):
        raise TypeError("media must be a DecodedMediaBatch")
    try:
        media.assert_integrity()
    except (TypeError, ValueError) as exc:
        raise RolloutContractError(f"decoded media contract drift: {exc}") from exc
    if task_type == "t2i":
        if media.layout != "BCHW":
            raise RolloutContractError("T2I decode must report BCHW layout")
        return "CHW"
    if task_type not in {"t2v", "i2v"}:
        raise RolloutContractError(f"unsupported decoded-media task {task_type!r}")
    if media.layout not in {"BFCHW", "BFHWC"}:
        raise RolloutContractError("video decode must report BFCHW or BFHWC layout")
    return "FCHW" if media.layout == "BFCHW" else "FHWC"


def _trajectory_context(
    request: RolloutRequest,
    row_index: int,
    *,
    strategy: str,
    strategy_identity: str,
) -> TrajectoryContext:
    row = request.samples.rows[row_index]
    seed = f"{strategy}\0{strategy_identity}\0{row.identity}".encode()
    digest = hashlib.sha256(seed).hexdigest()
    return TrajectoryContext(
        sample_id=f"sample-{digest[:24]}",
        trajectory_id=f"trajectory-{digest[24:48]}",
        batch_row=row,
    )


def _full_item(
    request: RolloutRequest,
    payloads: tuple[ConditionPayload, ...],
    result: RolloutStrategyResult,
    row_index: int,
    media: Any,
    media_layout: str,
) -> FullTrajectoryItem:
    if not isinstance(result.strategy_identity, str) or not result.strategy_identity:
        raise RolloutContractError("full-trajectory result lost strategy identity")
    return FullTrajectoryItem(
        context=_trajectory_context(
            request,
            row_index,
            strategy="full-trajectory",
            strategy_identity=result.strategy_identity,
        ),
        steps=result.steps_by_row[row_index],
        media=media,
        media_layout=media_layout,
        condition=payloads[row_index],
    )


def _single_step_item(
    request: RolloutRequest,
    payloads: tuple[ConditionPayload, ...],
    result: RolloutStrategyResult,
    row_index: int,
    media: Any,
    media_layout: str,
) -> SingleStepTrajectoryItem:
    if result.selected_timestep_index is None:
        raise RolloutContractError("single-step result lost selected index")
    if (
        result.selection_policy_identity is None
        or result.selection_mapping_identity is None
    ):
        raise RolloutContractError("single-step result lost selection identity")
    selected = result.selected_timestep_index[row_index]
    return SingleStepTrajectoryItem(
        context=_trajectory_context(
            request,
            row_index,
            strategy="single-step",
            strategy_identity=str(selected),
        ),
        steps=result.steps_by_row[row_index],
        media=media,
        media_layout=media_layout,
        condition=payloads[row_index],
        selected_timestep_index=selected,
        selection_policy_identity=result.selection_policy_identity,
        selection_mapping_identity=result.selection_mapping_identity,
    )


def _branching_item(
    request: RolloutRequest,
    payloads: tuple[ConditionPayload, ...],
    result: RolloutStrategyResult,
    row_index: int,
    media: Any,
    media_layout: str,
) -> BranchingTrajectoryItem:
    topology = result.branch_topology
    if not isinstance(topology, BranchTopology):
        raise TypeError("branching result lost its branch topology")
    row = request.samples.rows[row_index]
    if topology.kind == "every_policy_timestep":
        if result.transition_terminal_media is None:
            raise RolloutContractError("branching result lost terminal media")
        if result.transition_terminal_media_layout is None:
            raise RolloutContractError("branching result lost terminal media layout")
        if result.schedule_snapshot_identity is None:
            raise RolloutContractError(
                "paper topology result lost its schedule snapshot identity"
            )
        strategy_identity = hashlib.sha256(
            (
                f"{topology.topology_identity}\0{result.schedule_snapshot_identity}"
            ).encode()
        ).hexdigest()
        branch_step_index = None
        shared_prefix_id = None
        branch_step_identity = None
        terminal_media = result.transition_terminal_media[row_index]
        terminal_layout = result.transition_terminal_media_layout
    else:
        if result.branch_step_index is None:
            raise RolloutContractError("branching result lost branch step")
        if result.shared_prefix_id is None or result.branch_step_identity is None:
            raise RolloutContractError("branching result lost branch identities")
        strategy_identity = result.branch_step_identity[row_index]
        branch_step_index = result.branch_step_index
        shared_prefix_id = result.shared_prefix_id[row_index]
        branch_step_identity = result.branch_step_identity[row_index]
        terminal_media = None
        terminal_layout = None
    return BranchingTrajectoryItem(
        context=_trajectory_context(
            request,
            row_index,
            strategy="branching",
            strategy_identity=strategy_identity,
        ),
        steps=result.steps_by_row[row_index],
        media=media,
        media_layout=media_layout,
        condition=payloads[row_index],
        branch_topology=topology,
        exploration_member_index=row.member_id,
        branch_step_index=branch_step_index,
        shared_prefix_id=shared_prefix_id,
        branch_step_identity=branch_step_identity,
        transition_terminal_media=terminal_media,
        transition_terminal_media_layout=terminal_layout,
    )


def build_trajectory_batch(
    *,
    request: RolloutRequest,
    condition_payloads: tuple[ConditionPayload, ...],
    result: RolloutStrategyResult,
    media: Any,
    media_layout: str,
) -> TrajectoryBatch:
    """Cross the mutable-builder to immutable-data boundary exactly once."""

    batch_size = request.samples.batch_size
    if len(result.steps_by_row) != batch_size:
        raise RolloutContractError("strategy rows must match the sample batch")
    if len(condition_payloads) != batch_size:
        raise RolloutContractError("condition payloads must match the sample batch")
    item_factory = {
        "full-trajectory": _full_item,
        "branching": _branching_item,
        "single-step": _single_step_item,
    }[result.strategy]
    items = tuple(
        item_factory(
            request,
            condition_payloads,
            result,
            row_index,
            media[row_index].detach(),
            media_layout,
        )
        for row_index in range(batch_size)
    )
    return ExplicitCollator().collate_trajectories(items, media_batch=media)
