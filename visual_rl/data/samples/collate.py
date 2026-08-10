"""The explicit item-to-batch boundary for samples and trajectories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from visual_rl.data.samples.items import (
    BatchRowContext,
    CameraConditionPayload,
    I2VItem,
    NoCondition,
    SampleItem,
    SourceItemContext,
    camera_condition_batch_identity,
    camera_condition_identity,
)


def _indices(indices: Any, batch_size: int) -> list[int]:
    import torch

    if isinstance(indices, slice):
        resolved = list(range(batch_size))[indices]
    elif isinstance(indices, torch.Tensor):
        if indices.ndim != 1 or indices.dtype not in {torch.int32, torch.int64}:
            raise TypeError("batch indices tensor must be 1-D integer")
        resolved = [int(item) for item in indices.to(device="cpu").tolist()]
    elif isinstance(indices, (tuple, list)):
        if any(type(item) is not int for item in indices):
            raise TypeError("batch indices must contain integers, not bool")
        resolved = list(indices)
    else:
        raise TypeError("batch indices must be a slice, sequence, or tensor")
    if not resolved:
        raise ValueError("batch slice must not be empty")
    if any(not 0 <= item < batch_size for item in resolved):
        raise IndexError("batch index is out of range")
    return resolved


class ConditionBatchState(ABC):
    """Runtime-only batch state created from typed condition payloads."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def slice(self, indices: Any) -> ConditionBatchState:
        raise NotImplementedError

    @abstractmethod
    def to(self, device: Any) -> ConditionBatchState:
        raise NotImplementedError

    @abstractmethod
    def detach(self) -> ConditionBatchState:
        raise NotImplementedError


@dataclass(frozen=True)
class NoConditionBatchState(ConditionBatchState):
    size: int

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return self.size

    def validate(self) -> None:
        if type(self.size) is not int or self.size < 1:
            raise ValueError("NoConditionBatchState.size must be positive")

    def slice(self, indices: Any) -> NoConditionBatchState:
        return NoConditionBatchState(len(_indices(indices, self.size)))

    def to(self, device: Any) -> NoConditionBatchState:
        del device
        return self

    def detach(self) -> NoConditionBatchState:
        return self


@dataclass(frozen=True)
class CameraConditionBatchState(ConditionBatchState):
    """Stacked camera state with shape ``[B,F,4,4]``."""

    camera_trajectory: Any
    conditioner_config_identity: tuple[str, ...]
    row_condition_identities: tuple[str, ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.camera_trajectory.shape[0])

    @property
    def batch_condition_identity(self) -> str:
        return camera_condition_batch_identity(self.row_condition_identities)

    def validate(self) -> None:
        import torch

        value = self.camera_trajectory
        if not isinstance(value, torch.Tensor):
            raise TypeError("camera_trajectory must be a torch.Tensor")
        if value.ndim != 4 or value.shape[0] < 1 or tuple(value.shape[2:]) != (4, 4):
            raise ValueError("camera batch must have shape [B,F,4,4]")
        if not value.is_floating_point():
            raise TypeError("camera batch must be floating point")
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError("camera batch must be detached")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("camera batch must be finite")
        identities = self.conditioner_config_identity
        if type(identities) is not tuple or len(identities) != value.shape[0]:
            raise ValueError("conditioner_config_identity must have shape [B]")
        if any(not isinstance(item, str) or not item for item in identities):
            raise ValueError("conditioner identities must be non-empty strings")
        row_identities = self.row_condition_identities
        if type(row_identities) is not tuple or len(row_identities) != value.shape[0]:
            raise ValueError("row_condition_identities must have shape [B]")
        expected = tuple(
            camera_condition_identity(
                value[index],
                identities[index],
            )
            for index in range(value.shape[0])
        )
        if row_identities != expected:
            raise ValueError(
                "row_condition_identities must match camera payload content"
            )

    def slice(self, indices: Any) -> CameraConditionBatchState:
        import torch

        resolved = _indices(indices, self.batch_size)
        index = torch.tensor(
            resolved,
            dtype=torch.long,
            device=self.camera_trajectory.device,
        )
        return CameraConditionBatchState(
            camera_trajectory=self.camera_trajectory.index_select(0, index),
            conditioner_config_identity=tuple(
                self.conditioner_config_identity[item] for item in resolved
            ),
            row_condition_identities=tuple(
                self.row_condition_identities[item] for item in resolved
            ),
        )

    def to(self, device: Any) -> CameraConditionBatchState:
        # Camera matrices intentionally retain their declared precision.
        return replace(
            self,
            camera_trajectory=self.camera_trajectory.to(device=device),
        )

    def detach(self) -> CameraConditionBatchState:
        return replace(self, camera_trajectory=self.camera_trajectory.detach())


@dataclass(frozen=True)
class StackedSampleBatch:
    """Homogeneous runtime sample batch produced by :class:`ExplicitCollator`."""

    task_type: str
    prompts: tuple[str, ...]
    sources: tuple[SourceItemContext, ...]
    rows: tuple[BatchRowContext, ...]
    metadata: tuple[Mapping[str, object], ...]
    condition_state: ConditionBatchState
    input_images: Any | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return len(self.prompts)

    def validate(self) -> None:
        import torch

        if self.task_type not in {"t2i", "t2v", "i2v"}:
            raise ValueError(f"unknown task type: {self.task_type!r}")
        if type(self.prompts) is not tuple or not self.prompts:
            raise ValueError("prompts must be a non-empty tuple")
        batch_size = self.batch_size
        if any(not isinstance(item, str) or not item for item in self.prompts):
            raise ValueError("prompts must be non-empty strings")
        for name in ("sources", "rows", "metadata"):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != batch_size:
                raise ValueError(f"{name} must contain B entries")
        if any(not isinstance(item, SourceItemContext) for item in self.sources):
            raise TypeError("sources must contain SourceItemContext values")
        if any(not isinstance(item, BatchRowContext) for item in self.rows):
            raise TypeError("rows must contain BatchRowContext values")
        for source, row in zip(self.sources, self.rows, strict=True):
            source.validate()
            row.validate()
            if source.source_item_id != row.source_item_id:
                raise ValueError("row source_item_id must match its source item")
        identities = tuple(row.identity for row in self.rows)
        if len(identities) != len(set(identities)):
            raise ValueError("batch row identities must be unique after K-repeat")
        grouped_members: dict[str, set[int]] = {}
        for row in self.rows:
            members = grouped_members.setdefault(row.group_id, set())
            if row.member_id in members:
                raise ValueError("member_id must be unique within each group")
            members.add(row.member_id)
        if not isinstance(self.condition_state, ConditionBatchState):
            raise TypeError("condition_state must be a ConditionBatchState")
        self.condition_state.validate()
        if self.condition_state.batch_size != batch_size:
            raise ValueError("condition_state batch size mismatch")

        if self.task_type == "i2v":
            if not isinstance(self.input_images, torch.Tensor):
                raise TypeError("I2V batch requires input_images tensor")
            if self.input_images.ndim != 4 or self.input_images.shape[0] != batch_size:
                raise ValueError("input_images must have shape [B,C,H,W]")
            if not self.input_images.is_floating_point():
                raise TypeError("input_images must be floating point")
            if self.input_images.requires_grad or self.input_images.grad_fn is not None:
                raise ValueError("input_images must be detached")
            if not bool(torch.isfinite(self.input_images).all()):
                raise ValueError("input_images must be finite")
        elif self.input_images is not None:
            raise ValueError("only I2V batches accept input_images")

    def slice(self, indices: Any) -> StackedSampleBatch:
        resolved = _indices(indices, self.batch_size)
        import torch

        images = self.input_images
        if images is not None:
            index = torch.tensor(resolved, dtype=torch.long, device=images.device)
            images = images.index_select(0, index)
        return replace(
            self,
            prompts=tuple(self.prompts[item] for item in resolved),
            sources=tuple(self.sources[item] for item in resolved),
            rows=tuple(self.rows[item] for item in resolved),
            metadata=tuple(self.metadata[item] for item in resolved),
            condition_state=self.condition_state.slice(resolved),
            input_images=images,
        )

    def to(self, device: Any, dtype: Any = None) -> StackedSampleBatch:
        images = self.input_images
        if images is not None:
            target_dtype = dtype if dtype is not None else images.dtype
            images = images.to(device=device, dtype=target_dtype)
        return replace(
            self,
            condition_state=self.condition_state.to(device),
            input_images=images,
        )

    def detach(self) -> StackedSampleBatch:
        images = self.input_images
        return replace(
            self,
            condition_state=self.condition_state.detach(),
            input_images=None if images is None else images.detach(),
        )


class ExplicitCollator:
    """Sole owner of item stacking for the P4 typed data model."""

    def collate_samples(
        self,
        items: Sequence[SampleItem],
        rows: Sequence[BatchRowContext],
    ) -> StackedSampleBatch:
        import torch

        item_values = tuple(items)
        row_values = tuple(rows)
        if not item_values:
            raise ValueError("cannot collate an empty sample sequence")
        if len(item_values) != len(row_values):
            raise ValueError("items and rows must have the same length")
        if any(not isinstance(item, SampleItem) for item in item_values):
            raise TypeError("items must contain only SampleItem values")
        if any(not isinstance(row, BatchRowContext) for row in row_values):
            raise TypeError("rows must contain only BatchRowContext values")
        concrete_type = type(item_values[0])
        if any(type(item) is not concrete_type for item in item_values):
            raise ValueError("one sample batch cannot mix task item types")
        for item in item_values:
            item.validate()

        input_images = None
        if concrete_type is I2VItem:
            input_images = torch.stack(
                [item.input_image for item in item_values],
                dim=0,
            )
        return StackedSampleBatch(
            task_type=item_values[0].TASK_TYPE,
            prompts=tuple(item.prompt for item in item_values),
            sources=tuple(item.source for item in item_values),
            rows=row_values,
            metadata=tuple(item.metadata for item in item_values),
            condition_state=self._collate_conditions(
                tuple(item.condition for item in item_values)
            ),
            input_images=input_images,
        )

    def collate_trajectories(
        self,
        items: Sequence[Any],
        *,
        media_batch: Any | None = None,
    ):
        import torch

        from visual_rl.data.samples.trajectory import (
            BranchingTrajectoryItem,
            SingleStepTrajectoryItem,
            TrajectoryBatch,
            TrajectoryItem,
        )

        values = tuple(items)
        if not values:
            raise ValueError("cannot collate an empty trajectory sequence")
        if any(not isinstance(item, TrajectoryItem) for item in values):
            raise TypeError("items must contain only TrajectoryItem values")
        concrete_type = type(values[0])
        if any(type(item) is not concrete_type for item in values):
            raise ValueError("one trajectory batch cannot mix trajectory kinds")
        for item in values:
            item.validate()
        transition_count = len(values[0].steps)
        if any(len(item.steps) != transition_count for item in values):
            raise ValueError("trajectory collation requires equal transition counts")
        if any(item.media_layout != values[0].media_layout for item in values):
            raise ValueError("trajectory media layouts must match")
        if any(
            item.likelihood_semantics is not values[0].likelihood_semantics
            for item in values
        ):
            raise ValueError("trajectory likelihood semantics must match")

        def stack_step(name: str):
            first = getattr(values[0].steps[0], name)
            output = first.new_empty(
                (len(values), transition_count, *first.shape)
            )
            for row_index, item in enumerate(values):
                for step_index, step in enumerate(item.steps):
                    value = getattr(step, name)
                    if (
                        tuple(value.shape) != tuple(first.shape)
                        or value.dtype != first.dtype
                        or value.device != first.device
                    ):
                        raise ValueError(
                            f"trajectory {name} changed shape/dtype/device"
                        )
                    output[row_index, step_index].copy_(value)
            return output

        def stack_item_tensor(name: str):
            first = getattr(values[0], name)
            output = first.new_empty((len(values), *first.shape))
            for row_index, item in enumerate(values):
                value = getattr(item, name)
                if (
                    tuple(value.shape) != tuple(first.shape)
                    or value.dtype != first.dtype
                    or value.device != first.device
                ):
                    raise ValueError(
                        f"trajectory {name} changed shape/dtype/device"
                    )
                output[row_index].copy_(value)
            return output

        def stack_optional_step(name: str):
            entries = tuple(
                getattr(step, name) for item in values for step in item.steps
            )
            if all(value is None for value in entries):
                return None
            if any(value is None for value in entries):
                raise ValueError(
                    f"trajectory {name} must be present for every transition or none"
                )
            return stack_step(name)

        transition_mask = torch.tensor(
            [[step.active for step in item.steps] for item in values],
            dtype=torch.bool,
            device=values[0].steps[0].x_t.device,
        )
        transition_index = torch.tensor(
            [[step.transition_index for step in item.steps] for item in values],
            dtype=torch.int64,
            device=values[0].steps[0].x_t.device,
        )
        branch_index = None
        branch_topology = None
        exploration_member_index = None
        branch_timestep_index = None
        transition_terminal_media = None
        transition_terminal_media_layout = None
        selected_index = None
        shared_prefix_id = None
        branch_step_identity = None
        selection_policy_identity = None
        selection_mapping_identity = None
        if concrete_type is BranchingTrajectoryItem:
            topologies = {item.branch_topology for item in values}
            if len(topologies) != 1:
                raise ValueError("branch trajectory rows must share one topology")
            branch_topology = next(iter(topologies))
            assert branch_topology is not None
            exploration_member_index = torch.tensor(
                [item.exploration_member_index for item in values],
                dtype=torch.int64,
                device=transition_index.device,
            )
            if branch_topology.kind == "every_policy_timestep":
                branch_timestep_index = transition_index.clone()
                transition_terminal_media = stack_item_tensor(
                    "transition_terminal_media"
                )
                item_terminal_layouts = {
                    item.transition_terminal_media_layout for item in values
                }
                if len(item_terminal_layouts) != 1:
                    raise ValueError("transition terminal media layouts must match")
                transition_terminal_media_layout = {
                    "TCHW": "BTCHW",
                    "TFCHW": "BTFCHW",
                    "TFHWC": "BTFHWC",
                }[next(iter(item_terminal_layouts))]
            else:
                branch_index = torch.tensor(
                    [item.branch_step_index for item in values],
                    dtype=torch.int64,
                    device=transition_index.device,
                )
                shared_prefix_id = tuple(item.shared_prefix_id for item in values)
                branch_step_identity = tuple(
                    item.branch_step_identity for item in values
                )
        elif concrete_type is SingleStepTrajectoryItem:
            selected_index = torch.tensor(
                [item.selected_timestep_index for item in values],
                dtype=torch.int64,
                device=transition_index.device,
            )
            policy_identities = {item.selection_policy_identity for item in values}
            mapping_identities = {item.selection_mapping_identity for item in values}
            if len(policy_identities) != 1 or len(mapping_identities) != 1:
                raise ValueError(
                    "single-step trajectory rows must share selection identities"
                )
            selection_policy_identity = next(iter(policy_identities))
            selection_mapping_identity = next(iter(mapping_identities))
        layout = {
            "CHW": "BCHW",
            "FCHW": "BFCHW",
            "FHWC": "BFHWC",
        }[values[0].media_layout]
        if media_batch is None:
            collated_media = stack_item_tensor("media")
        else:
            if not isinstance(media_batch, torch.Tensor):
                raise TypeError("media_batch must be a torch.Tensor or None")
            if tuple(media_batch.shape) != (len(values), *values[0].media.shape):
                raise ValueError("media_batch does not match trajectory item rows")
            for row_index, item in enumerate(values):
                row = media_batch[row_index]
                if (
                    item.media.data_ptr() != row.data_ptr()
                    or item.media.dtype != row.dtype
                    or item.media.device != row.device
                    or tuple(item.media.shape) != tuple(row.shape)
                ):
                    raise ValueError(
                        "media_batch rows must be the exact item media views"
                    )
            collated_media = media_batch
        return TrajectoryBatch(
            kind=values[0].KIND,
            contexts=tuple(item.context for item in values),
            x_t=stack_step("x_t"),
            sampled_action=stack_step("sampled_action"),
            conditioned_next=stack_step("conditioned_next"),
            timesteps=stack_step("t"),
            next_timesteps=stack_step("t_next"),
            old_log_probs=stack_step("old_log_prob"),
            transition_mask=transition_mask,
            transition_index=transition_index,
            likelihood_semantics=values[0].likelihood_semantics,
            condition_identity=tuple(
                tuple(step.condition_identity for step in item.steps) for item in values
            ),
            guidance_identity=tuple(
                tuple(step.guidance_identity for step in item.steps) for item in values
            ),
            storage_dtype_identity=tuple(
                tuple(step.storage_dtype_identity for step in item.steps)
                for item in values
            ),
            quantization_identity=tuple(
                tuple(step.quantization_identity for step in item.steps)
                for item in values
            ),
            media=collated_media,
            media_layout=layout,
            condition_state=self._collate_conditions(
                tuple(item.condition for item in values)
            ),
            branch_topology=branch_topology,
            exploration_member_index=exploration_member_index,
            branch_timestep_index=branch_timestep_index,
            transition_terminal_media=transition_terminal_media,
            transition_terminal_media_layout=transition_terminal_media_layout,
            branch_step_index=branch_index,
            selected_timestep_index=selected_index,
            transition_std_dev=stack_optional_step("transition_std_dev"),
            rectification_coefficient=stack_optional_step("rectification_coefficient"),
            shared_prefix_id=shared_prefix_id,
            branch_step_identity=branch_step_identity,
            selection_policy_identity=selection_policy_identity,
            selection_mapping_identity=selection_mapping_identity,
        )

    @staticmethod
    def _collate_conditions(conditions: tuple[Any, ...]) -> ConditionBatchState:
        import torch

        first = conditions[0]
        if any(type(item) is not type(first) for item in conditions):
            raise ValueError("one batch cannot mix condition payload kinds")
        if isinstance(first, NoCondition):
            return NoConditionBatchState(len(conditions))
        if isinstance(first, CameraConditionPayload):
            frame_count = first.camera_trajectory.shape[0]
            if any(
                item.camera_trajectory.shape[0] != frame_count for item in conditions
            ):
                raise ValueError("camera trajectories must have equal frame counts")
            return CameraConditionBatchState(
                camera_trajectory=torch.stack(
                    [item.camera_trajectory for item in conditions],
                    dim=0,
                ),
                conditioner_config_identity=tuple(
                    item.conditioner_config_identity for item in conditions
                ),
                row_condition_identities=tuple(
                    item.condition_identity for item in conditions
                ),
            )
        raise TypeError(f"unsupported condition payload: {type(first).__name__}")


__all__ = [
    "CameraConditionBatchState",
    "ConditionBatchState",
    "ExplicitCollator",
    "NoConditionBatchState",
    "StackedSampleBatch",
]
