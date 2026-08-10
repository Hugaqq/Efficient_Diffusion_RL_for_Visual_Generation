"""Item-level sample contracts with no implicit batch dimension.

These objects describe dataset/storage rows.  Runtime components consume the
explicit batch types from :mod:`visual_rl.data.samples.collate`; they must not infer
batching from item tensor shapes.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Literal


def _non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tensor_payload(value: Any) -> dict[str, object]:
    """Return a complete JSON-safe representation of one detached tensor."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor payload requires a torch.Tensor")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError("serialized tensors must be detached")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "values": value.detach().to(device="cpu").tolist(),
    }


def _validate_plain_mapping(name: str, value: Mapping[str, object]) -> None:
    """Reject hidden tensors/classes in durable item metadata."""

    def visit(item: object, path: str) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{path} contains a non-finite float")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} keys must be strings")
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        raise TypeError(
            f"{path} must contain only JSON-safe plain values, got "
            f"{type(item).__name__}"
        )

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    visit(value, name)


def _plain_copy(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_copy(item) for item in value]
    return value


def _freeze_plain(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_plain(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_plain(item) for item in value)
    return value


@dataclass(frozen=True)
class SourceItemContext:
    """Stable dataset identity safe to persist in a preprocess cache."""

    source_item_id: str
    dataset_source_id: str
    dataset_index: int
    dataset_revision: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _non_empty_string("source_item_id", self.source_item_id)
        _non_empty_string("dataset_source_id", self.dataset_source_id)
        _non_negative_int("dataset_index", self.dataset_index)
        _non_empty_string("dataset_revision", self.dataset_revision)
        _positive_int("schema_version", self.schema_version)

    def serialize(self) -> dict[str, object]:
        self.validate()
        return {
            "source_item_id": self.source_item_id,
            "dataset_source_id": self.dataset_source_id,
            "dataset_index": self.dataset_index,
            "dataset_revision": self.dataset_revision,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class BatchRowContext:
    """Iteration-local identity created after phase routing and K-repeat."""

    occurrence_id: str
    group_id: str
    member_id: int
    phase: str
    optimizer_step: int
    source_item_id: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _non_empty_string("occurrence_id", self.occurrence_id)
        _non_empty_string("group_id", self.group_id)
        _non_negative_int("member_id", self.member_id)
        _non_empty_string("phase", self.phase)
        _non_negative_int("optimizer_step", self.optimizer_step)
        _non_empty_string("source_item_id", self.source_item_id)

    @property
    def identity(self) -> str:
        payload = json.dumps(
            self.serialize(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return f"batch-row-{hashlib.sha256(payload).hexdigest()[:24]}"

    def serialize(self) -> dict[str, object]:
        self.validate()
        return {
            "occurrence_id": self.occurrence_id,
            "group_id": self.group_id,
            "member_id": self.member_id,
            "phase": self.phase,
            "optimizer_step": self.optimizer_step,
            "source_item_id": self.source_item_id,
        }


@dataclass(frozen=True)
class TrajectoryContext:
    """Rollout identity bound to exactly one iteration-local batch row."""

    sample_id: str
    trajectory_id: str
    batch_row: BatchRowContext

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _non_empty_string("sample_id", self.sample_id)
        _non_empty_string("trajectory_id", self.trajectory_id)
        if not isinstance(self.batch_row, BatchRowContext):
            raise TypeError("batch_row must be a BatchRowContext")
        self.batch_row.validate()

    @property
    def batch_row_identity(self) -> str:
        return self.batch_row.identity

    def serialize(self) -> dict[str, object]:
        self.validate()
        return {
            "sample_id": self.sample_id,
            "trajectory_id": self.trajectory_id,
            "batch_row_identity": self.batch_row_identity,
            "batch_row": self.batch_row.serialize(),
        }


class ConditionPayload(ABC):
    """Discriminated, item-level condition payload."""

    KIND: ClassVar[str]

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def serialize(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class NoCondition(ConditionPayload):
    KIND: ClassVar[Literal["none"]] = "none"

    def validate(self) -> None:
        return None

    def serialize(self) -> dict[str, object]:
        return {"kind": self.KIND}


_CAMERA_CONDITION_IDENTITY_SCHEMA = "visual-rl.camera-condition.v1"


def camera_condition_identity(
    camera_trajectory: Any,
    conditioner_config_identity: str,
) -> str:
    """Canonical row identity shared by conditioning, replay, and reward."""

    import torch

    _non_empty_string(
        "conditioner_config_identity",
        conditioner_config_identity,
    )
    if not isinstance(camera_trajectory, torch.Tensor):
        raise TypeError("camera_trajectory must be a torch.Tensor")
    if (
        camera_trajectory.ndim != 3
        or camera_trajectory.shape[0] < 1
        or tuple(camera_trajectory.shape[1:]) != (4, 4)
    ):
        raise ValueError(
            "item camera_trajectory must have shape [F,4,4] without a batch dimension"
        )
    if not camera_trajectory.is_floating_point():
        raise TypeError("camera_trajectory must be floating point")
    if camera_trajectory.requires_grad or camera_trajectory.grad_fn is not None:
        raise ValueError("camera_trajectory must be detached")
    if not bool(torch.isfinite(camera_trajectory).all()):
        raise ValueError("camera_trajectory must be finite")

    canonical = (
        camera_trajectory.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )
    header = {
        "schema": _CAMERA_CONDITION_IDENTITY_SCHEMA,
        "conditioner_config_identity": conditioner_config_identity,
        "shape": list(canonical.shape),
        "dtype": str(canonical.dtype),
    }
    digest = hashlib.sha256(
        json.dumps(
            header,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def camera_condition_batch_identity(
    row_condition_identities: tuple[str, ...],
) -> str:
    """Order-sensitive provenance identity for one stacked camera batch."""

    if type(row_condition_identities) is not tuple or not row_condition_identities:
        raise ValueError("row_condition_identities must be a non-empty tuple")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in row_condition_identities
    ):
        raise ValueError("row condition identities must be SHA-256 strings")
    digest = hashlib.sha256(
        f"{_CAMERA_CONDITION_IDENTITY_SCHEMA}.batch".encode("utf-8")
    )
    for value in row_condition_identities:
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CameraConditionPayload(ConditionPayload):
    """One camera trajectory; shape is ``[F, 4, 4]`` with no B axis."""

    camera_trajectory: Any
    conditioner_config_identity: str
    condition_identity: str = field(init=False)

    KIND: ClassVar[Literal["camera"]] = "camera"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "condition_identity",
            camera_condition_identity(
                self.camera_trajectory,
                self.conditioner_config_identity,
            ),
        )
        self.validate()

    def validate(self) -> None:
        import torch

        _non_empty_string(
            "conditioner_config_identity",
            self.conditioner_config_identity,
        )
        value = self.camera_trajectory
        if not isinstance(value, torch.Tensor):
            raise TypeError("camera_trajectory must be a torch.Tensor")
        if value.ndim != 3 or value.shape[0] < 1 or tuple(value.shape[1:]) != (4, 4):
            raise ValueError(
                "item camera_trajectory must have shape [F,4,4] without a "
                "batch dimension"
            )
        if not value.is_floating_point():
            raise TypeError("camera_trajectory must be floating point")
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError("camera_trajectory must be detached")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("camera_trajectory must be finite")
        expected_identity = camera_condition_identity(
            value,
            self.conditioner_config_identity,
        )
        if self.condition_identity != expected_identity:
            raise ValueError(
                "condition_identity must match the canonical camera trajectory"
            )

    def serialize(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.KIND,
            "conditioner_config_identity": self.conditioner_config_identity,
            "condition_identity": self.condition_identity,
            "camera_trajectory": _tensor_payload(self.camera_trajectory),
        }


@dataclass(frozen=True)
class SampleItem(ABC):
    """Base dataset item.  Concrete items never contain a B dimension."""

    prompt: str
    source: SourceItemContext
    condition: ConditionPayload = NoCondition()
    metadata: Mapping[str, object] = field(default_factory=dict)

    TASK_TYPE: ClassVar[str]

    def __post_init__(self) -> None:
        _validate_plain_mapping("metadata", self.metadata)
        object.__setattr__(self, "metadata", _freeze_plain(self.metadata))
        self.validate()

    def validate(self) -> None:
        _non_empty_string("prompt", self.prompt)
        if not isinstance(self.source, SourceItemContext):
            raise TypeError("source must be a SourceItemContext")
        self.source.validate()
        if not isinstance(self.condition, ConditionPayload):
            raise TypeError("condition must be a ConditionPayload")
        self.condition.validate()
        _validate_plain_mapping("metadata", self.metadata)

    def _serialize_common(self) -> dict[str, object]:
        self.validate()
        return {
            "task_type": self.TASK_TYPE,
            "prompt": self.prompt,
            "source": self.source.serialize(),
            "condition": self.condition.serialize(),
            "metadata": _plain_copy(self.metadata),
        }

    @abstractmethod
    def serialize(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class T2IItem(SampleItem):
    TASK_TYPE: ClassVar[Literal["t2i"]] = "t2i"

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.condition, NoCondition):
            raise ValueError("T2IItem does not accept a camera condition")

    def serialize(self) -> dict[str, object]:
        return self._serialize_common()


@dataclass(frozen=True)
class T2VItem(SampleItem):
    TASK_TYPE: ClassVar[Literal["t2v"]] = "t2v"

    def serialize(self) -> dict[str, object]:
        return self._serialize_common()


@dataclass(frozen=True)
class I2VItem(SampleItem):
    """Image-conditioned video item; input image is ``[C,H,W]``."""

    input_image: Any = None

    TASK_TYPE: ClassVar[Literal["i2v"]] = "i2v"

    def validate(self) -> None:
        super().validate()
        import torch

        value = self.input_image
        if not isinstance(value, torch.Tensor):
            raise TypeError("input_image must be a torch.Tensor")
        if value.ndim != 3:
            raise ValueError(
                "I2VItem.input_image must have shape [C,H,W] without a batch dimension"
            )
        if not value.is_floating_point():
            raise TypeError("input_image must be floating point")
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError("input_image must be detached")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("input_image must be finite")

    def serialize(self) -> dict[str, object]:
        payload = self._serialize_common()
        payload["input_image"] = _tensor_payload(self.input_image)
        return payload


__all__ = [
    "BatchRowContext",
    "CameraConditionPayload",
    "camera_condition_batch_identity",
    "camera_condition_identity",
    "ConditionPayload",
    "I2VItem",
    "NoCondition",
    "SampleItem",
    "SourceItemContext",
    "T2IItem",
    "T2VItem",
    "TrajectoryContext",
]
