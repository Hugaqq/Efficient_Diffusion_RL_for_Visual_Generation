"""Typed item- and batch-level trajectory payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, ClassVar, Literal

from visual_rl.core.contracts import LikelihoodSemantics
from visual_rl.data.samples.items import (
    CameraConditionPayload,
    ConditionPayload,
    NoCondition,
    TrajectoryContext,
    _non_empty_string,
    _non_negative_int,
    _tensor_payload,
)

BranchTopologyKind = Literal[
    "every_policy_timestep",
    "single_point_branch_ablation",
]
BranchAxis = Literal[
    "prompt_group",
    "exploration_member",
    "policy_timestep",
]

_ROW_AXES: tuple[BranchAxis, ...] = (
    "prompt_group",
    "exploration_member",
)
_PAPER_AXES: tuple[BranchAxis, ...] = (*_ROW_AXES, "policy_timestep")


def _branch_topology_identity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class BranchTopology:
    """Exact stored-policy and reward-media axes for a branching rollout."""

    kind: BranchTopologyKind
    exploration_count: int
    stored_policy_axes: tuple[BranchAxis, ...]
    reward_media_axes: tuple[BranchAxis, ...]
    schema_version: int = 1
    topology_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if self.kind not in {
            "every_policy_timestep",
            "single_point_branch_ablation",
        }:
            raise ValueError(f"unknown branch topology kind: {self.kind!r}")
        if type(self.exploration_count) is not int or self.exploration_count < 2:
            raise ValueError("exploration_count must be an integer >= 2")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("branch topology schema_version must be 1")
        for name in ("stored_policy_axes", "reward_media_axes"):
            axes = getattr(self, name)
            if type(axes) is not tuple or not axes:
                raise TypeError(f"{name} must be a non-empty tuple")
            if len(axes) != len(set(axes)):
                raise ValueError(f"{name} must not repeat an axis")
            if any(axis not in _PAPER_AXES for axis in axes):
                raise ValueError(f"{name} contains an unknown branch axis")

        expected = _PAPER_AXES if self.kind == "every_policy_timestep" else _ROW_AXES
        if self.stored_policy_axes != expected:
            raise ValueError(f"{self.kind} stored_policy_axes must be {expected!r}")
        if self.reward_media_axes != expected:
            raise ValueError(f"{self.kind} reward_media_axes must be {expected!r}")
        object.__setattr__(
            self,
            "topology_identity",
            _branch_topology_identity(self._identity_payload()),
        )

    @classmethod
    def every_policy_timestep(cls, exploration_count: int) -> "BranchTopology":
        """TempFlow paper topology: K branches at every policy timestep."""

        return cls(
            kind="every_policy_timestep",
            exploration_count=exploration_count,
            stored_policy_axes=_PAPER_AXES,
            reward_media_axes=_PAPER_AXES,
        )

    @classmethod
    def single_point_branch_ablation(
        cls,
        exploration_count: int,
    ) -> "BranchTopology":
        """One branch point per prompt group; retained as an explicit ablation."""

        return cls(
            kind="single_point_branch_ablation",
            exploration_count=exploration_count,
            stored_policy_axes=_ROW_AXES,
            reward_media_axes=_ROW_AXES,
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "exploration_count": self.exploration_count,
            "stored_policy_axes": list(self.stored_policy_axes),
            "reward_media_axes": list(self.reward_media_axes),
        }

    def to_payload(self) -> dict[str, object]:
        payload = self._identity_payload()
        payload["topology_identity"] = self.topology_identity
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> "BranchTopology":
        if not isinstance(payload, Mapping):
            raise TypeError("branch_topology must be a mapping")
        expected = {
            "schema_version",
            "kind",
            "exploration_count",
            "stored_policy_axes",
            "reward_media_axes",
            "topology_identity",
        }
        if set(payload) != expected:
            raise ValueError("branch_topology payload has invalid fields")
        stored = payload["stored_policy_axes"]
        reward = payload["reward_media_axes"]
        if not isinstance(stored, (tuple, list)) or not isinstance(
            reward, (tuple, list)
        ):
            raise TypeError("branch topology axes must be sequences")
        result = cls(
            schema_version=payload["schema_version"],
            kind=payload["kind"],
            exploration_count=payload["exploration_count"],
            stored_policy_axes=tuple(stored),
            reward_media_axes=tuple(reward),
        )
        if payload["topology_identity"] != result.topology_identity:
            raise ValueError("branch_topology identity mismatch")
        return result


def _semantics(value: object) -> LikelihoodSemantics:
    if isinstance(value, LikelihoodSemantics):
        return value
    try:
        return LikelihoodSemantics(value)
    except (TypeError, ValueError):
        raise ValueError(f"unknown likelihood semantics: {value!r}") from None


def _validate_detached_tensor(
    name: str,
    value: Any,
    *,
    floating: bool = False,
) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if floating and not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} must be detached")
    if (value.is_floating_point() or value.is_complex()) and not bool(
        torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite")


def _validate_scalar_tensor(name: str, value: Any, *, floating: bool = False) -> None:
    _validate_detached_tensor(name, value, floating=floating)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")


@dataclass(frozen=True)
class TrajectoryStep:
    """One unbatched transition with both pre- and post-hook states."""

    x_t: Any
    sampled_action: Any
    conditioned_next: Any
    t: Any
    t_next: Any
    old_log_prob: Any
    likelihood_semantics: LikelihoodSemantics
    condition_identity: str
    guidance_identity: str
    transition_index: int
    storage_dtype_identity: str
    quantization_identity: str = "none"
    active: bool = True
    transition_std_dev: Any | None = None
    rectification_coefficient: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "likelihood_semantics",
            _semantics(self.likelihood_semantics),
        )
        self.validate()

    def validate(self) -> None:
        import torch

        for name in ("x_t", "sampled_action", "conditioned_next"):
            _validate_detached_tensor(name, getattr(self, name), floating=True)
        if self.x_t.ndim < 1:
            raise ValueError("trajectory latent items must have at least one axis")
        shape = tuple(self.x_t.shape)
        if tuple(self.sampled_action.shape) != shape:
            raise ValueError("sampled_action must have the same shape as x_t")
        if tuple(self.conditioned_next.shape) != shape:
            raise ValueError("conditioned_next must have the same shape as x_t")
        _validate_scalar_tensor("t", self.t)
        _validate_scalar_tensor("t_next", self.t_next)
        _validate_scalar_tensor("old_log_prob", self.old_log_prob, floating=True)
        if not isinstance(self.likelihood_semantics, LikelihoodSemantics):
            raise TypeError("likelihood_semantics must be a LikelihoodSemantics")
        _non_empty_string("condition_identity", self.condition_identity)
        _non_empty_string("guidance_identity", self.guidance_identity)
        _non_negative_int("transition_index", self.transition_index)
        _non_empty_string("storage_dtype_identity", self.storage_dtype_identity)
        _non_empty_string("quantization_identity", self.quantization_identity)
        if type(self.active) is not bool:
            raise TypeError("active must be bool")
        if self.storage_dtype_identity != str(self.x_t.dtype):
            raise ValueError("storage_dtype_identity must equal the stored x_t dtype")
        if self.sampled_action.dtype != self.x_t.dtype:
            raise TypeError("sampled_action and x_t must use the same dtype")
        if self.conditioned_next.dtype != self.x_t.dtype:
            raise TypeError("conditioned_next and x_t must use the same dtype")
        if self.t.device != self.x_t.device or self.t_next.device != self.x_t.device:
            raise ValueError("t/t_next must be stored on the latent device")
        if self.old_log_prob.device != self.x_t.device:
            raise ValueError("old_log_prob must be stored on the latent device")
        if not bool(torch.isfinite(self.old_log_prob)):
            raise ValueError("old_log_prob must be finite")
        for name in ("transition_std_dev", "rectification_coefficient"):
            value = getattr(self, name)
            if value is None:
                continue
            _validate_scalar_tensor(name, value, floating=True)
            if value.device != self.x_t.device:
                raise ValueError(f"{name} must be stored on the latent device")
            if not bool(value > 0):
                raise ValueError(f"{name} must be strictly positive")

    @property
    def scoring_target(self) -> Any:
        if self.likelihood_semantics is LikelihoodSemantics.EXACT_ENV_ACTION:
            return self.sampled_action
        return self.conditioned_next

    def serialize(self) -> dict[str, object]:
        self.validate()
        return {
            "x_t": _tensor_payload(self.x_t),
            "sampled_action": _tensor_payload(self.sampled_action),
            "conditioned_next": _tensor_payload(self.conditioned_next),
            "t": _tensor_payload(self.t),
            "t_next": _tensor_payload(self.t_next),
            "old_log_prob": _tensor_payload(self.old_log_prob),
            "likelihood_semantics": self.likelihood_semantics.value,
            "condition_identity": self.condition_identity,
            "guidance_identity": self.guidance_identity,
            "transition_index": self.transition_index,
            "storage_dtype_identity": self.storage_dtype_identity,
            "quantization_identity": self.quantization_identity,
            "active": self.active,
            "transition_std_dev": (
                None
                if self.transition_std_dev is None
                else _tensor_payload(self.transition_std_dev)
            ),
            "rectification_coefficient": (
                None
                if self.rectification_coefficient is None
                else _tensor_payload(self.rectification_coefficient)
            ),
        }


@dataclass(frozen=True)
class TrajectoryItem(ABC):
    """One trajectory and its decoded reward media, with no B axis."""

    context: TrajectoryContext
    steps: tuple[TrajectoryStep, ...]
    media: Any
    media_layout: Literal["CHW", "FCHW", "FHWC"]
    condition: ConditionPayload = NoCondition()

    KIND: ClassVar[str]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.context, TrajectoryContext):
            raise TypeError("context must be a TrajectoryContext")
        self.context.validate()
        if type(self.steps) is not tuple or not self.steps:
            raise ValueError("steps must be a non-empty tuple")
        if any(not isinstance(step, TrajectoryStep) for step in self.steps):
            raise TypeError("steps must contain only TrajectoryStep values")
        for step in self.steps:
            step.validate()
        semantics = {step.likelihood_semantics for step in self.steps}
        if len(semantics) != 1:
            raise ValueError("one trajectory cannot mix likelihood semantics")
        if not isinstance(self.condition, ConditionPayload):
            raise TypeError("condition must be a ConditionPayload")
        self.condition.validate()
        if isinstance(self.condition, NoCondition):
            expected_condition_identity = "none"
        elif isinstance(self.condition, CameraConditionPayload):
            expected_condition_identity = self.condition.condition_identity
        else:
            raise TypeError(
                f"unsupported condition payload: {type(self.condition).__name__}"
            )
        if any(
            step.condition_identity != expected_condition_identity
            for step in self.steps
        ):
            raise ValueError(
                "trajectory step condition identity must match its typed "
                "condition payload"
            )

        _validate_detached_tensor("media", self.media, floating=True)
        expected_ndim = {"CHW": 3, "FCHW": 4, "FHWC": 4}.get(self.media_layout)
        if expected_ndim is None:
            raise ValueError("media_layout must be CHW, FCHW, or FHWC")
        if self.media.ndim != expected_ndim:
            raise ValueError(
                f"media_layout {self.media_layout} requires an unbatched "
                f"{expected_ndim}-D tensor"
            )
        if isinstance(self.condition, CameraConditionPayload):
            if self.media_layout == "CHW":
                raise ValueError("image trajectory cannot carry a camera condition")
            if self.condition.camera_trajectory.shape[0] != self.media.shape[0]:
                raise ValueError(
                    "camera trajectory frame count must match video media frames"
                )

        indices = tuple(step.transition_index for step in self.steps)
        if len(indices) != len(set(indices)):
            raise ValueError("trajectory transition indices must be unique")
        for previous, current in zip(self.steps, self.steps[1:]):
            if current.transition_index != previous.transition_index + 1:
                raise ValueError("trajectory transition indices must be contiguous")
            if not self._requires_stochastic_state_chain():
                continue
            import torch

            if not torch.equal(previous.conditioned_next, current.x_t):
                raise ValueError(
                    "the next model state must equal the previous conditioned_next"
                )

    def _requires_stochastic_state_chain(self) -> bool:
        """Whether adjacent stored actions form one sampled state chain."""

        return True

    @property
    def likelihood_semantics(self) -> LikelihoodSemantics:
        return self.steps[0].likelihood_semantics

    def _serialize_common(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.KIND,
            "context": self.context.serialize(),
            "steps": [step.serialize() for step in self.steps],
            "media": _tensor_payload(self.media),
            "media_layout": self.media_layout,
            "condition": self.condition.serialize(),
        }

    @abstractmethod
    def serialize(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class FullTrajectoryItem(TrajectoryItem):
    KIND: ClassVar[Literal["full_trajectory"]] = "full_trajectory"

    def serialize(self) -> dict[str, object]:
        return self._serialize_common()


@dataclass(frozen=True)
class BranchingTrajectoryItem(TrajectoryItem):
    branch_topology: BranchTopology | None = None
    exploration_member_index: int = 0
    branch_step_index: int | None = None
    shared_prefix_id: str | None = None
    branch_step_identity: str | None = None
    transition_terminal_media: Any | None = None
    transition_terminal_media_layout: Literal["TCHW", "TFCHW", "TFHWC"] | None = None

    KIND: ClassVar[Literal["branching"]] = "branching"

    def validate(self) -> None:
        super().validate()
        topology = self.branch_topology
        if not isinstance(topology, BranchTopology):
            raise TypeError("branch_topology must be a BranchTopology")
        _non_negative_int(
            "exploration_member_index",
            self.exploration_member_index,
        )
        if self.exploration_member_index >= topology.exploration_count:
            raise ValueError("exploration_member_index is outside the branch topology")
        if topology.kind == "every_policy_timestep":
            if any(
                value is not None
                for value in (
                    self.branch_step_index,
                    self.shared_prefix_id,
                    self.branch_step_identity,
                )
            ):
                raise ValueError(
                    "every_policy_timestep does not accept single-point branch fields"
                )
            terminal = self.transition_terminal_media
            if terminal is None:
                raise ValueError(
                    "every_policy_timestep requires transition_terminal_media"
                )
            _validate_detached_tensor(
                "transition_terminal_media",
                terminal,
                floating=True,
            )
            expected_ndim = {
                "TCHW": 4,
                "TFCHW": 5,
                "TFHWC": 5,
            }.get(self.transition_terminal_media_layout)
            if expected_ndim is None or terminal.ndim != expected_ndim:
                raise ValueError(
                    "transition_terminal_media layout does not match its tensor"
                )
            if terminal.shape[0] != len(self.steps):
                raise ValueError(
                    "transition_terminal_media must contain one terminal medium "
                    "per stored policy timestep"
                )
        else:
            if self.branch_step_index is None:
                raise ValueError(
                    "single_point_branch_ablation requires branch_step_index"
                )
            _non_negative_int("branch_step_index", self.branch_step_index)
            _non_empty_string("shared_prefix_id", self.shared_prefix_id)
            _non_empty_string("branch_step_identity", self.branch_step_identity)
            if self.steps[0].transition_index != self.branch_step_index:
                raise ValueError(
                    "branch_step_index must identify the first stored branch transition"
                )
            if (
                self.transition_terminal_media is not None
                or self.transition_terminal_media_layout is not None
            ):
                raise ValueError(
                    "single_point_branch_ablation uses the ordinary row media field"
                )

    def _requires_stochastic_state_chain(self) -> bool:
        topology = self.branch_topology
        return not (
            isinstance(topology, BranchTopology)
            and topology.kind == "every_policy_timestep"
        )

    def serialize(self) -> dict[str, object]:
        result = self._serialize_common()
        assert self.branch_topology is not None
        result["branch_topology"] = self.branch_topology.to_payload()
        result["exploration_member_index"] = self.exploration_member_index
        if self.branch_topology.kind == "every_policy_timestep":
            result["transition_terminal_media"] = _tensor_payload(
                self.transition_terminal_media
            )
            result["transition_terminal_media_layout"] = (
                self.transition_terminal_media_layout
            )
        else:
            result["branch_step_index"] = self.branch_step_index
            result["shared_prefix_id"] = self.shared_prefix_id
            result["branch_step_identity"] = self.branch_step_identity
        return result


@dataclass(frozen=True)
class SingleStepTrajectoryItem(TrajectoryItem):
    selected_timestep_index: int = 0
    selection_policy_identity: str = ""
    selection_mapping_identity: str = ""

    KIND: ClassVar[Literal["single_step"]] = "single_step"

    def validate(self) -> None:
        super().validate()
        _non_negative_int("selected_timestep_index", self.selected_timestep_index)
        _non_empty_string("selection_policy_identity", self.selection_policy_identity)
        _non_empty_string("selection_mapping_identity", self.selection_mapping_identity)
        if len(self.steps) != 1:
            raise ValueError("single_step trajectory must contain exactly one step")
        if self.steps[0].transition_index != self.selected_timestep_index:
            raise ValueError(
                "selected_timestep_index must identify the stored transition"
            )

    def serialize(self) -> dict[str, object]:
        result = self._serialize_common()
        result["selected_timestep_index"] = self.selected_timestep_index
        result["selection_policy_identity"] = self.selection_policy_identity
        result["selection_mapping_identity"] = self.selection_mapping_identity
        return result


def _batch_indices(indices: Any, batch_size: int) -> list[int]:
    import torch

    if isinstance(indices, slice):
        resolved = list(range(batch_size))[indices]
    elif isinstance(indices, torch.Tensor):
        if indices.ndim != 1 or indices.dtype not in {
            torch.int32,
            torch.int64,
        }:
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


@dataclass(frozen=True)
class TrajectoryBatch:
    """Homogeneous padded-free batch produced only by trajectory collation."""

    kind: Literal["full_trajectory", "branching", "single_step"]
    contexts: tuple[TrajectoryContext, ...]
    x_t: Any
    sampled_action: Any
    conditioned_next: Any
    timesteps: Any
    next_timesteps: Any
    old_log_probs: Any
    transition_mask: Any
    transition_index: Any
    likelihood_semantics: LikelihoodSemantics
    condition_identity: tuple[tuple[str, ...], ...]
    guidance_identity: tuple[tuple[str, ...], ...]
    storage_dtype_identity: tuple[tuple[str, ...], ...]
    quantization_identity: tuple[tuple[str, ...], ...]
    media: Any
    media_layout: Literal["BCHW", "BFCHW", "BFHWC"]
    condition_state: Any
    branch_topology: BranchTopology | None = None
    branch_group_completeness: Literal["complete", "sliced_subset"] | None = None
    exploration_member_index: Any | None = None
    branch_timestep_index: Any | None = None
    transition_terminal_media: Any | None = None
    transition_terminal_media_layout: Literal["BTCHW", "BTFCHW", "BTFHWC"] | None = None
    branch_step_index: Any | None = None
    selected_timestep_index: Any | None = None
    shared_prefix_id: tuple[str, ...] | None = None
    branch_step_identity: tuple[str, ...] | None = None
    selection_policy_identity: str | None = None
    selection_mapping_identity: str | None = None
    transition_std_dev: Any | None = None
    rectification_coefficient: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "likelihood_semantics",
            _semantics(self.likelihood_semantics),
        )
        if self.kind == "branching" and self.branch_group_completeness is None:
            object.__setattr__(self, "branch_group_completeness", "complete")
        self.validate()

    @property
    def batch_size(self) -> int:
        return len(self.contexts)

    @property
    def transition_count(self) -> int:
        return int(self.old_log_probs.shape[1])

    @property
    def scoring_target(self) -> Any:
        if self.likelihood_semantics is LikelihoodSemantics.EXACT_ENV_ACTION:
            return self.sampled_action
        return self.conditioned_next

    def validate(self) -> None:
        import torch

        if self.kind not in {"full_trajectory", "branching", "single_step"}:
            raise ValueError(f"unknown trajectory kind: {self.kind!r}")
        if type(self.contexts) is not tuple or not self.contexts:
            raise ValueError("contexts must be a non-empty tuple")
        if any(not isinstance(item, TrajectoryContext) for item in self.contexts):
            raise TypeError("contexts must contain TrajectoryContext values")
        for context in self.contexts:
            context.validate()
        sample_ids = tuple(context.sample_id for context in self.contexts)
        trajectory_ids = tuple(context.trajectory_id for context in self.contexts)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_id must be unique within a trajectory batch")
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("trajectory_id must be unique within a batch")

        batch_size = self.batch_size
        latent_fields = (self.x_t, self.sampled_action, self.conditioned_next)
        for name, value in zip(
            ("x_t", "sampled_action", "conditioned_next"),
            latent_fields,
            strict=True,
        ):
            _validate_detached_tensor(name, value, floating=True)
            if value.ndim < 3 or value.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape [B,T,...]")
        if tuple(self.sampled_action.shape) != tuple(self.x_t.shape):
            raise ValueError("sampled_action must match x_t shape")
        if tuple(self.conditioned_next.shape) != tuple(self.x_t.shape):
            raise ValueError("conditioned_next must match x_t shape")
        transition_shape = tuple(self.x_t.shape[:2])
        for name in (
            "timesteps",
            "next_timesteps",
            "old_log_probs",
            "transition_mask",
            "transition_index",
        ):
            value = getattr(self, name)
            _validate_detached_tensor(name, value)
            if tuple(value.shape) != transition_shape:
                raise ValueError(f"{name} must have shape [B,T]")
        if not self.old_log_probs.is_floating_point():
            raise TypeError("old_log_probs must be floating point")
        if self.transition_mask.dtype != torch.bool:
            raise TypeError("transition_mask must use bool dtype")
        if self.transition_index.dtype != torch.int64:
            raise TypeError("transition_index must use torch.int64")
        if not bool(self.transition_mask.any()):
            raise ValueError("trajectory batch must contain an active transition")
        active_log_prob = self.old_log_probs.masked_select(self.transition_mask)
        if not bool(torch.isfinite(active_log_prob).all()):
            raise ValueError("active old_log_probs must be finite")
        for name in ("transition_std_dev", "rectification_coefficient"):
            value = getattr(self, name)
            if value is None:
                continue
            _validate_detached_tensor(name, value, floating=True)
            if tuple(value.shape) != transition_shape:
                raise ValueError(f"{name} must have shape [B,T]")
            if value.device != self.x_t.device or value.dtype != self.x_t.dtype:
                raise ValueError(f"{name} must share latent device/dtype")
            if not bool((value > 0).all()):
                raise ValueError(f"{name} must be strictly positive")
        if not isinstance(self.likelihood_semantics, LikelihoodSemantics):
            raise TypeError("invalid likelihood semantics")

        for name in (
            "condition_identity",
            "guidance_identity",
            "storage_dtype_identity",
            "quantization_identity",
        ):
            rows = getattr(self, name)
            if type(rows) is not tuple or len(rows) != batch_size:
                raise ValueError(f"{name} must contain B rows")
            if any(
                type(row) is not tuple or len(row) != transition_shape[1]
                for row in rows
            ):
                raise ValueError(f"{name} rows must contain T values")
            if any(
                not isinstance(value, str) or not value for row in rows for value in row
            ):
                raise ValueError(f"{name} values must be non-empty strings")
        if any(
            value != str(self.x_t.dtype)
            for row in self.storage_dtype_identity
            for value in row
        ):
            raise ValueError("storage dtype identity does not match x_t")

        _validate_detached_tensor("media", self.media, floating=True)
        expected_media_ndim = {
            "BCHW": 4,
            "BFCHW": 5,
            "BFHWC": 5,
        }.get(self.media_layout)
        if expected_media_ndim is None:
            raise ValueError("invalid batched media layout")
        if self.media.ndim != expected_media_ndim or self.media.shape[0] != batch_size:
            raise ValueError("media layout does not match batched media shape")

        from visual_rl.data.samples.collate import ConditionBatchState

        if not isinstance(self.condition_state, ConditionBatchState):
            raise TypeError("condition_state must be a ConditionBatchState")
        self.condition_state.validate()
        if self.condition_state.batch_size != batch_size:
            raise ValueError("condition_state batch size mismatch")
        if hasattr(self.condition_state, "row_condition_identities"):
            expected_identities = self.condition_state.row_condition_identities
        else:
            expected_identities = ("none",) * batch_size
        for row, expected_identity in zip(
            self.condition_identity,
            expected_identities,
            strict=True,
        ):
            if any(value != expected_identity for value in row):
                raise ValueError(
                    "trajectory condition identity must match condition_state"
                )

        for name in ("branch_step_index", "selected_timestep_index"):
            value = getattr(self, name)
            if value is not None:
                _validate_detached_tensor(name, value)
                if value.dtype != torch.int64 or tuple(value.shape) != (batch_size,):
                    raise ValueError(f"{name} must be int64 [B]")
        if self.kind == "branching":
            topology = self.branch_topology
            if not isinstance(topology, BranchTopology):
                raise TypeError("branching requires a BranchTopology")
            if self.branch_group_completeness not in {
                "complete",
                "sliced_subset",
            }:
                raise ValueError("branching requires explicit group completeness")
            if self.selected_timestep_index is not None:
                raise ValueError("branching does not accept selected_timestep_index")
            if (
                self.selection_policy_identity is not None
                or self.selection_mapping_identity is not None
            ):
                raise ValueError("branching does not accept selection identities")
            grouped: dict[str, list[int]] = {}
            for row_index, context in enumerate(self.contexts):
                grouped.setdefault(context.batch_row.group_id, []).append(row_index)
            expected_members = set(range(topology.exploration_count))
            for row_indices in grouped.values():
                members = {
                    self.contexts[index].batch_row.member_id for index in row_indices
                }
                if self.branch_group_completeness == "complete":
                    if (
                        len(row_indices) != topology.exploration_count
                        or members != expected_members
                    ):
                        raise ValueError(
                            "a complete branch group must enumerate exactly K "
                            "exploration members"
                        )
                elif (
                    len(row_indices) > topology.exploration_count
                    or not members
                    or not members.issubset(expected_members)
                ):
                    raise ValueError(
                        "branch rows contain an invalid exploration member"
                    )
                transition_sequences = {
                    tuple(
                        int(item)
                        for item in self.transition_index[index]
                        .detach()
                        .to(device="cpu")
                        .tolist()
                    )
                    for index in row_indices
                }
                if len(transition_sequences) != 1:
                    raise ValueError(
                        "branch members in one prompt group must share the "
                        "transition-index axis"
                    )
            member_index = self.exploration_member_index
            if not isinstance(member_index, torch.Tensor):
                raise TypeError("branching requires exploration_member_index")
            _validate_detached_tensor("exploration_member_index", member_index)
            if member_index.dtype != torch.int64 or tuple(member_index.shape) != (
                batch_size,
            ):
                raise ValueError("exploration_member_index must be int64 [B]")
            expected_member_index = torch.tensor(
                [context.batch_row.member_id for context in self.contexts],
                dtype=torch.int64,
                device=member_index.device,
            )
            if not torch.equal(member_index, expected_member_index):
                raise ValueError(
                    "exploration_member_index must match the batch row contexts"
                )
            if topology.kind == "every_policy_timestep":
                self._validate_every_policy_timestep_topology(transition_shape)
            else:
                self._validate_single_point_branch_ablation(grouped)
        elif self.kind == "single_step":
            if (
                self.selected_timestep_index is None
                or self.branch_step_index is not None
            ):
                raise ValueError("single_step requires only selected_timestep_index")
            if self.transition_count != 1:
                raise ValueError(
                    "single_step batch must contain exactly one transition"
                )
            if (
                self.branch_topology is not None
                or self.branch_group_completeness is not None
                or self.exploration_member_index is not None
                or self.branch_timestep_index is not None
                or self.transition_terminal_media is not None
                or self.transition_terminal_media_layout is not None
                or self.shared_prefix_id is not None
                or self.branch_step_identity is not None
            ):
                raise ValueError("single_step does not accept branch identities")
            for name in (
                "selection_policy_identity",
                "selection_mapping_identity",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"single_step requires {name}")
        elif (
            self.branch_topology is not None
            or self.branch_group_completeness is not None
            or self.exploration_member_index is not None
            or self.branch_timestep_index is not None
            or self.transition_terminal_media is not None
            or self.transition_terminal_media_layout is not None
            or self.branch_step_index is not None
            or self.selected_timestep_index is not None
            or self.shared_prefix_id is not None
            or self.branch_step_identity is not None
            or self.selection_policy_identity is not None
            or self.selection_mapping_identity is not None
        ):
            raise ValueError("full trajectory does not accept branch/selected fields")

    def _validate_every_policy_timestep_topology(
        self,
        transition_shape: tuple[int, int],
    ) -> None:
        import torch

        if any(
            value is not None
            for value in (
                self.branch_step_index,
                self.shared_prefix_id,
                self.branch_step_identity,
            )
        ):
            raise ValueError(
                "every_policy_timestep does not accept single-point branch fields"
            )
        branch_timestep = self.branch_timestep_index
        if not isinstance(branch_timestep, torch.Tensor):
            raise TypeError("every_policy_timestep requires branch_timestep_index")
        _validate_detached_tensor("branch_timestep_index", branch_timestep)
        if (
            branch_timestep.dtype != torch.int64
            or tuple(branch_timestep.shape) != transition_shape
        ):
            raise ValueError("branch_timestep_index must be int64 [B,T]")
        if not torch.equal(branch_timestep, self.transition_index):
            raise ValueError(
                "branch_timestep_index must identify each stored policy record"
            )
        expected_timestep_axis = torch.arange(
            transition_shape[1],
            dtype=torch.int64,
            device=branch_timestep.device,
        ).expand(transition_shape[0], -1)
        if not torch.equal(branch_timestep, expected_timestep_axis):
            raise ValueError("every_policy_timestep must store the ordered 0..N-1 axis")
        if not bool(self.transition_mask.all()):
            raise ValueError(
                "every_policy_timestep requires every stored transition to be active"
            )
        terminal = self.transition_terminal_media
        if not isinstance(terminal, torch.Tensor):
            raise TypeError("every_policy_timestep requires transition_terminal_media")
        _validate_detached_tensor(
            "transition_terminal_media",
            terminal,
            floating=True,
        )
        expected_ndim = {
            "BTCHW": 5,
            "BTFCHW": 6,
            "BTFHWC": 6,
        }.get(self.transition_terminal_media_layout)
        if expected_ndim is None or terminal.ndim != expected_ndim:
            raise ValueError(
                "transition_terminal_media layout does not match its tensor"
            )
        if tuple(terminal.shape[:2]) != transition_shape:
            raise ValueError("transition_terminal_media must have leading shape [B,T]")

    def _validate_single_point_branch_ablation(
        self,
        grouped: dict[str, list[int]],
    ) -> None:
        if self.branch_step_index is None:
            raise ValueError("single_point_branch_ablation requires branch_step_index")
        if (
            self.branch_timestep_index is not None
            or self.transition_terminal_media is not None
            or self.transition_terminal_media_layout is not None
        ):
            raise ValueError(
                "single_point_branch_ablation uses ordinary row media only"
            )
        for name in ("shared_prefix_id", "branch_step_identity"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != self.batch_size:
                raise ValueError(f"branching requires {name} with B entries")
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} values must be non-empty strings")
        assert self.shared_prefix_id is not None
        assert self.branch_step_identity is not None
        for row_indices in grouped.values():
            prefix_ids = {self.shared_prefix_id[index] for index in row_indices}
            step_ids = {self.branch_step_identity[index] for index in row_indices}
            branch_steps = {
                int(self.branch_step_index[index].item()) for index in row_indices
            }
            if len(prefix_ids) != 1 or len(step_ids) != 1 or len(branch_steps) != 1:
                raise ValueError(
                    "branch rows in one group must share prefix and branch-step identity"
                )

    def slice(self, indices: Any) -> "TrajectoryBatch":
        resolved = _batch_indices(indices, self.batch_size)
        import torch

        branch_group_completeness = self.branch_group_completeness
        if self.kind == "branching":
            assert self.branch_topology is not None
            selected_groups: dict[str, set[int]] = {}
            for index in resolved:
                row = self.contexts[index].batch_row
                selected_groups.setdefault(row.group_id, set()).add(row.member_id)
            expected_members = set(range(self.branch_topology.exploration_count))
            branch_group_completeness = (
                "complete"
                if all(
                    members == expected_members for members in selected_groups.values()
                )
                else "sliced_subset"
            )

        cache: dict[object, Any] = {}

        def select(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                key = value.device
                index = cache.get(key)
                if index is None:
                    index = torch.tensor(
                        resolved, dtype=torch.long, device=value.device
                    )
                    cache[key] = index
                return value.index_select(0, index)
            return tuple(value[item] for item in resolved)

        return replace(
            self,
            contexts=select(self.contexts),
            x_t=select(self.x_t),
            sampled_action=select(self.sampled_action),
            conditioned_next=select(self.conditioned_next),
            timesteps=select(self.timesteps),
            next_timesteps=select(self.next_timesteps),
            old_log_probs=select(self.old_log_probs),
            transition_mask=select(self.transition_mask),
            transition_index=select(self.transition_index),
            condition_identity=select(self.condition_identity),
            guidance_identity=select(self.guidance_identity),
            storage_dtype_identity=select(self.storage_dtype_identity),
            quantization_identity=select(self.quantization_identity),
            media=select(self.media),
            condition_state=self.condition_state.slice(resolved),
            branch_group_completeness=branch_group_completeness,
            exploration_member_index=select(self.exploration_member_index),
            branch_timestep_index=select(self.branch_timestep_index),
            transition_terminal_media=select(self.transition_terminal_media),
            branch_step_index=select(self.branch_step_index),
            selected_timestep_index=select(self.selected_timestep_index),
            shared_prefix_id=select(self.shared_prefix_id),
            branch_step_identity=select(self.branch_step_identity),
            transition_std_dev=select(self.transition_std_dev),
            rectification_coefficient=select(self.rectification_coefficient),
        )

    def to(self, device: Any, dtype: Any = None) -> "TrajectoryBatch":
        import torch

        def move(value: Any, *, cast: bool) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            target_dtype = (
                dtype
                if cast
                and dtype is not None
                and (value.is_floating_point() or value.is_complex())
                else None
            )
            return value.to(device=device, dtype=target_dtype)

        new_x = move(self.x_t, cast=True)
        storage = tuple(
            tuple(str(new_x.dtype) for _ in row) for row in self.storage_dtype_identity
        )
        return replace(
            self,
            x_t=new_x,
            sampled_action=move(self.sampled_action, cast=True),
            conditioned_next=move(self.conditioned_next, cast=True),
            timesteps=move(self.timesteps, cast=False),
            next_timesteps=move(self.next_timesteps, cast=False),
            old_log_probs=move(self.old_log_probs, cast=True),
            transition_mask=move(self.transition_mask, cast=False),
            transition_index=move(self.transition_index, cast=False),
            storage_dtype_identity=storage,
            media=move(self.media, cast=True),
            condition_state=self.condition_state.to(device),
            exploration_member_index=move(
                self.exploration_member_index,
                cast=False,
            ),
            branch_timestep_index=move(
                self.branch_timestep_index,
                cast=False,
            ),
            transition_terminal_media=move(
                self.transition_terminal_media,
                cast=True,
            ),
            branch_step_index=move(self.branch_step_index, cast=False),
            selected_timestep_index=move(
                self.selected_timestep_index,
                cast=False,
            ),
            transition_std_dev=move(self.transition_std_dev, cast=True),
            rectification_coefficient=move(
                self.rectification_coefficient,
                cast=True,
            ),
        )

    def detach(self) -> "TrajectoryBatch":
        def detached(value: Any) -> Any:
            return value.detach() if hasattr(value, "detach") else value

        return replace(
            self,
            x_t=detached(self.x_t),
            sampled_action=detached(self.sampled_action),
            conditioned_next=detached(self.conditioned_next),
            timesteps=detached(self.timesteps),
            next_timesteps=detached(self.next_timesteps),
            old_log_probs=detached(self.old_log_probs),
            transition_mask=detached(self.transition_mask),
            transition_index=detached(self.transition_index),
            media=detached(self.media),
            condition_state=self.condition_state.detach(),
            exploration_member_index=detached(self.exploration_member_index),
            branch_timestep_index=detached(self.branch_timestep_index),
            transition_terminal_media=detached(self.transition_terminal_media),
            branch_step_index=detached(self.branch_step_index),
            selected_timestep_index=detached(self.selected_timestep_index),
            transition_std_dev=detached(self.transition_std_dev),
            rectification_coefficient=detached(self.rectification_coefficient),
        )


__all__ = [
    "BranchAxis",
    "BranchingTrajectoryItem",
    "BranchTopology",
    "BranchTopologyKind",
    "FullTrajectoryItem",
    "LikelihoodSemantics",
    "SingleStepTrajectoryItem",
    "TrajectoryBatch",
    "TrajectoryItem",
    "TrajectoryStep",
]
