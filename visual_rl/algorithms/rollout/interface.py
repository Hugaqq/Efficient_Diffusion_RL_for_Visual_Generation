"""Canonical rollout runtime contracts, independent of registries and models."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from visual_rl.algorithms.conditioning.interface import (
    LatentConditioner,
    LatentSpec,
)
from visual_rl.algorithms.dynamics.interface import Dynamics
from visual_rl.algorithms.dynamics.replay import DynamicsReplayBinding
from visual_rl.algorithms.dynamics.session import DynamicsSession, ScheduleSnapshot
from visual_rl.core.contracts import LikelihoodSemantics
from visual_rl.core.contracts.runtime import (
    ExecutionPolicyReceipt,
    PolicyRuntimePort,
    RolloutExecutionPolicy,
)
from visual_rl.data.samples import (
    StackedSampleBatch,
    TrajectoryBatch,
)
from visual_rl.models.interface import BatchRowProjection, ModelLatentSpec

__all__ = (
    "ModelForwardReplayPlan",
    "RolloutComponent",
    "RolloutContractError",
    "RolloutExecution",
    "RolloutRequest",
    "project_model_payload_rows",
)


class RolloutContractError(ValueError):
    """Raised when runtime rollout ports or identities do not align."""


def _require_policy_port(value: object, *, methods: tuple[str, ...]) -> None:
    """Validate the operational port without importing a concrete Adapter.

    Lightweight unit adapters remain structurally usable, while production
    injects the fully bound ``PolicyRuntimePort`` implementation.
    """

    missing = tuple(
        name for name in methods if not callable(getattr(value, name, None))
    )
    if missing:
        raise TypeError(
            "adapter must implement the PolicyRuntimePort operations: "
            + ", ".join(missing)
        )


def _identities(name: str, values: object, batch_size: int) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) != batch_size:
        raise RolloutContractError(f"{name} must contain B entries")
    if any(not isinstance(item, str) or not item for item in values):
        raise RolloutContractError(f"{name} values must be non-empty strings")
    return values


def project_model_payload_rows(
    payload: object | None,
    row_indices: tuple[int, ...],
    *,
    label: str,
    identity_attribute: str,
    expected_identity: tuple[str, ...],
    require_projection: bool,
) -> object | None:
    """Project an explicitly row-batched model payload without guessing.

    ``None`` and batch-invariant payloads (for example a scalar guidance
    configuration) pass through unchanged.  A payload opts into row geometry
    by exposing ``project_rows``; after projection it must expose the matching
    per-row identity attribute.  This keeps rollout and differentiable replay
    on the same structural boundary without inspecting model-specific fields
    or treating arbitrary mappings as batched values.
    """

    if type(row_indices) is not tuple or not row_indices:
        raise RolloutContractError(f"{label} row_indices must be a non-empty tuple")
    if any(type(index) is not int or index < 0 for index in row_indices):
        raise RolloutContractError(
            f"{label} row_indices must contain non-negative integers"
        )
    _identities(f"expected {label} identity", expected_identity, len(row_indices))
    if payload is None:
        if require_projection:
            raise RolloutContractError(f"{label} payload must not be None")
        return None

    project_rows = getattr(payload, "project_rows", None)
    if not callable(project_rows):
        if require_projection:
            raise RolloutContractError(
                f"{label} row projection requires payload.project_rows()"
            )
        return payload

    source_batch_size = getattr(payload, "batch_size", None)
    if type(source_batch_size) is not int or source_batch_size < 1:
        raise RolloutContractError(
            f"row-batched {label} must expose a positive batch_size"
        )
    selected = project_rows(
        BatchRowProjection(
            source_batch_size=source_batch_size,
            row_indices=row_indices,
        )
    )
    if selected is None:
        raise RolloutContractError(f"{label}.project_rows() returned None")
    observed_identity = getattr(selected, identity_attribute, None)
    if observed_identity is None:
        raise RolloutContractError(
            f"row-batched {label} must expose {identity_attribute}"
        )
    _identities(f"projected {label} identity", observed_identity, len(row_indices))
    if observed_identity != expected_identity:
        raise RolloutContractError(f"projected {label} changed its per-row identity")
    return selected


def _model_condition_identities(
    conditioning: object,
    samples: StackedSampleBatch,
) -> tuple[str, ...]:
    values = getattr(conditioning, "condition_identity", None)
    if values is None:
        values = tuple(row.identity for row in samples.rows)
    return _identities(
        "model conditioning identity",
        values,
        samples.batch_size,
    )


@dataclass(frozen=True, slots=True)
class ModelForwardReplayPlan:
    """Freeze the model-forward batch geometry that produced policy actions.

    ``forward_row_indices`` are the canonical trajectory rows evaluated by the
    model in one forward. ``row_to_forward_position`` expands that prediction
    batch back onto every stored trajectory row.  Independent rollouts use the
    identity plan; grouped exploration may share one forward row across K
    sampled policy rows without exposing an algorithm or model name.
    """

    forward_row_indices: tuple[int, ...]
    row_to_forward_position: tuple[int, ...]
    forward_partitions: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if type(self.forward_row_indices) is not tuple or not self.forward_row_indices:
            raise ValueError("forward_row_indices must be a non-empty tuple")
        if any(type(row) is not int or row < 0 for row in self.forward_row_indices):
            raise ValueError("forward_row_indices must contain non-negative integers")
        if len(set(self.forward_row_indices)) != len(self.forward_row_indices):
            raise ValueError("forward_row_indices must not contain duplicates")
        if (
            type(self.row_to_forward_position) is not tuple
            or not self.row_to_forward_position
        ):
            raise ValueError("row_to_forward_position must be a non-empty tuple")
        forward_count = len(self.forward_row_indices)
        if any(
            type(position) is not int or not 0 <= position < forward_count
            for position in self.row_to_forward_position
        ):
            raise ValueError("row_to_forward_position must index forward_row_indices")
        batch_size = len(self.row_to_forward_position)
        if any(row >= batch_size for row in self.forward_row_indices):
            raise ValueError("forward row is outside the trajectory batch")
        if set(self.row_to_forward_position) != set(range(forward_count)):
            raise ValueError("every model-forward row must serve a trajectory row")
        for position, row in enumerate(self.forward_row_indices):
            if self.row_to_forward_position[row] != position:
                raise ValueError("each model-forward leader must map to itself")
        partitions = self.forward_partitions
        if partitions is None:
            partitions = (self.forward_row_indices,)
            object.__setattr__(self, "forward_partitions", partitions)
        if type(partitions) is not tuple or not partitions:
            raise ValueError("forward_partitions must be a non-empty tuple")
        if any(
            type(partition) is not tuple or not partition for partition in partitions
        ):
            raise ValueError("each forward partition must be a non-empty tuple")
        flattened = tuple(row for partition in partitions for row in partition)
        if flattened != self.forward_row_indices:
            raise ValueError(
                "forward_partitions must preserve the exact forward-row order"
            )

    @classmethod
    def independent(
        cls,
        batch_size: int,
        *,
        microbatch_size: int | None = None,
    ) -> ModelForwardReplayPlan:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if microbatch_size is not None and (
            type(microbatch_size) is not int or microbatch_size < 1
        ):
            raise ValueError("microbatch_size must be a positive integer or None")
        rows = tuple(range(batch_size))
        width = batch_size if microbatch_size is None else microbatch_size
        return cls(
            forward_row_indices=rows,
            row_to_forward_position=rows,
            forward_partitions=tuple(
                rows[start : start + width] for start in range(0, batch_size, width)
            ),
        )

    @property
    def batch_size(self) -> int:
        return len(self.row_to_forward_position)

    @property
    def is_independent(self) -> bool:
        return self.forward_row_indices == tuple(range(self.batch_size)) and (
            self.row_to_forward_position == tuple(range(self.batch_size))
        )

    @property
    def partition_identity(self) -> str:
        payload = {
            "schema_version": 1,
            "forward_row_indices": list(self.forward_row_indices),
            "row_to_forward_position": list(self.row_to_forward_position),
            "forward_partitions": [
                list(item) for item in self.forward_partitions or ()
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"model-forward-partitions.v1:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class RolloutExecution:
    """Trajectory plus the exact schedule snapshot needed for recomputation."""

    trajectory: TrajectoryBatch
    schedule_snapshot: ScheduleSnapshot
    encoded_conditioning: object
    model_condition_identity: tuple[str, ...]
    model_forward_replay: ModelForwardReplayPlan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, TrajectoryBatch):
            raise TypeError("trajectory must be a TrajectoryBatch")
        if not isinstance(self.schedule_snapshot, ScheduleSnapshot):
            raise TypeError("schedule_snapshot must be a ScheduleSnapshot")
        if self.encoded_conditioning is None:
            raise ValueError("encoded_conditioning must not be None")
        replay = self.model_forward_replay
        if replay is None:
            replay = ModelForwardReplayPlan.independent(self.trajectory.batch_size)
            object.__setattr__(self, "model_forward_replay", replay)
        if not isinstance(replay, ModelForwardReplayPlan):
            raise TypeError("model_forward_replay must be ModelForwardReplayPlan")
        if replay.batch_size != self.trajectory.batch_size:
            raise RolloutContractError(
                "model-forward replay plan does not match trajectory batch size"
            )
        _identities(
            "model_condition_identity",
            self.model_condition_identity,
            self.trajectory.batch_size,
        )
        encoded_identity = getattr(
            self.encoded_conditioning,
            "condition_identity",
            self.model_condition_identity,
        )
        if encoded_identity != self.model_condition_identity:
            raise RolloutContractError(
                "encoded conditioning identity changed before rollout handoff"
            )
        self._validate_model_forward_replay(replay)
        if self.trajectory.kind == "single_step":
            if self.trajectory.selection_policy_identity != (
                self.schedule_snapshot.selection_policy
            ):
                raise RolloutContractError(
                    "single-step trajectory selection policy differs from its "
                    "schedule snapshot"
                )
            if self.trajectory.selection_mapping_identity != (
                self.schedule_snapshot.randomness_identity
            ):
                raise RolloutContractError(
                    "single-step trajectory selection mapping differs from its "
                    "schedule snapshot"
                )

    def _validate_model_forward_replay(
        self,
        replay: ModelForwardReplayPlan,
    ) -> None:
        """Reject grouped replay metadata that cannot reproduce model inputs."""

        if replay.is_independent:
            return
        import torch

        trajectory = self.trajectory
        for row, position in enumerate(replay.row_to_forward_position):
            leader = replay.forward_row_indices[position]
            if (
                self.model_condition_identity[row]
                != self.model_condition_identity[leader]
            ):
                raise RolloutContractError(
                    "shared model-forward rows must share model condition identity"
                )
            if (
                trajectory.guidance_identity[row]
                != trajectory.guidance_identity[leader]
            ):
                raise RolloutContractError(
                    "shared model-forward rows must share guidance identity"
                )
            for name in ("x_t", "timesteps", "next_timesteps"):
                value = getattr(trajectory, name)
                if not torch.equal(value[row], value[leader]):
                    raise RolloutContractError(
                        f"shared model-forward rows must share stored {name}"
                    )


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    """Every runtime dependency required by a rollout, with no hidden RNG."""

    adapter: PolicyRuntimePort
    dynamics: Dynamics
    samples: StackedSampleBatch
    latent_spec: ModelLatentSpec
    generator: Any
    likelihood_semantics: LikelihoodSemantics
    guidance: object | None = None
    guidance_identity: tuple[str, ...] = ()
    quantization_identity: tuple[str, ...] = ()
    conditioner: LatentConditioner | None = None
    conditioner_latent_spec: LatentSpec | None = None
    selection_generator: Any | None = None
    dynamics_session: DynamicsSession | None = None
    dynamics_replay_binding: DynamicsReplayBinding[object] | None = None
    encoded_conditioning: object | None = None
    model_condition_identity: tuple[str, ...] = ()
    selection_contract_identity: str = ""

    def __post_init__(self) -> None:
        import torch

        _require_policy_port(
            self.adapter,
            methods=("encode", "prepare_latents", "predict", "decode"),
        )
        if not isinstance(self.dynamics, Dynamics):
            raise TypeError("dynamics must be a Dynamics")
        if not isinstance(self.samples, StackedSampleBatch):
            raise TypeError("samples must be a StackedSampleBatch")
        if not isinstance(self.latent_spec, ModelLatentSpec):
            raise TypeError("latent_spec must be a ModelLatentSpec")
        self.samples.validate()
        if self.latent_spec.batch_size != self.samples.batch_size:
            raise RolloutContractError(
                "latent_spec and sample batch must have the same B dimension"
            )
        if not isinstance(self.generator, torch.Generator):
            raise TypeError("rollout requires an explicit torch.Generator")
        generator_device = torch.device(self.generator.device)
        if generator_device.type != self.latent_spec.device.type:
            raise RolloutContractError(
                "generator device type must match latent device type"
            )
        selection_generator = self.selection_generator
        if selection_generator is None:
            selection_generator = self.generator
            object.__setattr__(self, "selection_generator", selection_generator)
        if not isinstance(selection_generator, torch.Generator):
            raise TypeError("selection_generator must be a torch.Generator")
        selection_device = torch.device(selection_generator.device)
        if selection_device.type != self.latent_spec.device.type:
            raise RolloutContractError(
                "selection generator device type must match latent device type"
            )
        if self.dynamics_session is not None:
            if not isinstance(self.dynamics_session, DynamicsSession):
                raise TypeError("dynamics_session must be a DynamicsSession")
            if self.dynamics_session.dynamics is not self.dynamics:
                raise RolloutContractError(
                    "dynamics_session must bind the request Dynamics instance"
                )
        bound_replay = getattr(self.dynamics, "replay_binding", None)
        request_replay = self.dynamics_replay_binding
        if bound_replay is None:
            if request_replay is not None:
                raise RolloutContractError(
                    "dynamics_replay_binding requires a replay-bound Dynamics"
                )
        else:
            if not isinstance(bound_replay, DynamicsReplayBinding):
                raise TypeError("Dynamics replay_binding has an invalid type")
            if not isinstance(request_replay, DynamicsReplayBinding):
                raise RolloutContractError(
                    "replay-bound Dynamics requires an explicit request binding"
                )
            if request_replay.binding_identity != bound_replay.binding_identity:
                raise RolloutContractError(
                    "request replay binding does not match its Dynamics"
                )
            if request_replay.replay_state is not getattr(
                self.dynamics, "replay_state", None
            ):
                raise RolloutContractError(
                    "request replay state is not the state owned by its Dynamics"
                )
        if self.encoded_conditioning is None:
            if self.model_condition_identity:
                raise RolloutContractError(
                    "model_condition_identity requires encoded_conditioning"
                )
        else:
            identities = self.model_condition_identity
            if not identities:
                identities = _model_condition_identities(
                    self.encoded_conditioning,
                    self.samples,
                )
                object.__setattr__(self, "model_condition_identity", identities)
            _identities(
                "model_condition_identity",
                identities,
                self.samples.batch_size,
            )
            encoded_identity = getattr(
                self.encoded_conditioning,
                "condition_identity",
                identities,
            )
            if encoded_identity != identities:
                raise RolloutContractError(
                    "encoded conditioning identity does not match request"
                )
        try:
            semantics = LikelihoodSemantics(self.likelihood_semantics)
        except (TypeError, ValueError):
            raise RolloutContractError("invalid likelihood semantics") from None
        object.__setattr__(self, "likelihood_semantics", semantics)

        if not isinstance(self.selection_contract_identity, str):
            raise TypeError("selection_contract_identity must be a string")
        if self.selection_contract_identity and (
            self.selection_contract_identity.strip() != self.selection_contract_identity
        ):
            raise RolloutContractError(
                "selection_contract_identity must be a canonical string"
            )

        batch_size = self.samples.batch_size
        guidance_identity = self.guidance_identity
        if not guidance_identity:
            if self.guidance is not None:
                raise RolloutContractError(
                    "guidance requires one explicit identity per batch row"
                )
            guidance_identity = ("none",) * batch_size
            object.__setattr__(self, "guidance_identity", guidance_identity)
        _identities("guidance_identity", guidance_identity, batch_size)
        guidance_project_rows = getattr(self.guidance, "project_rows", None)
        guidance_batch_size = getattr(self.guidance, "batch_size", None)
        payload_identity = getattr(self.guidance, "guidance_identity", None)
        if (
            callable(guidance_project_rows)
            or guidance_batch_size is not None
            or payload_identity is not None
        ):
            if not callable(guidance_project_rows):
                raise RolloutContractError(
                    "row-batched guidance must expose project_rows()"
                )
            if guidance_batch_size != batch_size:
                raise RolloutContractError(
                    "row-batched guidance batch_size does not match request"
                )
            if payload_identity is None:
                raise RolloutContractError(
                    "row-batched guidance must expose guidance_identity"
                )
            _identities(
                "guidance payload identity",
                payload_identity,
                batch_size,
            )
            if payload_identity != guidance_identity:
                raise RolloutContractError(
                    "guidance payload identity does not match request"
                )

        quantization_identity = self.quantization_identity
        if not quantization_identity:
            quantization_identity = ("none",) * batch_size
            object.__setattr__(
                self,
                "quantization_identity",
                quantization_identity,
            )
        _identities(
            "quantization_identity",
            quantization_identity,
            batch_size,
        )

        if self.conditioner is None:
            if self.conditioner_latent_spec is not None:
                raise RolloutContractError(
                    "conditioner_latent_spec requires a conditioner"
                )
        else:
            if not isinstance(self.conditioner, LatentConditioner):
                raise TypeError("conditioner must be a LatentConditioner")
            if not isinstance(self.conditioner_latent_spec, LatentSpec):
                raise TypeError(
                    "conditioner requires an explicit conditioners.LatentSpec"
                )
            expected = self.conditioner.bind_model_geometry(self.latent_spec)
            if not isinstance(expected, LatentSpec):
                raise TypeError(
                    "conditioner.bind_model_geometry must return conditioners.LatentSpec"
                )
            if self.conditioner_latent_spec != expected:
                raise RolloutContractError(
                    "model and conditioner latent geometry must match exactly"
                )


class RolloutComponent(ABC):
    """Registry-independent trajectory-control runtime boundary."""

    INTERFACE_VERSION = "1.0"

    @property
    @abstractmethod
    def execution_policy(self) -> ExecutionPolicyReceipt:
        """Return the validated core-owned receipt bound at construction."""

        raise NotImplementedError

    @property
    @abstractmethod
    def rollout_execution_policy(self) -> RolloutExecutionPolicy:
        """Return the receipt-derived physical rollout projection."""

        raise NotImplementedError

    @property
    def selection_contract_identity(self) -> str:
        return "visual-rl.rollout-selection.default.v1"

    @classmethod
    @abstractmethod
    def describe(cls, config: object) -> object:
        """Return the exact declared contract implemented by ``config``."""

        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> RolloutComponent:
        if not isinstance(runtime_context, Mapping):
            raise TypeError("runtime_context must be a mapping")
        return cls(config)

    @property
    @abstractmethod
    def num_steps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def run(self, request: RolloutRequest) -> TrajectoryBatch:
        raise NotImplementedError

    @abstractmethod
    def run_with_snapshot(self, request: RolloutRequest) -> RolloutExecution:
        raise NotImplementedError
