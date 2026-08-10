"""Typed construction boundary for per-rollout scheduler replay state.

The model loader resolves and freezes a scheduler *blueprint* once.  A
composition root then supplies a :class:`DynamicsReplayRequest` for each
rollout.  Materialization always creates a fresh scheduler instance and
immediately converts it into an immutable replay state; no live pipeline or
mutable scheduler cursor crosses this boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, runtime_checkable

from visual_rl.algorithms.dynamics.interface import (
    Dynamics,
    DynamicsComponent,
    DynamicsContractError,
)
from visual_rl.models.scheduler import ModelScheduleContext, SchedulerArtifactBlueprint

__all__ = (
    "DynamicsInstanceFactory",
    "DynamicsReplayBinding",
    "DynamicsReplayRequest",
    "DynamicsReplayStateFactory",
    "FlowMatchDynamicShiftConfig",
    "FlowMatchScheduleConditioning",
    "SchedulerDynamicsBinder",
    "materialize_scheduler_from_blueprint",
    "scheduler_dynamic_shift_config",
)

_STATE = TypeVar("_STATE")
_STATE_CO_co = TypeVar("_STATE_CO_co", covariant=True)


def _identity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise DynamicsContractError(f"{name} must be a non-empty string")
    return value


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise DynamicsContractError(f"{name} must be a positive integer")
    return value


def _finite_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise DynamicsContractError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True, eq=False)
class FlowMatchDynamicShiftConfig:
    """Typed resolution-to-``mu`` policy used by flow-match schedulers.

    The interpolation is the one used by Diffusers' SD3 pipeline and by the
    Flow-Factory scheduler helper.  Keeping the four scheduler values here
    makes the derived ``mu`` reproducible without retaining a mutable
    scheduler or reading its config at rollout time.
    """

    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15
    policy: str = "linear_sequence_length_interpolation.v1"
    schema_version: int = 1
    config_identity: str = field(init=False)

    def __post_init__(self) -> None:
        base_seq_len = _positive_int("base_image_seq_len", self.base_image_seq_len)
        max_seq_len = _positive_int("max_image_seq_len", self.max_image_seq_len)
        if max_seq_len <= base_seq_len:
            raise DynamicsContractError(
                "max_image_seq_len must be greater than base_image_seq_len"
            )
        base_shift = _finite_float("base_shift", self.base_shift)
        max_shift = _finite_float("max_shift", self.max_shift)
        if base_shift <= 0 or max_shift <= 0:
            raise DynamicsContractError(
                "dynamic shift endpoints must be strictly positive"
            )
        if self.policy != "linear_sequence_length_interpolation.v1":
            raise DynamicsContractError("unsupported dynamic shift policy")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise DynamicsContractError("dynamic shift config schema_version must be 1")
        object.__setattr__(self, "base_image_seq_len", base_seq_len)
        object.__setattr__(self, "max_image_seq_len", max_seq_len)
        object.__setattr__(self, "base_shift", base_shift)
        object.__setattr__(self, "max_shift", max_shift)
        object.__setattr__(self, "config_identity", _identity(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "base_image_seq_len": self.base_image_seq_len,
            "max_image_seq_len": self.max_image_seq_len,
            "base_shift": self.base_shift,
            "max_shift": self.max_shift,
        }

    def calculate_mu(self, image_seq_len: int) -> float:
        seq_len = _positive_int("image_seq_len", image_seq_len)
        slope = (self.max_shift - self.base_shift) / (
            self.max_image_seq_len - self.base_image_seq_len
        )
        intercept = self.base_shift - slope * self.base_image_seq_len
        return seq_len * slope + intercept

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FlowMatchDynamicShiftConfig)
            and self.config_identity == other.config_identity
        )

    def __hash__(self) -> int:
        return hash(self.config_identity)


@dataclass(frozen=True, slots=True, eq=False)
class FlowMatchScheduleConditioning:
    """Latent geometry and optional dynamic-shift policy for one schedule."""

    latent_height: int
    latent_width: int
    patch_size: int
    image_seq_len: int
    dynamic_shift: FlowMatchDynamicShiftConfig | None
    schema_version: int = 1
    conditioning_identity: str = field(init=False)

    def __post_init__(self) -> None:
        latent_height = _positive_int("latent_height", self.latent_height)
        latent_width = _positive_int("latent_width", self.latent_width)
        patch_size = _positive_int("patch_size", self.patch_size)
        image_seq_len = _positive_int("image_seq_len", self.image_seq_len)
        if latent_height % patch_size or latent_width % patch_size:
            raise DynamicsContractError(
                "latent height and width must be divisible by patch_size"
            )
        expected_seq_len = (latent_height // patch_size) * (latent_width // patch_size)
        if image_seq_len != expected_seq_len:
            raise DynamicsContractError(
                "image_seq_len does not match latent geometry and patch_size"
            )
        if self.dynamic_shift is not None and not isinstance(
            self.dynamic_shift, FlowMatchDynamicShiftConfig
        ):
            raise TypeError("dynamic_shift must be FlowMatchDynamicShiftConfig or None")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise DynamicsContractError(
                "schedule conditioning schema_version must be 1"
            )
        object.__setattr__(self, "latent_height", latent_height)
        object.__setattr__(self, "latent_width", latent_width)
        object.__setattr__(self, "patch_size", patch_size)
        object.__setattr__(self, "image_seq_len", image_seq_len)
        object.__setattr__(
            self,
            "conditioning_identity",
            _identity(self.to_payload()),
        )

    @classmethod
    def from_latent_geometry(
        cls,
        *,
        latent_height: int,
        latent_width: int,
        patch_size: int,
        dynamic_shift: FlowMatchDynamicShiftConfig | None,
    ) -> FlowMatchScheduleConditioning:
        height = _positive_int("latent_height", latent_height)
        width = _positive_int("latent_width", latent_width)
        patch = _positive_int("patch_size", patch_size)
        if height % patch or width % patch:
            raise DynamicsContractError(
                "latent height and width must be divisible by patch_size"
            )
        return cls(
            latent_height=height,
            latent_width=width,
            patch_size=patch,
            image_seq_len=(height // patch) * (width // patch),
            dynamic_shift=dynamic_shift,
        )

    @property
    def mu(self) -> float | None:
        if self.dynamic_shift is None:
            return None
        return self.dynamic_shift.calculate_mu(self.image_seq_len)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "patch_size": self.patch_size,
            "image_seq_len": self.image_seq_len,
            "dynamic_shift": (
                None if self.dynamic_shift is None else self.dynamic_shift.to_payload()
            ),
        }

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FlowMatchScheduleConditioning)
            and self.conditioning_identity == other.conditioning_identity
        )

    def __hash__(self) -> int:
        return hash(self.conditioning_identity)


@dataclass(frozen=True, slots=True, eq=False)
class DynamicsReplayRequest:
    """Semantic identity and schedule length for exactly one rollout."""

    rollout_identity: str
    num_steps: int
    schedule_conditioning: FlowMatchScheduleConditioning | None = None
    schema_version: int = 2
    request_identity: str = field(init=False)

    def __post_init__(self) -> None:
        rollout_identity = _non_empty("rollout_identity", self.rollout_identity)
        if type(self.num_steps) is not int or self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if self.schedule_conditioning is not None and not isinstance(
            self.schedule_conditioning, FlowMatchScheduleConditioning
        ):
            raise TypeError(
                "schedule_conditioning must be FlowMatchScheduleConditioning or None"
            )
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise DynamicsContractError("replay request schema_version must be 2")
        object.__setattr__(
            self,
            "request_identity",
            _identity(
                {
                    "schema_version": self.schema_version,
                    "rollout_identity": rollout_identity,
                    "num_steps": self.num_steps,
                    "schedule_conditioning": (
                        None
                        if self.schedule_conditioning is None
                        else self.schedule_conditioning.to_payload()
                    ),
                }
            ),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DynamicsReplayRequest)
            and self.request_identity == other.request_identity
        )

    def __hash__(self) -> int:
        return hash(self.request_identity)


@dataclass(frozen=True, slots=True, eq=False)
class DynamicsReplayBinding(Generic[_STATE]):
    """An immutable replay state bound to its factory and rollout request."""

    request: DynamicsReplayRequest
    factory_identity: str
    replay_state: _STATE
    binding_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, DynamicsReplayRequest):
            raise TypeError("request must be a DynamicsReplayRequest")
        factory_identity = _non_empty("factory_identity", self.factory_identity)
        state_steps = getattr(self.replay_state, "num_steps", None)
        if state_steps != self.request.num_steps:
            raise DynamicsContractError(
                "replay state step count does not match its rollout request"
            )
        replay_identity = _non_empty(
            "replay_state_identity",
            getattr(self.replay_state, "replay_state_identity", None),
        )
        _non_empty(
            "scheduler_identity",
            getattr(self.replay_state, "scheduler_identity", None),
        )
        object.__setattr__(
            self,
            "binding_identity",
            _identity(
                {
                    "schema_version": 1,
                    "request_identity": self.request.request_identity,
                    "factory_identity": factory_identity,
                    "replay_state_identity": replay_identity,
                }
            ),
        )

    @property
    def replay_state_identity(self) -> str:
        return self.replay_state.replay_state_identity  # type: ignore[attr-defined]

    @property
    def scheduler_identity(self) -> str:
        return self.replay_state.scheduler_identity  # type: ignore[attr-defined]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DynamicsReplayBinding)
            and self.binding_identity == other.binding_identity
        )

    def __hash__(self) -> int:
        return hash(self.binding_identity)


class DynamicsInstanceFactory(DynamicsComponent, Generic[_STATE]):
    """Run-level component that creates a fresh Dynamics for one binding."""

    @property
    @abstractmethod
    def replay_state_type(self) -> type[_STATE]:
        """The exact immutable replay-state type accepted by this factory."""

        raise NotImplementedError

    @abstractmethod
    def create(self, binding: DynamicsReplayBinding[_STATE]) -> Dynamics:
        """Bind one iteration state to a new transition kernel instance."""

        raise NotImplementedError


@runtime_checkable
class DynamicsReplayStateFactory(Protocol[_STATE_CO_co]):
    """Dynamics-owned port that materializes one owned state per rollout."""

    @property
    def factory_identity(self) -> str: ...

    @property
    def replay_state_type(self) -> type[_STATE_CO_co]: ...

    def create(
        self,
        request: DynamicsReplayRequest,
    ) -> DynamicsReplayBinding[_STATE_CO_co]: ...


@runtime_checkable
class SchedulerDynamicsBinder(Protocol):
    """Narrow typed bridge from a model scheduler artifact into Dynamics.

    Replay-state construction is bound once when the production graph is
    assembled.  Geometry-dependent schedule conditioning is derived later,
    once the model has produced the concrete latent context for an iteration.
    """

    @property
    def dynamics_binding_family(self) -> str: ...

    @property
    def replay_state_schema_id(self) -> str: ...

    def bind_replay_state_factory(
        self,
        blueprint: SchedulerArtifactBlueprint,
    ) -> DynamicsReplayStateFactory[object]: ...

    def schedule_conditioning(
        self,
        blueprint: SchedulerArtifactBlueprint,
        context: ModelScheduleContext,
    ) -> FlowMatchScheduleConditioning | None: ...


def scheduler_dynamic_shift_config(
    blueprint: SchedulerArtifactBlueprint,
) -> FlowMatchDynamicShiftConfig | None:
    """Interpret flow-match dynamic-shift semantics inside Dynamics."""

    if not isinstance(blueprint, SchedulerArtifactBlueprint):
        raise TypeError("blueprint must be a SchedulerArtifactBlueprint")
    config = blueprint.config_payload()
    enabled = config.get("use_dynamic_shifting", False)
    if type(enabled) is not bool:
        raise DynamicsContractError(
            "scheduler.config.use_dynamic_shifting must be bool"
        )
    if not enabled:
        return None
    return FlowMatchDynamicShiftConfig(
        base_image_seq_len=config.get("base_image_seq_len", 256),
        max_image_seq_len=config.get("max_image_seq_len", 4096),
        base_shift=config.get("base_shift", 0.5),
        max_shift=config.get("max_shift", 1.15),
    )


def materialize_scheduler_from_blueprint(
    blueprint: SchedulerArtifactBlueprint,
    request: DynamicsReplayRequest,
) -> object:
    """Bind an algorithm replay request to one fresh scheduler cursor."""

    if not isinstance(blueprint, SchedulerArtifactBlueprint):
        raise TypeError("blueprint must be a SchedulerArtifactBlueprint")
    if not isinstance(request, DynamicsReplayRequest):
        raise TypeError("request must be a DynamicsReplayRequest")
    scheduler = blueprint.instantiate_scheduler()
    set_timesteps = scheduler.set_timesteps
    dynamic_shift = scheduler_dynamic_shift_config(blueprint)
    conditioning = request.schedule_conditioning
    if dynamic_shift is None:
        if conditioning is not None and conditioning.dynamic_shift is not None:
            raise DynamicsContractError(
                "request provides dynamic shifting for a scheduler that does "
                "not declare that capability"
            )
        set_timesteps(num_inference_steps=request.num_steps, device="cpu")
        return scheduler
    if conditioning is None:
        raise DynamicsContractError(
            "dynamic-shift scheduler requires latent schedule conditioning"
        )
    if conditioning.dynamic_shift != dynamic_shift:
        raise DynamicsContractError(
            "request dynamic-shift config does not match the scheduler blueprint"
        )
    mu = conditioning.mu
    assert mu is not None
    try:
        set_timesteps(
            num_inference_steps=request.num_steps,
            device="cpu",
            mu=mu,
        )
    except TypeError as exc:
        raise DynamicsContractError(
            "dynamic-shift scheduler must accept a typed mu parameter"
        ) from exc
    return scheduler
