"""Per-rollout immutable schedule and transition session.

``Dynamics`` implementations own only immutable equation/configuration state.
Every rollout captures that state into a :class:`ScheduleSnapshot` and uses a
separate :class:`DynamicsSession`; neither object contains a mutable step
cursor.  A snapshot is also the portable replay boundary used to prove that a
recomputed action is scored against the exact schedule used for sampling.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from visual_rl.algorithms.dynamics.interface import (
    DeterministicTransitionOutput,
    Dynamics,
    DynamicsContractError,
    TransitionEvaluation,
    TransitionInput,
    TransitionMeanStd,
    TransitionOutput,
    TransitionRecord,
    TransitionSchedule,
)

_SCHEMA_VERSION = 1


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


def _tensor_payload(value: Any) -> dict[str, object]:
    """Encode an owned CPU tensor without losing dtype bits through JSON."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("schedule tensor payload requires a torch.Tensor")
    owned = value.detach().to(device="cpu").contiguous()
    raw = owned.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(owned.dtype),
        "shape": list(owned.shape),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }


_DTYPES = {
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.int8": "int8",
    "torch.uint8": "uint8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
}


def _tensor_from_payload(payload: object, *, name: str) -> Any:
    import torch

    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a tensor payload mapping")
    if set(payload) != {"dtype", "shape", "data_base64"}:
        raise DynamicsContractError(f"{name} tensor payload has invalid fields")
    dtype_name = payload["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise DynamicsContractError(f"{name} tensor payload has unsupported dtype")
    dtype = getattr(torch, _DTYPES[dtype_name])
    shape = payload["shape"]
    if not isinstance(shape, list) or any(
        type(item) is not int or item < 0 for item in shape
    ):
        raise DynamicsContractError(f"{name} tensor payload has invalid shape")
    encoded = payload["data_base64"]
    if not isinstance(encoded, str):
        raise DynamicsContractError(f"{name} tensor payload data must be base64")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        raise DynamicsContractError(
            f"{name} tensor payload data must be valid base64"
        ) from None
    try:
        flat_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        value = flat_bytes.view(dtype).reshape(tuple(shape))
    except (RuntimeError, ValueError):
        raise DynamicsContractError(
            f"{name} tensor payload byte count does not match its shape"
        ) from None
    return value


@dataclass(frozen=True, slots=True, eq=False)
class DynamicsSelectionState:
    """Portable state of the dedicated policy-step selection RNG stream."""

    generator_device: str
    _state: Any = field(repr=False)
    schema_version: int = _SCHEMA_VERSION
    state_identity: str = field(init=False)

    def __post_init__(self) -> None:
        import torch

        _non_empty("generator_device", self.generator_device)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise DynamicsContractError("selection state schema_version must be 1")
        if not isinstance(self._state, torch.Tensor):
            raise TypeError("selection generator state must be a torch.Tensor")
        if self._state.dtype != torch.uint8 or self._state.ndim != 1:
            raise DynamicsContractError(
                "selection generator state must be a 1-D uint8 tensor"
            )
        if self._state.numel() < 1:
            raise DynamicsContractError("selection generator state must not be empty")
        if self._state.requires_grad or self._state.grad_fn is not None:
            raise DynamicsContractError("selection generator state must be detached")
        owned = self._state.detach().to(device="cpu").contiguous().clone()
        object.__setattr__(self, "_state", owned)
        payload = self._identity_payload()
        object.__setattr__(self, "state_identity", _identity(payload))

    @classmethod
    def from_generator(cls, generator: Any) -> DynamicsSelectionState:
        import torch

        if not isinstance(generator, torch.Generator):
            raise TypeError("selection state requires an explicit torch.Generator")
        return cls(str(generator.device), generator.get_state())

    @property
    def state(self) -> Any:
        return self._state.clone()

    def restore_generator(self) -> Any:
        """Create a new generator at this exact selection boundary."""

        import torch

        try:
            generator = torch.Generator(device=self.generator_device)
            generator.set_state(self._state.clone())
        except (RuntimeError, ValueError):
            raise DynamicsContractError(
                "selection generator state is unavailable on this runtime"
            ) from None
        return generator

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator_device": self.generator_device,
            "state": _tensor_payload(self._state),
        }

    def to_payload(self) -> dict[str, object]:
        payload = self._identity_payload()
        payload["state_identity"] = self.state_identity
        return payload

    def to_checkpoint_payload(self) -> dict[str, object]:
        return self.to_payload()

    @classmethod
    def from_payload(cls, payload: object) -> DynamicsSelectionState:
        if not isinstance(payload, Mapping):
            raise TypeError("selection state payload must be a mapping")
        expected = {
            "schema_version",
            "generator_device",
            "state",
            "state_identity",
        }
        if set(payload) != expected:
            raise DynamicsContractError("selection state payload has invalid fields")
        result = cls(
            generator_device=payload["generator_device"],
            _state=_tensor_from_payload(payload["state"], name="selection state"),
            schema_version=payload["schema_version"],
        )
        if payload["state_identity"] != result.state_identity:
            raise DynamicsContractError("selection state identity mismatch")
        return result

    @classmethod
    def from_checkpoint_payload(cls, payload: object) -> DynamicsSelectionState:
        return cls.from_payload(payload)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DynamicsSelectionState)
            and self.state_identity == other.state_identity
        )

    def __hash__(self) -> int:
        return hash(self.state_identity)


@dataclass(frozen=True, slots=True)
class PolicyStepSelection:
    """Immutable result of one explicit policy-step selection operation."""

    policy: str
    indices: tuple[int, ...]
    randomness_identity: str
    state_before: DynamicsSelectionState
    next_state: DynamicsSelectionState

    def __post_init__(self) -> None:
        _non_empty("selection policy", self.policy)
        _non_empty("selection randomness_identity", self.randomness_identity)
        if type(self.indices) is not tuple or not self.indices:
            raise DynamicsContractError("selection indices must be a non-empty tuple")
        if any(type(item) is not int or item < 0 for item in self.indices):
            raise DynamicsContractError(
                "selection indices must contain non-negative integers"
            )
        if not isinstance(self.state_before, DynamicsSelectionState) or not isinstance(
            self.next_state, DynamicsSelectionState
        ):
            raise TypeError("selection states must be DynamicsSelectionState values")

    @property
    def selection_mapping_identity(self) -> str:
        """Identity of the exact row mapping selected for this session."""

        return self.randomness_identity

    @classmethod
    def all_steps(
        cls,
        *,
        num_steps: int,
        generator: Any,
    ) -> PolicyStepSelection:
        _positive_int("num_steps", num_steps)
        before = DynamicsSelectionState.from_generator(generator)
        after = DynamicsSelectionState.from_generator(generator)
        return cls._create(
            policy="all",
            indices=tuple(range(num_steps)),
            before=before,
            after=after,
        )

    @classmethod
    def fixed(
        cls,
        indices: Sequence[int],
        *,
        num_steps: int,
        generator: Any,
        policy: str,
    ) -> PolicyStepSelection:
        _positive_int("num_steps", num_steps)
        values = _indices(indices, num_steps=num_steps)
        before = DynamicsSelectionState.from_generator(generator)
        after = DynamicsSelectionState.from_generator(generator)
        return cls._create(
            policy=policy,
            indices=values,
            before=before,
            after=after,
        )

    @classmethod
    def uniform(
        cls,
        *,
        num_steps: int,
        cardinality: int,
        generator: Any | None = None,
        selection_state: DynamicsSelectionState | None = None,
        shared: bool = False,
        include_final: bool = True,
        policy: str = "uniform",
    ) -> PolicyStepSelection:
        """Draw indices and return the exact state needed by the next draw.

        Passing ``selection_state`` restores a dedicated selection stream and
        is the checkpoint/resume path.  Passing ``generator`` directly keeps
        backwards-compatible callers explicit while still recording both RNG
        boundaries.
        """

        import torch

        _positive_int("num_steps", num_steps)
        _positive_int("selection cardinality", cardinality)
        if type(shared) is not bool or type(include_final) is not bool:
            raise TypeError("shared/include_final must be bool")
        if generator is not None and selection_state is not None:
            raise ValueError("provide generator or selection_state, not both")
        if selection_state is not None:
            if not isinstance(selection_state, DynamicsSelectionState):
                raise TypeError("selection_state must be DynamicsSelectionState")
            generator = selection_state.restore_generator()
        if not isinstance(generator, torch.Generator):
            raise TypeError("uniform selection requires an explicit generator state")
        upper = num_steps if include_final else num_steps - 1
        if upper < 1:
            raise DynamicsContractError(
                "uniform selection range must contain at least one step"
            )
        before = DynamicsSelectionState.from_generator(generator)
        draw_count = 1 if shared else cardinality
        values = torch.randint(
            0,
            upper,
            (draw_count,),
            generator=generator,
            device=generator.device,
            dtype=torch.int64,
        ).to(device="cpu")
        indices = tuple(int(item) for item in values.tolist())
        if shared:
            indices = (indices[0],) * cardinality
        after = DynamicsSelectionState.from_generator(generator)
        return cls._create(
            policy=policy,
            indices=indices,
            before=before,
            after=after,
        )

    @classmethod
    def uniform_from_candidates_by_key(
        cls,
        *,
        num_steps: int,
        candidate_indices: Sequence[int],
        keys: Sequence[str],
        generator: Any | None = None,
        selection_state: DynamicsSelectionState | None = None,
        policy: str,
    ) -> PolicyStepSelection:
        """Uniformly map each unique key to one candidate transition.

        Keys are consumed in first-occurrence order.  Equal keys therefore
        share one draw (the Flash K-repeat rule) while the exact row mapping
        and RNG boundary remain content-addressed by ``randomness_identity``.
        Raw prompt text is never placed in that identity payload.
        """

        import torch

        _positive_int("num_steps", num_steps)
        candidates = _indices(candidate_indices, num_steps=num_steps)
        if len(candidates) != len(set(candidates)):
            raise DynamicsContractError("selection candidates must be unique")
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
            raise TypeError("selection keys must be a sequence")
        row_keys = tuple(keys)
        if not row_keys:
            raise DynamicsContractError("selection keys must not be empty")
        if any(not isinstance(item, str) or not item for item in row_keys):
            raise DynamicsContractError("selection keys must contain non-empty strings")
        if generator is not None and selection_state is not None:
            raise ValueError("provide generator or selection_state, not both")
        if selection_state is not None:
            if not isinstance(selection_state, DynamicsSelectionState):
                raise TypeError("selection_state must be DynamicsSelectionState")
            generator = selection_state.restore_generator()
        if not isinstance(generator, torch.Generator):
            raise TypeError("uniform selection requires an explicit generator state")

        unique_keys = tuple(dict.fromkeys(row_keys))
        before = DynamicsSelectionState.from_generator(generator)
        candidate_positions = torch.randint(
            0,
            len(candidates),
            (len(unique_keys),),
            generator=generator,
            device=generator.device,
            dtype=torch.int64,
        ).to(device="cpu")
        step_by_key = {
            key: candidates[int(position)]
            for key, position in zip(
                unique_keys,
                candidate_positions.tolist(),
                strict=True,
            )
        }
        indices = tuple(step_by_key[key] for key in row_keys)
        after = DynamicsSelectionState.from_generator(generator)
        key_identity_by_row = tuple(
            _identity({"schema_version": 1, "selection_key": key}) for key in row_keys
        )
        return cls._create(
            policy=policy,
            indices=indices,
            before=before,
            after=after,
            mapping_payload={
                "candidate_indices": list(candidates),
                "key_identity_by_row": list(key_identity_by_row),
            },
        )

    @classmethod
    def _create(
        cls,
        *,
        policy: str,
        indices: tuple[int, ...],
        before: DynamicsSelectionState,
        after: DynamicsSelectionState,
        mapping_payload: Mapping[str, object] | None = None,
    ) -> PolicyStepSelection:
        identity_payload: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "policy": policy,
            "indices": list(indices),
            "state_before": before.state_identity,
            "next_state": after.state_identity,
        }
        if mapping_payload is not None:
            identity_payload["mapping"] = dict(mapping_payload)
        randomness_identity = _identity(identity_payload)
        return cls(
            policy=policy,
            indices=indices,
            randomness_identity=randomness_identity,
            state_before=before,
            next_state=after,
        )


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _indices(values: Sequence[int], *, num_steps: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("selection indices must be a sequence")
    result = tuple(values)
    if not result:
        raise DynamicsContractError("selection indices must not be empty")
    if any(type(item) is not int or not 0 <= item < num_steps for item in result):
        raise DynamicsContractError("selection index is outside the schedule")
    return result


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ScheduleSnapshot:
    """Owned, JSON-serializable schedule and policy-step selection snapshot."""

    _timesteps: Any = field(repr=False)
    _next_timesteps: Any = field(repr=False)
    _sigmas: Any | None = field(repr=False)
    _dt: Any = field(repr=False)
    dynamics_config_identity: str
    scheduler_identity: str
    selection_policy: str
    selected_policy_step_indices: tuple[int, ...]
    randomness_identity: str
    next_selection_state: DynamicsSelectionState
    schema_version: int
    schedule_identity: str
    snapshot_identity: str

    def __init__(
        self,
        timesteps: Any,
        next_timesteps: Any,
        *,
        sigmas: Any | None,
        dt: Any,
        dynamics_config_identity: str,
        scheduler_identity: str,
        selection_policy: str,
        selected_policy_step_indices: Sequence[int],
        randomness_identity: str,
        next_selection_state: DynamicsSelectionState,
        schema_version: int = _SCHEMA_VERSION,
    ) -> None:
        import torch

        if type(schema_version) is not int or schema_version != 1:
            raise DynamicsContractError("schedule snapshot schema_version must be 1")
        config_identity = _non_empty(
            "dynamics_config_identity", dynamics_config_identity
        )
        scheduler = _non_empty("scheduler_identity", scheduler_identity)
        policy = _non_empty("selection_policy", selection_policy)
        randomness = _non_empty("randomness_identity", randomness_identity)
        if not isinstance(next_selection_state, DynamicsSelectionState):
            raise TypeError("next_selection_state must be a DynamicsSelectionState")
        owned_timesteps = _owned_schedule_vector("timesteps", timesteps)
        owned_next = _owned_schedule_vector("next_timesteps", next_timesteps)
        if (
            owned_timesteps.shape != owned_next.shape
            or owned_timesteps.dtype != owned_next.dtype
        ):
            raise DynamicsContractError(
                "snapshot timesteps and next_timesteps must share shape and dtype"
            )
        if owned_timesteps.numel() > 1 and not torch.equal(
            owned_next[:-1], owned_timesteps[1:]
        ):
            raise DynamicsContractError(
                "snapshot next timesteps must follow the ordered schedule"
            )
        owned_dt = _owned_schedule_vector("dt", dt, floating=True)
        if owned_dt.shape != owned_timesteps.shape:
            raise DynamicsContractError("snapshot dt must contain one value per step")
        if bool((owned_dt == 0).any()):
            raise DynamicsContractError("snapshot dt values must be non-zero")
        owned_sigmas = None
        if sigmas is not None:
            owned_sigmas = _owned_schedule_vector("sigmas", sigmas, floating=True)
            if owned_sigmas.numel() != owned_timesteps.numel() + 1:
                raise DynamicsContractError(
                    "snapshot sigmas must contain num_steps + 1 values"
                )
            expected_dt = owned_sigmas[1:] - owned_sigmas[:-1]
            if not torch.equal(owned_dt.to(dtype=expected_dt.dtype), expected_dt):
                raise DynamicsContractError("snapshot dt does not match sigma pairs")
        selected = _indices(
            selected_policy_step_indices,
            num_steps=int(owned_timesteps.numel()),
        )

        object.__setattr__(self, "_timesteps", owned_timesteps)
        object.__setattr__(self, "_next_timesteps", owned_next)
        object.__setattr__(self, "_sigmas", owned_sigmas)
        object.__setattr__(self, "_dt", owned_dt)
        object.__setattr__(self, "dynamics_config_identity", config_identity)
        object.__setattr__(self, "scheduler_identity", scheduler)
        object.__setattr__(self, "selection_policy", policy)
        object.__setattr__(self, "selected_policy_step_indices", selected)
        object.__setattr__(self, "randomness_identity", randomness)
        object.__setattr__(self, "next_selection_state", next_selection_state)
        object.__setattr__(self, "schema_version", schema_version)
        schedule_payload = self._schedule_payload()
        object.__setattr__(self, "schedule_identity", _identity(schedule_payload))
        object.__setattr__(
            self, "snapshot_identity", _identity(self._identity_payload())
        )

    @property
    def num_steps(self) -> int:
        return int(self._timesteps.numel())

    @property
    def timesteps(self) -> Any:
        return self._timesteps.clone()

    @property
    def next_timesteps(self) -> Any:
        return self._next_timesteps.clone()

    @property
    def sigmas(self) -> Any | None:
        return None if self._sigmas is None else self._sigmas.clone()

    @property
    def dt(self) -> Any:
        return self._dt.clone()

    def transition_schedule(self, *, device: Any) -> TransitionSchedule:
        return TransitionSchedule(
            timesteps=self._timesteps.to(device=device).clone(),
            next_timesteps=self._next_timesteps.to(device=device).clone(),
        )

    def _schedule_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dynamics_config_identity": self.dynamics_config_identity,
            "scheduler_identity": self.scheduler_identity,
            "timesteps": _tensor_payload(self._timesteps),
            "next_timesteps": _tensor_payload(self._next_timesteps),
            "sigmas": (None if self._sigmas is None else _tensor_payload(self._sigmas)),
            "dt": _tensor_payload(self._dt),
        }

    def _identity_payload(self) -> dict[str, object]:
        return {
            **self._schedule_payload(),
            "selection_policy": self.selection_policy,
            "selected_policy_step_indices": list(self.selected_policy_step_indices),
            "randomness_identity": self.randomness_identity,
            "next_selection_state_identity": (self.next_selection_state.state_identity),
        }

    def to_payload(self) -> dict[str, object]:
        payload = self._identity_payload()
        payload["next_selection_state"] = self.next_selection_state.to_payload()
        payload["schedule_identity"] = self.schedule_identity
        payload["snapshot_identity"] = self.snapshot_identity
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> ScheduleSnapshot:
        if not isinstance(payload, Mapping):
            raise TypeError("schedule snapshot payload must be a mapping")
        expected = {
            "schema_version",
            "dynamics_config_identity",
            "scheduler_identity",
            "timesteps",
            "next_timesteps",
            "sigmas",
            "dt",
            "selection_policy",
            "selected_policy_step_indices",
            "randomness_identity",
            "next_selection_state_identity",
            "next_selection_state",
            "schedule_identity",
            "snapshot_identity",
        }
        if set(payload) != expected:
            raise DynamicsContractError("schedule snapshot payload has invalid fields")
        sigmas_payload = payload["sigmas"]
        sigmas = (
            None
            if sigmas_payload is None
            else _tensor_from_payload(sigmas_payload, name="snapshot sigmas")
        )
        next_state = DynamicsSelectionState.from_payload(
            payload["next_selection_state"]
        )
        if payload["next_selection_state_identity"] != next_state.state_identity:
            raise DynamicsContractError(
                "snapshot next selection state identity mismatch"
            )
        result = cls(
            _tensor_from_payload(payload["timesteps"], name="snapshot timesteps"),
            _tensor_from_payload(
                payload["next_timesteps"], name="snapshot next_timesteps"
            ),
            sigmas=sigmas,
            dt=_tensor_from_payload(payload["dt"], name="snapshot dt"),
            dynamics_config_identity=payload["dynamics_config_identity"],
            scheduler_identity=payload["scheduler_identity"],
            selection_policy=payload["selection_policy"],
            selected_policy_step_indices=payload["selected_policy_step_indices"],
            randomness_identity=payload["randomness_identity"],
            next_selection_state=next_state,
            schema_version=payload["schema_version"],
        )
        if payload["schedule_identity"] != result.schedule_identity:
            raise DynamicsContractError("schedule snapshot schedule identity mismatch")
        if payload["snapshot_identity"] != result.snapshot_identity:
            raise DynamicsContractError("schedule snapshot identity mismatch")
        return result

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ScheduleSnapshot)
            and self.snapshot_identity == other.snapshot_identity
        )

    def __hash__(self) -> int:
        return hash(self.snapshot_identity)


def _owned_schedule_vector(name: str, value: Any, *, floating: bool = False) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"snapshot {name} must be a torch.Tensor")
    if value.ndim != 1 or value.numel() < 1:
        raise DynamicsContractError(f"snapshot {name} must be non-empty 1-D")
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError(f"snapshot {name} must use a real numeric dtype")
    if floating and not value.is_floating_point():
        raise TypeError(f"snapshot {name} must be floating point")
    if value.requires_grad or value.grad_fn is not None:
        raise DynamicsContractError(f"snapshot {name} must be detached")
    if not bool(torch.isfinite(value).all()):
        raise DynamicsContractError(f"snapshot {name} must be finite")
    return value.detach().to(device="cpu").contiguous().clone()


@dataclass(frozen=True, slots=True, eq=False)
class DynamicsSession:
    """Cursor-free binding of one Dynamics equation to one rollout snapshot."""

    dynamics: Dynamics
    snapshot: ScheduleSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.dynamics, Dynamics):
            raise TypeError("session dynamics must be a Dynamics")
        if not isinstance(self.snapshot, ScheduleSnapshot):
            raise TypeError("session snapshot must be a ScheduleSnapshot")
        current = self._capture_schedule(selection=None)
        if current.schedule_identity != self.snapshot.schedule_identity:
            raise DynamicsContractError(
                "schedule snapshot does not match the bound Dynamics configuration"
            )

    @classmethod
    def create(
        cls,
        dynamics: Dynamics,
        *,
        num_steps: int,
        device: Any,
        selection: PolicyStepSelection,
    ) -> DynamicsSession:
        if not isinstance(dynamics, Dynamics):
            raise TypeError("session dynamics must be a Dynamics")
        if not isinstance(selection, PolicyStepSelection):
            raise TypeError("selection must be a PolicyStepSelection")
        schedule = dynamics.transition_schedule(num_steps=num_steps, device=device)
        sigmas = dynamics.schedule_sigmas(num_steps=num_steps, device=device)
        dt = (
            schedule.next_timesteps - schedule.timesteps
            if sigmas is None
            else sigmas[1:] - sigmas[:-1]
        )
        snapshot = ScheduleSnapshot(
            schedule.timesteps,
            schedule.next_timesteps,
            sigmas=sigmas,
            dt=dt,
            dynamics_config_identity=dynamics.dynamics_config_identity,
            scheduler_identity=dynamics.scheduler_identity,
            selection_policy=selection.policy,
            selected_policy_step_indices=selection.indices,
            randomness_identity=selection.randomness_identity,
            next_selection_state=selection.next_state,
        )
        return cls(dynamics=dynamics, snapshot=snapshot)

    @classmethod
    def from_snapshot(
        cls,
        dynamics: Dynamics,
        snapshot: ScheduleSnapshot,
    ) -> DynamicsSession:
        return cls(dynamics=dynamics, snapshot=snapshot)

    def _capture_schedule(
        self,
        *,
        selection: PolicyStepSelection | None,
    ) -> ScheduleSnapshot:
        import torch

        schedule = self.dynamics.transition_schedule(
            num_steps=self.snapshot.num_steps,
            device="cpu",
        )
        sigmas = self.dynamics.schedule_sigmas(
            num_steps=self.snapshot.num_steps,
            device="cpu",
        )
        dt = (
            schedule.next_timesteps - schedule.timesteps
            if sigmas is None
            else sigmas[1:] - sigmas[:-1]
        )
        if selection is None:
            generator = torch.Generator(device="cpu").manual_seed(0)
            selection = PolicyStepSelection.fixed(
                self.snapshot.selected_policy_step_indices,
                num_steps=self.snapshot.num_steps,
                generator=generator,
                policy=self.snapshot.selection_policy,
            )
        return ScheduleSnapshot(
            schedule.timesteps,
            schedule.next_timesteps,
            sigmas=sigmas,
            dt=dt,
            dynamics_config_identity=self.dynamics.dynamics_config_identity,
            scheduler_identity=self.dynamics.scheduler_identity,
            selection_policy=selection.policy,
            selected_policy_step_indices=selection.indices,
            randomness_identity=selection.randomness_identity,
            next_selection_state=selection.next_state,
        )

    @property
    def schedule(self) -> TransitionSchedule:
        return self.snapshot.transition_schedule(device="cpu")

    def transition_schedule(self, *, device: Any) -> TransitionSchedule:
        return self.snapshot.transition_schedule(device=device)

    def _validate_transition(self, transition: TransitionInput) -> None:
        import torch

        transition.validate()
        indices = transition.transition_index
        if bool((indices >= self.snapshot.num_steps).any()):
            raise DynamicsContractError(
                "transition_index is outside the session snapshot"
            )
        timesteps = self.snapshot._timesteps.to(device=transition.x_t.device)
        next_timesteps = self.snapshot._next_timesteps.to(device=transition.x_t.device)
        if transition.t.dtype != timesteps.dtype or transition.t_next.dtype != (
            next_timesteps.dtype
        ):
            raise DynamicsContractError(
                "transition timestep dtype does not match the session snapshot"
            )
        if not torch.equal(transition.t, timesteps.index_select(0, indices)):
            raise DynamicsContractError("transition t does not match session snapshot")
        if not torch.equal(
            transition.t_next,
            next_timesteps.index_select(0, indices),
        ):
            raise DynamicsContractError(
                "transition t_next does not match session snapshot"
            )

    def _validate_dt(self, transition: TransitionInput, dt: Any) -> None:
        import torch

        expected = self.snapshot._dt.to(
            device=transition.x_t.device,
            dtype=dt.dtype,
        ).index_select(0, transition.transition_index)
        if not torch.equal(dt, expected):
            raise DynamicsContractError("transition dt does not match session snapshot")

    def transition_mean_std(
        self,
        transition: TransitionInput,
    ) -> TransitionMeanStd:
        self._validate_transition(transition)
        result = self.dynamics.transition_mean_std(transition)
        if not isinstance(result, TransitionMeanStd):
            raise TypeError("transition_mean_std() must return TransitionMeanStd")
        result.validate_against(transition)
        self._validate_dt(transition, result.dt)
        return result

    def deterministic_ode_step(
        self,
        transition: TransitionInput,
    ) -> DeterministicTransitionOutput:
        """Execute the bound Dynamics ODE port against this frozen schedule."""

        self._validate_transition(transition)
        result = self.dynamics.deterministic_ode_step(transition)
        if not isinstance(result, DeterministicTransitionOutput):
            raise TypeError(
                "deterministic_ode_step() must return DeterministicTransitionOutput"
            )
        result.validate_against(transition)
        self._validate_dt(transition, result.dt)
        return result

    def policy_metadata(
        self,
        transition: TransitionInput,
        stats: TransitionMeanStd,
    ) -> object:
        """Return credit metadata after validating it against this snapshot."""

        from visual_rl.algorithms.dynamics.interface import TransitionPolicyMetadata

        self._validate_transition(transition)
        stats.validate_against(transition)
        self._validate_dt(transition, stats.dt)
        result = self.dynamics.policy_metadata(transition, stats)
        if not isinstance(result, TransitionPolicyMetadata):
            raise TypeError("policy_metadata() must return TransitionPolicyMetadata")
        result.validate_against(transition)
        return result

    def sample_transition(
        self,
        transition: TransitionInput,
        *,
        generator: Any,
    ) -> TransitionOutput:
        self._validate_transition(transition)
        result = self.dynamics.sample_transition(transition, generator=generator)
        self._validate_dt(transition, result.dt)
        return result

    def transition_log_prob(
        self,
        transition: TransitionInput,
        action_latent: Any,
    ) -> Any:
        self._validate_transition(transition)
        return self.dynamics.transition_log_prob(transition, action_latent)

    def evaluate_transition(
        self,
        transition: TransitionInput,
        action_latent: Any,
    ) -> TransitionEvaluation:
        """Evaluate a stored action once and enforce the frozen schedule dt."""

        self._validate_transition(transition)
        result = self.dynamics.evaluate_transition(transition, action_latent)
        if not isinstance(result, TransitionEvaluation):
            raise TypeError("evaluate_transition() must return TransitionEvaluation")
        result.validate_against(transition)
        self._validate_dt(transition, result.stats.dt)
        return result

    def recompute_log_prob(
        self,
        record: TransitionRecord,
        model_prediction: Any,
    ) -> Any:
        """Score a stored action using this snapshot, never live scheduler state."""

        if not isinstance(record, TransitionRecord):
            raise TypeError("record must be a TransitionRecord")
        transition = TransitionInput(
            x_t=record.x_t,
            model_prediction=model_prediction,
            t=record.t,
            t_next=record.t_next,
            mask=record.mask,
            transition_index=record.transition_index,
            condition_identity=record.condition_identity,
            guidance_identity=record.guidance_identity,
            storage_dtype_identity=record.storage_dtype_identity,
            quantization_identity=record.quantization_identity,
        )
        return self.transition_log_prob(transition, record.scoring_target)

    def make_record(
        self,
        transition: TransitionInput,
        output: TransitionOutput,
        *,
        conditioned_next: Any,
        likelihood_semantics: Any,
    ) -> TransitionRecord:
        self._validate_transition(transition)
        self._validate_dt(transition, output.dt)
        return self.dynamics.make_record(
            transition,
            output,
            conditioned_next=conditioned_next,
            likelihood_semantics=likelihood_semantics,
        )

    def add_noise(self, clean: Any, noise: Any, timestep: Any) -> Any:
        import torch

        if not isinstance(timestep, torch.Tensor):
            raise TypeError("timestep must be a torch.Tensor")
        values = timestep.reshape(-1).to(device="cpu")
        candidates = self.snapshot._timesteps
        if not bool(((values[:, None] == candidates[None, :]).any(dim=1)).all()):
            raise DynamicsContractError("timestep is absent from session snapshot")
        return self.dynamics.add_noise(clean, noise, timestep)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DynamicsSession)
            and self.dynamics is other.dynamics
            and self.snapshot == other.snapshot
        )

    def __hash__(self) -> int:
        return hash((id(self.dynamics), self.snapshot))


__all__ = [
    "DynamicsSelectionState",
    "DynamicsSession",
    "PolicyStepSelection",
    "ScheduleSnapshot",
]
