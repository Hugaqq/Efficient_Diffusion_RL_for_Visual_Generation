"""Typed rank-local component, RNG, and Dynamics checkpoint state."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visual_rl.algorithms.dynamics.selection import DynamicsSelectionPolicyState
from visual_rl.artifacts.checkpoint.coordination import (
    CheckpointSafePoint,
    _rank_and_world_size,
)

_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True, eq=False)
class RankRNGSnapshot:
    """Portable Python/NumPy/Torch RNG state for one rank."""

    rank: int
    python_state: tuple[Any, ...] = field(repr=False)
    numpy_bit_generator: str
    numpy_state: tuple[int, ...] = field(repr=False)
    numpy_position: int
    numpy_has_gauss: int
    numpy_cached_gaussian: float
    torch_cpu: Any = field(repr=False)
    torch_cuda: Any | None = field(repr=False)
    state_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("RNG rank must be a non-negative integer")
        if type(self.python_state) is not tuple:
            raise TypeError("Python RNG state must be a tuple")
        try:
            probe = random.Random()
            probe.setstate(self.python_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("Python RNG state is invalid") from exc
        if self.numpy_bit_generator != "MT19937":
            raise ValueError("NumPy RNG bit generator must be MT19937")
        if (
            type(self.numpy_state) is not tuple
            or len(self.numpy_state) != 624
            or any(
                type(item) is not int or not 0 <= item <= 0xFFFF_FFFF
                for item in self.numpy_state
            )
        ):
            raise ValueError("NumPy RNG state must contain 624 uint32 words")
        if type(self.numpy_position) is not int or not 0 <= self.numpy_position <= 624:
            raise ValueError("NumPy RNG position must be in [0, 624]")
        if self.numpy_has_gauss not in {0, 1} or type(self.numpy_has_gauss) is not int:
            raise ValueError("NumPy RNG has_gauss must be integer 0 or 1")
        cached = float(self.numpy_cached_gaussian)
        if not np.isfinite(cached):
            raise ValueError("NumPy cached Gaussian must be finite")
        cpu = _owned_rng_tensor("torch_cpu", self.torch_cpu)
        cuda = (
            None
            if self.torch_cuda is None
            else _owned_rng_tensor("torch_cuda", self.torch_cuda)
        )
        object.__setattr__(self, "torch_cpu", cpu)
        object.__setattr__(self, "torch_cuda", cuda)
        object.__setattr__(self, "numpy_cached_gaussian", cached)
        object.__setattr__(
            self,
            "state_identity",
            _payload_digest(self._identity_payload()),
        )

    @classmethod
    def capture_current(cls, rank: int) -> RankRNGSnapshot:
        import torch

        numpy_state = np.random.get_state()
        cuda = (
            torch.cuda.get_rng_state().cpu().contiguous()
            if torch.cuda.is_available()
            else None
        )
        return cls(
            rank=rank,
            python_state=random.getstate(),
            numpy_bit_generator=str(numpy_state[0]),
            numpy_state=tuple(
                int(item)
                for item in np.asarray(numpy_state[1], dtype=np.uint32).tolist()
            ),
            numpy_position=int(numpy_state[2]),
            numpy_has_gauss=int(numpy_state[3]),
            numpy_cached_gaussian=float(numpy_state[4]),
            torch_cpu=torch.get_rng_state().cpu().contiguous(),
            torch_cuda=cuda,
        )

    def restore_current(self) -> None:
        """Restore this rank's global RNG streams after trusted validation."""

        import torch

        random.setstate(self.python_state)
        np.random.set_state(
            (
                self.numpy_bit_generator,
                np.asarray(self.numpy_state, dtype=np.uint32),
                self.numpy_position,
                self.numpy_has_gauss,
                self.numpy_cached_gaussian,
            )
        )
        torch.set_rng_state(self.torch_cpu.clone())
        if self.torch_cuda is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("checkpoint contains CUDA RNG on a CPU-only runtime")
            torch.cuda.set_rng_state(self.torch_cuda.clone())

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "rank": self.rank,
            "python_state": self.python_state,
            "numpy": {
                "bit_generator": self.numpy_bit_generator,
                "state": self.numpy_state,
                "position": self.numpy_position,
                "has_gauss": self.numpy_has_gauss,
                "cached_gaussian": self.numpy_cached_gaussian,
            },
            "torch_cpu": _tensor_identity_payload(self.torch_cpu),
            "torch_cuda": (
                None
                if self.torch_cuda is None
                else _tensor_identity_payload(self.torch_cuda)
            ),
        }

    def to_checkpoint_payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "rank": self.rank,
            "python_state": self.python_state,
            "numpy_bit_generator": self.numpy_bit_generator,
            "numpy_state": self.numpy_state,
            "numpy_position": self.numpy_position,
            "numpy_has_gauss": self.numpy_has_gauss,
            "numpy_cached_gaussian": self.numpy_cached_gaussian,
            "torch_cpu": self.torch_cpu.clone(),
            "torch_cuda": (
                None if self.torch_cuda is None else self.torch_cuda.clone()
            ),
            "state_identity": self.state_identity,
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: object) -> RankRNGSnapshot:
        if not isinstance(payload, Mapping):
            raise TypeError("RNG checkpoint payload must be a mapping")
        expected = {
            "schema_version",
            "rank",
            "python_state",
            "numpy_bit_generator",
            "numpy_state",
            "numpy_position",
            "numpy_has_gauss",
            "numpy_cached_gaussian",
            "torch_cpu",
            "torch_cuda",
            "state_identity",
        }
        if set(payload) != expected or payload["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("RNG checkpoint payload has invalid fields or version")
        result = cls(
            rank=payload["rank"],
            python_state=tuple(payload["python_state"]),
            numpy_bit_generator=payload["numpy_bit_generator"],
            numpy_state=tuple(payload["numpy_state"]),
            numpy_position=payload["numpy_position"],
            numpy_has_gauss=payload["numpy_has_gauss"],
            numpy_cached_gaussian=payload["numpy_cached_gaussian"],
            torch_cpu=payload["torch_cpu"],
            torch_cuda=payload["torch_cuda"],
        )
        if result.state_identity != payload["state_identity"]:
            raise ValueError("RNG checkpoint state identity mismatch")
        return result

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RankRNGSnapshot)
            and self.state_identity == other.state_identity
        )

    def __hash__(self) -> int:
        return hash(self.state_identity)


@dataclass(frozen=True, slots=True)
class RankCheckpointSnapshot:
    """Frozen state written to exactly one rank shard."""

    rank: int
    world_size: int
    safe_point: CheckpointSafePoint
    component_states: tuple[tuple[str, object], ...]
    rng_state: RankRNGSnapshot
    dynamics_selection_policy: DynamicsSelectionPolicyState

    def __post_init__(self) -> None:
        _rank_and_world_size(self.rank, self.world_size)
        if not isinstance(self.safe_point, CheckpointSafePoint):
            raise TypeError("safe_point must be CheckpointSafePoint")
        if (self.safe_point.rank, self.safe_point.world_size) != (
            self.rank,
            self.world_size,
        ):
            raise ValueError("safe point rank topology disagrees with shard")
        if type(self.component_states) is not tuple or not self.component_states:
            raise ValueError("component_states must be a non-empty tuple")
        names: list[str] = []
        for item in self.component_states:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("component state entries must be (name, state) pairs")
            name, _state = item
            names.append(_identifier("component state name", name))
        if names != sorted(set(names)):
            raise ValueError("component state names must be sorted and unique")
        if not isinstance(self.rng_state, RankRNGSnapshot):
            raise TypeError("rng_state must be RankRNGSnapshot")
        if self.rng_state.rank != self.rank:
            raise ValueError("RNG state rank disagrees with shard")
        if not isinstance(
            self.dynamics_selection_policy,
            DynamicsSelectionPolicyState,
        ):
            raise TypeError(
                "dynamics_selection_policy must be a DynamicsSelectionPolicyState"
            )

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(name for name, _state in self.component_states)

    def component_state(self, name: str) -> object:
        key = _identifier("component state name", name)
        for candidate, state in self.component_states:
            if candidate == key:
                return state
        raise KeyError(f"unknown checkpoint component state {key!r}")

    def to_checkpoint_payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "rank": self.rank,
            "world_size": self.world_size,
            "safe_point": self.safe_point.to_payload(),
            "safe_point_id": self.safe_point.safe_point_id,
            "component_states": [
                {"name": name, "state": state} for name, state in self.component_states
            ],
            "rng_state": self.rng_state.to_checkpoint_payload(),
            "dynamics_selection_policy": (
                self.dynamics_selection_policy.to_checkpoint_payload()
            ),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: object) -> RankCheckpointSnapshot:
        if not isinstance(payload, Mapping):
            raise TypeError("rank shard payload must be a mapping")
        expected = {
            "schema_version",
            "rank",
            "world_size",
            "safe_point",
            "safe_point_id",
            "component_states",
            "rng_state",
            "dynamics_selection_policy",
        }
        if set(payload) != expected or payload["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("rank shard payload has invalid fields or version")
        raw_states = payload["component_states"]
        if not isinstance(raw_states, list) or any(
            not isinstance(item, Mapping) or set(item) != {"name", "state"}
            for item in raw_states
        ):
            raise ValueError("rank shard component states are invalid")
        safe_point = CheckpointSafePoint.from_payload(payload["safe_point"])
        if safe_point.safe_point_id != payload["safe_point_id"]:
            raise ValueError("rank shard safe-point identity mismatch")
        return cls(
            rank=payload["rank"],
            world_size=payload["world_size"],
            safe_point=safe_point,
            component_states=tuple(
                (item["name"], item["state"]) for item in raw_states
            ),
            rng_state=RankRNGSnapshot.from_checkpoint_payload(payload["rng_state"]),
            dynamics_selection_policy=(
                DynamicsSelectionPolicyState.from_checkpoint_payload(
                    payload["dynamics_selection_policy"]
                )
            ),
        )


class CheckpointStateCollector:
    """Capture component/RNG state and the immutable selection policy."""

    def __init__(
        self,
        *,
        component_state_sources: Mapping[str, Callable[[], object]],
        dynamics_selection_policy_source: Callable[[], DynamicsSelectionPolicyState],
        rng_state_source: Callable[[int], RankRNGSnapshot] | None = None,
    ) -> None:
        if (
            not isinstance(component_state_sources, Mapping)
            or not component_state_sources
        ):
            raise ValueError("component_state_sources must be a non-empty mapping")
        sources: list[tuple[str, Callable[[], object]]] = []
        for name, source in component_state_sources.items():
            identifier = _identifier("component state source", name)
            if not callable(source):
                raise TypeError(
                    f"component state source {identifier!r} must be callable"
                )
            sources.append((identifier, source))
        sources.sort(key=lambda item: item[0])
        if len({name for name, _source in sources}) != len(sources):
            raise ValueError("component state source names must be unique")
        if not callable(dynamics_selection_policy_source):
            raise TypeError("dynamics_selection_policy_source must be callable")
        if rng_state_source is not None and not callable(rng_state_source):
            raise TypeError("rng_state_source must be callable or None")
        self._component_sources = tuple(sources)
        self._dynamics_policy_source = dynamics_selection_policy_source
        self._rng_source = rng_state_source or RankRNGSnapshot.capture_current

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(name for name, _source in self._component_sources)

    def capture(
        self,
        safe_point: CheckpointSafePoint,
    ) -> RankCheckpointSnapshot:
        if not isinstance(safe_point, CheckpointSafePoint):
            raise TypeError("safe_point must be CheckpointSafePoint")
        states = tuple(
            (name, _frozen_pickle_copy(source(), label=name))
            for name, source in self._component_sources
        )
        rng_state = self._rng_source(safe_point.rank)
        if not isinstance(rng_state, RankRNGSnapshot):
            raise TypeError("rng_state_source must return RankRNGSnapshot")
        dynamics_policy = self._dynamics_policy_source()
        if not isinstance(dynamics_policy, DynamicsSelectionPolicyState):
            raise TypeError(
                "dynamics_selection_policy_source must return "
                "DynamicsSelectionPolicyState"
            )
        return RankCheckpointSnapshot(
            rank=safe_point.rank,
            world_size=safe_point.world_size,
            safe_point=safe_point,
            component_states=states,
            rng_state=rng_state,
            dynamics_selection_policy=dynamics_policy,
        )


def _owned_rng_tensor(name: str, value: object):
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} RNG state must be a torch.Tensor")
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() < 1:
        raise ValueError(f"{name} RNG state must be non-empty 1-D uint8")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} RNG state must be detached")
    return value.detach().to(device="cpu").contiguous().clone()


def _tensor_identity_payload(value: object) -> dict[str, object]:
    tensor = _owned_rng_tensor("identity", value)
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "data_base64": base64.b64encode(tensor.numpy().tobytes()).decode("ascii"),
    }


def _frozen_pickle_copy(value: object, *, label: str) -> object:
    try:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return pickle.loads(payload)
    except BaseException as exc:
        raise TypeError(
            f"component state {label!r} must be pickle-serializable"
        ) from exc


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"{name} must not contain path separators")
    return value


def _payload_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = (
    "CheckpointStateCollector",
    "RankCheckpointSnapshot",
    "RankRNGSnapshot",
)
