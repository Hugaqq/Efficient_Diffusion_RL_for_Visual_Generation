"""Component-only parameter snapshots with prevalidated atomic restore."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from visual_rl.models.state.parameters import (
    ParameterStateManager,
    ParameterTopology,
)

__all__ = (
    "MODEL_PARAMETER_STATE_SCHEMA_VERSION",
    "AtomicRestoreError",
    "ModelParameterState",
    "ModelStateAdapter",
    "StateTopologyError",
)


MODEL_PARAMETER_STATE_SCHEMA_VERSION = 2


class StateTopologyError(ValueError):
    """Raised before mutation when checkpoint topology is incompatible."""


class AtomicRestoreError(RuntimeError):
    """Raised after a failed copy; rollback is attempted before propagation."""


@dataclass(frozen=True, slots=True, init=False)
class ModelParameterState:
    """Detached CPU snapshot of the canonical standalone model parameters."""

    topology: ParameterTopology
    _tensors: Mapping[str, Any] = field(repr=False)
    projection_id: str
    schema_version: int

    def __init__(
        self,
        topology: ParameterTopology,
        tensors: Mapping[str, Any],
        *,
        projection_id: str | None = None,
        schema_version: int = MODEL_PARAMETER_STATE_SCHEMA_VERSION,
    ) -> None:
        import torch

        if not isinstance(topology, ParameterTopology):
            raise TypeError("topology must be ParameterTopology")
        if not isinstance(tensors, Mapping):
            raise TypeError("state tensors must be a mapping")
        if (
            type(schema_version) is not int
            or schema_version != MODEL_PARAMETER_STATE_SCHEMA_VERSION
        ):
            raise StateTopologyError("model parameter state schema_version must be 2")
        bound_projection_id = topology.state_projection_id
        if bound_projection_id is None:
            raise StateTopologyError(
                "parameter topology is not bound to a model state projection"
            )
        resolved_projection_id = (
            bound_projection_id if projection_id is None else projection_id
        )
        if (
            not isinstance(resolved_projection_id, str)
            or len(resolved_projection_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in resolved_projection_id
            )
        ):
            raise StateTopologyError(
                "model parameter state projection_id must be a SHA-256 digest"
            )
        if resolved_projection_id != bound_projection_id:
            raise StateTopologyError("model state projection identity mismatch")
        expected = tuple(item.name for item in topology.entries)
        if set(tensors) != set(expected):
            missing = sorted(set(expected).difference(tensors))
            extra = sorted(set(tensors).difference(expected))
            raise StateTopologyError(
                f"state parameter keys mismatch; missing={missing}, extra={extra}"
            )
        owned: dict[str, Any] = {}
        descriptors = {item.name: item for item in topology.entries}
        for name in expected:
            value = tensors[name]
            descriptor = descriptors[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"state tensor {name!r} must be torch.Tensor")
            if tuple(value.shape) != descriptor.shape:
                raise StateTopologyError(f"state tensor {name!r} shape mismatch")
            if str(value.dtype) != descriptor.dtype:
                raise StateTopologyError(f"state tensor {name!r} dtype mismatch")
            if not bool(torch.isfinite(value).all()):
                raise StateTopologyError(f"state tensor {name!r} must be finite")
            owned[name] = value.detach().to(device="cpu").contiguous().clone()
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "_tensors", MappingProxyType(owned))
        object.__setattr__(self, "projection_id", resolved_projection_id)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def tensors(self) -> Mapping[str, Any]:
        """Return owned clones so callers cannot mutate the frozen snapshot."""

        return MappingProxyType(
            {name: value.clone() for name, value in self._tensors.items()}
        )

    def tensor(self, name: str) -> Any:
        try:
            return self._tensors[name].clone()
        except KeyError:
            raise KeyError(name) from None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "topology_identity": self.topology.identity,
            "projection_id": self.projection_id,
            "tensors": {name: value.clone() for name, value in self._tensors.items()},
        }

    @classmethod
    def from_payload(
        cls,
        topology: ParameterTopology,
        payload: Mapping[str, object],
    ) -> ModelParameterState:
        if not isinstance(payload, Mapping):
            raise TypeError("model state payload must be a mapping")
        if payload.get("schema_version") == 1:
            raise StateTopologyError(
                "legacy model parameter state schema_version 1 has no projection_id; "
                "regenerate the checkpoint with schema_version 2"
            )
        if set(payload) != {
            "schema_version",
            "topology_identity",
            "projection_id",
            "tensors",
        }:
            raise StateTopologyError("model state payload keys are invalid")
        if payload["topology_identity"] != topology.identity:
            raise StateTopologyError("model state topology identity mismatch")
        tensors = payload["tensors"]
        if not isinstance(tensors, Mapping):
            raise TypeError("model state payload tensors must be a mapping")
        return cls(
            topology,
            tensors,
            projection_id=payload["projection_id"],
            schema_version=payload["schema_version"],
        )


class ModelStateAdapter:
    """Expose model parameter state without owning run-level checkpoints."""

    def __init__(self, parameters: ParameterStateManager) -> None:
        if not isinstance(parameters, ParameterStateManager):
            raise TypeError("parameters must be ParameterStateManager")
        self.parameters = parameters

    def capture(self) -> ModelParameterState:
        named = self.parameters.named_trainable_parameters()
        projection = self.parameters.state_projection
        if {item.name for item in named} != set(projection.standalone_parameter_names):
            raise StateTopologyError(
                "live standalone tensor set disagrees with model state projection"
            )
        return ModelParameterState(
            self.parameters.topology,
            {
                item.name: item.parameter.detach().to(device="cpu").clone()
                for item in named
            },
            projection_id=projection.projection_id,
        )

    def restore(self, state: ModelParameterState) -> None:
        """Validate and stage every tensor before atomically copying any value."""

        import torch

        if not isinstance(state, ModelParameterState):
            raise TypeError("state must be ModelParameterState")
        named = self.parameters.named_trainable_parameters()
        topology = self.parameters.topology
        projection = self.parameters.state_projection
        if state.projection_id != projection.projection_id:
            raise StateTopologyError("model state projection identity mismatch")
        if state.topology.identity != topology.identity:
            raise StateTopologyError("model state topology identity mismatch")
        incoming = state._tensors
        targets = {item.name: item.parameter for item in named}

        staged: dict[str, Any] = {}
        backups: dict[str, Any] = {}
        for descriptor in topology.entries:
            name = descriptor.name
            target = targets[name]
            value = incoming[name]
            if tuple(target.shape) != descriptor.shape:
                raise StateTopologyError(
                    f"live parameter {name!r} shape changed before restore"
                )
            if str(target.dtype) != descriptor.dtype:
                raise StateTopologyError(
                    f"live parameter {name!r} dtype changed before restore"
                )
            if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
                raise StateTopologyError(f"state tensor {name!r} is incompatible")
            staged[name] = value.to(device=target.device).contiguous().clone()
            backups[name] = target.detach().clone()

        copied: list[str] = []
        try:
            with torch.no_grad():
                for descriptor in topology.entries:
                    name = descriptor.name
                    self._copy_into(targets[name], staged[name], name=name)
                    copied.append(name)
        except BaseException as exc:
            rollback_errors: list[BaseException] = []
            with torch.no_grad():
                for descriptor in reversed(topology.entries):
                    name = descriptor.name
                    try:
                        self._copy_into(targets[name], backups[name], name=name)
                    except BaseException as rollback_exc:  # noqa: BLE001
                        # Atomic restore must aggregate rollback failure even
                        # when cancellation interrupted the original copy.
                        rollback_errors.append(rollback_exc)
            if rollback_errors:
                raise AtomicRestoreError(
                    "model restore and rollback both failed; parameter state may be partial: "
                    + "; ".join(str(item) for item in rollback_errors)
                ) from exc
            raise AtomicRestoreError(
                "model restore failed and the original parameter state was restored; "
                f"successful copies before failure={copied}"
            ) from exc

    def _copy_into(self, target: Any, source: Any, *, name: str) -> None:
        del name
        target.copy_(source)
