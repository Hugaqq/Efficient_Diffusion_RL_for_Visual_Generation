"""Stable trainable-parameter identity and topology for model components."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from visual_rl.models.lifecycle.components import ComponentRole, ModelComponents

if TYPE_CHECKING:
    from visual_rl.models.state.projection import ModelStateProjection

__all__ = (
    "NamedTrainableParameter",
    "ParameterStateError",
    "ParameterStateManager",
    "ParameterTopology",
    "ParameterTopologyEntry",
)


class ParameterStateError(RuntimeError):
    """Raised when trainable parameter ownership or topology drifts."""


@dataclass(frozen=True, slots=True)
class NamedTrainableParameter:
    name: str
    component_name: str
    parameter_name: str
    parameter: Any


@dataclass(frozen=True, slots=True)
class ParameterTopologyEntry:
    name: str
    component_name: str
    parameter_name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int

    def __post_init__(self) -> None:
        for field_name in ("name", "component_name", "parameter_name", "dtype"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ParameterStateError(f"{field_name} must be a non-empty string")
        if type(self.shape) is not tuple or any(
            type(item) is not int or item < 0 for item in self.shape
        ):
            raise ParameterStateError("parameter shape must contain non-negative ints")
        if type(self.numel) is not int or self.numel < 0:
            raise ParameterStateError("parameter numel must be non-negative")


@dataclass(frozen=True, slots=True)
class ParameterTopology:
    entries: tuple[ParameterTopologyEntry, ...]
    identity: str
    total_numel: int
    state_projection_id: str | None = None

    @classmethod
    def from_entries(
        cls,
        entries: tuple[ParameterTopologyEntry, ...],
    ) -> ParameterTopology:
        if type(entries) is not tuple or not entries:
            raise ParameterStateError("trainable parameter topology must not be empty")
        if any(not isinstance(item, ParameterTopologyEntry) for item in entries):
            raise TypeError("topology entries must be ParameterTopologyEntry")
        names = tuple(item.name for item in entries)
        if len(names) != len(set(names)):
            raise ParameterStateError("trainable parameter names must be unique")
        return cls(
            entries=entries,
            identity=_topology_identity(entries),
            total_numel=sum(item.numel for item in entries),
        )

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or not self.entries:
            raise ParameterStateError("topology entries must not be empty")
        if any(not isinstance(item, ParameterTopologyEntry) for item in self.entries):
            raise TypeError("topology entries must be ParameterTopologyEntry")
        names = tuple(item.name for item in self.entries)
        if len(names) != len(set(names)):
            raise ParameterStateError("trainable parameter names must be unique")
        if (
            not isinstance(self.identity, str)
            or len(self.identity) != 64
            or any(character not in "0123456789abcdef" for character in self.identity)
        ):
            raise ParameterStateError("topology identity must be a SHA-256 hex digest")
        if self.identity != _topology_identity(self.entries):
            raise ParameterStateError("topology identity does not match its entries")
        if self.total_numel != sum(item.numel for item in self.entries):
            raise ParameterStateError("topology total_numel is inconsistent")
        if self.state_projection_id is not None and (
            not isinstance(self.state_projection_id, str)
            or len(self.state_projection_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.state_projection_id
            )
        ):
            raise ParameterStateError(
                "state_projection_id must be a SHA-256 hex digest when bound"
            )

    def bind_state_projection(self, projection_id: str) -> ParameterTopology:
        """Return this parameter topology bound to one model-state projection."""

        if self.state_projection_id is not None:
            if self.state_projection_id != projection_id:
                raise ParameterStateError(
                    "parameter topology is already bound to another state projection"
                )
            return self
        return replace(self, state_projection_id=projection_id)


def _topology_identity(entries: tuple[ParameterTopologyEntry, ...]) -> str:
    serializable = [
        {
            "name": item.name,
            "component_name": item.component_name,
            "parameter_name": item.parameter_name,
            "shape": list(item.shape),
            "dtype": item.dtype,
            "numel": item.numel,
        }
        for item in entries
    ]
    payload = json.dumps(
        serializable,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ParameterStateManager:
    """Freeze one canonical trainable-parameter path and object topology."""

    def __init__(self, components: ModelComponents) -> None:
        if not isinstance(components, ModelComponents):
            raise TypeError("components must be ModelComponents")
        self.components = components
        selected = self._inspect()
        self._object_ids = tuple(id(item.parameter) for item in selected)
        self._topology = self._topology_for(selected)
        from visual_rl.models.state.projection import ModelStateProjection

        projection = ModelStateProjection.from_components(components, self)
        self._topology = self._topology.bind_state_projection(projection.projection_id)
        self._state_projection = projection

    @property
    def topology(self) -> ParameterTopology:
        return self._topology

    @property
    def state_projection(self) -> ModelStateProjection:
        """Return the immutable projection after rechecking live ownership."""

        from visual_rl.models.state.projection import ModelStateProjection

        current = ModelStateProjection.from_components(self.components, self)
        if current.projection_id != self._state_projection.projection_id:
            raise ParameterStateError(
                "model state projection changed after topology bind"
            )
        return self._state_projection

    def named_trainable_parameters(self) -> tuple[NamedTrainableParameter, ...]:
        selected = self._inspect()
        object_ids = tuple(id(item.parameter) for item in selected)
        topology = self._topology_for(selected)
        if object_ids != self._object_ids:
            raise ParameterStateError(
                "trainable parameter object identity changed after topology bind"
            )
        if topology.identity != self._topology.identity:
            raise ParameterStateError(
                "trainable parameter topology changed after topology bind"
            )
        return selected

    def parameters(self) -> tuple[Any, ...]:
        return tuple(item.parameter for item in self.named_trainable_parameters())

    def _inspect(self) -> tuple[NamedTrainableParameter, ...]:
        import torch

        selected: list[NamedTrainableParameter] = []
        seen_names: set[str] = set()
        seen_objects: set[int] = set()
        trainable_bindings = self.components.bindings_for(ComponentRole.TRAINABLE)
        if not trainable_bindings:
            raise ParameterStateError("ModelComponents has no trainable role")
        for binding in self.components.bindings:
            if ComponentRole.TRAINABLE in binding.roles:
                continue
            named_parameters = getattr(binding.component, "named_parameters", None)
            if callable(named_parameters) and any(
                parameter.requires_grad
                for _name, parameter in tuple(named_parameters())
            ):
                raise ParameterStateError(
                    f"non-trainable component {binding.name!r} must be frozen"
                )
        for binding in trainable_bindings:
            named_parameters = getattr(binding.component, "named_parameters", None)
            if not callable(named_parameters):
                raise ParameterStateError(
                    f"trainable component {binding.name!r} lacks named_parameters()"
                )
            component_count = 0
            for parameter_name, parameter in tuple(named_parameters()):
                if not isinstance(parameter, torch.nn.Parameter):
                    raise ParameterStateError(
                        f"{binding.name}.{parameter_name} is not torch.nn.Parameter"
                    )
                if not parameter.requires_grad:
                    continue
                if not isinstance(parameter_name, str) or not parameter_name:
                    raise ParameterStateError(
                        f"component {binding.name!r} returned an empty parameter name"
                    )
                canonical_name = f"{binding.name}.{parameter_name}"
                if canonical_name in seen_names:
                    raise ParameterStateError(
                        f"duplicate trainable parameter name: {canonical_name}"
                    )
                if id(parameter) in seen_objects:
                    raise ParameterStateError(
                        "one trainable parameter object has multiple ownership paths"
                    )
                seen_names.add(canonical_name)
                seen_objects.add(id(parameter))
                selected.append(
                    NamedTrainableParameter(
                        name=canonical_name,
                        component_name=binding.name,
                        parameter_name=parameter_name,
                        parameter=parameter,
                    )
                )
                component_count += 1
            if component_count == 0:
                raise ParameterStateError(
                    f"trainable component {binding.name!r} has no trainable parameters"
                )
        return tuple(selected)

    @staticmethod
    def _topology_for(
        selected: tuple[NamedTrainableParameter, ...],
    ) -> ParameterTopology:
        entries = tuple(
            ParameterTopologyEntry(
                name=item.name,
                component_name=item.component_name,
                parameter_name=item.parameter_name,
                shape=tuple(int(size) for size in item.parameter.shape),
                dtype=str(item.parameter.dtype),
                numel=int(item.parameter.numel()),
            )
            for item in selected
        )
        return ParameterTopology.from_entries(entries)
