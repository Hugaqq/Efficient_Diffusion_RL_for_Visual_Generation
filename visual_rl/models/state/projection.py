"""Canonical model component-to-checkpoint state projection.

The projection is deliberately derived from component roles and the frozen
parameter topology.  It never inspects implementation class names or searches
module aliases, so save and restore can share one exact identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from visual_rl.models.lifecycle.components import ComponentRole, ModelComponents

if TYPE_CHECKING:
    from visual_rl.models.state.parameters import ParameterStateManager

__all__ = (
    "MODEL_STATE_PROJECTION_SCHEMA_VERSION",
    "ModelComponentStateMembership",
    "ModelParameterStateMembership",
    "ModelStateProjection",
    "ModelStateProjectionError",
)


MODEL_STATE_PROJECTION_SCHEMA_VERSION = 1


class ModelStateProjectionError(ValueError):
    """Raised when model state ownership is incomplete, overlapping, or stale."""


def _require_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelStateProjectionError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ModelStateProjectionError(f"{field_name} must be a SHA-256 hex digest")
    return value


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelComponentStateMembership:
    """One component's role-only membership in the prepared model bundle."""

    name: str
    roles: tuple[str, ...]
    managed_residency: bool

    def __post_init__(self) -> None:
        _require_name(self.name, "component membership name")
        if type(self.roles) is not tuple or not self.roles:
            raise ModelStateProjectionError(
                "component membership roles must be a non-empty tuple"
            )
        resolved: list[str] = []
        for value in self.roles:
            try:
                resolved.append(ComponentRole(value).value)
            except (TypeError, ValueError):
                raise ModelStateProjectionError(
                    f"invalid component membership role: {value!r}"
                ) from None
        canonical = tuple(sorted(resolved))
        if tuple(self.roles) != canonical or len(canonical) != len(set(canonical)):
            raise ModelStateProjectionError(
                "component membership roles must be unique and canonical"
            )
        if type(self.managed_residency) is not bool:
            raise TypeError("component membership managed_residency must be bool")

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "roles": list(self.roles),
            "managed_residency": self.managed_residency,
        }


@dataclass(frozen=True, slots=True)
class ModelParameterStateMembership:
    """One standalone tensor and its uniquely declared component owner."""

    name: str
    component_name: str
    parameter_name: str

    def __post_init__(self) -> None:
        _require_name(self.name, "parameter membership name")
        _require_name(self.component_name, "parameter membership component_name")
        _require_name(self.parameter_name, "parameter membership parameter_name")
        if self.name != f"{self.component_name}.{self.parameter_name}":
            raise ModelStateProjectionError(
                "parameter membership name must match its canonical ownership path"
            )

    def to_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "component_name": self.component_name,
            "parameter_name": self.parameter_name,
        }


@dataclass(frozen=True, slots=True)
class ModelStateProjection:
    """Immutable exact partition of standalone and artifact-rehydrated state."""

    standalone_saved_component_names: tuple[str, ...]
    standalone_parameter_names: tuple[str, ...]
    bundle_membership_identity: str
    artifact_rehydrated_component_names: tuple[str, ...]
    projection_id: str
    component_membership: tuple[ModelComponentStateMembership, ...]
    parameter_membership: tuple[ModelParameterStateMembership, ...]
    parameter_topology_identity: str
    schema_version: int = MODEL_STATE_PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != MODEL_STATE_PROJECTION_SCHEMA_VERSION
        ):
            raise ModelStateProjectionError(
                "model state projection schema_version must be 1"
            )
        if (
            type(self.component_membership) is not tuple
            or not self.component_membership
        ):
            raise ModelStateProjectionError(
                "model state projection component membership must not be empty"
            )
        if any(
            not isinstance(item, ModelComponentStateMembership)
            for item in self.component_membership
        ):
            raise TypeError(
                "component_membership entries must be ModelComponentStateMembership"
            )
        if (
            type(self.parameter_membership) is not tuple
            or not self.parameter_membership
        ):
            raise ModelStateProjectionError(
                "model state projection parameter membership must not be empty"
            )
        if any(
            not isinstance(item, ModelParameterStateMembership)
            for item in self.parameter_membership
        ):
            raise TypeError(
                "parameter_membership entries must be ModelParameterStateMembership"
            )

        component_names = tuple(item.name for item in self.component_membership)
        parameter_names = tuple(item.name for item in self.parameter_membership)
        if component_names != tuple(sorted(component_names)):
            raise ModelStateProjectionError(
                "component membership must be in canonical name order"
            )
        if parameter_names != tuple(sorted(parameter_names)):
            raise ModelStateProjectionError(
                "parameter membership must be in canonical name order"
            )
        if len(component_names) != len(set(component_names)):
            raise ModelStateProjectionError("component membership names must be unique")
        if len(parameter_names) != len(set(parameter_names)):
            raise ModelStateProjectionError("parameter membership names must be unique")

        standalone_components = tuple(
            sorted({item.component_name for item in self.parameter_membership})
        )
        if self.standalone_saved_component_names != standalone_components:
            raise ModelStateProjectionError(
                "standalone saved components must exactly equal parameter owners"
            )
        if self.standalone_parameter_names != parameter_names:
            raise ModelStateProjectionError(
                "standalone parameter names must exactly equal parameter membership"
            )
        component_name_set = set(component_names)
        standalone_set = set(self.standalone_saved_component_names)
        artifact_names = self.artifact_rehydrated_component_names
        if artifact_names != tuple(sorted(artifact_names)):
            raise ModelStateProjectionError(
                "artifact-rehydrated components must be in canonical name order"
            )
        if len(artifact_names) != len(set(artifact_names)):
            raise ModelStateProjectionError(
                "artifact-rehydrated component names must be unique"
            )
        artifact_set = set(artifact_names)
        overlap = sorted(standalone_set.intersection(artifact_set))
        if overlap:
            raise ModelStateProjectionError(
                f"standalone and artifact-rehydrated components overlap: {overlap}"
            )
        missing = sorted(component_name_set.difference(standalone_set | artifact_set))
        extra = sorted((standalone_set | artifact_set).difference(component_name_set))
        if missing or extra:
            raise ModelStateProjectionError(
                "model component state projection is incomplete; "
                f"missing={missing}, extra={extra}"
            )
        unknown_parameter_owners = sorted(standalone_set.difference(component_name_set))
        if unknown_parameter_owners:
            raise ModelStateProjectionError(
                "parameter owners are absent from bundle membership: "
                f"{unknown_parameter_owners}"
            )

        membership_by_name = {item.name: item for item in self.component_membership}
        non_trainable_owners = sorted(
            name
            for name in standalone_set
            if ComponentRole.TRAINABLE.value not in membership_by_name[name].roles
        )
        trainable_without_state = sorted(
            name
            for name, membership in membership_by_name.items()
            if ComponentRole.TRAINABLE.value in membership.roles
            and name not in standalone_set
        )
        if non_trainable_owners or trainable_without_state:
            raise ModelStateProjectionError(
                "trainable role and standalone state ownership disagree; "
                f"non_trainable_owners={non_trainable_owners}, "
                f"trainable_without_state={trainable_without_state}"
            )

        _require_sha256(
            self.parameter_topology_identity,
            "parameter_topology_identity",
        )
        _require_sha256(
            self.bundle_membership_identity,
            "bundle_membership_identity",
        )
        expected_bundle_identity = _digest(
            [item.to_payload() for item in self.component_membership]
        )
        if self.bundle_membership_identity != expected_bundle_identity:
            raise ModelStateProjectionError(
                "bundle membership identity does not match component membership"
            )
        _require_sha256(self.projection_id, "projection_id")
        if self.projection_id != _digest(self._identity_payload()):
            raise ModelStateProjectionError(
                "model state projection identity does not match its contents"
            )

    @classmethod
    def from_components(
        cls,
        components: ModelComponents,
        parameters: ParameterStateManager,
    ) -> ModelStateProjection:
        """Derive one exact projection from role and parameter ownership topology."""

        from visual_rl.models.state.parameters import ParameterStateManager

        if not isinstance(components, ModelComponents):
            raise TypeError("components must be ModelComponents")
        if not isinstance(parameters, ParameterStateManager):
            raise TypeError("parameters must be ParameterStateManager")
        if parameters.components is not components:
            raise ModelStateProjectionError(
                "parameter topology belongs to a different ModelComponents inventory"
            )

        named = parameters.named_trainable_parameters()
        topology = parameters.topology
        topology_entries = {item.name: item for item in topology.entries}
        if set(topology_entries) != {item.name for item in named}:
            raise ModelStateProjectionError(
                "live trainable parameters drifted from the frozen topology"
            )
        for item in named:
            descriptor = topology_entries[item.name]
            if (
                descriptor.component_name != item.component_name
                or descriptor.parameter_name != item.parameter_name
            ):
                raise ModelStateProjectionError(
                    "parameter ownership drifted from the frozen topology"
                )

        component_membership = tuple(
            sorted(
                (
                    ModelComponentStateMembership(
                        name=binding.name,
                        roles=tuple(sorted(role.value for role in binding.roles)),
                        managed_residency=binding.managed_residency,
                    )
                    for binding in components.bindings
                ),
                key=lambda item: item.name,
            )
        )
        parameter_membership = tuple(
            sorted(
                (
                    ModelParameterStateMembership(
                        name=item.name,
                        component_name=item.component_name,
                        parameter_name=item.parameter_name,
                    )
                    for item in named
                ),
                key=lambda item: item.name,
            )
        )
        standalone_components = tuple(
            sorted({item.component_name for item in parameter_membership})
        )
        role_trainable_components = tuple(
            sorted(
                membership.name
                for membership in component_membership
                if ComponentRole.TRAINABLE.value in membership.roles
            )
        )
        if standalone_components != role_trainable_components:
            raise ModelStateProjectionError(
                "trainable role set does not exactly match restorable parameter owners"
            )
        all_components = {item.name for item in component_membership}
        artifact_components = tuple(
            sorted(all_components.difference(standalone_components))
        )
        bundle_identity = _digest([item.to_payload() for item in component_membership])
        identity_payload = {
            "schema_version": MODEL_STATE_PROJECTION_SCHEMA_VERSION,
            "standalone_saved_component_names": list(standalone_components),
            "standalone_parameter_names": [item.name for item in parameter_membership],
            "bundle_membership_identity": bundle_identity,
            "artifact_rehydrated_component_names": list(artifact_components),
            "component_membership": [
                item.to_payload() for item in component_membership
            ],
            "parameter_membership": [
                item.to_payload() for item in parameter_membership
            ],
            "parameter_topology_identity": topology.identity,
        }
        return cls(
            standalone_saved_component_names=standalone_components,
            standalone_parameter_names=tuple(
                item.name for item in parameter_membership
            ),
            bundle_membership_identity=bundle_identity,
            artifact_rehydrated_component_names=artifact_components,
            projection_id=_digest(identity_payload),
            component_membership=component_membership,
            parameter_membership=parameter_membership,
            parameter_topology_identity=topology.identity,
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "standalone_saved_component_names": list(
                self.standalone_saved_component_names
            ),
            "standalone_parameter_names": list(self.standalone_parameter_names),
            "bundle_membership_identity": self.bundle_membership_identity,
            "artifact_rehydrated_component_names": list(
                self.artifact_rehydrated_component_names
            ),
            "component_membership": [
                item.to_payload() for item in self.component_membership
            ],
            "parameter_membership": [
                item.to_payload() for item in self.parameter_membership
            ],
            "parameter_topology_identity": self.parameter_topology_identity,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._identity_payload(), "projection_id": self.projection_id}
