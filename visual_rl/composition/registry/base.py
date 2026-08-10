"""Immutable alias registry for import-safe component descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from visual_rl.core.contracts.composition import (
    COMPONENT_KINDS,
    ComponentDescriptor,
    ComponentKind,
)
from visual_rl.errors import ComponentError

__all__ = (
    "COMPONENT_KINDS",
    "Registry",
    "RegistryError",
    "RegistryIssueCode",
)

class RegistryIssueCode(str, Enum):
    """Stable reasons why catalog registration or static resolution failed."""

    ALIAS_CONFLICT = "alias_conflict"
    INVALID_DECLARATION = "invalid_declaration"
    INVALID_PROVIDER = "invalid_provider"
    PROVIDER_FAILED = "provider_failed"
    PROVIDER_IMPORT_FAILED = "provider_import_failed"
    REMOVED_ALIAS = "removed_alias"
    UNKNOWN_ALIAS = "unknown_alias"


class RegistryError(ComponentError):
    """Structured internal failure at the composition registry boundary."""

    def __init__(
        self,
        code: RegistryIssueCode,
        message: str,
        *,
        kind: ComponentKind,
        alias: str,
    ) -> None:
        if not isinstance(code, RegistryIssueCode):
            raise TypeError("code must be a RegistryIssueCode")
        super().__init__(message, kind=kind, name=alias)
        self.code = code.value
        self.alias = alias


@dataclass(frozen=True, slots=True)
class Registry:
    """One immutable alias table containing descriptor strings only.

    ``register`` returns a new value instead of mutating global state. This
    makes catalog assembly deterministic and keeps plugin/catalog conflicts at
    the composition boundary.
    """

    kind: ComponentKind
    descriptors: tuple[ComponentDescriptor, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in COMPONENT_KINDS:
            raise ValueError(f"unsupported component kind: {self.kind!r}")
        if type(self.descriptors) is not tuple or any(
            not isinstance(item, ComponentDescriptor) for item in self.descriptors
        ):
            raise TypeError("descriptors must be a tuple of ComponentDescriptor")
        aliases = tuple(item.alias for item in self.descriptors)
        if len(aliases) != len(set(aliases)):
            duplicate = next(alias for alias in aliases if aliases.count(alias) > 1)
            raise RegistryError(
                RegistryIssueCode.ALIAS_CONFLICT,
                f"duplicate {self.kind} alias {duplicate!r}",
                kind=self.kind,
                alias=duplicate,
            )
        object.__setattr__(
            self,
            "descriptors",
            tuple(sorted(self.descriptors, key=lambda item: item.alias)),
        )

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return active aliases, excluding retained tombstones."""

        return tuple(
            item.alias for item in self.descriptors if item.removed_message is None
        )

    @property
    def tombstones(self) -> tuple[str, ...]:
        return tuple(
            item.alias for item in self.descriptors if item.removed_message is not None
        )

    def register(self, *descriptors: ComponentDescriptor) -> Registry:
        """Return a registry extended by new, conflict-free descriptors."""

        if any(not isinstance(item, ComponentDescriptor) for item in descriptors):
            raise TypeError("register expects ComponentDescriptor values")
        existing = {item.alias for item in self.descriptors}
        incoming: set[str] = set()
        for descriptor in descriptors:
            if descriptor.alias in existing or descriptor.alias in incoming:
                raise RegistryError(
                    RegistryIssueCode.ALIAS_CONFLICT,
                    f"duplicate {self.kind} alias {descriptor.alias!r}",
                    kind=self.kind,
                    alias=descriptor.alias,
                )
            incoming.add(descriptor.alias)
        if not descriptors:
            return self
        return Registry(self.kind, (*self.descriptors, *descriptors))

    def lookup(self, alias: str) -> ComponentDescriptor | None:
        if not isinstance(alias, str) or not alias:
            raise ValueError("alias must be a non-empty string")
        return next((item for item in self.descriptors if item.alias == alias), None)

    def resolve_descriptor(self, alias: str) -> ComponentDescriptor:
        """Resolve one active alias, failing explicitly for tombstones."""

        descriptor = self.lookup(alias)
        if descriptor is None:
            available = ", ".join(self.aliases) or "<none>"
            raise RegistryError(
                RegistryIssueCode.UNKNOWN_ALIAS,
                f"unknown {self.kind} alias {alias!r}; available: {available}",
                kind=self.kind,
                alias=alias,
            )
        if descriptor.removed_message is not None:
            raise RegistryError(
                RegistryIssueCode.REMOVED_ALIAS,
                descriptor.removed_message,
                kind=self.kind,
                alias=alias,
            )
        return descriptor
