"""Deterministic assembly of import-safe domain catalog fragments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from visual_rl.composition.registry.base import (
    COMPONENT_KINDS,
    Registry,
    RegistryError,
    RegistryIssueCode,
)
from visual_rl.core.contracts.composition import (
    CatalogFragment,
    ComponentKind,
)

__all__ = (
    "Catalog",
    "CatalogFragment",
    "build_catalog",
)

@dataclass(frozen=True, slots=True)
class Catalog:
    """The composition-owned union of all supplied domain fragments."""

    registries: tuple[Registry, ...]

    def __post_init__(self) -> None:
        if type(self.registries) is not tuple or any(
            not isinstance(item, Registry) for item in self.registries
        ):
            raise TypeError("registries must be a tuple of Registry values")
        kinds = tuple(item.kind for item in self.registries)
        if len(kinds) != len(set(kinds)):
            raise ValueError("catalog registry kinds must be unique")
        order = {kind: index for index, kind in enumerate(COMPONENT_KINDS)}
        object.__setattr__(
            self,
            "registries",
            tuple(sorted(self.registries, key=lambda item: order[item.kind])),
        )

    @property
    def kinds(self) -> tuple[ComponentKind, ...]:
        return tuple(item.kind for item in self.registries)

    def for_kind(self, kind: ComponentKind) -> Registry:
        if kind not in COMPONENT_KINDS:
            raise ValueError(f"unsupported component kind: {kind!r}")
        registry = next((item for item in self.registries if item.kind == kind), None)
        if registry is None:
            raise KeyError(f"catalog has no {kind!r} fragment")
        return registry


def build_catalog(fragments: Iterable[CatalogFragment]) -> Catalog:
    """Combine domain fragments without importing any implementation module."""

    if isinstance(fragments, (str, bytes)) or not isinstance(fragments, Iterable):
        raise TypeError("fragments must be an iterable of CatalogFragment values")
    registries: dict[ComponentKind, Registry] = {}
    owners: dict[tuple[ComponentKind, str], str] = {}
    for fragment in fragments:
        if not isinstance(fragment, CatalogFragment):
            raise TypeError("fragments must contain CatalogFragment values")
        registry = registries.get(fragment.kind, Registry(fragment.kind))
        for descriptor in fragment.descriptors:
            key = (fragment.kind, descriptor.alias)
            previous_owner = owners.get(key)
            if previous_owner is not None:
                raise RegistryError(
                    RegistryIssueCode.ALIAS_CONFLICT,
                    f"{fragment.kind} alias {descriptor.alias!r} is declared by "
                    f"both {previous_owner!r} and {fragment.owner!r}",
                    kind=fragment.kind,
                    alias=descriptor.alias,
                )
            owners[key] = fragment.owner
            registry = registry.register(descriptor)
        registries[fragment.kind] = registry
    return Catalog(tuple(registries.values()))
