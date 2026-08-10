"""Static declaration resolution without importing runtime implementations."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from visual_rl.composition.registry.base import (
    Registry,
    RegistryError,
    RegistryIssueCode,
)
from visual_rl.core.contracts.composition import (
    DECLARATION_PROVIDER_ABI,
    ComponentDeclaration,
    ComponentDescriptor,
    ComponentKind,
    DeclarationProvider,
    DeclaredContract,
)
from visual_rl.core.identity import canonical_identity, to_identity_value

__all__ = (
    "DECLARATION_PROVIDER_ABI",
    "ComponentDeclaration",
    "DeclarationProvider",
    "DeclarationResolver",
    "ResolvedComponentDeclaration",
)

_CLASS_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")


@dataclass(frozen=True, slots=True)
class ResolvedComponentDeclaration:
    """Serializable static result retaining no imported provider or callable."""

    kind: ComponentKind
    alias: str
    descriptor: ComponentDescriptor
    config_type_path: str
    config: object
    declared_contract: DeclaredContract

    def __post_init__(self) -> None:
        if self.alias != self.descriptor.alias:
            raise ValueError("resolved alias must match its descriptor")
        if self.descriptor.removed_message is not None:
            raise ValueError("a tombstone cannot produce a declaration")
        if (
            not isinstance(self.config_type_path, str)
            or _CLASS_PATH_RE.fullmatch(self.config_type_path) is None
        ):
            raise ValueError("config_type_path must be a canonical class path")
        observed_config_type = (
            f"{type(self.config).__module__}:{type(self.config).__qualname__}"
        )
        if observed_config_type != self.config_type_path:
            raise ValueError(
                "resolved config type differs from provider CONFIG_TYPE_PATH: "
                f"expected={self.config_type_path!r}, "
                f"observed={observed_config_type!r}"
            )
        if self.declared_contract.component_kind != self.kind:
            raise ValueError("declared contract kind differs from the registry")
        if self.declared_contract.component_id != self.alias:
            raise ValueError("declared contract id differs from the resolved alias")

    @property
    def implementation_class_path(self) -> str:
        value = self.descriptor.implementation_class_path
        if value is None:  # guarded by ComponentDescriptor for active entries
            raise AssertionError("active descriptor has no implementation path")
        return value

    @property
    def declaration_provider_path(self) -> str:
        value = self.descriptor.declaration_provider_path
        if value is None:  # guarded by ComponentDescriptor for active entries
            raise AssertionError("active descriptor has no provider path")
        return value

    def to_identity_payload(self) -> dict[str, object]:
        """Return the complete static declaration provenance manifest."""

        return {
            "schema_version": 1,
            "kind": self.kind,
            "alias": self.alias,
            "implementation_class_path": self.implementation_class_path,
            "declaration_provider_path": self.declaration_provider_path,
            "declaration_provider_abi": (self.descriptor.declaration_provider_abi),
            "config_type_path": self.config_type_path,
            "interface_version": self.descriptor.interface_version,
            "optional_dependencies": self.descriptor.optional_dependencies,
            "config": to_identity_value(self.config),
            "declared_contract": to_identity_value(self.declared_contract),
        }

    @property
    def declaration_id(self) -> str:
        """Identity of provider, parser, config, contract, and runtime target."""

        return canonical_identity(
            "component-declaration.v1",
            self.to_identity_payload(),
        )


class DeclarationResolver:
    """Resolve an alias by importing only its declaration provider."""

    def resolve(
        self,
        registry: Registry,
        alias: str,
        raw_params: Mapping[str, Any],
        *,
        context: object | None = None,
    ) -> ResolvedComponentDeclaration:
        if not isinstance(registry, Registry):
            raise TypeError("registry must be a Registry")
        if not isinstance(raw_params, Mapping):
            raise TypeError("raw_params must be a mapping")
        descriptor = registry.resolve_descriptor(alias)
        if descriptor.declaration_provider_abi != DECLARATION_PROVIDER_ABI:
            raise RegistryError(
                RegistryIssueCode.INVALID_PROVIDER,
                "component declaration requires a specialized resolver for "
                f"provider ABI {descriptor.declaration_provider_abi!r}",
                kind=registry.kind,
                alias=alias,
            )
        implementation_path = descriptor.implementation_class_path
        provider_path = descriptor.declaration_provider_path
        if implementation_path is None or provider_path is None:
            raise AssertionError("active descriptor paths were not validated")
        if provider_path == implementation_path:
            raise RegistryError(
                RegistryIssueCode.INVALID_PROVIDER,
                "declaration provider must be separate from runtime implementation",
                kind=registry.kind,
                alias=alias,
            )

        provider = _load_provider(provider_path, kind=registry.kind, alias=alias)
        config_type_path = _provider_config_type_path(
            provider,
            kind=registry.kind,
            alias=alias,
        )
        declaration_method = provider.declare_component
        try:
            declaration = declaration_method(dict(raw_params), context=context)
        except Exception as exc:
            raise RegistryError(
                RegistryIssueCode.PROVIDER_FAILED,
                f"declaration provider failed for {registry.kind} alias {alias!r}: "
                f"{type(exc).__name__}",
                kind=registry.kind,
                alias=alias,
            ) from exc
        if not isinstance(declaration, ComponentDeclaration):
            raise RegistryError(
                RegistryIssueCode.INVALID_DECLARATION,
                "declaration provider must return ComponentDeclaration",
                kind=registry.kind,
                alias=alias,
            )
        observed_config_type = (
            f"{type(declaration.config).__module__}:"
            f"{type(declaration.config).__qualname__}"
        )
        if observed_config_type != config_type_path:
            raise RegistryError(
                RegistryIssueCode.INVALID_DECLARATION,
                "resolved config type differs from provider CONFIG_TYPE_PATH: "
                f"expected={config_type_path!r}, "
                f"observed={observed_config_type!r}",
                kind=registry.kind,
                alias=alias,
            )
        contract = declaration.declared_contract
        if contract.component_kind != registry.kind or contract.component_id != alias:
            raise RegistryError(
                RegistryIssueCode.INVALID_DECLARATION,
                "provider declaration kind/id differs from the selected alias",
                kind=registry.kind,
                alias=alias,
            )
        return ResolvedComponentDeclaration(
            kind=registry.kind,
            alias=alias,
            descriptor=descriptor,
            config_type_path=config_type_path,
            config=declaration.config,
            declared_contract=contract,
        )


def _load_provider(
    class_path: str,
    *,
    kind: ComponentKind,
    alias: str,
) -> type[DeclarationProvider]:
    return cast(
        "type[DeclarationProvider]",
        _load_provider_for_abi(
            class_path,
            expected_abi=DECLARATION_PROVIDER_ABI,
            declaration_method="declare_component",
            kind=kind,
            alias=alias,
        ),
    )


def _load_provider_for_abi(
    class_path: str,
    *,
    expected_abi: str,
    declaration_method: str,
    kind: ComponentKind,
    alias: str,
) -> type:
    module_name, qualname = class_path.split(":", 1)
    try:
        value: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise RegistryError(
            RegistryIssueCode.PROVIDER_IMPORT_FAILED,
            f"cannot import declaration provider {class_path!r}",
            kind=kind,
            alias=alias,
        ) from exc
    if not isinstance(value, type):
        raise RegistryError(
            RegistryIssueCode.INVALID_PROVIDER,
            "declaration provider path must resolve to a class",
            kind=kind,
            alias=alias,
        )
    if getattr(value, "PROVIDER_ABI", None) != expected_abi:
        raise RegistryError(
            RegistryIssueCode.INVALID_PROVIDER,
            "declaration provider has an unsupported PROVIDER_ABI",
            kind=kind,
            alias=alias,
        )
    if not callable(getattr(value, declaration_method, None)):
        raise RegistryError(
            RegistryIssueCode.INVALID_PROVIDER,
            f"declaration provider must implement {declaration_method}()",
            kind=kind,
            alias=alias,
        )
    return value


def _provider_config_type_path(
    provider: type,
    *,
    kind: ComponentKind,
    alias: str,
) -> str:
    value = getattr(provider, "CONFIG_TYPE_PATH", None)
    if not isinstance(value, str) or _CLASS_PATH_RE.fullmatch(value) is None:
        raise RegistryError(
            RegistryIssueCode.INVALID_PROVIDER,
            "declaration provider must expose a canonical CONFIG_TYPE_PATH",
            kind=kind,
            alias=alias,
        )
    return value
