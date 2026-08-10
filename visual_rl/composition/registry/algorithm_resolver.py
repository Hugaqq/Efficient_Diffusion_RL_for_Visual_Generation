"""Atomic resolution for the specialized public-algorithm declaration ABI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from visual_rl.algorithms.modules.declarations import (
    ALGORITHM_DECLARATION_PROVIDER_ABI,
    AlgorithmDeclaration,
    AlgorithmDeclarationProvider,
)
from visual_rl.algorithms.modules.descriptor import AlgorithmBlueprint
from visual_rl.composition.registry.base import (
    Registry,
    RegistryError,
    RegistryIssueCode,
)
from visual_rl.composition.registry.resolver import (
    ResolvedComponentDeclaration,
    _load_provider_for_abi,
    _provider_config_type_path,
)
from visual_rl.core.contracts import AlgorithmRequirements
from visual_rl.core.identity import canonical_identity

__all__ = (
    "ALGORITHM_DECLARATION_PROVIDER_ABI",
    "AlgorithmDeclarationResolver",
    "ResolvedAlgorithmDeclaration",
)


@dataclass(frozen=True, slots=True)
class ResolvedAlgorithmDeclaration:
    """One algorithm config, contract, and blueprint resolved atomically."""

    component: ResolvedComponentDeclaration
    blueprint: AlgorithmBlueprint

    def __post_init__(self) -> None:
        if not isinstance(self.component, ResolvedComponentDeclaration):
            raise TypeError("component must be a ResolvedComponentDeclaration")
        if self.component.kind != "algorithm":
            raise ValueError("algorithm declaration must wrap an algorithm component")
        if (
            self.component.descriptor.declaration_provider_abi
            != ALGORITHM_DECLARATION_PROVIDER_ABI
        ):
            raise ValueError("algorithm declaration uses the wrong provider ABI")
        if not isinstance(self.blueprint, AlgorithmBlueprint):
            raise TypeError("blueprint must be an AlgorithmBlueprint")
        if self.blueprint.algorithm_component_id != self.component.alias:
            raise ValueError("algorithm blueprint id differs from the resolved alias")
        if not isinstance(
            self.component.declared_contract.algorithm,
            AlgorithmRequirements,
        ):
            raise TypeError("algorithm contract has no AlgorithmRequirements")

    @property
    def alias(self) -> str:
        return self.component.alias

    @property
    def config(self) -> object:
        return self.component.config

    @property
    def requirements(self) -> AlgorithmRequirements:
        """Return the sole requirements value from the component contract."""

        requirements = self.component.declared_contract.algorithm
        assert isinstance(requirements, AlgorithmRequirements)
        return requirements

    @property
    def component_declaration_id(self) -> str:
        return self.component.declaration_id

    def to_identity_payload(self) -> dict[str, object]:
        """Return the complete atomic algorithm declaration provenance."""

        return {
            "schema_version": 1,
            "component_declaration_id": self.component.declaration_id,
            "component_declaration": self.component.to_identity_payload(),
            "blueprint_id": self.blueprint.blueprint_id,
            "blueprint": self.blueprint.to_payload(),
            "requirement_id": self.requirements.requirement_id,
        }

    @property
    def declaration_id(self) -> str:
        return canonical_identity(
            "algorithm-declaration.v1",
            self.to_identity_payload(),
        )


class AlgorithmDeclarationResolver:
    """Resolve a public algorithm through its specialized provider exactly once."""

    def resolve(
        self,
        registry: Registry,
        alias: str,
        raw_params: Mapping[str, Any],
        *,
        context: object | None = None,
    ) -> ResolvedAlgorithmDeclaration:
        if not isinstance(registry, Registry):
            raise TypeError("registry must be a Registry")
        if registry.kind != "algorithm":
            raise ValueError(
                "AlgorithmDeclarationResolver requires an algorithm registry"
            )
        if not isinstance(raw_params, Mapping):
            raise TypeError("raw_params must be a mapping")

        descriptor = registry.resolve_descriptor(alias)
        if descriptor.declaration_provider_abi != ALGORITHM_DECLARATION_PROVIDER_ABI:
            raise RegistryError(
                RegistryIssueCode.INVALID_PROVIDER,
                "algorithm declaration requires provider ABI "
                f"{ALGORITHM_DECLARATION_PROVIDER_ABI!r}",
                kind="algorithm",
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
                kind="algorithm",
                alias=alias,
            )

        provider = cast(
            "type[AlgorithmDeclarationProvider]",
            _load_provider_for_abi(
                provider_path,
                expected_abi=ALGORITHM_DECLARATION_PROVIDER_ABI,
                declaration_method="declare_algorithm",
                kind="algorithm",
                alias=alias,
            ),
        )
        config_type_path = _provider_config_type_path(
            provider,
            kind="algorithm",
            alias=alias,
        )
        try:
            declaration = provider.declare_algorithm(
                dict(raw_params),
                context=context,
            )
        except Exception as exc:
            raise RegistryError(
                RegistryIssueCode.PROVIDER_FAILED,
                f"algorithm declaration provider failed for alias {alias!r}: "
                f"{type(exc).__name__}",
                kind="algorithm",
                alias=alias,
            ) from exc
        if not isinstance(declaration, AlgorithmDeclaration):
            raise RegistryError(
                RegistryIssueCode.INVALID_DECLARATION,
                "algorithm provider must return AlgorithmDeclaration",
                kind="algorithm",
                alias=alias,
            )

        component_declaration = declaration.component
        observed_config_type = (
            f"{type(component_declaration.config).__module__}:"
            f"{type(component_declaration.config).__qualname__}"
        )
        if observed_config_type != config_type_path:
            raise RegistryError(
                RegistryIssueCode.INVALID_DECLARATION,
                "resolved config type differs from provider CONFIG_TYPE_PATH: "
                f"expected={config_type_path!r}, "
                f"observed={observed_config_type!r}",
                kind="algorithm",
                alias=alias,
            )
        contract = component_declaration.declared_contract
        if contract.component_kind != "algorithm" or contract.component_id != alias:
            raise RegistryError(
                RegistryIssueCode.INVALID_DECLARATION,
                "provider declaration kind/id differs from the selected alias",
                kind="algorithm",
                alias=alias,
            )
        if declaration.blueprint.algorithm_component_id != alias:
            raise RegistryError(
                RegistryIssueCode.INVALID_DECLARATION,
                "provider blueprint id differs from the selected alias",
                kind="algorithm",
                alias=alias,
            )

        component = ResolvedComponentDeclaration(
            kind="algorithm",
            alias=alias,
            descriptor=descriptor,
            config_type_path=config_type_path,
            config=component_declaration.config,
            declared_contract=contract,
        )
        return ResolvedAlgorithmDeclaration(
            component=component,
            blueprint=declaration.blueprint,
        )
