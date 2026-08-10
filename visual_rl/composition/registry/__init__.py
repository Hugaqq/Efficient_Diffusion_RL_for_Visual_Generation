"""Import-safe registry, declaration resolver, and catalog assembly."""

from visual_rl.composition.registry.algorithm_resolver import (
    ALGORITHM_DECLARATION_PROVIDER_ABI,
    AlgorithmDeclarationResolver,
    ResolvedAlgorithmDeclaration,
)
from visual_rl.composition.registry.base import (
    COMPONENT_KINDS,
    Registry,
    RegistryError,
    RegistryIssueCode,
)
from visual_rl.composition.registry.catalog import (
    Catalog,
    CatalogFragment,
    build_catalog,
)
from visual_rl.composition.registry.resolver import (
    DECLARATION_PROVIDER_ABI,
    ComponentDeclaration,
    DeclarationProvider,
    DeclarationResolver,
    ResolvedComponentDeclaration,
)

__all__ = (
    "ALGORITHM_DECLARATION_PROVIDER_ABI",
    "COMPONENT_KINDS",
    "DECLARATION_PROVIDER_ABI",
    "AlgorithmDeclarationResolver",
    "Catalog",
    "CatalogFragment",
    "ComponentDeclaration",
    "DeclarationProvider",
    "DeclarationResolver",
    "Registry",
    "RegistryError",
    "RegistryIssueCode",
    "ResolvedAlgorithmDeclaration",
    "ResolvedComponentDeclaration",
    "build_catalog",
)
