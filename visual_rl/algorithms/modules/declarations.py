"""Import-safe declaration providers for the public algorithm axis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from visual_rl.algorithms.modules.config import (
    FlashGRPOAlgorithmConfig,
    FlowGRPOAlgorithmConfig,
    TempFlowGRPOAlgorithmConfig,
)
from visual_rl.algorithms.modules.descriptor import AlgorithmBlueprint
from visual_rl.core.contracts import (
    AlgorithmRequirements,
    ComponentDeclaration,
)

__all__ = (
    "ALGORITHM_DECLARATION_PROVIDER_ABI",
    "AlgorithmDeclaration",
    "AlgorithmDeclarationProvider",
    "FlashGRPODeclarationProvider",
    "FlowGRPODeclarationProvider",
    "TempFlowGRPODeclarationProvider",
)

ALGORITHM_DECLARATION_PROVIDER_ABI = "visual-rl.algorithm-declaration-provider.v1"


@dataclass(frozen=True, slots=True)
class AlgorithmDeclaration:
    """One atomic, import-safe declaration of a public algorithm axis."""

    component: ComponentDeclaration
    blueprint: AlgorithmBlueprint

    def __post_init__(self) -> None:
        if not isinstance(self.component, ComponentDeclaration):
            raise TypeError("component must be a ComponentDeclaration")
        if not isinstance(self.blueprint, AlgorithmBlueprint):
            raise TypeError("blueprint must be an AlgorithmBlueprint")
        contract = self.component.declared_contract
        if contract.component_kind != "algorithm" or not isinstance(
            contract.algorithm,
            AlgorithmRequirements,
        ):
            raise ValueError(
                "algorithm declaration component must carry AlgorithmRequirements"
            )
        if self.blueprint.algorithm_component_id != contract.component_id:
            raise ValueError(
                "algorithm blueprint id differs from the component declaration"
            )

    @property
    def requirements(self) -> AlgorithmRequirements:
        """Return the sole requirements value carried by the declared contract."""

        requirements = self.component.declared_contract.algorithm
        assert isinstance(requirements, AlgorithmRequirements)
        return requirements


class AlgorithmDeclarationProvider(Protocol):
    """Specialized provider ABI for one atomic algorithm declaration."""

    PROVIDER_ABI: ClassVar[str]
    CONFIG_TYPE_PATH: ClassVar[str]

    @classmethod
    def declare_algorithm(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> AlgorithmDeclaration:
        """Parse once and return config, requirements, and blueprint together."""


def _algorithm_declaration(
    config: (
        FlowGRPOAlgorithmConfig | TempFlowGRPOAlgorithmConfig | FlashGRPOAlgorithmConfig
    ),
) -> AlgorithmDeclaration:
    return AlgorithmDeclaration(
        component=ComponentDeclaration(
            config=config,
            declared_contract=config.describe_contract(),
        ),
        blueprint=config.describe_blueprint(),
    )


class FlowGRPODeclarationProvider:
    PROVIDER_ABI = ALGORITHM_DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.modules.config:FlowGRPOAlgorithmConfig"

    @classmethod
    def declare_algorithm(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> AlgorithmDeclaration:
        del cls
        config = FlowGRPOAlgorithmConfig.from_mapping(raw_params, context=context)
        return _algorithm_declaration(config)


class TempFlowGRPODeclarationProvider:
    PROVIDER_ABI = ALGORITHM_DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.modules.config:TempFlowGRPOAlgorithmConfig"

    @classmethod
    def declare_algorithm(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> AlgorithmDeclaration:
        del cls
        config = TempFlowGRPOAlgorithmConfig.from_mapping(
            raw_params,
            context=context,
        )
        return _algorithm_declaration(config)


class FlashGRPODeclarationProvider:
    PROVIDER_ABI = ALGORITHM_DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.modules.config:FlashGRPOAlgorithmConfig"

    @classmethod
    def declare_algorithm(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> AlgorithmDeclaration:
        del cls
        config = FlashGRPOAlgorithmConfig.from_mapping(raw_params, context=context)
        return _algorithm_declaration(config)
