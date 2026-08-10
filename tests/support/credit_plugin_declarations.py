"""Import-safe declaration provider for the canonical credit plugin tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from visual_rl.algorithms.optimization.config import GRPOCreditConfig
from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    ComponentDeclaration,
)


class CreditPluginDeclarationProvider:
    """Declare the test plugin without importing its runtime implementation."""

    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.optimization.config:GRPOCreditConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = GRPOCreditConfig.from_mapping(raw_params, context=context)
        contract = replace(config.describe_contract(), component_id="plugin-credit")
        return ComponentDeclaration(config=config, declared_contract=contract)
