"""Import-safe declarations for GRPO-family credit assignment.

This module owns only immutable configuration and static capability contracts.
Runtime credit implementations are referenced by class path in the catalog and
must not be imported while declarations are resolved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    CatalogFragment,
    ComponentDeclaration,
    ComponentDescriptor,
    CreditContract,
    DeclaredContract,
    GroupingKind,
    LikelihoodSemantics,
    ReferenceRequirement,
    TrajectoryKind,
)

__all__ = (
    "CREDIT_CATALOG_FRAGMENT",
    "FlashCreditConfig",
    "FlashCreditDeclarationProvider",
    "GRPOCreditConfig",
    "GRPOCreditDeclarationProvider",
    "TempFlowCreditConfig",
    "TempFlowCreditDeclarationProvider",
    "credit_catalog_fragment",
)

_POLICY_FIELDS = (
    "active_mask",
    "algorithm_weight",
    "base_advantage",
    "clip_range",
    "reference_kl_weight",
)
_GRPO_CONFIG_KEYS = frozenset(
    {
        "advantage_normalization",
        "advantage_epsilon",
        "advantage_std_domain",
        "clip_range",
        "advantage_clip",
    }
)
_TEMPFLOW_CONFIG_KEYS = frozenset({*_GRPO_CONFIG_KEYS, "transition_noise_scale"})
_FLASH_CONFIG_KEYS = frozenset(
    {
        "normalization",
        "advantage_epsilon",
        "advantage_std_domain",
        "clip_range",
        "advantage_clip",
    }
)


def _strict_values(
    values: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} params must be a mapping")
    unknown = tuple(sorted(set(values) - allowed))
    if unknown:
        raise ValueError(f"unknown {label} params: {list(unknown)}")
    return dict(values)


def _positive_number(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _clip_range(value: object) -> float:
    observed = _positive_number("clip_range", value)
    if observed >= 1.0:
        raise ValueError("clip_range must satisfy 0 < clip_range < 1")
    return observed


def _credit_contract(
    *,
    component_id: str,
    trajectories: tuple[TrajectoryKind, ...],
    grouping: tuple[GroupingKind, ...],
    reference: ReferenceRequirement,
    likelihoods: tuple[LikelihoodSemantics, ...],
    required_policy_metadata_fields: tuple[str, ...],
) -> DeclaredContract:
    return DeclaredContract(
        component_kind="credit",
        component_id=component_id,
        credit=CreditContract(
            accepted_trajectories=trajectories,
            accepted_grouping=grouping,
            minimum_group_size=2,
            requires_current_log_prob=True,
            reference_requirement=reference,
            requires_differentiable_log_prob=True,
            accepted_likelihoods=likelihoods,
            produced_policy_fields=_POLICY_FIELDS,
            required_policy_metadata_fields=required_policy_metadata_fields,
        ),
    )


@dataclass(frozen=True, slots=True)
class GRPOCreditConfig:
    """Credit semantics for full-trajectory GRPO."""

    advantage_normalization: str = "group"
    advantage_epsilon: float = 1.0e-4
    advantage_std_domain: str = "group"
    clip_range: float = 1.0e-4
    advantage_clip: float = 5.0

    def __post_init__(self) -> None:
        if self.advantage_normalization != "group":
            raise ValueError("GRPO advantage_normalization must be group")
        if self.advantage_std_domain not in {"group", "batch"}:
            raise ValueError("GRPO advantage_std_domain must be group or batch")
        object.__setattr__(
            self,
            "advantage_epsilon",
            _positive_number("advantage_epsilon", self.advantage_epsilon),
        )
        object.__setattr__(
            self,
            "clip_range",
            _clip_range(self.clip_range),
        )
        object.__setattr__(
            self,
            "advantage_clip",
            _positive_number("advantage_clip", self.advantage_clip),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> GRPOCreditConfig:
        del context
        return cls(
            **_strict_values(
                values,
                allowed=_GRPO_CONFIG_KEYS,
                label="grpo credit",
            )
        )

    def describe_contract(self) -> DeclaredContract:
        return _credit_contract(
            component_id="grpo",
            trajectories=(TrajectoryKind.FULL,),
            grouping=(GroupingKind.PROMPT_COMPLETIONS,),
            reference=ReferenceRequirement.WHEN_BETA_POSITIVE,
            likelihoods=(
                LikelihoodSemantics.EXACT_ENV_ACTION,
                LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE,
            ),
            required_policy_metadata_fields=(),
        )


@dataclass(frozen=True, slots=True)
class TempFlowCreditConfig:
    """Credit semantics for branch-grouped TempFlow-GRPO."""

    advantage_normalization: str = "branches"
    advantage_epsilon: float = 1.0e-4
    advantage_std_domain: str = "group"
    clip_range: float = 1.0e-4
    advantage_clip: float = 5.0
    transition_noise_scale: float = 2.25

    def __post_init__(self) -> None:
        if self.advantage_normalization != "branches":
            raise ValueError("TempFlow advantage_normalization must be branches")
        if self.advantage_std_domain != "group":
            raise ValueError("TempFlow advantage_std_domain must be group")
        object.__setattr__(
            self,
            "advantage_epsilon",
            _positive_number("advantage_epsilon", self.advantage_epsilon),
        )
        object.__setattr__(
            self,
            "clip_range",
            _clip_range(self.clip_range),
        )
        object.__setattr__(
            self,
            "advantage_clip",
            _positive_number("advantage_clip", self.advantage_clip),
        )
        object.__setattr__(
            self,
            "transition_noise_scale",
            _positive_number(
                "transition_noise_scale",
                self.transition_noise_scale,
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> TempFlowCreditConfig:
        del context
        return cls(
            **_strict_values(
                values,
                allowed=_TEMPFLOW_CONFIG_KEYS,
                label="tempflow credit",
            )
        )

    def describe_contract(self) -> DeclaredContract:
        return _credit_contract(
            component_id="tempflow",
            trajectories=(TrajectoryKind.BRANCHING,),
            grouping=(GroupingKind.BRANCHES,),
            reference=ReferenceRequirement.NEVER,
            likelihoods=(LikelihoodSemantics.EXACT_ENV_ACTION,),
            required_policy_metadata_fields=("transition_std_dev",),
        )


@dataclass(frozen=True, slots=True)
class FlashCreditConfig:
    """Credit semantics for selected-timestep Flash-GRPO."""

    normalization: str = "global"
    advantage_epsilon: float = 1.0e-4
    advantage_std_domain: str = "batch"
    clip_range: float = 0.001
    advantage_clip: float = 5.0

    def __post_init__(self) -> None:
        if self.normalization != "global":
            raise ValueError("Flash normalization must be global")
        if self.advantage_std_domain != "batch":
            raise ValueError("Flash advantage_std_domain must be batch")
        object.__setattr__(
            self,
            "advantage_epsilon",
            _positive_number("advantage_epsilon", self.advantage_epsilon),
        )
        object.__setattr__(
            self,
            "clip_range",
            _clip_range(self.clip_range),
        )
        object.__setattr__(
            self,
            "advantage_clip",
            _positive_number("advantage_clip", self.advantage_clip),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> FlashCreditConfig:
        del context
        return cls(
            **_strict_values(
                values,
                allowed=_FLASH_CONFIG_KEYS,
                label="flash credit",
            )
        )

    def describe_contract(self) -> DeclaredContract:
        return _credit_contract(
            component_id="flash",
            trajectories=(TrajectoryKind.SINGLE_STEP,),
            grouping=(GroupingKind.SELECTED_TIMESTEP,),
            reference=ReferenceRequirement.NEVER,
            likelihoods=(LikelihoodSemantics.EXACT_ENV_ACTION,),
            required_policy_metadata_fields=("rectification_coefficient",),
        )


class GRPOCreditDeclarationProvider:
    """Declare GRPO credit without importing its runtime implementation."""

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
        return ComponentDeclaration(
            config=config,
            declared_contract=config.describe_contract(),
        )


class TempFlowCreditDeclarationProvider:
    """Declare TempFlow credit without importing its runtime implementation."""

    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.optimization.config:TempFlowCreditConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = TempFlowCreditConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config,
            declared_contract=config.describe_contract(),
        )


class FlashCreditDeclarationProvider:
    """Declare Flash credit without importing its runtime implementation."""

    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.optimization.config:FlashCreditConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = FlashCreditConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config,
            declared_contract=config.describe_contract(),
        )


CREDIT_CATALOG_FRAGMENT = CatalogFragment(
    owner="algorithms.optimization",
    kind="credit",
    descriptors=(
        ComponentDescriptor(
            alias="grpo",
            implementation_class_path=(
                "visual_rl.algorithms.optimization.credit:RegisteredGRPOCredit"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.optimization.config:GRPOCreditDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
        ComponentDescriptor(
            alias="tempflow",
            implementation_class_path=(
                "visual_rl.algorithms.optimization.credit:RegisteredTempFlowCredit"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.optimization.config:"
                "TempFlowCreditDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
        ComponentDescriptor(
            alias="flash",
            implementation_class_path=(
                "visual_rl.algorithms.optimization.credit:RegisteredFlashCredit"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.optimization.config:"
                "FlashCreditDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
    ),
)


def credit_catalog_fragment() -> CatalogFragment:
    """Return the immutable credit descriptor contribution."""

    return CREDIT_CATALOG_FRAGMENT
