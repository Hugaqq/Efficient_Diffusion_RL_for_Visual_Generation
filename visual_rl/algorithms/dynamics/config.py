"""Import-safe declarations for model-bound diffusion dynamics.

Numerical kernels and scheduler materialization live in the runtime modules
named by the descriptors below.  Importing this file only parses immutable
configuration and exposes static compatibility contracts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    CatalogFragment,
    ComponentDeclaration,
    ComponentDescriptor,
    DeclaredContract,
    DynamicsContract,
    LatentLayout,
    LikelihoodSemantics,
    PredictionType,
    ReplayTarget,
    TimeCoordinate,
    TransitionKind,
)
from visual_rl.models.scheduler import SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA

__all__ = (
    "DYNAMICS_CATALOG_FRAGMENT",
    "FlowSDEConfig",
    "SD3FlowSDEDeclarationProvider",
    "WanFlowSDEConfig",
    "WanFlowSDEDeclarationProvider",
    "WanFlowSDEProfile",
    "dynamics_catalog_fragment",
)

_EnumT = TypeVar("_EnumT", bound=Enum)


class WanFlowSDEProfile(str, Enum):
    """Name-neutral transition profiles for Wan flow-SDE kernels."""

    STANDARD = "standard"
    FLASH = "flash"
    CONDITIONED = "conditioned"


def _strict_values(
    values: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
    label: str,
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} params must be a mapping")
    unknown = tuple(sorted(set(values) - allowed))
    if unknown:
        raise ValueError(f"unknown {label} params: {list(unknown)}")
    missing = tuple(sorted(required - set(values)))
    if missing:
        raise ValueError(f"missing {label} params: {list(missing)}")
    return dict(values)


def _enum_value(name: str, value: object, enum_type: type[_EnumT]) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {allowed}") from error


@dataclass(frozen=True, slots=True)
class FlowSDEConfig:
    """SD3 flow-SDE equation parameters, excluding bound schedule state."""

    noise_level: float = 0.7

    def __post_init__(self) -> None:
        if (
            isinstance(self.noise_level, bool)
            or not isinstance(self.noise_level, (int, float))
            or not math.isfinite(float(self.noise_level))
            or float(self.noise_level) <= 0.0
        ):
            raise ValueError("noise_level must be a finite positive number")
        object.__setattr__(self, "noise_level", float(self.noise_level))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> FlowSDEConfig:
        del context
        return cls(
            **_strict_values(
                values,
                allowed=frozenset({"noise_level"}),
                label="flow-sde",
            )
        )

    def describe_contract(self) -> DeclaredContract:
        return DeclaredContract(
            component_kind="dynamics",
            component_id="flow-sde",
            dynamics=DynamicsContract(
                accepted_latent_layouts=(LatentLayout.BCHW,),
                accepted_prediction_types=(PredictionType.FLOW,),
                accepted_time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
                accepted_transition_dtypes=("float32",),
                transition_kind=TransitionKind.SDE,
                stochastic=True,
                exposes_mean_std=True,
                scores_arbitrary_action=True,
                differentiable_log_prob=True,
                replayable=True,
                branchable=True,
                supports_deterministic_ode=True,
                log_prob_reduction="latent_mean.v1",
                supported_likelihoods=(LikelihoodSemantics.EXACT_ENV_ACTION,),
                produced_policy_metadata_fields=("transition_std_dev",),
                accepted_scheduler_blueprint_schemas=(
                    SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA,
                ),
                accepted_model_binding_families=("sd3.flow-sde.v1",),
                produced_replay_state_schema_id="sd3.schedule-replay.v1",
            ),
        )


@dataclass(frozen=True, slots=True)
class WanFlowSDEConfig:
    """Wan kernel profile plus explicit typed score/replay semantics.

    The ordinary Flow-GRPO route uses ``STANDARD`` and contains no World-R1
    name.  ``CONDITIONED`` describes camera-hook-compatible transition math;
    its exact-action extension and its post-hook surrogate are deliberately
    separate, closed pairs rather than stringly-typed score targets.
    """

    profile: WanFlowSDEProfile
    likelihood_semantics: LikelihoodSemantics
    replay_target: ReplayTarget
    stochastic_sampling: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.profile, WanFlowSDEProfile):
            raise TypeError("profile must be a WanFlowSDEProfile")
        if not isinstance(self.likelihood_semantics, LikelihoodSemantics):
            raise TypeError("likelihood_semantics must be a LikelihoodSemantics")
        if not isinstance(self.replay_target, ReplayTarget):
            raise TypeError("replay_target must be a ReplayTarget")
        if type(self.stochastic_sampling) is not bool:
            raise TypeError("stochastic_sampling must be bool")

        pair = (self.likelihood_semantics, self.replay_target)
        exact_action = (
            LikelihoodSemantics.EXACT_ENV_ACTION,
            ReplayTarget.SAMPLED_ACTION,
        )
        conditioned_surrogate = (
            LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE,
            ReplayTarget.CONDITIONED_NEXT,
        )
        if self.profile in {WanFlowSDEProfile.STANDARD, WanFlowSDEProfile.FLASH}:
            if pair != exact_action:
                raise ValueError(
                    f"{self.profile.value} Wan profile requires exact_env_action "
                    "likelihood with sampled_action replay"
                )
        elif pair not in {exact_action, conditioned_surrogate}:
            raise ValueError(
                "conditioned Wan profile requires either exact_env_action with "
                "sampled_action replay or post_hook_base_density_surrogate with "
                "conditioned_next replay"
            )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> WanFlowSDEConfig:
        del context
        resolved = _strict_values(
            values,
            allowed=frozenset(
                {
                    "profile",
                    "likelihood_semantics",
                    "replay_target",
                    "stochastic_sampling",
                }
            ),
            required=frozenset({"profile", "likelihood_semantics", "replay_target"}),
            label="wan-flow-sde",
        )
        resolved["profile"] = _enum_value(
            "profile", resolved["profile"], WanFlowSDEProfile
        )
        resolved["likelihood_semantics"] = _enum_value(
            "likelihood_semantics",
            resolved["likelihood_semantics"],
            LikelihoodSemantics,
        )
        resolved["replay_target"] = _enum_value(
            "replay_target", resolved["replay_target"], ReplayTarget
        )
        return cls(**resolved)

    def describe_contract(self) -> DeclaredContract:
        metadata = (
            ("rectification_coefficient",)
            if self.profile is WanFlowSDEProfile.FLASH
            else ()
        )
        return DeclaredContract(
            component_kind="dynamics",
            component_id="wan-flow-sde",
            dynamics=DynamicsContract(
                accepted_latent_layouts=(LatentLayout.BCTHW,),
                accepted_prediction_types=(PredictionType.FLOW,),
                accepted_time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
                accepted_transition_dtypes=("float32",),
                transition_kind=TransitionKind.SDE,
                stochastic=self.stochastic_sampling,
                exposes_mean_std=True,
                scores_arbitrary_action=True,
                differentiable_log_prob=True,
                replayable=True,
                branchable=False,
                supports_deterministic_ode=True,
                log_prob_reduction="latent_mean.v1",
                supported_likelihoods=(self.likelihood_semantics,),
                produced_policy_metadata_fields=metadata,
                accepted_scheduler_blueprint_schemas=(
                    SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA,
                ),
                accepted_model_binding_families=("wan.flow-sde.v1",),
                produced_replay_state_schema_id="wan.schedule-replay.v1",
            ),
        )


class SD3FlowSDEDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.dynamics.config:FlowSDEConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = FlowSDEConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config, declared_contract=config.describe_contract()
        )


class WanFlowSDEDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.dynamics.config:WanFlowSDEConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = WanFlowSDEConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config, declared_contract=config.describe_contract()
        )


DYNAMICS_CATALOG_FRAGMENT = CatalogFragment(
    owner="algorithms.dynamics",
    kind="dynamics",
    descriptors=(
        ComponentDescriptor(
            alias="flow-sde",
            implementation_class_path=(
                "visual_rl.algorithms.dynamics.sd3_flow_sde:RegisteredSD3FlowSDE"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.dynamics.config:SD3FlowSDEDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
        ComponentDescriptor(
            alias="wan-flow-sde",
            implementation_class_path=(
                "visual_rl.algorithms.dynamics.wan_flow_sde:RegisteredWanFlowSDE"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.dynamics.config:WanFlowSDEDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
    ),
)


def dynamics_catalog_fragment() -> CatalogFragment:
    return DYNAMICS_CATALOG_FRAGMENT
