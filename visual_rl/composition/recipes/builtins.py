"""Canonical typed recipe definitions for the six v0.8 integration routes.

Definitions in this module name only public composition axes and typed plans.
Internal trainer/Dynamics/rollout/credit declarations are deliberately absent:
the compiler derives them from the resolved algorithm blueprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from visual_rl.composition.config.integration import (
    DynamicsConditioningMode,
    DynamicsIntegrationSpec,
)
from visual_rl.composition.config.specs import (
    ExecutionPolicySpec,
    RolloutExecutionPolicySpec,
    TrainingSpec,
)
from visual_rl.core.contracts import (
    ComputePrecision,
    DistributionMode,
    ExecutionTransformPlan,
    LikelihoodSemantics,
    ReplayTarget,
    RewardRouteBinding,
    RewardRouteSpec,
    TrainingMode,
    TrainingParadigm,
)
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict
from visual_rl.data.phase_schedule import (
    PeriodicPhaseSchedule,
    world_r1_release_phase_schedule,
)
from visual_rl.data.source_plan import DatasetSourceSpec, SourcePlanSpec
from visual_rl.errors import ConfigError

__all__ = (
    "BUILTIN_RECIPE_DEFINITIONS",
    "LogicalRewardDefinition",
    "RecipeDefinition",
    "RequestedComponent",
    "apply_recipe_overrides",
    "builtin_recipe_definitions",
    "get_recipe_definition",
)


@dataclass(frozen=True, slots=True)
class RequestedComponent:
    """One unresolved public/integration component selection."""

    alias: str
    params: FrozenMapping

    def __post_init__(self) -> None:
        if (
            not isinstance(self.alias, str)
            or not self.alias
            or self.alias.strip() != self.alias
        ):
            raise ValueError("component alias must be canonical text")
        if not isinstance(self.params, FrozenMapping):
            object.__setattr__(self, "params", FrozenMapping(self.params))


@dataclass(frozen=True, slots=True)
class LogicalRewardDefinition:
    """One logical reward id and its public reward adapter selection."""

    logical_reward_id: str
    component: RequestedComponent

    def __post_init__(self) -> None:
        if (
            not isinstance(self.logical_reward_id, str)
            or not self.logical_reward_id
            or self.logical_reward_id.strip() != self.logical_reward_id
        ):
            raise ValueError("logical_reward_id must be canonical text")
        if not isinstance(self.component, RequestedComponent):
            raise TypeError("component must be a RequestedComponent")


@dataclass(frozen=True, slots=True)
class RecipeDefinition:
    """One versioned typed integration definition before declaration resolution."""

    name: str
    version: int
    fidelity_target: str
    algorithm: RequestedComponent
    model: RequestedComponent
    rewards: tuple[LogicalRewardDefinition, ...]
    source_plan: SourcePlanSpec
    reward_routes: tuple[RewardRouteSpec, ...]
    execution_policy: ExecutionPolicySpec
    training: TrainingSpec
    phase_schedule: PeriodicPhaseSchedule | None
    dynamics_integration: DynamicsIntegrationSpec
    conditioner: RequestedComponent | None
    conditioner_implementation_family: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("recipe name must be non-empty")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("recipe version must be positive")
        if self.fidelity_target not in {
            "paper",
            "reference_release",
            "visualrl_extension",
        }:
            raise ValueError("unsupported fidelity target")
        if not isinstance(self.algorithm, RequestedComponent):
            raise TypeError("algorithm must be a RequestedComponent")
        if not isinstance(self.model, RequestedComponent):
            raise TypeError("model must be a RequestedComponent")
        if type(self.rewards) is not tuple or not self.rewards:
            raise ValueError("recipe rewards must be a non-empty tuple")
        if any(not isinstance(item, LogicalRewardDefinition) for item in self.rewards):
            raise TypeError("rewards must contain LogicalRewardDefinition values")
        rewards = tuple(sorted(self.rewards, key=lambda item: item.logical_reward_id))
        logical_ids = tuple(item.logical_reward_id for item in rewards)
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("logical reward ids must be unique")
        object.__setattr__(self, "rewards", rewards)
        for name, value, expected in (
            ("source_plan", self.source_plan, SourcePlanSpec),
            ("execution_policy", self.execution_policy, ExecutionPolicySpec),
            ("training", self.training, TrainingSpec),
            (
                "dynamics_integration",
                self.dynamics_integration,
                DynamicsIntegrationSpec,
            ),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be a {expected.__name__}")
        if type(self.reward_routes) is not tuple or not self.reward_routes:
            raise ValueError("reward_routes must be a non-empty tuple")
        if any(not isinstance(item, RewardRouteSpec) for item in self.reward_routes):
            raise TypeError("reward_routes must contain RewardRouteSpec values")
        object.__setattr__(
            self,
            "reward_routes",
            tuple(
                sorted(
                    self.reward_routes, key=lambda item: (item.source_id, item.phase_id)
                )
            ),
        )
        if self.phase_schedule is not None and not isinstance(
            self.phase_schedule, PeriodicPhaseSchedule
        ):
            raise TypeError("phase_schedule must be a PeriodicPhaseSchedule or None")
        if self.conditioner is not None and not isinstance(
            self.conditioner, RequestedComponent
        ):
            raise TypeError("conditioner must be a RequestedComponent or None")
        if self.conditioner_implementation_family is not None and (
            not isinstance(self.conditioner_implementation_family, str)
            or not self.conditioner_implementation_family
        ):
            raise ValueError("conditioner_implementation_family must be canonical")
        if (self.conditioner is None) is not (
            self.conditioner_implementation_family is None
        ):
            raise ValueError(
                "conditioner and conditioner_implementation_family must co-exist"
            )
        conditioned = (
            self.dynamics_integration.conditioning
            is DynamicsConditioningMode.CONDITIONED
        )
        if conditioned is (self.conditioner is None):
            raise ValueError(
                "conditioned integration requires exactly one conditioner selection"
            )

    @property
    def definition_id(self) -> str:
        return f"{self.name}_v{self.version}"


def _component(alias: str, **params: Any) -> RequestedComponent:
    return RequestedComponent(alias=alias, params=FrozenMapping(params))


def _upstream_training_defaults() -> TrainingSpec:
    baseline = TrainingSpec.default()
    return replace(
        baseline,
        adamw=replace(baseline.adamw, weight_decay=1.0e-4),
    )


def _execution(
    *,
    group_size: int,
    rollout: RolloutExecutionPolicySpec,
) -> ExecutionPolicySpec:
    return ExecutionPolicySpec(
        training_mode=TrainingMode.LORA,
        distribution_mode=DistributionMode.SINGLE,
        precision=ComputePrecision.BF16,
        group_size=group_size,
        rollout=rollout,
        transform_plan=ExecutionTransformPlan(
            paradigm=TrainingParadigm.COUPLED,
            transforms=(),
        ),
    )


_MODEL_STORAGE = RolloutExecutionPolicySpec(
    forward_microbatch_size=None,
    decode_microbatch_size=None,
    trajectory_storage_device="model",
)
_BRANCH_STORAGE = RolloutExecutionPolicySpec(
    forward_microbatch_size=1,
    decode_microbatch_size=None,
    trajectory_storage_device="model",
)
_VIDEO_STORAGE = RolloutExecutionPolicySpec(
    forward_microbatch_size=1,
    decode_microbatch_size=1,
    trajectory_storage_device="cpu",
)


def _reward_resource(
    *,
    factory_class: str,
    artifact_ref: str,
    protocol: str,
    protocol_version: str,
    semantic_factory_config: Mapping[str, Any],
    allowed_devices: tuple[str, ...],
    allowed_dtypes: tuple[str, ...],
    allowed_worker_domains: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "factory_class": factory_class,
        "artifact_ref": artifact_ref,
        "protocol": protocol,
        "protocol_version": protocol_version,
        "semantic_factory_config": dict(semantic_factory_config),
        "allowed_runtime_policy": {
            "allowed_devices": allowed_devices,
            "allowed_dtypes": allowed_dtypes,
            "allowed_worker_domains": allowed_worker_domains,
        },
    }


_IMAGE_QUALITY_RESOURCE = _reward_resource(
    factory_class="prompt_color_guarded",
    artifact_ref="reward_quality",
    protocol="visual_rl_reward_client",
    protocol_version="v1",
    semantic_factory_config={
        "default_color": "red",
        "luminance_max": 0.95,
        "luminance_min": 0.05,
        "luminance_penalty_weight": 1.0,
        "margin_clip": 0.5,
        "saturation_max": 0.95,
        "saturation_penalty_weight": 1.0,
        "spatial_penalty_weight": 1.0,
        "spatial_std_max": 0.5,
        "spatial_std_min": 0.01,
    },
    allowed_devices=("cpu",),
    allowed_dtypes=("fp32",),
    allowed_worker_domains=("in_process",),
)


def _world_resource(*, factory_class: str, artifact_ref: str) -> dict[str, Any]:
    semantic_factory_config: dict[str, Any] = {
        "server_revision_expectation": "world-r1-8e46b1b63498",
    }
    if factory_class == "reward_general":
        semantic_factory_config["input_selection_policy"] = {
            "schema_version": 1,
            "domain": "video_frame",
            "candidate_indices": "all",
            "selection": "keyed_uniform",
            "sharing": "batch",
            "seed_derivation_schema": "sha256_rejection_v1",
        }
    return _reward_resource(
        factory_class=factory_class,
        artifact_ref=artifact_ref,
        protocol="world_r1_json",
        protocol_version="strict_v2",
        semantic_factory_config=semantic_factory_config,
        allowed_devices=("cpu",),
        allowed_dtypes=("fp32",),
        allowed_worker_domains=("remote",),
    )


_WORLD_GENERAL_RESOURCE = _world_resource(
    factory_class="reward_general",
    artifact_ref="reward_general",
)
_WORLD_3D_RESOURCE = _world_resource(
    factory_class="reward_3d",
    artifact_ref="reward_3d",
)


def _source(source_id: str, selector: str, artifact_ref: str) -> DatasetSourceSpec:
    return DatasetSourceSpec(
        source_id=source_id,
        selector=selector,
        artifact_ref=artifact_ref,
        artifact_kind="file",
        format="text",
    )


def _route(
    source_id: str,
    phase_id: str,
    *logical_reward_ids: str,
) -> RewardRouteSpec:
    return RewardRouteSpec(
        source_id=source_id,
        phase_id=phase_id,
        rewards=tuple(RewardRouteBinding(item, 1.0) for item in logical_reward_ids),
    )


_FLOW_GRPO = RecipeDefinition(
    name="flow_grpo",
    version=1,
    fidelity_target="paper",
    algorithm=_component("flow-grpo", beta=0.004),
    model=_component("sd3", artifact_ref="main"),
    rewards=(
        LogicalRewardDefinition(
            "reward_quality",
            _component("image-quality", resource=_IMAGE_QUALITY_RESOURCE),
        ),
    ),
    source_plan=SourcePlanSpec((_source("main", "prompt-image", "main"),)),
    reward_routes=(_route("main", "main", "reward_quality"),),
    execution_policy=_execution(group_size=8, rollout=_MODEL_STORAGE),
    training=_upstream_training_defaults(),
    phase_schedule=None,
    dynamics_integration=DynamicsIntegrationSpec.unconditioned(),
    conditioner=None,
    conditioner_implementation_family=None,
)


_TEMPFLOW_GRPO = RecipeDefinition(
    name="tempflow_grpo",
    version=1,
    fidelity_target="paper",
    algorithm=_component("tempflow-grpo"),
    model=_component("sd3", artifact_ref="main"),
    rewards=_FLOW_GRPO.rewards,
    source_plan=_FLOW_GRPO.source_plan,
    reward_routes=_FLOW_GRPO.reward_routes,
    execution_policy=_execution(group_size=6, rollout=_BRANCH_STORAGE),
    training=_upstream_training_defaults(),
    phase_schedule=None,
    dynamics_integration=DynamicsIntegrationSpec.unconditioned(),
    conditioner=None,
    conditioner_implementation_family=None,
)


_FLASH_GRPO = RecipeDefinition(
    name="flash_grpo",
    version=1,
    fidelity_target="paper",
    algorithm=_component("flash-grpo"),
    model=_component("wan-t2v", artifact_ref="main", max_sequence_length=512),
    rewards=(
        LogicalRewardDefinition(
            "reward_general",
            _component("video-general", resource=_WORLD_GENERAL_RESOURCE),
        ),
    ),
    source_plan=SourcePlanSpec((_source("main", "prompt-video", "main"),)),
    reward_routes=(_route("main", "main", "reward_general"),),
    execution_policy=_execution(group_size=4, rollout=_VIDEO_STORAGE),
    training=_upstream_training_defaults(),
    phase_schedule=None,
    dynamics_integration=DynamicsIntegrationSpec.unconditioned(),
    conditioner=None,
    conditioner_implementation_family=None,
)


def _world_definition(
    *,
    name: str,
    fidelity_target: str,
    likelihood: LikelihoodSemantics,
    release_schedule: bool,
) -> RecipeDefinition:
    replay = (
        ReplayTarget.SAMPLED_ACTION
        if likelihood is LikelihoodSemantics.EXACT_ENV_ACTION
        else ReplayTarget.CONDITIONED_NEXT
    )
    source_plan = SourcePlanSpec(
        (
            *(
                (_source("dynamic", "world-r1-dynamic-prompts", "dynamic"),)
                if release_schedule
                else ()
            ),
            _source("main", "world-r1-prompts", "main"),
        )
    )
    routes = (
        *(
            (_route("dynamic", "dynamic", "reward_general"),)
            if release_schedule
            else ()
        ),
        _route("main", "main", "reward_general", "reward_3d"),
    )
    return RecipeDefinition(
        name=name,
        version=1,
        fidelity_target=fidelity_target,
        algorithm=_component("flow-grpo", beta=0.0),
        model=_component("wan-t2v", artifact_ref="main", max_sequence_length=512),
        rewards=(
            LogicalRewardDefinition(
                "reward_general",
                _component("world-r1-general", resource=_WORLD_GENERAL_RESOURCE),
            ),
            LogicalRewardDefinition(
                "reward_3d",
                _component("world-r1-3d", resource=_WORLD_3D_RESOURCE),
            ),
        ),
        source_plan=source_plan,
        reward_routes=routes,
        execution_policy=_execution(group_size=4, rollout=_VIDEO_STORAGE),
        training=_upstream_training_defaults(),
        phase_schedule=(
            world_r1_release_phase_schedule() if release_schedule else None
        ),
        dynamics_integration=DynamicsIntegrationSpec(
            conditioning=DynamicsConditioningMode.CONDITIONED,
            likelihood_semantics=likelihood,
            replay_target=replay,
        ),
        conditioner=_component(
            "world-r1-camera",
            wrap_strength=0.35,
            guidance_steps=8,
            noise_downspatial_mode="resize_noise",
        ),
        conditioner_implementation_family="camera-trajectory",
    )


_WORLD_R1_CORE = _world_definition(
    name="world_r1_core",
    fidelity_target="visualrl_extension",
    likelihood=LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE,
    release_schedule=False,
)
_WORLD_R1_RELEASE_SURROGATE = _world_definition(
    name="world_r1_release_surrogate",
    fidelity_target="reference_release",
    likelihood=LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE,
    release_schedule=True,
)
_WORLD_R1_EXACT_ENV_HOOK = _world_definition(
    name="world_r1_exact_env_hook",
    fidelity_target="visualrl_extension",
    likelihood=LikelihoodSemantics.EXACT_ENV_ACTION,
    release_schedule=True,
)


BUILTIN_RECIPE_DEFINITIONS = (
    _FLOW_GRPO,
    _TEMPFLOW_GRPO,
    _FLASH_GRPO,
    _WORLD_R1_CORE,
    _WORLD_R1_RELEASE_SURROGATE,
    _WORLD_R1_EXACT_ENV_HOOK,
)


def builtin_recipe_definitions() -> tuple[RecipeDefinition, ...]:
    return BUILTIN_RECIPE_DEFINITIONS


def get_recipe_definition(definition_id: str) -> RecipeDefinition:
    if not isinstance(definition_id, str) or not definition_id:
        raise ConfigError("recipe id must be a non-empty string", key="recipe")
    for definition in BUILTIN_RECIPE_DEFINITIONS:
        if definition.definition_id == definition_id:
            return definition
    available = ", ".join(item.definition_id for item in BUILTIN_RECIPE_DEFINITIONS)
    raise ConfigError(
        f"unknown recipe {definition_id!r}; available recipes: {available}",
        key="recipe",
    )


def apply_recipe_overrides(
    definition: RecipeDefinition,
    overrides: Mapping[str, Any],
) -> RecipeDefinition:
    """Apply only public-axis and typed-policy overrides, fail closed otherwise."""

    if not isinstance(definition, RecipeDefinition):
        raise TypeError("definition must be a RecipeDefinition")
    if not isinstance(overrides, Mapping):
        raise ConfigError("overrides must be a mapping", key="overrides")
    allowed = {"algorithm", "model", "rewards", "sources", "training", "execution"}
    unknown = tuple(sorted(set(overrides) - allowed))
    if unknown:
        raise ConfigError(
            f"unsupported override roots: {list(unknown)}",
            key=f"overrides.{unknown[0]}",
        )

    result = definition
    if "algorithm" in overrides:
        raw = _exact_mapping(
            overrides["algorithm"],
            allowed={"params"},
            required={"params"},
            key="overrides.algorithm",
        )
        result = replace(
            result,
            algorithm=RequestedComponent(
                result.algorithm.alias,
                FrozenMapping(
                    _deep_merge(
                        to_plain_dict(result.algorithm.params),
                        _plain_mapping(raw["params"], key="overrides.algorithm.params"),
                    )
                ),
            ),
        )
    if "model" in overrides:
        result = replace(
            result,
            model=_override_component(
                result.model,
                overrides["model"],
                key="overrides.model",
            ),
        )
    if "rewards" in overrides:
        raw_rewards = _plain_mapping(overrides["rewards"], key="overrides.rewards")
        known = {item.logical_reward_id for item in result.rewards}
        if set(raw_rewards) != known:
            raise ConfigError(
                "reward overrides must exactly cover existing logical rewards: "
                f"expected={sorted(known)}, observed={sorted(raw_rewards)}",
                key="overrides.rewards",
            )
        result = replace(
            result,
            rewards=tuple(
                LogicalRewardDefinition(
                    item.logical_reward_id,
                    _override_component(
                        item.component,
                        raw_rewards[item.logical_reward_id],
                        key=f"overrides.rewards.{item.logical_reward_id}",
                    ),
                )
                for item in result.rewards
            ),
        )
    if "sources" in overrides:
        raw_sources = _plain_mapping(overrides["sources"], key="overrides.sources")
        known_sources = {item.source_id for item in result.source_plan.sources}
        if set(raw_sources) != known_sources:
            raise ConfigError(
                "source overrides must exactly cover existing source ids: "
                f"expected={sorted(known_sources)}, observed={sorted(raw_sources)}",
                key="overrides.sources",
            )
        sources = []
        for current in result.source_plan.sources:
            raw = _exact_mapping(
                raw_sources[current.source_id],
                allowed={"selector", "artifact_ref", "artifact_kind", "format"},
                required={"selector", "artifact_ref", "artifact_kind", "format"},
                key=f"overrides.sources.{current.source_id}",
            )
            sources.append(DatasetSourceSpec(source_id=current.source_id, **raw))
        result = replace(result, source_plan=SourcePlanSpec(tuple(sources)))
    if "training" in overrides:
        payload = _deep_merge(
            result.training.to_payload(),
            _plain_mapping(overrides["training"], key="overrides.training"),
        )
        try:
            training = TrainingSpec.from_mapping(payload)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"invalid typed training override: {exc}",
                key="overrides.training",
            ) from exc
        result = replace(result, training=training)
    if "execution" in overrides:
        payload = _deep_merge(
            result.execution_policy.to_payload(),
            _plain_mapping(overrides["execution"], key="overrides.execution"),
        )
        try:
            execution = ExecutionPolicySpec.from_mapping(payload)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"invalid typed execution override: {exc}",
                key="overrides.execution",
            ) from exc
        result = replace(result, execution_policy=execution)
    return result


def _override_component(
    current: RequestedComponent,
    value: object,
    *,
    key: str,
) -> RequestedComponent:
    raw = _exact_mapping(
        value,
        allowed={"id", "params"},
        required=set(),
        key=key,
    )
    if not raw:
        raise ConfigError("component override must not be empty", key=key)
    alias = raw.get("id", current.alias)
    if not isinstance(alias, str) or not alias:
        raise ConfigError("component id must be non-empty", key=f"{key}.id")
    if alias != current.alias and "params" not in raw:
        raise ConfigError(
            "changing a component id requires an explicit complete params mapping",
            key=f"{key}.params",
        )
    supplied = _plain_mapping(raw.get("params", {}), key=f"{key}.params")
    params = (
        supplied
        if alias != current.alias
        else _deep_merge(to_plain_dict(current.params), supplied)
    )
    return RequestedComponent(alias, FrozenMapping(params))


def _exact_mapping(
    value: object,
    *,
    allowed: set[str],
    required: set[str],
    key: str,
) -> dict[str, Any]:
    raw = _plain_mapping(value, key=key)
    unknown = tuple(sorted(set(raw) - allowed))
    missing = tuple(sorted(required - set(raw)))
    if unknown or missing:
        raise ConfigError(
            f"invalid exact key set: missing={list(missing)}, unknown={list(unknown)}",
            key=key,
        )
    return raw


def _plain_mapping(value: object, *, key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError("expected a mapping", key=key)
    return {name: to_plain_dict(item) for name, item in value.items()}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = {key: to_plain_dict(value) for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = to_plain_dict(value)
    return merged
