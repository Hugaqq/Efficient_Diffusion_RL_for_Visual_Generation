"""Frozen, import-safe semantic configs for the GRPO-family modules.

Only algorithm mathematics and selection semantics belong here.  Forward and
decode microbatch sizes, storage placement, precision, and optimizer resources
are execution policy and are deliberately rejected by these parsers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from visual_rl.algorithms.modules.descriptor import (
    AlgorithmBlueprint,
    AlgorithmSlotBlueprint,
)
from visual_rl.core.contracts import (
    AlgorithmComponentResolution,
    AlgorithmComponentRole,
    AlgorithmRequirements,
    ComputePrecision,
    DeclaredContract,
    DistributionMode,
    GroupingKind,
    LatentLayout,
    LikelihoodSemantics,
    MediaKind,
    PredictionType,
    ReferenceRequirement,
    TaskKind,
    TimeCoordinate,
    TrainingMode,
    TrajectoryKind,
    TransitionKind,
)
from visual_rl.core.immutable import FrozenMapping
from visual_rl.data.samples.trajectory import BranchTopology

__all__ = (
    "FlashGRPOAlgorithmConfig",
    "FlowGRPOAlgorithmConfig",
    "TempFlowGRPOAlgorithmConfig",
)

_ALL_TASKS = (TaskKind.T2I, TaskKind.T2V, TaskKind.I2V)
_ALL_MEDIA = (MediaKind.IMAGE, MediaKind.VIDEO)
_DENSE_LATENTS = (LatentLayout.BCHW, LatentLayout.BCTHW)
_PRECISIONS = (
    ComputePrecision.FP32,
    ComputePrecision.FP16,
    ComputePrecision.BF16,
)
_TRAINING_MODES = (TrainingMode.LORA, TrainingMode.FULL)
_DISTRIBUTION_MODES = (DistributionMode.SINGLE,)
_OBJECTIVE_IDENTITY = (
    "visual_rl.algorithms.optimization.objective:ClippedSurrogateObjective.compute@1"
)
_RESOURCE_POLICY_KEYS = frozenset(
    {
        "decode_microbatch_size",
        "decoded_media_layout",
        "forward_microbatch_size",
        "precision",
        "row_microbatch_size",
        "trajectory_storage_device",
        "transition_window_size",
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
    resource_keys = tuple(sorted(set(values) & _RESOURCE_POLICY_KEYS))
    if resource_keys:
        raise ValueError(
            f"{label} params contain execution resource policy: {list(resource_keys)}"
        )
    unknown = tuple(sorted(set(values) - allowed))
    if unknown:
        raise ValueError(f"unknown {label} params: {list(unknown)}")
    return dict(values)


def _positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite(
    name: str,
    value: object,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    observed = float(value)
    if positive and observed <= 0.0:
        raise ValueError(f"{name} must be positive")
    if non_negative and observed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return observed


def _credit_values(
    *,
    normalization_key: str,
    normalization: str,
    advantage_epsilon: object,
    advantage_std_domain: str,
    clip_range: object,
    advantage_clip: object,
) -> tuple[float, float, float]:
    del normalization_key, normalization
    epsilon = _finite("advantage_epsilon", advantage_epsilon, positive=True)
    clip = _finite("clip_range", clip_range, positive=True)
    if clip >= 1.0:
        raise ValueError("clip_range must be smaller than one")
    if advantage_std_domain not in {"group", "batch"}:
        raise ValueError("advantage_std_domain must be group or batch")
    advantage_clip_value = _finite(
        "advantage_clip",
        advantage_clip,
        positive=True,
    )
    return epsilon, clip, advantage_clip_value


def _branch_topology_payload(kind: str, exploration_count: int) -> dict[str, Any]:
    topology = (
        BranchTopology.every_policy_timestep(exploration_count)
        if kind == "every_policy_timestep"
        else BranchTopology.single_point_branch_ablation(exploration_count)
    )
    return topology.to_payload()


def _slot(
    role: AlgorithmComponentRole,
    *,
    family: str,
    resolution: AlgorithmComponentResolution,
    component_id: str | None,
    params: Mapping[str, Any] = FrozenMapping(),
) -> AlgorithmSlotBlueprint:
    return AlgorithmSlotBlueprint(
        role=role,
        implementation_family=family,
        resolution=resolution,
        component_id=component_id,
        params=FrozenMapping(params),
    )


def _blueprint(
    *,
    component_id: str,
    beta: float,
    dynamics_params: Mapping[str, Any],
    rollout_component_id: str,
    rollout_params: Mapping[str, Any],
    credit_component_id: str,
    credit_params: Mapping[str, Any],
) -> AlgorithmBlueprint:
    return AlgorithmBlueprint(
        algorithm_component_id=component_id,
        slots=(
            _slot(
                AlgorithmComponentRole.TRAINER,
                family="grpo-trainer",
                resolution=AlgorithmComponentResolution.ALGORITHM_DEFAULT,
                component_id="grpo",
            ),
            _slot(
                AlgorithmComponentRole.DYNAMICS,
                family="flow-sde",
                resolution=AlgorithmComponentResolution.MODEL_BOUND,
                component_id=None,
                params=dynamics_params,
            ),
            _slot(
                AlgorithmComponentRole.ROLLOUT,
                family="trajectory-rollout",
                resolution=AlgorithmComponentResolution.ALGORITHM_DEFAULT,
                component_id=rollout_component_id,
                params=rollout_params,
            ),
            _slot(
                AlgorithmComponentRole.CREDIT,
                family="grpo-credit",
                resolution=AlgorithmComponentResolution.ALGORITHM_DEFAULT,
                component_id=credit_component_id,
                params=credit_params,
            ),
        ),
        objective_identity=_OBJECTIVE_IDENTITY,
        beta=beta,
    )


def _requirements(
    *,
    trajectory_kind: TrajectoryKind,
    grouping: GroupingKind,
    likelihoods: tuple[LikelihoodSemantics, ...],
    reference_requirement: ReferenceRequirement,
    reference_required: bool,
    required_transition_features: tuple[str, ...],
    required_policy_metadata_fields: tuple[str, ...],
) -> AlgorithmRequirements:
    return AlgorithmRequirements(
        accepted_tasks=_ALL_TASKS,
        accepted_media=_ALL_MEDIA,
        accepted_latent_layouts=_DENSE_LATENTS,
        accepted_prediction_types=(PredictionType.FLOW,),
        accepted_time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
        accepted_training_modes=_TRAINING_MODES,
        accepted_precisions=_PRECISIONS,
        transition_kind=TransitionKind.SDE,
        trajectory_kind=trajectory_kind,
        grouping=grouping,
        likelihood_semantics=likelihoods,
        accepted_distribution_modes=_DISTRIBUTION_MODES,
        reference_requirement=reference_requirement,
        reference_required=reference_required,
        minimum_group_size=2,
        required_transition_features=required_transition_features,
        required_policy_metadata_fields=required_policy_metadata_fields,
    )


@dataclass(frozen=True, slots=True)
class FlowGRPOAlgorithmConfig:
    """Full-trajectory Flow-GRPO mathematics and rollout semantics."""

    num_steps: int = 28
    beta: float = 0.0
    advantage_normalization: Literal["group"] = "group"
    advantage_epsilon: float = 1.0e-4
    advantage_std_domain: Literal["group", "batch"] = "batch"
    clip_range: float = 1.0e-4
    advantage_clip: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "num_steps", _positive_int("num_steps", self.num_steps)
        )
        beta = _finite("beta", self.beta, non_negative=True)
        object.__setattr__(self, "beta", 0.0 if beta == 0.0 else beta)
        if self.advantage_normalization != "group":
            raise ValueError("Flow-GRPO advantage_normalization must be group")
        epsilon, clip, advantage_clip = _credit_values(
            normalization_key="advantage_normalization",
            normalization=self.advantage_normalization,
            advantage_epsilon=self.advantage_epsilon,
            advantage_std_domain=self.advantage_std_domain,
            clip_range=self.clip_range,
            advantage_clip=self.advantage_clip,
        )
        object.__setattr__(self, "advantage_epsilon", epsilon)
        object.__setattr__(self, "clip_range", clip)
        object.__setattr__(self, "advantage_clip", advantage_clip)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> FlowGRPOAlgorithmConfig:
        del context
        return cls(
            **_strict_values(
                values,
                allowed=frozenset(
                    {
                        "num_steps",
                        "beta",
                        "advantage_normalization",
                        "advantage_epsilon",
                        "advantage_std_domain",
                        "clip_range",
                        "advantage_clip",
                    }
                ),
                label="Flow-GRPO",
            )
        )

    def describe_blueprint(self) -> AlgorithmBlueprint:
        return _blueprint(
            component_id="flow-grpo",
            beta=self.beta,
            dynamics_params={},
            rollout_component_id="full-trajectory",
            rollout_params={"num_steps": self.num_steps},
            credit_component_id="grpo",
            credit_params={
                "advantage_normalization": self.advantage_normalization,
                "advantage_epsilon": self.advantage_epsilon,
                "advantage_std_domain": self.advantage_std_domain,
                "clip_range": self.clip_range,
                "advantage_clip": self.advantage_clip,
            },
        )

    def describe_requirements(self) -> AlgorithmRequirements:
        return _requirements(
            trajectory_kind=TrajectoryKind.FULL,
            grouping=GroupingKind.PROMPT_COMPLETIONS,
            likelihoods=(
                LikelihoodSemantics.EXACT_ENV_ACTION,
                LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE,
            ),
            reference_requirement=ReferenceRequirement.WHEN_BETA_POSITIVE,
            reference_required=self.beta > 0.0,
            required_transition_features=(
                "differentiable_log_prob",
                "replayable",
                "scores_arbitrary_action",
                "stochastic",
            ),
            required_policy_metadata_fields=(),
        )

    def describe_contract(self) -> DeclaredContract:
        return DeclaredContract(
            component_kind="algorithm",
            component_id="flow-grpo",
            algorithm=self.describe_requirements(),
        )


@dataclass(frozen=True, slots=True)
class TempFlowGRPOAlgorithmConfig:
    """TempFlow branching, selection, and transition-weight semantics."""

    num_steps: int = 28
    branch_count: int = 6
    branch_topology: Literal[
        "every_policy_timestep",
        "single_point_branch_ablation",
    ] = "every_policy_timestep"
    branch_step_policy: Literal["uniform_intermediate"] | None = None
    branch_step_index: int | None = None
    advantage_normalization: Literal["branches"] = "branches"
    advantage_epsilon: float = 1.0e-4
    advantage_std_domain: Literal["group"] = "group"
    clip_range: float = 1.0e-4
    advantage_clip: float = 5.0
    transition_noise_scale: float = 2.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_steps",
            _positive_int("num_steps", self.num_steps, minimum=2),
        )
        object.__setattr__(
            self,
            "branch_count",
            _positive_int("branch_count", self.branch_count, minimum=2),
        )
        if self.branch_topology not in {
            "every_policy_timestep",
            "single_point_branch_ablation",
        }:
            raise ValueError("unsupported TempFlow branch_topology")
        if self.branch_topology == "every_policy_timestep":
            if (
                self.branch_step_policy is not None
                or self.branch_step_index is not None
            ):
                raise ValueError(
                    "every_policy_timestep does not accept branch-step selection"
                )
        else:
            if self.branch_step_policy != "uniform_intermediate":
                raise ValueError(
                    "single_point_branch_ablation requires "
                    "branch_step_policy=uniform_intermediate"
                )
            if self.branch_step_index is not None and (
                type(self.branch_step_index) is not int
                or not 0 <= self.branch_step_index < self.num_steps - 1
            ):
                raise ValueError("branch_step_index must precede the final step")
        if self.advantage_normalization != "branches":
            raise ValueError("TempFlow advantage_normalization must be branches")
        if self.advantage_std_domain != "group":
            raise ValueError("TempFlow advantage_std_domain must be group")
        epsilon, clip, advantage_clip = _credit_values(
            normalization_key="advantage_normalization",
            normalization=self.advantage_normalization,
            advantage_epsilon=self.advantage_epsilon,
            advantage_std_domain=self.advantage_std_domain,
            clip_range=self.clip_range,
            advantage_clip=self.advantage_clip,
        )
        object.__setattr__(self, "advantage_epsilon", epsilon)
        object.__setattr__(self, "clip_range", clip)
        object.__setattr__(self, "advantage_clip", advantage_clip)
        object.__setattr__(
            self,
            "transition_noise_scale",
            _finite(
                "transition_noise_scale",
                self.transition_noise_scale,
                positive=True,
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> TempFlowGRPOAlgorithmConfig:
        del context
        resolved = _strict_values(
            values,
            allowed=frozenset(
                {
                    "num_steps",
                    "branch_count",
                    "branch_topology",
                    "branch_step_policy",
                    "branch_step_index",
                    "advantage_normalization",
                    "advantage_epsilon",
                    "advantage_std_domain",
                    "clip_range",
                    "advantage_clip",
                    "transition_noise_scale",
                }
            ),
            label="TempFlow-GRPO",
        )
        topology = resolved.get("branch_topology")
        if isinstance(topology, Mapping):
            parsed_topology = BranchTopology.from_payload(topology)
            if "branch_count" in resolved and (
                parsed_topology.exploration_count != resolved["branch_count"]
            ):
                raise ValueError(
                    "branch_topology exploration_count must equal branch_count"
                )
            resolved.setdefault("branch_count", parsed_topology.exploration_count)
            resolved["branch_topology"] = parsed_topology.kind
        return cls(**resolved)

    def describe_blueprint(self) -> AlgorithmBlueprint:
        rollout_params = {
            "num_steps": self.num_steps,
            "branch_count": self.branch_count,
            "branch_topology": _branch_topology_payload(
                self.branch_topology,
                self.branch_count,
            ),
            "branch_step_policy": self.branch_step_policy,
            "branch_step_index": self.branch_step_index,
        }
        return _blueprint(
            component_id="tempflow-grpo",
            beta=0.0,
            dynamics_params={},
            rollout_component_id="branching",
            rollout_params=rollout_params,
            credit_component_id="tempflow",
            credit_params={
                "advantage_normalization": self.advantage_normalization,
                "advantage_epsilon": self.advantage_epsilon,
                "advantage_std_domain": self.advantage_std_domain,
                "clip_range": self.clip_range,
                "advantage_clip": self.advantage_clip,
                "transition_noise_scale": self.transition_noise_scale,
            },
        )

    def describe_requirements(self) -> AlgorithmRequirements:
        return _requirements(
            trajectory_kind=TrajectoryKind.BRANCHING,
            grouping=GroupingKind.BRANCHES,
            likelihoods=(LikelihoodSemantics.EXACT_ENV_ACTION,),
            reference_requirement=ReferenceRequirement.NEVER,
            reference_required=False,
            required_transition_features=(
                "branchable",
                "deterministic_ode",
                "differentiable_log_prob",
                "replayable",
                "scores_arbitrary_action",
                "stochastic",
            ),
            required_policy_metadata_fields=("transition_std_dev",),
        )

    def describe_contract(self) -> DeclaredContract:
        return DeclaredContract(
            component_kind="algorithm",
            component_id="tempflow-grpo",
            algorithm=self.describe_requirements(),
        )


@dataclass(frozen=True, slots=True)
class FlashGRPOAlgorithmConfig:
    """Flash single-step selection and rectified credit semantics."""

    num_steps: int = 40
    selected_timestep_policy: Literal["uniform"] = "uniform"
    selected_timestep_index: int | None = None
    candidate_timestep_window: tuple[int, int] | None = (0, 10)
    candidate_timestep_indices: tuple[int, ...] | None = None
    selection_key: Literal["row", "prompt"] = "prompt"
    selection_domain: Literal["single_process", "global_rank_broadcast"] = (
        "single_process"
    )
    normalization: Literal["global"] = "global"
    advantage_epsilon: float = 1.0e-4
    advantage_std_domain: Literal["batch"] = "batch"
    clip_range: float = 0.001
    advantage_clip: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "num_steps", _positive_int("num_steps", self.num_steps)
        )
        if self.selected_timestep_policy != "uniform":
            raise ValueError("selected_timestep_policy must be uniform")
        window = self.candidate_timestep_window
        indices = self.candidate_timestep_indices
        if window is not None and indices is not None:
            raise ValueError(
                "candidate_timestep_window and candidate_timestep_indices "
                "are mutually exclusive"
            )
        if window is not None:
            if (
                type(window) is not tuple
                or len(window) != 2
                or any(type(item) is not int for item in window)
            ):
                raise TypeError("candidate_timestep_window must be two integers")
            start, stop = window
            if start < 0 or stop <= start or start >= self.num_steps:
                raise ValueError("candidate_timestep_window must overlap the schedule")
        if indices is not None:
            if type(indices) is not tuple or not indices:
                raise ValueError("candidate_timestep_indices must be non-empty")
            if any(type(item) is not int for item in indices):
                raise TypeError("candidate_timestep_indices must contain integers")
            if len(indices) != len(set(indices)) or any(
                not 0 <= item < self.num_steps for item in indices
            ):
                raise ValueError("candidate_timestep_indices must be unique and valid")
            object.__setattr__(
                self, "candidate_timestep_indices", tuple(sorted(indices))
            )
        if self.selection_key not in {"row", "prompt"}:
            raise ValueError("selection_key must be row or prompt")
        if self.selection_domain not in {
            "single_process",
            "global_rank_broadcast",
        }:
            raise ValueError("unsupported selection_domain")
        if self.selected_timestep_index is not None and (
            type(self.selected_timestep_index) is not int
            or self.selected_timestep_index not in self.candidate_indices
        ):
            raise ValueError("selected_timestep_index must be a candidate")
        if self.normalization != "global":
            raise ValueError("Flash normalization must be global")
        if self.advantage_std_domain != "batch":
            raise ValueError("Flash advantage_std_domain must be batch")
        epsilon, clip, advantage_clip = _credit_values(
            normalization_key="normalization",
            normalization=self.normalization,
            advantage_epsilon=self.advantage_epsilon,
            advantage_std_domain=self.advantage_std_domain,
            clip_range=self.clip_range,
            advantage_clip=self.advantage_clip,
        )
        object.__setattr__(self, "advantage_epsilon", epsilon)
        object.__setattr__(self, "clip_range", clip)
        object.__setattr__(self, "advantage_clip", advantage_clip)

    @property
    def candidate_indices(self) -> tuple[int, ...]:
        if self.candidate_timestep_indices is not None:
            return self.candidate_timestep_indices
        if self.candidate_timestep_window is None:
            return tuple(range(self.num_steps))
        start, stop = self.candidate_timestep_window
        return tuple(range(start, min(stop, self.num_steps)))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> FlashGRPOAlgorithmConfig:
        del context
        resolved = _strict_values(
            values,
            allowed=frozenset(
                {
                    "num_steps",
                    "selected_timestep_policy",
                    "selected_timestep_index",
                    "candidate_timestep_window",
                    "candidate_timestep_indices",
                    "selection_key",
                    "selection_domain",
                    "normalization",
                    "advantage_epsilon",
                    "advantage_std_domain",
                    "clip_range",
                    "advantage_clip",
                }
            ),
            label="Flash-GRPO",
        )
        for name in ("candidate_timestep_window", "candidate_timestep_indices"):
            if name in resolved and isinstance(resolved[name], list):
                resolved[name] = tuple(resolved[name])
        return cls(**resolved)

    def describe_blueprint(self) -> AlgorithmBlueprint:
        return _blueprint(
            component_id="flash-grpo",
            beta=0.0,
            dynamics_params={
                "profile": "flash",
                "likelihood_semantics": "exact_env_action",
                "replay_target": "sampled_action",
                "stochastic_sampling": True,
            },
            rollout_component_id="single-step",
            rollout_params={
                "num_steps": self.num_steps,
                "selected_timestep_policy": self.selected_timestep_policy,
                "selected_timestep_index": self.selected_timestep_index,
                "candidate_timestep_window": self.candidate_timestep_window,
                "candidate_timestep_indices": self.candidate_timestep_indices,
                "selection_key": self.selection_key,
                "selection_domain": self.selection_domain,
            },
            credit_component_id="flash",
            credit_params={
                "normalization": self.normalization,
                "advantage_epsilon": self.advantage_epsilon,
                "advantage_std_domain": self.advantage_std_domain,
                "clip_range": self.clip_range,
                "advantage_clip": self.advantage_clip,
            },
        )

    def describe_requirements(self) -> AlgorithmRequirements:
        return _requirements(
            trajectory_kind=TrajectoryKind.SINGLE_STEP,
            grouping=GroupingKind.SELECTED_TIMESTEP,
            likelihoods=(LikelihoodSemantics.EXACT_ENV_ACTION,),
            reference_requirement=ReferenceRequirement.NEVER,
            reference_required=False,
            required_transition_features=(
                "deterministic_ode",
                "differentiable_log_prob",
                "replayable",
                "scores_arbitrary_action",
                "stochastic",
            ),
            required_policy_metadata_fields=("rectification_coefficient",),
        )

    def describe_contract(self) -> DeclaredContract:
        return DeclaredContract(
            component_kind="algorithm",
            component_id="flash-grpo",
            algorithm=self.describe_requirements(),
        )
