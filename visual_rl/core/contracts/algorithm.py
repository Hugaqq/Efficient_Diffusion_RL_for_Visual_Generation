"""Import-safe post-training algorithm capability contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from visual_rl.core.contracts.model import (
    ComputePrecision,
    LatentLayout,
    MediaKind,
    PredictionType,
    TaskKind,
    TimeCoordinate,
    TrainingMode,
)
from visual_rl.core.identity import canonical_identity

__all__ = (
    "CANONICAL_FLOATING_DTYPE_NAMES",
    "AlgorithmComponentResolution",
    "AlgorithmComponentRole",
    "AlgorithmComponentSelection",
    "AlgorithmMaterializationSpec",
    "AlgorithmRequirements",
    "ConditionerContract",
    "CreditContract",
    "DistributionMode",
    "DynamicsContract",
    "GroupingKind",
    "LikelihoodSemantics",
    "ReferenceRequirement",
    "ReplayTarget",
    "RolloutContract",
    "TrainerContract",
    "TrainingParadigm",
    "TrajectoryKind",
    "TransitionKind",
    "TransitionSelectionKind",
    "TransitionSelectionSpec",
)

CANONICAL_FLOATING_DTYPE_NAMES = (
    "bfloat16",
    "float16",
    "float32",
    "float64",
)


class _ValueEnum(str, Enum):
    pass


class TransitionKind(_ValueEnum):
    ODE = "ode"
    SDE = "sde"


class TrajectoryKind(_ValueEnum):
    FULL = "full"
    BRANCHING = "branching"
    SINGLE_STEP = "single_step"


class GroupingKind(_ValueEnum):
    PROMPT_COMPLETIONS = "prompt_completions"
    BRANCHES = "branches"
    SELECTED_TIMESTEP = "selected_timestep"


class LikelihoodSemantics(_ValueEnum):
    EXACT_ENV_ACTION = "exact_env_action"
    POST_HOOK_BASE_DENSITY_SURROGATE = "post_hook_base_density_surrogate"


class ReferenceRequirement(_ValueEnum):
    NEVER = "never"
    WHEN_BETA_POSITIVE = "when_beta_positive"
    ALWAYS = "always"


class DistributionMode(_ValueEnum):
    SINGLE = "single"
    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"


class TrainingParadigm(_ValueEnum):
    COUPLED = "coupled"
    DECOUPLED = "decoupled"
    DISTILLATION = "distillation"


class AlgorithmComponentRole(_ValueEnum):
    TRAINER = "trainer"
    DYNAMICS = "dynamics"
    ROLLOUT = "rollout"
    CONDITIONER = "conditioner"
    CREDIT = "credit"


class AlgorithmComponentResolution(_ValueEnum):
    ALGORITHM_DEFAULT = "algorithm_default"
    MODEL_BOUND = "model_bound"
    INTEGRATION = "integration"


class TransitionSelectionKind(_ValueEnum):
    ALL = "all_stored_policy_actions"
    BRANCH_STEP = "one_branch_step"
    SELECTED_TIMESTEP = "one_selected_timestep"


class ReplayTarget(_ValueEnum):
    SAMPLED_ACTION = "sampled_action"
    CONDITIONED_NEXT = "conditioned_next"


@dataclass(frozen=True)
class DynamicsContract:
    accepted_latent_layouts: tuple[LatentLayout, ...]
    accepted_prediction_types: tuple[PredictionType, ...]
    accepted_time_coordinates: tuple[TimeCoordinate, ...]
    accepted_transition_dtypes: tuple[str, ...]
    transition_kind: TransitionKind
    stochastic: bool
    exposes_mean_std: bool
    scores_arbitrary_action: bool
    differentiable_log_prob: bool
    replayable: bool
    branchable: bool
    supports_deterministic_ode: bool
    log_prob_reduction: str | None
    supported_likelihoods: tuple[LikelihoodSemantics, ...]
    produced_policy_metadata_fields: tuple[str, ...] = ()
    accepted_scheduler_blueprint_schemas: tuple[str, ...] = ()
    accepted_model_binding_families: tuple[str, ...] = ()
    produced_replay_state_schema_id: str | None = None

    def __post_init__(self) -> None:
        _unique("accepted_latent_layouts", self.accepted_latent_layouts)
        _unique("accepted_prediction_types", self.accepted_prediction_types)
        _unique("accepted_time_coordinates", self.accepted_time_coordinates)
        _unique("accepted_transition_dtypes", self.accepted_transition_dtypes)
        _unique("supported_likelihoods", self.supported_likelihoods)
        _unique(
            "produced_policy_metadata_fields",
            self.produced_policy_metadata_fields,
        )
        _scheduler_binding_contract(
            accepted_scheduler_blueprint_schemas=(
                self.accepted_scheduler_blueprint_schemas
            ),
            accepted_model_binding_families=(self.accepted_model_binding_families),
            produced_replay_state_schema_id=(self.produced_replay_state_schema_id),
        )
        if not self.accepted_transition_dtypes:
            raise ValueError("accepted_transition_dtypes must not be empty")
        if any(not isinstance(item, str) for item in self.accepted_transition_dtypes):
            raise TypeError(
                "accepted_transition_dtypes must contain canonical dtype names"
            )
        if self.accepted_transition_dtypes != tuple(
            sorted(self.accepted_transition_dtypes)
        ):
            raise ValueError("accepted_transition_dtypes must be sorted")
        unsupported_dtypes = tuple(
            item
            for item in self.accepted_transition_dtypes
            if item not in CANONICAL_FLOATING_DTYPE_NAMES
        )
        if unsupported_dtypes:
            raise ValueError(
                "accepted_transition_dtypes must contain only canonical floating "
                f"dtype names: {unsupported_dtypes}"
            )
        if self.transition_kind is TransitionKind.ODE and self.stochastic:
            raise ValueError("an ODE transition cannot declare stochastic=True")
        if self.differentiable_log_prob and not self.scores_arbitrary_action:
            raise ValueError(
                "differentiable log-prob requires arbitrary-action scoring"
            )
        if type(self.supports_deterministic_ode) is not bool:
            raise TypeError("supports_deterministic_ode must be bool")
        if any(
            not isinstance(item, str) or not item
            for item in self.produced_policy_metadata_fields
        ):
            raise ValueError(
                "produced_policy_metadata_fields must contain non-empty strings"
            )
        if self.produced_policy_metadata_fields != tuple(
            sorted(self.produced_policy_metadata_fields)
        ):
            raise ValueError("produced_policy_metadata_fields must be sorted")

    @property
    def declares_scheduler_binding(self) -> bool:
        """Whether this Dynamics descriptor carries the complete model ABI."""

        return bool(self.accepted_scheduler_blueprint_schemas)


@dataclass(frozen=True)
class RolloutContract:
    """Static schedule, work-cardinality, and stored-trajectory contract."""

    accepted_tasks: tuple[TaskKind, ...]
    accepted_transitions: tuple[TransitionKind, ...]
    trajectory_kind: TrajectoryKind
    grouping: GroupingKind
    requires_branchable: bool
    requires_deterministic_ode: bool
    required_transition_fields: tuple[str, ...]
    produced_trajectory_fields: tuple[str, ...]
    schedule_step_count: tuple[int, int | None]
    physical_transition_count: tuple[int, int | None]
    stored_policy_transition_count: tuple[int, int | None]

    def __post_init__(self) -> None:
        _unique("accepted_tasks", self.accepted_tasks)
        _unique("accepted_transitions", self.accepted_transitions)
        _unique("required_transition_fields", self.required_transition_fields)
        _unique("produced_trajectory_fields", self.produced_trajectory_fields)
        if type(self.requires_deterministic_ode) is not bool:
            raise TypeError("requires_deterministic_ode must be bool")
        _transition_count_range(
            "physical_transition_count",
            self.physical_transition_count,
        )
        _transition_count_range(
            "stored_policy_transition_count",
            self.stored_policy_transition_count,
        )
        schedule_count = self.schedule_step_count
        _transition_count_range("schedule_step_count", schedule_count)
        schedule_minimum, schedule_maximum = schedule_count
        if schedule_maximum is None or schedule_minimum != schedule_maximum:
            raise ValueError("schedule_step_count must be one exact positive count")
        _, physical_maximum = self.physical_transition_count
        stored_minimum, stored_maximum = self.stored_policy_transition_count
        if physical_maximum is not None and (
            stored_minimum > physical_maximum
            or stored_maximum is None
            or stored_maximum > physical_maximum
        ):
            raise ValueError(
                "stored policy transitions cannot exceed physical transitions"
            )
        if self.trajectory_kind is TrajectoryKind.FULL:
            if self.stored_policy_transition_count != self.physical_transition_count:
                raise ValueError(
                    "full trajectories must store every physical transition"
                )
            if self.physical_transition_count != schedule_count:
                raise ValueError(
                    "full trajectory schedule, physical, and stored counts must match"
                )
        if (
            self.trajectory_kind is TrajectoryKind.SINGLE_STEP
            and self.stored_policy_transition_count != (1, 1)
        ):
            raise ValueError("single-step trajectories must store one transition")
        if (
            self.trajectory_kind is TrajectoryKind.SINGLE_STEP
            and self.physical_transition_count != schedule_count
        ):
            raise ValueError(
                "single-step schedule and physical transition counts must match"
            )


@dataclass(frozen=True)
class ConditionerContract:
    accepted_tasks: tuple[TaskKind, ...]
    accepted_latent_layouts: tuple[LatentLayout, ...]
    payload_type: str
    has_initialize_hook: bool
    has_after_step_hook: bool
    deterministic_given_state: bool
    replay_state_serializable: bool
    independent_of_policy_parameters: bool
    required_modalities: tuple[str, ...] = ()
    provided_output_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _unique("accepted_tasks", self.accepted_tasks)
        _unique("accepted_latent_layouts", self.accepted_latent_layouts)
        _unique("required_modalities", self.required_modalities)
        _unique("provided_output_fields", self.provided_output_fields)
        if not self.payload_type:
            raise ValueError("conditioner payload_type must be non-empty")
        for name in ("required_modalities", "provided_output_fields"):
            values = getattr(self, name)
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"conditioner {name} must be non-empty strings")


@dataclass(frozen=True)
class CreditContract:
    accepted_trajectories: tuple[TrajectoryKind, ...]
    accepted_grouping: tuple[GroupingKind, ...]
    minimum_group_size: int
    requires_current_log_prob: bool
    reference_requirement: ReferenceRequirement
    requires_differentiable_log_prob: bool
    accepted_likelihoods: tuple[LikelihoodSemantics, ...]
    produced_policy_fields: tuple[str, ...]
    required_policy_metadata_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _unique("accepted_trajectories", self.accepted_trajectories)
        _unique("accepted_grouping", self.accepted_grouping)
        _unique("accepted_likelihoods", self.accepted_likelihoods)
        _unique("produced_policy_fields", self.produced_policy_fields)
        _unique(
            "required_policy_metadata_fields",
            self.required_policy_metadata_fields,
        )
        if self.minimum_group_size < 1:
            raise ValueError("minimum_group_size must be positive")
        if any(
            not isinstance(item, str) or not item
            for item in self.required_policy_metadata_fields
        ):
            raise ValueError(
                "required_policy_metadata_fields must contain non-empty strings"
            )
        if self.required_policy_metadata_fields != tuple(
            sorted(self.required_policy_metadata_fields)
        ):
            raise ValueError("required_policy_metadata_fields must be sorted")


@dataclass(frozen=True)
class TrainerContract:
    accepted_training_modes: tuple[TrainingMode, ...]
    accepted_distribution_modes: tuple[DistributionMode, ...]
    required_policy_fields: tuple[str, ...]
    supports_reference_policy: bool

    def __post_init__(self) -> None:
        _unique("accepted_training_modes", self.accepted_training_modes)
        _unique("accepted_distribution_modes", self.accepted_distribution_modes)
        _unique("required_policy_fields", self.required_policy_fields)


@dataclass(frozen=True)
class AlgorithmRequirements:
    """Coarse public contract for one complete post-training algorithm."""

    accepted_tasks: tuple[TaskKind, ...]
    accepted_media: tuple[MediaKind, ...]
    accepted_latent_layouts: tuple[LatentLayout, ...]
    accepted_prediction_types: tuple[PredictionType, ...]
    accepted_time_coordinates: tuple[TimeCoordinate, ...]
    accepted_training_modes: tuple[TrainingMode, ...]
    accepted_precisions: tuple[ComputePrecision, ...]
    transition_kind: TransitionKind
    trajectory_kind: TrajectoryKind
    grouping: GroupingKind
    likelihood_semantics: tuple[LikelihoodSemantics, ...]
    accepted_distribution_modes: tuple[DistributionMode, ...]
    reference_requirement: ReferenceRequirement
    reference_required: bool
    minimum_group_size: int
    required_condition_payload_types: tuple[str, ...] = ()
    required_transition_features: tuple[str, ...] = ()
    required_policy_metadata_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "accepted_tasks",
            "accepted_media",
            "accepted_latent_layouts",
            "accepted_prediction_types",
            "accepted_time_coordinates",
            "accepted_training_modes",
            "accepted_precisions",
            "likelihood_semantics",
            "accepted_distribution_modes",
            "required_condition_payload_types",
            "required_transition_features",
            "required_policy_metadata_fields",
        ):
            _unique(name, getattr(self, name))
        if not self.accepted_tasks:
            raise ValueError("algorithm accepted_tasks must not be empty")
        if not self.accepted_media:
            raise ValueError("algorithm accepted_media must not be empty")
        if not self.accepted_latent_layouts:
            raise ValueError("algorithm accepted_latent_layouts must not be empty")
        if not self.accepted_prediction_types:
            raise ValueError("algorithm accepted_prediction_types must not be empty")
        if not self.accepted_time_coordinates:
            raise ValueError("algorithm accepted_time_coordinates must not be empty")
        if not self.accepted_training_modes:
            raise ValueError("algorithm accepted_training_modes must not be empty")
        if not self.accepted_precisions:
            raise ValueError("algorithm accepted_precisions must not be empty")
        if not self.likelihood_semantics:
            raise ValueError("algorithm likelihood_semantics must not be empty")
        if not self.accepted_distribution_modes:
            raise ValueError("algorithm accepted_distribution_modes must not be empty")
        if type(self.reference_required) is not bool:
            raise TypeError("algorithm reference_required must be bool")
        if (
            self.reference_requirement is ReferenceRequirement.NEVER
            and self.reference_required
        ):
            raise ValueError("reference_required conflicts with NEVER requirement")
        if (
            self.reference_requirement is ReferenceRequirement.ALWAYS
            and not self.reference_required
        ):
            raise ValueError("ALWAYS reference requirement must be required")
        if type(self.minimum_group_size) is not int or self.minimum_group_size < 1:
            raise ValueError("algorithm minimum_group_size must be positive")
        for name in (
            "required_condition_payload_types",
            "required_transition_features",
            "required_policy_metadata_fields",
        ):
            values = getattr(self, name)
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"algorithm {name} must contain non-empty strings")
            if values != tuple(sorted(values)):
                raise ValueError(f"algorithm {name} must be sorted")

    @classmethod
    def from_contract(
        cls,
        contract: AlgorithmRequirements,
    ) -> AlgorithmRequirements:
        """Normalize a legacy aggregate contract without copying its type."""

        if not isinstance(contract, cls):
            raise TypeError("contract must be an AlgorithmRequirements")
        return contract

    @property
    def requirement_id(self) -> str:
        return canonical_identity("algorithm-requirements.v1", self)


@dataclass(frozen=True, slots=True)
class AlgorithmComponentSelection:
    """One resolved role carrying the full static declaration identity."""

    role: AlgorithmComponentRole
    selected_component_id: str
    component_declaration_id: str
    implementation_family: str
    resolution: AlgorithmComponentResolution

    def __post_init__(self) -> None:
        if not isinstance(self.role, AlgorithmComponentRole):
            raise TypeError("role must be an AlgorithmComponentRole")
        if not isinstance(self.resolution, AlgorithmComponentResolution):
            raise TypeError("resolution must be an AlgorithmComponentResolution")
        for name in (
            "selected_component_id",
            "implementation_family",
        ):
            _canonical_text(name, getattr(self, name))
        _namespaced_digest(
            "component_declaration_id",
            self.component_declaration_id,
            "component-declaration.v1",
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "role": self.role.value,
            "selected_component_id": self.selected_component_id,
            "component_declaration_id": self.component_declaration_id,
            "implementation_family": self.implementation_family,
            "resolution": self.resolution.value,
        }


@dataclass(frozen=True, slots=True)
class TransitionSelectionSpec:
    """Name-independent selection of replay-visible policy transitions."""

    kind: TransitionSelectionKind
    policy: str
    selected_transition_count: int
    fixed_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TransitionSelectionKind):
            raise TypeError("kind must be a TransitionSelectionKind")
        _canonical_text("policy", self.policy)
        if (
            type(self.selected_transition_count) is not int
            or self.selected_transition_count < 1
        ):
            raise ValueError("selected_transition_count must be positive")
        if self.fixed_index is not None and (
            type(self.fixed_index) is not int or self.fixed_index < 0
        ):
            raise ValueError("fixed_index must be a non-negative integer or None")
        if self.kind is TransitionSelectionKind.ALL:
            if self.policy != "all" or self.fixed_index is not None:
                raise ValueError("ALL selection requires policy='all' and no index")
        elif self.selected_transition_count != 1:
            raise ValueError("branch/single-step selection must select one transition")

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "policy": self.policy,
            "selected_transition_count": self.selected_transition_count,
            "fixed_index": self.fixed_index,
        }


@dataclass(frozen=True, slots=True)
class AlgorithmMaterializationSpec:
    """Canonical compiler projection consumed by algorithm materialization."""

    algorithm_component_id: str
    blueprint_id: str
    requirement_id: str
    components: tuple[AlgorithmComponentSelection, ...]
    objective_identity: str
    execution_policy_id: str
    trajectory_kind: TrajectoryKind
    grouping: GroupingKind
    schedule_step_count: int
    physical_transition_count: int
    stored_policy_transition_count: int
    transition_selection: TransitionSelectionSpec
    replay_target: ReplayTarget
    likelihood_semantics: LikelihoodSemantics
    reference_requirement: ReferenceRequirement
    requires_reference_statistics: bool
    beta: float
    inner_epochs: int
    optimizer_updates_per_iteration: int
    required_trajectory_fields: tuple[str, ...]
    required_policy_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "algorithm_component_id",
            "objective_identity",
        ):
            _canonical_text(name, getattr(self, name))
        _namespaced_digest("blueprint_id", self.blueprint_id, "algorithm-blueprint.v1")
        _namespaced_digest(
            "requirement_id",
            self.requirement_id,
            "algorithm-requirements.v1",
        )
        _namespaced_digest(
            "execution_policy_id",
            self.execution_policy_id,
            "execution-policy.v1",
        )
        if type(self.components) is not tuple or any(
            not isinstance(item, AlgorithmComponentSelection)
            for item in self.components
        ):
            raise TypeError(
                "components must contain AlgorithmComponentSelection values"
            )
        role_order = {role: index for index, role in enumerate(AlgorithmComponentRole)}
        components = tuple(
            sorted(self.components, key=lambda item: role_order[item.role])
        )
        roles = tuple(item.role for item in components)
        if len(roles) != len(set(roles)):
            raise ValueError("algorithm component roles must be unique")
        required_roles = {
            AlgorithmComponentRole.TRAINER,
            AlgorithmComponentRole.DYNAMICS,
            AlgorithmComponentRole.ROLLOUT,
            AlgorithmComponentRole.CREDIT,
        }
        missing = tuple(sorted(role.value for role in required_roles - set(roles)))
        if missing:
            raise ValueError(
                f"algorithm materialization is missing roles {list(missing)}"
            )
        expected_resolution = {
            AlgorithmComponentRole.TRAINER: (
                AlgorithmComponentResolution.ALGORITHM_DEFAULT
            ),
            AlgorithmComponentRole.DYNAMICS: AlgorithmComponentResolution.MODEL_BOUND,
            AlgorithmComponentRole.ROLLOUT: (
                AlgorithmComponentResolution.ALGORITHM_DEFAULT
            ),
            AlgorithmComponentRole.CONDITIONER: (
                AlgorithmComponentResolution.INTEGRATION
            ),
            AlgorithmComponentRole.CREDIT: (
                AlgorithmComponentResolution.ALGORITHM_DEFAULT
            ),
        }
        invalid_resolution = tuple(
            item.role.value
            for item in components
            if item.resolution is not expected_resolution[item.role]
        )
        if invalid_resolution:
            raise ValueError(
                "component resolution violates algorithm ownership for roles "
                f"{list(invalid_resolution)}"
            )
        object.__setattr__(self, "components", components)
        for name, expected_type in (
            ("trajectory_kind", TrajectoryKind),
            ("grouping", GroupingKind),
            ("replay_target", ReplayTarget),
            ("likelihood_semantics", LikelihoodSemantics),
            ("reference_requirement", ReferenceRequirement),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")
        for name in (
            "schedule_step_count",
            "physical_transition_count",
            "stored_policy_transition_count",
            "inner_epochs",
            "optimizer_updates_per_iteration",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stored_policy_transition_count > self.physical_transition_count:
            raise ValueError("stored transitions cannot exceed physical transitions")
        if self.trajectory_kind is TrajectoryKind.FULL and not (
            self.schedule_step_count
            == self.physical_transition_count
            == self.stored_policy_transition_count
        ):
            raise ValueError(
                "full trajectory schedule/physical/stored counts must match"
            )
        if self.trajectory_kind is TrajectoryKind.SINGLE_STEP and not (
            self.physical_transition_count == self.schedule_step_count
            and self.stored_policy_transition_count == 1
        ):
            raise ValueError(
                "single-step physical count must match schedule and store one action"
            )
        if not isinstance(self.transition_selection, TransitionSelectionSpec):
            raise TypeError("transition_selection must be a TransitionSelectionSpec")
        if (
            self.transition_selection.selected_transition_count
            != self.stored_policy_transition_count
        ):
            raise ValueError("transition selection count must equal stored transitions")
        allowed_selection = {
            TrajectoryKind.FULL: (TransitionSelectionKind.ALL,),
            TrajectoryKind.BRANCHING: (
                TransitionSelectionKind.ALL,
                TransitionSelectionKind.BRANCH_STEP,
            ),
            TrajectoryKind.SINGLE_STEP: (TransitionSelectionKind.SELECTED_TIMESTEP,),
        }[self.trajectory_kind]
        if self.transition_selection.kind not in allowed_selection:
            raise ValueError("transition selection differs from trajectory kind")
        if (
            self.transition_selection.fixed_index is not None
            and self.transition_selection.fixed_index >= self.schedule_step_count
        ):
            raise ValueError("transition selection fixed_index exceeds schedule")
        expected_grouping = {
            TrajectoryKind.FULL: GroupingKind.PROMPT_COMPLETIONS,
            TrajectoryKind.BRANCHING: GroupingKind.BRANCHES,
            TrajectoryKind.SINGLE_STEP: GroupingKind.SELECTED_TIMESTEP,
        }[self.trajectory_kind]
        if self.grouping is not expected_grouping:
            raise ValueError("grouping differs from trajectory kind")
        expected_replay = {
            LikelihoodSemantics.EXACT_ENV_ACTION: ReplayTarget.SAMPLED_ACTION,
            LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE: (
                ReplayTarget.CONDITIONED_NEXT
            ),
        }[self.likelihood_semantics]
        if self.replay_target is not expected_replay:
            raise ValueError("replay target differs from likelihood semantics")
        if type(self.requires_reference_statistics) is not bool:
            raise TypeError("requires_reference_statistics must be bool")
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, (int, float))
            or not math.isfinite(float(self.beta))
            or float(self.beta) < 0.0
        ):
            raise ValueError("beta must be finite and non-negative")
        beta = float(self.beta)
        object.__setattr__(self, "beta", 0.0 if beta == 0.0 else beta)
        expected_reference = (
            self.reference_requirement is ReferenceRequirement.ALWAYS
            or (
                self.reference_requirement is ReferenceRequirement.WHEN_BETA_POSITIVE
                and self.beta > 0.0
            )
        )
        if self.requires_reference_statistics is not expected_reference:
            raise ValueError("reference-statistics flag differs from requirement/beta")
        if self.reference_requirement is ReferenceRequirement.NEVER and self.beta > 0.0:
            raise ValueError("positive beta conflicts with NEVER reference requirement")
        for name in ("required_trajectory_fields", "required_policy_fields"):
            _canonical_string_tuple(name, getattr(self, name))

    @property
    def spec_id(self) -> str:
        return canonical_identity(
            "algorithm-materialization-spec.v1",
            self.to_payload(),
        )

    @property
    def algorithm_semantics_id(self) -> str:
        """Identify algorithm facts independently of the bound execution policy."""

        return canonical_identity(
            "algorithm-semantics.v1",
            self.to_algorithm_identity_payload(),
        )

    def to_algorithm_identity_payload(self) -> dict[str, object]:
        """Return algorithm-owned identity facts without execution geometry."""

        payload = self.to_payload()
        del payload["execution_policy_id"]
        return payload

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "algorithm_component_id": self.algorithm_component_id,
            "blueprint_id": self.blueprint_id,
            "requirement_id": self.requirement_id,
            "components": tuple(item.to_payload() for item in self.components),
            "objective_identity": self.objective_identity,
            "execution_policy_id": self.execution_policy_id,
            "trajectory_kind": self.trajectory_kind.value,
            "grouping": self.grouping.value,
            "schedule_step_count": self.schedule_step_count,
            "physical_transition_count": self.physical_transition_count,
            "stored_policy_transition_count": self.stored_policy_transition_count,
            "transition_selection": self.transition_selection.to_payload(),
            "replay_target": self.replay_target.value,
            "likelihood_semantics": self.likelihood_semantics.value,
            "reference_requirement": self.reference_requirement.value,
            "requires_reference_statistics": self.requires_reference_statistics,
            "beta": self.beta,
            "inner_epochs": self.inner_epochs,
            "optimizer_updates_per_iteration": self.optimizer_updates_per_iteration,
            "required_trajectory_fields": self.required_trajectory_fields,
            "required_policy_fields": self.required_policy_fields,
        }


def _transition_count_range(
    name: str,
    value: tuple[int, int | None],
) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{name} must be a (minimum, maximum) tuple")
    minimum, maximum = value
    if type(minimum) is not int or minimum <= 0:
        raise ValueError(f"{name} minimum must be a positive integer")
    if maximum is not None and (type(maximum) is not int or maximum < minimum):
        raise ValueError(f"invalid {name} range")


def _unique(name: str, values: tuple[object, ...]) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


_SCHEMA_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


def _scheduler_binding_contract(
    *,
    accepted_scheduler_blueprint_schemas: tuple[str, ...],
    accepted_model_binding_families: tuple[str, ...],
    produced_replay_state_schema_id: str | None,
) -> None:
    """Validate the complete load-before Model-to-Dynamics binding ABI."""

    for name, values in (
        (
            "accepted_scheduler_blueprint_schemas",
            accepted_scheduler_blueprint_schemas,
        ),
        (
            "accepted_model_binding_families",
            accepted_model_binding_families,
        ),
    ):
        _unique(name, values)
        if values != tuple(sorted(values)):
            raise ValueError(f"{name} must be sorted")
        if any(
            not isinstance(item, str) or _SCHEMA_ID.fullmatch(item) is None
            for item in values
        ):
            raise ValueError(f"{name} must contain canonical schema identifiers")

    values_present = (
        bool(accepted_scheduler_blueprint_schemas),
        bool(accepted_model_binding_families),
        produced_replay_state_schema_id is not None,
    )
    if not any(values_present):
        return
    if not all(values_present):
        raise ValueError("dynamics scheduler binding fields must be declared together")
    if (
        not isinstance(produced_replay_state_schema_id, str)
        or _SCHEMA_ID.fullmatch(produced_replay_state_schema_id) is None
    ):
        raise ValueError(
            "produced_replay_state_schema_id must be a canonical schema identifier"
        )


def _canonical_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    if "\r" in value or "\n" in value:
        raise ValueError(f"{name} must not contain line breaks")
    return value


def _canonical_string_tuple(name: str, values: tuple[str, ...]) -> None:
    _unique(name, values)
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if values != tuple(sorted(values)):
        raise ValueError(f"{name} must be sorted")


def _namespaced_digest(name: str, value: object, namespace: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        re.escape(namespace) + r":[0-9a-f]{64}",
        value,
    ):
        raise ValueError(f"{name} must be a {namespace} namespaced SHA-256")
    return value
