"""Canonical M2.5 production recipe identity contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from visual_rl.composition.compatibility.identity import CompatibilitySnapshot
from visual_rl.composition.config.integration import (
    DynamicsConditioningMode,
    DynamicsIntegrationSpec,
    ModelBoundDynamicsProjection,
    bind_model_bound_dynamics_declaration,
)
from visual_rl.composition.config.specs import ExecutionPolicySpec, TrainingSpec
from visual_rl.composition.registry.algorithm_resolver import (
    ResolvedAlgorithmDeclaration,
)
from visual_rl.composition.registry.resolver import ResolvedComponentDeclaration
from visual_rl.core.contracts.algorithm import (
    AlgorithmComponentResolution,
    AlgorithmComponentRole,
    AlgorithmMaterializationSpec,
    CreditContract,
    DynamicsContract,
    RolloutContract,
    TrainerContract,
    TrajectoryKind,
)
from visual_rl.core.contracts.model import ModelDescriptorContract
from visual_rl.core.contracts.reward import RewardPlanSpec
from visual_rl.core.identity import canonical_identity
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict
from visual_rl.data.phase_schedule import PeriodicPhaseSchedule
from visual_rl.data.source_plan import SourceContentBinding, SourcePlanSpec

__all__ = (
    "FidelityTarget",
    "MaterializedRecipe",
    "ResolvedRecipe",
    "ResolvedSlotDeclaration",
)


FidelityTarget: TypeAlias = Literal[
    "paper",
    "reference_release",
    "visualrl_extension",
]

_FIDELITY_TARGETS = frozenset({"paper", "reference_release", "visualrl_extension"})
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REWARD_SLOT_PREFIX = "rewards."
_REQUIRED_INTERNAL_SLOTS = frozenset({"trainer", "dynamics", "rollout", "credit"})
_INTERNAL_SLOT_ORDER = {
    role.value: index for index, role in enumerate(AlgorithmComponentRole)
}
_FILESYSTEM_IDENTITY_KEYS = frozenset(
    {
        "identity_schema",
        "content_policy",
        "node_type",
        "content_sha256",
        "file_count",
        "byte_count",
    }
)
_FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "absolute_path",
        "artifact_location",
        "cache_dir",
        "ca_bundle",
        "client",
        "config_source_id",
        "cwd",
        "endpoint",
        "evidence_label",
        "global_rank",
        "handle",
        "host",
        "hostname",
        "launch_id",
        "local_rank",
        "logging_dir",
        "master_addr",
        "master_port",
        "node_rank",
        "output_dir",
        "override_paths",
        "path",
        "pid",
        "process_index",
        "rank",
        "raw_config",
        "raw_yaml",
        "resume_path",
        "run_dir",
        "semantic_config",
        "source_path",
        "validation_evidence_scope",
        "world_size",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedSlotDeclaration:
    """One logical recipe slot retaining one complete resolved declaration.

    The wrapper owns only ``slot``.  Provider, config, implementation, and
    contract identity remain owned by ``ResolvedComponentDeclaration`` and are
    nested directly instead of being copied into a second identity DTO.
    """

    slot: str
    declaration: ResolvedComponentDeclaration

    def __post_init__(self) -> None:
        _slot(self.slot)
        if not isinstance(self.declaration, ResolvedComponentDeclaration):
            raise TypeError("declaration must be a ResolvedComponentDeclaration")
        if self.slot == "model":
            expected_kind = "model"
        elif self.slot in _INTERNAL_SLOT_ORDER:
            expected_kind = self.slot
        elif self.slot.startswith(_REWARD_SLOT_PREFIX):
            logical_id = self.slot.removeprefix(_REWARD_SLOT_PREFIX)
            _identifier(logical_id, field_name="logical reward slot")
            expected_kind = "reward"
        else:
            raise ValueError(f"unsupported canonical recipe slot {self.slot!r}")
        if self.declaration.kind != expected_kind:
            raise ValueError(
                f"slot {self.slot!r} requires a {expected_kind!r} declaration"
            )
        _reject_launch_or_source_facts(
            self.declaration.to_identity_payload(),
            location=f"slot.{self.slot}.declaration",
        )

    @property
    def component_declaration_id(self) -> str:
        return self.declaration.declaration_id

    @property
    def logical_reward_id(self) -> str:
        if not self.slot.startswith(_REWARD_SLOT_PREFIX):
            raise AttributeError("only reward slots have a logical_reward_id")
        return self.slot.removeprefix(_REWARD_SLOT_PREFIX)

    def to_identity_payload(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "component_declaration_id": self.declaration.declaration_id,
            "declaration": self.declaration.to_identity_payload(),
        }


@dataclass(frozen=True, slots=True)
class ResolvedRecipe:
    """The sole typed, path-free static graph for the M2.5 identity cut."""

    definition_id: str
    name: str
    version: int
    fidelity_target: FidelityTarget
    algorithm: ResolvedAlgorithmDeclaration
    algorithm_spec: AlgorithmMaterializationSpec
    model: ResolvedSlotDeclaration
    internal_components: tuple[ResolvedSlotDeclaration, ...]
    reward_components: tuple[ResolvedSlotDeclaration, ...]
    reward_plan: RewardPlanSpec
    source_plan: SourcePlanSpec
    execution_policy: ExecutionPolicySpec
    training: TrainingSpec
    phase_schedule: PeriodicPhaseSchedule | None
    dynamics_integration: DynamicsIntegrationSpec
    dynamics_projection: ModelBoundDynamicsProjection
    compatibility: CompatibilitySnapshot
    resolved_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.name, field_name="recipe name")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("recipe version must be a positive integer")
        if self.definition_id != f"{self.name}_v{self.version}":
            raise ValueError("definition_id must equal name + '_v' + version")
        if self.fidelity_target not in _FIDELITY_TARGETS:
            raise ValueError("unsupported fidelity_target")
        if not isinstance(self.algorithm, ResolvedAlgorithmDeclaration):
            raise TypeError("algorithm must be a ResolvedAlgorithmDeclaration")
        if not isinstance(self.algorithm_spec, AlgorithmMaterializationSpec):
            raise TypeError("algorithm_spec must be an AlgorithmMaterializationSpec")
        if not isinstance(self.model, ResolvedSlotDeclaration):
            raise TypeError("model must be a ResolvedSlotDeclaration")
        if self.model.slot != "model":
            raise ValueError("model must occupy the canonical 'model' slot")
        if type(self.internal_components) is not tuple or any(
            not isinstance(item, ResolvedSlotDeclaration)
            for item in self.internal_components
        ):
            raise TypeError(
                "internal_components must contain ResolvedSlotDeclaration values"
            )
        if type(self.reward_components) is not tuple or any(
            not isinstance(item, ResolvedSlotDeclaration)
            for item in self.reward_components
        ):
            raise TypeError(
                "reward_components must contain ResolvedSlotDeclaration values"
            )
        internal = tuple(
            sorted(
                self.internal_components,
                key=lambda item: _INTERNAL_SLOT_ORDER.get(item.slot, 1_000),
            )
        )
        rewards = tuple(sorted(self.reward_components, key=lambda item: item.slot))
        _unique_slots(internal, field_name="internal_components")
        _unique_slots(rewards, field_name="reward_components")
        if any(item.slot not in _INTERNAL_SLOT_ORDER for item in internal):
            raise ValueError("internal_components contains a non-internal slot")
        if any(not item.slot.startswith(_REWARD_SLOT_PREFIX) for item in rewards):
            raise ValueError("reward_components contains a non-reward slot")
        object.__setattr__(self, "internal_components", internal)
        object.__setattr__(self, "reward_components", rewards)

        for name, value, expected_type in (
            ("reward_plan", self.reward_plan, RewardPlanSpec),
            ("source_plan", self.source_plan, SourcePlanSpec),
            ("execution_policy", self.execution_policy, ExecutionPolicySpec),
            ("training", self.training, TrainingSpec),
            (
                "dynamics_integration",
                self.dynamics_integration,
                DynamicsIntegrationSpec,
            ),
            (
                "dynamics_projection",
                self.dynamics_projection,
                ModelBoundDynamicsProjection,
            ),
            ("compatibility", self.compatibility, CompatibilitySnapshot),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")
        if self.phase_schedule is not None and not isinstance(
            self.phase_schedule, PeriodicPhaseSchedule
        ):
            raise TypeError("phase_schedule must be a PeriodicPhaseSchedule or None")
        if self.compatibility.status == "invalid":
            raise ValueError("an invalid compatibility result cannot form a recipe")

        _validate_algorithm_graph(self)
        _validate_reward_and_source_graph(self)
        _validate_phase_graph(self)
        payload = self.canonical_semantic_payload()
        _reject_launch_or_source_facts(payload, location="resolved_recipe")
        object.__setattr__(
            self,
            "resolved_fingerprint",
            canonical_identity("resolved-recipe.v2", payload),
        )

    def component(self, slot: str) -> ResolvedSlotDeclaration:
        """Return one exact public/internal/logical slot declaration."""

        _slot(slot)
        if slot == "model":
            return self.model
        for item in (*self.internal_components, *self.reward_components):
            if item.slot == slot:
                return item
        raise KeyError(slot)

    def canonical_semantic_payload(self) -> dict[str, object]:
        """Return every typed static field and no source or launch audit fact."""

        return {
            "schema_version": 2,
            "definition_id": self.definition_id,
            "name": self.name,
            "version": self.version,
            "fidelity_target": self.fidelity_target,
            "algorithm": {
                "declaration_id": self.algorithm.declaration_id,
                "declaration": self.algorithm.to_identity_payload(),
            },
            "algorithm_spec": {
                "spec_id": self.algorithm_spec.spec_id,
                "spec": self.algorithm_spec.to_payload(),
            },
            "model": self.model.to_identity_payload(),
            "internal_components": tuple(
                item.to_identity_payload() for item in self.internal_components
            ),
            "reward_components": tuple(
                item.to_identity_payload() for item in self.reward_components
            ),
            "reward_plan": {
                "plan_id": self.reward_plan.plan_id,
                "plan": self.reward_plan.to_payload(),
            },
            "source_plan": {
                "plan_id": self.source_plan.plan_id,
                "plan": self.source_plan.to_payload(),
            },
            "execution_policy": {
                "policy_id": self.execution_policy.policy_id,
                "policy": self.execution_policy.to_payload(),
            },
            "training": self.training.to_payload(),
            "phase_schedule": (
                None
                if self.phase_schedule is None
                else {
                    "schedule_id": self.phase_schedule.schedule_id,
                    "schedule": self.phase_schedule.to_payload(),
                }
            ),
            "dynamics_integration": {
                "integration_id": self.dynamics_integration.integration_id,
                "integration": self.dynamics_integration.to_payload(),
            },
            "dynamics_projection": {
                "projection_id": self.dynamics_projection.projection_id,
                "projection": self.dynamics_projection.to_payload(),
            },
            "compatibility": self.compatibility.identity_payload(),
        }


@dataclass(frozen=True, slots=True)
class MaterializedRecipe:
    """A resolved graph with exact path-free content identities attached."""

    resolved: ResolvedRecipe
    model_artifact_identity: FrozenMapping
    source_content_binding: SourceContentBinding
    reward_plan: RewardPlanSpec
    code_artifact_identity: FrozenMapping
    recipe_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, ResolvedRecipe):
            raise TypeError("resolved must be a ResolvedRecipe")
        if not isinstance(self.source_content_binding, SourceContentBinding):
            raise TypeError("source_content_binding must be a SourceContentBinding")

        # This check is intentionally first and unconditional once the two typed
        # operands exist.  Source under/over-coverage must fail before recipe_id
        # construction and cannot be hidden by another materialization error.
        self.source_content_binding.validate_against(self.resolved.source_plan)

        _filesystem_artifact_identity(
            self.model_artifact_identity,
            field_name="model_artifact_identity",
            content_policy="all-files.v1",
        )
        if not isinstance(self.reward_plan, RewardPlanSpec):
            raise TypeError("reward_plan must be a RewardPlanSpec")
        _validate_materialized_reward_plan(
            provisional=self.resolved.reward_plan,
            materialized=self.reward_plan,
        )
        _filesystem_artifact_identity(
            self.code_artifact_identity,
            field_name="code_artifact_identity",
            content_policy="python-code.v1",
        )
        payload = self.canonical_semantic_payload()
        _reject_launch_or_source_facts(payload, location="materialized_recipe")
        object.__setattr__(
            self,
            "recipe_id",
            canonical_identity("materialized-recipe.v2", payload),
        )

    def canonical_semantic_payload(self) -> dict[str, object]:
        """Return the exact materialized payload hashed as ``recipe_id``."""

        return {
            "schema_version": 2,
            "resolved_fingerprint": self.resolved.resolved_fingerprint,
            "resolved_recipe": self.resolved.canonical_semantic_payload(),
            "model_artifact_identity": to_plain_dict(self.model_artifact_identity),
            "source_content_binding": self.source_content_binding.to_payload(),
            "reward_plan": {
                "plan_id": self.reward_plan.plan_id,
                "plan": self.reward_plan.to_payload(),
            },
            "code_artifact_identity": to_plain_dict(self.code_artifact_identity),
        }


def _validate_algorithm_graph(recipe: ResolvedRecipe) -> None:
    algorithm = recipe.algorithm
    blueprint = algorithm.blueprint
    requirements = algorithm.requirements
    spec = recipe.algorithm_spec
    execution = recipe.execution_policy
    integration = recipe.dynamics_integration
    projection = recipe.dynamics_projection

    exact_pairs = (
        ("algorithm component", spec.algorithm_component_id, algorithm.alias),
        ("algorithm blueprint", spec.blueprint_id, blueprint.blueprint_id),
        ("algorithm requirements", spec.requirement_id, requirements.requirement_id),
        ("algorithm objective", spec.objective_identity, blueprint.objective_identity),
        ("execution policy", spec.execution_policy_id, execution.policy_id),
        ("trajectory kind", spec.trajectory_kind, requirements.trajectory_kind),
        ("grouping", spec.grouping, requirements.grouping),
        (
            "likelihood semantics",
            spec.likelihood_semantics,
            integration.likelihood_semantics,
        ),
        ("replay target", spec.replay_target, integration.replay_target),
        (
            "reference requirement",
            spec.reference_requirement,
            requirements.reference_requirement,
        ),
        (
            "reference-statistics requirement",
            spec.requires_reference_statistics,
            requirements.reference_required,
        ),
        ("algorithm beta", spec.beta, blueprint.beta),
    )
    for label, observed, expected in exact_pairs:
        if observed != expected:
            raise ValueError(f"{label} differs across declaration/spec projection")
    if integration.likelihood_semantics not in requirements.likelihood_semantics:
        raise ValueError("integration likelihood is not accepted by the algorithm")
    if execution.training_mode not in requirements.accepted_training_modes:
        raise ValueError("execution training mode is not accepted by the algorithm")
    if execution.precision not in requirements.accepted_precisions:
        raise ValueError("execution precision is not accepted by the algorithm")
    if execution.distribution_mode not in requirements.accepted_distribution_modes:
        raise ValueError("execution distribution mode is not accepted by the algorithm")
    if execution.group_size < requirements.minimum_group_size:
        raise ValueError("execution group_size is below the algorithm minimum")

    model_contract = recipe.model.declaration.declared_contract.model
    if not isinstance(model_contract, ModelDescriptorContract):
        raise TypeError("model declaration has no ModelDescriptorContract")
    if execution.training_mode not in model_contract.training_modes:
        raise ValueError("execution training mode is not accepted by the model")
    if execution.precision not in model_contract.supported_precisions:
        raise ValueError("execution precision is not accepted by the model")
    if projection.model_binding_family != model_contract.dynamics_binding_family:
        raise ValueError("Dynamics projection differs from the model binding family")
    if projection.integration_id != integration.integration_id:
        raise ValueError("Dynamics projection differs from the integration identity")

    internal_by_slot = {item.slot: item for item in recipe.internal_components}
    observed_internal = set(internal_by_slot)
    expected_internal = set(_REQUIRED_INTERNAL_SLOTS)
    conditioned = integration.conditioning is DynamicsConditioningMode.CONDITIONED
    if conditioned:
        expected_internal.add("conditioner")
    if observed_internal != expected_internal:
        raise ValueError(
            "internal component slots differ from algorithm/integration roles: "
            f"expected={sorted(expected_internal)}, "
            f"observed={sorted(observed_internal)}"
        )
    selections = {item.role: item for item in spec.components}
    expected_roles = {AlgorithmComponentRole(item) for item in expected_internal}
    if set(selections) != expected_roles:
        raise ValueError("algorithm spec component roles do not exactly cover slots")

    for slot_name, slot_declaration in internal_by_slot.items():
        role = AlgorithmComponentRole(slot_name)
        selection = selections[role]
        declaration = slot_declaration.declaration
        if selection.selected_component_id != declaration.alias:
            raise ValueError(f"{slot_name} selection differs from declaration alias")
        if selection.component_declaration_id != declaration.declaration_id:
            raise ValueError(
                f"{slot_name} selection differs from full declaration identity"
            )
        if role is AlgorithmComponentRole.CONDITIONER:
            if selection.resolution is not AlgorithmComponentResolution.INTEGRATION:
                raise ValueError("conditioner selection must be integration-owned")
            continue
        blueprint_slot = blueprint.slot(role)
        if selection.implementation_family != blueprint_slot.implementation_family:
            raise ValueError(
                f"{slot_name} implementation family differs from blueprint"
            )
        if selection.resolution is not blueprint_slot.resolution:
            raise ValueError(f"{slot_name} resolution differs from blueprint")
        if (
            blueprint_slot.resolution is AlgorithmComponentResolution.ALGORITHM_DEFAULT
            and selection.selected_component_id != blueprint_slot.component_id
        ):
            raise ValueError(f"{slot_name} selection differs from blueprint default")

    dynamics_slot = internal_by_slot["dynamics"]
    dynamics_selection = selections[AlgorithmComponentRole.DYNAMICS]
    expected_dynamics = bind_model_bound_dynamics_declaration(
        projection=projection,
        declaration=dynamics_slot.declaration,
        model=recipe.model.declaration,
        algorithm=algorithm,
        integration=integration,
    )
    if dynamics_selection != expected_dynamics:
        raise ValueError("Dynamics selection differs from the validated projection")
    trainer_contract = internal_by_slot["trainer"].declaration.declared_contract.trainer
    rollout_contract = internal_by_slot["rollout"].declaration.declared_contract.rollout
    dynamics_contract = dynamics_slot.declaration.declared_contract.dynamics
    credit_contract = internal_by_slot["credit"].declaration.declared_contract.credit
    if not isinstance(trainer_contract, TrainerContract):
        raise TypeError("trainer declaration has no TrainerContract")
    if not isinstance(rollout_contract, RolloutContract):
        raise TypeError("rollout declaration has no RolloutContract")
    if not isinstance(dynamics_contract, DynamicsContract):
        raise TypeError("dynamics declaration has no DynamicsContract")
    if not isinstance(credit_contract, CreditContract):
        raise TypeError("credit declaration has no CreditContract")

    if execution.training_mode not in trainer_contract.accepted_training_modes:
        raise ValueError("trainer does not accept the execution training mode")
    if execution.distribution_mode not in trainer_contract.accepted_distribution_modes:
        raise ValueError("trainer does not accept the execution distribution mode")
    if (
        spec.requires_reference_statistics
        and not trainer_contract.supports_reference_policy
    ):
        raise ValueError("trainer cannot provide required reference statistics")
    if rollout_contract.trajectory_kind is not spec.trajectory_kind:
        raise ValueError("rollout trajectory kind differs from algorithm spec")
    if rollout_contract.grouping is not spec.grouping:
        raise ValueError("rollout grouping differs from algorithm spec")
    if requirements.transition_kind not in rollout_contract.accepted_transitions:
        raise ValueError("rollout does not accept the algorithm transition kind")
    for label, count, accepted in (
        ("schedule", spec.schedule_step_count, rollout_contract.schedule_step_count),
        (
            "physical transition",
            spec.physical_transition_count,
            rollout_contract.physical_transition_count,
        ),
        (
            "stored policy transition",
            spec.stored_policy_transition_count,
            rollout_contract.stored_policy_transition_count,
        ),
    ):
        if not _count_in_range(count, accepted):
            raise ValueError(f"{label} count is outside the rollout contract")
    expected_trajectory_fields = tuple(
        sorted(
            set(rollout_contract.required_transition_fields)
            | {spec.replay_target.value}
        )
    )
    if spec.required_trajectory_fields != expected_trajectory_fields:
        raise ValueError(
            "algorithm spec trajectory fields differ from rollout contract"
        )
    if dynamics_contract.transition_kind is not requirements.transition_kind:
        raise ValueError("Dynamics transition kind differs from algorithm requirements")
    if spec.likelihood_semantics not in dynamics_contract.supported_likelihoods:
        raise ValueError("Dynamics does not support the selected likelihood")
    if not set(requirements.required_policy_metadata_fields).issubset(
        dynamics_contract.produced_policy_metadata_fields
    ):
        raise ValueError("Dynamics does not produce required policy metadata")
    if spec.trajectory_kind not in credit_contract.accepted_trajectories:
        raise ValueError("credit component does not accept the trajectory kind")
    if spec.grouping not in credit_contract.accepted_grouping:
        raise ValueError("credit component does not accept the grouping")
    if spec.likelihood_semantics not in credit_contract.accepted_likelihoods:
        raise ValueError("credit component does not accept the likelihood")
    if credit_contract.reference_requirement is not spec.reference_requirement:
        raise ValueError("credit reference requirement differs from algorithm spec")
    if execution.group_size < credit_contract.minimum_group_size:
        raise ValueError("execution group_size is below the credit minimum")
    if (
        credit_contract.required_policy_metadata_fields
        != requirements.required_policy_metadata_fields
    ):
        raise ValueError("credit metadata requirements differ from the algorithm")
    expected_policy_fields = tuple(
        sorted(
            set(trainer_contract.required_policy_fields)
            | set(credit_contract.produced_policy_fields)
        )
    )
    if spec.required_policy_fields != expected_policy_fields:
        raise ValueError("algorithm spec policy fields differ from trainer/credit")

    if spec.trajectory_kind is TrajectoryKind.BRANCHING:
        branch_count = blueprint.slot(AlgorithmComponentRole.ROLLOUT).params.get(
            "branch_count"
        )
        if type(branch_count) is not int or branch_count < 1:
            raise ValueError("branching rollout blueprint requires branch_count")
        if execution.group_size != branch_count:
            raise ValueError(
                "branching execution group_size must equal blueprint branch_count"
            )


def _validate_reward_and_source_graph(recipe: ResolvedRecipe) -> None:
    plan = recipe.reward_plan
    if not plan.provisional:
        raise ValueError("resolved reward_plan must be provisional")
    logical_by_id = {item.logical_reward_id: item for item in plan.logical_rewards}
    declarations = {
        item.logical_reward_id: item.declaration for item in recipe.reward_components
    }
    if set(logical_by_id) != set(declarations):
        raise ValueError(
            "reward component slots must exactly cover logical rewards: "
            f"expected={sorted(logical_by_id)}, observed={sorted(declarations)}"
        )
    for logical_id, logical in logical_by_id.items():
        declaration = declarations[logical_id]
        if logical.component_declaration_id != declaration.declaration_id:
            raise ValueError(
                f"logical reward {logical_id!r} differs from declaration identity"
            )
        if logical.contract != declaration.declared_contract.reward:
            raise ValueError(
                f"logical reward {logical_id!r} differs from declaration contract"
            )

    source_ids = {item.source_id for item in recipe.source_plan.sources}
    routed_source_ids = {item.source_id for item in plan.routes}
    if routed_source_ids != source_ids:
        raise ValueError(
            "reward routes must exactly cover source plan ids: "
            f"expected={sorted(source_ids)}, observed={sorted(routed_source_ids)}"
        )


def _validate_phase_graph(recipe: ResolvedRecipe) -> None:
    source_ids = frozenset(item.source_id for item in recipe.source_plan.sources)
    reward_ids = frozenset(
        item.logical_reward_id for item in recipe.reward_plan.logical_rewards
    )
    route_map = {
        (route.source_id, route.phase_id): frozenset(
            item.logical_reward_id for item in route.rewards
        )
        for route in recipe.reward_plan.routes
    }
    schedule = recipe.phase_schedule
    if schedule is None:
        if len(source_ids) != 1 or len(route_map) != 1:
            raise ValueError(
                "phase_schedule=None requires one source and one source/phase route"
            )
        (source_id, _phase_id), active_rewards = next(iter(route_map.items()))
        if source_id not in source_ids or active_rewards != reward_ids:
            raise ValueError(
                "implicit phase route must exactly cover the sole source and rewards"
            )
        return
    if schedule.known_source_ids != source_ids:
        raise ValueError("phase schedule source ids differ from source_plan")
    if schedule.known_reward_ids != reward_ids:
        raise ValueError("phase schedule reward ids differ from reward_plan")
    scheduled = {
        (phase.source_id, phase.phase_id): frozenset(phase.active_rewards)
        for phase in schedule.phases
    }
    if scheduled != route_map:
        raise ValueError("phase schedule routes differ from reward_plan routes")


def _validate_materialized_reward_plan(
    *,
    provisional: RewardPlanSpec,
    materialized: RewardPlanSpec,
) -> None:
    if not provisional.provisional:
        raise ValueError("resolved reward plan must be provisional")
    if not materialized.materialized:
        raise ValueError("materialized reward_plan must have bound artifacts")
    bindings: dict[str, FrozenMapping] = {}
    for resource in materialized.resources:
        identity = resource.artifact_identity
        if not isinstance(identity, FrozenMapping):
            raise TypeError("materialized reward resource has no artifact identity")
        previous = bindings.get(resource.artifact_ref)
        if previous is not None and previous != identity:
            raise ValueError(
                "one reward artifact ref cannot have multiple content identities"
            )
        bindings[resource.artifact_ref] = identity
    expected = provisional.bind_artifacts(bindings)
    if expected != materialized:
        raise ValueError(
            "materialized reward_plan is not an exact binding of resolved reward_plan"
        )


def _filesystem_artifact_identity(
    value: object,
    *,
    field_name: str,
    content_policy: str,
) -> FrozenMapping:
    if not isinstance(value, FrozenMapping):
        raise TypeError(f"{field_name} must be a FrozenMapping")
    if set(value) != _FILESYSTEM_IDENTITY_KEYS:
        raise ValueError(f"{field_name} has an invalid exact key set")
    if value["identity_schema"] != "filesystem-artifact.v1":
        raise ValueError(f"{field_name} must use filesystem-artifact.v1")
    if value["content_policy"] != content_policy:
        raise ValueError(f"{field_name} must use {content_policy}")
    if value["node_type"] not in {"file", "tree"}:
        raise ValueError(f"{field_name}.node_type must be file or tree")
    digest = value["content_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{field_name}.content_sha256 must be a SHA-256 digest")
    file_count = value["file_count"]
    byte_count = value["byte_count"]
    if type(file_count) is not int or file_count < 0:
        raise ValueError(f"{field_name}.file_count must be non-negative")
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"{field_name}.byte_count must be non-negative")
    if value["node_type"] == "file" and file_count != 1:
        raise ValueError(f"{field_name} file identity must have file_count=1")
    _reject_launch_or_source_facts(value, location=field_name)
    return value


def _reject_launch_or_source_facts(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.lower()
            if normalized in _FORBIDDEN_IDENTITY_KEYS:
                raise ValueError(
                    f"{location}.{key} is source/launch audit data and cannot enter "
                    "recipe identity"
                )
            _reject_launch_or_source_facts(item, location=f"{location}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_launch_or_source_facts(item, location=f"{location}[{index}]")
    elif isinstance(value, Path):
        raise TypeError(f"{location} cannot contain a filesystem Path")


def _count_in_range(value: int, accepted: tuple[int, int | None]) -> bool:
    minimum, maximum = accepted
    return value >= minimum and (maximum is None or value <= maximum)


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical identifier")
    return value


def _slot(value: object) -> str:
    return _identifier(value, field_name="slot")


def _unique_slots(
    values: tuple[ResolvedSlotDeclaration, ...],
    *,
    field_name: str,
) -> None:
    slots = tuple(item.slot for item in values)
    if len(slots) != len(set(slots)):
        raise ValueError(f"{field_name} slots must be unique")
