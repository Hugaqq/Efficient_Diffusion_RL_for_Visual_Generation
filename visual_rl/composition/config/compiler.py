"""Canonical schema-v2 compiler and sole static recipe authority."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache

from visual_rl.algorithms.catalog import algorithm_domain_catalog_fragments
from visual_rl.algorithms.modules.descriptor import AlgorithmSlotBlueprint
from visual_rl.algorithms.rewards.resource_descriptor import RewardResourceDescriptor
from visual_rl.composition.compatibility import (
    CompatibilitySnapshot,
    ModelAlgorithmMismatch,
    bind_model_algorithm,
    match_model_algorithm_dynamics,
)
from visual_rl.composition.config.bootstrap import bootstrap_recipe_v2
from visual_rl.composition.config.integration import (
    DynamicsProjectionRegistry,
    bind_model_bound_dynamics_declaration,
    default_dynamics_projection_registry,
    project_model_bound_dynamics,
)
from visual_rl.composition.config.source import SourceRecipe
from visual_rl.composition.recipes.builtins import (
    RecipeDefinition,
    apply_recipe_overrides,
    get_recipe_definition,
)
from visual_rl.composition.recipes.schema import (
    ResolvedRecipe,
    ResolvedSlotDeclaration,
)
from visual_rl.composition.registry import (
    AlgorithmDeclarationResolver,
    Catalog,
    DeclarationResolver,
    ResolvedAlgorithmDeclaration,
    ResolvedComponentDeclaration,
    build_catalog,
)
from visual_rl.core.contracts import (
    AlgorithmComponentResolution,
    AlgorithmComponentRole,
    AlgorithmComponentSelection,
    AlgorithmMaterializationSpec,
    BoundPolicyCapabilities,
    ConditionerContract,
    CreditContract,
    DynamicsContract,
    LogicalRewardSpec,
    ModelDescriptorContract,
    RewardContract,
    RewardPlanSpec,
    RewardResourceSpec,
    RolloutContract,
    TrainerContract,
    TrajectoryKind,
    TransitionSelectionKind,
    TransitionSelectionSpec,
)
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict
from visual_rl.data.phase_schedule import PeriodicPhaseSchedule, PhaseDefinition
from visual_rl.errors import ConfigError
from visual_rl.models.catalog import model_catalog_fragment

__all__ = ("compile_recipe_v2", "default_catalog")


@cache
def default_catalog() -> Catalog:
    """Return the immutable union of canonical model and algorithm fragments."""

    return build_catalog(
        (model_catalog_fragment(), *algorithm_domain_catalog_fragments())
    )


def compile_recipe_v2(
    source: SourceRecipe,
    catalog: Catalog | None = None,
    *,
    declaration_resolver: DeclarationResolver | None = None,
    algorithm_declaration_resolver: AlgorithmDeclarationResolver | None = None,
    dynamics_projection_registry: DynamicsProjectionRegistry | None = None,
) -> ResolvedRecipe:
    """Compile one source into the canonical typed graph without runtime imports."""

    if not isinstance(source, SourceRecipe):
        raise TypeError("source must be a SourceRecipe")
    resolved_catalog = default_catalog() if catalog is None else catalog
    if not isinstance(resolved_catalog, Catalog):
        raise TypeError("catalog must be a Catalog or None")
    component_resolver = declaration_resolver or DeclarationResolver()
    algorithm_resolver = (
        algorithm_declaration_resolver or AlgorithmDeclarationResolver()
    )
    if not isinstance(component_resolver, DeclarationResolver):
        raise TypeError("declaration_resolver must be a DeclarationResolver")
    if not isinstance(algorithm_resolver, AlgorithmDeclarationResolver):
        raise TypeError(
            "algorithm_declaration_resolver must be an AlgorithmDeclarationResolver"
        )
    projection_registry = (
        default_dynamics_projection_registry()
        if dynamics_projection_registry is None
        else dynamics_projection_registry
    )
    if not isinstance(projection_registry, DynamicsProjectionRegistry):
        raise TypeError(
            "dynamics_projection_registry must be a "
            "DynamicsProjectionRegistry or None"
        )

    bootstrap = bootstrap_recipe_v2(source)
    try:
        definition = apply_recipe_overrides(
            get_recipe_definition(bootstrap.recipe_id),
            bootstrap.overrides,
        )
    except ConfigError as exc:
        if exc.path is not None:
            raise
        raise ConfigError(str(exc), key=exc.key, path=str(source.path)) from exc

    try:
        return _compile_definition(
            definition,
            catalog=resolved_catalog,
            component_resolver=component_resolver,
            algorithm_resolver=algorithm_resolver,
            projection_registry=projection_registry,
            context=source.context,
        )
    except ModelAlgorithmMismatch as exc:
        codes = ", ".join(item.code for item in exc.mismatches)
        raise ConfigError(
            f"incompatible public model/algorithm axes: {codes}",
            key="overrides.model",
            path=str(source.path),
        ) from exc
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"canonical recipe compilation failed: {exc}",
            key="recipe",
            path=str(source.path),
        ) from exc


def _compile_definition(
    definition: RecipeDefinition,
    *,
    catalog: Catalog,
    component_resolver: DeclarationResolver,
    algorithm_resolver: AlgorithmDeclarationResolver,
    projection_registry: DynamicsProjectionRegistry,
    context: object,
) -> ResolvedRecipe:
    algorithm = algorithm_resolver.resolve(
        catalog.for_kind("algorithm"),
        definition.algorithm.alias,
        definition.algorithm.params,
        context=context,
    )
    model = component_resolver.resolve(
        catalog.for_kind("model"),
        definition.model.alias,
        definition.model.params,
        context=context,
    )
    projection = project_model_bound_dynamics(
        model=model,
        algorithm=algorithm,
        integration=definition.dynamics_integration,
        projector_registry=projection_registry,
    )

    declarations, selections = _resolve_internal_components(
        definition,
        algorithm=algorithm,
        model=model,
        projection=projection,
        catalog=catalog,
        resolver=component_resolver,
        projection_registry=projection_registry,
        context=context,
    )
    reward_declarations = _resolve_rewards(
        definition,
        catalog=catalog,
        resolver=component_resolver,
        context=context,
    )
    reward_plan = _build_reward_plan(definition, reward_declarations)
    algorithm_spec = _build_algorithm_spec(
        definition,
        algorithm=algorithm,
        declarations=declarations,
        selections=selections,
    )
    compatibility = _compatibility_snapshot(
        algorithm=algorithm,
        model=model,
        declarations=declarations,
        reward_declarations=reward_declarations,
        definition=definition,
    )
    return ResolvedRecipe(
        definition_id=definition.definition_id,
        name=definition.name,
        version=definition.version,
        fidelity_target=definition.fidelity_target,  # type: ignore[arg-type]
        algorithm=algorithm,
        algorithm_spec=algorithm_spec,
        model=ResolvedSlotDeclaration("model", model),
        internal_components=tuple(
            ResolvedSlotDeclaration(role.value, declarations[role])
            for role in AlgorithmComponentRole
            if role in declarations
        ),
        reward_components=tuple(
            ResolvedSlotDeclaration(f"rewards.{logical_id}", declaration)
            for logical_id, declaration in reward_declarations
        ),
        reward_plan=reward_plan,
        source_plan=definition.source_plan,
        execution_policy=definition.execution_policy,
        training=definition.training,
        phase_schedule=_materialize_phase_schedule(
            definition,
            reward_plan=reward_plan,
        ),
        dynamics_integration=definition.dynamics_integration,
        dynamics_projection=projection,
        compatibility=compatibility,
    )


def _materialize_phase_schedule(
    definition: RecipeDefinition,
    *,
    reward_plan: RewardPlanSpec,
) -> PeriodicPhaseSchedule:
    """Return the sole typed runtime phase fact emitted by the compiler.

    An explicit schedule is already a fully validated immutable value.  The
    only implicit form accepted by the canonical compiler is the unambiguous
    one-source, one-source/phase-route case whose route covers every logical
    reward exactly.  Runtime consumers therefore never reconstruct phase
    semantics from a recipe or raw configuration.
    """

    if not isinstance(definition, RecipeDefinition):
        raise TypeError("definition must be a RecipeDefinition")
    if not isinstance(reward_plan, RewardPlanSpec):
        raise TypeError("reward_plan must be a RewardPlanSpec")
    if definition.phase_schedule is not None:
        return definition.phase_schedule

    source_ids = tuple(source.source_id for source in definition.source_plan.sources)
    routes = reward_plan.routes
    if len(source_ids) != 1 or len(routes) != 1:
        raise ValueError(
            "phase_schedule=None requires exactly one source and one "
            "source/phase reward route"
        )
    route = routes[0]
    if route.source_id != source_ids[0]:
        raise ValueError("implicit phase route must select the sole source")
    active_rewards = route.logical_reward_ids
    known_rewards = tuple(
        reward.logical_reward_id for reward in reward_plan.logical_rewards
    )
    if frozenset(active_rewards) != frozenset(known_rewards):
        raise ValueError("implicit phase route must exactly cover all logical rewards")
    return PeriodicPhaseSchedule(
        phases=(
            PhaseDefinition(
                phase_id=route.phase_id,
                start_offset=0,
                end_offset=1,
                source_id=route.source_id,
                active_rewards=active_rewards,
            ),
        ),
        known_source_ids=frozenset(source_ids),
        known_reward_ids=frozenset(known_rewards),
    )


def _resolve_internal_components(
    definition: RecipeDefinition,
    *,
    algorithm: ResolvedAlgorithmDeclaration,
    model: ResolvedComponentDeclaration,
    projection,
    catalog: Catalog,
    resolver: DeclarationResolver,
    projection_registry: DynamicsProjectionRegistry,
    context: object,
) -> tuple[
    dict[AlgorithmComponentRole, ResolvedComponentDeclaration],
    dict[AlgorithmComponentRole, AlgorithmComponentSelection],
]:
    declarations: dict[AlgorithmComponentRole, ResolvedComponentDeclaration] = {}
    selections: dict[AlgorithmComponentRole, AlgorithmComponentSelection] = {}
    for slot in algorithm.blueprint.slots:
        if slot.role is AlgorithmComponentRole.DYNAMICS:
            declaration = resolver.resolve(
                catalog.for_kind("dynamics"),
                projection.component_id,
                projection.params,
                context=context,
            )
            selection = bind_model_bound_dynamics_declaration(
                projection=projection,
                declaration=declaration,
                model=model,
                algorithm=algorithm,
                integration=definition.dynamics_integration,
                projector_registry=projection_registry,
            )
        else:
            component_id = slot.component_id
            if component_id is None:
                raise ValueError(f"{slot.role.value} blueprint has no component id")
            raw_params = to_plain_dict(slot.params)
            declaration = resolver.resolve(
                catalog.for_kind(slot.role.value),
                component_id,
                raw_params,
                context=context,
            )
            selection = _blueprint_selection(slot, declaration)
        declarations[slot.role] = declaration
        selections[slot.role] = selection

    if definition.conditioner is not None:
        declaration = resolver.resolve(
            catalog.for_kind("conditioner"),
            definition.conditioner.alias,
            definition.conditioner.params,
            context=context,
        )
        family = definition.conditioner_implementation_family
        if family is None:  # guarded by RecipeDefinition
            raise AssertionError("conditioner family is missing")
        declarations[AlgorithmComponentRole.CONDITIONER] = declaration
        selections[AlgorithmComponentRole.CONDITIONER] = AlgorithmComponentSelection(
            role=AlgorithmComponentRole.CONDITIONER,
            selected_component_id=declaration.alias,
            component_declaration_id=declaration.declaration_id,
            implementation_family=family,
            resolution=AlgorithmComponentResolution.INTEGRATION,
        )
    return declarations, selections


def _blueprint_selection(
    slot: AlgorithmSlotBlueprint,
    declaration: ResolvedComponentDeclaration,
) -> AlgorithmComponentSelection:
    if slot.component_id != declaration.alias:
        raise ValueError(f"{slot.role.value} declaration differs from blueprint")
    return AlgorithmComponentSelection(
        role=slot.role,
        selected_component_id=declaration.alias,
        component_declaration_id=declaration.declaration_id,
        implementation_family=slot.implementation_family,
        resolution=slot.resolution,
    )


def _resolve_rewards(
    definition: RecipeDefinition,
    *,
    catalog: Catalog,
    resolver: DeclarationResolver,
    context: object,
) -> tuple[tuple[str, ResolvedComponentDeclaration], ...]:
    return tuple(
        (
            item.logical_reward_id,
            resolver.resolve(
                catalog.for_kind("reward"),
                item.component.alias,
                item.component.params,
                context=context,
            ),
        )
        for item in definition.rewards
    )


def _build_reward_plan(
    definition: RecipeDefinition,
    declarations: tuple[tuple[str, ResolvedComponentDeclaration], ...],
) -> RewardPlanSpec:
    resource_by_descriptor: dict[FrozenMapping, RewardResourceSpec] = {}
    logical_rewards: list[LogicalRewardSpec] = []
    for logical_id, declaration in declarations:
        resource = getattr(declaration.config, "resource", None)
        if not isinstance(resource, RewardResourceDescriptor):
            raise TypeError(
                f"reward {logical_id!r} config has no RewardResourceDescriptor"
            )
        descriptor = FrozenMapping(resource.to_payload())
        resource_spec = resource_by_descriptor.setdefault(
            descriptor,
            RewardResourceSpec(descriptor=descriptor),
        )
        contract = declaration.declared_contract.reward
        if not isinstance(contract, RewardContract):
            raise TypeError(f"reward {logical_id!r} has no RewardContract")
        logical_rewards.append(
            LogicalRewardSpec(
                logical_reward_id=logical_id,
                component_declaration_id=declaration.declaration_id,
                resource_identity=resource_spec.resource_identity,
                contract=contract,
            )
        )
    return RewardPlanSpec(
        resources=tuple(resource_by_descriptor.values()),
        logical_rewards=tuple(logical_rewards),
        routes=definition.reward_routes,
    )


def _build_algorithm_spec(
    definition: RecipeDefinition,
    *,
    algorithm: ResolvedAlgorithmDeclaration,
    declarations: Mapping[AlgorithmComponentRole, ResolvedComponentDeclaration],
    selections: Mapping[AlgorithmComponentRole, AlgorithmComponentSelection],
) -> AlgorithmMaterializationSpec:
    rollout = _required_contract(
        declarations[AlgorithmComponentRole.ROLLOUT].declared_contract.rollout,
        RolloutContract,
        "rollout",
    )
    trainer = _required_contract(
        declarations[AlgorithmComponentRole.TRAINER].declared_contract.trainer,
        TrainerContract,
        "trainer",
    )
    credit = _required_contract(
        declarations[AlgorithmComponentRole.CREDIT].declared_contract.credit,
        CreditContract,
        "credit",
    )
    schedule_count = _exact_count(rollout.schedule_step_count, "schedule_step_count")
    physical_count = _exact_count(
        rollout.physical_transition_count,
        "physical_transition_count",
    )
    stored_count = _exact_count(
        rollout.stored_policy_transition_count,
        "stored_policy_transition_count",
    )
    selection = _transition_selection(
        algorithm.blueprint.slot(AlgorithmComponentRole.ROLLOUT),
        algorithm.requirements.trajectory_kind,
        stored_count,
    )
    return AlgorithmMaterializationSpec(
        algorithm_component_id=algorithm.alias,
        blueprint_id=algorithm.blueprint.blueprint_id,
        requirement_id=algorithm.requirements.requirement_id,
        components=tuple(selections.values()),
        objective_identity=algorithm.blueprint.objective_identity,
        execution_policy_id=definition.execution_policy.policy_id,
        trajectory_kind=algorithm.requirements.trajectory_kind,
        grouping=algorithm.requirements.grouping,
        schedule_step_count=schedule_count,
        physical_transition_count=physical_count,
        stored_policy_transition_count=stored_count,
        transition_selection=selection,
        replay_target=definition.dynamics_integration.replay_target,
        likelihood_semantics=definition.dynamics_integration.likelihood_semantics,
        reference_requirement=algorithm.requirements.reference_requirement,
        requires_reference_statistics=algorithm.requirements.reference_required,
        beta=algorithm.blueprint.beta,
        inner_epochs=1,
        optimizer_updates_per_iteration=1,
        required_trajectory_fields=tuple(
            sorted(
                set(rollout.required_transition_fields)
                | {definition.dynamics_integration.replay_target.value}
            )
        ),
        required_policy_fields=tuple(
            sorted(
                set(trainer.required_policy_fields) | set(credit.produced_policy_fields)
            )
        ),
    )


def _transition_selection(
    slot: AlgorithmSlotBlueprint,
    trajectory: TrajectoryKind,
    stored_count: int,
) -> TransitionSelectionSpec:
    params = slot.params
    if trajectory is TrajectoryKind.FULL:
        return TransitionSelectionSpec(
            kind=TransitionSelectionKind.ALL,
            policy="all",
            selected_transition_count=stored_count,
        )
    if trajectory is TrajectoryKind.BRANCHING:
        topology = params.get("branch_topology")
        if isinstance(topology, Mapping) and topology.get("kind") == (
            "every_policy_timestep"
        ):
            return TransitionSelectionSpec(
                kind=TransitionSelectionKind.ALL,
                policy="all",
                selected_transition_count=stored_count,
            )
        index = params.get("branch_step_index")
        return TransitionSelectionSpec(
            kind=TransitionSelectionKind.BRANCH_STEP,
            policy="fixed_index" if index is not None else params["branch_step_policy"],
            selected_transition_count=stored_count,
            fixed_index=index,
        )
    if trajectory is TrajectoryKind.SINGLE_STEP:
        index = params.get("selected_timestep_index")
        return TransitionSelectionSpec(
            kind=TransitionSelectionKind.SELECTED_TIMESTEP,
            policy=(
                "fixed_index"
                if index is not None
                else params["selected_timestep_policy"]
            ),
            selected_transition_count=stored_count,
            fixed_index=index,
        )
    raise ValueError(f"unsupported trajectory kind {trajectory.value!r}")


def _compatibility_snapshot(
    *,
    algorithm: ResolvedAlgorithmDeclaration,
    model: ResolvedComponentDeclaration,
    declarations: Mapping[AlgorithmComponentRole, ResolvedComponentDeclaration],
    reward_declarations: tuple[tuple[str, ResolvedComponentDeclaration], ...],
    definition: RecipeDefinition,
) -> CompatibilitySnapshot:
    model_contract = _required_contract(
        model.declared_contract.model,
        ModelDescriptorContract,
        "model",
    )
    dynamics = _required_contract(
        declarations[AlgorithmComponentRole.DYNAMICS].declared_contract.dynamics,
        DynamicsContract,
        "dynamics",
    )
    trainer = _required_contract(
        declarations[AlgorithmComponentRole.TRAINER].declared_contract.trainer,
        TrainerContract,
        "trainer",
    )
    capabilities = BoundPolicyCapabilities.from_contracts(
        model_contract,
        dynamics=dynamics,
        trainer=trainer,
    )
    public_binding = bind_model_algorithm(capabilities, algorithm.requirements)
    dynamics_match = match_model_algorithm_dynamics(
        model=model_contract,
        dynamics=dynamics,
        algorithm=algorithm.requirements,
        likelihood_semantics=definition.dynamics_integration.likelihood_semantics,
        beta=algorithm.blueprint.beta,
    )
    if not dynamics_match.is_compatible:
        codes = ", ".join(item.code for item in dynamics_match.mismatches)
        raise ValueError(f"resolved Dynamics is incompatible: {codes}")
    _validate_conditioner_and_rewards(
        model=model_contract,
        conditioner=declarations.get(AlgorithmComponentRole.CONDITIONER),
        rewards=reward_declarations,
    )
    all_declarations = (
        model,
        algorithm.component,
        *declarations.values(),
        *(item for _logical_id, item in reward_declarations),
    )
    pending = tuple(
        sorted(
            {
                field
                for declaration in all_declarations
                for field in declaration.declared_contract.pending_fields
            }
        )
    )
    bindings = {
        *public_binding.selections,
        *dynamics_match.bindings,
        ("model_algorithm_binding_id", public_binding.binding_id),
        ("execution.training_mode", definition.execution_policy.training_mode.value),
        (
            "execution.distribution_mode",
            definition.execution_policy.distribution_mode.value,
        ),
        ("execution.precision", definition.execution_policy.precision.value),
    }
    return CompatibilitySnapshot(
        status="pending_artifact_bind" if pending else "compatible",
        issues=(),
        pending_fields=pending,
        bindings=tuple(sorted(bindings)),
    )


def _validate_conditioner_and_rewards(
    *,
    model: ModelDescriptorContract,
    conditioner: ResolvedComponentDeclaration | None,
    rewards: tuple[tuple[str, ResolvedComponentDeclaration], ...],
) -> None:
    conditioner_contract: ConditionerContract | None = None
    if conditioner is not None:
        conditioner_contract = _required_contract(
            conditioner.declared_contract.conditioner,
            ConditionerContract,
            "conditioner",
        )
        if not set(model.tasks).intersection(conditioner_contract.accepted_tasks):
            raise ValueError("conditioner does not accept the model task")
        if not set(model.latent_layouts).intersection(
            conditioner_contract.accepted_latent_layouts
        ):
            raise ValueError("conditioner does not accept the model latent layout")
    for logical_id, declaration in rewards:
        reward = _required_contract(
            declaration.declared_contract.reward,
            RewardContract,
            f"reward {logical_id}",
        )
        if not set(model.output_media).intersection(reward.accepted_media):
            raise ValueError(f"reward {logical_id!r} does not accept model media")
        if reward.required_payload_type is not None and (
            conditioner_contract is None
            or conditioner_contract.payload_type != reward.required_payload_type
        ):
            raise ValueError(
                f"reward {logical_id!r} requires an unavailable conditioner payload"
            )


def _required_contract(value: object, expected: type, label: str):
    if not isinstance(value, expected):
        raise TypeError(f"{label} declaration has no {expected.__name__}")
    return value


def _exact_count(value: tuple[int, int | None], label: str) -> int:
    minimum, maximum = value
    if maximum is None or minimum != maximum:
        raise ValueError(f"{label} must resolve to one exact count")
    return minimum
