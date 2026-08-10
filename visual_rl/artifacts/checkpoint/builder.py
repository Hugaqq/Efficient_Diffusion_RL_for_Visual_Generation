"""Build the complete checkpoint contract from the bound runtime graph."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from visual_rl.core.serialization import canonical_json_text
from visual_rl.artifacts.checkpoint.protocol import (
    CheckpointContract,
    ComponentContractRef,
    OptimizerGroupContract,
    ParameterContract,
    PreparedCheckpointContract,
)
from visual_rl.composition.recipes.schema import MaterializedRecipe
from visual_rl.core.contracts.composition import (
    ComponentArtifactBindingSet,
    RuntimeBoundContract,
)
from visual_rl.core.contracts.runtime import ExecutionTransformPlan
from visual_rl.models.numerics.policy import ModelExecutionNumericsEvidence
from visual_rl.models.state.parameters import ParameterStateManager
from visual_rl.composition.preflight.types import (
    RuntimeFacts,
    RuntimeGraphBindResult,
    runtime_graph_payload_id,
    runtime_launch_payload_id,
)

__all__ = (
    "CheckpointBuildInput",
    "PreparedCheckpointBuildInput",
    "build_checkpoint_contract",
    "build_prepared_checkpoint_contract",
)


_PREPARED_STATE_SCHEMA_NAMES = frozenset(
    {"ema", "lr_scheduler", "model", "optimizer", "reference", "scaler"}
)


@dataclass(frozen=True, slots=True)
class PreparedCheckpointBuildInput:
    """Live prepare-stage facts available before transforms are applied and G3."""

    recipe: MaterializedRecipe
    artifact_binding_set: ComponentArtifactBindingSet
    runtime_facts: RuntimeFacts
    parameter_state: ParameterStateManager
    model_execution_numerics: ModelExecutionNumericsEvidence
    optimizer: object
    scaler: object | None
    lr_scheduler: object | None
    execution_transform_plan: ExecutionTransformPlan
    ema_state_schema: str = "none.v1"
    reference_state_schema: str = "optional-frozen-reference.v1"
    state_schema_versions: tuple[tuple[str, int], ...] = (
        ("model", 2),
        ("optimizer", 1),
    )

    def __post_init__(self) -> None:
        _validate_prepared_build_input(
            recipe=self.recipe,
            artifact_binding_set=self.artifact_binding_set,
            runtime_facts=self.runtime_facts,
            parameter_state=self.parameter_state,
            model_execution_numerics=self.model_execution_numerics,
            execution_transform_plan=self.execution_transform_plan,
        )


@dataclass(frozen=True, slots=True)
class CheckpointBuildInput:
    """All G3 facts that may affect resume compatibility."""

    recipe: MaterializedRecipe
    artifact_binding_set: ComponentArtifactBindingSet
    runtime_facts: RuntimeFacts
    graph_binding: RuntimeGraphBindResult
    runtime_bound_contracts: tuple[tuple[str, RuntimeBoundContract], ...]
    parameter_state: ParameterStateManager
    model_execution_numerics: ModelExecutionNumericsEvidence
    optimizer: object
    scaler: object | None
    lr_scheduler: object | None
    execution_transform_plan: ExecutionTransformPlan
    preprocess_identity: str
    preprocess_requirement_set_id: str
    data_sharding_version: str = "strict-rank-shard.v1"
    sampler_state_schema: str = "per-source-cursor.v1"
    rng_policy: str = "rank-local-explicit-generator.v1"
    dynamics_state_schema: str = "iteration-keyed-selection-policy.v2"
    progress_state_schema: str = "logical-safe-point.v2"
    ema_state_schema: str = "none.v1"
    reference_state_schema: str = "optional-frozen-reference.v1"
    state_schema_versions: tuple[tuple[str, int], ...] = (
        ("dynamics_selection_policy", 2),
        ("model", 2),
        ("optimizer", 1),
        ("phase_schedule", 1),
        ("progress", 2),
        ("rng", 1),
        ("sampler", 1),
    )

    def __post_init__(self) -> None:
        _validate_prepared_build_input(
            recipe=self.recipe,
            artifact_binding_set=self.artifact_binding_set,
            runtime_facts=self.runtime_facts,
            parameter_state=self.parameter_state,
            model_execution_numerics=self.model_execution_numerics,
            execution_transform_plan=self.execution_transform_plan,
        )
        runtime_contracts = _runtime_contracts_by_slot(
            self.runtime_bound_contracts,
            artifact_binding_set=self.artifact_binding_set,
        )
        _validate_graph_binding(
            self.graph_binding,
            recipe=self.recipe,
            runtime_facts=self.runtime_facts,
            runtime_contracts=runtime_contracts,
        )
        _digest("preprocess_identity", self.preprocess_identity)
        _digest(
            "preprocess_requirement_set_id",
            self.preprocess_requirement_set_id,
        )


def build_prepared_checkpoint_contract(
    request: PreparedCheckpointBuildInput,
) -> PreparedCheckpointContract:
    """Build the compatibility gate used before loading prepared owner state."""

    if not isinstance(request, PreparedCheckpointBuildInput):
        raise TypeError("request must be PreparedCheckpointBuildInput")
    return _build_prepared_checkpoint_contract(
        recipe=request.recipe,
        artifact_binding_set=request.artifact_binding_set,
        runtime_facts=request.runtime_facts,
        parameter_state=request.parameter_state,
        model_execution_numerics=request.model_execution_numerics,
        optimizer=request.optimizer,
        scaler=request.scaler,
        lr_scheduler=request.lr_scheduler,
        execution_transform_plan=request.execution_transform_plan,
        ema_state_schema=request.ema_state_schema,
        reference_state_schema=request.reference_state_schema,
        state_schema_versions=request.state_schema_versions,
    )


def build_checkpoint_contract(request: CheckpointBuildInput) -> CheckpointContract:
    """Derive one canonical contract and verify the live optimizer topology."""

    if not isinstance(request, CheckpointBuildInput):
        raise TypeError("request must be CheckpointBuildInput")
    recipe = request.recipe
    prepared = _build_prepared_checkpoint_contract(
        recipe=recipe,
        artifact_binding_set=request.artifact_binding_set,
        runtime_facts=request.runtime_facts,
        parameter_state=request.parameter_state,
        model_execution_numerics=request.model_execution_numerics,
        optimizer=request.optimizer,
        scaler=request.scaler,
        lr_scheduler=request.lr_scheduler,
        execution_transform_plan=request.execution_transform_plan,
        ema_state_schema=request.ema_state_schema,
        reference_state_schema=request.reference_state_schema,
        state_schema_versions=request.state_schema_versions,
    )

    runtime_contracts = _runtime_contracts_by_slot(
        request.runtime_bound_contracts,
        artifact_binding_set=request.artifact_binding_set,
    )

    component_refs = tuple(
        ComponentContractRef(
            slot=binding.slot,
            kind=binding.declared.component_kind,
            component_declaration_id=binding.component_declaration_id,
            artifact_binding_id=binding.binding_id,
            runtime_bound_contract_id=runtime_contracts[binding.slot].contract_id,
        )
        for binding in request.artifact_binding_set.bindings
    )
    resolved = recipe.resolved
    execution = resolved.execution_policy
    training = resolved.training
    return CheckpointContract(
        recipe_id=prepared.recipe_id,
        resolved_fingerprint=prepared.resolved_fingerprint,
        algorithm_materialization_spec_id=(prepared.algorithm_materialization_spec_id),
        execution_policy_id=prepared.execution_policy_id,
        reward_plan_id=prepared.reward_plan_id,
        source_content_binding_id=prepared.source_content_binding_id,
        component_artifact_binding_set_id=(prepared.component_artifact_binding_set_id),
        runtime_bound_contract_id=request.graph_binding.bound_contract_id,
        immutable_model_revision=prepared.immutable_model_revision,
        code_identity=prepared.code_identity,
        components=component_refs,
        model_state_projection=prepared.model_state_projection,
        model_state_projection_id=prepared.model_state_projection_id,
        model_execution_numerics=prepared.model_execution_numerics,
        model_execution_numerics_id=prepared.model_execution_numerics_id,
        trainable_parameters=prepared.trainable_parameters,
        optimizer_groups=prepared.optimizer_groups,
        scaler_schema=prepared.scaler_schema,
        lr_scheduler_schema=prepared.lr_scheduler_schema,
        precision=prepared.precision,
        preprocess_identity=request.preprocess_identity,
        preprocess_requirement_set_id=request.preprocess_requirement_set_id,
        group_size=execution.group_size,
        global_batch_size=training.global_prompt_batch_size * execution.group_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        data_sharding_version=request.data_sharding_version,
        sampler_state_schema=request.sampler_state_schema,
        rng_policy=request.rng_policy,
        dynamics_state_schema=request.dynamics_state_schema,
        progress_state_schema=request.progress_state_schema,
        ema_state_schema=prepared.ema_state_schema,
        reference_state_schema=prepared.reference_state_schema,
        execution_transform_plan_id=prepared.execution_transform_plan_id,
        execution_transform_chain=prepared.execution_transform_chain,
        world_size=prepared.world_size,
        world_size_policy="strict",
        state_schema_versions=request.state_schema_versions,
    )


def _build_prepared_checkpoint_contract(
    *,
    recipe: MaterializedRecipe,
    artifact_binding_set: ComponentArtifactBindingSet,
    runtime_facts: RuntimeFacts,
    parameter_state: ParameterStateManager,
    model_execution_numerics: ModelExecutionNumericsEvidence,
    optimizer: object,
    scaler: object | None,
    lr_scheduler: object | None,
    execution_transform_plan: ExecutionTransformPlan,
    ema_state_schema: str,
    reference_state_schema: str,
    state_schema_versions: tuple[tuple[str, int], ...],
) -> PreparedCheckpointContract:
    resolved = recipe.resolved
    execution = resolved.execution_policy
    if runtime_facts.precision != execution.precision.value:
        raise ValueError("runtime precision differs from MaterializedRecipe")
    if runtime_facts.distribution_mode != execution.distribution_mode.value:
        raise ValueError("runtime distribution mode differs from MaterializedRecipe")
    if execution_transform_plan.plan_id != execution.transform_plan.plan_id:
        raise ValueError("runtime execution transform chain differs from recipe")

    projection = parameter_state.state_projection
    if model_execution_numerics.source_projection_id != projection.projection_id:
        raise ValueError(
            "model execution numerics use a different model_state_projection"
        )
    parameters = parameter_state.named_trainable_parameters()
    parameter_contracts = tuple(
        ParameterContract(
            name=item.name,
            shape=tuple(int(size) for size in item.parameter.shape),
            dtype=str(item.parameter.dtype),
        )
        for item in parameters
    )
    optimizer_groups = _optimizer_groups(
        optimizer,
        {id(item.parameter): item.name for item in parameters},
    )
    return PreparedCheckpointContract(
        recipe_id=recipe.recipe_id,
        resolved_fingerprint=resolved.resolved_fingerprint,
        algorithm_materialization_spec_id=resolved.algorithm_spec.spec_id,
        execution_policy_id=execution.policy_id,
        reward_plan_id=recipe.reward_plan.plan_id,
        source_content_binding_id=recipe.source_content_binding.content_binding_id,
        component_artifact_binding_set_id=artifact_binding_set.binding_set_id,
        immutable_model_revision=_model_revision(recipe.model_artifact_identity),
        code_identity=_content_sha256(recipe.code_artifact_identity),
        model_state_projection=projection,
        model_state_projection_id=projection.projection_id,
        model_execution_numerics=model_execution_numerics,
        model_execution_numerics_id=(model_execution_numerics.execution_numerics_id),
        trainable_parameters=parameter_contracts,
        optimizer_groups=optimizer_groups,
        scaler_schema=_state_schema(scaler, none="none.v1"),
        lr_scheduler_schema=_state_schema(lr_scheduler, none="none.v1"),
        precision=execution.precision.value,
        ema_state_schema=ema_state_schema,
        reference_state_schema=reference_state_schema,
        execution_transform_plan_id=execution_transform_plan.plan_id,
        execution_transform_chain=tuple(
            (item.transform_id, item.contract_id)
            for item in execution_transform_plan.transforms
        ),
        world_size=runtime_facts.world_size,
        state_schema_versions=_prepared_state_schema_versions(state_schema_versions),
    )


def _validate_prepared_build_input(
    *,
    recipe: object,
    artifact_binding_set: object,
    runtime_facts: object,
    parameter_state: object,
    model_execution_numerics: object,
    execution_transform_plan: object,
) -> None:
    if not isinstance(recipe, MaterializedRecipe):
        raise TypeError("recipe must be a MaterializedRecipe")
    if not isinstance(artifact_binding_set, ComponentArtifactBindingSet):
        raise TypeError("artifact_binding_set must be a ComponentArtifactBindingSet")
    if artifact_binding_set.recipe_id != recipe.recipe_id:
        raise ValueError("artifact binding set differs from MaterializedRecipe")
    expected_declarations = _recipe_component_declarations(recipe)
    expected_slots = tuple(expected_declarations)
    if artifact_binding_set.slots != expected_slots:
        raise ValueError(
            "artifact binding set must exactly cover canonical recipe component slots"
        )
    for binding in artifact_binding_set.bindings:
        if binding.component_declaration_id != expected_declarations[binding.slot]:
            raise ValueError(
                f"artifact binding for {binding.slot!r} differs from the canonical "
                "recipe declaration"
            )
    if not isinstance(runtime_facts, RuntimeFacts):
        raise TypeError("runtime_facts must be RuntimeFacts")
    if not isinstance(parameter_state, ParameterStateManager):
        raise TypeError("parameter_state must be a ParameterStateManager")
    if not isinstance(model_execution_numerics, ModelExecutionNumericsEvidence):
        raise TypeError(
            "model_execution_numerics must be ModelExecutionNumericsEvidence"
        )
    if not isinstance(execution_transform_plan, ExecutionTransformPlan):
        raise TypeError("execution_transform_plan must be ExecutionTransformPlan")


def _prepared_state_schema_versions(
    values: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    if type(values) is not tuple:
        raise TypeError("state_schema_versions must be a tuple")
    selected: list[tuple[str, int]] = []
    for item in values:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("state schema versions must be (name, version) pairs")
        name, version = item
        if not isinstance(name, str) or not name:
            raise ValueError("state schema names must be non-empty strings")
        if type(version) is not int or version < 1:
            raise ValueError("state schema versions must be positive integers")
        if name in _PREPARED_STATE_SCHEMA_NAMES:
            selected.append((name, version))
    return tuple(sorted(selected))


def _optimizer_groups(
    optimizer: object,
    parameter_names: Mapping[int, str],
) -> tuple[OptimizerGroupContract, ...]:
    import torch

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    groups: list[OptimizerGroupContract] = []
    seen: list[int] = []
    for index, group in enumerate(optimizer.param_groups):
        raw_parameters = tuple(group.get("params", ()))
        if not raw_parameters:
            raise ValueError("optimizer groups must not be empty")
        ids = tuple(id(item) for item in raw_parameters)
        unknown = tuple(item for item in ids if item not in parameter_names)
        if unknown:
            raise ValueError(
                "optimizer contains a parameter outside trainable topology"
            )
        seen.extend(ids)
        hyperparameters = {
            key: _plain_optimizer_value(value, path=f"optimizer.group[{index}].{key}")
            for key, value in group.items()
            if key not in {"params", "lr", "initial_lr"}
        }
        initial_lr = group.get("initial_lr", group.get("lr"))
        if initial_lr is None:
            raise ValueError("optimizer group must expose an initial learning rate")
        hyperparameters["lr"] = _plain_optimizer_value(
            initial_lr,
            path=f"optimizer.group[{index}].initial_lr",
        )
        groups.append(
            OptimizerGroupContract(
                group_id=f"group-{index}",
                parameter_names=tuple(parameter_names[item] for item in ids),
                hyperparameters_id=_payload_digest(hyperparameters),
            )
        )
    if len(seen) != len(set(seen)):
        raise ValueError("optimizer repeats a trainable parameter")
    if set(seen) != set(parameter_names):
        raise ValueError("optimizer must cover the exact trainable parameter subset")
    return tuple(groups)


def _plain_optimizer_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [
            _plain_optimizer_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    module = type(value).__module__
    if module == "torch" and type(value).__name__ in {"dtype", "device"}:
        return str(value)
    raise TypeError(
        f"{path} has unsupported non-canonical value {type(value).__name__}"
    )


def _state_schema(value: object | None, *, none: str) -> str:
    if value is None:
        return none
    state_dict = getattr(value, "state_dict", None)
    load_state_dict = getattr(value, "load_state_dict", None)
    if not callable(state_dict) or not callable(load_state_dict):
        raise TypeError(
            "stateful runtime objects must implement state_dict/load_state_dict"
        )
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}.v1"


def _recipe_component_declarations(
    recipe: MaterializedRecipe,
) -> dict[str, str]:
    resolved = recipe.resolved
    items = {
        "algorithm": resolved.algorithm.component_declaration_id,
        "model": resolved.model.component_declaration_id,
        **{
            item.slot: item.component_declaration_id
            for item in (*resolved.internal_components, *resolved.reward_components)
        },
    }
    return dict(sorted(items.items()))


def _runtime_contracts_by_slot(
    values: object,
    *,
    artifact_binding_set: ComponentArtifactBindingSet,
) -> dict[str, RuntimeBoundContract]:
    if type(values) is not tuple or not values:
        raise ValueError("runtime_bound_contracts must be a non-empty tuple")
    result: dict[str, RuntimeBoundContract] = {}
    for item in values:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(
                "runtime_bound_contracts must contain "
                "(slot, RuntimeBoundContract) pairs"
            )
        slot, contract = item
        if not isinstance(slot, str) or not slot:
            raise ValueError("runtime bound contract slots must be non-empty")
        if not isinstance(contract, RuntimeBoundContract):
            raise TypeError(
                "runtime_bound_contracts values must be RuntimeBoundContract"
            )
        if not contract.is_declaration_bound:
            raise ValueError(
                "checkpoint contracts reject legacy runtime-bound contracts"
            )
        try:
            expected_binding = artifact_binding_set.binding(slot)
        except KeyError as exc:
            raise ValueError(
                f"runtime contract slot {slot!r} is absent from the G1 binding set"
            ) from exc
        if contract.artifact != expected_binding:
            raise ValueError(
                f"runtime contract for {slot!r} does not reference the exact G1 binding"
            )
        result[slot] = contract
    if tuple(result) != tuple(sorted(result)) or len(result) != len(values):
        raise ValueError(
            "runtime_bound_contracts must be sorted by unique component slot"
        )
    if tuple(result) != artifact_binding_set.slots:
        raise ValueError(
            "runtime-bound contracts must exactly cover the G1 component slots"
        )
    return result


def _validate_graph_binding(
    value: object,
    *,
    recipe: MaterializedRecipe,
    runtime_facts: RuntimeFacts,
    runtime_contracts: Mapping[str, RuntimeBoundContract],
) -> None:
    if not isinstance(value, RuntimeGraphBindResult):
        raise TypeError("graph_binding must be a RuntimeGraphBindResult")
    if value.recipe_id != recipe.recipe_id:
        raise ValueError("runtime graph binding differs from MaterializedRecipe")
    expected_graph_id = runtime_graph_payload_id(value.canonical_payload())
    if value.bound_contract_id != expected_graph_id:
        raise ValueError(
            "runtime graph bound_contract_id differs from its canonical payload"
        )
    expected_launch_id = runtime_launch_payload_id(recipe.recipe_id, runtime_facts)
    if value.launch_id != expected_launch_id:
        raise ValueError(
            "runtime graph launch_id differs from recipe/runtime compatibility payload"
        )
    expected_component_ids = {
        slot: contract.contract_id for slot, contract in runtime_contracts.items()
    }
    if dict(value.component_bound_contract_ids) != expected_component_ids:
        raise ValueError(
            "runtime graph component ids differ from typed runtime-bound contracts"
        )


def _model_revision(value: Mapping[str, Any]) -> str:
    return "sha256:" + _content_sha256(value)


def _content_sha256(value: Mapping[str, Any]) -> str:
    candidate = value.get("content_sha256")
    return _digest("content_sha256", candidate)


def _payload_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_text(value).encode("utf-8")).hexdigest()


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value
