"""Canonical v0.8 checkpoint compatibility contract and field diff."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from visual_rl.algorithms.dynamics.selection import DynamicsSelectionPolicyState
from visual_rl.models.numerics.policy import ModelExecutionNumericsEvidence
from visual_rl.models.state.projection import ModelStateProjection

__all__ = (
    "CHECKPOINT_CONTRACT_SCHEMA_VERSION",
    "CHECKPOINT_PROGRESS_SCHEMA_VERSION",
    "PREPARED_CHECKPOINT_CONTRACT_SCHEMA_VERSION",
    "CheckpointContract",
    "CheckpointProgress",
    "ComponentContractRef",
    "ContractDiff",
    "OptimizerGroupContract",
    "ParameterContract",
    "PreparedCheckpointContract",
    "assert_compatible_contract",
    "assert_compatible_prepared_contract",
    "diff_contracts",
    "diff_prepared_contracts",
)

CHECKPOINT_CONTRACT_SCHEMA_VERSION = 4
CHECKPOINT_PROGRESS_SCHEMA_VERSION = 2
PREPARED_CHECKPOINT_CONTRACT_SCHEMA_VERSION = 4
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _namespaced_identity(name: str, value: object, namespace: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            re.escape(namespace) + r":[0-9a-f]{64}",
            value,
        )
        is None
    ):
        raise ValueError(f"{name} must be a {namespace} namespaced SHA-256 identity")
    return value


@dataclass(frozen=True, slots=True)
class ParameterContract:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        _text("parameter name", self.name)
        _text("parameter dtype", self.dtype)
        if type(self.shape) is not tuple or any(
            type(item) is not int or item < 0 for item in self.shape
        ):
            raise ValueError("parameter shape must be a tuple of non-negative integers")

    def to_payload(self) -> dict[str, object]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True, slots=True)
class OptimizerGroupContract:
    group_id: str
    parameter_names: tuple[str, ...]
    hyperparameters_id: str

    def __post_init__(self) -> None:
        _text("optimizer group_id", self.group_id)
        _digest("optimizer hyperparameters_id", self.hyperparameters_id)
        if type(self.parameter_names) is not tuple or not self.parameter_names:
            raise ValueError("optimizer parameter_names must be a non-empty tuple")
        for name in self.parameter_names:
            _text("optimizer parameter name", name)
        if len(self.parameter_names) != len(set(self.parameter_names)):
            raise ValueError("optimizer parameter_names must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "parameter_names": list(self.parameter_names),
            "hyperparameters_id": self.hyperparameters_id,
        }


@dataclass(frozen=True, slots=True)
class ComponentContractRef:
    """Checkpoint projection of one exact G1 receipt plus its G3 identity."""

    slot: str
    kind: str
    component_declaration_id: str
    artifact_binding_id: str
    runtime_bound_contract_id: str

    def __post_init__(self) -> None:
        _text("component slot", self.slot)
        if self.kind not in {
            "model",
            "trainer",
            "dynamics",
            "rollout",
            "reward",
            "conditioner",
            "credit",
            "algorithm",
        }:
            raise ValueError("component kind is invalid")
        _namespaced_identity(
            "component_declaration_id",
            self.component_declaration_id,
            "component-declaration.v1",
        )
        _namespaced_identity(
            "artifact_binding_id",
            self.artifact_binding_id,
            "component-artifact-binding.v1",
        )
        _digest("runtime_bound_contract_id", self.runtime_bound_contract_id)

    def to_payload(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "kind": self.kind,
            "component_declaration_id": self.component_declaration_id,
            "artifact_binding_id": self.artifact_binding_id,
            "runtime_bound_contract_id": self.runtime_bound_contract_id,
        }


@dataclass(frozen=True, slots=True)
class PreparedCheckpointContract:
    """Compatibility facts available before G3 and required before state load."""

    recipe_id: str
    resolved_fingerprint: str
    algorithm_materialization_spec_id: str
    execution_policy_id: str
    reward_plan_id: str
    source_content_binding_id: str
    component_artifact_binding_set_id: str
    immutable_model_revision: str
    code_identity: str
    model_state_projection: ModelStateProjection
    model_state_projection_id: str
    model_execution_numerics: ModelExecutionNumericsEvidence
    model_execution_numerics_id: str
    trainable_parameters: tuple[ParameterContract, ...]
    optimizer_groups: tuple[OptimizerGroupContract, ...]
    scaler_schema: str
    lr_scheduler_schema: str
    precision: str
    ema_state_schema: str
    reference_state_schema: str
    execution_transform_plan_id: str
    execution_transform_chain: tuple[tuple[str, str], ...]
    world_size: int
    state_schema_versions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name, namespace in (
            ("recipe_id", "materialized-recipe.v2"),
            ("resolved_fingerprint", "resolved-recipe.v2"),
            (
                "algorithm_materialization_spec_id",
                "algorithm-materialization-spec.v1",
            ),
            ("execution_policy_id", "execution-policy.v1"),
            ("reward_plan_id", "reward-plan-spec.v1"),
            ("source_content_binding_id", "source-content-binding.v1"),
            (
                "component_artifact_binding_set_id",
                "component-artifact-binding-set.v1",
            ),
        ):
            _namespaced_identity(name, getattr(self, name), namespace)
        for name in ("code_identity", "execution_transform_plan_id"):
            _digest(name, getattr(self, name))
        if not isinstance(self.model_state_projection, ModelStateProjection):
            raise TypeError("model_state_projection must be a ModelStateProjection")
        _digest("model_state_projection_id", self.model_state_projection_id)
        if self.model_state_projection_id != self.model_state_projection.projection_id:
            raise ValueError(
                "model_state_projection_id must match the canonical projection"
            )
        if not isinstance(
            self.model_execution_numerics,
            ModelExecutionNumericsEvidence,
        ):
            raise TypeError(
                "model_execution_numerics must be ModelExecutionNumericsEvidence"
            )
        _digest("model_execution_numerics_id", self.model_execution_numerics_id)
        if (
            self.model_execution_numerics_id
            != self.model_execution_numerics.execution_numerics_id
        ):
            raise ValueError(
                "model_execution_numerics_id must match the canonical evidence"
            )
        if (
            self.model_execution_numerics.source_projection_id
            != self.model_state_projection_id
        ):
            raise ValueError(
                "model execution numerics must use the checkpoint state projection"
            )
        for name in (
            "immutable_model_revision",
            "scaler_schema",
            "lr_scheduler_schema",
            "precision",
            "ema_state_schema",
            "reference_state_schema",
        ):
            _text(name, getattr(self, name))
        if (
            type(self.trainable_parameters) is not tuple
            or not self.trainable_parameters
        ):
            raise ValueError("trainable_parameters must be a non-empty tuple")
        if any(
            not isinstance(item, ParameterContract)
            for item in self.trainable_parameters
        ):
            raise TypeError(
                "trainable_parameters must contain ParameterContract values"
            )
        names = tuple(item.name for item in self.trainable_parameters)
        if len(names) != len(set(names)):
            raise ValueError("trainable parameter names must be unique")
        expected_trainable_dtype = (
            "torch."
            + self.model_execution_numerics.parameter_dtype_policy.trainable_parameter_dtype
        )
        if any(
            item.dtype != expected_trainable_dtype for item in self.trainable_parameters
        ):
            raise ValueError(
                "trainable parameter dtypes disagree with ParameterDTypePolicy"
            )
        if tuple(sorted(names)) != (
            self.model_state_projection.standalone_parameter_names
        ):
            raise ValueError(
                "trainable parameters must exactly match the model state projection"
            )
        if type(self.optimizer_groups) is not tuple or not self.optimizer_groups:
            raise ValueError("optimizer_groups must be a non-empty tuple")
        if any(
            not isinstance(item, OptimizerGroupContract)
            for item in self.optimizer_groups
        ):
            raise TypeError(
                "optimizer_groups must contain OptimizerGroupContract values"
            )
        group_names = tuple(
            name for group in self.optimizer_groups for name in group.parameter_names
        )
        if sorted(group_names) != sorted(names):
            raise ValueError(
                "optimizer groups must cover every trainable parameter exactly once"
            )
        if len(group_names) != len(set(group_names)):
            raise ValueError("optimizer groups must not repeat parameters")
        if type(self.execution_transform_chain) is not tuple:
            raise TypeError("execution_transform_chain must be a tuple")
        transform_ids: list[str] = []
        for item in self.execution_transform_chain:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError(
                    "execution transform entries must be (transform_id, contract_id)"
                )
            transform_id, contract_id = item
            transform_ids.append(_text("execution transform id", transform_id))
            _digest("execution transform contract id", contract_id)
        if len(transform_ids) != len(set(transform_ids)):
            raise ValueError("execution transform ids must be unique")
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if type(self.state_schema_versions) is not tuple:
            raise TypeError("state_schema_versions must be a tuple")
        schema_names: list[str] = []
        for item in self.state_schema_versions:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("state schema versions must be (name, version) pairs")
            name, version = item
            schema_names.append(_text("state schema name", name))
            if type(version) is not int or version < 1:
                raise ValueError("state schema versions must be positive integers")
        if len(schema_names) != len(set(schema_names)):
            raise ValueError("state schema names must be unique")
        if self.state_schema_versions != tuple(sorted(self.state_schema_versions)):
            raise ValueError("state_schema_versions must use canonical name order")
        if dict(self.state_schema_versions).get("model") != 2:
            raise ValueError("model parameter state schema_version must be 2")

    @property
    def prepared_checkpoint_contract_id(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_payload())).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": PREPARED_CHECKPOINT_CONTRACT_SCHEMA_VERSION,
            "recipe_id": self.recipe_id,
            "resolved_fingerprint": self.resolved_fingerprint,
            "algorithm_materialization_spec_id": (
                self.algorithm_materialization_spec_id
            ),
            "execution_policy_id": self.execution_policy_id,
            "reward_plan_id": self.reward_plan_id,
            "source_content_binding_id": self.source_content_binding_id,
            "component_artifact_binding_set_id": (
                self.component_artifact_binding_set_id
            ),
            "immutable_model_revision": self.immutable_model_revision,
            "code_identity": self.code_identity,
            "model_state_projection": self.model_state_projection.to_payload(),
            "model_state_projection_id": self.model_state_projection_id,
            "model_execution_numerics": self.model_execution_numerics.to_payload(),
            "model_execution_numerics_id": self.model_execution_numerics_id,
            "trainable_parameters": [
                item.to_payload() for item in self.trainable_parameters
            ],
            "optimizer_groups": [item.to_payload() for item in self.optimizer_groups],
            "scaler_schema": self.scaler_schema,
            "lr_scheduler_schema": self.lr_scheduler_schema,
            "precision": self.precision,
            "ema_state_schema": self.ema_state_schema,
            "reference_state_schema": self.reference_state_schema,
            "execution_transform_plan_id": self.execution_transform_plan_id,
            "execution_transform_chain": [
                {"transform_id": transform_id, "contract_id": contract_id}
                for transform_id, contract_id in self.execution_transform_chain
            ],
            "world_size": self.world_size,
            "state_schema_versions": [
                {"name": name, "version": version}
                for name, version in self.state_schema_versions
            ],
        }


@dataclass(frozen=True, slots=True)
class CheckpointContract:
    """Complete compatibility payload checked before loading mutable state."""

    recipe_id: str
    resolved_fingerprint: str
    algorithm_materialization_spec_id: str
    execution_policy_id: str
    reward_plan_id: str
    source_content_binding_id: str
    component_artifact_binding_set_id: str
    runtime_bound_contract_id: str
    immutable_model_revision: str
    code_identity: str
    components: tuple[ComponentContractRef, ...]
    model_state_projection: ModelStateProjection
    model_state_projection_id: str
    model_execution_numerics: ModelExecutionNumericsEvidence
    model_execution_numerics_id: str
    trainable_parameters: tuple[ParameterContract, ...]
    optimizer_groups: tuple[OptimizerGroupContract, ...]
    scaler_schema: str
    lr_scheduler_schema: str
    precision: str
    preprocess_identity: str
    preprocess_requirement_set_id: str
    group_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    data_sharding_version: str
    sampler_state_schema: str
    rng_policy: str
    dynamics_state_schema: str
    progress_state_schema: str
    ema_state_schema: str
    reference_state_schema: str
    execution_transform_plan_id: str
    execution_transform_chain: tuple[tuple[str, str], ...]
    world_size: int
    world_size_policy: str
    state_schema_versions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name, namespace in (
            ("recipe_id", "materialized-recipe.v2"),
            ("resolved_fingerprint", "resolved-recipe.v2"),
            (
                "algorithm_materialization_spec_id",
                "algorithm-materialization-spec.v1",
            ),
            ("execution_policy_id", "execution-policy.v1"),
            ("reward_plan_id", "reward-plan-spec.v1"),
            ("source_content_binding_id", "source-content-binding.v1"),
            (
                "component_artifact_binding_set_id",
                "component-artifact-binding-set.v1",
            ),
        ):
            _namespaced_identity(name, getattr(self, name), namespace)
        for name in (
            "runtime_bound_contract_id",
            "code_identity",
            "preprocess_identity",
            "preprocess_requirement_set_id",
            "execution_transform_plan_id",
        ):
            _digest(name, getattr(self, name))
        if not isinstance(self.model_state_projection, ModelStateProjection):
            raise TypeError("model_state_projection must be a ModelStateProjection")
        _digest("model_state_projection_id", self.model_state_projection_id)
        if self.model_state_projection_id != self.model_state_projection.projection_id:
            raise ValueError(
                "model_state_projection_id must match the canonical projection"
            )
        if not isinstance(
            self.model_execution_numerics,
            ModelExecutionNumericsEvidence,
        ):
            raise TypeError(
                "model_execution_numerics must be ModelExecutionNumericsEvidence"
            )
        _digest("model_execution_numerics_id", self.model_execution_numerics_id)
        if (
            self.model_execution_numerics_id
            != self.model_execution_numerics.execution_numerics_id
        ):
            raise ValueError(
                "model_execution_numerics_id must match the canonical evidence"
            )
        if (
            self.model_execution_numerics.source_projection_id
            != self.model_state_projection_id
        ):
            raise ValueError(
                "model execution numerics must use the checkpoint state projection"
            )
        for name in (
            "immutable_model_revision",
            "scaler_schema",
            "lr_scheduler_schema",
            "precision",
            "data_sharding_version",
            "sampler_state_schema",
            "rng_policy",
            "dynamics_state_schema",
            "progress_state_schema",
            "ema_state_schema",
            "reference_state_schema",
        ):
            _text(name, getattr(self, name))
        if type(self.components) is not tuple or not self.components:
            raise ValueError("components must be a non-empty tuple")
        if any(not isinstance(item, ComponentContractRef) for item in self.components):
            raise TypeError("components must contain ComponentContractRef values")
        slots = tuple(item.slot for item in self.components)
        if len(slots) != len(set(slots)):
            raise ValueError("component slots must be unique")
        if slots != tuple(sorted(slots)):
            raise ValueError("components must use canonical slot order")
        kinds = tuple(item.kind for item in self.components)
        duplicate_non_rewards = tuple(
            kind for kind in set(kinds) if kind != "reward" and kinds.count(kind) > 1
        )
        if duplicate_non_rewards:
            raise ValueError("only rewards may have multiple component refs")
        if (
            type(self.trainable_parameters) is not tuple
            or not self.trainable_parameters
        ):
            raise ValueError("trainable_parameters must be a non-empty tuple")
        names = tuple(item.name for item in self.trainable_parameters)
        if len(names) != len(set(names)):
            raise ValueError("trainable parameter names must be unique")
        expected_trainable_dtype = (
            "torch."
            + self.model_execution_numerics.parameter_dtype_policy.trainable_parameter_dtype
        )
        if any(
            item.dtype != expected_trainable_dtype for item in self.trainable_parameters
        ):
            raise ValueError(
                "trainable parameter dtypes disagree with ParameterDTypePolicy"
            )
        if tuple(sorted(names)) != (
            self.model_state_projection.standalone_parameter_names
        ):
            raise ValueError(
                "trainable parameters must exactly match the model state projection"
            )
        if type(self.optimizer_groups) is not tuple or not self.optimizer_groups:
            raise ValueError("optimizer_groups must be a non-empty tuple")
        group_names = tuple(
            name for group in self.optimizer_groups for name in group.parameter_names
        )
        if sorted(group_names) != sorted(names):
            raise ValueError(
                "optimizer groups must cover every trainable parameter exactly once"
            )
        if len(group_names) != len(set(group_names)):
            raise ValueError("optimizer groups must not repeat parameters")
        for name in (
            "group_size",
            "global_batch_size",
            "gradient_accumulation_steps",
            "world_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.global_batch_size % self.group_size:
            raise ValueError("global_batch_size must be divisible by group_size")
        if self.world_size_policy != "strict":
            raise ValueError("v0.8 supports only strict world_size compatibility")
        if type(self.execution_transform_chain) is not tuple:
            raise TypeError("execution_transform_chain must be a tuple")
        transform_ids: list[str] = []
        for item in self.execution_transform_chain:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError(
                    "execution transform entries must be (transform_id, contract_id)"
                )
            transform_id, contract_id = item
            transform_ids.append(_text("execution transform id", transform_id))
            _digest("execution transform contract id", contract_id)
        if len(transform_ids) != len(set(transform_ids)):
            raise ValueError("execution transform ids must be unique")
        if (
            type(self.state_schema_versions) is not tuple
            or not self.state_schema_versions
        ):
            raise ValueError("state_schema_versions must be a non-empty tuple")
        schema_names = tuple(name for name, _ in self.state_schema_versions)
        if len(schema_names) != len(set(schema_names)):
            raise ValueError("state schema names must be unique")
        if any(
            not isinstance(name, str)
            or not name
            or type(version) is not int
            or version < 1
            for name, version in self.state_schema_versions
        ):
            raise ValueError("state schema versions must be positive integers")
        if dict(self.state_schema_versions).get("model") != 2:
            raise ValueError("model parameter state schema_version must be 2")

    @property
    def checkpoint_contract_id(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_payload())).hexdigest()

    def prepared_projection(self) -> PreparedCheckpointContract:
        """Project only facts safe to validate before runtime graph binding."""

        return PreparedCheckpointContract(
            recipe_id=self.recipe_id,
            resolved_fingerprint=self.resolved_fingerprint,
            algorithm_materialization_spec_id=(self.algorithm_materialization_spec_id),
            execution_policy_id=self.execution_policy_id,
            reward_plan_id=self.reward_plan_id,
            source_content_binding_id=self.source_content_binding_id,
            component_artifact_binding_set_id=(self.component_artifact_binding_set_id),
            immutable_model_revision=self.immutable_model_revision,
            code_identity=self.code_identity,
            model_state_projection=self.model_state_projection,
            model_state_projection_id=self.model_state_projection_id,
            model_execution_numerics=self.model_execution_numerics,
            model_execution_numerics_id=self.model_execution_numerics_id,
            trainable_parameters=self.trainable_parameters,
            optimizer_groups=self.optimizer_groups,
            scaler_schema=self.scaler_schema,
            lr_scheduler_schema=self.lr_scheduler_schema,
            precision=self.precision,
            ema_state_schema=self.ema_state_schema,
            reference_state_schema=self.reference_state_schema,
            execution_transform_plan_id=self.execution_transform_plan_id,
            execution_transform_chain=self.execution_transform_chain,
            world_size=self.world_size,
            state_schema_versions=_prepared_state_schema_versions(
                self.state_schema_versions
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_CONTRACT_SCHEMA_VERSION,
            "recipe_id": self.recipe_id,
            "resolved_fingerprint": self.resolved_fingerprint,
            "algorithm_materialization_spec_id": (
                self.algorithm_materialization_spec_id
            ),
            "execution_policy_id": self.execution_policy_id,
            "reward_plan_id": self.reward_plan_id,
            "source_content_binding_id": self.source_content_binding_id,
            "component_artifact_binding_set_id": (
                self.component_artifact_binding_set_id
            ),
            "runtime_bound_contract_id": self.runtime_bound_contract_id,
            "immutable_model_revision": self.immutable_model_revision,
            "code_identity": self.code_identity,
            "components": [item.to_payload() for item in self.components],
            "model_state_projection": self.model_state_projection.to_payload(),
            "model_state_projection_id": self.model_state_projection_id,
            "model_execution_numerics": self.model_execution_numerics.to_payload(),
            "model_execution_numerics_id": self.model_execution_numerics_id,
            "trainable_parameters": [
                item.to_payload() for item in self.trainable_parameters
            ],
            "optimizer_groups": [item.to_payload() for item in self.optimizer_groups],
            "scaler_schema": self.scaler_schema,
            "lr_scheduler_schema": self.lr_scheduler_schema,
            "precision": self.precision,
            "preprocess_identity": self.preprocess_identity,
            "preprocess_requirement_set_id": self.preprocess_requirement_set_id,
            "group_size": self.group_size,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "data_sharding_version": self.data_sharding_version,
            "sampler_state_schema": self.sampler_state_schema,
            "rng_policy": self.rng_policy,
            "dynamics_state_schema": self.dynamics_state_schema,
            "progress_state_schema": self.progress_state_schema,
            "ema_state_schema": self.ema_state_schema,
            "reference_state_schema": self.reference_state_schema,
            "execution_transform_plan_id": self.execution_transform_plan_id,
            "execution_transform_chain": [
                {"transform_id": transform_id, "contract_id": contract_id}
                for transform_id, contract_id in self.execution_transform_chain
            ],
            "world_size": self.world_size,
            "world_size_policy": self.world_size_policy,
            "state_schema_versions": [
                {"name": name, "version": version}
                for name, version in self.state_schema_versions
            ],
        }


@dataclass(frozen=True, slots=True)
class CheckpointProgress:
    """Logical safe-point state required for resume equivalence."""

    global_step: int
    iteration: int
    next_optimizer_step: int
    next_source_id: str
    next_prompt_batch_id: str
    next_phase_id: str
    active_reward_ids: tuple[str, ...]
    source_cursors: tuple[tuple[str, int], ...]
    dynamics_selection_policy: DynamicsSelectionPolicyState
    gradient_accumulation_position: int
    ema_state_saved: bool
    reference_state_saved: bool
    execution_transform_plan_id: str
    rng_state_id: str

    def __post_init__(self) -> None:
        for name in ("global_step", "iteration", "next_optimizer_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.next_optimizer_step != self.global_step:
            raise ValueError(
                "next_optimizer_step must equal global_step at a safe point"
            )
        _text("next_source_id", self.next_source_id)
        _digest("next_prompt_batch_id", self.next_prompt_batch_id)
        _text("next_phase_id", self.next_phase_id)
        if type(self.active_reward_ids) is not tuple or not self.active_reward_ids:
            raise ValueError("active_reward_ids must be a non-empty tuple")
        for reward_id in self.active_reward_ids:
            _text("active reward id", reward_id)
        if len(self.active_reward_ids) != len(set(self.active_reward_ids)):
            raise ValueError("active_reward_ids must be unique")
        if type(self.source_cursors) is not tuple or not self.source_cursors:
            raise ValueError("source_cursors must be a non-empty tuple")
        source_ids: list[str] = []
        for item in self.source_cursors:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("source cursors must be (source_id, cursor) pairs")
            source_id, cursor = item
            source_ids.append(_text("source id", source_id))
            if type(cursor) is not int or cursor < 0:
                raise ValueError("source cursor must be a non-negative integer")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source cursor ids must be unique")
        if tuple(sorted(self.source_cursors)) != self.source_cursors:
            raise ValueError("source_cursors must use canonical source-id order")
        if not isinstance(
            self.dynamics_selection_policy,
            DynamicsSelectionPolicyState,
        ):
            raise TypeError(
                "dynamics_selection_policy must be a DynamicsSelectionPolicyState"
            )
        if self.gradient_accumulation_position != 0:
            raise ValueError(
                "v0.8 checkpoints are allowed only at an accumulation safe point"
            )
        if type(self.ema_state_saved) is not bool:
            raise TypeError("ema_state_saved must be bool")
        if type(self.reference_state_saved) is not bool:
            raise TypeError("reference_state_saved must be bool")
        _digest(
            "execution_transform_plan_id",
            self.execution_transform_plan_id,
        )
        _digest("rng_state_id", self.rng_state_id)

    @property
    def progress_id(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_payload())).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_PROGRESS_SCHEMA_VERSION,
            "global_step": self.global_step,
            "iteration": self.iteration,
            "next_optimizer_step": self.next_optimizer_step,
            "next_source_id": self.next_source_id,
            "next_prompt_batch_id": self.next_prompt_batch_id,
            "next_phase_id": self.next_phase_id,
            "active_reward_ids": list(self.active_reward_ids),
            "source_cursors": [
                {"source_id": source_id, "cursor": cursor}
                for source_id, cursor in self.source_cursors
            ],
            "dynamics_selection_policy": (
                self.dynamics_selection_policy.to_checkpoint_payload()
            ),
            "gradient_accumulation_position": (self.gradient_accumulation_position),
            "ema_state_saved": self.ema_state_saved,
            "reference_state_saved": self.reference_state_saved,
            "execution_transform_plan_id": self.execution_transform_plan_id,
            "rng_state_id": self.rng_state_id,
        }


@dataclass(frozen=True, slots=True)
class ContractDiff:
    path: str
    expected: object
    found: object


def diff_contracts(
    expected: CheckpointContract,
    found: CheckpointContract,
) -> tuple[ContractDiff, ...]:
    if not isinstance(expected, CheckpointContract) or not isinstance(
        found, CheckpointContract
    ):
        raise TypeError("expected and found must be CheckpointContract values")
    output: list[ContractDiff] = []
    _diff_payload(expected.to_payload(), found.to_payload(), "", output)
    return tuple(output)


def assert_compatible_contract(
    expected: CheckpointContract,
    found: CheckpointContract,
) -> None:
    differences = diff_contracts(expected, found)
    if differences:
        summary = ", ".join(item.path for item in differences[:8])
        raise ValueError(f"checkpoint contract mismatch at: {summary}")


def diff_prepared_contracts(
    expected: PreparedCheckpointContract,
    found: PreparedCheckpointContract,
) -> tuple[ContractDiff, ...]:
    if not isinstance(expected, PreparedCheckpointContract) or not isinstance(
        found, PreparedCheckpointContract
    ):
        raise TypeError("expected and found must be PreparedCheckpointContract values")
    output: list[ContractDiff] = []
    _diff_payload(expected.to_payload(), found.to_payload(), "", output)
    return tuple(output)


def assert_compatible_prepared_contract(
    expected: PreparedCheckpointContract,
    found: PreparedCheckpointContract,
) -> None:
    differences = diff_prepared_contracts(expected, found)
    if differences:
        summary = ", ".join(item.path for item in differences[:8])
        raise ValueError(f"prepared checkpoint contract mismatch at: {summary}")


_PREPARED_STATE_SCHEMA_NAMES = frozenset(
    {"ema", "lr_scheduler", "model", "optimizer", "reference", "scaler"}
)


def _prepared_state_schema_versions(
    values: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(item for item in values if item[0] in _PREPARED_STATE_SCHEMA_NAMES)
    )


def _diff_payload(
    expected: object,
    found: object,
    path: str,
    output: list[ContractDiff],
) -> None:
    if isinstance(expected, dict) and isinstance(found, dict):
        for key in sorted(set(expected) | set(found)):
            child = f"{path}.{key}" if path else key
            if key not in expected:
                output.append(ContractDiff(child, "<absent>", found[key]))
            elif key not in found:
                output.append(ContractDiff(child, expected[key], "<absent>"))
            else:
                _diff_payload(expected[key], found[key], child, output)
        return
    if isinstance(expected, list) and isinstance(found, list):
        if len(expected) != len(found):
            output.append(ContractDiff(f"{path}.length", len(expected), len(found)))
        for index, (left, right) in enumerate(zip(expected, found)):
            _diff_payload(left, right, f"{path}[{index}]", output)
        return
    if expected != found:
        output.append(ContractDiff(path or "<root>", expected, found))


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
