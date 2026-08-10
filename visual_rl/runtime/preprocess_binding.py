"""Typed preprocess identity binding for one prepared model runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from visual_rl.algorithms.trainer.execution_plan import AlgorithmExecutionPlan
from visual_rl.composition.preflight.types import RuntimeFacts
from visual_rl.composition.recipes.schema import MaterializedRecipe
from visual_rl.core.contracts import (
    ConditionerContract,
    DeclaredContract,
    RuntimeBoundContract,
)
from visual_rl.core.types import to_plain_dict
from visual_rl.data import (
    PreprocessConsumerRequirement,
    PreprocessContractError,
    PreprocessProducerSpec,
    PreprocessRequirementProvider,
    PreprocessRequirementSet,
)
from visual_rl.models import (
    ModelAdapter,
    ModelPreprocessConsumerSpec,
)
from visual_rl.models.lifecycle.components import ComponentManager
from visual_rl.models.lifecycle.prepared import PreparedComponentHandle
from visual_rl.runtime.component_graph import (
    ComponentRuntimeBindingError,
    RuntimeComponentBinding,
)

__all__ = (
    "PreprocessIdentityProvider",
    "PreprocessIdentityRequest",
    "PreprocessIdentityResult",
    "PreprocessRuntimeBinding",
    "PreprocessRequirementCompileError",
    "compile_preprocess_requirement_set",
    "resolve_preprocess_runtime_binding",
)


class PreprocessRequirementCompileError(PreprocessContractError):
    """The resolved consumer graph cannot form one exact requirement set."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class PreprocessIdentityRequest:
    """Narrow leaf input for content-addressing the actual preprocess path."""

    materialized: MaterializedRecipe
    model_binding: RuntimeComponentBinding
    manager: ComponentManager
    handle: PreparedComponentHandle
    runtime_facts: RuntimeFacts
    model_runtime_contract: RuntimeBoundContract
    requirement_set: PreprocessRequirementSet | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.materialized, MaterializedRecipe):
            raise TypeError("materialized must be a MaterializedRecipe")
        if not isinstance(self.model_binding, RuntimeComponentBinding):
            raise TypeError("model_binding must be RuntimeComponentBinding")
        if self.model_binding.slot != "model" or self.model_binding.kind != "model":
            raise ValueError("model_binding must be the declared model slot")
        if not isinstance(self.manager, ComponentManager):
            raise TypeError("manager must be ComponentManager")
        if not isinstance(self.handle, PreparedComponentHandle):
            raise TypeError("handle must be PreparedComponentHandle")
        if not isinstance(self.runtime_facts, RuntimeFacts):
            raise TypeError("runtime_facts must be RuntimeFacts")
        if not isinstance(self.model_runtime_contract, RuntimeBoundContract):
            raise TypeError("model_runtime_contract must be RuntimeBoundContract")
        if self.requirement_set is not None and not isinstance(
            self.requirement_set, PreprocessRequirementSet
        ):
            raise TypeError("requirement_set must be PreprocessRequirementSet or None")


@dataclass(frozen=True, slots=True)
class PreprocessIdentityResult:
    """Typed receipt binding one plan identity to its consumer requirements."""

    preprocess_identity: str
    requirement_set_id: str

    def __post_init__(self) -> None:
        _digest("preprocess_identity", self.preprocess_identity)
        _digest("requirement_set_id", self.requirement_set_id)


@dataclass(frozen=True, slots=True)
class PreprocessRuntimeBinding:
    """Effective consumer requirements plus the resolved preprocess identity."""

    requirement_set: PreprocessRequirementSet
    identity_result: PreprocessIdentityResult

    def __post_init__(self) -> None:
        if not isinstance(self.requirement_set, PreprocessRequirementSet):
            raise TypeError("requirement_set must be PreprocessRequirementSet")
        if not isinstance(self.identity_result, PreprocessIdentityResult):
            raise TypeError("identity_result must be PreprocessIdentityResult")
        if (
            self.identity_result.requirement_set_id
            != self.requirement_set.requirement_set_id
        ):
            raise ComponentRuntimeBindingError(
                "preprocess identity receipt differs from effective requirements"
            )


@runtime_checkable
class PreprocessIdentityProvider(Protocol):
    """Identify the concrete preprocess implementation and its runtime inputs."""

    def resolve(
        self, request: PreprocessIdentityRequest
    ) -> PreprocessIdentityResult: ...


def _consumer_identity(kind: str, payload: object) -> str:
    encoded = json.dumps(
        to_plain_dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{kind}.v1:{hashlib.sha256(encoded).hexdigest()}"


def _require_binding(
    bindings: dict[str, RuntimeComponentBinding],
    slot: str,
    kind: str,
) -> RuntimeComponentBinding:
    try:
        binding = bindings[slot]
    except KeyError:
        raise PreprocessRequirementCompileError(
            f"resolved runtime graph is missing {slot!r}"
        ) from None
    if not isinstance(binding, RuntimeComponentBinding):
        raise TypeError("runtime graph values must be RuntimeComponentBinding")
    if binding.kind != kind:
        raise PreprocessRequirementCompileError(
            f"resolved slot {slot!r} is not a {kind} component"
        )
    return binding


def compile_preprocess_requirement_set(
    *,
    bindings: dict[str, RuntimeComponentBinding],
    producer_spec: PreprocessProducerSpec,
    model_consumer: ModelPreprocessConsumerSpec,
    algorithm: AlgorithmExecutionPlan,
) -> PreprocessRequirementSet:
    """Compile the exact union of model, algorithm, and conditioner demands."""

    if not isinstance(bindings, dict) or not bindings:
        raise TypeError("bindings must be a non-empty dict")
    if not isinstance(producer_spec, PreprocessProducerSpec):
        raise TypeError("producer_spec must be a PreprocessProducerSpec")
    if not isinstance(model_consumer, ModelPreprocessConsumerSpec):
        raise TypeError("model_consumer must be a ModelPreprocessConsumerSpec")
    if not isinstance(algorithm, AlgorithmExecutionPlan):
        raise TypeError("algorithm must be an AlgorithmExecutionPlan")

    model = _require_binding(bindings, "model", "model")
    rollout = _require_binding(bindings, "rollout", "rollout")
    declared_model = model.declared_contract
    if not isinstance(declared_model, DeclaredContract) or declared_model.model is None:
        raise PreprocessRequirementCompileError(
            "resolved model binding has no typed ModelContract"
        )
    payload_types = declared_model.model.condition_payload_types
    if (
        producer_spec.port.output_payload_type not in payload_types
        or model_consumer.payload_type not in payload_types
    ):
        raise PreprocessRequirementCompileError(
            "model producer/consumer payload type is absent from ModelContract"
        )

    effective_fields = model_consumer.required_output_fields
    negative_fields = (
        model_consumer.negative_output_fields
        if model_consumer.uses_negative_condition
        else ()
    )
    requirements: list[PreprocessConsumerRequirement] = [
        PreprocessConsumerRequirement(
            consumer_identity=_consumer_identity(
                "rollout-model-conditioning",
                {
                    "resolved_rollout_manifest": to_plain_dict(
                        rollout.declaration.to_identity_payload()
                    ),
                    "model_consumer_spec_id": model_consumer.consumer_spec_id,
                },
            ),
            provider=PreprocessRequirementProvider.MODEL,
            payload_type=model_consumer.payload_type,
            required_modalities=model_consumer.required_modalities,
            required_output_fields=effective_fields,
            required_negative_condition_fields=negative_fields,
            requires_negative_condition=model_consumer.uses_negative_condition,
        )
    ]

    if algorithm.requires_reference_statistics:
        requirements.append(
            PreprocessConsumerRequirement(
                consumer_identity=_consumer_identity(
                    "algorithm-reference-conditioning",
                    {
                        "algorithm_plan_id": algorithm.plan_id,
                        "reference_requirement": algorithm.reference_requirement.value,
                        "model_consumer_spec_id": model_consumer.consumer_spec_id,
                    },
                ),
                provider=PreprocessRequirementProvider.MODEL,
                payload_type=model_consumer.payload_type,
                required_modalities=model_consumer.required_modalities,
                required_output_fields=effective_fields,
                required_negative_condition_fields=negative_fields,
                requires_negative_condition=model_consumer.uses_negative_condition,
            )
        )

    conditioner = bindings.get("conditioner")
    if conditioner is not None:
        if not isinstance(conditioner, RuntimeComponentBinding):
            raise TypeError("conditioner binding must be RuntimeComponentBinding")
        if conditioner.kind != "conditioner":
            raise PreprocessRequirementCompileError(
                "conditioner slot is not a Conditioner component"
            )
        declared = conditioner.declared_contract
        if not isinstance(declared, DeclaredContract) or not isinstance(
            declared.conditioner,
            ConditionerContract,
        ):
            raise PreprocessRequirementCompileError(
                "resolved Conditioner has no typed ConditionerContract"
            )
        contract = declared.conditioner
        if not contract.required_modalities or not contract.provided_output_fields:
            raise PreprocessRequirementCompileError(
                "Conditioner must explicitly declare required modalities and "
                "provided output fields"
            )
        requirements.append(
            PreprocessConsumerRequirement(
                consumer_identity=_consumer_identity(
                    "conditioner-payload",
                    {
                        "resolved_conditioner_manifest": to_plain_dict(
                            conditioner.declaration.to_identity_payload()
                        ),
                        "algorithm_trajectory_kind": algorithm.trajectory_kind.value,
                    },
                ),
                provider=PreprocessRequirementProvider.CONDITIONER,
                payload_type=contract.payload_type,
                required_modalities=contract.required_modalities,
                required_output_fields=contract.provided_output_fields,
            )
        )

    result = PreprocessRequirementSet(tuple(requirements))
    result.validate_model_producer(producer_spec.port)
    return result


def resolve_preprocess_runtime_binding(
    *,
    materialized: MaterializedRecipe,
    bindings: dict[str, RuntimeComponentBinding],
    model_binding: RuntimeComponentBinding,
    manager: ComponentManager,
    handle: PreparedComponentHandle,
    runtime_facts: RuntimeFacts,
    model_runtime_contract: RuntimeBoundContract,
    algorithm: AlgorithmExecutionPlan,
    identity_provider: PreprocessIdentityProvider,
) -> PreprocessRuntimeBinding:
    """Compile demands and bind them to the concrete prepared preprocess path."""

    adapter = model_binding.instance
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("model binding must contain a ModelAdapter")
    producer_spec = adapter.describe_preprocess()
    if not isinstance(producer_spec, PreprocessProducerSpec):
        raise TypeError("describe_preprocess() must return PreprocessProducerSpec")
    model_consumer = adapter.describe_preprocess_consumption()
    if not isinstance(model_consumer, ModelPreprocessConsumerSpec):
        raise TypeError(
            "describe_preprocess_consumption() must return ModelPreprocessConsumerSpec"
        )
    requirement_set = compile_preprocess_requirement_set(
        bindings=bindings,
        producer_spec=producer_spec,
        model_consumer=model_consumer,
        algorithm=algorithm,
    )
    identity_result = identity_provider.resolve(
        PreprocessIdentityRequest(
            materialized=materialized,
            model_binding=model_binding,
            manager=manager,
            handle=handle,
            runtime_facts=runtime_facts,
            model_runtime_contract=model_runtime_contract,
            requirement_set=requirement_set,
        )
    )
    if not isinstance(identity_result, PreprocessIdentityResult):
        raise TypeError(
            "preprocess identity provider must return PreprocessIdentityResult"
        )
    return PreprocessRuntimeBinding(
        requirement_set=requirement_set,
        identity_result=identity_result,
    )
