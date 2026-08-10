"""Typed runtime facade for a compiler-materialized algorithm.

This module deliberately does not resolve compatibility, registry entries, or
internal callbacks.  The composition root supplies a validated core binding,
an immutable materialization spec, and one typed execution port.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from visual_rl.algorithms.modules.descriptor import AlgorithmBlueprint
from visual_rl.core.contracts import (
    AlgorithmMaterializationSpec,
    AlgorithmRequirements,
    AlgorithmStepResult,
    ExecutionPolicyReceipt,
    ModelAlgorithmBinding,
    PolicyRuntimePort,
)

__all__ = (
    "AlgorithmExecutionPort",
    "AlgorithmMaterializationPort",
    "AlgorithmMaterializationRequest",
    "AlgorithmModule",
    "AlgorithmModuleState",
    "BoundAlgorithm",
)


class AlgorithmModuleState(str, Enum):
    NEW = "new"
    PREPARED = "prepared"
    CLOSED = "closed"


@runtime_checkable
class AlgorithmExecutionPort(Protocol):
    """The assembled internal trainer graph behind a stable typed boundary."""

    def prepare_run(self, context: object) -> None: ...

    def run_iteration(self, optimizer_step: int) -> object: ...

    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AlgorithmMaterializationRequest:
    component_id: str
    config: object
    blueprint: AlgorithmBlueprint
    requirements: AlgorithmRequirements
    spec: AlgorithmMaterializationSpec
    execution_policy: ExecutionPolicyReceipt
    policy: PolicyRuntimePort
    binding: ModelAlgorithmBinding

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError("component_id must be non-empty")
        if not isinstance(self.blueprint, AlgorithmBlueprint):
            raise TypeError("blueprint must be an AlgorithmBlueprint")
        if self.blueprint.algorithm_component_id != self.component_id:
            raise ValueError("blueprint belongs to a different algorithm")
        if not isinstance(self.requirements, AlgorithmRequirements):
            raise TypeError("requirements must be AlgorithmRequirements")
        if not isinstance(self.spec, AlgorithmMaterializationSpec):
            raise TypeError("spec must be an AlgorithmMaterializationSpec")
        if self.spec.algorithm_component_id != self.component_id:
            raise ValueError("materialization spec belongs to a different algorithm")
        if self.spec.blueprint_id != self.blueprint.blueprint_id:
            raise ValueError("materialization spec blueprint identity mismatch")
        if self.spec.requirement_id != self.requirements.requirement_id:
            raise ValueError("materialization spec requirement identity mismatch")
        for field_name in (
            "trajectory_kind",
            "grouping",
            "reference_requirement",
        ):
            if getattr(self.spec, field_name) != getattr(
                self.requirements,
                field_name,
            ):
                raise ValueError(
                    f"materialization spec {field_name} differs from requirements"
                )
        if self.spec.likelihood_semantics not in (
            self.requirements.likelihood_semantics
        ):
            raise ValueError(
                "materialization spec likelihood_semantics differs from requirements"
            )
        if (
            self.spec.requires_reference_statistics
            is not self.requirements.reference_required
        ):
            raise ValueError(
                "materialization spec requires_reference_statistics differs from "
                "requirements"
            )
        if self.spec.objective_identity != self.blueprint.objective_identity:
            raise ValueError("materialization spec objective identity mismatch")
        if self.spec.beta != self.blueprint.beta:
            raise ValueError("materialization spec beta differs from the blueprint")
        if type(self.execution_policy) is not ExecutionPolicyReceipt:
            raise TypeError("execution_policy must be an ExecutionPolicyReceipt")
        try:
            self.execution_policy.validated_projection(self.spec.execution_policy_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "materialization spec execution policy identity/projection mismatch"
            ) from exc
        if not isinstance(self.policy, PolicyRuntimePort):
            raise TypeError("policy must implement PolicyRuntimePort")
        if not isinstance(self.binding, ModelAlgorithmBinding):
            raise TypeError("binding must be a ModelAlgorithmBinding")
        if self.binding.model_capabilities != self.policy.capabilities:
            raise ValueError("binding capabilities differ from the policy port")
        if self.binding.algorithm_requirements != self.requirements:
            raise ValueError("binding requirements differ from the algorithm")


@runtime_checkable
class AlgorithmMaterializationPort(Protocol):
    def materialize(
        self,
        request: AlgorithmMaterializationRequest,
    ) -> AlgorithmExecutionPort: ...


class AlgorithmModule(ABC):
    """Runtime algorithm axis with no registry or compatibility ownership."""

    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = ""

    @property
    @abstractmethod
    def config(self) -> object:
        raise NotImplementedError

    @property
    @abstractmethod
    def requirements(self) -> AlgorithmRequirements:
        raise NotImplementedError

    @property
    @abstractmethod
    def blueprint(self) -> AlgorithmBlueprint:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> AlgorithmModule:
        raise NotImplementedError

    def materialize(
        self,
        policy: PolicyRuntimePort,
        binding: ModelAlgorithmBinding,
        spec: AlgorithmMaterializationSpec,
        materializer: AlgorithmMaterializationPort,
        *,
        execution_policy: ExecutionPolicyReceipt,
    ) -> BoundAlgorithm:
        request = AlgorithmMaterializationRequest(
            component_id=self.blueprint.algorithm_component_id,
            config=self.config,
            blueprint=self.blueprint,
            requirements=self.requirements,
            spec=spec,
            execution_policy=execution_policy,
            policy=policy,
            binding=binding,
        )
        if not isinstance(materializer, AlgorithmMaterializationPort):
            raise TypeError("materializer must implement AlgorithmMaterializationPort")
        execution = materializer.materialize(request)
        if not isinstance(execution, AlgorithmExecutionPort):
            raise TypeError("materializer must return AlgorithmExecutionPort")
        return BoundAlgorithm(request=request, execution=execution)


class BoundAlgorithm:
    """Lifecycle-checked facade over one typed materialized execution port."""

    _STATE_KEYS = frozenset(
        {
            "schema_version",
            "component_id",
            "materialization_spec_id",
            "algorithm_binding_id",
            "lifecycle_state",
            "execution_state",
        }
    )
    _RESUMABLE_LIFECYCLE_STATES = frozenset(
        {
            AlgorithmModuleState.NEW.value,
            AlgorithmModuleState.PREPARED.value,
        }
    )

    def __init__(
        self,
        *,
        request: AlgorithmMaterializationRequest,
        execution: AlgorithmExecutionPort,
    ) -> None:
        if not isinstance(request, AlgorithmMaterializationRequest):
            raise TypeError("request must be an AlgorithmMaterializationRequest")
        if not isinstance(execution, AlgorithmExecutionPort):
            raise TypeError("execution must implement AlgorithmExecutionPort")
        self._request = request
        self._execution = execution
        self._state = AlgorithmModuleState.NEW

    @property
    def component_id(self) -> str:
        return self._request.component_id

    @property
    def blueprint(self) -> AlgorithmBlueprint:
        return self._request.blueprint

    @property
    def requirements(self) -> AlgorithmRequirements:
        return self._request.requirements

    @property
    def spec(self) -> AlgorithmMaterializationSpec:
        return self._request.spec

    @property
    def binding(self) -> ModelAlgorithmBinding:
        return self._request.binding

    @property
    def policy(self) -> PolicyRuntimePort:
        """Return the exact G3-bound policy facade supplied at materialization."""

        return self._request.policy

    @property
    def state(self) -> AlgorithmModuleState:
        return self._state

    def prepare_run(self, context: object) -> None:
        if self._state is not AlgorithmModuleState.NEW:
            raise RuntimeError("prepare_run() may be called exactly once")
        self._execution.prepare_run(context)
        self._state = AlgorithmModuleState.PREPARED

    def run_iteration(self, optimizer_step: int) -> AlgorithmStepResult:
        if self._state is not AlgorithmModuleState.PREPARED:
            raise RuntimeError("algorithm must be prepared before run_iteration()")
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("optimizer_step must be a non-negative integer")
        observed = self._execution.run_iteration(optimizer_step)
        if isinstance(observed, AlgorithmStepResult):
            if observed.algorithm_binding_id != self.binding.binding_id:
                raise ValueError("execution returned a different binding id")
            return observed
        return AlgorithmStepResult(
            optimizer_step=optimizer_step,
            iteration=observed,
            algorithm_binding_id=self.binding.binding_id,
        )

    def state_dict(self) -> dict[str, object]:
        if self._state is AlgorithmModuleState.CLOSED:
            raise RuntimeError("a closed algorithm has no capturable state")
        payload = self._execution.state_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("execution state_dict() must return a mapping")
        return {
            "schema_version": 1,
            "component_id": self.component_id,
            "materialization_spec_id": self.spec.spec_id,
            "algorithm_binding_id": self.binding.binding_id,
            "lifecycle_state": self._state.value,
            "execution_state": dict(payload),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._state is not AlgorithmModuleState.NEW:
            raise RuntimeError("algorithm state must be restored before prepare_run()")
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if set(state) != self._STATE_KEYS:
            raise ValueError("algorithm state has unexpected fields")
        expected = {
            "schema_version": 1,
            "component_id": self.component_id,
            "materialization_spec_id": self.spec.spec_id,
            "algorithm_binding_id": self.binding.binding_id,
        }
        for name, value in expected.items():
            if state[name] != value:
                raise ValueError(f"algorithm state {name} mismatch")
        lifecycle_state = state["lifecycle_state"]
        if (
            type(lifecycle_state) is not str
            or lifecycle_state not in self._RESUMABLE_LIFECYCLE_STATES
        ):
            raise ValueError("algorithm state lifecycle is not resumable")
        payload = state["execution_state"]
        if not isinstance(payload, Mapping):
            raise TypeError("execution_state must be a mapping")
        self._execution.load_state_dict(payload)

    def close(self) -> None:
        if self._state is AlgorithmModuleState.CLOSED:
            return
        self._execution.close()
        self._state = AlgorithmModuleState.CLOSED
