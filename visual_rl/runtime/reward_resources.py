"""Identity-deduplicated lifecycle owner for heavy reward resources."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from visual_rl.algorithms.rewards.components import (
    PointwiseRewardAdapterConfig,
    RewardResourceDescriptor,
)
from visual_rl.algorithms.rewards.clients.world_r1 import (
    WORLD_R1_RESOURCE_PROTOCOL,
    WorldR1HealthAttestation,
)
from visual_rl.algorithms.rewards.input_selection import RewardInputSelectionPolicy
from visual_rl.core.contracts import LogicalRewardSpec
from visual_rl.algorithms.rewards.resource_port import RewardResourceState
from visual_rl.core.contracts import RewardPlanSpec
from visual_rl.composition.config.specs import LaunchSpec, RewardRuntimeBindingSpec
from visual_rl.composition.preflight.types import RuntimeFacts
from visual_rl.composition.recipes.schema import MaterializedRecipe
from visual_rl.core.protocols.world_r1 import (
    PROTOCOL_VERSION,
    REWARD_3D,
    REWARD_GENERAL,
    validate_server_revision,
)
from visual_rl.core.serialization import canonical_json_text
from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RuntimeBuildContext,
    to_plain_dict,
)

if TYPE_CHECKING:
    from visual_rl.runtime.component_graph import RuntimeComponentBinding
    from visual_rl.runtime.resources import (
        DefaultRuntimeResourceContainer,
    )


_BOUND_ID_DOMAIN = b"visual_rl.bound-reward-resource.v1\0"
_ACQUISITION_REQUEST_FINGERPRINT_DOMAIN = (
    b"visual_rl.reward-resource-acquisition-request.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MATERIALIZED_SPEC = re.compile(r"^reward-resource-spec\.v1:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RewardLogicalBinding:
    """One logical reward adapter bound to one deduplicated physical resource."""

    logical_id: str
    graph_binding: RuntimeComponentBinding
    plan_binding: LogicalRewardSpec
    descriptor: RewardResourceDescriptor


def resolve_reward_bindings(
    *,
    materialized: MaterializedRecipe,
    launch: LaunchSpec,
    runtime_facts: RuntimeFacts,
    plan: RewardPlanSpec,
    bindings: Mapping[str, RuntimeComponentBinding],
) -> tuple[
    tuple[RewardLogicalBinding, ...],
    tuple[RewardResourceAcquireRequest, ...],
]:
    """Validate logical reward slots and reconstruct exact acquisition requests."""

    from visual_rl.runtime.component_graph import (
        ComponentRuntimeBindingError,
        RuntimeComponentBinding,
    )
    if not isinstance(plan, RewardPlanSpec) or not plan.materialized:
        raise ComponentRuntimeBindingError("G3 requires a materialized RewardPlanSpec")
    expected_slots = {f"rewards.{item}" for item in plan.logical_reward_ids}
    actual_slots = {
        slot for slot, binding in bindings.items() if binding.kind == "reward"
    }
    if actual_slots != expected_slots:
        raise ComponentRuntimeBindingError(
            "runtime reward slots do not exactly cover the reward plan: "
            f"missing={sorted(expected_slots - actual_slots)}, "
            f"unknown={sorted(actual_slots - expected_slots)}"
        )

    logical: list[RewardLogicalBinding] = []
    component_instances: set[int] = set()
    for logical_id in plan.logical_reward_ids:
        slot = f"rewards.{logical_id}"
        graph_binding = bindings[slot]
        if not isinstance(graph_binding, RuntimeComponentBinding):
            raise TypeError("reward graph bindings must be RuntimeComponentBinding")
        if graph_binding.slot != slot:
            raise ComponentRuntimeBindingError(
                f"reward slot identity drifted for {slot!r}"
            )
        if id(graph_binding.instance) in component_instances:
            raise ComponentRuntimeBindingError(
                "logical reward slots must use distinct adapter instances"
            )
        component_instances.add(id(graph_binding.instance))

        config = graph_binding.config
        if not isinstance(config, PointwiseRewardAdapterConfig):
            raise TypeError(f"{slot} must use PointwiseRewardAdapterConfig at G3")
        descriptor = config.resource
        if not isinstance(descriptor, RewardResourceDescriptor):
            raise TypeError(f"{slot} has no RewardResourceDescriptor")
        validate_descriptor = getattr(
            graph_binding.instance,
            "validate_resource_descriptor",
            None,
        )
        if callable(validate_descriptor):
            try:
                validate_descriptor(descriptor)
            except (TypeError, ValueError) as exc:
                raise ComponentRuntimeBindingError(
                    "reward resource factory role mismatch for logical id "
                    f"{logical_id!r}: {exc}"
                ) from exc
        plan_binding = plan.logical_reward(logical_id)
        if (
            plan_binding.component_declaration_id
            != graph_binding.declaration.declaration_id
        ):
            raise ComponentRuntimeBindingError(
                f"reward declaration drifted for logical id {logical_id!r}"
            )
        resource_spec = plan.resource(plan_binding.resource_identity)
        if to_plain_dict(resource_spec.descriptor) != descriptor.to_payload():
            raise ComponentRuntimeBindingError(
                f"reward descriptor drifted for logical id {logical_id!r}"
            )
        logical.append(
            RewardLogicalBinding(
                logical_id=logical_id,
                graph_binding=graph_binding,
                plan_binding=plan_binding,
                descriptor=descriptor,
            )
        )

    try:
        compiled_plan, acquisition_requests = compile_reward_resource_acquisition(
            materialized,
            launch,
            runtime_facts,
        )
    except RewardAcquisitionPlanningError as exc:
        raise ComponentRuntimeBindingError(
            "reward acquisition inputs cannot be reconstructed at G3"
        ) from exc
    if compiled_plan != plan:
        raise ComponentRuntimeBindingError(
            "G3 reward plan differs from the pre-model acquisition plan"
        )
    request_by_spec = {
        item.reward_resource_spec_id: item for item in acquisition_requests
    }
    for item in logical:
        acquire_request = request_by_spec.get(item.plan_binding.resource_identity)
        if acquire_request is None or acquire_request.descriptor != item.descriptor:
            raise ComponentRuntimeBindingError(
                "graph reward descriptor differs from the pre-model acquisition "
                f"request for logical id {item.logical_id!r}"
            )
    return tuple(logical), acquisition_requests


def acquire_reward_view(
    container: DefaultRuntimeResourceContainer,
    plan: RewardPlanSpec,
    requests: tuple[RewardResourceAcquireRequest, ...],
) -> RewardPoolView:
    """Acquire or exactly reuse the session-owned physical reward pool."""

    from visual_rl.runtime.component_graph import ComponentRuntimeBindingError
    if container.state is RewardResourceState.DECLARED:
        view = container.acquire(plan, requests)
    elif container.state is RewardResourceState.ACQUIRED:
        if container.plan != plan:
            raise ComponentRuntimeBindingError(
                "pre-acquired reward container holds a different plan"
            )
        try:
            container.assert_acquisition_requests_match(plan, requests)
        except RuntimeResourceAcquisitionError as exc:
            raise ComponentRuntimeBindingError(
                "pre-acquired reward container acquisition inputs differ"
            ) from exc
        view = container.view()
    else:
        raise ComponentRuntimeBindingError(
            "G3 reward binding requires a DECLARED or ACQUIRED container; "
            f"found {container.state.value!r}"
        )
    if not isinstance(view, RewardPoolView):
        raise TypeError("runtime resource container must return RewardPoolView")
    if container.is_active:
        raise ComponentRuntimeBindingError("G3 must not activate reward resources")
    return view


def bind_logical_rewards(
    logical_rewards: tuple[RewardLogicalBinding, ...],
    view: RewardPoolView,
) -> None:
    """Bind each logical reward adapter once to its non-owning pool handle."""

    from visual_rl.runtime.component_graph import ComponentRuntimeBindingError

    for logical in logical_rewards:
        adapter = logical.graph_binding.instance
        bind_resource = getattr(adapter, "bind_resource", None)
        is_bound_to = getattr(adapter, "is_bound_to", None)
        if not callable(bind_resource) or not callable(is_bound_to):
            raise TypeError(
                f"reward adapter {logical.logical_id!r} lacks its bind-once port"
            )
        handle = view.handle(logical.plan_binding.resource_identity)
        state = getattr(adapter, "resource_state", None)
        if state is RewardResourceState.DECLARED:
            bind_resource(handle)
        elif state is RewardResourceState.ACQUIRED:
            if not is_bound_to(handle):
                raise ComponentRuntimeBindingError(
                    f"reward adapter {logical.logical_id!r} is prebound to a "
                    "different handle"
                )
        else:
            state_name = getattr(state, "value", repr(state))
            raise ComponentRuntimeBindingError(
                f"reward adapter {logical.logical_id!r} has invalid G3 state "
                f"{state_name!r}"
            )
        if not is_bound_to(handle) or (
            getattr(adapter, "resource_state", None) is not RewardResourceState.ACQUIRED
        ):
            raise ComponentRuntimeBindingError(
                f"reward adapter {logical.logical_id!r} did not retain the exact "
                "ACQUIRED handle"
            )


class BorrowedRewardResourceHandle:
    """Non-owning handle; only its owner pool may activate or close it."""

    __slots__ = ("_resource", "_resource_identity", "_state")

    def __init__(self, resource_identity: str, resource: object) -> None:
        if not isinstance(resource_identity, str) or not resource_identity:
            raise ValueError("resource_identity must be a non-empty string")
        if not callable(getattr(resource, "close", None)):
            raise TypeError("reward resources must define close()")
        self._resource_identity = resource_identity
        self._resource = resource
        self._state = RewardResourceState.ACQUIRED

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    @property
    def state(self) -> RewardResourceState:
        return self._state

    def require_method(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("resource method name must be non-empty")
        if not callable(getattr(self._resource, name, None)):
            raise TypeError(f"reward resource must define {name}()")

    def resource_for_execution(self) -> object:
        """Borrow the resource only while the owner marks it ACTIVE."""

        if self._state is not RewardResourceState.ACTIVE:
            raise RuntimeError(
                "reward resource cannot execute unless its state is ACTIVE; "
                f"identity={self._resource_identity!r}, state={self._state.value!r}"
            )
        return self._resource

    def _activate(self) -> None:
        if self._state is not RewardResourceState.ACQUIRED:
            raise RuntimeError(
                "reward resource activation requires ACQUIRED state; "
                f"identity={self._resource_identity!r}, state={self._state.value!r}"
            )
        activate = getattr(self._resource, "activate", None)
        if callable(activate):
            activate()
        self._state = RewardResourceState.ACTIVE

    def _mark_closed(self) -> None:
        self._state = RewardResourceState.CLOSED


class RewardPoolView:
    """Stage-facing view over session-owned handles; deliberately not closable."""

    __slots__ = ("_handles", "plan")

    def __init__(
        self,
        plan: RewardPlanSpec,
        handles: dict[str, BorrowedRewardResourceHandle],
    ) -> None:
        if not isinstance(plan, RewardPlanSpec):
            raise TypeError("plan must be a RewardPlanSpec")
        if set(handles) != set(plan.resource_identities):
            raise ValueError("borrowed handles must exactly cover the reward plan")
        if any(
            not isinstance(handle, BorrowedRewardResourceHandle)
            for handle in handles.values()
        ):
            raise TypeError("handles must contain BorrowedRewardResourceHandle values")
        self.plan = plan
        self._handles = MappingProxyType(dict(handles))

    @property
    def resource_identities(self) -> tuple[str, ...]:
        return self.plan.resource_identities

    def handle(self, resource_identity: str) -> BorrowedRewardResourceHandle:
        try:
            return self._handles[resource_identity]
        except KeyError as exc:
            raise KeyError(
                f"unknown reward resource identity {resource_identity!r}"
            ) from exc

    def get(self, resource_identity: str) -> object:
        """Return an ACTIVE borrowed resource without transferring ownership."""

        return self.handle(resource_identity).resource_for_execution()


class RewardPoolBuildError(RuntimeError):
    """A resource failed to construct after prior resources were unwound."""

    def __init__(
        self,
        resource_identity: str,
        cleanup_errors: tuple[BaseException, ...],
    ) -> None:
        super().__init__(f"failed to construct reward resource {resource_identity!r}")
        self.resource_identity = resource_identity
        self.cleanup_errors = cleanup_errors


class RewardPoolCloseError(RuntimeError):
    """One or more resources failed while the complete pool was unwound."""

    def __init__(self, errors: tuple[tuple[str, BaseException], ...]) -> None:
        super().__init__("one or more reward resources failed to close")
        self.errors = errors


class RewardPoolActivationError(RuntimeError):
    """Activation failed and the owner pool was closed fail-closed."""

    def __init__(
        self,
        resource_identity: str,
        cleanup_errors: tuple[tuple[str, BaseException], ...],
    ) -> None:
        super().__init__(f"failed to activate reward resource {resource_identity!r}")
        self.resource_identity = resource_identity
        self.cleanup_errors = cleanup_errors


class RewardPool:
    """Session-side owner: acquire once, activate explicitly, close in reverse."""

    def __init__(
        self,
        plan: RewardPlanSpec,
        factory: Callable[[str], object],
    ) -> None:
        if not isinstance(plan, RewardPlanSpec):
            raise TypeError("plan must be a RewardPlanSpec")
        if plan.provisional:
            raise ValueError(
                "RewardPool cannot deduplicate provisional reward descriptors; "
                "compile from MaterializedRecipe first"
            )
        if not callable(factory):
            raise TypeError("factory must be callable")
        self.plan = plan
        self._resources: dict[str, object] = {}
        self._handles: dict[str, BorrowedRewardResourceHandle] = {}
        self._construction_order: list[str] = []
        self._closed = False
        self._active = False
        current_identity = "<none>"
        try:
            for current_identity in plan.resource_identities:
                resource = factory(current_identity)
                if not callable(getattr(resource, "close", None)):
                    raise TypeError("reward resources must define close()")
                self._resources[current_identity] = resource
                self._handles[current_identity] = BorrowedRewardResourceHandle(
                    current_identity,
                    resource,
                )
                self._construction_order.append(current_identity)
        except BaseException as exc:
            cleanup_errors = self._close_constructed()
            self._closed = True
            raise RewardPoolBuildError(current_identity, cleanup_errors) from exc

    @property
    def resource_identities(self) -> tuple[str, ...]:
        return tuple(self._construction_order)

    @property
    def is_active(self) -> bool:
        return self._active and not self._closed

    def view(self) -> RewardPoolView:
        if self._closed:
            raise RuntimeError("RewardPool is closed")
        return RewardPoolView(self.plan, self._handles)

    def activate(self) -> None:
        """Move every acquired resource to ACTIVE exactly once."""

        if self._closed:
            raise RuntimeError("RewardPool is closed")
        if self._active:
            raise RuntimeError("RewardPool is already ACTIVE")
        current_identity = "<none>"
        try:
            for current_identity in self._construction_order:
                self._handles[current_identity]._activate()
        except BaseException as exc:
            cleanup_errors = self._close_constructed_with_identity()
            self._closed = True
            raise RewardPoolActivationError(
                current_identity,
                cleanup_errors,
            ) from exc
        self._active = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._active = False
        errors = self._close_constructed_with_identity()
        if errors:
            raise RewardPoolCloseError(errors)

    def __enter__(self) -> RewardPool:
        if self._closed:
            raise RuntimeError("RewardPool is closed")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _close_constructed(self) -> tuple[BaseException, ...]:
        return tuple(
            error for _identity, error in self._close_constructed_with_identity()
        )

    def _close_constructed_with_identity(
        self,
    ) -> tuple[tuple[str, BaseException], ...]:
        errors: list[tuple[str, BaseException]] = []
        for identity in reversed(self._construction_order):
            resource = self._resources.pop(identity)
            try:
                resource.close()  # type: ignore[attr-defined]
            except BaseException as exc:
                errors.append((identity, exc))
            finally:
                self._handles[identity]._mark_closed()
        self._construction_order.clear()
        return tuple(errors)


class RuntimeResourceAcquisitionError(RuntimeError):
    """Reward acquisition or exact pre-acquisition reuse validation failed."""


def _canonical_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _materialized_spec_id(value: object) -> str:
    if not isinstance(value, str) or not _MATERIALIZED_SPEC.fullmatch(value):
        raise ValueError(
            "reward_resource_spec_id must be a materialized reward resource id"
        )
    return value


@dataclass(frozen=True, slots=True)
class RewardResourceAcquireRequest:
    """Complete leaf-factory input for one unique materialized resource spec."""

    reward_resource_spec_id: str
    descriptor: RewardResourceDescriptor
    immutable_artifact_identity: FrozenMapping
    artifact_location: Path = field(compare=False, hash=False, repr=False)
    runtime_facts: RuntimeFacts = field(compare=False, hash=False, repr=False)
    runtime_binding: RewardRuntimeBindingSpec | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _materialized_spec_id(self.reward_resource_spec_id)
        if not isinstance(self.descriptor, RewardResourceDescriptor):
            raise TypeError("descriptor must be a RewardResourceDescriptor")
        if (
            not isinstance(self.immutable_artifact_identity, FrozenMapping)
            or not self.immutable_artifact_identity
        ):
            raise ValueError(
                "immutable_artifact_identity must be a non-empty FrozenMapping"
            )
        if not isinstance(self.artifact_location, Path) or not (
            self.artifact_location.is_absolute()
        ):
            raise ValueError("artifact_location must be an absolute Path")
        if not isinstance(self.runtime_facts, RuntimeFacts):
            raise TypeError("runtime_facts must be RuntimeFacts")
        if self.runtime_binding is not None:
            if not isinstance(self.runtime_binding, RewardRuntimeBindingSpec):
                raise TypeError(
                    "runtime_binding must be RewardRuntimeBindingSpec or None"
                )
            if self.runtime_binding.artifact_ref != self.descriptor.artifact_ref:
                raise ValueError("runtime binding artifact_ref differs from descriptor")


def _acquisition_request_fingerprint(
    request: RewardResourceAcquireRequest,
) -> str:
    """Hash every exact leaf-factory input into one canonical reuse guard."""

    if not isinstance(request, RewardResourceAcquireRequest):
        raise TypeError("request must be RewardResourceAcquireRequest")
    payload = {
        "schema_version": 1,
        "reward_resource_spec_id": request.reward_resource_spec_id,
        "descriptor": request.descriptor.to_payload(),
        "immutable_artifact_identity": to_plain_dict(
            request.immutable_artifact_identity
        ),
        "artifact_location": str(request.artifact_location),
        "runtime_facts": request.runtime_facts.to_payload(),
        "runtime_binding": (
            None
            if request.runtime_binding is None
            else request.runtime_binding.to_payload()
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_ACQUISITION_REQUEST_FINGERPRINT_DOMAIN + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RewardResourceBindingFacts:
    """Actual factory-observed facts, separate from recipe semantics."""

    endpoint_identity: str
    protocol: str
    protocol_version: str
    device: str
    dtype: str
    worker_domain: str

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_identity, str) or not _SHA256.fullmatch(
            self.endpoint_identity
        ):
            raise ValueError("endpoint_identity must be a lowercase SHA-256 digest")
        for name in (
            "protocol",
            "protocol_version",
            "device",
            "dtype",
            "worker_domain",
        ):
            _canonical_text(name, getattr(self, name))

    def to_payload(self) -> dict[str, str]:
        return {
            "endpoint_identity": self.endpoint_identity,
            "protocol": self.protocol,
            "protocol_version": self.protocol_version,
            "device": self.device,
            "dtype": self.dtype,
            "worker_domain": self.worker_domain,
        }


@dataclass(frozen=True, slots=True)
class AcquiredRewardResource:
    """Factory result whose physical resource ownership transfers to a pool."""

    resource: object = field(compare=False, hash=False, repr=False)
    binding_facts: RewardResourceBindingFacts

    def __post_init__(self) -> None:
        if not callable(getattr(self.resource, "close", None)):
            raise TypeError("acquired reward resource must define close()")
        if not isinstance(self.binding_facts, RewardResourceBindingFacts):
            raise TypeError("binding_facts must be RewardResourceBindingFacts")


@runtime_checkable
class RewardResourceFactory(Protocol):
    """Acquire one owned resource or roll it back before raising."""

    def __call__(
        self,
        request: RewardResourceAcquireRequest,
    ) -> AcquiredRewardResource: ...


def bound_reward_resource_payload(
    reward_resource_spec_id: str,
    binding_facts: RewardResourceBindingFacts,
) -> dict[str, object]:
    """Return the exact runtime-only payload hashed for G3 compatibility."""

    _materialized_spec_id(reward_resource_spec_id)
    if not isinstance(binding_facts, RewardResourceBindingFacts):
        raise TypeError("binding_facts must be RewardResourceBindingFacts")
    return {
        "schema_version": 1,
        "reward_resource_spec_id": reward_resource_spec_id,
        "actual": binding_facts.to_payload(),
    }


def bound_reward_resource_id(
    reward_resource_spec_id: str,
    binding_facts: RewardResourceBindingFacts,
) -> str:
    """Hash a domain-separated canonical runtime binding payload."""

    payload = bound_reward_resource_payload(
        reward_resource_spec_id,
        binding_facts,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_BOUND_ID_DOMAIN + encoded).hexdigest()


class RewardAcquisitionPlanningError(RuntimeError):
    """Materialized reward inputs cannot form one exact acquisition set."""


def compile_reward_resource_acquisition(
    materialized: MaterializedRecipe,
    launch: LaunchSpec,
    runtime_facts: RuntimeFacts,
) -> tuple[RewardPlanSpec, tuple[RewardResourceAcquireRequest, ...]]:
    """Return the recipe-owned plan and one request per physical resource.

    No reward route or descriptor is reconstructed here.  Runtime-only paths,
    endpoints, devices, and worker placement are joined to the already
    materialized :class:`RewardResourceSpec` solely for acquisition.
    """

    if not isinstance(materialized, MaterializedRecipe):
        raise TypeError("materialized must be a MaterializedRecipe")
    if not isinstance(launch, LaunchSpec):
        raise TypeError("launch must be a LaunchSpec")
    if not isinstance(runtime_facts, RuntimeFacts):
        raise TypeError("runtime_facts must be RuntimeFacts")

    plan = materialized.reward_plan
    if not isinstance(plan, RewardPlanSpec) or not plan.materialized:
        raise RewardAcquisitionPlanningError(
            "reward acquisition requires a materialized RewardPlanSpec"
        )

    requests: list[RewardResourceAcquireRequest] = []
    for resource in plan.resources:
        try:
            descriptor = RewardResourceDescriptor.from_mapping(resource.descriptor)
        except (TypeError, ValueError) as exc:
            raise RewardAcquisitionPlanningError(
                f"reward resource {resource.resource_identity!r} has an invalid "
                "typed descriptor"
            ) from exc
        immutable_artifact_identity = resource.artifact_identity
        if not isinstance(immutable_artifact_identity, FrozenMapping) or not (
            immutable_artifact_identity
        ):
            raise RewardAcquisitionPlanningError(
                f"reward artifact identity {resource.artifact_ref!r} is invalid"
            )
        try:
            artifact_location = launch.artifacts.reward(resource.artifact_ref)
        except (KeyError, TypeError, ValueError) as exc:
            raise RewardAcquisitionPlanningError(
                f"reward artifact {resource.artifact_ref!r} has no launch location"
            ) from exc

        runtime_binding = launch.reward_runtime_binding(resource.artifact_ref)
        _validate_runtime_binding(
            descriptor,
            runtime_binding=runtime_binding,
        )
        requests.append(
            RewardResourceAcquireRequest(
                reward_resource_spec_id=resource.resource_identity,
                descriptor=descriptor,
                immutable_artifact_identity=immutable_artifact_identity,
                artifact_location=artifact_location,
                runtime_facts=runtime_facts,
                runtime_binding=runtime_binding,
            )
        )

    request_ids = tuple(item.reward_resource_spec_id for item in requests)
    if request_ids != plan.resource_identities:
        raise RewardAcquisitionPlanningError(
            "reward acquisition requests do not exactly cover the materialized plan"
        )
    return plan, tuple(requests)


def _validate_runtime_binding(
    descriptor: RewardResourceDescriptor,
    *,
    runtime_binding: object | None,
) -> None:
    allowed = descriptor.allowed_runtime_policy
    if runtime_binding is None:
        if "in_process" not in allowed.allowed_worker_domains:
            raise RewardAcquisitionPlanningError(
                f"reward artifact {descriptor.artifact_ref!r} requires an explicit "
                "launch.reward_runtime_bindings entry"
            )
        return

    execution_domain = getattr(runtime_binding, "execution_domain", None)
    device = getattr(runtime_binding, "device", None)
    dtype = getattr(runtime_binding, "dtype", None)
    if execution_domain not in allowed.allowed_worker_domains:
        raise RewardAcquisitionPlanningError(
            f"reward runtime domain for {descriptor.artifact_ref!r} violates its "
            "declared policy"
        )
    if not isinstance(device, str) or (
        device.split(":", 1)[0] not in allowed.allowed_devices
    ):
        raise RewardAcquisitionPlanningError(
            f"reward runtime device for {descriptor.artifact_ref!r} violates its "
            "declared policy"
        )
    if dtype not in allowed.allowed_dtypes:
        raise RewardAcquisitionPlanningError(
            f"reward runtime dtype for {descriptor.artifact_ref!r} violates its "
            "declared policy"
        )


_ENDPOINT_ID_DOMAIN = b"visual-rl.default-reward-endpoint.v1\0"
_REDACTED_VALUE_DOMAIN = b"visual-rl.reward-runtime-redacted.v1\0"
_CA_BUNDLE_DOMAIN = b"visual-rl.reward-ca-bundle.v1\0"
_HASH_CHUNK_SIZE = 1024 * 1024
_LOCAL_PROTOCOL = "visual_rl_reward_client"
_LOCAL_PROTOCOL_VERSION = "v1"
_WORLD_RESULT_DEVICE = "cpu"
_WORLD_RESULT_DTYPE = "fp32"
_LOCAL_FACTORY_PATHS = {
    "mock": "visual_rl.algorithms.rewards.clients.mock:MockRewardClient",
    "prompt_color": (
        "visual_rl.algorithms.rewards.clients.image:PromptColorRewardClient"
    ),
    "prompt_color_guarded": (
        "visual_rl.algorithms.rewards.clients.image:PromptColorGuardedRewardClient"
    ),
    "prompt_color_margin": (
        "visual_rl.algorithms.rewards.clients.image:PromptColorMarginRewardClient"
    ),
}
_REMOTE_FACTORY_PATHS = {
    "reward_3d": (
        "visual_rl.algorithms.rewards.clients.world_r1:WorldR1Reward3DClient"
    ),
    "reward_general": (
        "visual_rl.algorithms.rewards.clients.world_r1:WorldR1RewardGeneralClient"
    ),
}


class DefaultRewardResourceFactoryError(RuntimeResourceAcquisitionError):
    """A declared reward leaf cannot be acquired from its exact inputs."""


class DefaultRewardResourceFactory:
    """Construct builtin or explicit plugin reward resources at G3."""

    def __call__(
        self,
        request: RewardResourceAcquireRequest,
    ) -> AcquiredRewardResource:
        if not isinstance(request, RewardResourceAcquireRequest):
            raise TypeError("request must be RewardResourceAcquireRequest")
        factory_class = request.descriptor.factory_class
        if factory_class in _LOCAL_FACTORY_PATHS:
            return self._local(request, _LOCAL_FACTORY_PATHS[factory_class])
        if factory_class in _REMOTE_FACTORY_PATHS:
            return self._remote(request, _REMOTE_FACTORY_PATHS[factory_class])
        if ":" in factory_class:
            return self._plugin(request, factory_class)
        raise DefaultRewardResourceFactoryError(
            f"unsupported builtin reward resource factory {factory_class!r}"
        )

    @staticmethod
    def _local(
        request: RewardResourceAcquireRequest,
        class_path: str,
    ) -> AcquiredRewardResource:
        _require_protocol(
            request,
            protocol=_LOCAL_PROTOCOL,
            protocol_version=_LOCAL_PROTOCOL_VERSION,
        )
        binding = request.runtime_binding
        if binding is not None:
            _require_domain(binding, "in_process")
            if (binding.device, binding.dtype) != ("cpu", "fp32"):
                raise DefaultRewardResourceFactoryError(
                    "builtin in-process rewards execute on cpu/fp32"
                )
        component = _load_symbol(class_path)
        semantic = to_plain_dict(request.descriptor.semantic_factory_config)
        resource = _construct_client(component, semantic, request)
        endpoint_identity = _endpoint_identity(
            {
                "kind": "in_process",
                "factory_class": request.descriptor.factory_class,
                "implementation": class_path,
                "immutable_artifact_identity": to_plain_dict(
                    request.immutable_artifact_identity
                ),
            }
        )
        return AcquiredRewardResource(
            resource=resource,
            binding_facts=RewardResourceBindingFacts(
                endpoint_identity=endpoint_identity,
                protocol=_LOCAL_PROTOCOL,
                protocol_version=_LOCAL_PROTOCOL_VERSION,
                device="cpu",
                dtype="fp32",
                worker_domain="in_process",
            ),
        )

    @staticmethod
    def _remote(
        request: RewardResourceAcquireRequest,
        class_path: str,
    ) -> AcquiredRewardResource:
        binding = request.runtime_binding
        if binding is None:
            raise DefaultRewardResourceFactoryError(
                "remote rewards require launch.reward_runtime_bindings"
            )
        _require_domain(binding, "remote")
        if (binding.device, binding.dtype) != (
            _WORLD_RESULT_DEVICE,
            _WORLD_RESULT_DTYPE,
        ):
            raise DefaultRewardResourceFactoryError(
                "World-R1 clients observe cpu/fp32 reward results"
            )
        _require_protocol(
            request,
            protocol=WORLD_R1_RESOURCE_PROTOCOL,
            protocol_version=PROTOCOL_VERSION,
        )
        semantic = to_plain_dict(request.descriptor.semantic_factory_config)
        expected_semantic_keys = {"server_revision_expectation"}
        if request.descriptor.factory_class == "reward_general":
            expected_semantic_keys.add("input_selection_policy")
        if set(semantic) != expected_semantic_keys:
            raise DefaultRewardResourceFactoryError(
                "World-R1 semantic config has an invalid exact key set: "
                f"expected={sorted(expected_semantic_keys)}, "
                f"actual={sorted(semantic)}"
            )
        server_revision = semantic["server_revision_expectation"]
        try:
            server_revision = validate_server_revision(server_revision)
        except (TypeError, ValueError) as exc:
            raise DefaultRewardResourceFactoryError(
                "server_revision_expectation violates the World-R1 contract"
            ) from exc
        ca_bundle_identity = _ca_bundle_content_identity(binding.ca_bundle)
        component = _load_symbol(class_path)
        runtime_config = {
            "url": binding.endpoint,
            "timeout_s": binding.timeout_s,
            "trusted_hosts": binding.trusted_hosts,
            "ca_bundle": binding.ca_bundle,
            "max_response_bytes": binding.max_response_bytes,
            "server_revision": server_revision,
            "input_selection_policy": None,
        }
        if request.descriptor.factory_class == "reward_general":
            try:
                runtime_config["input_selection_policy"] = (
                    RewardInputSelectionPolicy.from_mapping(
                        semantic["input_selection_policy"]
                    )
                )
            except (TypeError, ValueError) as exc:
                raise DefaultRewardResourceFactoryError(
                    "input_selection_policy violates the World-R1 contract"
                ) from exc
        resource = _construct_client(component, runtime_config, request)
        try:
            healthcheck = getattr(resource, "healthcheck", None)
            if not callable(healthcheck):
                raise TypeError("remote reward resource must implement healthcheck()")
            attestation = healthcheck()
            _validate_world_attestation(
                attestation,
                expected_reward=(
                    REWARD_3D
                    if request.descriptor.factory_class == "reward_3d"
                    else REWARD_GENERAL
                ),
                expected_revision=server_revision,
            )
            if _ca_bundle_content_identity(binding.ca_bundle) != ca_bundle_identity:
                raise DefaultRewardResourceFactoryError(
                    "World-R1 CA bundle changed during endpoint acquisition"
                )
        except BaseException as primary:
            _close_after_failure(resource, primary)
            raise
        endpoint_identity = _remote_endpoint_identity(
            factory_class=request.descriptor.factory_class,
            binding=binding,
            attestation=attestation,
            ca_bundle_identity=ca_bundle_identity,
        )
        return AcquiredRewardResource(
            resource=resource,
            binding_facts=RewardResourceBindingFacts(
                endpoint_identity=endpoint_identity,
                protocol=attestation.protocol,
                protocol_version=attestation.protocol_version,
                device=_WORLD_RESULT_DEVICE,
                dtype=_WORLD_RESULT_DTYPE,
                worker_domain="remote",
            ),
        )

    @staticmethod
    def _plugin(
        request: RewardResourceAcquireRequest,
        class_path: str,
    ) -> AcquiredRewardResource:
        loaded = _load_symbol(class_path)
        candidate = loaded() if isinstance(loaded, type) else loaded
        if not isinstance(candidate, RewardResourceFactory):
            raise TypeError(
                "explicit reward factory must implement RewardResourceFactory"
            )
        result = candidate(request)
        if not isinstance(result, AcquiredRewardResource):
            primary = TypeError(
                "explicit reward factory must return AcquiredRewardResource"
            )
            _close_after_failure(result, primary)
            raise primary
        return result


def _construct_client(
    component: object,
    raw: Mapping[str, object],
    request: RewardResourceAcquireRequest,
) -> object:
    resolve = getattr(component, "resolve_params", None)
    construct = getattr(component, "from_config", None)
    if not callable(resolve) or not callable(construct):
        raise TypeError(
            "builtin reward client must implement resolve_params/from_config"
        )
    location = request.artifact_location
    config_path = location if location.is_file() else location / "reward-artifact"
    resolved = resolve(
        raw,
        ResolutionContext(
            config_path=config_path,
            config_dir=config_path.parent,
        ),
    )
    facts = request.runtime_facts
    try:
        import torch

        device = torch.device(facts.device)
    except (ModuleNotFoundError, RuntimeError, TypeError) as exc:
        raise DefaultRewardResourceFactoryError(
            "reward runtime device is not a valid torch device"
        ) from exc
    resource = construct(
        resolved,
        RuntimeBuildContext(
            rank=facts.rank,
            local_rank=facts.local_rank,
            world_size=facts.world_size,
            backend=facts.backend,
            device=device,
            precision=facts.precision,
        ),
    )
    if not callable(getattr(resource, "score", None)) or not callable(
        getattr(resource, "close", None)
    ):
        primary = TypeError("reward client must implement score() and close()")
        _close_after_failure(resource, primary)
        raise primary
    return resource


def _load_symbol(class_path: str) -> object:
    if not isinstance(class_path, str) or class_path.count(":") != 1:
        raise ValueError("class_path must use module:attribute syntax")
    module_name, attribute_name = class_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise DefaultRewardResourceFactoryError(
            f"cannot load reward resource factory {class_path!r}"
        ) from exc


def _require_domain(
    binding: RewardRuntimeBindingSpec,
    expected: str,
) -> None:
    if not isinstance(binding, RewardRuntimeBindingSpec):
        raise TypeError("runtime binding must be RewardRuntimeBindingSpec")
    if binding.execution_domain != expected:
        raise DefaultRewardResourceFactoryError(
            f"reward runtime binding must use execution_domain={expected!r}"
        )


def _require_protocol(
    request: RewardResourceAcquireRequest,
    *,
    protocol: str,
    protocol_version: str,
) -> None:
    descriptor = request.descriptor
    if (descriptor.protocol, descriptor.protocol_version) != (
        protocol,
        protocol_version,
    ):
        raise DefaultRewardResourceFactoryError(
            "reward descriptor protocol/version differs from the concrete client"
        )


def _validate_world_attestation(
    value: object,
    *,
    expected_reward: str,
    expected_revision: str,
) -> WorldR1HealthAttestation:
    if not isinstance(value, WorldR1HealthAttestation):
        raise TypeError("World-R1 healthcheck must return WorldR1HealthAttestation")
    if value.reward != expected_reward:
        raise DefaultRewardResourceFactoryError(
            "World-R1 health attested a different reward kind"
        )
    if value.protocol != WORLD_R1_RESOURCE_PROTOCOL or (
        value.protocol_version != PROTOCOL_VERSION
    ):
        raise DefaultRewardResourceFactoryError(
            "World-R1 health attested a different protocol"
        )
    if value.server_revision != expected_revision:
        raise DefaultRewardResourceFactoryError(
            "World-R1 health attested a different server revision"
        )
    return value


def _endpoint_identity(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _ENDPOINT_ID_DOMAIN + canonical_json_text(payload).encode("utf-8")
    ).hexdigest()


def _remote_endpoint_identity(
    *,
    factory_class: str,
    binding: RewardRuntimeBindingSpec,
    attestation: WorldR1HealthAttestation,
    ca_bundle_identity: str,
) -> str:
    return _endpoint_identity(
        {
            "kind": "remote",
            "factory_class": factory_class,
            "endpoint_origin_sha256": _redacted_value_identity(
                "endpoint-origin",
                attestation.endpoint_origin,
            ),
            "trusted_hosts_sha256": _redacted_value_identity(
                "trusted-hosts",
                list(binding.trusted_hosts),
            ),
            "ca_bundle_content_identity": ca_bundle_identity,
            "timeout_s": binding.timeout_s,
            "max_response_bytes": binding.max_response_bytes,
            "server_revision": attestation.server_revision,
        }
    )


def _redacted_value_identity(kind: str, value: object) -> str:
    payload = {"kind": kind, "value": value}
    return hashlib.sha256(
        _REDACTED_VALUE_DOMAIN + canonical_json_text(payload).encode("utf-8")
    ).hexdigest()


def _ca_bundle_content_identity(path: Path | None) -> str:
    if path is None:
        return "system-default-ca"
    if not isinstance(path, Path) or not path.is_absolute():
        raise TypeError("CA bundle must be an absolute Path or None")
    digest = hashlib.sha256(_CA_BUNDLE_DOMAIN)
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise DefaultRewardResourceFactoryError(
                    "World-R1 CA bundle must be a regular file"
                )
            while True:
                chunk = handle.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise DefaultRewardResourceFactoryError(
            "World-R1 CA bundle cannot be read for identity"
        ) from exc
    before_facts = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_facts = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_facts != after_facts:
        raise DefaultRewardResourceFactoryError(
            "World-R1 CA bundle changed while its content was hashed"
        )
    return digest.hexdigest()


def _close_after_failure(resource: object, primary: BaseException) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except BaseException as cleanup_error:  # noqa: BLE001
        if hasattr(primary, "add_note"):
            primary.add_note(
                "reward resource cleanup failed after acquisition error: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


__all__ = (
    "AcquiredRewardResource",
    "BorrowedRewardResourceHandle",
    "DefaultRewardResourceFactory",
    "DefaultRewardResourceFactoryError",
    "RewardAcquisitionPlanningError",
    "RewardLogicalBinding",
    "RewardPool",
    "RewardPoolActivationError",
    "RewardPoolBuildError",
    "RewardPoolCloseError",
    "RewardPoolView",
    "RewardResourceAcquireRequest",
    "RewardResourceBindingFacts",
    "RewardResourceFactory",
    "RewardResourceState",
    "RuntimeResourceAcquisitionError",
    "acquire_reward_view",
    "bind_logical_rewards",
    "bound_reward_resource_id",
    "bound_reward_resource_payload",
    "compile_reward_resource_acquisition",
    "resolve_reward_bindings",
)
