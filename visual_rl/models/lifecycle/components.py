"""Single owner for model components, residency, and execution modes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visual_rl.models.interface import ModelAdapter
    from visual_rl.models.lifecycle.prepared import PreparedComponentHandle
    from visual_rl.models.numerics.execution import StageExecutionPolicy
    from visual_rl.models.numerics.policy import (
        ModelExecutionNumericsEvidence,
        ParameterDTypePolicy,
        ParameterDTypePolicyOwner,
    )
    from visual_rl.models.state.parameters import ParameterStateManager

__all__ = (
    "ComponentBinding",
    "ComponentLifecycleError",
    "ComponentLoadSession",
    "ComponentManager",
    "ComponentManagerError",
    "ComponentResidencyError",
    "ComponentRole",
    "ExecutionMode",
    "ModelComponents",
    "OwnershipState",
    "Residency",
    "ResourceOwner",
    "ResourcePlan",
)


class ComponentManagerError(RuntimeError):
    """Base error for model component ownership failures."""


class ComponentLifecycleError(ComponentManagerError):
    """Raised for an illegal lifecycle or execution-mode transition."""


class ComponentResidencyError(ComponentManagerError):
    """Raised when a manager-owned device transition cannot complete."""


class ComponentRole(str, Enum):
    PREPROCESS = "preprocess"
    INFERENCE = "inference"
    TRAINABLE = "trainable"
    REFERENCE = "reference"
    DECODER = "decoder"


class OwnershipState(str, Enum):
    UNLOADED = "unloaded"
    CREATED = "unloaded"  # noqa: PIE796  # compatibility alias
    LOADED = "loaded"
    CONFIGURED = "configured"
    PREPARED = "prepared"
    CLOSED = "closed"


class ExecutionMode(str, Enum):
    IDLE = "idle"
    PREPROCESS = "preprocess"
    ROLLOUT = "rollout"
    SAMPLING = "rollout"  # noqa: PIE796  # compatibility alias
    TRAIN = "train"
    TRAINING = "train"  # noqa: PIE796  # compatibility alias
    EVAL = "eval"


class Residency(str, Enum):
    OFFLOADED = "offloaded"
    RESIDENT = "resident"
    PREPARED = "prepared"
    STATIC = "static"


class ResourceOwner(str, Enum):
    COMPONENT_MANAGER = "component_manager"
    PREPARED_BACKEND = "prepared_backend"
    STATIC = "static"


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    """Explicit role residency requirements for every execution mode."""

    mode_roles: tuple[tuple[ExecutionMode, tuple[ComponentRole, ...]], ...]

    def __post_init__(self) -> None:
        if type(self.mode_roles) is not tuple:
            raise TypeError("ResourcePlan.mode_roles must be a tuple")
        expected = {
            ExecutionMode.PREPROCESS,
            ExecutionMode.ROLLOUT,
            ExecutionMode.TRAIN,
            ExecutionMode.EVAL,
        }
        resolved: list[tuple[ExecutionMode, tuple[ComponentRole, ...]]] = []
        for mode, roles in self.mode_roles:
            try:
                mode = ExecutionMode(mode)
            except (TypeError, ValueError):
                raise ComponentManagerError(
                    f"invalid resource-plan mode: {mode!r}"
                ) from None
            if mode is ExecutionMode.IDLE:
                raise ComponentManagerError("IDLE does not have resident roles")
            resolved.append((mode, _roles(roles, allow_trainable_reference=True)))
        modes = tuple(mode for mode, _roles_value in resolved)
        if len(modes) != len(set(modes)):
            raise ComponentManagerError("ResourcePlan modes must be unique")
        if set(modes) != expected:
            raise ComponentManagerError(
                "ResourcePlan must cover PREPROCESS, ROLLOUT, TRAIN, and EVAL"
            )
        object.__setattr__(self, "mode_roles", tuple(resolved))

    @classmethod
    def default(cls) -> ResourcePlan:
        return cls(
            mode_roles=(
                (ExecutionMode.PREPROCESS, (ComponentRole.PREPROCESS,)),
                (
                    ExecutionMode.ROLLOUT,
                    (ComponentRole.INFERENCE, ComponentRole.DECODER),
                ),
                (
                    ExecutionMode.TRAIN,
                    (
                        ComponentRole.INFERENCE,
                        ComponentRole.TRAINABLE,
                        ComponentRole.REFERENCE,
                    ),
                ),
                (
                    ExecutionMode.EVAL,
                    (
                        ComponentRole.PREPROCESS,
                        ComponentRole.INFERENCE,
                        ComponentRole.REFERENCE,
                        ComponentRole.DECODER,
                    ),
                ),
            )
        )

    def roles_for(self, mode: ExecutionMode) -> tuple[ComponentRole, ...]:
        try:
            resolved = ExecutionMode(mode)
        except (TypeError, ValueError):
            raise ComponentManagerError(f"invalid execution mode: {mode!r}") from None
        matches = tuple(
            roles for candidate, roles in self.mode_roles if candidate is resolved
        )
        if len(matches) != 1:
            raise ComponentManagerError(
                f"ResourcePlan does not define mode {resolved.value!r}"
            )
        return matches[0]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "modes": [
                {
                    "mode": mode.value,
                    "required_roles": [role.value for role in roles],
                }
                for mode, roles in self.mode_roles
            ],
        }

    @property
    def plan_id(self) -> str:
        payload = json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def owner_for(
        binding: ComponentBinding,
        *,
        prepared_component_names: frozenset[str],
    ) -> ResourceOwner:
        if binding.name in prepared_component_names:
            return ResourceOwner.PREPARED_BACKEND
        if not binding.managed_residency:
            return ResourceOwner.STATIC
        return ResourceOwner.COMPONENT_MANAGER


def _roles(
    values: object,
    *,
    allow_trainable_reference: bool = False,
) -> tuple[ComponentRole, ...]:
    if type(values) is not tuple or not values:
        raise ComponentManagerError("component roles must be a non-empty tuple")
    resolved: list[ComponentRole] = []
    for value in values:
        try:
            role = ComponentRole(value)
        except (TypeError, ValueError):
            raise ComponentManagerError(f"invalid component role: {value!r}") from None
        resolved.append(role)
    result = tuple(resolved)
    if len(result) != len(set(result)):
        raise ComponentManagerError("component roles must not contain duplicates")
    if (
        not allow_trainable_reference
        and ComponentRole.REFERENCE in result
        and ComponentRole.TRAINABLE in result
    ):
        raise ComponentManagerError(
            "one component cannot be both trainable and a frozen reference"
        )
    return result


@dataclass(frozen=True, slots=True)
class ComponentBinding:
    """One uniquely owned object with all of its explicit runtime roles."""

    name: str
    component: object
    roles: tuple[ComponentRole, ...]
    managed_residency: bool = True
    closer: Callable[[object], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ComponentManagerError("component name must be a non-empty string")
        if self.component is None:
            raise ComponentManagerError("component must not be None")
        object.__setattr__(self, "roles", _roles(self.roles))
        if type(self.managed_residency) is not bool:
            raise TypeError("managed_residency must be bool")
        if self.closer is not None and not callable(self.closer):
            raise TypeError("component closer must be callable")


@dataclass(frozen=True, slots=True)
class ModelComponents:
    """Frozen post-load inventory; aliases never duplicate object ownership."""

    bindings: tuple[ComponentBinding, ...]

    def __post_init__(self) -> None:
        if type(self.bindings) is not tuple or not self.bindings:
            raise ComponentManagerError(
                "ModelComponents.bindings must be a non-empty tuple"
            )
        if any(not isinstance(item, ComponentBinding) for item in self.bindings):
            raise TypeError("ModelComponents entries must be ComponentBinding")
        names = tuple(item.name for item in self.bindings)
        identities = tuple(id(item.component) for item in self.bindings)
        if len(names) != len(set(names)):
            raise ComponentManagerError("component names must be uniquely owned")
        if len(identities) != len(set(identities)):
            raise ComponentManagerError(
                "one component object cannot have multiple ownership bindings"
            )

    def bindings_for(self, role: ComponentRole) -> tuple[ComponentBinding, ...]:
        try:
            resolved = ComponentRole(role)
        except (TypeError, ValueError):
            raise ComponentManagerError(f"invalid component role: {role!r}") from None
        return tuple(item for item in self.bindings if resolved in item.roles)

    def components_for(self, role: ComponentRole) -> tuple[object, ...]:
        return tuple(item.component for item in self.bindings_for(role))

    def binding(self, name: str) -> ComponentBinding:
        matches = tuple(item for item in self.bindings if item.name == name)
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    @property
    def preprocess(self) -> tuple[object, ...]:
        return self.components_for(ComponentRole.PREPROCESS)

    @property
    def inference(self) -> tuple[object, ...]:
        return self.components_for(ComponentRole.INFERENCE)

    @property
    def trainable(self) -> tuple[object, ...]:
        return self.components_for(ComponentRole.TRAINABLE)

    @property
    def reference(self) -> tuple[object, ...]:
        return self.components_for(ComponentRole.REFERENCE)

    @property
    def decoder(self) -> tuple[object, ...]:
        return self.components_for(ComponentRole.DECODER)


class ComponentLoadSession:
    """Manager-owned acquisition log enabling reverse-order rollback."""

    def __init__(self) -> None:
        self._bindings: list[ComponentBinding] = []
        self._frozen: ModelComponents | None = None
        self._committed = False

    def acquire(
        self,
        name: str,
        factory: Callable[[], object],
        *,
        roles: tuple[ComponentRole, ...],
        managed_residency: bool = True,
        closer: Callable[[object], None] | None = None,
    ) -> object:
        """Construct and immediately register one manager-owned component."""

        if self._frozen is not None or self._committed:
            raise ComponentLifecycleError("component load session is already frozen")
        if not callable(factory):
            raise TypeError("component factory must be callable")
        component = factory()
        try:
            self.register(
                name,
                component,
                roles=roles,
                managed_residency=managed_residency,
                closer=closer,
            )
        except BaseException as exc:
            if all(item.component is not component for item in self._bindings):
                try:
                    _close_unbound(component, closer)
                except BaseException as cleanup_exc:  # noqa: BLE001
                    if hasattr(exc, "add_note"):
                        exc.add_note(f"unbound component cleanup failed: {cleanup_exc}")
            raise
        return component

    def register(
        self,
        name: str,
        component: object,
        *,
        roles: tuple[ComponentRole, ...],
        managed_residency: bool = True,
        closer: Callable[[object], None] | None = None,
    ) -> None:
        if self._frozen is not None or self._committed:
            raise ComponentLifecycleError("component load session is already frozen")
        binding = ComponentBinding(
            name=name,
            component=component,
            roles=roles,
            managed_residency=managed_residency,
            closer=closer,
        )
        if any(item.name == binding.name for item in self._bindings):
            raise ComponentManagerError(f"duplicate component name: {binding.name}")
        if any(item.component is binding.component for item in self._bindings):
            raise ComponentManagerError(
                "one component object cannot be registered more than once"
            )
        self._bindings.append(binding)

    def freeze(self) -> ModelComponents:
        if self._committed:
            raise ComponentLifecycleError("component load session is committed")
        if self._frozen is None:
            self._frozen = ModelComponents(tuple(self._bindings))
        return self._frozen

    def _accepts(self, components: ModelComponents) -> bool:
        return components is self._frozen

    def _commit(self) -> None:
        if self._frozen is None:
            raise ComponentLifecycleError("cannot commit an unfrozen load session")
        self._committed = True

    def _rollback(self) -> tuple[BaseException, ...]:
        if self._committed:
            raise ComponentLifecycleError("cannot roll back a committed load session")
        errors: list[BaseException] = []
        for binding in reversed(self._bindings):
            try:
                _close_binding(binding)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        self._bindings.clear()
        self._frozen = None
        return tuple(errors)


def _close_binding(binding: ComponentBinding) -> None:
    if binding.closer is not None:
        binding.closer(binding.component)
        return
    close = getattr(binding.component, "close", None)
    if callable(close):
        close()


def _close_unbound(
    component: object,
    closer: Callable[[object], None] | None,
) -> None:
    if closer is not None:
        closer(component)
        return
    close = getattr(component, "close", None)
    if callable(close):
        close()


class ComponentManager:
    """Own component lifecycle and all device transitions for one adapter."""

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        execution_device: Any,
        offload_device: Any = "cpu",
        resource_plan: ResourcePlan | None = None,
    ) -> None:
        import torch

        from visual_rl.models.interface import ModelAdapter

        if not isinstance(adapter, ModelAdapter):
            raise TypeError("adapter must inherit visual_rl.models.ModelAdapter")
        try:
            self.execution_device = torch.device(execution_device)
            self.offload_device = torch.device(offload_device)
        except (TypeError, RuntimeError):
            raise TypeError(
                "component devices must be torch.device-compatible"
            ) from None
        self.adapter = adapter
        if resource_plan is not None and not isinstance(resource_plan, ResourcePlan):
            raise TypeError("resource_plan must be ResourcePlan")
        self.resource_plan = (
            ResourcePlan.default() if resource_plan is None else resource_plan
        )
        self._state = OwnershipState.UNLOADED
        self._mode = ExecutionMode.IDLE
        self._mode_depth = 0
        self._mode_snapshot: dict[str, Residency] | None = None
        self._module_mode_snapshot: tuple[tuple[Any, bool], ...] | None = None
        self._active_execution_policy: StageExecutionPolicy | None = None
        self._components: ModelComponents | None = None
        self._residency: dict[str, Residency] = {}
        self._runtime_bound: object | None = None
        self._parameter_state: ParameterStateManager | None = None
        self._parameter_dtype_owner: ParameterDTypePolicyOwner | None = None
        self._model_execution_numerics: ModelExecutionNumericsEvidence | None = None
        self._prepared_handle: PreparedComponentHandle | None = None
        self._prepared_component_names: frozenset[str] = frozenset()
        self._prepare_attempted = False
        self._prepare_error: BaseException | None = None
        self._lock = RLock()

    @property
    def state(self) -> OwnershipState:
        return self._state

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    @property
    def active_execution_policy(self) -> StageExecutionPolicy | None:
        return self._active_execution_policy

    @property
    def components(self) -> ModelComponents:
        if (
            self._state
            not in {
                OwnershipState.LOADED,
                OwnershipState.CONFIGURED,
                OwnershipState.PREPARED,
            }
            or self._components is None
        ):
            raise ComponentLifecycleError("model components are not loaded")
        return self._components

    @property
    def runtime_bound(self) -> object:
        if self._runtime_bound is None:
            raise ComponentLifecycleError("G3 runtime contract is not bound")
        return self._runtime_bound

    @property
    def parameter_state(self) -> ParameterStateManager:
        if self._parameter_state is None:
            raise ComponentLifecycleError(
                "trainable parameter topology is not configured"
            )
        return self._parameter_state

    @property
    def parameter_dtype_owner(self) -> ParameterDTypePolicyOwner:
        if self._parameter_dtype_owner is None:
            raise ComponentLifecycleError("parameter dtype policy is not applied")
        return self._parameter_dtype_owner

    @property
    def model_execution_numerics(self) -> ModelExecutionNumericsEvidence:
        if self._model_execution_numerics is None:
            raise ComponentLifecycleError("model execution numerics are not bound")
        return self._model_execution_numerics

    @property
    def prepared_handle(self) -> PreparedComponentHandle:
        if self._prepared_handle is None:
            raise ComponentLifecycleError("model bundle is not prepared")
        return self._prepared_handle

    @property
    def prepare_error(self) -> BaseException | None:
        return self._prepare_error

    def component(self, name: str) -> object:
        return self.components.binding(name).component

    def residency(self, name: str) -> Residency:
        if self._state not in {
            OwnershipState.LOADED,
            OwnershipState.CONFIGURED,
            OwnershipState.PREPARED,
        }:
            raise ComponentLifecycleError("model components are not loaded")
        try:
            return self._residency[name]
        except KeyError:
            raise KeyError(name) from None

    def load(self) -> ModelComponents:
        with self._lock:
            if self._state is not OwnershipState.UNLOADED:
                raise ComponentLifecycleError(
                    f"load requires UNLOADED state, found {self._state.value}"
                )
            session = ComponentLoadSession()
            try:
                components = self.adapter.load_components(session)
                if not isinstance(components, ModelComponents):
                    raise TypeError("load_components() must return ModelComponents")
                if not session._accepts(components):
                    raise ComponentManagerError(
                        "load_components() must return its manager-owned "
                        "session.freeze()"
                    )
                residency: dict[str, Residency] = {}
                for binding in components.bindings:
                    if binding.managed_residency:
                        self._move(binding, self.offload_device)
                        residency[binding.name] = Residency.OFFLOADED
                    else:
                        residency[binding.name] = Residency.STATIC
            except BaseException as exc:
                cleanup_errors = session._rollback()
                if cleanup_errors and hasattr(exc, "add_note"):
                    exc.add_note(
                        "component rollback errors: "
                        + "; ".join(str(item) for item in cleanup_errors)
                    )
                raise
            session._commit()
            self._components = components
            self._residency = residency
            self._state = OwnershipState.LOADED
            return components

    def configure(
        self,
        configurator: Callable[[ModelComponents], None] | None = None,
    ) -> None:
        """Apply manager-scoped model configuration before distributed prepare."""

        with self._lock:
            if self._state is not OwnershipState.LOADED:
                raise ComponentLifecycleError(
                    f"configure requires LOADED state, found {self._state.value}"
                )
            if configurator is not None:
                if not callable(configurator):
                    raise TypeError("configurator must be callable")
                configurator(self.components)
            from visual_rl.models.state.parameters import ParameterStateManager

            parameter_state = ParameterStateManager(self.components)
            self._parameter_state = parameter_state
            self._state = OwnershipState.CONFIGURED

    def apply_parameter_dtype_policy(
        self,
        policy: ParameterDTypePolicy,
    ) -> ParameterStateManager:
        """Apply and atomically rebind the only pre-prepare dtype owner."""

        from visual_rl.models.numerics.policy import (
            ParameterDTypePolicy,
            ParameterDTypePolicyOwner,
        )

        if not isinstance(policy, ParameterDTypePolicy):
            raise TypeError("policy must be ParameterDTypePolicy")
        with self._lock:
            if self._state is not OwnershipState.CONFIGURED:
                raise ComponentLifecycleError(
                    "parameter dtype application requires CONFIGURED state, "
                    f"found {self._state.value}"
                )
            if self._prepare_attempted:
                raise ComponentLifecycleError(
                    "parameter dtype application must precede prepare"
                )
            if self._parameter_dtype_owner is not None:
                raise ComponentLifecycleError(
                    "parameter dtype policy is already owned by this manager"
                )
            prior = self.parameter_state
            owner = ParameterDTypePolicyOwner(
                self.components,
                ownership_state=lambda: self.state,
                parameter_state=prior,
            )
            rebound = owner.apply(policy)
            if rebound.components is not self.components:
                raise ComponentLifecycleError(
                    "dtype application rebound a foreign component inventory"
                )
            self._parameter_state = rebound
            self._parameter_dtype_owner = owner
            return rebound

    def bind_model_execution_numerics(
        self,
        evidence: ModelExecutionNumericsEvidence,
    ) -> None:
        """Bind exact forward/view evidence before optimizer construction."""

        from visual_rl.models.numerics.policy import ModelExecutionNumericsEvidence

        if not isinstance(evidence, ModelExecutionNumericsEvidence):
            raise TypeError("evidence must be ModelExecutionNumericsEvidence")
        with self._lock:
            if self._state is not OwnershipState.CONFIGURED:
                raise ComponentLifecycleError(
                    "model execution numerics require CONFIGURED state"
                )
            if self._prepare_attempted:
                raise ComponentLifecycleError(
                    "model execution numerics must be bound before prepare"
                )
            if self._model_execution_numerics is not None:
                raise ComponentLifecycleError(
                    "model execution numerics are already bound"
                )
            dtype_owner = self.parameter_dtype_owner
            dtype_owner.validate_applied()
            if evidence.parameter_dtype_policy.policy_id != dtype_owner.policy_id:
                raise ComponentLifecycleError(
                    "execution numerics use a different parameter dtype policy"
                )
            if (
                evidence.source_projection_id
                != self.parameter_state.state_projection.projection_id
            ):
                raise ComponentLifecycleError(
                    "parameter view evidence uses a stale model-state projection"
                )
            self.adapter._bind_model_execution_numerics(
                evidence,
                execution_policy_provider=lambda: self.active_execution_policy,
            )
            self._model_execution_numerics = evidence

    def prepare(
        self,
        *,
        accelerator: object,
        optimizer: object,
        scheduler: object | None = None,
    ) -> PreparedComponentHandle:
        """Prepare exactly one model root, optimizer, and optional scheduler."""

        with self._lock:
            if self._prepare_attempted:
                raise ComponentLifecycleError(
                    "ComponentManager permits exactly one prepare attempt"
                )
            if self._state is not OwnershipState.CONFIGURED:
                raise ComponentLifecycleError(
                    f"prepare requires CONFIGURED state, found {self._state.value}"
                )
            if self._parameter_state is None:
                raise ComponentLifecycleError(
                    "parameter topology must be configured before prepare"
                )
            from visual_rl.models.lifecycle.prepared import prepare_model_bundle

            self._prepare_attempted = True
            handle = None
            try:
                handle = prepare_model_bundle(
                    self.components,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                self.adapter._bind_prepared_components(handle)
            except BaseException as exc:
                if handle is not None:
                    close_handle = getattr(handle, "close", None)
                    if callable(close_handle):
                        try:
                            close_handle()
                        except BaseException as cleanup_exc:  # noqa: BLE001
                            if hasattr(exc, "add_note"):
                                exc.add_note(
                                    f"prepared handle rollback failed: {cleanup_exc}"
                                )
                self.adapter._clear_prepared_components()
                self._prepare_error = exc
                self._prepared_handle = None
                self._prepared_component_names = frozenset()
                raise
            self._prepared_handle = handle
            self._prepared_component_names = frozenset(handle.component_names)
            for name in self._prepared_component_names:
                self._residency[name] = Residency.PREPARED
            self._state = OwnershipState.PREPARED
            return handle

    def bind_runtime(self, contract: object) -> None:
        """Bind successful G3 evidence before any execution mode is legal."""

        from visual_rl.core.contracts import RuntimeBoundContract
        from visual_rl.models.interface import ModelAdapter

        with self._lock:
            if self._state is not OwnershipState.PREPARED:
                raise ComponentLifecycleError(
                    "runtime bind requires PREPARED component ownership"
                )
            if not isinstance(contract, RuntimeBoundContract):
                raise TypeError("contract must be RuntimeBoundContract")
            if contract.artifact.declared.component_kind != "model":
                raise ComponentLifecycleError(
                    "ComponentManager requires a model RuntimeBoundContract"
                )
            model_contract = contract.artifact.declared.model
            assert model_contract is not None
            if model_contract.provides_reference_policy is True:
                if (
                    type(self.adapter).predict_reference
                    is ModelAdapter.predict_reference
                ):
                    raise ComponentLifecycleError(
                        "G3 rejected declared reference policy without a typed "
                        "predict_reference implementation"
                    )
                verified = dict(contract.verified_fields)
                if verified.get("model.reference_forward") != "verified":
                    raise ComponentLifecycleError(
                        "G3 requires a verified model.reference_forward probe"
                    )
            if self._runtime_bound is not None:
                raise ComponentLifecycleError("G3 runtime contract is already bound")
            self._runtime_bound = contract

    @contextmanager
    def sampling(self) -> Iterator[ModelComponents]:
        """Compatibility alias for the canonical rollout execution mode."""

        with self.rollout() as components:
            yield components

    @contextmanager
    def training(self) -> Iterator[ModelComponents]:
        """Compatibility alias for the canonical train execution mode."""

        with self.train() as components:
            yield components

    @contextmanager
    def preprocess(self) -> Iterator[ModelComponents]:
        from visual_rl.models.numerics.execution import StageExecutionPolicy

        policy = StageExecutionPolicy.canonical(ExecutionMode.PREPROCESS)
        with self.execution(policy) as components:
            yield components

    @contextmanager
    def rollout(self) -> Iterator[ModelComponents]:
        from visual_rl.models.numerics.execution import StageExecutionPolicy

        policy = StageExecutionPolicy.canonical(ExecutionMode.ROLLOUT)
        with self.execution(policy) as components:
            yield components

    @contextmanager
    def train(self) -> Iterator[ModelComponents]:
        from visual_rl.models.numerics.execution import StageExecutionPolicy

        policy = StageExecutionPolicy.canonical(ExecutionMode.TRAIN)
        with self.execution(policy) as components:
            yield components

    @contextmanager
    def train_reference(
        self,
        *,
        parameter_view_context: Callable[[], object] | None = None,
    ) -> Iterator[ModelComponents]:
        """Run a frozen reference forward without nesting the current view."""

        from visual_rl.models.numerics.execution import (
            ParameterView,
            StageExecutionPolicy,
        )

        policy = StageExecutionPolicy.canonical(
            ExecutionMode.TRAIN,
            parameter_view=ParameterView.REFERENCE,
        )
        with self.execution(
            policy,
            parameter_view_context=parameter_view_context,
        ) as components:
            yield components

    @contextmanager
    def evaluate(self) -> Iterator[ModelComponents]:
        from visual_rl.models.numerics.execution import StageExecutionPolicy

        policy = StageExecutionPolicy.canonical(ExecutionMode.EVAL)
        with self.execution(policy) as components:
            yield components

    @contextmanager
    def execution(
        self,
        policy: StageExecutionPolicy,
        *,
        parameter_view_context: Callable[[], object] | None = None,
    ) -> Iterator[ModelComponents]:
        """Apply residency, module mode, grad mode, and one logical view."""

        import torch

        from visual_rl.models.numerics.execution import StageExecutionPolicy

        if not isinstance(policy, StageExecutionPolicy):
            raise TypeError("policy must be a StageExecutionPolicy")
        if parameter_view_context is not None and not callable(parameter_view_context):
            raise TypeError("parameter_view_context must be callable or None")

        with self._execution_policy(policy) as components:
            view_context = (
                nullcontext()
                if parameter_view_context is None
                else parameter_view_context()
            )
            if not callable(getattr(view_context, "__enter__", None)) or not callable(
                getattr(view_context, "__exit__", None)
            ):
                raise TypeError("parameter_view_context must return a context manager")
            with torch.set_grad_enabled(policy.grad_enabled), view_context:
                yield components

    @contextmanager
    def _execution_policy(
        self,
        policy: StageExecutionPolicy,
    ) -> Iterator[ModelComponents]:
        requested = policy.stage
        with self._lock:
            if self._state is not OwnershipState.PREPARED:
                raise ComponentLifecycleError(
                    f"{requested.value} mode requires PREPARED components"
                )
            if self._runtime_bound is None:
                raise ComponentLifecycleError(
                    f"{requested.value} mode requires successful G3 runtime bind"
                )
            active = self._active_execution_policy
            if self._mode is not ExecutionMode.IDLE and (
                active is None
                or active.execution_policy_id != policy.execution_policy_id
            ):
                active_view = (
                    "unknown" if active is None else active.parameter_view.value
                )
                if self._mode is not requested:
                    raise ComponentLifecycleError(
                        f"cannot enter {requested.value} while "
                        f"{self._mode.value} is active "
                        f"(parameter views {active_view} -> "
                        f"{policy.parameter_view.value})"
                    )
                raise ComponentLifecycleError(
                    f"cannot enter {requested.value}/{policy.parameter_view.value} "
                    f"while {self._mode.value}/{active_view} is active"
                )
            if self._mode is requested:
                self._mode_depth += 1
            else:
                residency_snapshot = dict(self._residency)
                module_mode_snapshot = self._snapshot_module_modes()
                try:
                    self._apply_mode(requested, residency_snapshot)
                    self._apply_module_modes(policy)
                except BaseException as exc:
                    try:
                        self._restore_module_modes(module_mode_snapshot)
                    except BaseException as rollback_exc:  # noqa: BLE001
                        if hasattr(exc, "add_note"):
                            exc.add_note(f"module-mode rollback failed: {rollback_exc}")
                    try:
                        self._restore_residency(residency_snapshot)
                    except BaseException as rollback_exc:  # noqa: BLE001
                        if hasattr(exc, "add_note"):
                            exc.add_note(f"residency rollback failed: {rollback_exc}")
                    raise
                self._mode_snapshot = residency_snapshot
                self._module_mode_snapshot = module_mode_snapshot
                self._active_execution_policy = policy
                self._mode = requested
                self._mode_depth = 1
        body_error: BaseException | None = None
        try:
            yield self.components
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            with self._lock:
                active = self._active_execution_policy
                if (
                    self._mode is not requested
                    or self._mode_depth < 1
                    or active is None
                    or active.execution_policy_id != policy.execution_policy_id
                ):
                    raise ComponentLifecycleError("execution mode stack is corrupted")
                self._mode_depth -= 1
                if self._mode_depth == 0:
                    residency_snapshot = self._mode_snapshot
                    module_mode_snapshot = self._module_mode_snapshot
                    if residency_snapshot is None or module_mode_snapshot is None:
                        raise ComponentLifecycleError(
                            "execution mode restoration snapshot is missing"
                        )
                    cleanup_errors: list[BaseException] = []
                    try:
                        self._restore_module_modes(module_mode_snapshot)
                    except BaseException as exc:  # noqa: BLE001
                        cleanup_errors.append(exc)
                    try:
                        self._restore_residency(residency_snapshot)
                    except BaseException as exc:  # noqa: BLE001
                        cleanup_errors.append(exc)
                    finally:
                        self._mode = ExecutionMode.IDLE
                        self._mode_snapshot = None
                        self._module_mode_snapshot = None
                        self._active_execution_policy = None
                    if cleanup_errors:
                        message = "; ".join(str(item) for item in cleanup_errors)
                        if body_error is not None:
                            if hasattr(body_error, "add_note"):
                                body_error.add_note(
                                    f"execution policy restoration failed: {message}"
                                )
                        else:
                            raise ComponentLifecycleError(
                                "execution policy restoration failed: " + message
                            ) from cleanup_errors[0]

    def _snapshot_module_modes(self) -> tuple[tuple[Any, bool], ...]:
        import torch

        snapshot: list[tuple[Any, bool]] = []
        seen: set[int] = set()
        roots = tuple(
            binding.component
            for binding in self.components.bindings
            if isinstance(binding.component, torch.nn.Module)
        )

        # Restore every root before any descendant.  This preserves exact child
        # flags even when a module graph is shared or intentionally mixes modes.
        for module in roots:
            identity = id(module)
            if identity in seen:
                continue
            seen.add(identity)
            snapshot.append((module, module.training))
        for component in roots:
            for module in tuple(component.modules())[1:]:
                identity = id(module)
                if identity in seen:
                    continue
                seen.add(identity)
                snapshot.append((module, module.training))
        return tuple(snapshot)

    def _apply_module_modes(self, policy: StageExecutionPolicy) -> None:
        import torch

        from visual_rl.models.numerics.execution import ModuleExecutionMode

        for binding in self.components.bindings:
            component = binding.component
            if not isinstance(component, torch.nn.Module):
                continue
            target = policy.mode_for_roles(binding.roles)
            if target is None:
                continue
            component.train(target is ModuleExecutionMode.TRAIN)

    def _restore_module_modes(
        self,
        snapshot: tuple[tuple[Any, bool], ...],
    ) -> None:
        errors: list[BaseException] = []
        for module, training in snapshot:
            try:
                module.train(training)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise ComponentLifecycleError(
                "failed to restore component module modes: "
                + "; ".join(str(item) for item in errors)
            ) from errors[0]

    def _apply_mode(
        self,
        mode: ExecutionMode,
        snapshot: dict[str, Residency],
    ) -> None:
        required = self.resource_plan.roles_for(mode)
        try:
            for binding in self.components.bindings:
                owner = self.resource_plan.owner_for(
                    binding,
                    prepared_component_names=self._prepared_component_names,
                )
                if owner is ResourceOwner.PREPARED_BACKEND:
                    if self._residency[binding.name] is not Residency.PREPARED:
                        raise ComponentResidencyError(
                            f"prepared component {binding.name!r} lost backend "
                            "ownership"
                        )
                    continue
                if owner is ResourceOwner.STATIC:
                    continue
                target = (
                    Residency.RESIDENT
                    if any(role in required for role in binding.roles)
                    else Residency.OFFLOADED
                )
                self._set_residency(binding, target)
        except BaseException as exc:
            try:
                self._restore_residency(snapshot)
            except BaseException as rollback_exc:  # noqa: BLE001
                if hasattr(exc, "add_note"):
                    exc.add_note(f"residency rollback failed: {rollback_exc}")
            raise

    def _restore_residency(self, snapshot: dict[str, Residency]) -> None:
        errors: list[BaseException] = []
        for binding in reversed(self.components.bindings):
            target = snapshot[binding.name]
            try:
                self._set_residency(binding, target)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise ComponentResidencyError(
                "failed to restore component residency: "
                + "; ".join(str(item) for item in errors)
            ) from errors[0]

    def _set_residency(
        self,
        binding: ComponentBinding,
        target: Residency,
    ) -> None:
        current = self._residency[binding.name]
        if current is target:
            return
        if current in {Residency.STATIC, Residency.PREPARED} or target in {
            Residency.STATIC,
            Residency.PREPARED,
        }:
            raise ComponentResidencyError(
                f"backend/static component {binding.name!r} cannot change residency"
            )
        device = (
            self.execution_device
            if target is Residency.RESIDENT
            else self.offload_device
        )
        self._move(binding, device)
        self._residency[binding.name] = target

    @staticmethod
    def _move(binding: ComponentBinding, device: Any) -> None:
        move = getattr(binding.component, "to", None)
        if not callable(move):
            raise ComponentResidencyError(
                f"managed component {binding.name!r} does not expose to(device)"
            )
        result = move(device)
        if result is not None and result is not binding.component:
            raise ComponentResidencyError(
                f"component {binding.name!r} returned a new object from to(device)"
            )

    def close(self) -> None:
        with self._lock:
            if self._state is OwnershipState.CLOSED:
                return
            if self._mode is not ExecutionMode.IDLE:
                raise ComponentLifecycleError(
                    "cannot close ComponentManager while an execution mode is active"
                )
            if self._state is OwnershipState.UNLOADED:
                self._state = OwnershipState.CLOSED
                return
            errors: list[BaseException] = []
            assert self._components is not None
            if self._prepared_handle is not None:
                close_handle = getattr(self._prepared_handle, "close", None)
                if callable(close_handle):
                    try:
                        close_handle()
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)
            for binding in reversed(self._components.bindings):
                if self._residency[binding.name] is Residency.RESIDENT:
                    try:
                        self._move(binding, self.offload_device)
                        self._residency[binding.name] = Residency.OFFLOADED
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)
                try:
                    _close_binding(binding)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
            self._components = None
            self._residency.clear()
            self._runtime_bound = None
            self._parameter_state = None
            self._parameter_dtype_owner = None
            self._model_execution_numerics = None
            self._prepared_handle = None
            self._prepared_component_names = frozenset()
            self.adapter._clear_prepared_components()
            self.adapter._clear_model_execution_numerics()
            self._state = OwnershipState.CLOSED
            if errors:
                raise ComponentManagerError(
                    "component close errors: " + "; ".join(str(item) for item in errors)
                ) from errors[0]
