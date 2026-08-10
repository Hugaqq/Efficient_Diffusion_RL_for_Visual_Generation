"""One distributed root and one forward handle for all shardable model modules."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch

from visual_rl.models.lifecycle.components import (
    ComponentRole,
    ModelComponents,
)

__all__ = (
    "PreparedBundleError",
    "PreparedComponentHandle",
    "PreparedModelBundle",
    "prepare_model_bundle",
    "validate_optimizer_parameter_subset",
)


class PreparedBundleError(RuntimeError):
    """Raised when distributed root ownership or preparation is ambiguous."""


_SHARDABLE_ROLES = frozenset(
    {
        ComponentRole.INFERENCE,
        ComponentRole.TRAINABLE,
        ComponentRole.REFERENCE,
    }
)
_ACTIVE_BUNDLE: ContextVar[int | None] = ContextVar(
    "visual_rl_active_prepared_bundle",
    default=None,
)


class PreparedModelBundle(torch.nn.Module):
    """The only root passed to ``accelerator.prepare`` for model modules."""

    def __init__(self, components: ModelComponents) -> None:
        super().__init__()
        if not isinstance(components, ModelComponents):
            raise TypeError("components must be ModelComponents")
        selected = tuple(
            binding
            for binding in components.bindings
            if any(role in _SHARDABLE_ROLES for role in binding.roles)
        )
        if not selected:
            raise PreparedBundleError("no shardable model components were declared")
        if not any(ComponentRole.TRAINABLE in item.roles for item in selected):
            raise PreparedBundleError("prepared bundle has no trainable component")

        names: list[str] = []
        no_split_modules: list[str] = []
        for binding in selected:
            if not isinstance(binding.component, torch.nn.Module):
                raise PreparedBundleError(
                    f"shardable component {binding.name!r} must be torch.nn.Module"
                )
            if "." in binding.name:
                raise PreparedBundleError(
                    f"component name {binding.name!r} cannot contain '.'"
                )
            if hasattr(self, binding.name):
                raise PreparedBundleError(
                    f"component name {binding.name!r} collides with nn.Module root"
                )
            self.add_module(binding.name, binding.component)
            names.append(binding.name)
            declared = getattr(binding.component, "_no_split_modules", ())
            if declared is None:
                declared = ()
            if not isinstance(declared, (tuple, list)) or any(
                not isinstance(item, str) or not item for item in declared
            ):
                raise PreparedBundleError(
                    f"component {binding.name!r} has invalid _no_split_modules"
                )
            for item in declared:
                if item not in no_split_modules:
                    no_split_modules.append(item)

        self._component_names = tuple(names)
        self._no_split_modules = no_split_modules
        self._validate_parameter_ownership()
        self._forward_guard_handles = tuple(
            self.component(name).register_forward_pre_hook(self._forward_guard)
            for name in self._component_names
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        return self._component_names

    def owns(self, component_name: str) -> bool:
        return component_name in self._component_names

    def component(self, component_name: str) -> torch.nn.Module:
        if not self.owns(component_name):
            raise KeyError(component_name)
        return self._modules[component_name]

    def forward(self, component_name: str, *args: Any, **kwargs: Any) -> Any:
        """Route one child call through the prepared root's ``__call__`` path."""

        if not isinstance(component_name, str) or not component_name:
            raise TypeError("component_name must be a non-empty string")
        token = _ACTIVE_BUNDLE.set(id(self))
        try:
            return self.component(component_name)(*args, **kwargs)
        finally:
            _ACTIVE_BUNDLE.reset(token)

    def release_forward_guards(self) -> None:
        handles = self._forward_guard_handles
        self._forward_guard_handles = ()
        for handle in handles:
            handle.remove()

    def _forward_guard(self, _module: object, _args: object) -> None:
        if _ACTIVE_BUNDLE.get() != id(self):
            raise PreparedBundleError(
                "prepared component forward bypassed PreparedComponentHandle"
            )

    def trainable_named_parameters(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def trainable_parameters(self) -> tuple[Any, ...]:
        return tuple(
            parameter for _name, parameter in self.trainable_named_parameters()
        )

    def _validate_parameter_ownership(self) -> None:
        paths = tuple(self.named_parameters(remove_duplicate=False))
        names = tuple(name for name, _parameter in paths)
        identities = tuple(id(parameter) for _name, parameter in paths)
        if len(names) != len(set(names)):
            raise PreparedBundleError("prepared parameter names must be unique")
        if len(identities) != len(set(identities)):
            duplicates = sorted(
                {
                    name
                    for name, parameter in paths
                    if identities.count(id(parameter)) > 1
                }
            )
            raise PreparedBundleError(
                f"one parameter object has multiple prepared paths: {duplicates}"
            )
        if not self.trainable_parameters():
            raise PreparedBundleError(
                "prepared bundle requires at least one requires_grad parameter"
            )


class PreparedComponentHandle:
    """Only Adapter-visible route to the distributed model root."""

    def __init__(
        self,
        *,
        source_bundle: PreparedModelBundle,
        prepared_root: torch.nn.Module,
        optimizer: object,
        scheduler: object | None,
        accelerator: object,
    ) -> None:
        if not isinstance(source_bundle, PreparedModelBundle):
            raise TypeError("source_bundle must be PreparedModelBundle")
        if not isinstance(prepared_root, torch.nn.Module):
            raise TypeError("prepared_root must be torch.nn.Module")
        if optimizer is None:
            raise TypeError("prepared optimizer must not be None")
        self._source_bundle: PreparedModelBundle | None = source_bundle
        self._prepared_root: torch.nn.Module | None = prepared_root
        self._optimizer: object | None = optimizer
        self._scheduler: object | None = scheduler
        self._accelerator: object | None = accelerator
        self._closed = False

    @property
    def component_names(self) -> tuple[str, ...]:
        source_bundle, _prepared_root, _optimizer, _accelerator = self._live_state()
        return source_bundle.component_names

    @property
    def prepared_root(self) -> torch.nn.Module:
        _source_bundle, prepared_root, _optimizer, _accelerator = self._live_state()
        return prepared_root

    @property
    def optimizer(self) -> object:
        _source_bundle, _prepared_root, optimizer, _accelerator = self._live_state()
        return optimizer

    @property
    def scheduler(self) -> object | None:
        self._require_open()
        return self._scheduler

    @property
    def accumulation_root(self) -> torch.nn.Module:
        _source_bundle, prepared_root, _optimizer, _accelerator = self._live_state()
        return prepared_root

    def owns(self, component_name: str) -> bool:
        source_bundle, _prepared_root, _optimizer, _accelerator = self._live_state()
        return source_bundle.owns(component_name)

    def forward(self, component_name: str, *args: Any, **kwargs: Any) -> Any:
        source_bundle, prepared_root, _optimizer, _accelerator = self._live_state()
        if not source_bundle.owns(component_name):
            raise KeyError(component_name)
        return prepared_root(component_name, *args, **kwargs)

    @contextmanager
    def accumulate(self) -> Iterator[torch.nn.Module]:
        """Bind gradient accumulation to the prepared root, never a raw child."""

        _source_bundle, prepared_root, _optimizer, accelerator = self._live_state()
        accumulate = getattr(accelerator, "accumulate", None)
        if not callable(accumulate):
            raise PreparedBundleError("accelerator does not expose accumulate(root)")
        with accumulate(prepared_root):
            yield prepared_root

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        source_bundle = self._source_bundle
        try:
            if source_bundle is not None:
                source_bundle.release_forward_guards()
        finally:
            # A propagated OOM can keep controller traceback frames alive.  Do
            # not let such a frame retain the prepared model, optimizer, or
            # Accelerator through this otherwise-closed handle.
            self._source_bundle = None
            self._prepared_root = None
            self._optimizer = None
            self._scheduler = None
            self._accelerator = None

    def _require_open(self) -> None:
        if self._closed:
            raise PreparedBundleError("prepared component handle is closed")

    def _live_state(
        self,
    ) -> tuple[PreparedModelBundle, torch.nn.Module, object, object]:
        self._require_open()
        source_bundle = self._source_bundle
        prepared_root = self._prepared_root
        optimizer = self._optimizer
        accelerator = self._accelerator
        if (
            source_bundle is None
            or prepared_root is None
            or optimizer is None
            or accelerator is None
        ):
            raise PreparedBundleError("prepared component handle lost live resources")
        return source_bundle, prepared_root, optimizer, accelerator


def validate_optimizer_parameter_subset(
    bundle: PreparedModelBundle,
    optimizer: object,
) -> None:
    """Require optimizer parameters to equal the root's requires-grad subset."""

    if not isinstance(bundle, PreparedModelBundle):
        raise TypeError("bundle must be PreparedModelBundle")
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise PreparedBundleError("optimizer must expose non-empty param_groups")
    optimizer_parameters: list[object] = []
    for group in groups:
        if not isinstance(group, dict) or "params" not in group:
            raise PreparedBundleError("optimizer param group is malformed")
        parameters = group["params"]
        if not isinstance(parameters, (tuple, list)):
            raise PreparedBundleError("optimizer group params must be a sequence")
        optimizer_parameters.extend(parameters)
    optimizer_ids = tuple(id(parameter) for parameter in optimizer_parameters)
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise PreparedBundleError("optimizer contains duplicate parameter objects")
    required = bundle.trainable_parameters()
    required_ids = tuple(id(parameter) for parameter in required)
    if set(optimizer_ids) != set(required_ids) or len(optimizer_ids) != len(
        required_ids
    ):
        raise PreparedBundleError(
            "optimizer parameter set must exactly equal prepared requires_grad subset"
        )


def prepare_model_bundle(
    components: ModelComponents,
    *,
    accelerator: object,
    optimizer: object,
    scheduler: object | None = None,
) -> PreparedComponentHandle:
    """Build and prepare one root with exactly one accelerator call."""

    prepare = getattr(accelerator, "prepare", None)
    if not callable(prepare):
        raise TypeError("accelerator must expose prepare(...)")
    bundle = PreparedModelBundle(components)
    try:
        validate_optimizer_parameter_subset(bundle, optimizer)
        arguments = (
            (bundle, optimizer) if scheduler is None else (bundle, optimizer, scheduler)
        )
        prepared = prepare(*arguments)
        if not isinstance(prepared, (tuple, list)) or len(prepared) != len(arguments):
            raise PreparedBundleError(
                "accelerator.prepare must return one prepared value per input"
            )
    except BaseException:
        bundle.release_forward_guards()
        raise
    try:
        prepared_root = prepared[0]
        prepared_optimizer = prepared[1]
        prepared_scheduler = prepared[2] if scheduler is not None else None
        return PreparedComponentHandle(
            source_bundle=bundle,
            prepared_root=prepared_root,
            optimizer=prepared_optimizer,
            scheduler=prepared_scheduler,
            accelerator=accelerator,
        )
    except BaseException:
        bundle.release_forward_guards()
        raise
