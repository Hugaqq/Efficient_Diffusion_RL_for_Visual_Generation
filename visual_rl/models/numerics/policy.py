"""Typed, content-addressed model parameter and forward numerics policies.

Parameter storage dtype, forward autocast, and logical parameter views are
different concerns.  This module keeps them separate and deliberately has no
distributed-runtime integration: the dtype owner must run before preparation,
while callers open a fresh autocast context around each forward.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

from visual_rl.models.lifecycle.components import (
    ExecutionMode,
    ModelComponents,
    OwnershipState,
)
from visual_rl.models.numerics.execution import ParameterView
from visual_rl.models.state.parameters import ParameterStateManager

__all__ = (
    "FloatingBufferPolicy",
    "ForwardAutocastPolicy",
    "FrozenParameterPolicy",
    "ModelExecutionNumericsEvidence",
    "ModelNumericsPolicyError",
    "ParameterDTypePolicy",
    "ParameterDTypePolicyOwner",
    "ParameterViewEvidence",
    "ParameterViewMode",
    "require_parameter_view_evidence",
)


_SCHEMA_VERSION = 1
_FLOATING_DTYPES = frozenset(
    {
        "bfloat16",
        "float16",
        "float32",
        "float64",
    }
)
_ENABLED_AUTOCAST_DTYPES = {
    "cpu": frozenset({"bfloat16"}),
    "cuda": frozenset({"bfloat16", "float16"}),
}


class ModelNumericsPolicyError(RuntimeError):
    """A model numerics policy is invalid, stale, or used out of order."""


class FrozenParameterPolicy(str, Enum):
    """Storage policy for parameters whose ``requires_grad`` is false."""

    PRESERVE_LOADED = "preserve_loaded"
    EXPLICIT_DTYPE = "explicit_dtype"


class FloatingBufferPolicy(str, Enum):
    """Storage policy for floating-point module buffers."""

    PRESERVE_LOADED = "preserve_loaded"
    EXPLICIT_DTYPE = "explicit_dtype"


class ParameterViewMode(str, Enum):
    """Concrete realization used to expose one logical parameter view."""

    CURRENT = "current"
    LORA_DISABLE = "lora_disable"
    FROZEN_COPY = "frozen_copy"
    IN_PLACE_SWAP = "in_place_swap"
    EMA_COPY = "ema_copy"


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _identity(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ModelNumericsPolicyError(
            f"{field_name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _dtype_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value not in _FLOATING_DTYPES:
        raise ModelNumericsPolicyError(
            f"{field_name} must be one of {sorted(_FLOATING_DTYPES)}"
        )
    return value


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def _enum_value(value: object, enum_type: type[Enum], *, field_name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        allowed = tuple(item.value for item in enum_type)
        raise ModelNumericsPolicyError(
            f"{field_name} must be one of {allowed}"
        ) from None


def _strict_keys(
    payload: Mapping[str, object],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected.difference(payload))
        unknown = sorted(set(payload).difference(expected))
        raise ModelNumericsPolicyError(
            f"{context} has invalid fields; missing={missing}, unknown={unknown}"
        )


@dataclass(frozen=True, slots=True)
class ParameterDTypePolicy:
    """Canonical storage dtype policy applied once before model preparation."""

    trainable_parameter_dtype: str
    frozen_parameter_policy: FrozenParameterPolicy = (
        FrozenParameterPolicy.PRESERVE_LOADED
    )
    frozen_parameter_dtype: str | None = None
    floating_buffer_policy: FloatingBufferPolicy = FloatingBufferPolicy.PRESERVE_LOADED
    floating_buffer_dtype: str | None = None
    schema_version: int = _SCHEMA_VERSION
    _policy_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ModelNumericsPolicyError(
                "parameter dtype policy schema_version must be 1"
            )
        trainable_dtype = _dtype_name(
            self.trainable_parameter_dtype,
            field_name="trainable_parameter_dtype",
        )
        frozen_policy = _enum_value(
            self.frozen_parameter_policy,
            FrozenParameterPolicy,
            field_name="frozen_parameter_policy",
        )
        buffer_policy = _enum_value(
            self.floating_buffer_policy,
            FloatingBufferPolicy,
            field_name="floating_buffer_policy",
        )
        frozen_dtype = self.frozen_parameter_dtype
        buffer_dtype = self.floating_buffer_dtype
        if frozen_policy is FrozenParameterPolicy.PRESERVE_LOADED:
            if frozen_dtype is not None:
                raise ModelNumericsPolicyError(
                    "frozen_parameter_dtype must be None when frozen parameters "
                    "preserve their loaded dtype"
                )
        elif frozen_dtype is None:
            raise ModelNumericsPolicyError(
                "explicit frozen parameter policy requires frozen_parameter_dtype"
            )
        else:
            frozen_dtype = _dtype_name(
                frozen_dtype,
                field_name="frozen_parameter_dtype",
            )
        if buffer_policy is FloatingBufferPolicy.PRESERVE_LOADED:
            if buffer_dtype is not None:
                raise ModelNumericsPolicyError(
                    "floating_buffer_dtype must be None when buffers preserve "
                    "their loaded dtype"
                )
        elif buffer_dtype is None:
            raise ModelNumericsPolicyError(
                "explicit floating buffer policy requires floating_buffer_dtype"
            )
        else:
            buffer_dtype = _dtype_name(
                buffer_dtype,
                field_name="floating_buffer_dtype",
            )
        object.__setattr__(self, "trainable_parameter_dtype", trainable_dtype)
        object.__setattr__(self, "frozen_parameter_policy", frozen_policy)
        object.__setattr__(self, "frozen_parameter_dtype", frozen_dtype)
        object.__setattr__(self, "floating_buffer_policy", buffer_policy)
        object.__setattr__(self, "floating_buffer_dtype", buffer_dtype)
        object.__setattr__(self, "_policy_id", _identity(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trainable_parameter_dtype": self.trainable_parameter_dtype,
            "frozen_parameter_policy": self.frozen_parameter_policy.value,
            "frozen_parameter_dtype": self.frozen_parameter_dtype,
            "floating_buffer_policy": self.floating_buffer_policy.value,
            "floating_buffer_dtype": self.floating_buffer_dtype,
        }

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def to_evidence_payload(self) -> dict[str, object]:
        return {**self.to_payload(), "policy_id": self.policy_id}

    @classmethod
    def from_evidence_payload(cls, payload: object) -> ParameterDTypePolicy:
        if not isinstance(payload, Mapping):
            raise TypeError("parameter dtype policy payload must be a mapping")
        expected = frozenset(
            {
                "schema_version",
                "trainable_parameter_dtype",
                "frozen_parameter_policy",
                "frozen_parameter_dtype",
                "floating_buffer_policy",
                "floating_buffer_dtype",
                "policy_id",
            }
        )
        _strict_keys(payload, expected=expected, context="parameter dtype policy")
        result = cls(
            trainable_parameter_dtype=payload["trainable_parameter_dtype"],  # type: ignore[arg-type]
            frozen_parameter_policy=payload["frozen_parameter_policy"],  # type: ignore[arg-type]
            frozen_parameter_dtype=payload["frozen_parameter_dtype"],  # type: ignore[arg-type]
            floating_buffer_policy=payload["floating_buffer_policy"],  # type: ignore[arg-type]
            floating_buffer_dtype=payload["floating_buffer_dtype"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if payload["policy_id"] != result.policy_id:
            raise ModelNumericsPolicyError("parameter dtype policy identity mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ForwardAutocastPolicy:
    """One forward-only autocast scope for a stage and logical parameter view."""

    stage: ExecutionMode
    parameter_view: ParameterView
    device_type: str
    compute_dtype: str
    enabled: bool
    schema_version: int = _SCHEMA_VERSION
    _policy_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ModelNumericsPolicyError(
                "forward autocast policy schema_version must be 1"
            )
        try:
            stage = ExecutionMode(self.stage)
        except (TypeError, ValueError):
            raise ModelNumericsPolicyError(
                f"invalid forward autocast stage: {self.stage!r}"
            ) from None
        if stage is ExecutionMode.IDLE:
            raise ModelNumericsPolicyError("IDLE has no forward autocast policy")
        view = _enum_value(
            self.parameter_view,
            ParameterView,
            field_name="parameter_view",
        )
        if not isinstance(self.device_type, str) or self.device_type not in {
            "cpu",
            "cuda",
        }:
            raise ModelNumericsPolicyError("device_type must be one of ('cpu', 'cuda')")
        dtype = _dtype_name(self.compute_dtype, field_name="compute_dtype")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        if self.enabled and dtype not in _ENABLED_AUTOCAST_DTYPES[self.device_type]:
            allowed = sorted(_ENABLED_AUTOCAST_DTYPES[self.device_type])
            raise ModelNumericsPolicyError(
                f"enabled {self.device_type} autocast only supports {allowed}"
            )
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "parameter_view", view)
        object.__setattr__(self, "compute_dtype", dtype)
        object.__setattr__(self, "_policy_id", _identity(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "parameter_view": self.parameter_view.value,
            "device_type": self.device_type,
            "compute_dtype": self.compute_dtype,
            "enabled": self.enabled,
        }

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def to_evidence_payload(self) -> dict[str, object]:
        return {**self.to_payload(), "policy_id": self.policy_id}

    @classmethod
    def from_evidence_payload(cls, payload: object) -> ForwardAutocastPolicy:
        if not isinstance(payload, Mapping):
            raise TypeError("forward autocast policy payload must be a mapping")
        expected = frozenset(
            {
                "schema_version",
                "stage",
                "parameter_view",
                "device_type",
                "compute_dtype",
                "enabled",
                "policy_id",
            }
        )
        _strict_keys(payload, expected=expected, context="forward autocast policy")
        result = cls(
            stage=payload["stage"],  # type: ignore[arg-type]
            parameter_view=payload["parameter_view"],  # type: ignore[arg-type]
            device_type=payload["device_type"],  # type: ignore[arg-type]
            compute_dtype=payload["compute_dtype"],  # type: ignore[arg-type]
            enabled=payload["enabled"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if payload["policy_id"] != result.policy_id:
            raise ModelNumericsPolicyError("forward autocast policy identity mismatch")
        return result

    @contextmanager
    def forward_context(self) -> Iterator[None]:
        """Open exactly one forward autocast scope and restore prior state.

        The returned context is intentionally named for forward use and this
        module exposes no optimizer wrapper.  Callers must leave it before
        backward/optimizer execution.
        """

        import torch

        if not self.enabled:
            with nullcontext():
                yield
            return
        available = getattr(torch.amp.autocast_mode, "is_autocast_available", None)
        if not callable(available) or not available(self.device_type):
            raise ModelNumericsPolicyError(
                f"autocast is unavailable for device type {self.device_type!r}"
            )
        if self.device_type == "cuda":
            if not torch.cuda.is_available():
                raise ModelNumericsPolicyError(
                    "CUDA autocast requested but CUDA is unavailable"
                )
            if self.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ModelNumericsPolicyError(
                    "CUDA bfloat16 autocast is unsupported by the active runtime"
                )
        with torch.autocast(
            device_type=self.device_type,
            dtype=_torch_dtype(self.compute_dtype),
            enabled=True,
        ):
            yield

    def run_forward(
        self, forward: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Invoke one callable inside this policy's forward-only scope."""

        if not callable(forward):
            raise TypeError("forward must be callable")
        with self.forward_context():
            return forward(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class ParameterViewEvidence:
    """Typed proof of how a current, reference, or EMA view is realized."""

    parameter_view: ParameterView
    mode: ParameterViewMode
    owner_component_names: tuple[str, ...]
    restorable_state_names: tuple[str, ...]
    source_projection_id: str
    mutates_parameters_in_place: bool
    schema_version: int = _SCHEMA_VERSION
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ModelNumericsPolicyError(
                "parameter view evidence schema_version must be 1"
            )
        view = _enum_value(
            self.parameter_view,
            ParameterView,
            field_name="parameter_view",
        )
        mode = _enum_value(self.mode, ParameterViewMode, field_name="mode")
        owners = _canonical_names(
            self.owner_component_names,
            field_name="owner_component_names",
        )
        state_names = _canonical_names(
            self.restorable_state_names,
            field_name="restorable_state_names",
        )
        projection_id = _sha256(
            self.source_projection_id,
            field_name="source_projection_id",
        )
        if type(self.mutates_parameters_in_place) is not bool:
            raise TypeError("mutates_parameters_in_place must be bool")
        allowed_modes = {
            ParameterView.CURRENT: frozenset({ParameterViewMode.CURRENT}),
            ParameterView.REFERENCE: frozenset(
                {
                    ParameterViewMode.LORA_DISABLE,
                    ParameterViewMode.FROZEN_COPY,
                    ParameterViewMode.IN_PLACE_SWAP,
                }
            ),
            ParameterView.EMA: frozenset(
                {
                    ParameterViewMode.EMA_COPY,
                    ParameterViewMode.FROZEN_COPY,
                    ParameterViewMode.IN_PLACE_SWAP,
                }
            ),
        }
        if mode not in allowed_modes[view]:
            raise ModelNumericsPolicyError(
                f"mode {mode.value!r} cannot realize {view.value!r} parameters"
            )
        expected_mutation = mode is ParameterViewMode.IN_PLACE_SWAP
        if self.mutates_parameters_in_place is not expected_mutation:
            raise ModelNumericsPolicyError(
                "mutates_parameters_in_place disagrees with parameter view mode"
            )
        unknown_owners = sorted(
            state_name
            for state_name in state_names
            if "." not in state_name or state_name.split(".", 1)[0] not in owners
        )
        if unknown_owners:
            raise ModelNumericsPolicyError(
                "restorable state names must use an owner component prefix: "
                f"{unknown_owners}"
            )
        object.__setattr__(self, "parameter_view", view)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "owner_component_names", owners)
        object.__setattr__(self, "restorable_state_names", state_names)
        object.__setattr__(self, "source_projection_id", projection_id)
        object.__setattr__(self, "evidence_id", _identity(self._identity_payload()))

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parameter_view": self.parameter_view.value,
            "mode": self.mode.value,
            "owner_component_names": list(self.owner_component_names),
            "restorable_state_names": list(self.restorable_state_names),
            "source_projection_id": self.source_projection_id,
            "mutates_parameters_in_place": self.mutates_parameters_in_place,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._identity_payload(), "evidence_id": self.evidence_id}

    def assert_integrity(self) -> None:
        if self.evidence_id != _identity(self._identity_payload()):
            raise ModelNumericsPolicyError("parameter view evidence identity mismatch")

    def assert_matches(
        self,
        *,
        parameter_view: ParameterView,
        source_projection_id: str,
        owner_component_names: tuple[str, ...] | None = None,
    ) -> None:
        self.assert_integrity()
        expected_view = _enum_value(
            parameter_view,
            ParameterView,
            field_name="parameter_view",
        )
        expected_projection = _sha256(
            source_projection_id,
            field_name="source_projection_id",
        )
        if self.parameter_view is not expected_view:
            raise ModelNumericsPolicyError("parameter view evidence has the wrong view")
        if self.source_projection_id != expected_projection:
            raise ModelNumericsPolicyError(
                "parameter view evidence has the wrong source projection"
            )
        if owner_component_names is not None and self.owner_component_names != (
            _canonical_names(
                owner_component_names,
                field_name="owner_component_names",
            )
        ):
            raise ModelNumericsPolicyError(
                "parameter view evidence has the wrong component owners"
            )

    @classmethod
    def from_payload(cls, payload: object) -> ParameterViewEvidence:
        if not isinstance(payload, Mapping):
            raise TypeError("parameter view evidence payload must be a mapping")
        expected = frozenset(
            {
                "schema_version",
                "parameter_view",
                "mode",
                "owner_component_names",
                "restorable_state_names",
                "source_projection_id",
                "mutates_parameters_in_place",
                "evidence_id",
            }
        )
        _strict_keys(payload, expected=expected, context="parameter view evidence")
        if payload["schema_version"] != _SCHEMA_VERSION:
            raise ModelNumericsPolicyError(
                "parameter view evidence schema_version is unsupported"
            )
        raw_owners = payload["owner_component_names"]
        raw_states = payload["restorable_state_names"]
        if not isinstance(raw_owners, list) or not isinstance(raw_states, list):
            raise TypeError("parameter view evidence owner/state names must be lists")
        result = cls(
            parameter_view=payload["parameter_view"],  # type: ignore[arg-type]
            mode=payload["mode"],  # type: ignore[arg-type]
            owner_component_names=tuple(raw_owners),
            restorable_state_names=tuple(raw_states),
            source_projection_id=payload["source_projection_id"],  # type: ignore[arg-type]
            mutates_parameters_in_place=payload["mutates_parameters_in_place"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if payload["evidence_id"] != result.evidence_id:
            raise ModelNumericsPolicyError("parameter view evidence identity mismatch")
        return result


def _canonical_names(value: object, *, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ModelNumericsPolicyError(f"{field_name} must be a non-empty tuple")
    if any(
        not isinstance(item, str) or not item or item.strip() != item for item in value
    ):
        raise ModelNumericsPolicyError(
            f"{field_name} entries must be non-empty canonical strings"
        )
    if len(value) != len(set(value)):
        raise ModelNumericsPolicyError(f"{field_name} entries must be unique")
    return tuple(sorted(value))


def require_parameter_view_evidence(
    value: object,
    *,
    parameter_view: ParameterView,
    source_projection_id: str,
    owner_component_names: tuple[str, ...] | None = None,
) -> ParameterViewEvidence:
    """Require typed view evidence; a boolean readiness flag is never proof."""

    if not isinstance(value, ParameterViewEvidence):
        raise TypeError(
            "parameter view readiness requires ParameterViewEvidence; bool or "
            "untyped values are forbidden"
        )
    value.assert_matches(
        parameter_view=parameter_view,
        source_projection_id=source_projection_id,
        owner_component_names=owner_component_names,
    )
    return value


@dataclass(frozen=True, slots=True)
class ModelExecutionNumericsEvidence:
    """Canonical G3/checkpoint identity for storage, forward, and view policies."""

    parameter_dtype_policy: ParameterDTypePolicy
    forward_autocast_policies: tuple[ForwardAutocastPolicy, ...]
    parameter_view_evidence: tuple[ParameterViewEvidence, ...]
    schema_version: int = _SCHEMA_VERSION
    execution_numerics_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ModelNumericsPolicyError(
                "model execution numerics schema_version must be 1"
            )
        if not isinstance(self.parameter_dtype_policy, ParameterDTypePolicy):
            raise TypeError("parameter_dtype_policy must be ParameterDTypePolicy")
        policies = self.forward_autocast_policies
        if type(policies) is not tuple or not policies:
            raise ModelNumericsPolicyError(
                "forward_autocast_policies must be a non-empty tuple"
            )
        if any(not isinstance(item, ForwardAutocastPolicy) for item in policies):
            raise TypeError(
                "forward_autocast_policies must contain ForwardAutocastPolicy"
            )
        policy_keys = tuple(
            (item.stage.value, item.parameter_view.value) for item in policies
        )
        if len(policy_keys) != len(set(policy_keys)):
            raise ModelNumericsPolicyError(
                "forward autocast stage/view bindings must be unique"
            )
        canonical_policies = tuple(
            sorted(
                policies,
                key=lambda item: (item.stage.value, item.parameter_view.value),
            )
        )

        views = self.parameter_view_evidence
        if type(views) is not tuple or not views:
            raise ModelNumericsPolicyError(
                "parameter_view_evidence must be a non-empty tuple"
            )
        if any(not isinstance(item, ParameterViewEvidence) for item in views):
            raise TypeError(
                "parameter_view_evidence must contain ParameterViewEvidence"
            )
        view_keys = tuple(item.parameter_view for item in views)
        if len(view_keys) != len(set(view_keys)):
            raise ModelNumericsPolicyError(
                "parameter view evidence must define each logical view once"
            )
        if ParameterView.CURRENT not in view_keys:
            raise ModelNumericsPolicyError(
                "model execution numerics requires current parameter view evidence"
            )
        canonical_views = tuple(
            sorted(views, key=lambda item: item.parameter_view.value)
        )
        projection_ids = {item.source_projection_id for item in canonical_views}
        if len(projection_ids) != 1:
            raise ModelNumericsPolicyError(
                "all parameter views must use one source model-state projection"
            )
        missing_views = sorted(
            {
                item.parameter_view.value
                for item in canonical_policies
                if item.parameter_view not in view_keys
            }
        )
        if missing_views:
            raise ModelNumericsPolicyError(
                "forward autocast policies lack typed parameter view evidence: "
                f"{missing_views}"
            )
        object.__setattr__(self, "forward_autocast_policies", canonical_policies)
        object.__setattr__(self, "parameter_view_evidence", canonical_views)
        object.__setattr__(
            self,
            "execution_numerics_id",
            _identity(self._identity_payload()),
        )

    @property
    def source_projection_id(self) -> str:
        return self.parameter_view_evidence[0].source_projection_id

    def autocast_policy(
        self,
        stage: ExecutionMode,
        parameter_view: ParameterView,
    ) -> ForwardAutocastPolicy:
        try:
            key = (ExecutionMode(stage), ParameterView(parameter_view))
        except (TypeError, ValueError):
            raise ModelNumericsPolicyError(
                "invalid stage/view forward autocast lookup"
            ) from None
        matches = tuple(
            item
            for item in self.forward_autocast_policies
            if (item.stage, item.parameter_view) == key
        )
        if len(matches) != 1:
            raise ModelNumericsPolicyError(
                "no exact forward autocast policy exists for "
                f"{key[0].value}/{key[1].value}"
            )
        return matches[0]

    def view_evidence(self, parameter_view: ParameterView) -> ParameterViewEvidence:
        try:
            resolved = ParameterView(parameter_view)
        except (TypeError, ValueError):
            raise ModelNumericsPolicyError("invalid parameter view lookup") from None
        matches = tuple(
            item
            for item in self.parameter_view_evidence
            if item.parameter_view is resolved
        )
        if len(matches) != 1:
            raise ModelNumericsPolicyError(
                f"no exact evidence exists for parameter view {resolved.value!r}"
            )
        return matches[0]

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parameter_dtype_policy": (
                self.parameter_dtype_policy.to_evidence_payload()
            ),
            "forward_autocast_policies": [
                item.to_evidence_payload() for item in self.forward_autocast_policies
            ],
            "parameter_view_evidence": [
                item.to_payload() for item in self.parameter_view_evidence
            ],
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._identity_payload(),
            "execution_numerics_id": self.execution_numerics_id,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ModelExecutionNumericsEvidence:
        if not isinstance(payload, Mapping):
            raise TypeError("model execution numerics payload must be a mapping")
        expected = frozenset(
            {
                "schema_version",
                "parameter_dtype_policy",
                "forward_autocast_policies",
                "parameter_view_evidence",
                "execution_numerics_id",
            }
        )
        _strict_keys(payload, expected=expected, context="model execution numerics")
        raw_autocast = payload["forward_autocast_policies"]
        raw_views = payload["parameter_view_evidence"]
        if not isinstance(raw_autocast, list) or not isinstance(raw_views, list):
            raise TypeError("model execution numerics policy collections must be lists")
        result = cls(
            parameter_dtype_policy=ParameterDTypePolicy.from_evidence_payload(
                payload["parameter_dtype_policy"]
            ),
            forward_autocast_policies=tuple(
                ForwardAutocastPolicy.from_evidence_payload(item)
                for item in raw_autocast
            ),
            parameter_view_evidence=tuple(
                ParameterViewEvidence.from_payload(item) for item in raw_views
            ),
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if payload["execution_numerics_id"] != result.execution_numerics_id:
            raise ModelNumericsPolicyError("model execution numerics identity mismatch")
        return result


@dataclass(frozen=True, slots=True)
class _TensorRecord:
    kind: str
    path: str
    tensor: object = field(repr=False, compare=False)
    object_id: int
    shape: tuple[int, ...]
    dtype: str
    dtype_object: object = field(repr=False, compare=False)
    requires_grad: bool | None

    def semantic_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
        }


class ParameterDTypePolicyOwner:
    """Single pre-prepare owner of all parameter and floating-buffer casts.

    The owner snapshots both object identity and semantic tensor topology.  It
    walks individual parameters and buffers, never calling ``Module.to`` on a
    whole component.  A new :class:`ParameterStateManager` is bound after a
    successful cast because dtype is part of trainable topology identity.

    Distributed prepare may legitimately rebind buffer objects while moving a
    module between devices.  Parameter object identity remains strict because
    it is shared with the optimizer.  The first prepared validation accepts
    only semantically identical buffer replacements, then locks their new
    identities for every subsequent validation.
    """

    def __init__(
        self,
        components: ModelComponents,
        *,
        ownership_state: Callable[[], OwnershipState],
        parameter_state: ParameterStateManager | None = None,
    ) -> None:
        if not isinstance(components, ModelComponents):
            raise TypeError("components must be ModelComponents")
        if not callable(ownership_state):
            raise TypeError("ownership_state must be callable")
        if parameter_state is None:
            parameter_state = ParameterStateManager(components)
        if not isinstance(parameter_state, ParameterStateManager):
            raise TypeError("parameter_state must be ParameterStateManager")
        if parameter_state.components is not components:
            raise ModelNumericsPolicyError(
                "parameter_state belongs to a different ModelComponents inventory"
            )
        self._components = components
        self._ownership_state = ownership_state
        self._parameter_state = parameter_state
        self._baseline = _tensor_inventory(components)
        self._applied_inventory: tuple[_TensorRecord, ...] | None = None
        self._prepared_inventory: tuple[_TensorRecord, ...] | None = None
        self._policy: ParameterDTypePolicy | None = None
        self._apply_attempted = False
        self._lock = RLock()
        self._require_pre_prepare_state()
        self._validate_parameter_state(parameter_state, self._baseline)

    @property
    def parameter_state(self) -> ParameterStateManager:
        return self._parameter_state

    @property
    def policy(self) -> ParameterDTypePolicy:
        if self._policy is None:
            raise ModelNumericsPolicyError("parameter dtype policy is not applied")
        return self._policy

    @property
    def policy_id(self) -> str:
        return self.policy.policy_id

    @property
    def application_id(self) -> str:
        if self._applied_inventory is None:
            raise ModelNumericsPolicyError("parameter dtype policy is not applied")
        return _identity(
            {
                "schema_version": _SCHEMA_VERSION,
                "policy_id": self.policy_id,
                "before_inventory_id": _inventory_id(self._baseline),
                "after_inventory_id": _inventory_id(self._applied_inventory),
                "state_projection_id": (
                    self._parameter_state.state_projection.projection_id
                ),
            }
        )

    def apply(self, policy: ParameterDTypePolicy) -> ParameterStateManager:
        """Apply ``policy`` exactly once and return the rebound topology owner."""

        if not isinstance(policy, ParameterDTypePolicy):
            raise TypeError("policy must be ParameterDTypePolicy")
        with self._lock:
            if self._apply_attempted:
                raise ModelNumericsPolicyError(
                    "parameter dtype policy owner permits exactly one apply attempt"
                )
            self._apply_attempted = True
            self._require_pre_prepare_state()
            current = _tensor_inventory(self._components)
            _require_same_inventory(
                self._baseline,
                current,
                context="before parameter dtype application",
            )
            self._validate_parameter_state(self._parameter_state, current)
            targets = self._target_dtypes(policy, current)
            changed: list[_TensorRecord] = []
            try:
                for record, target_dtype in targets:
                    if record.dtype_object == target_dtype:
                        continue
                    tensor = record.tensor
                    tensor.data = tensor.data.to(dtype=target_dtype)
                    changed.append(record)
                self._require_pre_prepare_state()
                applied = _tensor_inventory(self._components)
                _require_same_structure(current, applied)
                rebound = ParameterStateManager(self._components)
                self._validate_parameter_state(rebound, applied)
            except BaseException:
                for record in reversed(changed):
                    tensor = record.tensor
                    tensor.data = tensor.data.to(dtype=record.dtype_object)
                raise
            self._parameter_state = rebound
            self._applied_inventory = applied
            self._policy = policy
            return rebound

    def validate_applied(self) -> ParameterStateManager:
        """Reject unsanctioned topology, identity, requires-grad, or dtype drift."""

        with self._lock:
            if self._applied_inventory is None or self._policy is None:
                raise ModelNumericsPolicyError("parameter dtype policy is not applied")
            current = _tensor_inventory(self._components)
            state = self._current_ownership_state()
            if state is OwnershipState.PREPARED:
                if self._prepared_inventory is None:
                    _require_same_prepared_inventory(
                        self._applied_inventory,
                        current,
                        context="during distributed prepare",
                    )
                else:
                    _require_same_inventory(
                        self._prepared_inventory,
                        current,
                        context="after prepared inventory lock",
                    )
            else:
                _require_same_inventory(
                    self._applied_inventory,
                    current,
                    context="after parameter dtype application",
                )
            self._validate_parameter_state(self._parameter_state, current)
            if state is OwnershipState.PREPARED and self._prepared_inventory is None:
                self._prepared_inventory = current
            return self._parameter_state

    def _current_ownership_state(self) -> OwnershipState:
        try:
            state = OwnershipState(self._ownership_state())
        except (TypeError, ValueError):
            raise ModelNumericsPolicyError(
                "ownership_state returned an invalid lifecycle state"
            ) from None
        return state

    def _require_pre_prepare_state(self) -> None:
        state = self._current_ownership_state()
        if state not in {OwnershipState.LOADED, OwnershipState.CONFIGURED}:
            raise ModelNumericsPolicyError(
                "parameter dtype policy may only run before prepare in LOADED "
                f"or CONFIGURED state, found {state.value}"
            )

    @staticmethod
    def _validate_parameter_state(
        state: ParameterStateManager,
        inventory: tuple[_TensorRecord, ...],
    ) -> None:
        selected = state.named_trainable_parameters()
        selected_paths = tuple(item.name for item in selected)
        inventory_paths = tuple(
            record.path
            for record in inventory
            if record.kind == "parameter" and record.requires_grad
        )
        if set(selected_paths) != set(inventory_paths):
            raise ModelNumericsPolicyError(
                "ParameterStateManager trainable ownership disagrees with live inventory"
            )
        selected_ids = {item.name: id(item.parameter) for item in selected}
        inventory_ids = {
            record.path: record.object_id
            for record in inventory
            if record.kind == "parameter" and record.requires_grad
        }
        if selected_ids != inventory_ids:
            raise ModelNumericsPolicyError(
                "ParameterStateManager parameter objects disagree with live inventory"
            )

    @staticmethod
    def _target_dtypes(
        policy: ParameterDTypePolicy,
        inventory: tuple[_TensorRecord, ...],
    ) -> tuple[tuple[_TensorRecord, object], ...]:
        targets: list[tuple[_TensorRecord, object]] = []
        for record in inventory:
            tensor = record.tensor
            if record.kind == "parameter" and record.requires_grad:
                if not tensor.is_floating_point():
                    raise ModelNumericsPolicyError(
                        f"trainable parameter {record.path!r} is not floating point"
                    )
                targets.append((record, _torch_dtype(policy.trainable_parameter_dtype)))
                continue
            if not tensor.is_floating_point():
                continue
            if record.kind == "parameter":
                if (
                    policy.frozen_parameter_policy
                    is FrozenParameterPolicy.EXPLICIT_DTYPE
                ):
                    assert policy.frozen_parameter_dtype is not None
                    targets.append(
                        (record, _torch_dtype(policy.frozen_parameter_dtype))
                    )
            elif policy.floating_buffer_policy is FloatingBufferPolicy.EXPLICIT_DTYPE:
                assert policy.floating_buffer_dtype is not None
                targets.append((record, _torch_dtype(policy.floating_buffer_dtype)))
        if not any(
            record.kind == "parameter" and record.requires_grad
            for record, _dtype in targets
        ):
            raise ModelNumericsPolicyError(
                "parameter dtype policy found no trainable floating parameters"
            )
        return tuple(targets)


def _tensor_inventory(components: ModelComponents) -> tuple[_TensorRecord, ...]:
    import torch

    records: list[_TensorRecord] = []
    keys: set[tuple[str, str]] = set()
    tensor_owners: dict[int, str] = {}
    for binding in components.bindings:
        named_parameters = getattr(binding.component, "named_parameters", None)
        if callable(named_parameters):
            for name, parameter in tuple(named_parameters()):
                if not isinstance(name, str) or not name:
                    raise ModelNumericsPolicyError(
                        f"component {binding.name!r} returned an invalid parameter name"
                    )
                if not isinstance(parameter, torch.nn.Parameter):
                    raise ModelNumericsPolicyError(
                        f"{binding.name}.{name} is not torch.nn.Parameter"
                    )
                record = _record(
                    kind="parameter",
                    path=f"{binding.name}.{name}",
                    tensor=parameter,
                    requires_grad=bool(parameter.requires_grad),
                )
                key = (record.kind, record.path)
                if key in keys:
                    raise ModelNumericsPolicyError(
                        f"duplicate tensor ownership path: {record.path}"
                    )
                prior_owner = tensor_owners.get(record.object_id)
                if prior_owner is not None:
                    raise ModelNumericsPolicyError(
                        "one tensor object has multiple component ownership paths: "
                        f"{prior_owner!r}, {record.path!r}"
                    )
                keys.add(key)
                tensor_owners[record.object_id] = record.path
                records.append(record)
        named_buffers = getattr(binding.component, "named_buffers", None)
        if callable(named_buffers):
            for name, buffer in tuple(named_buffers()):
                if not isinstance(name, str) or not name:
                    raise ModelNumericsPolicyError(
                        f"component {binding.name!r} returned an invalid buffer name"
                    )
                if not isinstance(buffer, torch.Tensor):
                    raise ModelNumericsPolicyError(
                        f"{binding.name}.{name} is not torch.Tensor"
                    )
                record = _record(
                    kind="buffer",
                    path=f"{binding.name}.{name}",
                    tensor=buffer,
                    requires_grad=None,
                )
                key = (record.kind, record.path)
                if key in keys:
                    raise ModelNumericsPolicyError(
                        f"duplicate tensor ownership path: {record.path}"
                    )
                prior_owner = tensor_owners.get(record.object_id)
                if prior_owner is not None:
                    raise ModelNumericsPolicyError(
                        "one tensor object has multiple component ownership paths: "
                        f"{prior_owner!r}, {record.path!r}"
                    )
                keys.add(key)
                tensor_owners[record.object_id] = record.path
                records.append(record)
    return tuple(sorted(records, key=lambda item: (item.kind, item.path)))


def _record(
    *,
    kind: str,
    path: str,
    tensor: object,
    requires_grad: bool | None,
) -> _TensorRecord:
    return _TensorRecord(
        kind=kind,
        path=path,
        tensor=tensor,
        object_id=id(tensor),
        shape=tuple(int(item) for item in tensor.shape),
        dtype=str(tensor.dtype),
        dtype_object=tensor.dtype,
        requires_grad=requires_grad,
    )


def _inventory_id(inventory: tuple[_TensorRecord, ...]) -> str:
    return _identity(
        {
            "schema_version": _SCHEMA_VERSION,
            "tensors": [record.semantic_payload() for record in inventory],
        }
    )


def _require_same_structure(
    before: tuple[_TensorRecord, ...],
    after: tuple[_TensorRecord, ...],
) -> None:
    before_structure = tuple(
        (item.kind, item.path, item.object_id, item.shape, item.requires_grad)
        for item in before
    )
    after_structure = tuple(
        (item.kind, item.path, item.object_id, item.shape, item.requires_grad)
        for item in after
    )
    if before_structure != after_structure:
        raise ModelNumericsPolicyError(
            "live tensor structure changed during parameter dtype application"
        )


def _require_same_inventory(
    expected: tuple[_TensorRecord, ...],
    actual: tuple[_TensorRecord, ...],
    *,
    context: str,
) -> None:
    expected_values = tuple(
        (
            item.kind,
            item.path,
            item.object_id,
            item.shape,
            item.dtype,
            item.requires_grad,
        )
        for item in expected
    )
    actual_values = tuple(
        (
            item.kind,
            item.path,
            item.object_id,
            item.shape,
            item.dtype,
            item.requires_grad,
        )
        for item in actual
    )
    if expected_values != actual_values:
        raise ModelNumericsPolicyError(
            f"live tensor topology or dtype drifted {context}"
        )


def _require_same_prepared_inventory(
    expected: tuple[_TensorRecord, ...],
    actual: tuple[_TensorRecord, ...],
    *,
    context: str,
) -> None:
    """Validate post-prepare tensors without treating buffer ids as topology.

    ``torch.nn.Module._apply`` preserves ``Parameter`` objects so optimizers
    keep their ownership, but replaces registered buffers during device
    placement.  The replacement is valid only when every buffer's semantic
    contract remains unchanged.
    """

    def comparison_value(item: _TensorRecord) -> tuple[object, ...]:
        return (
            item.kind,
            item.path,
            item.object_id if item.kind == "parameter" else None,
            item.shape,
            item.dtype,
            item.requires_grad,
        )

    if tuple(map(comparison_value, expected)) != tuple(map(comparison_value, actual)):
        raise ModelNumericsPolicyError(
            f"live tensor topology or dtype drifted {context}"
        )
