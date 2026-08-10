"""Content-addressed reference-policy ownership evidence for checkpoints.

Reference capability is a prepared model fact; active ownership is an
algorithm fact.  This module binds both without component-name or recipe-name
branches and preserves enough parameter-view evidence for checkpoint code to
distinguish derived, independently owned, and inapplicable state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Final

from visual_rl.algorithms.trainer.execution_plan import AlgorithmExecutionPlan
from visual_rl.core.contracts import ModelContract
from visual_rl.models.lifecycle.components import ExecutionMode
from visual_rl.models.numerics.execution import ParameterView
from visual_rl.models.numerics.policy import (
    ModelExecutionNumericsEvidence,
    ModelNumericsPolicyError,
    ParameterViewEvidence,
    ParameterViewMode,
)

__all__ = (
    "DERIVED_REFERENCE_STATE_SCHEMA",
    "INDEPENDENT_REFERENCE_STATE_SCHEMA",
    "NO_REFERENCE_STATE_SCHEMA",
    "ReferencePolicyStateError",
    "ReferencePolicyStateEvidence",
    "derive_reference_policy_state_evidence",
)


DERIVED_REFERENCE_STATE_SCHEMA: Final = "derived-from-model-artifact.v1"
INDEPENDENT_REFERENCE_STATE_SCHEMA: Final = "independent-reference-state.v1"
NO_REFERENCE_STATE_SCHEMA: Final = "none.v1"
_SCHEMA_VERSION = 1


class ReferencePolicyStateError(ValueError):
    """Algorithm, model contract, and prepared parameter views disagree."""


@dataclass(frozen=True, slots=True)
class ReferencePolicyStateEvidence:
    """Immutable proof of effective reference capability and checkpoint ownership."""

    mode: ParameterViewMode | None
    owner_component_names: tuple[str, ...]
    restorable_state_names: tuple[str, ...]
    state_schema: str
    source_projection_id: str
    parameter_view_evidence_id: str | None
    model_execution_numerics_id: str
    algorithm_plan_id: str
    algorithm_requires_reference_statistics: bool
    model_provides_reference_policy: bool
    schema_version: int = _SCHEMA_VERSION
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ReferencePolicyStateError(
                "reference policy state evidence schema_version must be 1"
            )
        for name in (
            "source_projection_id",
            "model_execution_numerics_id",
            "algorithm_plan_id",
        ):
            _digest(name, getattr(self, name))
        if type(self.algorithm_requires_reference_statistics) is not bool:
            raise TypeError("algorithm_requires_reference_statistics must be bool")
        if type(self.model_provides_reference_policy) is not bool:
            raise TypeError("model_provides_reference_policy must be bool")
        mode = None if self.mode is None else _mode(self.mode)
        owners = _canonical_names(
            self.owner_component_names,
            field_name="owner_component_names",
        )
        state_names = _canonical_names(
            self.restorable_state_names,
            field_name="restorable_state_names",
        )
        state_schema = _text("state_schema", self.state_schema)

        if self.model_provides_reference_policy:
            if mode is None:
                raise ReferencePolicyStateError(
                    "model reference capability requires a concrete parameter-view mode"
                )
            if not owners or not state_names:
                raise ReferencePolicyStateError(
                    "model reference capability requires non-empty owners and "
                    "restorable state names"
                )
            if self.parameter_view_evidence_id is None:
                raise ReferencePolicyStateError(
                    "model reference capability requires parameter-view evidence"
                )
            _digest(
                "parameter_view_evidence_id",
                self.parameter_view_evidence_id,
            )
            if state_schema != _state_schema_for_mode(mode):
                raise ReferencePolicyStateError(
                    "reference state schema disagrees with parameter-view mode"
                )
        elif (
            mode is not None
            or owners
            or state_names
            or self.parameter_view_evidence_id is not None
            or state_schema != NO_REFERENCE_STATE_SCHEMA
        ):
            raise ReferencePolicyStateError(
                "model without reference capability cannot carry reference state "
                "evidence"
            )
        if (
            self.algorithm_requires_reference_statistics
            and not self.model_provides_reference_policy
        ):
            raise ReferencePolicyStateError(
                "algorithm requires reference statistics but the model provides no "
                "reference policy"
            )

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "owner_component_names", owners)
        object.__setattr__(self, "restorable_state_names", state_names)
        object.__setattr__(self, "state_schema", state_schema)
        object.__setattr__(self, "evidence_id", _identity(self._identity_payload()))

    @property
    def has_reference_capability(self) -> bool:
        return self.model_provides_reference_policy

    @property
    def has_active_reference_owner(self) -> bool:
        """Whether this algorithm must reconstruct reference state on restore."""

        return self.algorithm_requires_reference_statistics

    @property
    def checkpoint_state_schema(self) -> str:
        """Project capability state into the schema owned by this checkpoint."""

        return (
            self.state_schema
            if self.has_active_reference_owner
            else NO_REFERENCE_STATE_SCHEMA
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": None if self.mode is None else self.mode.value,
            "owner_component_names": list(self.owner_component_names),
            "restorable_state_names": list(self.restorable_state_names),
            "state_schema": self.state_schema,
            "source_projection_id": self.source_projection_id,
            "parameter_view_evidence_id": self.parameter_view_evidence_id,
            "model_execution_numerics_id": self.model_execution_numerics_id,
            "algorithm_plan_id": self.algorithm_plan_id,
            "algorithm_requires_reference_statistics": (
                self.algorithm_requires_reference_statistics
            ),
            "model_provides_reference_policy": (self.model_provides_reference_policy),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._identity_payload(), "evidence_id": self.evidence_id}

    def assert_integrity(self) -> None:
        if self.evidence_id != _identity(self._identity_payload()):
            raise ReferencePolicyStateError(
                "reference policy state evidence identity mismatch"
            )


def derive_reference_policy_state_evidence(
    *,
    algorithm: AlgorithmExecutionPlan,
    model: ModelContract,
    model_execution_numerics: ModelExecutionNumericsEvidence,
) -> ReferencePolicyStateEvidence:
    """Derive prepared reference ownership from typed, content-addressed sources."""

    if not isinstance(algorithm, AlgorithmExecutionPlan):
        raise TypeError("algorithm must be AlgorithmExecutionPlan")
    if not isinstance(model, ModelContract):
        raise TypeError("model must be ModelContract")
    if not isinstance(model_execution_numerics, ModelExecutionNumericsEvidence):
        raise TypeError(
            "model_execution_numerics must be ModelExecutionNumericsEvidence"
        )
    _validate_numerics_integrity(model_execution_numerics)
    provides_reference = model.provides_reference_policy
    if type(provides_reference) is not bool:
        raise ReferencePolicyStateError(
            "checkpoint reference ownership requires a resolved boolean model contract"
        )

    current = _exact_view(model_execution_numerics, ParameterView.CURRENT)
    reference_views = tuple(
        item
        for item in model_execution_numerics.parameter_view_evidence
        if item.parameter_view is ParameterView.REFERENCE
    )
    if provides_reference:
        if len(reference_views) != 1:
            raise ReferencePolicyStateError(
                "model declares reference capability but prepared numerics do not "
                "contain exactly one reference parameter view"
            )
        reference = reference_views[0]
        _validate_reference_view(
            reference,
            current=current,
            numerics=model_execution_numerics,
        )
        return ReferencePolicyStateEvidence(
            mode=reference.mode,
            owner_component_names=reference.owner_component_names,
            restorable_state_names=reference.restorable_state_names,
            state_schema=_state_schema_for_mode(reference.mode),
            source_projection_id=reference.source_projection_id,
            parameter_view_evidence_id=reference.evidence_id,
            model_execution_numerics_id=(
                model_execution_numerics.execution_numerics_id
            ),
            algorithm_plan_id=algorithm.plan_id,
            algorithm_requires_reference_statistics=(
                algorithm.requires_reference_statistics
            ),
            model_provides_reference_policy=True,
        )

    if reference_views:
        raise ReferencePolicyStateError(
            "model declares no reference capability but prepared numerics contain a "
            "reference parameter view"
        )
    return ReferencePolicyStateEvidence(
        mode=None,
        owner_component_names=(),
        restorable_state_names=(),
        state_schema=NO_REFERENCE_STATE_SCHEMA,
        source_projection_id=current.source_projection_id,
        parameter_view_evidence_id=None,
        model_execution_numerics_id=model_execution_numerics.execution_numerics_id,
        algorithm_plan_id=algorithm.plan_id,
        algorithm_requires_reference_statistics=(
            algorithm.requires_reference_statistics
        ),
        model_provides_reference_policy=False,
    )


def _validate_numerics_integrity(
    numerics: ModelExecutionNumericsEvidence,
) -> None:
    try:
        ModelExecutionNumericsEvidence.from_payload(numerics.to_payload())
    except (ModelNumericsPolicyError, TypeError, ValueError) as exc:
        raise ReferencePolicyStateError(
            "model execution numerics evidence failed content-address integrity"
        ) from exc


def _exact_view(
    numerics: ModelExecutionNumericsEvidence,
    parameter_view: ParameterView,
) -> ParameterViewEvidence:
    try:
        evidence = numerics.view_evidence(parameter_view)
        evidence.assert_integrity()
    except (ModelNumericsPolicyError, TypeError, ValueError) as exc:
        raise ReferencePolicyStateError(
            f"prepared numerics lack exact {parameter_view.value!r} view evidence"
        ) from exc
    return evidence


def _validate_reference_view(
    reference: ParameterViewEvidence,
    *,
    current: ParameterViewEvidence,
    numerics: ModelExecutionNumericsEvidence,
) -> None:
    try:
        reference.assert_matches(
            parameter_view=ParameterView.REFERENCE,
            source_projection_id=current.source_projection_id,
        )
        numerics.autocast_policy(ExecutionMode.TRAIN, ParameterView.REFERENCE)
    except (ModelNumericsPolicyError, TypeError, ValueError) as exc:
        raise ReferencePolicyStateError(
            "reference view is stale, corrupt, or lacks a train-forward policy"
        ) from exc
    if reference.mode is ParameterViewMode.LORA_DISABLE and (
        reference.owner_component_names != current.owner_component_names
        or reference.restorable_state_names != current.restorable_state_names
    ):
        raise ReferencePolicyStateError(
            "LoRA-disable reference state must derive from the exact current model "
            "artifact owners and restorable state names"
        )


def _state_schema_for_mode(mode: ParameterViewMode) -> str:
    resolved = _mode(mode)
    if resolved is ParameterViewMode.LORA_DISABLE:
        return DERIVED_REFERENCE_STATE_SCHEMA
    if resolved in {ParameterViewMode.FROZEN_COPY, ParameterViewMode.IN_PLACE_SWAP}:
        return INDEPENDENT_REFERENCE_STATE_SCHEMA
    raise ReferencePolicyStateError(
        f"unsupported reference parameter-view mode: {resolved.value!r}"
    )


def _mode(value: object) -> ParameterViewMode:
    try:
        resolved = ParameterViewMode(value)
    except (TypeError, ValueError):
        raise ReferencePolicyStateError(
            f"invalid reference parameter-view mode: {value!r}"
        ) from None
    if resolved not in {
        ParameterViewMode.LORA_DISABLE,
        ParameterViewMode.FROZEN_COPY,
        ParameterViewMode.IN_PLACE_SWAP,
    }:
        raise ReferencePolicyStateError(
            f"invalid reference parameter-view mode: {resolved.value!r}"
        )
    return resolved


def _canonical_names(value: object, *, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple or any(
        not isinstance(item, str) or not item or item.strip() != item for item in value
    ):
        raise ReferencePolicyStateError(
            f"{field_name} must be a tuple of canonical strings"
        )
    if value != tuple(sorted(set(value))):
        raise ReferencePolicyStateError(f"{field_name} must be sorted and unique")
    return value


def _digest(field_name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReferencePolicyStateError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ReferencePolicyStateError(
            f"{field_name} must be a non-empty canonical string"
        )
    return value


def _identity(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
