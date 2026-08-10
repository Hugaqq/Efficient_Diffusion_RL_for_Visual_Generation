"""Canonical stage execution semantics independent of resource residency."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from visual_rl.models.lifecycle.components import ComponentRole, ExecutionMode

__all__ = (
    "GradPolicy",
    "ModuleExecutionMode",
    "ParameterView",
    "StageExecutionPolicy",
    "StageExecutionPolicyError",
)


class StageExecutionPolicyError(ValueError):
    """Stage execution semantics are incomplete or internally inconsistent."""


class ModuleExecutionMode(str, Enum):
    TRAIN = "train"
    EVAL = "eval"


class GradPolicy(str, Enum):
    ENABLED = "enabled"
    NO_GRAD = "no_grad"

    @property
    def enabled(self) -> bool:
        return self is GradPolicy.ENABLED


class ParameterView(str, Enum):
    CURRENT = "current"
    REFERENCE = "reference"
    EMA = "ema"


@dataclass(frozen=True, slots=True)
class StageExecutionPolicy:
    """Immutable execution policy for one stage and one parameter view.

    ``ResourcePlan`` remains the sole owner of placement.  This object only
    declares module train/eval state, autograd state, and the logical parameter
    view.  The initial v0.8 policies are intentionally closed and canonical so
    two semantically identical policies cannot acquire different identities.
    """

    stage: ExecutionMode
    component_mode_by_role: tuple[tuple[ComponentRole, ModuleExecutionMode], ...]
    grad_policy: GradPolicy
    parameter_view: ParameterView
    transform_plan_id: str = "identity"

    def __post_init__(self) -> None:
        try:
            stage = ExecutionMode(self.stage)
        except (TypeError, ValueError):
            raise StageExecutionPolicyError(
                f"invalid execution stage: {self.stage!r}"
            ) from None
        if stage is ExecutionMode.IDLE:
            raise StageExecutionPolicyError("IDLE has no execution policy")
        try:
            grad_policy = GradPolicy(self.grad_policy)
        except (TypeError, ValueError):
            raise StageExecutionPolicyError(
                f"invalid grad policy: {self.grad_policy!r}"
            ) from None
        try:
            parameter_view = ParameterView(self.parameter_view)
        except (TypeError, ValueError):
            raise StageExecutionPolicyError(
                f"invalid parameter view: {self.parameter_view!r}"
            ) from None
        if not isinstance(self.transform_plan_id, str) or not self.transform_plan_id:
            raise StageExecutionPolicyError(
                "transform_plan_id must be a non-empty string"
            )
        if type(self.component_mode_by_role) is not tuple:
            raise TypeError("component_mode_by_role must be a tuple")

        resolved: list[tuple[ComponentRole, ModuleExecutionMode]] = []
        for item in self.component_mode_by_role:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError(
                    "component_mode_by_role entries must be (role, mode) tuples"
                )
            role_value, mode_value = item
            try:
                role = ComponentRole(role_value)
            except (TypeError, ValueError):
                raise StageExecutionPolicyError(
                    f"invalid component role: {role_value!r}"
                ) from None
            try:
                mode = ModuleExecutionMode(mode_value)
            except (TypeError, ValueError):
                raise StageExecutionPolicyError(
                    f"invalid module execution mode: {mode_value!r}"
                ) from None
            resolved.append((role, mode))
        if not resolved:
            raise StageExecutionPolicyError("component_mode_by_role must not be empty")
        roles = tuple(role for role, _mode in resolved)
        if len(roles) != len(set(roles)):
            raise StageExecutionPolicyError(
                "component_mode_by_role must define each role once"
            )

        canonical = tuple(sorted(resolved, key=lambda item: item[0].value))
        expected_modes, expected_grad = _canonical_semantics(
            stage,
            parameter_view,
        )
        if canonical != expected_modes or grad_policy is not expected_grad:
            raise StageExecutionPolicyError(
                "stage execution policy does not match the canonical v0.8 "
                f"semantics for {stage.value}/{parameter_view.value}"
            )

        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "component_mode_by_role", canonical)
        object.__setattr__(self, "grad_policy", grad_policy)
        object.__setattr__(self, "parameter_view", parameter_view)

    @classmethod
    def canonical(
        cls,
        stage: ExecutionMode,
        *,
        parameter_view: ParameterView = ParameterView.CURRENT,
        transform_plan_id: str = "identity",
    ) -> StageExecutionPolicy:
        try:
            resolved_stage = ExecutionMode(stage)
        except (TypeError, ValueError):
            raise StageExecutionPolicyError(
                f"invalid execution stage: {stage!r}"
            ) from None
        try:
            resolved_view = ParameterView(parameter_view)
        except (TypeError, ValueError):
            raise StageExecutionPolicyError(
                f"invalid parameter view: {parameter_view!r}"
            ) from None
        modes, grad_policy = _canonical_semantics(resolved_stage, resolved_view)
        return cls(
            stage=resolved_stage,
            component_mode_by_role=modes,
            grad_policy=grad_policy,
            parameter_view=resolved_view,
            transform_plan_id=transform_plan_id,
        )

    @property
    def grad_enabled(self) -> bool:
        return self.grad_policy.enabled

    def mode_for_roles(
        self,
        roles: tuple[ComponentRole, ...],
    ) -> ModuleExecutionMode | None:
        configured = dict(self.component_mode_by_role)
        matches = {configured[role] for role in roles if role in configured}
        if len(matches) > 1:
            raise StageExecutionPolicyError(
                "one component resolves to conflicting train/eval modes"
            )
        return next(iter(matches), None)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "stage": self.stage.value,
            "component_mode_by_role": [
                {"role": role.value, "mode": mode.value}
                for role, mode in self.component_mode_by_role
            ],
            "grad_policy": self.grad_policy.value,
            "parameter_view": self.parameter_view.value,
            "transform_plan_id": self.transform_plan_id,
        }

    @property
    def execution_policy_id(self) -> str:
        payload = json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _canonical_semantics(
    stage: ExecutionMode,
    parameter_view: ParameterView,
) -> tuple[
    tuple[tuple[ComponentRole, ModuleExecutionMode], ...],
    GradPolicy,
]:
    if stage is ExecutionMode.PREPROCESS and parameter_view is ParameterView.CURRENT:
        modes = ((ComponentRole.PREPROCESS, ModuleExecutionMode.EVAL),)
        return tuple(sorted(modes, key=lambda item: item[0].value)), GradPolicy.NO_GRAD
    if stage is ExecutionMode.ROLLOUT and parameter_view is ParameterView.CURRENT:
        modes = (
            (ComponentRole.INFERENCE, ModuleExecutionMode.EVAL),
            (ComponentRole.DECODER, ModuleExecutionMode.EVAL),
        )
        return tuple(sorted(modes, key=lambda item: item[0].value)), GradPolicy.NO_GRAD
    if stage is ExecutionMode.TRAIN and parameter_view is ParameterView.CURRENT:
        modes = (
            # Policy-ratio recompute must match rollout module semantics exactly.
            # EVAL disables dropout/stateful training behavior without disabling
            # autograd; LoRA/current-policy gradients still flow below.
            (ComponentRole.INFERENCE, ModuleExecutionMode.EVAL),
            (ComponentRole.TRAINABLE, ModuleExecutionMode.EVAL),
            (ComponentRole.REFERENCE, ModuleExecutionMode.EVAL),
        )
        return tuple(sorted(modes, key=lambda item: item[0].value)), GradPolicy.ENABLED
    if stage is ExecutionMode.TRAIN and parameter_view is ParameterView.REFERENCE:
        modes = (
            (ComponentRole.INFERENCE, ModuleExecutionMode.EVAL),
            (ComponentRole.TRAINABLE, ModuleExecutionMode.EVAL),
            (ComponentRole.REFERENCE, ModuleExecutionMode.EVAL),
        )
        return tuple(sorted(modes, key=lambda item: item[0].value)), GradPolicy.NO_GRAD
    if stage is ExecutionMode.EVAL and parameter_view is ParameterView.CURRENT:
        modes = tuple(
            (role, ModuleExecutionMode.EVAL)
            for role in (
                ComponentRole.PREPROCESS,
                ComponentRole.INFERENCE,
                ComponentRole.TRAINABLE,
                ComponentRole.REFERENCE,
                ComponentRole.DECODER,
            )
        )
        return tuple(sorted(modes, key=lambda item: item[0].value)), GradPolicy.NO_GRAD
    raise StageExecutionPolicyError(
        "no canonical v0.8 execution policy exists for "
        f"{stage.value}/{parameter_view.value}"
    )
