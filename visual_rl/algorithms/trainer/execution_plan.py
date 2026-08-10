"""Spec-only immutable execution plan for the six-stage trainer graph.

This module never reads recipes, registry objects, component graphs, or live
runtime resources. ``AlgorithmExecutionPlan.from_spec`` is the sole production
construction path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from visual_rl.core.contracts import (
    AlgorithmComponentRole,
    AlgorithmMaterializationSpec,
    ExecutionPolicyReceipt,
    GroupingKind,
    LikelihoodSemantics,
    ReferenceRequirement,
    ReplayTarget,
    TrainingParadigm,
    TrajectoryKind,
    TransitionSelectionKind,
    TransitionSelectionSpec,
)
from visual_rl.core.serialization import to_plain_dict

__all__ = (
    "AlgorithmExecutionPlan",
    "AlgorithmPlanError",
    "PolicyMicrobatchCardinality",
    "ReplayTarget",
    "StageTraceEntry",
    "TransitionSelectionKind",
    "TransitionSelectionSpec",
    "UpdateCardinality",
)


class AlgorithmPlanError(ValueError):
    """A materialization spec cannot form the canonical trainer plan."""


@dataclass(frozen=True, slots=True)
class PolicyMicrobatchCardinality:
    """Exact cardinality of one complete group-scoped policy microbatch."""

    scope: str
    row_count: int
    stored_transition_slots: int
    physical_transition_slots: int

    def __post_init__(self) -> None:
        if self.scope != "one_complete_group":
            raise ValueError("policy microbatch scope must be one_complete_group")
        for name in (
            "row_count",
            "stored_transition_slots",
            "physical_transition_slots",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stored_transition_slots > self.physical_transition_slots:
            raise ValueError(
                "stored transition slots cannot exceed physical transition slots"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "row_count": self.row_count,
            "stored_transition_slots": self.stored_transition_slots,
            "physical_transition_slots": self.physical_transition_slots,
        }


@dataclass(frozen=True, slots=True)
class UpdateCardinality:
    """Optimizer commits performed by one logical trainer iteration."""

    inner_epochs: int
    optimizer_updates_per_iteration: int

    def __post_init__(self) -> None:
        for name in ("inner_epochs", "optimizer_updates_per_iteration"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_payload(self) -> dict[str, int]:
        return {
            "inner_epochs": self.inner_epochs,
            "optimizer_updates_per_iteration": self.optimizer_updates_per_iteration,
        }


@dataclass(frozen=True, slots=True)
class StageTraceEntry:
    """One typed handoff in the canonical Trainer stage graph."""

    stage: str
    input_type: str
    output_type: str

    def __post_init__(self) -> None:
        _canonical_text(self.stage, field_name="stage")
        _class_path(self.input_type, field_name="stage input_type")
        _class_path(self.output_type, field_name="stage output_type")

    def to_payload(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "input_type": self.input_type,
            "output_type": self.output_type,
        }


@dataclass(frozen=True, slots=True)
class AlgorithmExecutionPlan:
    """Complete immutable semantics consumed by the six-stage trainer."""

    trainer_family: str
    trainer_implementation_identity: str
    training_paradigm: TrainingParadigm
    objective_identity: str
    trajectory_kind: TrajectoryKind
    grouping: GroupingKind
    schedule_step_count: int
    physical_transition_count: int
    stored_policy_transition_count: int
    transition_selection: TransitionSelectionSpec
    replay_target: ReplayTarget
    likelihood_semantics: LikelihoodSemantics
    credit_family: str
    credit_assignment_identity: str
    reference_requirement: ReferenceRequirement
    requires_reference_statistics: bool
    beta: float
    policy_microbatch_cardinality: PolicyMicrobatchCardinality
    update_cardinality: UpdateCardinality
    required_trajectory_fields: tuple[str, ...]
    required_policy_fields: tuple[str, ...]
    stage_trace_schema: tuple[StageTraceEntry, ...]
    _plan_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "trainer_family",
            "trainer_implementation_identity",
            "objective_identity",
            "credit_family",
            "credit_assignment_identity",
        ):
            _canonical_text(getattr(self, name), field_name=name)
        for name, expected_type in (
            ("training_paradigm", TrainingParadigm),
            ("trajectory_kind", TrajectoryKind),
            ("grouping", GroupingKind),
            ("replay_target", ReplayTarget),
            ("likelihood_semantics", LikelihoodSemantics),
            ("reference_requirement", ReferenceRequirement),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")
        for name in (
            "schedule_step_count",
            "physical_transition_count",
            "stored_policy_transition_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stored_policy_transition_count > self.physical_transition_count:
            raise ValueError(
                "stored policy transitions cannot exceed physical transitions"
            )
        if self.trajectory_kind is TrajectoryKind.FULL and not (
            self.schedule_step_count
            == self.physical_transition_count
            == self.stored_policy_transition_count
        ):
            raise ValueError(
                "full trajectory schedule/physical/stored counts must match"
            )
        if self.trajectory_kind is TrajectoryKind.SINGLE_STEP and not (
            self.physical_transition_count == self.schedule_step_count
            and self.stored_policy_transition_count == 1
        ):
            raise ValueError(
                "single-step physical count must match schedule and store one action"
            )
        if not isinstance(self.transition_selection, TransitionSelectionSpec):
            raise TypeError("transition_selection must be TransitionSelectionSpec")
        if (
            self.transition_selection.selected_transition_count
            != self.stored_policy_transition_count
        ):
            raise ValueError(
                "transition selection count must equal stored policy transitions"
            )
        allowed_selection = {
            TrajectoryKind.FULL: (TransitionSelectionKind.ALL,),
            TrajectoryKind.BRANCHING: (
                TransitionSelectionKind.ALL,
                TransitionSelectionKind.BRANCH_STEP,
            ),
            TrajectoryKind.SINGLE_STEP: (TransitionSelectionKind.SELECTED_TIMESTEP,),
        }[self.trajectory_kind]
        if self.transition_selection.kind not in allowed_selection:
            raise ValueError("transition selection differs from trajectory kind")
        if (
            self.transition_selection.fixed_index is not None
            and self.transition_selection.fixed_index >= self.schedule_step_count
        ):
            raise ValueError("transition selection fixed_index exceeds schedule")
        expected_grouping = {
            TrajectoryKind.FULL: GroupingKind.PROMPT_COMPLETIONS,
            TrajectoryKind.BRANCHING: GroupingKind.BRANCHES,
            TrajectoryKind.SINGLE_STEP: GroupingKind.SELECTED_TIMESTEP,
        }[self.trajectory_kind]
        if self.grouping is not expected_grouping:
            raise ValueError("grouping differs from trajectory kind")
        expected_replay_target = {
            LikelihoodSemantics.EXACT_ENV_ACTION: ReplayTarget.SAMPLED_ACTION,
            LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE: (
                ReplayTarget.CONDITIONED_NEXT
            ),
        }[self.likelihood_semantics]
        if self.replay_target is not expected_replay_target:
            raise ValueError("replay target differs from likelihood semantics")
        if type(self.requires_reference_statistics) is not bool:
            raise TypeError("requires_reference_statistics must be bool")
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, (int, float))
            or not math.isfinite(float(self.beta))
            or float(self.beta) < 0.0
        ):
            raise ValueError("beta must be finite and non-negative")
        object.__setattr__(self, "beta", float(self.beta))
        expected_reference = (
            self.reference_requirement is ReferenceRequirement.ALWAYS
            or (
                self.reference_requirement is ReferenceRequirement.WHEN_BETA_POSITIVE
                and self.beta > 0.0
            )
        )
        if self.requires_reference_statistics is not expected_reference:
            raise ValueError("reference-statistics flag differs from requirement/beta")
        if self.reference_requirement is ReferenceRequirement.NEVER and self.beta > 0.0:
            raise ValueError("positive beta conflicts with NEVER reference requirement")
        if not isinstance(
            self.policy_microbatch_cardinality,
            PolicyMicrobatchCardinality,
        ):
            raise TypeError(
                "policy_microbatch_cardinality must be PolicyMicrobatchCardinality"
            )
        cardinality = self.policy_microbatch_cardinality
        if (
            cardinality.stored_transition_slots
            != cardinality.row_count * self.stored_policy_transition_count
        ):
            raise ValueError(
                "policy microbatch stored slots differ from row/transition counts"
            )
        if (
            cardinality.physical_transition_slots
            != cardinality.row_count * self.physical_transition_count
        ):
            raise ValueError(
                "policy microbatch physical slots differ from row/transition counts"
            )
        if not isinstance(self.update_cardinality, UpdateCardinality):
            raise TypeError("update_cardinality must be UpdateCardinality")
        _canonical_string_tuple(
            self.required_trajectory_fields,
            field_name="required_trajectory_fields",
        )
        _canonical_string_tuple(
            self.required_policy_fields,
            field_name="required_policy_fields",
        )
        if type(self.stage_trace_schema) is not tuple or any(
            not isinstance(item, StageTraceEntry) for item in self.stage_trace_schema
        ):
            raise TypeError("stage_trace_schema must contain StageTraceEntry values")
        if tuple(item.stage for item in self.stage_trace_schema) != (
            "prelude",
            "rollout",
            "reward",
            "advantage",
            "credit",
            "optimize",
        ):
            raise ValueError("stage_trace_schema must use the canonical stage order")
        object.__setattr__(self, "_plan_id", _sha256_payload(self.to_payload()))

    @classmethod
    def from_spec(
        cls,
        spec: AlgorithmMaterializationSpec,
        *,
        execution_policy: ExecutionPolicyReceipt,
    ) -> AlgorithmExecutionPlan:
        """Bind a spec only after rebuilding its complete policy receipt."""

        if not isinstance(spec, AlgorithmMaterializationSpec):
            raise TypeError("spec must be an AlgorithmMaterializationSpec")
        if type(execution_policy) is not ExecutionPolicyReceipt:
            raise TypeError("execution_policy must be an ExecutionPolicyReceipt")
        try:
            policy = execution_policy.validated_projection(spec.execution_policy_id)
        except (TypeError, ValueError) as exc:
            raise AlgorithmPlanError(
                "execution policy identity/projection differs from the "
                "materialization spec"
            ) from exc
        by_role = {item.role: item for item in spec.components}
        try:
            trainer = by_role[AlgorithmComponentRole.TRAINER]
            credit = by_role[AlgorithmComponentRole.CREDIT]
        except KeyError as exc:  # protected by the core contract, kept fail-closed
            raise AlgorithmPlanError(
                f"materialization spec is missing {exc.args[0].value}"
            ) from exc
        return cls(
            trainer_family=trainer.selected_component_id,
            trainer_implementation_identity=trainer.component_declaration_id,
            training_paradigm=policy.transform_plan.paradigm,
            objective_identity=spec.objective_identity,
            trajectory_kind=spec.trajectory_kind,
            grouping=spec.grouping,
            schedule_step_count=spec.schedule_step_count,
            physical_transition_count=spec.physical_transition_count,
            stored_policy_transition_count=spec.stored_policy_transition_count,
            transition_selection=spec.transition_selection,
            replay_target=spec.replay_target,
            likelihood_semantics=spec.likelihood_semantics,
            credit_family=credit.selected_component_id,
            credit_assignment_identity=credit.component_declaration_id,
            reference_requirement=spec.reference_requirement,
            requires_reference_statistics=spec.requires_reference_statistics,
            beta=spec.beta,
            policy_microbatch_cardinality=PolicyMicrobatchCardinality(
                scope="one_complete_group",
                row_count=policy.group_size,
                stored_transition_slots=(
                    policy.group_size * spec.stored_policy_transition_count
                ),
                physical_transition_slots=(
                    policy.group_size * spec.physical_transition_count
                ),
            ),
            update_cardinality=UpdateCardinality(
                inner_epochs=spec.inner_epochs,
                optimizer_updates_per_iteration=(spec.optimizer_updates_per_iteration),
            ),
            required_trajectory_fields=spec.required_trajectory_fields,
            required_policy_fields=spec.required_policy_fields,
            stage_trace_schema=_CANONICAL_STAGE_TRACE_SCHEMA,
        )

    @property
    def plan_id(self) -> str:
        return self._plan_id

    @property
    def fingerprint(self) -> str:
        return self._plan_id

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trainer_family": self.trainer_family,
            "trainer_implementation_identity": self.trainer_implementation_identity,
            "training_paradigm": self.training_paradigm.value,
            "objective_identity": self.objective_identity,
            "trajectory_kind": self.trajectory_kind.value,
            "grouping": self.grouping.value,
            "schedule_step_count": self.schedule_step_count,
            "physical_transition_count": self.physical_transition_count,
            "stored_policy_transition_count": self.stored_policy_transition_count,
            "transition_selection": self.transition_selection.to_payload(),
            "replay_target": self.replay_target.value,
            "likelihood_semantics": self.likelihood_semantics.value,
            "credit_family": self.credit_family,
            "credit_assignment_identity": self.credit_assignment_identity,
            "reference_requirement": self.reference_requirement.value,
            "requires_reference_statistics": self.requires_reference_statistics,
            "beta": self.beta,
            "policy_microbatch_cardinality": (
                self.policy_microbatch_cardinality.to_payload()
            ),
            "update_cardinality": self.update_cardinality.to_payload(),
            "required_trajectory_fields": list(self.required_trajectory_fields),
            "required_policy_fields": list(self.required_policy_fields),
            "stage_trace_schema": [
                item.to_payload() for item in self.stage_trace_schema
            ],
        }


def _canonical_text(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _class_path(value: Any, *, field_name: str) -> str:
    text = _canonical_text(value, field_name=field_name)
    if ":" not in text:
        raise ValueError(f"{field_name} must use module:Class syntax")
    return text


def _canonical_string_tuple(value: Any, *, field_name: str) -> None:
    if type(value) is not tuple or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    for item in value:
        _canonical_text(item, field_name=f"{field_name} item")
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{field_name} must be sorted and unique")


def _sha256_payload(value: object) -> str:
    encoded = json.dumps(
        to_plain_dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_CANONICAL_STAGE_TRACE_SCHEMA = (
    StageTraceEntry(
        stage="prelude",
        input_type="builtins:int",
        output_type="visual_rl.data.prelude:PreludeBatchPayload",
    ),
    StageTraceEntry(
        stage="rollout",
        input_type="visual_rl.data.prelude:PreludeBatchPayload",
        output_type="visual_rl.algorithms.trainer.stages:RolloutStagePayload",
    ),
    StageTraceEntry(
        stage="reward",
        input_type="visual_rl.algorithms.trainer.stages:RolloutStagePayload",
        output_type="visual_rl.algorithms.trainer.stages:RewardedRollout",
    ),
    StageTraceEntry(
        stage="advantage",
        input_type="visual_rl.algorithms.trainer.stages:RewardedRollout",
        output_type="visual_rl.algorithms.trainer.stages:AdvantagedRollout",
    ),
    StageTraceEntry(
        stage="credit",
        input_type="visual_rl.algorithms.trainer.stages:AdvantagedRollout",
        output_type="visual_rl.algorithms.trainer.stages:CreditAssignedRollout",
    ),
    StageTraceEntry(
        stage="optimize",
        input_type="visual_rl.algorithms.trainer.stages:CreditAssignedRollout",
        output_type="visual_rl.algorithms.trainer.stages:OptimizedIteration",
    ),
)
