"""Import-safe rollout declarations owned by the algorithm domain.

This module deliberately contains no rollout implementation imports.  It is
the static half of the rollout component boundary: frozen configuration,
declared contracts, declaration providers, and the domain catalog fragment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    CatalogFragment,
    ComponentDeclaration,
    ComponentDescriptor,
    DeclaredContract,
    GroupingKind,
    RolloutContract,
    TaskKind,
    TrajectoryKind,
    TransitionKind,
)
from visual_rl.data.samples.trajectory import BranchTopology

__all__ = (
    "ROLLOUT_CATALOG_FRAGMENT",
    "BranchingRolloutConfig",
    "BranchingRolloutDeclarationProvider",
    "FullTrajectoryRolloutConfig",
    "FullTrajectoryRolloutDeclarationProvider",
    "SingleStepRolloutConfig",
    "SingleStepRolloutDeclarationProvider",
    "rollout_catalog_fragment",
)

_BASE_TRANSITION_FIELDS = (
    "x_t",
    "sampled_action",
    "conditioned_next",
    "t",
    "t_next",
    "old_log_prob",
    "transition_index",
    "condition_identity",
    "guidance_identity",
)
_BRANCH_ABLATION_TRANSITION_FIELDS = (
    *_BASE_TRANSITION_FIELDS,
    "shared_prefix_id",
    "branch_step_identity",
)
_BRANCH_PAPER_TRANSITION_FIELDS = (
    *_BASE_TRANSITION_FIELDS,
    "branch_topology",
    "exploration_member_index",
    "branch_timestep_index",
    "transition_terminal_media",
)
_SINGLE_STEP_TRANSITION_FIELDS = (
    *_BASE_TRANSITION_FIELDS[:7],
    "selected_timestep_index",
    *_BASE_TRANSITION_FIELDS[7:],
)


def _values(
    values: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} params must be a mapping")
    unknown = tuple(sorted(set(values) - allowed))
    if unknown:
        raise ValueError(f"unknown {label} params: {list(unknown)}")
    missing = tuple(sorted(required - set(values)))
    if missing:
        raise ValueError(f"missing {label} params: {list(missing)}")
    return dict(values)


def _positive(name: str, value: object, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True)
class FullTrajectoryRolloutConfig:
    num_steps: int

    def __post_init__(self) -> None:
        _positive("num_steps", self.num_steps)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> FullTrajectoryRolloutConfig:
        del context
        return cls(
            **_values(
                values,
                allowed=frozenset({"num_steps"}),
                required=frozenset({"num_steps"}),
                label="full-trajectory rollout",
            )
        )

    def describe_contract(self) -> DeclaredContract:
        count = (self.num_steps, self.num_steps)
        return DeclaredContract(
            component_kind="rollout",
            component_id="full-trajectory",
            rollout=RolloutContract(
                accepted_tasks=(TaskKind.T2I, TaskKind.T2V, TaskKind.I2V),
                accepted_transitions=(TransitionKind.SDE,),
                trajectory_kind=TrajectoryKind.FULL,
                grouping=GroupingKind.PROMPT_COMPLETIONS,
                requires_branchable=False,
                requires_deterministic_ode=False,
                required_transition_fields=_BASE_TRANSITION_FIELDS,
                produced_trajectory_fields=_BASE_TRANSITION_FIELDS,
                schedule_step_count=count,
                physical_transition_count=count,
                stored_policy_transition_count=count,
            ),
        )


@dataclass(frozen=True, slots=True)
class BranchingRolloutConfig:
    num_steps: int
    branch_count: int
    branch_topology: BranchTopology
    branch_step_policy: Literal["uniform_intermediate"] | None = None
    branch_step_index: int | None = None

    def __post_init__(self) -> None:
        _positive("num_steps", self.num_steps, minimum=2)
        _positive("branch_count", self.branch_count, minimum=2)
        topology = self.branch_topology
        if not isinstance(topology, BranchTopology):
            raise TypeError("branch_topology must be a BranchTopology")
        if topology.exploration_count != self.branch_count:
            raise ValueError(
                "branch_topology exploration_count must equal branch_count"
            )
        if topology.kind == "every_policy_timestep":
            if (
                self.branch_step_policy is not None
                or self.branch_step_index is not None
            ):
                raise ValueError(
                    "every_policy_timestep does not accept a single branch-step policy"
                )
        else:
            if self.branch_step_policy != "uniform_intermediate":
                raise ValueError(
                    "single_point_branch_ablation requires "
                    "branch_step_policy=uniform_intermediate"
                )
            if self.branch_step_index is not None and (
                type(self.branch_step_index) is not int
                or not 0 <= self.branch_step_index < self.num_steps - 1
            ):
                raise ValueError(
                    "branch_step_index must precede the final schedule step"
                )

    @property
    def selection_contract_identity(self) -> str:
        payload = {
            "schema_version": 1,
            "strategy": "branching",
            "num_steps": self.num_steps,
            "branch_count": self.branch_count,
            "branch_topology": self.branch_topology.to_payload(),
            "branch_step_policy": self.branch_step_policy,
            "branch_step_index": self.branch_step_index,
            "paper_selection_rule": (
                "all_nonterminal"
                if self.branch_topology.kind == "every_policy_timestep"
                else None
            ),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"branching-selection.v1:{hashlib.sha256(encoded).hexdigest()}"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> BranchingRolloutConfig:
        del context
        resolved = _values(
            values,
            allowed=frozenset(
                {
                    "num_steps",
                    "branch_count",
                    "branch_step_policy",
                    "branch_step_index",
                    "branch_topology",
                }
            ),
            required=frozenset({"num_steps", "branch_count", "branch_topology"}),
            label="branching rollout",
        )
        topology = resolved.get("branch_topology")
        if topology is not None and not isinstance(topology, BranchTopology):
            resolved["branch_topology"] = BranchTopology.from_payload(topology)
        return cls(**resolved)

    def describe_contract(self) -> DeclaredContract:
        paper = self.branch_topology.kind == "every_policy_timestep"
        fields = (
            _BRANCH_PAPER_TRANSITION_FIELDS
            if paper
            else _BRANCH_ABLATION_TRANSITION_FIELDS
        )
        physical = (
            (self.num_steps - 1) * (self.num_steps + 4) // 2
            if paper
            else self.num_steps
        )
        stored = self.num_steps - 1 if paper else 1
        return DeclaredContract(
            component_kind="rollout",
            component_id="branching",
            rollout=RolloutContract(
                accepted_tasks=(TaskKind.T2I, TaskKind.T2V, TaskKind.I2V),
                accepted_transitions=(TransitionKind.SDE,),
                trajectory_kind=TrajectoryKind.BRANCHING,
                grouping=GroupingKind.BRANCHES,
                requires_branchable=True,
                requires_deterministic_ode=True,
                required_transition_fields=fields,
                produced_trajectory_fields=fields,
                schedule_step_count=(self.num_steps, self.num_steps),
                physical_transition_count=(physical, physical),
                stored_policy_transition_count=(stored, stored),
            ),
        )


@dataclass(frozen=True, slots=True)
class SingleStepRolloutConfig:
    """One stored stochastic action plus deterministic media continuations."""

    selected_timestep_policy: Literal["uniform"]
    num_steps: int = 40
    selected_timestep_index: int | None = None
    candidate_timestep_window: tuple[int, int] | None = None
    candidate_timestep_indices: tuple[int, ...] | None = None
    selection_key: Literal["row", "prompt"] = "row"
    selection_domain: Literal["single_process", "global_rank_broadcast"] = (
        "single_process"
    )

    def __post_init__(self) -> None:
        if self.selected_timestep_policy != "uniform":
            raise ValueError("selected_timestep_policy must be uniform")
        _positive("num_steps", self.num_steps)
        window = self.candidate_timestep_window
        indices = self.candidate_timestep_indices
        if window is not None and indices is not None:
            raise ValueError(
                "candidate_timestep_window and candidate_timestep_indices "
                "are mutually exclusive"
            )
        if window is not None:
            if (
                type(window) is not tuple
                or len(window) != 2
                or any(type(item) is not int for item in window)
            ):
                raise TypeError("candidate_timestep_window must be a two-integer tuple")
            start, stop = window
            if start < 0 or stop <= start or start >= self.num_steps:
                raise ValueError("candidate_timestep_window must overlap the schedule")
        if indices is not None:
            if type(indices) is not tuple or not indices:
                raise ValueError("candidate_timestep_indices must be a non-empty tuple")
            if any(type(item) is not int for item in indices):
                raise TypeError("candidate_timestep_indices must contain integers")
            if len(indices) != len(set(indices)):
                raise ValueError("candidate_timestep_indices must be unique")
            if any(not 0 <= item < self.num_steps for item in indices):
                raise ValueError(
                    "candidate_timestep_indices must be inside the schedule"
                )
            object.__setattr__(
                self, "candidate_timestep_indices", tuple(sorted(indices))
            )
        if self.selection_key not in {"row", "prompt"}:
            raise ValueError("selection_key must be row or prompt")
        if self.selection_domain not in {"single_process", "global_rank_broadcast"}:
            raise ValueError(
                "selection_domain must be single_process or global_rank_broadcast"
            )
        if self.selected_timestep_index is not None:
            if (
                type(self.selected_timestep_index) is not int
                or not 0 <= self.selected_timestep_index < self.num_steps
            ):
                raise ValueError("selected_timestep_index must be inside the schedule")
            if self.selected_timestep_index not in self.candidate_indices:
                raise ValueError(
                    "selected_timestep_index must belong to the candidate set"
                )

    @property
    def candidate_indices(self) -> tuple[int, ...]:
        if self.candidate_timestep_indices is not None:
            return self.candidate_timestep_indices
        if self.candidate_timestep_window is None:
            return tuple(range(self.num_steps))
        start, stop = self.candidate_timestep_window
        return tuple(range(start, min(stop, self.num_steps)))

    @property
    def selection_contract_identity(self) -> str:
        payload = {
            "schema_version": 1,
            "strategy": "single_step",
            "policy": self.selected_timestep_policy,
            "num_steps": self.num_steps,
            "candidate_timestep_indices": list(self.candidate_indices),
            "selected_timestep_index": self.selected_timestep_index,
            "selection_key": self.selection_key,
            "selection_domain": self.selection_domain,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"single-step-selection.v1:{hashlib.sha256(encoded).hexdigest()}"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> SingleStepRolloutConfig:
        del context
        resolved = _values(
            values,
            allowed=frozenset(
                {
                    "selected_timestep_policy",
                    "num_steps",
                    "selected_timestep_index",
                    "candidate_timestep_window",
                    "candidate_timestep_indices",
                    "selection_key",
                    "selection_domain",
                }
            ),
            required=frozenset({"selected_timestep_policy"}),
            label="single-step rollout",
        )
        for name in ("candidate_timestep_window", "candidate_timestep_indices"):
            if name in resolved and isinstance(resolved[name], list):
                resolved[name] = tuple(resolved[name])
        return cls(**resolved)

    def describe_contract(self) -> DeclaredContract:
        count = (self.num_steps, self.num_steps)
        return DeclaredContract(
            component_kind="rollout",
            component_id="single-step",
            rollout=RolloutContract(
                accepted_tasks=(TaskKind.T2I, TaskKind.T2V, TaskKind.I2V),
                accepted_transitions=(TransitionKind.SDE,),
                trajectory_kind=TrajectoryKind.SINGLE_STEP,
                grouping=GroupingKind.SELECTED_TIMESTEP,
                requires_branchable=False,
                requires_deterministic_ode=self.num_steps > 1,
                required_transition_fields=_SINGLE_STEP_TRANSITION_FIELDS,
                produced_trajectory_fields=_SINGLE_STEP_TRANSITION_FIELDS,
                schedule_step_count=count,
                physical_transition_count=count,
                stored_policy_transition_count=(1, 1),
            ),
        )


class FullTrajectoryRolloutDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.rollout.config:FullTrajectoryRolloutConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = FullTrajectoryRolloutConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config, declared_contract=config.describe_contract()
        )


class BranchingRolloutDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.rollout.config:BranchingRolloutConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = BranchingRolloutConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config, declared_contract=config.describe_contract()
        )


class SingleStepRolloutDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.rollout.config:SingleStepRolloutConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = SingleStepRolloutConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config, declared_contract=config.describe_contract()
        )


ROLLOUT_CATALOG_FRAGMENT = CatalogFragment(
    owner="algorithms.rollout",
    kind="rollout",
    descriptors=(
        ComponentDescriptor(
            alias="full-trajectory",
            implementation_class_path=(
                "visual_rl.algorithms.rollout.full_trajectory:FullTrajectoryRollout"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.rollout.config:"
                "FullTrajectoryRolloutDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
        ComponentDescriptor(
            alias="branching",
            implementation_class_path=(
                "visual_rl.algorithms.rollout.branching:BranchingRollout"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.rollout.config:"
                "BranchingRolloutDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
        ComponentDescriptor(
            alias="single-step",
            implementation_class_path=(
                "visual_rl.algorithms.rollout.single_step:SingleStepRollout"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.rollout.config:"
                "SingleStepRolloutDeclarationProvider"
            ),
            optional_dependencies=("torch",),
        ),
    ),
)


def rollout_catalog_fragment() -> CatalogFragment:
    return ROLLOUT_CATALOG_FRAGMENT
