"""Final rollout-engine contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import hashlib
import json
from typing import Any, Literal

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RolloutBatch,
    RolloutRequest,
    RuntimeBuildContext,
    StepContext,
    ValidationCheck,
    ValidationContext,
)
from visual_rl.model_adapters.base import ModelAdapter


class RolloutEngine(ABC):
    """Expand one dataset batch, construct one request, and sample once."""

    @abstractmethod
    def sample(
        self,
        *,
        adapter: ModelAdapter,
        prompts: tuple[str, ...],
        metadata: tuple[Mapping[str, object], ...],
        context: StepContext,
    ) -> RolloutBatch:
        raise NotImplementedError

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__}.resolve_params() requires a mapping, "
                f"got {type(raw).__name__}"
            )
        del context
        return FrozenMapping(raw)

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, object],
        context: ValidationContext,
    ) -> tuple[ValidationCheck, ...]:
        del resolved, context
        return ()

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> RolloutEngine:
        del resolved, context
        raise NotImplementedError(f"{cls.__name__} must implement from_config()")

    @classmethod
    def required_capabilities(
        cls,
        resolved_params: Mapping[str, object],
    ) -> frozenset[str]:
        del resolved_params
        return frozenset()

    def close(self) -> None:
        """Release resources owned by this engine; pure engines are no-op."""


def _build_rollout_request(
    *,
    prompts: tuple[str, ...],
    metadata: tuple[Mapping[str, object], ...],
    context: StepContext,
    kind: Literal["full_trajectory", "single_step", "branching"],
    num_steps: int,
    group_size: int,
    selected_by_occurrence: tuple[int, ...] | None = None,
    branch_step: int | None = None,
) -> RolloutRequest:
    """Expand occurrence rows once and assign the sole sample identifiers."""

    if not isinstance(context, StepContext):
        raise TypeError("context must be a StepContext")
    if type(prompts) is not tuple or not prompts:
        raise ValueError("prompts must be a non-empty tuple")
    if type(metadata) is not tuple or len(metadata) != len(prompts):
        raise ValueError("metadata must contain one mapping per prompt")
    if selected_by_occurrence is not None and (
        type(selected_by_occurrence) is not tuple
        or len(selected_by_occurrence) != len(prompts)
    ):
        raise ValueError("selected timestep plan must contain one row per prompt")
    if kind == "branching" and type(branch_step) is not int:
        raise TypeError("branching requires an integer branch_step")

    occurrence_rows = tuple(
        _extract_occurrence(prompt, item)
        for prompt, item in zip(prompts, metadata, strict=True)
    )
    group_ids = tuple(row[2] for row in occurrence_rows)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("dataset occurrence group_id values must be unique")

    expanded_prompts: list[str] = []
    expanded_metadata: list[FrozenMapping] = []
    sample_ids: list[str] = []
    prompt_ids: list[str] = []
    expanded_group_ids: list[str] = []
    branch_id_values: list[int] = []
    selected_indices: list[int] = []
    branch_indices: list[int] = []

    for occurrence_index, (
        prompt,
        prompt_id,
        group_id,
        item_metadata,
    ) in enumerate(occurrence_rows):
        for member_index in range(group_size):
            branch_id: int | None = (
                member_index if kind == "branching" else None
            )
            local_row = len(expanded_prompts)
            expanded_prompts.append(prompt)
            expanded_metadata.append(item_metadata)
            prompt_ids.append(prompt_id)
            expanded_group_ids.append(group_id)
            if branch_id is not None:
                branch_id_values.append(branch_id)
            if selected_by_occurrence is not None:
                selected_indices.append(selected_by_occurrence[occurrence_index])
            if branch_step is not None:
                branch_indices.append(branch_step)
            sample_ids.append(
                _sample_id(
                    context=context,
                    prompt_id=prompt_id,
                    group_id=group_id,
                    branch_id=branch_id,
                    local_row=local_row,
                )
            )

    return RolloutRequest(
        prompts=tuple(expanded_prompts),
        metadata=tuple(expanded_metadata),
        sample_id=tuple(sample_ids),
        prompt_id=tuple(prompt_ids),
        group_id=tuple(expanded_group_ids),
        branch_id=(
            tuple(branch_id_values) if kind == "branching" else None
        ),
        context=context,
        kind=kind,
        num_steps=num_steps,
        group_size=group_size,
        selected_timestep_index=(
            tuple(selected_indices)
            if selected_by_occurrence is not None
            else None
        ),
        branch_step_index=(
            tuple(branch_indices) if branch_step is not None else None
        ),
    )


def _extract_occurrence(
    prompt: str,
    metadata: Mapping[str, object],
) -> tuple[str, str, str, FrozenMapping]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("dataset prompts must be non-empty strings")
    if not isinstance(metadata, Mapping):
        raise TypeError("dataset metadata rows must be mappings")
    row: dict[str, Any] = dict(metadata)
    for key in ("dataset_epoch", "dataset_index", "prompt_id", "group_id"):
        if key not in row:
            raise ValueError(f"dataset metadata is missing reserved key {key!r}")
    for key in ("dataset_epoch", "dataset_index"):
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError(f"metadata.{key} must be a non-negative integer")
    prompt_id = row.pop("prompt_id")
    group_id = row.pop("group_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("metadata.prompt_id must be a non-empty string")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("metadata.group_id must be a non-empty string")
    forbidden = {
        "sample_id",
        "branch_id",
        "selected_timestep_index",
        "branch_step_index",
    }
    overlap = forbidden.intersection(row)
    if overlap:
        raise ValueError(
            f"dataset metadata contains rollout-owned keys: {sorted(overlap)}"
        )
    return prompt, prompt_id, group_id, FrozenMapping(row)


def _sample_id(
    *,
    context: StepContext,
    prompt_id: str,
    group_id: str,
    branch_id: int | None,
    local_row: int,
) -> str:
    payload = json.dumps(
        {
            "step": context.step,
            "seed": context.seed,
            "rank": context.rank,
            "prompt_id": prompt_id,
            "group_id": group_id,
            "branch_id": branch_id,
            "local_row": local_row,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return (
        f"step-{context.step:06d}-rank-{context.rank:04d}-{digest}"
    )
