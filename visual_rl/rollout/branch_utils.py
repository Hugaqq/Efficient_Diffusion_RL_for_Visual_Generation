"""Branching utilities used by TempFlow-style rollout."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BranchingSpec:
    branch_count: int = 4
    include_main: bool = False
    branch_timesteps: list[int] | str = "auto"
    branch_timestep_strategy: str = "cycle"


def branching_spec_from_config(config: dict[str, Any]) -> BranchingSpec:
    branch_config = dict(config.get("branch") or {})
    branch_count = int(
        config.get(
            "branch_count",
            config.get("exploration_k", branch_config.get("branch_count", 4)),
        )
    )
    exploration_k = config.get("exploration_k", branch_config.get("exploration_k"))
    if exploration_k is not None and int(exploration_k) != branch_count:
        raise ValueError("exploration_k and branch_count must describe the same number of branches")
    if branch_count < 2:
        raise ValueError("branch_count must be >= 2 for TempFlow group advantages")
    return BranchingSpec(
        branch_count=branch_count,
        include_main=bool(config.get("include_main", branch_config.get("include_main", False))),
        branch_timesteps=config.get("branch_timesteps", branch_config.get("branch_timesteps", "auto")),
        branch_timestep_strategy=str(
            config.get("branch_timestep_strategy", branch_config.get("branch_timestep_strategy", "cycle"))
        ),
    )


def resolve_branch_timesteps(num_steps: int, branch_timesteps: list[int] | str = "auto") -> list[int]:
    """Resolve candidate denoising step indices, not scheduler timestep values."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if branch_timesteps == "auto" or branch_timesteps is None:
        return list(range(num_steps))
    if isinstance(branch_timesteps, str):
        values = [int(item.strip()) for item in branch_timesteps.split(",") if item.strip()]
    else:
        values = [int(item) for item in branch_timesteps]
    if not values:
        raise ValueError("branch_timesteps cannot be empty")
    invalid = [item for item in values if item < 0 or item >= num_steps]
    if invalid:
        raise ValueError(f"branch_timesteps out of range for num_steps={num_steps}: {invalid}")
    return values


def select_branch_timestep(timesteps: list[int], epoch_tag: int | None, strategy: str = "cycle") -> int:
    if strategy != "cycle":
        raise ValueError(f"Unknown branch_timestep_strategy: {strategy}")
    index = int(epoch_tag or 0) % len(timesteps)
    return int(timesteps[index])


def expand_branch_inputs(
    prompts: list[str],
    metadata: list[dict[str, Any]],
    spec: BranchingSpec,
    branch_step_index: int,
) -> tuple[list[str], list[dict[str, Any]], list[int], list[int]]:
    if len(prompts) != len(metadata):
        raise ValueError("prompts and metadata must have the same length")
    branch_ids = [-1] if spec.include_main else []
    branch_ids.extend(range(spec.branch_count))

    expanded_prompts: list[str] = []
    expanded_metadata: list[dict[str, Any]] = []
    expanded_branch_ids: list[int] = []
    parent_indices: list[int] = []
    for parent_index, (prompt, item) in enumerate(zip(prompts, metadata, strict=True)):
        for branch_id in branch_ids:
            branch_metadata = deepcopy(item)
            branch_metadata.update(
                {
                    "parent_prompt_index": parent_index,
                    "branch_id": branch_id,
                    "branch_step_index": branch_step_index,
                    "exploration_k": spec.branch_count,
                    "is_main_branch": branch_id == -1,
                    "rollout_kind": "tempflow_branching",
                }
            )
            expanded_prompts.append(prompt)
            expanded_metadata.append(branch_metadata)
            expanded_branch_ids.append(branch_id)
            parent_indices.append(parent_index)
    return expanded_prompts, expanded_metadata, expanded_branch_ids, parent_indices
