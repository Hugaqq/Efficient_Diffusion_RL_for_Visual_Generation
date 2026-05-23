"""Branching utilities used by TempFlow-style rollout."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BranchingSpec:
    branch_count: int = 4
    exploration_k: int = 6
    include_main: bool = True
    branch_timesteps: list[int] | str = "auto"
    branch_timestep_strategy: str = "cycle"


def branching_spec_from_config(config: dict[str, Any]) -> BranchingSpec:
    branch_config = dict(config.get("branch") or {})
    return BranchingSpec(
        branch_count=int(config.get("branch_count", branch_config.get("branch_count", 4))),
        exploration_k=int(config.get("exploration_k", branch_config.get("exploration_k", 6))),
        include_main=bool(config.get("include_main", branch_config.get("include_main", True))),
        branch_timesteps=config.get("branch_timesteps", branch_config.get("branch_timesteps", "auto")),
        branch_timestep_strategy=str(
            config.get("branch_timestep_strategy", branch_config.get("branch_timestep_strategy", "cycle"))
        ),
    )


def resolve_branch_timesteps(num_steps: int, branch_timesteps: list[int] | str = "auto") -> list[int]:
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
    branch_timestep: int,
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
                    "branch_timestep": branch_timestep,
                    "exploration_k": spec.exploration_k,
                    "is_main_branch": branch_id == -1,
                    "rollout_kind": "tempflow_branching",
                }
            )
            expanded_prompts.append(prompt)
            expanded_metadata.append(branch_metadata)
            expanded_branch_ids.append(branch_id)
            parent_indices.append(parent_index)
    return expanded_prompts, expanded_metadata, expanded_branch_ids, parent_indices

