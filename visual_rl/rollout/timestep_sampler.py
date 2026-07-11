"""Timestep sampling helpers for Flash-GRPO single-step rollouts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SingleStepSpec:
    samples_per_prompt: int = 2
    selected_step_strategy: str = "iso_temporal"
    timestep_range: tuple[int, int] | None = None


def single_step_spec_from_config(config: dict[str, Any]) -> SingleStepSpec:
    samples_per_prompt = int(config.get("samples_per_prompt", 2))
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    raw_range = config.get("timestep_range")
    timestep_range = None
    if raw_range is not None:
        if len(raw_range) != 2:
            raise ValueError("timestep_range must be [start, end]")
        timestep_range = (int(raw_range[0]), int(raw_range[1]))
    return SingleStepSpec(
        samples_per_prompt=samples_per_prompt,
        selected_step_strategy=str(config.get("selected_step_strategy", "iso_temporal")),
        timestep_range=timestep_range,
    )


def resolve_timestep_indices(num_steps: int, timestep_range: tuple[int, int] | None = None) -> list[int]:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if timestep_range is None:
        return list(range(num_steps))
    start, end = timestep_range
    start = max(0, min(num_steps - 1, start))
    end = max(0, min(num_steps - 1, end))
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def select_prompt_timestep_indices(
    prompt_count: int,
    candidates: list[int],
    strategy: str,
    epoch_tag: int | None = None,
    seed: int | None = None,
) -> list[int]:
    if prompt_count < 1:
        return []
    if not candidates:
        raise ValueError("candidates must contain at least one timestep index")

    strategy = strategy.lower()
    epoch = int(epoch_tag or 0)
    if strategy in {"iso_temporal", "cycle"}:
        return [candidates[(epoch + prompt_index) % len(candidates)] for prompt_index in range(prompt_count)]
    if strategy in {"first", "min"}:
        return [candidates[0] for _ in range(prompt_count)]
    if strategy in {"last", "max"}:
        return [candidates[-1] for _ in range(prompt_count)]
    if strategy == "middle":
        return [candidates[len(candidates) // 2] for _ in range(prompt_count)]
    if strategy in {"random", "seeded_random"}:
        rng = random.Random(int(seed or 0) + epoch)
        return [rng.choice(candidates) for _ in range(prompt_count)]
    raise ValueError(f"Unknown selected_step_strategy: {strategy}")


def expand_prompt_groups(
    prompts: list[str],
    metadata: list[dict[str, Any]],
    samples_per_prompt: int,
    selected_indices: list[int],
) -> tuple[list[str], list[dict[str, Any]], list[int], list[int]]:
    if len(prompts) != len(metadata):
        raise ValueError("prompts and metadata must have the same length")
    if len(selected_indices) != len(prompts):
        raise ValueError("selected_indices must have one entry per prompt")

    expanded_prompts: list[str] = []
    expanded_metadata: list[dict[str, Any]] = []
    expanded_selected_indices: list[int] = []
    parent_indices: list[int] = []

    for parent_index, (prompt, item) in enumerate(zip(prompts, metadata, strict=True)):
        selected_index = int(selected_indices[parent_index])
        for sample_index in range(samples_per_prompt):
            expanded_prompts.append(prompt)
            expanded_item = dict(item)
            expanded_item.update(
                {
                    "parent_prompt_index": parent_index,
                    "sample_index": sample_index,
                    "selected_timestep": selected_index,
                    "selected_timestep_index": selected_index,
                    "rollout_kind": "flash_single_step",
                }
            )
            expanded_metadata.append(expanded_item)
            expanded_selected_indices.append(selected_index)
            parent_indices.append(parent_index)
    return expanded_prompts, expanded_metadata, expanded_selected_indices, parent_indices
