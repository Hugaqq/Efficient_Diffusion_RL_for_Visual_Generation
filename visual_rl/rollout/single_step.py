"""Flash-GRPO single-step rollout through one typed request."""

from __future__ import annotations

from collections.abc import Mapping
import random

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.rollout.base import RolloutEngine, _build_rollout_request


_STRATEGIES = frozenset(
    {"iso_temporal", "first", "last", "middle", "seeded_random"}
)


class SingleStepRollout(RolloutEngine):
    def __init__(
        self,
        *,
        num_steps: int,
        samples_per_prompt: int,
        selected_step_strategy: str,
        timestep_range: tuple[int, int] | None,
    ) -> None:
        self.num_steps = num_steps
        self.samples_per_prompt = samples_per_prompt
        self.selected_step_strategy = selected_step_strategy
        self.timestep_range = timestep_range

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> FrozenMapping:
        del context
        if not isinstance(raw, Mapping):
            raise TypeError("single_step params must be a mapping")
        allowed = {
            "num_steps",
            "samples_per_prompt",
            "selected_step_strategy",
            "timestep_range",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown single_step params: {sorted(unknown)}")
        num_steps = _positive_int("num_steps", raw.get("num_steps", 2))
        samples = _positive_int(
            "samples_per_prompt", raw.get("samples_per_prompt", 2)
        )
        strategy = raw.get("selected_step_strategy", "iso_temporal")
        if not isinstance(strategy, str) or strategy not in _STRATEGIES:
            raise ValueError(
                "selected_step_strategy must be one of "
                f"{sorted(_STRATEGIES)}"
            )
        timestep_range = _resolve_range(raw.get("timestep_range"), num_steps)
        return FrozenMapping(
            {
                "num_steps": num_steps,
                "samples_per_prompt": samples,
                "selected_step_strategy": strategy,
                "timestep_range": timestep_range,
            }
        )

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> "SingleStepRollout":
        del context
        timestep_range = resolved["timestep_range"]
        return cls(
            num_steps=int(resolved["num_steps"]),
            samples_per_prompt=int(resolved["samples_per_prompt"]),
            selected_step_strategy=str(resolved["selected_step_strategy"]),
            timestep_range=(
                None
                if timestep_range is None
                else (int(timestep_range[0]), int(timestep_range[1]))
            ),
        )

    def sample(
        self,
        *,
        adapter: ModelAdapter,
        prompts: tuple[str, ...],
        metadata: tuple[Mapping[str, object], ...],
        context: StepContext,
    ) -> RolloutBatch:
        selected = self._selected_indices(len(prompts), context)
        request = _build_rollout_request(
            prompts=prompts,
            metadata=metadata,
            context=context,
            kind="single_step",
            num_steps=self.num_steps,
            group_size=self.samples_per_prompt,
            selected_by_occurrence=selected,
        )
        batch = adapter.sample(request)
        batch.validate_against(request)
        return batch

    def _selected_indices(
        self,
        occurrence_count: int,
        context: StepContext,
    ) -> tuple[int, ...]:
        if occurrence_count < 1:
            raise ValueError("single_step requires at least one prompt")
        start, end = self.timestep_range or (0, self.num_steps - 1)
        candidates = tuple(range(start, end + 1))
        if self.selected_step_strategy == "iso_temporal":
            return tuple(
                candidates[(context.step + index) % len(candidates)]
                for index in range(occurrence_count)
            )
        if self.selected_step_strategy == "first":
            return (candidates[0],) * occurrence_count
        if self.selected_step_strategy == "last":
            return (candidates[-1],) * occurrence_count
        if self.selected_step_strategy == "middle":
            return (candidates[len(candidates) // 2],) * occurrence_count
        generator = random.Random(context.seed)
        return tuple(
            generator.choice(candidates) for _ in range(occurrence_count)
        )


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_range(
    value: object,
    num_steps: int,
) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("timestep_range must be null or [start, end]")
    start, end = value
    if type(start) is not int or type(end) is not int:
        raise TypeError("timestep_range values must be integers, not bool")
    if not 0 <= start <= end < num_steps:
        raise ValueError(
            "timestep_range must satisfy 0 <= start <= end < num_steps"
        )
    return start, end
