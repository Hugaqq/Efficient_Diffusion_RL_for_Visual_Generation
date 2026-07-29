"""Full-trajectory rollout through the one typed Adapter request."""

from __future__ import annotations

from collections.abc import Mapping

from visual_rl.core.types import (
    FrozenMapping,
    ResolutionContext,
    RolloutBatch,
    RuntimeBuildContext,
    StepContext,
)
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.rollout.base import RolloutEngine, _build_rollout_request


class FullTrajectoryRollout(RolloutEngine):
    def __init__(self, *, num_steps: int, samples_per_prompt: int) -> None:
        self.num_steps = num_steps
        self.samples_per_prompt = samples_per_prompt

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> FrozenMapping:
        del context
        values = _params(raw, {"num_steps", "samples_per_prompt"})
        num_steps = _positive_int("num_steps", values.get("num_steps", 2))
        samples = _positive_int(
            "samples_per_prompt", values.get("samples_per_prompt", 2)
        )
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        return FrozenMapping(
            {
                "num_steps": num_steps,
                "samples_per_prompt": samples,
            }
        )

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> "FullTrajectoryRollout":
        del context
        return cls(
            num_steps=int(resolved["num_steps"]),
            samples_per_prompt=int(resolved["samples_per_prompt"]),
        )

    def sample(
        self,
        *,
        adapter: ModelAdapter,
        prompts: tuple[str, ...],
        metadata: tuple[Mapping[str, object], ...],
        context: StepContext,
    ) -> RolloutBatch:
        request = _build_rollout_request(
            prompts=prompts,
            metadata=metadata,
            context=context,
            kind="full_trajectory",
            num_steps=self.num_steps,
            group_size=self.samples_per_prompt,
        )
        batch = adapter.sample(request)
        batch.validate_against(request)
        return batch


def _params(
    raw: Mapping[str, object],
    allowed: set[str],
) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError("full_trajectory params must be a mapping")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown full_trajectory params: {sorted(unknown)}")
    return raw


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value
