"""TempFlow shared-prefix branching through one typed Adapter request."""

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


class BranchingRollout(RolloutEngine):
    def __init__(
        self,
        *,
        num_steps: int,
        branch_count: int,
        branch_timesteps: tuple[int, ...],
    ) -> None:
        self.num_steps = num_steps
        self.branch_count = branch_count
        self.branch_timesteps = branch_timesteps

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> FrozenMapping:
        del context
        if not isinstance(raw, Mapping):
            raise TypeError("branching params must be a mapping")
        allowed = {"num_steps", "branch_count", "branch_timesteps"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown branching params: {sorted(unknown)}")
        num_steps = _positive_int("num_steps", raw.get("num_steps", 2))
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        branch_count = _positive_int(
            "branch_count", raw.get("branch_count", 4)
        )
        if branch_count < 2:
            raise ValueError("branch_count must be at least 2")
        candidates = _branch_timesteps(
            raw.get("branch_timesteps", "auto"),
            num_steps=num_steps,
        )
        return FrozenMapping(
            {
                "num_steps": num_steps,
                "branch_count": branch_count,
                "branch_timesteps": candidates,
            }
        )

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> "BranchingRollout":
        del context
        return cls(
            num_steps=int(resolved["num_steps"]),
            branch_count=int(resolved["branch_count"]),
            branch_timesteps=tuple(
                int(item) for item in resolved["branch_timesteps"]
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
        branch_step = self.branch_timesteps[
            context.step % len(self.branch_timesteps)
        ]
        request = _build_rollout_request(
            prompts=prompts,
            metadata=metadata,
            context=context,
            kind="branching",
            num_steps=self.num_steps,
            group_size=self.branch_count,
            branch_step=branch_step,
        )
        batch = adapter.sample(request)
        batch.validate_against(request)
        self._validate_branch_payload(batch, branch_step)
        return batch

    @staticmethod
    def _validate_branch_payload(
        batch: RolloutBatch,
        branch_step: int,
    ) -> None:
        import torch

        if batch.trajectory_step_index is None:
            raise ValueError("branching requires trajectory_step_index")
        expected = torch.tensor(
            [branch_step],
            dtype=torch.int64,
            device=batch.trajectory_step_index.device,
        )
        if not torch.equal(batch.trajectory_step_index, expected):
            raise ValueError(
                "trajectory_step_index must identify the selected branch step"
            )
        if batch.transition_std_dev is None:
            raise ValueError("branching requires transition_std_dev")


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _branch_timesteps(
    value: object,
    *,
    num_steps: int,
) -> tuple[int, ...]:
    transition_count = num_steps - 1
    if value == "auto":
        return tuple(range(transition_count))
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise TypeError(
            "branch_timesteps must be 'auto' or an integer sequence"
        )
    candidates = tuple(value)
    if not candidates:
        raise ValueError("branch_timesteps must not be empty")
    if any(type(item) is not int for item in candidates):
        raise TypeError("branch_timesteps values must be integers, not bool")
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError(
            "branch_timesteps must be strictly increasing and unique"
        )
    if any(not 0 <= item < transition_count for item in candidates):
        raise ValueError(
            "branch_timesteps values must satisfy "
            "0 <= value < num_steps - 1"
        )
    return candidates
