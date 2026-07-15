"""TempFlow rollout that requires an adapter-level shared-prefix implementation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from visual_rl.core.types import RolloutBatch, StepContext
from visual_rl.core.registry import ROLLOUT_ENGINES
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.rollout.base import RolloutEngine
from visual_rl.rollout.branch_utils import (
    branching_spec_from_config,
    resolve_branch_timesteps,
    select_branch_timestep,
)


class BranchingRollout(RolloutEngine):
    """Select one branch point and delegate the actual shared-prefix work."""

    def sample(
        self,
        adapter: ModelAdapter,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        context: StepContext | None = None,
    ) -> RolloutBatch:
        context = self.resolve_context(context)
        spec = branching_spec_from_config(self.config)
        base_runtime_config = self.runtime_config(context)
        transition_counter = getattr(adapter, "branch_transition_count", None)
        transition_count = int(
            transition_counter(base_runtime_config)
            if callable(transition_counter)
            else self.config.get("num_steps", 1)
        )
        if transition_count < 1:
            raise ValueError("Branching rollout requires at least one transition")
        candidates = resolve_branch_timesteps(
            transition_count,
            spec.branch_timesteps,
        )
        branch_step_index = select_branch_timestep(
            candidates,
            context.epoch_tag,
            spec.branch_timestep_strategy,
        )
        sample_branching = getattr(adapter, "sample_branching", None)
        if not callable(sample_branching):
            raise NotImplementedError(
                f"Adapter {adapter.name!r} does not implement shared-prefix sample_branching()."
            )

        branch_config = self.runtime_config(
            context,
            branch_step_index=branch_step_index,
            branch_step_candidates=candidates,
            branch_count=spec.branch_count,
            exploration_k=spec.branch_count,
            include_main=spec.include_main,
            transition_count=transition_count,
        )
        batch = sample_branching(prompts, metadata, branch_config)
        batch.model_metadata.update(
            {
                "rollout": "branching",
                "branching_mode": "shared_prefix",
                "branch_count": spec.branch_count,
                "include_main": spec.include_main,
                "branch_step_index": branch_step_index,
                "branch_step_candidates": candidates,
                "transition_count": transition_count,
            }
        )
        finalized = self.finalize_batch(
            batch,
            context,
            media_type=getattr(adapter, "media_type", None),
        )
        self._validate_result(
            finalized,
            len(prompts),
            spec.branch_count,
            spec.include_main,
        )
        return finalized

    @staticmethod
    def _validate_result(
        batch: RolloutBatch,
        parent_count: int,
        branch_count: int,
        include_main: bool,
    ) -> None:
        batch.validate_lightweight(strict=True)
        expected_group_size = branch_count + int(include_main)
        expected_size = parent_count * expected_group_size
        if len(batch.prompts) != expected_size:
            raise ValueError(
                f"sample_branching returned {len(batch.prompts)} samples; expected {expected_size}"
            )
        parent_indices = [item.get("parent_prompt_index") for item in batch.metadata]
        counts = Counter(parent_indices)
        if set(counts.values()) != {expected_group_size}:
            raise ValueError(
                "Every parent prompt must produce the same complete branch group"
            )
        required = {
            "branch_id",
            "branch_step_index",
            "branch_timestep_value",
        }
        for item in batch.metadata:
            missing = required.difference(item)
            if missing:
                raise ValueError(f"Branch metadata is missing fields: {sorted(missing)}")


ROLLOUT_ENGINES.register("branching", BranchingRollout)
