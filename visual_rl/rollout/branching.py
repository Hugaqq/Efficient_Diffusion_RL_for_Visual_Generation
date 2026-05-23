"""TempFlow branching rollout engine."""

from __future__ import annotations

from visual_rl.core.types import RolloutBatch
from visual_rl.integrations.tempflow_grpo.branching import (
    branching_spec_from_config,
    expand_branch_inputs,
    resolve_branch_timesteps,
    select_branch_timestep,
)
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.rollout.base import RolloutEngine


class BranchingRollout(RolloutEngine):
    """Create main-plus-branch samples for process-reward credit assignment."""

    def sample(self, adapter: ModelAdapter, prompts: list[str], metadata: list[dict]) -> RolloutBatch:
        import torch

        spec = branching_spec_from_config(self.config)
        branch_timesteps = resolve_branch_timesteps(int(self.config.get("num_steps", 2)), spec.branch_timesteps)
        branch_timestep = select_branch_timestep(
            branch_timesteps,
            self.config.get("epoch_tag"),
            spec.branch_timestep_strategy,
        )
        expanded_prompts, expanded_metadata, branch_ids, parent_indices = expand_branch_inputs(
            prompts,
            metadata,
            spec,
            branch_timestep,
        )

        branch_config = dict(self.config)
        branch_config.update(
            {
                "branch_timestep": branch_timestep,
                "branch_timesteps": branch_timesteps,
                "branch_count": spec.branch_count,
                "exploration_k": spec.exploration_k,
                "include_main": spec.include_main,
            }
        )

        sample_branching = getattr(adapter, "sample_branching", None)
        if callable(sample_branching):
            batch = sample_branching(prompts, metadata, branch_config)
        else:
            batch = adapter.sample(expanded_prompts, expanded_metadata, branch_config)

        batch.branch_ids = torch.as_tensor(branch_ids, dtype=torch.long)
        batch.model_metadata.update(
            {
                "rollout": "branching",
                "branch_count": spec.branch_count,
                "include_main": spec.include_main,
                "exploration_k": spec.exploration_k,
                "branch_timestep": branch_timestep,
                "branch_timesteps": branch_timesteps,
                "branch_ids": branch_ids,
                "parent_prompt_indices": parent_indices,
            }
        )
        return batch
