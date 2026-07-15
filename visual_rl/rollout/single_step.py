"""Flash-GRPO single-step rollout engine."""

from __future__ import annotations

from typing import Any

from visual_rl.core.types import RolloutBatch, StepContext
from visual_rl.core.registry import ROLLOUT_ENGINES
from visual_rl.rollout.rectification import scheduler_rectification_weights
from visual_rl.rollout.timestep_sampler import (
    expand_prompt_groups,
    resolve_timestep_indices,
    select_prompt_timestep_indices,
    single_step_spec_from_config,
)
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.rollout.base import RolloutEngine


class SingleStepRollout(RolloutEngine):
    """Expand prompt groups and keep the selected timestep logprob only."""

    def sample(
        self,
        adapter: ModelAdapter,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        context: StepContext | None = None,
    ) -> RolloutBatch:
        context = self.resolve_context(context)
        spec = single_step_spec_from_config(self.config)
        num_steps = int(self.config.get("num_steps", 2))
        candidates = resolve_timestep_indices(num_steps, spec.timestep_range)
        selected_per_prompt = select_prompt_timestep_indices(
            prompt_count=len(prompts),
            candidates=candidates,
            strategy=spec.selected_step_strategy,
            epoch_tag=context.epoch_tag,
            seed=context.seed,
        )
        expanded_prompts, expanded_metadata, selected_indices, parent_indices = expand_prompt_groups(
            prompts,
            metadata,
            spec.samples_per_prompt,
            selected_per_prompt,
        )

        single_step_config = self.runtime_config(
            context,
            rollout_kind="flash_single_step",
            selected_timestep_indices=selected_indices,
            selected_timesteps=selected_indices,
            timestep_candidates=candidates,
            samples_per_prompt=spec.samples_per_prompt,
        )

        sample_single_step = getattr(adapter, "sample_single_step", None)
        if callable(sample_single_step):
            batch = sample_single_step(expanded_prompts, expanded_metadata, single_step_config)
        else:
            batch = adapter.sample(expanded_prompts, expanded_metadata, single_step_config)
            batch = self._narrow_to_selected_timestep(batch, selected_indices)

        rectification_mode = str(
            self.config.get("rectification_mode", "scheduler_formula")
        )
        timestep_values = None
        if rectification_mode.lower() == "flash_reference_table":
            timestep_values = self._single_timestep_values(batch)
        rectification_weights = scheduler_rectification_weights(
            selected_indices,
            num_steps=num_steps,
            mode=rectification_mode,
            timestep_values=timestep_values,
        )
        batch.model_metadata.update(
            {
                "rollout": "single_step",
                "selected_step_strategy": spec.selected_step_strategy,
                "selected_timestep_indices": selected_indices,
                "selected_timesteps": selected_indices,
                "timestep_candidates": candidates,
                "samples_per_prompt": spec.samples_per_prompt,
                "parent_prompt_indices": parent_indices,
                "num_steps": num_steps,
                "rectification_mode": rectification_mode,
                "selected_timestep_values": timestep_values,
                "flash_rectification_weights": [[value] for value in rectification_weights],
            }
        )
        return self.finalize_batch(
            batch,
            context,
            media_type=getattr(adapter, "media_type", None),
        )

    @staticmethod
    def _single_timestep_values(batch: RolloutBatch) -> list[int]:
        import torch

        timesteps = batch.timesteps
        if not isinstance(timesteps, torch.Tensor):
            raise ValueError(
                "flash_reference_table requires tensor scheduler timesteps"
            )
        if timesteps.ndim != 2 or timesteps.shape[1] != 1:
            raise ValueError(
                "flash_reference_table requires exactly one retained timestep per sample"
            )
        return [int(value) for value in timesteps[:, 0].detach().cpu().tolist()]

    @staticmethod
    def _narrow_to_selected_timestep(batch: RolloutBatch, selected_indices: list[int]) -> RolloutBatch:
        import torch

        def narrow(value):
            if value is None or not isinstance(value, torch.Tensor):
                return value
            if value.ndim < 2 or value.shape[0] != len(selected_indices):
                return value
            if value.shape[1] <= max(selected_indices):
                raise ValueError(
                    f"Cannot select timestep {max(selected_indices)} from tensor with second dimension {value.shape[1]}"
                )
            return torch.stack([value[row, index : index + 1] for row, index in enumerate(selected_indices)], dim=0)

        return batch.replace(
            latents=narrow(batch.latents),
            next_latents=narrow(batch.next_latents),
            timesteps=narrow(batch.timesteps),
            old_log_probs=narrow(batch.old_log_probs),
            kl=narrow(batch.kl),
            transition_mask=narrow(batch.transition_mask),
        )


ROLLOUT_ENGINES.register("single_step", SingleStepRollout)
ROLLOUT_ENGINES.register("flash_single_step", SingleStepRollout)
