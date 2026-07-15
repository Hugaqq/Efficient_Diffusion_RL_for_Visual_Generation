"""Full-trajectory rollout engine."""

from __future__ import annotations

from typing import Any

from visual_rl.core.types import RolloutBatch, StepContext
from visual_rl.core.registry import ROLLOUT_ENGINES
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.rollout.base import RolloutEngine


class FullTrajectoryRollout(RolloutEngine):
    def sample(
        self,
        adapter: ModelAdapter,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        context: StepContext | None = None,
    ) -> RolloutBatch:
        context = self.resolve_context(context)
        samples_per_prompt = int(self.config.get("samples_per_prompt", 2))
        if samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be >= 1")
        expanded_prompts: list[str] = []
        expanded_metadata: list[dict[str, Any]] = []
        parent_indices: list[int] = []
        for parent_index, (prompt, item) in enumerate(
            zip(prompts, metadata, strict=True)
        ):
            for sample_index in range(samples_per_prompt):
                sample_metadata = dict(item)
                sample_metadata.update(
                    {
                        "parent_prompt_index": parent_index,
                        "sample_index": sample_index,
                        "rollout_kind": "full_trajectory",
                    }
                )
                expanded_prompts.append(prompt)
                expanded_metadata.append(sample_metadata)
                parent_indices.append(parent_index)

        adapter_config = self.runtime_config(
            context,
            samples_per_prompt=1,
            num_video_per_prompt=1,
            num_videos_per_prompt=1,
        )
        batch = adapter.sample(expanded_prompts, expanded_metadata, adapter_config)
        if len(batch.prompts) != len(expanded_prompts):
            raise ValueError(
                "Adapter returned a different batch size after full-trajectory expansion"
            )
        batch.model_metadata.update(
            {
                "rollout": "full_trajectory",
                "samples_per_prompt": samples_per_prompt,
                "parent_prompt_indices": parent_indices,
            }
        )
        return self.finalize_batch(
            batch,
            context,
            media_type=getattr(adapter, "media_type", None),
        )


def build_rollout_engine(config: dict[str, Any]) -> RolloutEngine:
    from visual_rl.builtins import register_builtin_plugins

    register_builtin_plugins()
    engine_cls = ROLLOUT_ENGINES.get(config.get("name", "full_trajectory"))
    engine = engine_cls(config)
    if not isinstance(engine, RolloutEngine):
        raise TypeError(
            f"Rollout engine {config.get('name')!r} must implement RolloutEngine"
        )
    return engine


ROLLOUT_ENGINES.register("full_trajectory", FullTrajectoryRollout)
