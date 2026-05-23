"""Wan/World-R1 adapter skeleton with lazy legacy imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.third_party.legacy import legacy_repo_path


class WorldR1WanLegacyAdapter(ModelAdapter):
    """Thin v0.1 bridge for the World-R1 Wan implementation.

    The real heavy training path is intentionally lazy so `import visual_rl` and
    mock smoke tests do not require CUDA, diffusers, or Wan checkpoints.
    """

    name = "world_r1_wan_legacy"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.repo_root = Path(config.get("repo_root", "World-R1-main"))
        self.pipeline = None
        self.transformer = None

    def parameters(self):
        if self.transformer is None:
            raise RuntimeError("WorldR1WanLegacyAdapter must be loaded before parameters()")
        return self.transformer.parameters()

    def load(self):
        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("model.model_path is required for world_r1_wan_legacy")
        with legacy_repo_path(self.repo_root):
            from diffusers import WanPipeline

            self.pipeline = WanPipeline.from_pretrained(model_path)
            self.transformer = self.pipeline.transformer
        return self

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        del prompts, metadata, rollout_config
        raise NotImplementedError(
            "v0.1 exposes the legacy Wan adapter and launcher, but full heavy rollout "
            "is delegated to World-R1-main/scripts/run_training.sh until GPU smoke is run."
        )

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        del batch
        raise NotImplementedError("Use the World-R1 legacy training path for real Wan logprob in v0.1.")


MODEL_ADAPTERS.register("world_r1_wan_legacy", WorldR1WanLegacyAdapter)

