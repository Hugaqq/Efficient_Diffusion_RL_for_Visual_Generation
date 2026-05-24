"""Wan/World-R1 adapter skeleton with lazy legacy imports."""

from __future__ import annotations

from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.diffusers_common import require_model_path, resolve_torch_dtype
from visual_rl.third_party.legacy import legacy_repo_path, resolve_legacy_repo


class WorldR1WanLegacyAdapter(ModelAdapter):
    """Thin v0.1 bridge for the World-R1 Wan implementation.

    The real heavy training path is intentionally lazy so `import visual_rl` and
    mock smoke tests do not require CUDA, diffusers, or Wan checkpoints.
    """

    name = "world_r1_wan_legacy"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.repo_root = resolve_legacy_repo(config.get("repo_root", "reference_code/World-R1-main"))
        self.pipeline = None
        self.transformer = None

    def parameters(self):
        if self.transformer is None:
            raise RuntimeError("WorldR1WanLegacyAdapter must be loaded before parameters()")
        return self.transformer.parameters()

    def load(self):
        model_path = require_model_path(self.config, self.name)
        from_pretrained_kwargs: dict[str, Any] = {
            "local_files_only": bool(self.config.get("local_files_only", True)),
        }
        dtype = resolve_torch_dtype(self.config.get("torch_dtype") or self.config.get("dtype"))
        if dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = dtype
        if "low_cpu_mem_usage" in self.config:
            from_pretrained_kwargs["low_cpu_mem_usage"] = bool(self.config["low_cpu_mem_usage"])

        with legacy_repo_path(self.repo_root):
            from diffusers import WanPipeline

            self.pipeline = WanPipeline.from_pretrained(model_path, **from_pretrained_kwargs)
            device = str(self.config.get("device", "")).strip()
            if device:
                self.pipeline = self.pipeline.to(device)
            self.transformer = self.pipeline.transformer
        return self

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        del prompts, metadata, rollout_config
        raise NotImplementedError(
            "v0.1 exposes the legacy Wan adapter and launcher, but full heavy rollout "
            "is delegated to reference_code/World-R1-main/scripts/run_training.sh until GPU smoke is run."
        )

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        del batch
        raise NotImplementedError("Use the World-R1 legacy training path for real Wan logprob in v0.1.")


MODEL_ADAPTERS.register("world_r1_wan_legacy", WorldR1WanLegacyAdapter)
