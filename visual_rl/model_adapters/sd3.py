"""Lazy SD3 adapter shell for TempFlow image RL."""

from __future__ import annotations

from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.third_party.legacy import resolve_legacy_repo


class SD3Adapter(ModelAdapter):
    name = "tempflow_sd3_legacy"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.repo_root = resolve_legacy_repo(config.get("repo_root", "reference_code/TempFlow-GRPO-main"))

    def parameters(self):
        raise RuntimeError("SD3 TempFlow adapter is a lazy bridge; load a real SD3 policy before training.")

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        del prompts, metadata, rollout_config
        raise NotImplementedError(
            "SD3 TempFlow rollout will be wired after tiny_diffusion and SD1.5 branching are stable."
        )

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        del batch
        raise NotImplementedError("SD3 TempFlow logprob recomputation is not wired yet.")


MODEL_ADAPTERS.register("tempflow_sd3_legacy", SD3Adapter)
