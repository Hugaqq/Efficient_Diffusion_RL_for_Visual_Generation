"""Tiny trainable adapter used for local v0.1 dry-runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter


class MockWanAdapter(ModelAdapter):
    name = "mock_wan"

    def __init__(self, config: dict[str, Any]):
        import torch

        self.config = config
        self.latent_shape = tuple(config.get("latent_shape", [4, 2, 2, 2]))
        self.media_shape = tuple(config.get("media_shape", [4, 3, 16, 16]))
        self.policy_bias = torch.nn.Parameter(torch.tensor(0.0))

    def parameters(self):
        return [self.policy_bias]

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        import torch

        batch_size = len(prompts)
        num_steps = int(rollout_config.get("num_steps", 2))
        generator = None
        seed = rollout_config.get("seed")
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        latents = torch.randn(batch_size, num_steps, *self.latent_shape, generator=generator)
        next_latents = latents + 0.05 * torch.randn(latents.shape, generator=generator)
        timesteps = torch.arange(num_steps).repeat(batch_size, 1)
        with torch.no_grad():
            old_log_probs = -((next_latents - latents) ** 2).mean(dim=tuple(range(2, next_latents.ndim)))
        media = torch.rand(batch_size, *self.media_shape, generator=generator)
        return RolloutBatch(
            prompts=list(prompts),
            metadata=[dict(item) for item in metadata],
            media=media,
            latents=latents,
            next_latents=next_latents,
            timesteps=timesteps,
            old_log_probs=old_log_probs,
            kl=torch.zeros(batch_size, num_steps),
            epoch_tag=rollout_config.get("epoch_tag"),
            seed=seed,
            model_metadata={"adapter": self.name},
        )

    def recompute_log_probs(self, batch: RolloutBatch):
        delta = batch.next_latents - batch.latents - self.policy_bias
        return -(delta**2).mean(dim=tuple(range(2, delta.ndim)))

    def save_pretrained(self, output_dir: str) -> None:
        import torch

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"policy_bias": self.policy_bias.detach().cpu()}, path / "mock_adapter.pt")


MODEL_ADAPTERS.register("mock_wan", MockWanAdapter)
