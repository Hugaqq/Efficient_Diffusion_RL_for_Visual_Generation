"""Tiny trainable diffusion-like image adapter for local RL tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter


class TinyDiffusionAdapter(ModelAdapter):
    name = "tiny_diffusion"

    def __init__(self, config: dict[str, Any]):
        import torch

        self.config = config
        self.image_size = int(config.get("image_size", config.get("extra", {}).get("image_size", 16)))
        self.channels = int(config.get("channels", config.get("extra", {}).get("channels", 3)))
        if self.channels != 3:
            raise ValueError("TinyDiffusionAdapter currently expects 3 channels for prompt_color reward.")
        self.color_bias = torch.nn.Parameter(torch.zeros(self.channels))

    def parameters(self):
        return [self.color_bias]

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        import torch

        batch_size = len(prompts)
        num_steps = int(rollout_config.get("num_steps", 4))
        seed = rollout_config.get("seed")
        generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        shape = (batch_size, num_steps, self.channels, self.image_size, self.image_size)
        latents = torch.randn(shape, generator=generator) * 0.25
        noise = torch.randn(shape, generator=generator) * 0.05
        bias = self.color_bias.detach().view(1, 1, self.channels, 1, 1)
        next_latents = latents + noise + bias
        old_log_probs = -((next_latents - latents - bias) ** 2).mean(dim=(2, 3, 4)).detach()
        timesteps = torch.arange(num_steps).repeat(batch_size, 1)
        media = torch.sigmoid(next_latents[:, -1]).detach()

        return RolloutBatch(
            prompts=list(prompts),
            metadata=[dict(item) for item in metadata],
            media=media,
            latents=latents.detach(),
            next_latents=next_latents.detach(),
            timesteps=timesteps,
            old_log_probs=old_log_probs,
            kl=torch.zeros(batch_size, num_steps),
            epoch_tag=rollout_config.get("epoch_tag"),
            seed=seed,
            model_metadata={"adapter": self.name, "image_size": self.image_size},
        )

    def recompute_log_probs(self, batch: RolloutBatch):
        bias = self.color_bias.view(1, 1, self.channels, 1, 1)
        return -((batch.next_latents - batch.latents - bias) ** 2).mean(dim=(2, 3, 4))

    def save_pretrained(self, output_dir: str) -> None:
        import torch

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"color_bias": self.color_bias.detach().cpu()}, path / "tiny_diffusion.pt")


MODEL_ADAPTERS.register("tiny_diffusion", TinyDiffusionAdapter)
