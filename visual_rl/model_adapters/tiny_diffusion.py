"""Tiny trainable diffusion-like image adapter for local RL tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter


class TinyDiffusionAdapter(ModelAdapter):
    name = "tiny_diffusion"
    media_type = "image"

    def __init__(self, config: dict[str, Any]):
        import torch

        self.config = config
        extra = config.get("extra", {})
        self.image_size = int(config.get("image_size", config.get("extra", {}).get("image_size", 16)))
        self.channels = int(config.get("channels", config.get("extra", {}).get("channels", 3)))
        if self.channels != 3:
            raise ValueError("TinyDiffusionAdapter currently expects 3 channels for prompt_color reward.")
        self.device = torch.device(config.get("device", extra.get("device", "cpu")))
        self.color_bias = torch.nn.Parameter(torch.zeros(self.channels, device=self.device))

    def parameters(self):
        return [self.color_bias]

    def named_parameters(self):
        return [("color_bias", self.color_bias)]

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        import torch

        batch_size = len(prompts)
        num_steps = int(rollout_config.get("num_steps", 4))
        seed = rollout_config.get("seed")
        generator_device = self.device if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed)) if seed is not None else None
        shape = (batch_size, num_steps, self.channels, self.image_size, self.image_size)
        latents = torch.randn(shape, generator=generator, device=self.device) * 0.25
        noise = torch.randn(shape, generator=generator, device=self.device) * 0.05
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

    def sample_branching(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> RolloutBatch:
        import torch

        if len(prompts) != len(metadata):
            raise ValueError("prompts and metadata must have the same length")
        num_steps = int(rollout_config.get("num_steps", 4))
        branch_step_index = int(rollout_config["branch_step_index"])
        branch_count = int(rollout_config["branch_count"])
        include_main = bool(rollout_config.get("include_main", False))
        group_size = branch_count + int(include_main)
        seed = rollout_config.get("seed")
        generator_device = self.device if self.device.type == "cuda" else "cpu"
        generator = (
            torch.Generator(device=generator_device).manual_seed(int(seed))
            if seed is not None
            else None
        )
        timestep_values = rollout_config.get("timestep_values", range(num_steps))
        timestep_values = torch.as_tensor(list(timestep_values), dtype=torch.long)
        if timestep_values.numel() != num_steps:
            raise ValueError("timestep_values must contain one value per denoising step")

        sample_shape = (self.channels, self.image_size, self.image_size)
        parent_initial = torch.randn(
            len(prompts), *sample_shape, generator=generator, device=self.device
        ) * 0.25
        prefix_states = [parent_initial]
        bias = self.color_bias.detach().view(1, self.channels, 1, 1)
        for _step in range(branch_step_index):
            noise = torch.randn(
                len(prompts), *sample_shape, generator=generator, device=self.device
            ) * 0.05
            prefix_states.append(prefix_states[-1] + noise + bias)

        rows = len(prompts) * group_size
        latents = torch.empty(rows, num_steps, *sample_shape, device=self.device)
        next_latents = torch.empty_like(latents)
        expanded_prompts: list[str] = []
        expanded_metadata: list[dict[str, Any]] = []
        branch_ids: list[int] = []
        row = 0
        ids = ([-1] if include_main else []) + list(range(branch_count))
        branch_timestep_value = int(timestep_values[branch_step_index])
        for parent_index, (prompt, item) in enumerate(
            zip(prompts, metadata, strict=True)
        ):
            for branch_id in ids:
                for step in range(branch_step_index):
                    latents[row, step] = prefix_states[step][parent_index]
                    next_latents[row, step] = prefix_states[step + 1][parent_index]
                state = prefix_states[-1][parent_index]
                for step in range(branch_step_index, num_steps):
                    noise = torch.randn(
                        sample_shape, generator=generator, device=self.device
                    ) * 0.05
                    next_state = state + noise + self.color_bias.detach().view(
                        self.channels, 1, 1
                    )
                    latents[row, step] = state
                    next_latents[row, step] = next_state
                    state = next_state
                expanded_prompts.append(prompt)
                branch_metadata = dict(item)
                branch_metadata.update(
                    {
                        "parent_prompt_index": parent_index,
                        "branch_id": branch_id,
                        "branch_step_index": branch_step_index,
                        "branch_timestep_value": branch_timestep_value,
                        "is_main_branch": branch_id == -1,
                        "rollout_kind": "tempflow_branching",
                    }
                )
                expanded_metadata.append(branch_metadata)
                branch_ids.append(branch_id)
                row += 1

        detached_bias = self.color_bias.detach().view(1, 1, self.channels, 1, 1)
        old_log_probs = -(
            (next_latents - latents - detached_bias) ** 2
        ).mean(dim=(2, 3, 4)).detach()
        timesteps = timestep_values.repeat(rows, 1)
        media = torch.sigmoid(next_latents[:, -1]).detach()
        batch = RolloutBatch(
            prompts=expanded_prompts,
            metadata=expanded_metadata,
            media=media,
            latents=latents.detach(),
            next_latents=next_latents.detach(),
            timesteps=timesteps,
            old_log_probs=old_log_probs,
            kl=torch.zeros(rows, num_steps),
            branch_ids=torch.as_tensor(branch_ids, dtype=torch.long),
            epoch_tag=rollout_config.get("epoch_tag"),
            seed=seed,
            model_metadata={
                "adapter": self.name,
                "image_size": self.image_size,
                "branching_mode": "shared_prefix",
                "branch_step_index": branch_step_index,
                "branch_timestep_value": branch_timestep_value,
            },
        )
        batch.validate_lightweight(strict=True)
        return batch

    def recompute_log_probs(self, batch: RolloutBatch):
        bias = self.color_bias.view(1, 1, self.channels, 1, 1)
        return -((batch.next_latents - batch.latents - bias) ** 2).mean(dim=(2, 3, 4))

    def save_pretrained(self, output_dir: str) -> None:
        import torch

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"color_bias": self.color_bias.detach().cpu()}, path / "tiny_diffusion.pt")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        import torch

        state = torch.load(
            Path(checkpoint_dir) / "tiny_diffusion.pt",
            map_location=self.device,
            weights_only=False,
        )
        with torch.no_grad():
            self.color_bias.copy_(state["color_bias"].to(self.device))


MODEL_ADAPTERS.register("tiny_diffusion", TinyDiffusionAdapter)
