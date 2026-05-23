"""Stable Diffusion 1.5 LoRA adapter for small image RL probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.diffusers_common import (
    AdapterNotLoadedError,
    apply_peft_lora,
    make_generator,
    require_model_path,
    resolve_torch_dtype,
    surrogate_transition_log_prob,
    trainable_parameters,
)


DEFAULT_SD15_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


class SD15LoRAAdapter(ModelAdapter):
    """A low-resolution SD1.5 adapter with LoRA trainable UNet parameters.

    The transition logprob is a DDIM-style surrogate around the scheduler
    predicted previous latent. This is intended for adapter/trainer correctness
    checks before heavier SD3/FLUX/QwenImage integration.
    """

    name = "sd15_lora"

    def __init__(self, config: dict[str, Any]):
        import torch

        self.config = config
        extra = dict(config.get("extra", {}))
        self.device = torch.device(config.get("device", extra.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
        self.dtype = resolve_torch_dtype(config.get("dtype", extra.get("dtype", "float16" if self.device.type == "cuda" else "float32")))
        self.resolution = int(config.get("resolution", extra.get("resolution", 256)))
        self.logprob_std = float(extra.get("logprob_std", 0.1))
        self.eta = float(extra.get("eta", 0.0))
        self.negative_prompt = str(extra.get("negative_prompt", ""))
        self.use_lora = bool(config.get("use_lora", extra.get("use_lora", True)))
        self.lora_rank = int(extra.get("lora_rank", 4))
        self.lora_alpha = int(extra.get("lora_alpha", self.lora_rank * 2))
        self.lora_path = config.get("lora_path", extra.get("lora_path"))
        self.lora_targets = list(extra.get("lora_target_modules", DEFAULT_SD15_LORA_TARGETS))
        self.pipe = None
        self.unet = None
        if not extra.get("defer_load", False):
            self.load()

    def load(self) -> None:
        try:
            from diffusers import DDIMScheduler, StableDiffusionPipeline
        except ImportError as exc:  # pragma: no cover - optional train dependency
            raise ImportError("Install visual-rl[train] to use SD1.5 LoRA adapter.") from exc

        model_path = require_model_path(self.config, self.name)
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=self.dtype)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        if self.use_lora:
            self.pipe.unet = apply_peft_lora(
                self.pipe.unet,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                target_modules=self.lora_targets,
                lora_path=self.lora_path,
            )
        else:
            self.pipe.unet.requires_grad_(True)
        self.pipe.to(self.device)
        self.unet = self.pipe.unet
        if self.dtype is not None and self.device.type == "cuda":
            self.unet.to(self.device, dtype=self.dtype)

    def _ensure_loaded(self) -> None:
        if self.pipe is None or self.unet is None:
            raise AdapterNotLoadedError("SD1.5 adapter is not loaded. Provide model_path or call load().")

    def parameters(self):
        self._ensure_loaded()
        params = trainable_parameters(self.unet)
        if not params:
            raise AdapterNotLoadedError("SD1.5 adapter has no trainable parameters.")
        return params

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        import torch

        self._ensure_loaded()
        num_steps = int(rollout_config.get("num_steps", 10))
        guidance_scale = float(rollout_config.get("guidance_scale", 7.5))
        generator = make_generator(self.device, rollout_config.get("seed"))
        batch_size = len(prompts)

        with torch.no_grad():
            prompt_embeds = self._encode_prompts(prompts, guidance_scale)
            latents = self._initial_latents(batch_size, generator)
            self.pipe.scheduler.set_timesteps(num_steps, device=self.device)
            latent_history = []
            next_history = []
            log_probs = []
            for timestep in self.pipe.scheduler.timesteps:
                latent_history.append(latents.detach())
                noise_pred = self._predict_noise(latents, timestep, prompt_embeds, guidance_scale)
                mean_prev = self.pipe.scheduler.step(noise_pred, timestep, latents, eta=self.eta).prev_sample
                log_prob = surrogate_transition_log_prob(mean_prev, mean_prev, self.logprob_std)
                latents = mean_prev
                next_history.append(latents.detach())
                log_probs.append(log_prob.detach())
            media = self._decode_latents(latents).detach()

        return RolloutBatch(
            prompts=list(prompts),
            metadata=[dict(item) for item in metadata],
            media=media,
            latents=torch.stack(latent_history, dim=1).detach(),
            next_latents=torch.stack(next_history, dim=1).detach(),
            timesteps=self.pipe.scheduler.timesteps.repeat(batch_size, 1).detach().cpu(),
            old_log_probs=torch.stack(log_probs, dim=1).detach(),
            kl=torch.zeros(batch_size, num_steps, device=self.device),
            epoch_tag=rollout_config.get("epoch_tag"),
            seed=rollout_config.get("seed"),
            model_metadata={
                "adapter": self.name,
                "resolution": self.resolution,
                "guidance_scale": guidance_scale,
                "logprob": "ddim_surrogate",
                "logprob_std": self.logprob_std,
            },
        )

    def recompute_log_probs(self, batch: RolloutBatch):
        import torch

        self._ensure_loaded()
        guidance_scale = float(batch.model_metadata.get("guidance_scale", self.config.get("guidance_scale", 7.5)))
        prompt_embeds = self._encode_prompts(batch.prompts, guidance_scale)
        latents = batch.latents.to(self.device, dtype=self._unet_dtype())
        next_latents = batch.next_latents.to(self.device, dtype=self._unet_dtype())
        timesteps = batch.timesteps.to(self.device)
        log_probs = []
        for index in range(latents.shape[1]):
            timestep = timesteps[:, index]
            timestep_value = timestep[0] if timestep.ndim > 0 else timestep
            noise_pred = self._predict_noise(latents[:, index], timestep_value, prompt_embeds, guidance_scale)
            mean_prev = self.pipe.scheduler.step(noise_pred, timestep_value, latents[:, index], eta=self.eta).prev_sample
            log_probs.append(surrogate_transition_log_prob(next_latents[:, index], mean_prev, self.logprob_std))
        return torch.stack(log_probs, dim=1)

    def save_pretrained(self, output_dir: str) -> None:
        self._ensure_loaded()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.unet, "save_pretrained"):
            self.unet.save_pretrained(path)
        else:
            import torch

            torch.save({name: value.detach().cpu() for name, value in self.unet.state_dict().items()}, path / "unet.pt")

    def _encode_prompts(self, prompts: list[str], guidance_scale: float):
        import torch

        do_cfg = guidance_scale > 1.0
        if hasattr(self.pipe, "_encode_prompt"):
            return self.pipe._encode_prompt(  # noqa: SLF001 - Diffusers exposes this helper across SD1.x versions
                prompts,
                self.device,
                1,
                do_cfg,
                negative_prompt=[self.negative_prompt] * len(prompts),
            )
        encoded = self.pipe.encode_prompt(
            prompt=prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=[self.negative_prompt] * len(prompts),
        )
        prompt_embeds, negative_prompt_embeds = encoded[:2]
        return torch.cat([negative_prompt_embeds, prompt_embeds]) if do_cfg else prompt_embeds

    def _initial_latents(self, batch_size: int, generator):
        import torch

        latent_channels = int(self.unet.config.in_channels)
        scale = int(self.pipe.vae_scale_factor)
        shape = (batch_size, latent_channels, self.resolution // scale, self.resolution // scale)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self._unet_dtype())
        return latents * self.pipe.scheduler.init_noise_sigma

    def _predict_noise(self, latents, timestep, prompt_embeds, guidance_scale: float):
        import torch

        do_cfg = guidance_scale > 1.0
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample
        if not do_cfg:
            return noise_pred
        noise_uncond, noise_text = noise_pred.chunk(2)
        return noise_uncond + guidance_scale * (noise_text - noise_uncond)

    def _decode_latents(self, latents):
        image_latents = latents / self.pipe.vae.config.scaling_factor
        images = self.pipe.vae.decode(image_latents.to(dtype=self.pipe.vae.dtype), return_dict=False)[0]
        return ((images.float() + 1.0) / 2.0).clamp(0.0, 1.0)

    def _unet_dtype(self):
        import torch

        return next(self.unet.parameters(), torch.empty((), dtype=torch.float32)).dtype


MODEL_ADAPTERS.register("sd15_lora", SD15LoRAAdapter)
