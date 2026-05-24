"""SD3/SD3.5 TempFlow adapter backed by the verified reference path."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.diffusers_common import (
    AdapterNotLoadedError,
    apply_peft_lora,
    make_generator,
    module_or_self,
    require_model_path,
    resolve_torch_dtype,
    stack_steps,
    trainable_parameters,
)
from visual_rl.third_party.legacy import legacy_repo_path, resolve_legacy_repo


DEFAULT_SD3_LORA_TARGETS = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
]


class SD3TempFlowAdapter(ModelAdapter):
    """In-process wrapper around TempFlow-GRPO's SD3 patched pipeline."""

    name = "tempflow_sd3_legacy"

    def __init__(self, config: dict[str, Any]):
        import torch

        self.config = config
        extra = dict(config.get("extra", {}))
        self.repo_root = resolve_legacy_repo(config.get("repo_root", extra.get("repo_root", "reference_code/TempFlow-GRPO-main")))
        self.device = torch.device(config.get("device", extra.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
        self.dtype = resolve_torch_dtype(config.get("dtype", extra.get("dtype", "bfloat16" if self.device.type == "cuda" else "float32")))
        self.resolution = int(config.get("resolution", extra.get("resolution", 256)))
        self.max_sequence_length = int(extra.get("max_sequence_length", 128))
        self.use_lora = bool(config.get("use_lora", extra.get("use_lora", True)))
        self.lora_rank = int(extra.get("lora_rank", 32))
        self.lora_alpha = int(extra.get("lora_alpha", self.lora_rank * 2))
        self.lora_path = config.get("lora_path", extra.get("lora_path"))
        self.lora_targets = list(extra.get("lora_target_modules", DEFAULT_SD3_LORA_TARGETS))
        self.pipeline = None
        self.transformer = None
        self._pipeline_with_logprob = None
        self._sde_step_with_logprob = None
        self._encode_prompt = None
        if not extra.get("defer_load", False):
            self.load()

    def load(self) -> None:
        import torch

        try:
            from diffusers import StableDiffusion3Pipeline
        except ImportError as exc:  # pragma: no cover - optional train dependency
            raise ImportError("Install visual-rl[train] to use SD3 TempFlow adapter.") from exc

        model_path = require_model_path(self.config, self.name)
        with legacy_repo_path(self.repo_root):
            from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
            from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
            from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

        self._pipeline_with_logprob = pipeline_with_logprob
        self._sde_step_with_logprob = sde_step_with_logprob
        self._encode_prompt = encode_prompt
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=self.dtype)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        self.pipeline.text_encoder_3.requires_grad_(False)
        if self.use_lora:
            self.pipeline.transformer = apply_peft_lora(
                self.pipeline.transformer,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                target_modules=self.lora_targets,
                lora_path=self.lora_path,
            )
        else:
            self.pipeline.transformer.requires_grad_(True)
        self.pipeline.to(self.device)
        if self.dtype is not None and self.device.type == "cuda":
            self.pipeline.text_encoder.to(self.device, dtype=self.dtype)
            self.pipeline.text_encoder_2.to(self.device, dtype=self.dtype)
            self.pipeline.text_encoder_3.to(self.device, dtype=self.dtype)
            self.pipeline.transformer.to(self.device)
        self.pipeline.vae.to(self.device, dtype=torch.float32)
        self.transformer = self.pipeline.transformer

    def _ensure_loaded(self) -> None:
        if self.pipeline is None or self.transformer is None:
            raise AdapterNotLoadedError("SD3 TempFlow adapter is not loaded. Provide model_path or call load().")

    def parameters(self):
        self._ensure_loaded()
        params = trainable_parameters(self.transformer)
        if not params:
            raise AdapterNotLoadedError("SD3 TempFlow adapter has no trainable parameters.")
        return params

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        import torch

        self._ensure_loaded()
        num_steps = int(rollout_config.get("num_steps", 3))
        guidance_scale = float(rollout_config.get("guidance_scale", 4.5))
        output_type = str(rollout_config.get("output_type", "pt"))
        generator = make_generator(self.device, rollout_config.get("seed"))

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_text(prompts)
            neg_prompt_embeds, neg_pooled_prompt_embeds = self._encode_text([""] * len(prompts))
            result = self._call_pipeline_with_logprob(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type=output_type,
                height=self.resolution,
                width=self.resolution,
                kl_reward=float(rollout_config.get("kl_reward", 0.0)),
            )
            if len(result) == 4:
                images, latents, log_probs, kls = result
            elif len(result) == 3:
                images, latents, log_probs = result
                kls = []
            else:
                raise ValueError(f"Unexpected SD3 TempFlow pipeline result length: {len(result)}")
            latents = stack_steps(latents, dim=1)
            log_probs = stack_steps(log_probs, dim=1)
            kls = stack_steps(kls, dim=1) if kls else torch.zeros_like(log_probs)

        return RolloutBatch(
            prompts=list(prompts),
            metadata=[dict(item) for item in metadata],
            media=images.detach() if isinstance(images, torch.Tensor) else images,
            latents=latents[:, :-1].detach(),
            next_latents=latents[:, 1:].detach(),
            timesteps=self.pipeline.scheduler.timesteps.repeat(len(prompts), 1).detach(),
            old_log_probs=log_probs.detach(),
            kl=kls.detach(),
            epoch_tag=rollout_config.get("epoch_tag"),
            seed=rollout_config.get("seed"),
            model_metadata={
                "adapter": self.name,
                "reference_repo": str(self.repo_root),
                "reference_pipeline": "sd3_pipeline_with_logprob",
                "resolution": self.resolution,
                "guidance_scale": guidance_scale,
            },
            model_tensors={
                "prompt_embeds": prompt_embeds.detach(),
                "pooled_prompt_embeds": pooled_prompt_embeds.detach(),
                "negative_prompt_embeds": neg_prompt_embeds.detach(),
                "negative_pooled_prompt_embeds": neg_pooled_prompt_embeds.detach(),
            },
        )

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        import torch

        self._ensure_loaded()
        guidance_scale = float(batch.model_metadata.get("guidance_scale", 4.5))
        prompt_embeds, pooled_prompt_embeds = self._batch_embeddings(batch, positive=True)
        neg_prompt_embeds, neg_pooled_prompt_embeds = self._batch_embeddings(batch, positive=False)
        latents = batch.latents.to(self.device)
        next_latents = batch.next_latents.to(self.device)
        timesteps = batch.timesteps.to(self.device)
        log_probs = []
        transformer = module_or_self(self.transformer)
        transformer_dtype = self._transformer_dtype()
        for index in range(latents.shape[1]):
            latent_step = latents[:, index].to(dtype=transformer_dtype)
            if guidance_scale > 1.0:
                hidden_states = torch.cat([latent_step] * 2)
                timestep = torch.cat([timesteps[:, index]] * 2)
                embeds = torch.cat([neg_prompt_embeds, prompt_embeds])
                pooled = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds])
                noise_pred = transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=embeds,
                    pooled_projections=pooled,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = transformer(
                    hidden_states=latent_step,
                    timestep=timesteps[:, index],
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
            _prev_sample, log_prob, _mean, _std = self._sde_step_with_logprob(
                self.pipeline.scheduler,
                noise_pred.float(),
                timesteps[:, index],
                latents[:, index].float(),
                prev_sample=next_latents[:, index].float(),
            )
            log_probs.append(log_prob)
        return torch.stack(log_probs, dim=1)

    def save_pretrained(self, output_dir: str) -> None:
        self._ensure_loaded()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.transformer, "save_pretrained"):
            self.transformer.save_pretrained(path)
        else:
            import torch

            torch.save(self.transformer.state_dict(), path / "sd3_transformer.pt")

    def _call_pipeline_with_logprob(self, **kwargs):
        signature = inspect.signature(self._pipeline_with_logprob)
        supported = set(signature.parameters)
        call_kwargs = {key: value for key, value in kwargs.items() if key in supported}
        if "return_dict" in supported:
            call_kwargs["return_dict"] = False
        return self._pipeline_with_logprob(self.pipeline, **call_kwargs)

    def _encode_text(self, prompts: list[str]):
        embeds, pooled = self._encode_prompt(
            [self.pipeline.text_encoder, self.pipeline.text_encoder_2, self.pipeline.text_encoder_3],
            [self.pipeline.tokenizer, self.pipeline.tokenizer_2, self.pipeline.tokenizer_3],
            prompts,
            self.max_sequence_length,
        )
        return embeds.to(self.device), pooled.to(self.device)

    def _transformer_dtype(self):
        import torch

        return next(self.transformer.parameters(), torch.empty((), dtype=torch.float32)).dtype

    def _batch_embeddings(self, batch: RolloutBatch, *, positive: bool):
        if positive:
            embed_key = "prompt_embeds"
            pooled_key = "pooled_prompt_embeds"
            prompt_text = batch.prompts
        else:
            embed_key = "negative_prompt_embeds"
            pooled_key = "negative_pooled_prompt_embeds"
            prompt_text = [""] * len(batch.prompts)
        if embed_key in batch.model_tensors and pooled_key in batch.model_tensors:
            return (
                batch.model_tensors[embed_key].to(self.device),
                batch.model_tensors[pooled_key].to(self.device),
            )
        return self._encode_text(prompt_text)


MODEL_ADAPTERS.register("tempflow_sd3_legacy", SD3TempFlowAdapter)
MODEL_ADAPTERS.register("sd3_tempflow", SD3TempFlowAdapter)
