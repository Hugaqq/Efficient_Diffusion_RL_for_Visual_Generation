"""FLUX TempFlow adapter backed by the TempFlow-GRPO reference patches."""

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
    module_or_self,
    require_model_path,
    resolve_torch_dtype,
    stack_steps,
    trainable_parameters,
)
from visual_rl.third_party.legacy import legacy_repo_path, resolve_legacy_repo


DEFAULT_FLUX_LORA_TARGETS = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


class FluxTempFlowAdapter(ModelAdapter):
    name = "tempflow_flux_legacy"

    def __init__(self, config: dict[str, Any]):
        import torch

        self.config = config
        extra = dict(config.get("extra", {}))
        self.repo_root = resolve_legacy_repo(config.get("repo_root", extra.get("repo_root", "reference_code/TempFlow-GRPO-main")))
        self.device = torch.device(config.get("device", extra.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
        self.dtype = resolve_torch_dtype(config.get("dtype", extra.get("dtype", "bfloat16" if self.device.type == "cuda" else "float32")))
        self.resolution = int(config.get("resolution", extra.get("resolution", 256)))
        self.max_sequence_length = int(extra.get("max_sequence_length", 256))
        self.use_lora = bool(config.get("use_lora", extra.get("use_lora", True)))
        self.lora_rank = int(extra.get("lora_rank", 32))
        self.lora_alpha = int(extra.get("lora_alpha", self.lora_rank * 2))
        self.lora_path = config.get("lora_path", extra.get("lora_path"))
        self.lora_targets = list(extra.get("lora_target_modules", DEFAULT_FLUX_LORA_TARGETS))
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
            from diffusers import FluxPipeline
        except ImportError as exc:  # pragma: no cover - optional train dependency
            raise ImportError("Install visual-rl[train] to use FLUX TempFlow adapter.") from exc

        model_path = require_model_path(self.config, self.name)
        with legacy_repo_path(self.repo_root):
            from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
            from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
            from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt

        self._pipeline_with_logprob = pipeline_with_logprob
        self._sde_step_with_logprob = sde_step_with_logprob
        self._encode_prompt = encode_prompt
        self.pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=self.dtype)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
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
            self.pipeline.transformer.to(self.device)
        self.pipeline.vae.to(self.device, dtype=torch.float32)
        self.transformer = self.pipeline.transformer

    def _ensure_loaded(self) -> None:
        if self.pipeline is None or self.transformer is None:
            raise AdapterNotLoadedError("FLUX TempFlow adapter is not loaded. Provide model_path or call load().")

    def parameters(self):
        self._ensure_loaded()
        params = trainable_parameters(self.transformer)
        if not params:
            raise AdapterNotLoadedError("FLUX TempFlow adapter has no trainable parameters.")
        return params

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        import torch

        self._ensure_loaded()
        num_steps = int(rollout_config.get("num_steps", 3))
        guidance_scale = float(rollout_config.get("guidance_scale", 3.5))
        output_type = str(rollout_config.get("output_type", "pt"))
        generator = make_generator(self.device, rollout_config.get("seed"))
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_text(prompts)
            images, latents, image_ids, text_ids, log_probs = self._pipeline_with_logprob(
                self.pipeline,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type=output_type,
                height=self.resolution,
                width=self.resolution,
                determistic=False,
            )
            latents = stack_steps(latents, dim=1)
            log_probs = stack_steps(log_probs, dim=1)

        return RolloutBatch(
            prompts=list(prompts),
            metadata=[dict(item) for item in metadata],
            media=images.detach() if isinstance(images, torch.Tensor) else images,
            latents=latents[:, :-1].detach(),
            next_latents=latents[:, 1:].detach(),
            timesteps=self.pipeline.scheduler.timesteps.repeat(len(prompts), 1).detach(),
            old_log_probs=log_probs.detach(),
            epoch_tag=rollout_config.get("epoch_tag"),
            seed=rollout_config.get("seed"),
            model_metadata={
                "adapter": self.name,
                "reference_repo": str(self.repo_root),
                "reference_pipeline": "flux_pipeline_with_logprob",
                "resolution": self.resolution,
                "guidance_scale": guidance_scale,
            },
            model_tensors={
                "prompt_embeds": prompt_embeds.detach(),
                "pooled_prompt_embeds": pooled_prompt_embeds.detach(),
                "image_ids": image_ids.detach(),
                "text_ids": text_ids.detach(),
            },
        )

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        import torch

        self._ensure_loaded()
        guidance_scale = float(batch.model_metadata.get("guidance_scale", 3.5))
        prompt_embeds, pooled_prompt_embeds = self._batch_embeddings(batch)
        latents = batch.latents.to(self.device)
        next_latents = batch.next_latents.to(self.device)
        timesteps = batch.timesteps.to(self.device)
        image_ids = batch.model_tensors["image_ids"].to(self.device)
        text_ids = batch.model_tensors.get("text_ids")
        text_ids = text_ids.to(self.device) if text_ids is not None else torch.zeros(prompt_embeds.shape[1], 3, device=self.device)
        img_ids = image_ids[0] if image_ids.ndim == 3 else image_ids
        transformer = module_or_self(self.transformer)
        guidance = None
        transformer_config = getattr(transformer, "config", None)
        if getattr(transformer_config, "guidance_embeds", False):
            guidance = torch.full((latents.shape[0],), guidance_scale, device=self.device)
        log_probs = []
        for index in range(latents.shape[1]):
            model_pred = transformer(
                hidden_states=latents[:, index],
                timestep=timesteps[:, index] / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
            _prev_sample, log_prob, _mean, _std = self._sde_step_with_logprob(
                self.pipeline.scheduler,
                model_pred.float(),
                timesteps[:, index],
                latents[:, index].float(),
                prev_sample=next_latents[:, index].float(),
                determistic=False,
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

            torch.save(self.transformer.state_dict(), path / "flux_transformer.pt")

    def _encode_text(self, prompts: list[str]):
        prompt_embeds, pooled_prompt_embeds, _text_ids = self._encode_prompt(
            [self.pipeline.text_encoder, self.pipeline.text_encoder_2],
            [self.pipeline.tokenizer, self.pipeline.tokenizer_2],
            prompts,
            self.max_sequence_length,
        )
        return prompt_embeds.to(self.device), pooled_prompt_embeds.to(self.device)

    def _batch_embeddings(self, batch: RolloutBatch):
        if "prompt_embeds" in batch.model_tensors and "pooled_prompt_embeds" in batch.model_tensors:
            return (
                batch.model_tensors["prompt_embeds"].to(self.device),
                batch.model_tensors["pooled_prompt_embeds"].to(self.device),
            )
        return self._encode_text(batch.prompts)


MODEL_ADAPTERS.register("tempflow_flux_legacy", FluxTempFlowAdapter)
MODEL_ADAPTERS.register("flux_tempflow", FluxTempFlowAdapter)
