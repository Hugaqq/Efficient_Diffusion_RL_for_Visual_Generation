"""QwenImage TempFlow adapter backed by the TempFlow-GRPO reference patches."""

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


DEFAULT_QWENIMAGE_LORA_TARGETS = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]


class QwenImageTempFlowAdapter(ModelAdapter):
    name = "tempflow_qwenimage_legacy"

    def __init__(self, config: dict[str, Any]):
        import torch

        self.config = config
        extra = dict(config.get("extra", {}))
        self.repo_root = resolve_legacy_repo(config.get("repo_root", extra.get("repo_root", "reference_code/TempFlow-GRPO-main")))
        self.device = torch.device(config.get("device", extra.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
        self.dtype = resolve_torch_dtype(config.get("dtype", extra.get("dtype", "bfloat16" if self.device.type == "cuda" else "float32")))
        self.resolution = int(config.get("resolution", extra.get("resolution", 256)))
        self.use_lora = bool(config.get("use_lora", extra.get("use_lora", True)))
        self.lora_rank = int(extra.get("lora_rank", 64))
        self.lora_alpha = int(extra.get("lora_alpha", self.lora_rank * 2))
        self.lora_path = config.get("lora_path", extra.get("lora_path"))
        self.lora_targets = list(extra.get("lora_target_modules", DEFAULT_QWENIMAGE_LORA_TARGETS))
        self.pipeline = None
        self.transformer = None
        self._pipeline_with_logprob = None
        self._sde_step_with_logprob = None
        if not extra.get("defer_load", False):
            self.load()

    def load(self) -> None:
        import torch

        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:  # pragma: no cover - optional train dependency
            raise ImportError("Install visual-rl[train] to use QwenImage TempFlow adapter.") from exc

        model_path = require_model_path(self.config, self.name)
        with legacy_repo_path(self.repo_root):
            from flow_grpo.diffusers_patch.qwenimage_pipeline_with_logprob import pipeline_with_logprob
            from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob

        self._pipeline_with_logprob = pipeline_with_logprob
        self._sde_step_with_logprob = sde_step_with_logprob
        self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=self.dtype)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
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
            self.pipeline.transformer.to(self.device)
        self.pipeline.vae.to(self.device, dtype=torch.float32)
        self.transformer = self.pipeline.transformer

    def _ensure_loaded(self) -> None:
        if self.pipeline is None or self.transformer is None:
            raise AdapterNotLoadedError("QwenImage TempFlow adapter is not loaded. Provide model_path or call load().")

    def parameters(self):
        self._ensure_loaded()
        params = trainable_parameters(self.transformer)
        if not params:
            raise AdapterNotLoadedError("QwenImage TempFlow adapter has no trainable parameters.")
        return params

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        import torch

        self._ensure_loaded()
        num_steps = int(rollout_config.get("num_steps", 3))
        guidance_scale = float(rollout_config.get("guidance_scale", 4.0))
        output_type = str(rollout_config.get("output_type", "pt"))
        generator = make_generator(self.device, rollout_config.get("seed"))
        with torch.no_grad():
            collected = self._pipeline_with_logprob(
                self.pipeline,
                prompts,
                negative_prompt=[" "] * len(prompts),
                num_inference_steps=num_steps,
                true_cfg_scale=guidance_scale,
                generator=generator,
                output_type=output_type,
                height=self.resolution,
                width=self.resolution,
                determistic=False,
            )
            latents = stack_steps(collected["all_latents"], dim=1)
            log_probs = stack_steps(collected["all_log_probs"], dim=1)
            timesteps = torch.stack(collected["all_timesteps"]).unsqueeze(0).repeat(len(prompts), 1)

        return RolloutBatch(
            prompts=list(prompts),
            metadata=[dict(item) for item in metadata],
            media=collected["images"].detach() if isinstance(collected["images"], torch.Tensor) else collected["images"],
            latents=latents[:, :-1].detach(),
            next_latents=latents[:, 1:].detach(),
            timesteps=timesteps.detach(),
            old_log_probs=log_probs.detach(),
            epoch_tag=rollout_config.get("epoch_tag"),
            seed=rollout_config.get("seed"),
            model_metadata={
                "adapter": self.name,
                "reference_repo": str(self.repo_root),
                "reference_pipeline": "qwenimage_pipeline_with_logprob",
                "resolution": self.resolution,
                "guidance_scale": guidance_scale,
            },
            model_tensors={
                "prompt_embeds": collected["prompt_embeds"].detach(),
                "negative_prompt_embeds": collected["negative_prompt_embeds"].detach(),
                "prompt_embeds_mask": collected["prompt_embeds_mask"].detach(),
                "negative_prompt_embeds_mask": collected["negative_prompt_embeds_mask"].detach(),
            },
        )

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        import torch

        self._ensure_loaded()
        guidance_scale = float(batch.model_metadata.get("guidance_scale", 4.0))
        latents = batch.latents.to(self.device)
        next_latents = batch.next_latents.to(self.device)
        timesteps = batch.timesteps.to(self.device)
        prompt_embeds = batch.model_tensors["prompt_embeds"].to(self.device)
        negative_prompt_embeds = batch.model_tensors["negative_prompt_embeds"].to(self.device)
        prompt_mask = batch.model_tensors["prompt_embeds_mask"].to(self.device)
        negative_prompt_mask = batch.model_tensors["negative_prompt_embeds_mask"].to(self.device)
        txt_seq_lens = prompt_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_prompt_mask.sum(dim=1).tolist()
        max_len = max(txt_seq_lens + negative_txt_seq_lens)
        prompt_embeds = prompt_embeds[:, :max_len]
        negative_prompt_embeds = negative_prompt_embeds[:, :max_len]
        prompt_mask = prompt_mask[:, :max_len]
        negative_prompt_mask = negative_prompt_mask[:, :max_len]
        img_shapes = [[(1, self.resolution // self.pipeline.vae_scale_factor // 2, self.resolution // self.pipeline.vae_scale_factor // 2)]] * len(batch.prompts)
        transformer = module_or_self(self.transformer)
        log_probs = []
        for index in range(latents.shape[1]):
            noise_pred = transformer(
                hidden_states=torch.cat([latents[:, index], latents[:, index]], dim=0),
                timestep=torch.cat([timesteps[:, index], timesteps[:, index]], dim=0) / 1000,
                guidance=None,
                encoder_hidden_states_mask=torch.cat([prompt_mask, negative_prompt_mask], dim=0),
                encoder_hidden_states=torch.cat([prompt_embeds, negative_prompt_embeds], dim=0),
                img_shapes=img_shapes * 2,
                txt_seq_lens=txt_seq_lens + negative_txt_seq_lens,
            )[0]
            noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)
            combined = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(combined, dim=-1, keepdim=True).clamp_min(1e-6)
            noise_pred = combined * (cond_norm / noise_norm)
            _prev_sample, log_prob, _mean, _std = self._sde_step_with_logprob(
                self.pipeline.scheduler,
                noise_pred.float(),
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

            torch.save(self.transformer.state_dict(), path / "qwenimage_transformer.pt")


MODEL_ADAPTERS.register("tempflow_qwenimage_legacy", QwenImageTempFlowAdapter)
MODEL_ADAPTERS.register("qwenimage_tempflow", QwenImageTempFlowAdapter)
