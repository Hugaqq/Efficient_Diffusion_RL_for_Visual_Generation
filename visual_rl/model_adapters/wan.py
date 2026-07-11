"""Wan/World-R1 adapter with lazy legacy imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.diffusers_common import AdapterNotLoadedError, make_generator, require_model_path
from visual_rl.model_adapters.diffusers_common import resolve_torch_dtype
from visual_rl.third_party.legacy import legacy_repo_path, resolve_legacy_repo


class WorldR1WanLegacyAdapter(ModelAdapter):
    """Bounded bridge for Wan rollout/logprob behavior from the reference repos.

    The real heavy training path is intentionally lazy so `import visual_rl` and
    mock smoke tests do not require CUDA, diffusers, or Wan checkpoints.
    """

    name = "world_r1_wan_legacy"
    media_type = "video"

    def __init__(self, config: dict[str, Any]):
        extra = dict(config.get("extra") or {})
        self.config = {**extra, **{key: value for key, value in config.items() if key != "extra"}}
        self.repo_root = resolve_legacy_repo(
            self.config.get("world_r1_root") or self.config.get("repo_root", "World-R1-main")
        )
        self.pipeline = None
        self.transformer = None
        self.scheduler = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.device = None
        self.dtype = None
        self.model_path = None

    def parameters(self):
        if self.transformer is None:
            raise RuntimeError("WorldR1WanLegacyAdapter must be loaded before parameters()")
        return self.transformer.parameters()

    def named_parameters(self):
        if self.transformer is None:
            raise RuntimeError(
                "WorldR1WanLegacyAdapter must be loaded before named_parameters()"
            )
        return [
            (f"transformer.{name}", parameter)
            for name, parameter in self.transformer.named_parameters()
            if parameter.requires_grad
        ]

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
            self.scheduler = getattr(self.pipeline, "scheduler", None)
            self.tokenizer = getattr(self.pipeline, "tokenizer", None)
            self.tokenizer_2 = getattr(self.pipeline, "tokenizer_2", None)
            self.text_encoder = getattr(self.pipeline, "text_encoder", None)
            self.text_encoder_2 = getattr(self.pipeline, "text_encoder_2", None)
            self.device = self._infer_device(device)
            self.dtype = dtype or getattr(self.transformer, "dtype", None)
            self.model_path = str(model_path)
        return self

    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        self._ensure_loaded()
        import torch

        batch_size = len(prompts)
        runtime = self._runtime_options(rollout_config)
        generator = make_generator(runtime["device"], runtime["seed"])
        prompt_embeds, negative_prompt_embeds = self._encode_prompt_embeds(
            prompts,
            negative_prompts=[""] * batch_size,
            num_videos_per_prompt=runtime["num_videos_per_prompt"],
            max_sequence_length=runtime["max_sequence_length"],
        )
        pipeline_with_logprob, reference_path = self._load_wan_pipeline_with_logprob()

        with torch.no_grad():
            result = pipeline_with_logprob(
                self.pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=runtime["num_steps"],
                guidance_scale=runtime["guidance_scale"],
                output_type="pt",
                return_dict=False,
                num_frames=runtime["frames"],
                height=runtime["height"],
                width=runtime["width"],
                kl_reward=runtime["kl_reward"],
                generator=generator,
                noise_level=runtime["noise_level"],
                sde_type=runtime["sde_type"],
                diffusion_clip=runtime["diffusion_clip"],
                diffusion_clip_value=runtime["diffusion_clip_value"],
                sde_window_size=runtime["sde_window_size"],
                sde_window_range=runtime["sde_window_range"],
            )

        videos, all_latents, all_log_probs, all_kl, all_timesteps = self._normalize_pipeline_result(result)
        latents = torch.stack(list(all_latents), dim=1)
        old_log_probs = torch.stack(list(all_log_probs), dim=1)
        if all_kl:
            kl = torch.stack(list(all_kl), dim=1)
        else:
            kl = torch.zeros_like(old_log_probs)
        timesteps = torch.stack(list(all_timesteps)).to(old_log_probs.device).unsqueeze(0).repeat(batch_size, 1)

        batch = RolloutBatch(
            prompts=list(prompts),
            metadata=[dict(item) for item in metadata],
            media=videos,
            latents=latents[:, :-1],
            next_latents=latents[:, 1:],
            timesteps=timesteps,
            old_log_probs=old_log_probs,
            kl=kl,
            epoch_tag=rollout_config.get("epoch_tag"),
            seed=runtime["seed"],
            model_metadata={
                **self.runtime_metadata(),
                "adapter": self.name,
                "reference_path": str(reference_path),
                "frame_shape": list(getattr(videos, "shape", [])),
                "scheduler": self._scheduler_metadata(),
                "sample_config": self._serializable_runtime_options(runtime),
            },
            model_tensors={
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
            },
        )
        batch.validate_lightweight(strict=True)
        return batch

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        self._ensure_loaded()
        import torch

        sde_step_with_logprob = self._load_sde_step_with_logprob()

        prompt_embeds = batch.model_tensors.get("prompt_embeds")
        negative_prompt_embeds = batch.model_tensors.get("negative_prompt_embeds")
        if prompt_embeds is None:
            raise ValueError("RolloutBatch.model_tensors must contain prompt_embeds for Wan logprob recomputation.")

        runtime = dict(batch.model_metadata.get("sample_config") or {})
        guidance_scale = float(runtime.get("guidance_scale", self.config.get("guidance_scale", 5.0)))
        train_cfg = bool(runtime.get("train_cfg", self.config.get("train_cfg", True)))
        attention_kwargs = self.config.get("attention_kwargs")
        steps = int(batch.old_log_probs.shape[1])
        new_log_probs = []
        for j in range(steps):
            latents_j = batch.latents[:, j].to(device=batch.old_log_probs.device)
            timesteps_j = batch.timesteps[:, j].to(device=batch.old_log_probs.device)
            next_latents_j = batch.next_latents[:, j].to(device=batch.old_log_probs.device)
            noise_pred_text = self.transformer(
                hidden_states=latents_j,
                timestep=timesteps_j,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            if train_cfg:
                if negative_prompt_embeds is None:
                    raise ValueError("negative_prompt_embeds are required when train_cfg=True.")
                noise_pred_uncond = self.transformer(
                    hidden_states=latents_j,
                    timestep=timesteps_j,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_text

            _prev_sample, log_prob, *_rest = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                timesteps_j,
                latents_j.float(),
                noise_level=float(runtime.get("noise_level", self.config.get("noise_level", 0.7))),
                prev_sample=next_latents_j.float(),
                sde_type=runtime.get("sde_type", self.config.get("sde_type", "flow_sde")),
                diffusion_clip=bool(runtime.get("diffusion_clip", self.config.get("diffusion_clip", False))),
                diffusion_clip_value=float(
                    runtime.get("diffusion_clip_value", self.config.get("diffusion_clip_value", 0.45))
                ),
            )
            new_log_probs.append(log_prob)
        return torch.stack(new_log_probs, dim=1).to(device=batch.old_log_probs.device)

    def save_pretrained(self, output_dir: str) -> None:
        if self.transformer is None:
            raise RuntimeError("WorldR1WanLegacyAdapter must be loaded before save_pretrained()")
        import torch

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.transformer, "save_pretrained"):
            self.transformer.save_pretrained(path / "transformer")
            save_kind = "transformer_save_pretrained"
        else:
            save_kind = "transformer_state_dict"
        torch.save(self.transformer.state_dict(), path / "transformer_state.pt")

        from visual_rl.artifacts.checkpoint import save_json

        save_json(
            path / "adapter_metadata.json",
            {
                **self.runtime_metadata(),
                "save_kind": save_kind,
                "resume_support": "full transformer state",
            },
        )

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        if self.transformer is None:
            raise RuntimeError("WorldR1WanLegacyAdapter must be loaded before load_checkpoint()")
        import torch

        path = Path(checkpoint_dir)
        state_path = path / "transformer_state.pt"
        if not state_path.exists():
            raise RuntimeError(
                "WorldR1WanLegacyAdapter resume currently requires transformer_state.pt. "
                "LoRA-only/save_pretrained resume is not yet wired for this bounded path."
            )
        state = torch.load(state_path, map_location=self.device or "cpu", weights_only=False)
        self.transformer.load_state_dict(state)

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path or self.config.get("model_path") or self.config.get("pretrained_model"),
            "repo_root": str(self.repo_root),
            "frames": self.config.get("frames"),
            "height": self.config.get("height"),
            "width": self.config.get("width"),
            "num_steps": self.config.get("num_steps"),
            "guidance_scale": self.config.get("guidance_scale"),
            "noise_level": self.config.get("noise_level"),
            "sde_type": self.config.get("sde_type"),
            "diffusion_clip": self.config.get("diffusion_clip"),
            "device": str(self.device) if self.device is not None else None,
            "dtype": str(self.dtype) if self.dtype is not None else None,
            "pipeline_class": type(self.pipeline).__name__ if self.pipeline is not None else None,
            "transformer_class": type(self.transformer).__name__ if self.transformer is not None else None,
        }

    def _ensure_loaded(self) -> None:
        if self.pipeline is None or self.transformer is None or self.scheduler is None:
            raise AdapterNotLoadedError("WorldR1WanLegacyAdapter.load() must run before Wan sampling/logprob.")

    def _infer_device(self, configured_device: str):
        import torch

        if configured_device:
            return torch.device(configured_device)
        if execution_device := getattr(self.pipeline, "_execution_device", None):
            return torch.device(execution_device)
        try:
            return next(self.transformer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _runtime_options(self, rollout_config: dict[str, Any]) -> dict[str, Any]:
        def pick(name: str, default: Any) -> Any:
            return rollout_config.get(name, self.config.get(name, default))

        sde_window_range = pick("sde_window_range", None)
        return {
            "num_steps": int(pick("num_steps", 2)),
            "guidance_scale": float(pick("guidance_scale", 5.0)),
            "frames": int(pick("frames", pick("num_frames", 81))),
            "height": int(pick("height", 480)),
            "width": int(pick("width", 832)),
            "kl_reward": float(pick("kl_reward", 0.0)),
            "noise_level": float(pick("noise_level", 0.7)),
            "sde_type": pick("sde_type", "flow_sde"),
            "diffusion_clip": bool(pick("diffusion_clip", False)),
            "diffusion_clip_value": float(pick("diffusion_clip_value", 0.45)),
            "sde_window_size": int(pick("sde_window_size", 0) or 0),
            "sde_window_range": tuple(sde_window_range) if sde_window_range else None,
            "seed": int(pick("seed", self.config.get("seed", 0))),
            "device": self.device or "cpu",
            "max_sequence_length": int(pick("max_sequence_length", 512)),
            "num_videos_per_prompt": int(pick("num_videos_per_prompt", pick("num_video_per_prompt", 1))),
            "train_cfg": bool(pick("train_cfg", True)),
        }

    @staticmethod
    def _serializable_runtime_options(runtime: dict[str, Any]) -> dict[str, Any]:
        serializable = dict(runtime)
        serializable["device"] = str(serializable.get("device"))
        if serializable.get("sde_window_range") is not None:
            serializable["sde_window_range"] = list(serializable["sde_window_range"])
        return serializable

    def _scheduler_metadata(self) -> dict[str, Any]:
        scheduler = self.scheduler
        metadata = {"class": type(scheduler).__name__ if scheduler is not None else None}
        timesteps = getattr(scheduler, "timesteps", None)
        if timesteps is not None:
            metadata["timesteps"] = [int(item) for item in timesteps.detach().cpu().flatten().tolist()]
        return metadata

    def _encode_prompt_embeds(
        self,
        prompts: list[str],
        *,
        negative_prompts: list[str],
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ):
        if hasattr(self.pipeline, "encode_prompt"):
            encoded = self.pipeline.encode_prompt(
                prompt=prompts,
                negative_prompt=negative_prompts,
                do_classifier_free_guidance=bool(self.config.get("train_cfg", True)),
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=max_sequence_length,
                device=self.device,
            )
            if isinstance(encoded, tuple):
                return encoded[0], encoded[1]
            raise TypeError("WanPipeline.encode_prompt must return (prompt_embeds, negative_prompt_embeds).")

        with legacy_repo_path(self.repo_root):
            from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt

        text_encoders = [item for item in [self.text_encoder, self.text_encoder_2] if item is not None]
        tokenizers = [item for item in [self.tokenizer, self.tokenizer_2] if item is not None]
        prompt_embeds = encode_prompt(
            text_encoders,
            tokenizers,
            prompts,
            max_sequence_length=max_sequence_length,
            num_videos_per_prompt=num_videos_per_prompt,
            device=self.device,
            dtype=self.dtype,
        )
        negative_embeds = encode_prompt(
            text_encoders,
            tokenizers,
            negative_prompts,
            max_sequence_length=max_sequence_length,
            num_videos_per_prompt=num_videos_per_prompt,
            device=self.device,
            dtype=self.dtype,
        )
        return prompt_embeds, negative_embeds

    def _load_wan_pipeline_with_logprob(self):
        injected = self.config.get("wan_pipeline_with_logprob")
        if injected is not None:
            return injected, "<injected>"
        with legacy_repo_path(self.repo_root):
            from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob

        return wan_pipeline_with_logprob, self.repo_root

    def _load_sde_step_with_logprob(self):
        injected = self.config.get("sde_step_with_logprob")
        if injected is not None:
            return injected
        with legacy_repo_path(self.repo_root):
            from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import (
                sde_step_with_logprob,
            )

        return sde_step_with_logprob

    def _normalize_pipeline_result(self, result: Any):
        if len(result) == 5:
            return result
        if len(result) == 4:
            videos, latents, log_probs, kl = result
            timesteps = getattr(self.scheduler, "timesteps", None)
            if timesteps is None:
                raise ValueError("4-item Wan pipeline result requires scheduler.timesteps to reconstruct timesteps.")
            return videos, latents, log_probs, kl, list(timesteps[: len(log_probs)])
        raise ValueError(f"Wan pipeline result must have 4 or 5 items, got {len(result)}.")


MODEL_ADAPTERS.register("world_r1_wan_legacy", WorldR1WanLegacyAdapter)
