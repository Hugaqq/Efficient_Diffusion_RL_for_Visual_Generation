"""Final Wan2.1 adapters for Flash-GRPO and World-R1 rollouts."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, ClassVar, Literal

from visual_rl.core.types import (
    FrozenMapping,
    PolicyRecomputeStats,
    ResolutionContext,
    RolloutBatch,
    RolloutRequest,
    RuntimeBuildContext,
    ValidationCheck,
    ValidationContext,
)
from visual_rl.errors import ConfigError, RunError
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.diffusers_common import (
    AdapterNotLoadedError,
    apply_peft_lora,
    configure_gradient_checkpointing,
    make_generator,
    resolve_torch_dtype,
    verify_gradient_checkpointing,
)
from visual_rl.model_adapters.diffusion_transition import (
    wan_sde_step_with_logprob,
)

WAN_VAE_TEMPORAL_STRIDE = 4

_COMMON_REQUIRED = frozenset(
    {
        "checkpoint",
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
        "gradient_checkpointing",
        "guidance_scale",
        "height",
        "width",
        "max_sequence_length",
    }
)
_OPTIONAL_DEFAULTS: Mapping[str, object] = {
    "local_files_only": True,
    "low_cpu_mem_usage": True,
    "offload_frozen_modules_during_update": False,
    "vae_tiling": False,
}


class _WanAdapterCore(ModelAdapter):
    """Shared private implementation; public components freeze the backend."""

    _COMPONENT_NAME: ClassVar[str]
    _BACKEND: ClassVar[Literal["flash", "world_r1"]]
    _FRAMES: ClassVar[int | None]

    def __init__(
        self,
        *,
        checkpoint: Path,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: tuple[str, ...],
        gradient_checkpointing: bool,
        guidance_scale: float,
        height: int,
        width: int,
        frames: int,
        max_sequence_length: int,
        local_files_only: bool,
        low_cpu_mem_usage: bool,
        context: RuntimeBuildContext,
        offload_frozen_modules_during_update: bool = False,
        vae_tiling: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.gradient_checkpointing = gradient_checkpointing
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.frames = frames
        self.max_sequence_length = max_sequence_length
        self.local_files_only = local_files_only
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.offload_frozen_modules_during_update = (
            offload_frozen_modules_during_update
        )
        self.vae_tiling = vae_tiling
        self.device = context.device
        self.dtype = resolve_torch_dtype(context.precision)
        self.train_cfg = guidance_scale > 1.0
        self.pipeline = None
        self.transformer = None
        self.scheduler = None
        self._gradient_checkpointing_state = None
        self._frozen_modules_offloaded = False
        self._text_encoders_active = False
        self._vae_active = False
        self._policy_active = False

    @property
    def train_module(self):
        if self.transformer is None:
            raise AdapterNotLoadedError(
                f"{self._COMPONENT_NAME} is not fully constructed"
            )
        return self.transformer

    @classmethod
    def _resolve(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
        *,
        include_frames: bool,
    ) -> Mapping[str, object]:
        if not isinstance(raw, Mapping):
            raise ConfigError("model.params must be a mapping", key="model.params")
        required = set(_COMMON_REQUIRED)
        if include_frames:
            required.add("frames")
        allowed = required | set(_OPTIONAL_DEFAULTS)
        unknown = sorted(set(raw) - allowed)
        missing = sorted(required - set(raw))
        if unknown:
            raise ConfigError(
                f"Unknown model.params: {unknown}",
                key="model.params",
            )
        if missing:
            raise ConfigError(
                f"Missing model.params: {missing}",
                key="model.params",
            )

        values = {**_OPTIONAL_DEFAULTS, **dict(raw)}
        values["checkpoint"] = _resolve_path(
            values["checkpoint"],
            context,
            "checkpoint",
        )
        for key in ("lora_rank", "lora_alpha", "height", "width"):
            values[key] = _positive_int(values[key], key)
        values["max_sequence_length"] = _positive_int(
            values["max_sequence_length"],
            "max_sequence_length",
        )
        if values["height"] % 8 or values["width"] % 8:
            raise ConfigError(
                "model.params.height and width must be multiples of 8",
                key="model.params",
            )
        values["lora_target_modules"] = _target_modules(
            values["lora_target_modules"]
        )
        for key in (
            "gradient_checkpointing",
            "local_files_only",
            "low_cpu_mem_usage",
            "offload_frozen_modules_during_update",
            "vae_tiling",
        ):
            if type(values[key]) is not bool:
                raise ConfigError(
                    f"model.params.{key} must be bool",
                    key=f"model.params.{key}",
                )
        guidance = values["guidance_scale"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, (int, float))
            or not math.isfinite(float(guidance))
            or float(guidance) <= 0
        ):
            raise ConfigError(
                "model.params.guidance_scale must be finite and > 0",
                key="model.params.guidance_scale",
            )
        values["guidance_scale"] = float(guidance)
        if include_frames:
            frames = _positive_int(values["frames"], "frames")
            if (frames - 1) % WAN_VAE_TEMPORAL_STRIDE:
                raise ConfigError(
                    "model.params.frames must satisfy (frames - 1) % 4 == 0",
                    key="model.params.frames",
                )
            values["frames"] = frames
        return FrozenMapping(values)

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, object],
        context: ValidationContext,
    ) -> tuple[ValidationCheck, ...]:
        del context
        checks: list[ValidationCheck] = []
        for key in ("checkpoint",):
            value = resolved.get(key)
            if not isinstance(value, Path) or not value.is_absolute():
                checks.append(
                    ValidationCheck(
                        level="error",
                        code="model_path_not_resolved",
                        path=f"model.params.{key}",
                        message=f"{key} must resolve to an absolute path",
                    )
                )
            elif not value.is_dir():
                checks.append(
                    ValidationCheck(
                        level="error",
                        code="model_path_missing",
                        path=f"model.params.{key}",
                        message=f"{key} directory does not exist: {value}",
                    )
                )
        return tuple(checks)

    @classmethod
    def _from_resolved(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
        *,
        frames: int,
    ) -> _WanAdapterCore:
        adapter = cls(
            checkpoint=_require_path(resolved, "checkpoint"),
            lora_rank=int(resolved["lora_rank"]),
            lora_alpha=int(resolved["lora_alpha"]),
            lora_target_modules=tuple(resolved["lora_target_modules"]),
            gradient_checkpointing=bool(resolved["gradient_checkpointing"]),
            guidance_scale=float(resolved["guidance_scale"]),
            height=int(resolved["height"]),
            width=int(resolved["width"]),
            frames=frames,
            max_sequence_length=int(resolved["max_sequence_length"]),
            local_files_only=bool(resolved["local_files_only"]),
            low_cpu_mem_usage=bool(resolved["low_cpu_mem_usage"]),
            context=context,
            offload_frozen_modules_during_update=bool(
                resolved["offload_frozen_modules_during_update"]
            ),
            vae_tiling=bool(resolved["vae_tiling"]),
        )
        adapter._load_base_pipeline()
        return adapter

    def _load_base_pipeline(self) -> None:
        import torch
        from diffusers import WanPipeline

        pipeline = WanPipeline.from_pretrained(
            str(self.checkpoint),
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )
        vae = getattr(pipeline, "vae", None)
        if vae is not None:
            vae.requires_grad_(False)
            vae.eval()
            self._configure_vae_decode(vae)
        for name in ("text_encoder", "text_encoder_2"):
            encoder = getattr(pipeline, name, None)
            if encoder is None:
                continue
            encoder.requires_grad_(False)
            encoder.eval()
        base_transformer = pipeline.transformer
        checkpointing_state = configure_gradient_checkpointing(
            base_transformer,
            self.gradient_checkpointing,
            context=f"{self._COMPONENT_NAME} base transformer",
        )
        transformer = apply_peft_lora(
            base_transformer,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
        )
        pipeline.transformer = transformer
        if (
            self.offload_frozen_modules_during_update
            and torch.device(self.device).type == "cuda"
        ):
            for name in ("text_encoder", "text_encoder_2"):
                encoder = getattr(pipeline, name, None)
                if encoder is not None:
                    encoder.to("cpu", dtype=self.dtype)
            if vae is not None:
                vae.to("cpu", dtype=torch.float32)
            transformer.to(self.device)
            self._frozen_modules_offloaded = True
            self._policy_active = True
        else:
            pipeline = pipeline.to(self.device)
            if vae is not None:
                vae.to(device=self.device, dtype=torch.float32)
            for name in ("text_encoder", "text_encoder_2"):
                encoder = getattr(pipeline, name, None)
                if encoder is not None:
                    encoder.to(device=self.device, dtype=self.dtype)
            self._text_encoders_active = True
            self._vae_active = vae is not None
            self._policy_active = True
        self.pipeline = pipeline
        self.transformer = transformer
        self.scheduler = pipeline.scheduler
        self._gradient_checkpointing_state = verify_gradient_checkpointing(
            transformer,
            checkpointing_state,
            context=f"{self._COMPONENT_NAME} active transformer",
        )

    def _configure_vae_decode(self, vae) -> None:
        if not self.vae_tiling:
            return
        enable_tiling = getattr(vae, "enable_tiling", None)
        if not callable(enable_tiling):
            raise RunError(
                "Wan vae_tiling requires VAE.enable_tiling() support"
            )
        enable_tiling()

    def sample(self, request: RolloutRequest) -> RolloutBatch:
        self._ensure_loaded()
        expected_kind = (
            "single_step" if self._BACKEND == "flash" else "full_trajectory"
        )
        if request.kind != expected_kind:
            raise RunError(
                f"{self._COMPONENT_NAME} only supports {expected_kind} rollout"
            )
        was_training = self.train_module.training
        self.train_module.eval()
        try:
            if self._BACKEND == "flash":
                batch = self._sample_flash(request)
            else:
                batch = self._sample_world(request)
            batch.validate_against(request)
            return batch
        finally:
            try:
                self.train_module.train(was_training)
            finally:
                self._offload_frozen_modules_for_update()

    def _sample_world(self, request: RolloutRequest) -> RolloutBatch:
        import torch

        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            request.prompts
        )
        generator = make_generator(self.device, request.context.seed)
        base_latents, callback, camera = self._prepare_world_camera(
            request.prompts,
            generator,
        )
        with torch.no_grad():
            result = self._run_world_pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_steps=request.num_steps,
                generator=generator,
                latents=base_latents,
                callback=callback,
            )
        media, states, log_probs, timesteps = self._normalize_world_result(
            result,
            batch_size=len(request.prompts),
            num_steps=request.num_steps,
        )
        latents = states[:, :-1]
        next_latents = states[:, 1:]
        transition_mask = torch.ones_like(log_probs, dtype=torch.bool)
        payload = {"prompt_embeds": prompt_embeds.detach()}
        if negative_prompt_embeds is not None:
            payload["negative_prompt_embeds"] = negative_prompt_embeds.detach()
        return RolloutBatch(
            prompts=request.prompts,
            metadata=request.metadata,
            media=media.detach(),
            latents=latents.detach(),
            next_latents=next_latents.detach(),
            timesteps=timesteps.detach(),
            old_log_probs=log_probs.detach(),
            transition_mask=transition_mask.detach(),
            sample_id=request.sample_id,
            prompt_id=request.prompt_id,
            group_id=request.group_id,
            branch_id=request.branch_id,
            media_layout="BFCHW",
            camera_trajectory=camera.detach(),
            context=request.context,
            selected_timestep_index=None,
            flash_coefficient=None,
            branch_step_index=None,
            trajectory_step_index=None,
            transition_std_dev=None,
            recompute_payload=payload,
            artifact_metadata={
                "adapter": self._COMPONENT_NAME,
                "frames": self.frames,
                "height": self.height,
                "width": self.width,
            },
        )

    def _sample_flash(self, request: RolloutRequest) -> RolloutBatch:
        import torch

        if request.selected_timestep_index is None:
            raise RunError("wan_flash requires selected timestep indices")
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            request.prompts
        )
        grouped_positions: list[int] = []
        chunks: dict[str, list[Any]] = {
            key: []
            for key in (
                "media",
                "latents",
                "next_latents",
                "timesteps",
                "log_probs",
                "coefficient",
                "prompt_embeds",
                "negative_prompt_embeds",
            )
        }
        selected = request.selected_timestep_index
        for selected_index in sorted(set(selected)):
            positions = [
                row for row, value in enumerate(selected) if value == selected_index
            ]
            grouped_positions.extend(positions)
            prompt_group = _select_rows(prompt_embeds, positions)
            negative_group = (
                None
                if negative_prompt_embeds is None
                else _select_rows(negative_prompt_embeds, positions)
            )
            seed = _selected_seed(request.context.seed, selected_index)
            generator = make_generator(self.device, seed)
            with self._fork_rng(seed), torch.no_grad():
                result = self._run_flash_pipeline(
                    prompt_embeds=prompt_group,
                    negative_prompt_embeds=negative_group,
                    num_steps=request.num_steps,
                    generator=generator,
                    selected_index=selected_index,
                )
            (
                media,
                latent,
                next_latent,
                log_prob,
                timestep,
                coefficient,
            ) = self._normalize_flash_result(
                result,
                batch_size=len(positions),
                selected_index=selected_index,
            )
            chunks["media"].append(media)
            chunks["latents"].append(latent)
            chunks["next_latents"].append(next_latent)
            chunks["log_probs"].append(log_prob)
            chunks["timesteps"].append(timestep)
            chunks["coefficient"].append(coefficient)
            chunks["prompt_embeds"].append(prompt_group)
            if negative_group is not None:
                chunks["negative_prompt_embeds"].append(negative_group)

        restored = {
            key: _restore_order(torch.cat(values, dim=0), grouped_positions)
            for key, values in chunks.items()
            if values
        }
        transition_mask = torch.ones_like(
            restored["log_probs"],
            dtype=torch.bool,
        )
        payload = {"prompt_embeds": restored["prompt_embeds"].detach()}
        if "negative_prompt_embeds" in restored:
            payload["negative_prompt_embeds"] = restored[
                "negative_prompt_embeds"
            ].detach()
        selected_tensor = torch.tensor(
            selected,
            dtype=torch.int64,
            device=restored["timesteps"].device,
        )
        return RolloutBatch(
            prompts=request.prompts,
            metadata=request.metadata,
            media=restored["media"].detach(),
            latents=restored["latents"].detach(),
            next_latents=restored["next_latents"].detach(),
            timesteps=restored["timesteps"].detach(),
            old_log_probs=restored["log_probs"].detach(),
            transition_mask=transition_mask.detach(),
            sample_id=request.sample_id,
            prompt_id=request.prompt_id,
            group_id=request.group_id,
            branch_id=request.branch_id,
            media_layout="BFCHW",
            camera_trajectory=None,
            context=request.context,
            selected_timestep_index=selected_tensor,
            flash_coefficient=restored["coefficient"].detach(),
            branch_step_index=None,
            trajectory_step_index=None,
            transition_std_dev=None,
            recompute_payload=payload,
            artifact_metadata={
                "adapter": self._COMPONENT_NAME,
                "frames": self.frames,
                "height": self.height,
                "width": self.width,
            },
        )

    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        if require_reference:
            raise RunError(f"{self._COMPONENT_NAME} has no reference-policy stats")
        self._ensure_loaded()
        prompt_embeds = batch.recompute_payload.get("prompt_embeds")
        negative_prompt_embeds = batch.recompute_payload.get(
            "negative_prompt_embeds"
        )
        if prompt_embeds is None:
            raise RunError("Wan recompute payload is missing prompt_embeds")

        import torch

        was_training = self.train_module.training
        # Match rollout mode exactly.  Evaluation mode keeps autograd enabled
        # while preventing dropout from perturbing the policy ratio.
        self.train_module.eval()
        try:
            device, dtype = _wan_module_device_and_dtype(
                self.train_module,
                fallback_device=self.device,
                fallback_dtype=self.dtype,
            )
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    device=device,
                    dtype=dtype,
                )
            sde_step = self._load_sde_function()
            values = []
            for step in range(batch.transition_count):
                sample = batch.latents[:, step].to(device=device)
                timestep = batch.timesteps[:, step].to(device=device)
                next_sample = batch.next_latents[:, step].to(device=device)
                prediction = self._wan_prediction(
                    sample,
                    timestep,
                    prompt_embeds,
                    negative_prompt_embeds,
                )
                kwargs: dict[str, object] = {
                    "prev_sample": next_sample.float(),
                }
                if self._BACKEND == "flash":
                    kwargs["return_dt_and_std_dev_t"] = True
                step_result = sde_step(
                    self.scheduler,
                    prediction.float(),
                    timestep,
                    sample.float(),
                    **kwargs,
                )
                if not isinstance(step_result, (tuple, list)) or len(step_result) < 2:
                    raise RunError("Wan transition helper returned an invalid result")
                log_prob = _scalar_transition(
                    step_result[1],
                    batch.batch_size,
                    "new_log_probs",
                )
                if self._BACKEND == "flash":
                    if len(step_result) != 6:
                        raise RunError(
                            "Flash transition helper must return six values"
                        )
                    recomputed = _normalize_coefficient(
                        step_result[5],
                        batch.batch_size,
                    )
                    stored = batch.flash_coefficient
                    if stored is None or not torch.allclose(
                        stored.to(recomputed),
                        recomputed,
                        rtol=1e-5,
                        atol=1e-7,
                    ):
                        raise RunError("Flash coefficient changed during recompute")
                values.append(log_prob)
            new_log_probs = torch.cat(values, dim=1).to(
                device=batch.old_log_probs.device
            )
            stats = PolicyRecomputeStats(new_log_probs=new_log_probs)
            stats.validate_against(batch, require_reference=False)
            return stats
        finally:
            if self.pipeline is not None:
                self.pipeline._current_timestep = None
            self.train_module.train(was_training)

    def _encode_prompt(self, prompts: tuple[str, ...]):
        if self.pipeline is None or not callable(
            getattr(self.pipeline, "encode_prompt", None)
        ):
            raise RunError("WanPipeline.encode_prompt is required")
        self._activate_text_encoders_for_prompt()
        try:
            encoded = self.pipeline.encode_prompt(
                prompt=list(prompts),
                negative_prompt=[""] * len(prompts),
                do_classifier_free_guidance=self.train_cfg,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=self.max_sequence_length,
                device=self.device,
            )
        finally:
            self._offload_text_encoders()
            self._activate_policy_module()
        if not isinstance(encoded, tuple) or len(encoded) < 2:
            raise RunError(
                "WanPipeline.encode_prompt must return positive/negative embeddings"
            )
        return encoded[0], encoded[1]

    def _load_sde_function(self):
        variant = "flash" if self._BACKEND == "flash" else "world_r1"

        def step(
            scheduler,
            model_output,
            timestep,
            sample,
            *,
            prev_sample=None,
            generator=None,
            determistic: bool = False,
            return_dt_and_std_dev_t: bool = False,
        ):
            return wan_sde_step_with_logprob(
                scheduler,
                model_output,
                timestep,
                sample,
                variant=variant,
                prev_sample=prev_sample,
                generator=generator,
                deterministic=determistic,
                return_flash_coefficient=return_dt_and_std_dev_t,
            )

        return step

    def _run_world_pipeline(
        self,
        *,
        prompt_embeds,
        negative_prompt_embeds,
        num_steps: int,
        generator,
        latents,
        callback,
    ):
        import torch

        timesteps, latents = self._prepare_native_wan_rollout(
            prompt_embeds=prompt_embeds,
            num_steps=num_steps,
            generator=generator,
            latents=latents,
        )
        states = [latents]
        log_probs = []
        for index, timestep in enumerate(timesteps):
            prediction = self._wan_prediction(
                latents,
                timestep,
                prompt_embeds,
                negative_prompt_embeds,
            )
            next_latents = self.scheduler.step(
                prediction.float(),
                timestep,
                latents.float(),
                return_dict=False,
            )[0]
            transition = wan_sde_step_with_logprob(
                self.scheduler,
                prediction.float(),
                timestep.reshape(1),
                latents.float(),
                variant="world_r1",
                prev_sample=next_latents.float(),
            )
            log_prob = transition[1]
            if callback is not None:
                outputs = callback(
                    self.pipeline,
                    index,
                    timestep,
                    {"latents": next_latents},
                )
                if not isinstance(outputs, Mapping):
                    raise RunError(
                        "World-R1 camera callback must return a mapping"
                    )
                next_latents = outputs.get("latents", next_latents)
                log_prob = wan_sde_step_with_logprob(
                    self.scheduler,
                    prediction.float(),
                    timestep.reshape(1),
                    latents.float(),
                    variant="world_r1",
                    prev_sample=next_latents.float(),
                )[1]
            latents = next_latents
            states.append(latents)
            log_probs.append(log_prob)
        self.pipeline._current_timestep = None
        media = self._decode_wan_latents(latents)
        zeros = [torch.zeros_like(value) for value in log_probs]
        return media, states, log_probs, zeros, timesteps

    def _run_flash_pipeline(
        self,
        *,
        prompt_embeds,
        negative_prompt_embeds,
        num_steps: int,
        generator,
        selected_index: int,
    ):
        import torch

        timesteps, latents = self._prepare_native_wan_rollout(
            prompt_embeds=prompt_embeds,
            num_steps=num_steps,
            generator=generator,
            latents=None,
        )
        if not 0 <= selected_index < len(timesteps):
            raise RunError("Flash selected timestep is outside the scheduler")
        selected_source = None
        selected_target = None
        selected_log_prob = None
        for index, timestep in enumerate(timesteps):
            prediction = self._wan_prediction(
                latents,
                timestep,
                prompt_embeds,
                negative_prompt_embeds,
            )
            if index == selected_index:
                selected_source = latents
                transition = wan_sde_step_with_logprob(
                    self.scheduler,
                    prediction.float(),
                    timestep.reshape(1),
                    latents.float(),
                    variant="flash",
                )
                latents, selected_log_prob = transition[:2]
                selected_target = latents
            else:
                latents = wan_sde_step_with_logprob(
                    self.scheduler,
                    prediction.float(),
                    timestep.reshape(1),
                    latents.float(),
                    variant="flash",
                    deterministic=True,
                )[0]
        self.pipeline._current_timestep = None
        if (
            selected_source is None
            or selected_target is None
            or selected_log_prob is None
        ):
            raise RunError("Flash sampler did not execute its transition")
        media = self._decode_wan_latents(latents)
        zeros = [torch.zeros_like(selected_log_prob)]
        return (
            media,
            [selected_source, selected_target],
            [selected_log_prob],
            zeros,
            selected_index,
        )

    def _prepare_native_wan_rollout(
        self,
        *,
        prompt_embeds,
        num_steps: int,
        generator,
        latents,
    ):
        import torch

        pipeline = self.pipeline
        pipeline._guidance_scale = self.guidance_scale
        pipeline._attention_kwargs = None
        pipeline._current_timestep = None
        pipeline._interrupt = False
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        config = getattr(self.train_module, "config", None)
        channels = getattr(config, "in_channels", None)
        if type(channels) is not int or channels <= 0:
            raise RunError("Wan transformer.config.in_channels must be positive")
        prepared = pipeline.prepare_latents(
            prompt_embeds.shape[0],
            channels,
            self.height,
            self.width,
            self.frames,
            torch.float32,
            self.device,
            generator,
            latents,
        )
        return timesteps, prepared

    def _wan_prediction(
        self,
        latents,
        timestep,
        prompt_embeds,
        negative_prompt_embeds,
    ):
        import torch

        pipeline_config = getattr(self.pipeline, "config", object())
        if bool(getattr(pipeline_config, "expand_timesteps", False)):
            mask = torch.ones_like(latents, dtype=torch.float32)
            values = (mask[0][0][:, ::2, ::2] * timestep).flatten()
            model_timestep = values.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            model_timestep = timestep.expand(latents.shape[0])
        _, dtype = _wan_module_device_and_dtype(
            self.train_module,
            fallback_device=self.device,
            fallback_dtype=self.dtype,
        )
        hidden_states = latents.to(dtype=dtype)
        self.pipeline._current_timestep = timestep
        with _cache_context(self.train_module, "cond"):
            prediction = self.train_module(
                hidden_states=hidden_states,
                timestep=model_timestep,
                encoder_hidden_states=prompt_embeds.to(dtype=dtype),
                attention_kwargs=None,
                return_dict=False,
            )[0]
        if self.train_cfg:
            if negative_prompt_embeds is None:
                raise RunError("Wan CFG requires negative prompt embeddings")
            with _cache_context(self.train_module, "uncond"):
                negative = self.train_module(
                    hidden_states=hidden_states,
                    timestep=model_timestep,
                    encoder_hidden_states=negative_prompt_embeds.to(dtype=dtype),
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            prediction = negative + self.guidance_scale * (
                prediction - negative
            )
        return prediction

    def _decode_wan_latents(self, latents):
        import torch

        self._activate_vae_for_decode()
        try:
            vae = self.pipeline.vae
            dtype = _module_dtype(vae, fallback=torch.float32)
            parameter = next(iter(vae.parameters()), None)
            device = parameter.device if parameter is not None else self.device
            latents = latents.to(device=device, dtype=dtype)
            mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents)
            )
            inverse_std = (
                1.0
                / torch.tensor(vae.config.latents_std)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents)
            )
            decoded = vae.decode(
                latents / inverse_std + mean,
                return_dict=False,
            )[0]
            return self.pipeline.video_processor.postprocess_video(
                decoded,
                output_type="pt",
            )
        finally:
            self._offload_vae_after_decode()

    def _activate_text_encoders_for_prompt(self) -> None:
        if not self._cuda_offload_enabled():
            return
        if self._text_encoders_active:
            return

        import torch

        if self._policy_active:
            self.train_module.to("cpu")
            self._policy_active = False
            torch.cuda.empty_cache()
        for name in ("text_encoder", "text_encoder_2"):
            encoder = getattr(self.pipeline, name, None)
            if encoder is not None:
                encoder.to(self.device, dtype=self.dtype)
        self._text_encoders_active = True
        self._frozen_modules_offloaded = False

    def _offload_text_encoders(self) -> None:
        if not self._cuda_offload_enabled() or not self._text_encoders_active:
            return

        import torch

        for name in ("text_encoder", "text_encoder_2"):
            encoder = getattr(self.pipeline, name, None)
            if encoder is not None:
                encoder.to("cpu", dtype=self.dtype)
        torch.cuda.empty_cache()
        self._text_encoders_active = False
        self._frozen_modules_offloaded = not self._vae_active

    def _activate_policy_module(self) -> None:
        if not self._cuda_offload_enabled() or self._policy_active:
            return
        if self._text_encoders_active:
            raise RunError(
                "Wan policy cannot move to CUDA while text encoders are active"
            )
        self.train_module.to(self.device)
        self._policy_active = True

    def _activate_vae_for_decode(self) -> None:
        if not self._cuda_offload_enabled() or self._vae_active:
            return

        import torch

        if self._text_encoders_active:
            raise RunError(
                "Wan VAE cannot move to CUDA while text encoders are active"
            )
        if self._policy_active:
            self.train_module.to("cpu")
            self._policy_active = False
            torch.cuda.empty_cache()
        try:
            self.pipeline.vae.to(self.device, dtype=torch.float32)
        except BaseException:
            # A partially moved VAE must not remain beside the restored policy.
            self.pipeline.vae.to("cpu", dtype=torch.float32)
            torch.cuda.empty_cache()
            self._activate_policy_module()
            raise
        self._vae_active = True
        self._frozen_modules_offloaded = False

    def _offload_vae_after_decode(self) -> None:
        if not self._cuda_offload_enabled():
            return

        if self._vae_active:
            import torch

            self.pipeline.vae.to("cpu", dtype=torch.float32)
            torch.cuda.empty_cache()
            self._vae_active = False
        self._frozen_modules_offloaded = not self._text_encoders_active
        self._activate_policy_module()

    def _offload_frozen_modules_for_update(self) -> None:
        self._offload_text_encoders()
        self._offload_vae_after_decode()
        self._activate_policy_module()

    def _cuda_offload_enabled(self) -> bool:
        import torch

        return (
            self.offload_frozen_modules_during_update
            and torch.device(self.device).type == "cuda"
        )

    def _prepare_world_camera(
        self,
        prompts: tuple[str, ...],
        generator: object,
    ):
        raise RunError("camera preparation is only implemented by wan_world_r1")

    def _normalize_world_result(
        self,
        result: object,
        *,
        batch_size: int,
        num_steps: int,
    ):
        import torch

        if not isinstance(result, (tuple, list)) or len(result) < 3:
            raise RunError("World-R1 Wan pipeline returned an invalid result")
        media, all_states, all_log_probs = result[:3]
        states = _stack_state_sequence(
            all_states,
            batch_size=batch_size,
            expected=num_steps + 1,
            name="latents",
        )
        log_probs = _stack_scalar_sequence(
            all_log_probs,
            batch_size=batch_size,
            expected=num_steps,
            name="old_log_probs",
        )
        raw_timesteps = result[4] if len(result) > 4 else self.scheduler.timesteps
        timesteps = _timesteps(
            raw_timesteps,
            batch_size=batch_size,
            expected=num_steps,
            device=log_probs.device,
        )
        if not isinstance(media, torch.Tensor) or media.ndim != 5:
            raise RunError("Wan media must be a [B,F,C,H,W] tensor")
        return media, states, log_probs, timesteps

    def _normalize_flash_result(
        self,
        result: object,
        *,
        batch_size: int,
        selected_index: int,
    ):
        import torch

        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise RunError("Flash Wan pipeline must return exactly five values")
        media, all_states, all_log_probs, _all_kl, returned_index = result
        if type(returned_index) is not int or returned_index != selected_index:
            raise RunError("Flash Wan pipeline returned a different selected index")
        states = _stack_state_sequence(
            all_states,
            batch_size=batch_size,
            expected=2,
            name="latents",
        )
        log_probs = _stack_scalar_sequence(
            all_log_probs,
            batch_size=batch_size,
            expected=1,
            name="old_log_probs",
        )
        scheduler_timesteps = getattr(self.scheduler, "timesteps", None)
        if scheduler_timesteps is None or len(scheduler_timesteps) <= selected_index:
            raise RunError("Wan scheduler does not contain the selected timestep")
        timestep = torch.as_tensor(
            scheduler_timesteps[selected_index],
            device=log_probs.device,
            dtype=torch.int64,
        ).reshape(1, 1).expand(batch_size, 1).clone()
        coefficient = self._reference_coefficient(
            states[:, 0],
            states[:, 1],
            timestep[:, 0],
        )
        if not isinstance(media, torch.Tensor) or media.ndim != 5:
            raise RunError("Wan media must be a [B,F,C,H,W] tensor")
        return (
            media,
            states[:, :1],
            states[:, 1:],
            log_probs,
            timestep,
            coefficient,
        )

    def _reference_coefficient(self, latents, next_latents, timesteps):
        import torch

        result = self._load_sde_function()(
            self.scheduler,
            torch.zeros_like(latents),
            timesteps,
            latents.float(),
            prev_sample=next_latents.float(),
            return_dt_and_std_dev_t=True,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 6:
            raise RunError("Flash coefficient helper must return six values")
        return _normalize_coefficient(result[5], latents.shape[0])

    @contextmanager
    def _fork_rng(self, seed: int):
        import torch

        target = torch.device(self.device)
        cuda_devices = []
        if target.type == "cuda":
            cuda_devices = [
                target.index
                if target.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed(seed)
            yield

    def _ensure_loaded(self) -> None:
        if (
            self.pipeline is None
            or self.transformer is None
            or self.scheduler is None
        ):
            raise AdapterNotLoadedError(
                f"{self._COMPONENT_NAME} is not fully constructed"
            )

    def close(self) -> None:
        self.pipeline = None
        self.transformer = None
        self.scheduler = None
        self._frozen_modules_offloaded = False
        self._text_encoders_active = False
        self._vae_active = False
        self._policy_active = False


class WanFlashAdapter(_WanAdapterCore):
    """Wan2.1 selected-timestep policy used by Flash-GRPO."""

    MEDIA_TYPE: ClassVar[Literal["video"]] = "video"
    _COMPONENT_NAME = "wan_flash"
    _BACKEND = "flash"
    _FRAMES = None

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        return cls._resolve(raw, context, include_frames=True)

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> WanFlashAdapter:
        return cls._from_resolved(
            resolved,
            context,
            frames=int(resolved["frames"]),
        )


class WanWorldR1Adapter(_WanAdapterCore):
    """Wan2.1 full trajectory policy with frozen World-R1 camera semantics."""

    MEDIA_TYPE: ClassVar[Literal["video"]] = "video"
    WORLD_R1_FRAMES: ClassVar[int] = 81
    WORLD_R1_CAMERA_NOISE_WRAP: ClassVar[Mapping[str, object]] = FrozenMapping(
        {
            "remove_camera_keywords_from_prompt": False,
            "force_camera_movement": None,
            "noise_wrap_compute_dtype": "fp32",
            "noise_downtemp_interp": "nearest",
            "noise_downspatial_mode": "resize_noise",
            "noise_degradation": 0.35,
            "noise_wrap_flow_scale": 16,
            "wrap_strength": 0.35,
            "wrap_injection_mode": "stepwise_delta",
            "delta_lowpass_kernel": 9,
            "stepwise_guidance_steps": 8,
        }
    )
    _COMPONENT_NAME = "wan_world_r1"
    _BACKEND = "world_r1"
    _FRAMES = WORLD_R1_FRAMES

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        return cls._resolve(raw, context, include_frames=False)

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> WanWorldR1Adapter:
        return cls._from_resolved(
            resolved,
            context,
            frames=cls.WORLD_R1_FRAMES,
        )

    def _prepare_world_camera(
        self,
        prompts: tuple[str, ...],
        generator: object,
    ):
        from visual_rl.model_adapters.world_r1_camera import (
            build_stepwise_delta_callback,
            get_camera_trajectories_for_batch,
            lowpass_latent_delta,
            prepare_latents_with_camera,
        )

        _require_camera_signatures(
            get_camera_trajectories_for_batch,
            prepare_latents_with_camera,
            lowpass_latent_delta,
            build_stepwise_delta_callback,
        )
        config = self.WORLD_R1_CAMERA_NOISE_WRAP
        trajectories_result = get_camera_trajectories_for_batch(
            list(prompts),
            batch_size=len(prompts),
            frames_per_trajectory=self.WORLD_R1_FRAMES,
            force_camera_movement=config["force_camera_movement"],
        )
        if (
            not isinstance(trajectories_result, (tuple, list))
            or len(trajectories_result) != 4
        ):
            raise RunError(
                "World-R1 camera helper must return exactly four values"
            )
        trajectories, detected, _expanded, _profiles = trajectories_result
        camera = _camera_tensor(
            trajectories,
            batch_size=len(prompts),
            frames=self.WORLD_R1_FRAMES,
        )
        transformer_config = getattr(self.train_module, "config", None)
        channels = getattr(transformer_config, "in_channels", None)
        if type(channels) is not int or channels <= 0:
            raise RunError("Wan transformer.config.in_channels must be positive")
        prepared = prepare_latents_with_camera(
            prompt=list(prompts),
            batch_size=len(prompts),
            num_channels_latents=channels,
            height=self.height,
            width=self.width,
            num_frames=self.WORLD_R1_FRAMES,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
            latents=None,
            vae_scale_factor_temporal=getattr(
                self.pipeline,
                "vae_scale_factor_temporal",
                WAN_VAE_TEMPORAL_STRIDE,
            ),
            frames_per_trajectory=self.WORLD_R1_FRAMES,
            camera_trajectories=trajectories,
            detected_movements_batch=detected,
            remove_camera_keywords_from_prompt=config[
                "remove_camera_keywords_from_prompt"
            ],
            force_camera_movement=config["force_camera_movement"],
            noise_wrap_compute_dtype=config["noise_wrap_compute_dtype"],
            noise_downtemp_interp=config["noise_downtemp_interp"],
            noise_downspatial_mode=config["noise_downspatial_mode"],
            noise_degradation=config["noise_degradation"],
            noise_wrap_flow_scale=config["noise_wrap_flow_scale"],
            return_base_latents=True,
        )
        if not isinstance(prepared, (tuple, list)) or len(prepared) != 2:
            raise RunError(
                "prepare_latents_with_camera must return wrapped/base latents"
            )
        wrapped_latents, base_latents = prepared
        if config["wrap_strength"] <= 0:
            return base_latents, None, camera
        delta_low = lowpass_latent_delta(
            wrapped_latents - base_latents,
            config["delta_lowpass_kernel"],
        )
        callback = build_stepwise_delta_callback(
            delta_low=delta_low,
            wrap_strength=config["wrap_strength"],
            guidance_steps=config["stepwise_guidance_steps"],
        )
        if not callable(callback):
            raise RunError("World-R1 camera callback must be callable")
        return base_latents, callback, camera


def _resolve_path(
    value: object,
    context: ResolutionContext,
    key: str,
) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(
            f"model.params.{key} must be a non-empty path",
            key=f"model.params.{key}",
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = context.config_dir / path
    return path.resolve(strict=False)


def _require_path(values: Mapping[str, object], key: str) -> Path:
    value = values[key]
    if not isinstance(value, Path) or not value.is_absolute():
        raise ConfigError(
            f"resolved model.params.{key} must be an absolute Path",
            key=f"model.params.{key}",
        )
    return value


def _positive_int(value: object, key: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(
            f"model.params.{key} must be a positive integer",
            key=f"model.params.{key}",
        )
    return value


def _target_modules(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ConfigError(
            "model.params.lora_target_modules must be a non-empty string list",
            key="model.params.lora_target_modules",
        )
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise ConfigError(
            "model.params.lora_target_modules must not contain duplicates",
            key="model.params.lora_target_modules",
        )
    return normalized


def _selected_seed(seed: int, selected_index: int) -> int:
    return (seed + 1_000_003 * (selected_index + 1)) % (2**63 - 1)


def _select_rows(value, positions: Sequence[int]):
    import torch

    if not isinstance(value, torch.Tensor):
        raise RunError("Wan prompt embeddings must be tensors")
    index = torch.tensor(positions, dtype=torch.int64, device=value.device)
    return value.index_select(0, index)


def _restore_order(value, grouped_positions: Sequence[int]):
    import torch

    if not isinstance(value, torch.Tensor) or value.shape[0] != len(
        grouped_positions
    ):
        raise RunError("Wan grouped result does not match request rows")
    inverse = [0] * len(grouped_positions)
    for grouped_row, original_row in enumerate(grouped_positions):
        inverse[original_row] = grouped_row
    index = torch.tensor(inverse, dtype=torch.int64, device=value.device)
    return value.index_select(0, index)


def _module_dtype(module, *, fallback):
    parameter = next(iter(module.parameters()), None)
    return parameter.dtype if parameter is not None else fallback


def _wan_module_device_and_dtype(
    module,
    *,
    fallback_device,
    fallback_dtype,
):
    """Resolve Wan's activation contract from its input convolution.

    PEFT can enumerate a FP32 LoRA parameter before the frozen BF16 Wan
    transformer.  That parameter is not a valid activation-dtype oracle:
    feeding FP32 latents into Wan's BF16 ``patch_embedding`` fails before the
    LoRA branch is reached.
    """

    base_model = module
    get_base_model = getattr(module, "get_base_model", None)
    if callable(get_base_model):
        resolved = get_base_model()
        if resolved is not None:
            base_model = resolved

    patch_embedding = getattr(base_model, "patch_embedding", None)
    if patch_embedding is not None:
        parameter = next(iter(patch_embedding.parameters()), None)
        if parameter is not None:
            return parameter.device, parameter.dtype

    for name, parameter in base_model.named_parameters():
        if "patch_embedding." in name:
            return parameter.device, parameter.dtype
    for parameter in base_model.parameters():
        if not parameter.requires_grad:
            return parameter.device, parameter.dtype

    parameter = next(iter(module.parameters()), None)
    if parameter is not None:
        return parameter.device, parameter.dtype
    return fallback_device, fallback_dtype


def _cache_context(module, name: str):
    factory = getattr(module, "cache_context", None)
    return factory(name) if callable(factory) else nullcontext()


def _stack_state_sequence(
    values: object,
    *,
    batch_size: int,
    expected: int,
    name: str,
):
    import torch

    if isinstance(values, torch.Tensor):
        if values.shape[:2] == (batch_size, expected):
            return values
        raise RunError(f"Wan {name} tensor must have shape [B,{expected},...]")
    if not isinstance(values, (tuple, list)) or len(values) != expected:
        raise RunError(f"Wan {name} must contain {expected} states")
    tensors = []
    for value in values:
        if not isinstance(value, torch.Tensor) or value.shape[0] != batch_size:
            raise RunError(f"Wan {name} states must align with batch size")
        tensors.append(value)
    return torch.stack(tensors, dim=1)


def _scalar_transition(value: object, batch_size: int, name: str):
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.numel() != batch_size:
        raise RunError(f"Wan {name} must contain one value per row")
    return tensor.reshape(batch_size, 1)


def _stack_scalar_sequence(
    values: object,
    *,
    batch_size: int,
    expected: int,
    name: str,
):
    import torch

    if isinstance(values, torch.Tensor):
        if tuple(values.shape) == (batch_size, expected):
            return values
        raise RunError(f"Wan {name} must have shape [B,{expected}]")
    if not isinstance(values, (tuple, list)) or len(values) != expected:
        raise RunError(f"Wan {name} must contain {expected} transitions")
    return torch.cat(
        [_scalar_transition(value, batch_size, name) for value in values],
        dim=1,
    )


def _timesteps(
    values: object,
    *,
    batch_size: int,
    expected: int,
    device: object,
):
    import torch

    tensor = torch.as_tensor(values, dtype=torch.int64, device=device)
    if tensor.ndim == 1 and tensor.numel() == expected:
        return tensor[None, :].expand(batch_size, expected).clone()
    if tuple(tensor.shape) == (batch_size, expected):
        return tensor
    raise RunError(f"Wan timesteps must have shape [{expected}] or [B,{expected}]")


def _normalize_coefficient(value: object, batch_size: int):
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.numel() % batch_size:
        raise RunError("Flash coefficient does not align with batch size")
    flattened = tensor.reshape(batch_size, -1)
    coefficient = flattened[:, :1]
    if not torch.allclose(flattened, coefficient.expand_as(flattened)):
        raise RunError("Flash coefficient must be scalar per row")
    if not bool(torch.isfinite(coefficient).all()) or not bool(
        (coefficient > 0).all()
    ):
        raise RunError("Flash coefficient must be finite and positive")
    return coefficient


def _camera_tensor(
    trajectories: object,
    *,
    batch_size: int,
    frames: int,
):
    import torch

    if not isinstance(trajectories, list) or len(trajectories) != batch_size:
        raise RunError("World-R1 requires one camera trajectory per request row")
    rows: list[list[list[list[float]]]] = []
    expected_keys = [f"frame{index}" for index in range(frames)]
    for trajectory in trajectories:
        if not isinstance(trajectory, Mapping):
            raise RunError("World-R1 prompt did not resolve a camera movement")
        if list(trajectory) != expected_keys:
            raise RunError("World-R1 camera trajectory must contain frame0..frame80")
        row = [_parse_camera_matrix(trajectory[key]) for key in expected_keys]
        rows.append(row)
    tensor = torch.tensor(rows, dtype=torch.float64)
    if tuple(tensor.shape) != (batch_size, frames, 4, 4):
        raise RunError("World-R1 camera trajectory must have shape [B,F,4,4]")
    if not bool(torch.isfinite(tensor).all()):
        raise RunError("World-R1 camera trajectory must be finite")
    return tensor


def _parse_camera_matrix(value: object) -> list[list[float]]:
    if not isinstance(value, str):
        raise RunError("World-R1 camera matrix must be a string")
    columns = value.strip().split("] [")
    parsed_columns = []
    for column in columns:
        numbers = column.replace("[", "").replace("]", "").split()
        if len(numbers) != 4:
            raise RunError("World-R1 camera matrix columns must contain four values")
        try:
            parsed_columns.append([float(item) for item in numbers])
        except ValueError as exc:
            raise RunError("World-R1 camera matrix contains a non-number") from exc
    if len(parsed_columns) != 4:
        raise RunError("World-R1 camera matrix must contain four columns")
    return [
        [parsed_columns[column][row] for column in range(4)]
        for row in range(4)
    ]


def _require_camera_signatures(*functions: object) -> None:
    required = (
        (
            "get_camera_trajectories_for_batch",
            {
                "prompts",
                "batch_size",
                "frames_per_trajectory",
                "force_camera_movement",
            },
        ),
        (
            "prepare_latents_with_camera",
            {
                "prompt",
                "batch_size",
                "num_channels_latents",
                "height",
                "width",
                "num_frames",
                "dtype",
                "device",
                "generator",
                "latents",
                "vae_scale_factor_temporal",
                "frames_per_trajectory",
                "camera_trajectories",
                "detected_movements_batch",
                "remove_camera_keywords_from_prompt",
                "force_camera_movement",
                "noise_wrap_compute_dtype",
                "noise_downtemp_interp",
                "noise_downspatial_mode",
                "noise_degradation",
                "noise_wrap_flow_scale",
                "return_base_latents",
            },
        ),
        (
            "lowpass_latent_delta",
            {"delta", "kernel_size"},
        ),
        (
            "build_stepwise_delta_callback",
            {"delta_low", "wrap_strength", "guidance_steps"},
        ),
    )
    for (expected_name, expected_parameters), function in zip(
        required,
        functions,
        strict=True,
    ):
        if not callable(function):
            raise RunError(f"World-R1 helper {expected_name} must be callable")
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError) as exc:
            raise RunError(
                f"World-R1 helper {expected_name} has no inspectable signature"
            ) from exc
        missing = sorted(expected_parameters - set(parameters))
        if missing:
            raise RunError(
                f"World-R1 helper {expected_name} is missing parameters: "
                f"{missing}"
            )


__all__ = ["WanFlashAdapter", "WanWorldR1Adapter"]
