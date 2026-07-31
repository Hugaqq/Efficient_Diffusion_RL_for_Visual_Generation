"""Final SD3/SD3.5 TempFlow model adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager
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
    sd3_sde_step_with_logprob,
)

SD3_REFERENCE_NOISE_LEVEL = 0.7

_REQUIRED_PARAMS = frozenset(
    {
        "checkpoint",
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
        "gradient_checkpointing",
        "guidance_scale",
        "resolution",
        "max_sequence_length",
    }
)
_DEFAULT_PARAMS: Mapping[str, object] = {
    "local_files_only": True,
    "low_cpu_mem_usage": True,
    "offload_frozen_modules_during_update": False,
    "policy_forward_microbatch_size": None,
}
_PROMPT_PAYLOAD_KEYS = (
    "prompt_embeds",
    "pooled_prompt_embeds",
    "negative_prompt_embeds",
    "negative_pooled_prompt_embeds",
)


class SD3TempFlowAdapter(ModelAdapter):
    """One LoRA-only SD3 policy supporting full and shared-prefix rollouts."""

    MEDIA_TYPE: ClassVar[Literal["image"]] = "image"
    SD3_REFERENCE_NOISE_LEVEL: ClassVar[float] = SD3_REFERENCE_NOISE_LEVEL

    def __init__(
        self,
        *,
        checkpoint: Path,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: tuple[str, ...],
        gradient_checkpointing: bool,
        guidance_scale: float,
        resolution: int,
        max_sequence_length: int,
        local_files_only: bool,
        low_cpu_mem_usage: bool,
        context: RuntimeBuildContext,
        offload_frozen_modules_during_update: bool = False,
        policy_forward_microbatch_size: int | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.gradient_checkpointing = gradient_checkpointing
        self.guidance_scale = guidance_scale
        self.resolution = resolution
        self.max_sequence_length = max_sequence_length
        self.local_files_only = local_files_only
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.offload_frozen_modules_during_update = (
            offload_frozen_modules_during_update
        )
        self.policy_forward_microbatch_size = policy_forward_microbatch_size
        self.device = context.device
        self.dtype = resolve_torch_dtype(context.precision)
        self.pipeline = None
        self.transformer = None
        self._sde_step = None
        self._encode_prompt = None
        self._gradient_checkpointing_state = None
        self._policy_dtype_hook = None
        self._frozen_modules_offloaded = False
        self._text_encoders_active = False
        self._vae_active = False
        self._policy_active = False

    @property
    def train_module(self):
        if self.transformer is None:
            raise AdapterNotLoadedError("sd3_tempflow is not fully constructed")
        return self.transformer

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        if not isinstance(raw, Mapping):
            raise ConfigError("model.params must be a mapping", key="model.params")
        allowed = set(_REQUIRED_PARAMS) | set(_DEFAULT_PARAMS)
        unknown = sorted(set(raw) - allowed)
        missing = sorted(set(_REQUIRED_PARAMS) - set(raw))
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
        values = {**_DEFAULT_PARAMS, **dict(raw)}
        values["checkpoint"] = _resolve_path(
            values["checkpoint"],
            context,
            "checkpoint",
        )
        for key in ("lora_rank", "lora_alpha", "resolution", "max_sequence_length"):
            values[key] = _positive_int(values[key], key)
        if values["policy_forward_microbatch_size"] is not None:
            values["policy_forward_microbatch_size"] = _positive_int(
                values["policy_forward_microbatch_size"],
                "policy_forward_microbatch_size",
            )
        values["lora_target_modules"] = _target_modules(
            values["lora_target_modules"]
        )
        for key in (
            "gradient_checkpointing",
            "local_files_only",
            "low_cpu_mem_usage",
            "offload_frozen_modules_during_update",
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
        ):
            raise ConfigError(
                "model.params.guidance_scale must be finite",
                key="model.params.guidance_scale",
            )
        values["guidance_scale"] = float(guidance)
        return FrozenMapping(values)

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, object],
        context: ValidationContext,
    ) -> tuple[ValidationCheck, ...]:
        del cls, context
        checks = []
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
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> SD3TempFlowAdapter:
        adapter = cls(
            checkpoint=_require_path(resolved, "checkpoint"),
            lora_rank=int(resolved["lora_rank"]),
            lora_alpha=int(resolved["lora_alpha"]),
            lora_target_modules=tuple(resolved["lora_target_modules"]),
            gradient_checkpointing=bool(resolved["gradient_checkpointing"]),
            guidance_scale=float(resolved["guidance_scale"]),
            resolution=int(resolved["resolution"]),
            max_sequence_length=int(resolved["max_sequence_length"]),
            local_files_only=bool(resolved["local_files_only"]),
            low_cpu_mem_usage=bool(resolved["low_cpu_mem_usage"]),
            context=context,
            offload_frozen_modules_during_update=bool(
                resolved["offload_frozen_modules_during_update"]
            ),
            policy_forward_microbatch_size=(
                None
                if resolved["policy_forward_microbatch_size"] is None
                else int(resolved["policy_forward_microbatch_size"])
            ),
        )
        adapter._load_base_pipeline()
        return adapter

    def _load_base_pipeline(self) -> None:
        import torch

        try:
            from diffusers import StableDiffusion3Pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install visual-rl[train] to use sd3_tempflow"
            ) from exc

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            str(self.checkpoint),
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )
        for module_name in (
            "vae",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
        ):
            module = getattr(pipeline, module_name)
            module.requires_grad_(False)
            module.eval()
        base_transformer = pipeline.transformer
        checkpointing_state = configure_gradient_checkpointing(
            base_transformer,
            self.gradient_checkpointing,
            context="sd3_tempflow base transformer",
        )
        pipeline.transformer = apply_peft_lora(
            base_transformer,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
        )
        if (
            self.offload_frozen_modules_during_update
            and torch.device(self.device).type == "cuda"
        ):
            for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                getattr(pipeline, name).to("cpu", dtype=self.dtype)
            pipeline.transformer.to(self.device)
            pipeline.vae.to("cpu", dtype=torch.float32)
            self._frozen_modules_offloaded = True
            self._policy_active = True
        else:
            pipeline = pipeline.to(self.device)
            if torch.device(self.device).type == "cuda":
                for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                    getattr(pipeline, name).to(self.device, dtype=self.dtype)
                pipeline.transformer.to(self.device)
            pipeline.vae.to(self.device, dtype=torch.float32)
            self._text_encoders_active = True
            self._vae_active = True
            self._policy_active = True
        self._install_policy_dtype_guard(pipeline.transformer)
        self.pipeline = pipeline
        self.transformer = pipeline.transformer
        self._sde_step = sd3_sde_step_with_logprob
        self._encode_prompt = pipeline.encode_prompt
        self._gradient_checkpointing_state = verify_gradient_checkpointing(
            self.transformer,
            checkpointing_state,
            context="sd3_tempflow active transformer",
        )

    def sample(self, request: RolloutRequest) -> RolloutBatch:
        self._ensure_loaded()
        if request.kind not in {"full_trajectory", "branching"}:
            raise RunError(
                "sd3_tempflow only supports full_trajectory and branching"
            )
        was_training = self.train_module.training
        self.train_module.eval()
        try:
            if request.kind == "full_trajectory":
                batch = self._sample_full(request)
            else:
                batch = self._sample_branches(request)
            batch.validate_against(request)
            return batch
        finally:
            try:
                self.train_module.train(was_training)
            finally:
                self._offload_frozen_modules_for_update()

    def _sample_full(self, request: RolloutRequest) -> RolloutBatch:
        import torch

        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._prompt_payload(request.prompts)
        generator = make_generator(self.device, request.context.seed)
        with torch.no_grad():
            result = self._run_full_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_steps=request.num_steps,
                generator=generator,
            )
        if not isinstance(result, (tuple, list)) or len(result) not in {3, 4}:
            raise RunError("SD3 full pipeline must return three or four values")
        media, states_raw, log_probs_raw = result[:3]
        log_probs = _stack_scalars(
            log_probs_raw,
            batch_size=len(request.prompts),
            expected=request.num_steps,
            name="old_log_probs",
        )
        states = _stack_states(
            states_raw,
            batch_size=len(request.prompts),
            expected=request.num_steps + 1,
            name="latents",
        )
        timesteps = _scheduler_timesteps(
            self.pipeline.scheduler,
            batch_size=len(request.prompts),
            expected=request.num_steps,
            device=log_probs.device,
        )
        if not isinstance(media, torch.Tensor) or media.ndim != 4:
            raise RunError("SD3 tensor output must have shape [B,C,H,W]")
        payload = {
            name: value.detach()
            for name, value in zip(
                _PROMPT_PAYLOAD_KEYS,
                (
                    prompt_embeds,
                    pooled_prompt_embeds,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ),
                strict=True,
            )
        }
        return RolloutBatch(
            prompts=request.prompts,
            metadata=request.metadata,
            media=media.detach(),
            latents=states[:, :-1].detach(),
            next_latents=states[:, 1:].detach(),
            timesteps=timesteps.detach(),
            old_log_probs=log_probs.detach(),
            transition_mask=torch.ones_like(log_probs, dtype=torch.bool).detach(),
            sample_id=request.sample_id,
            prompt_id=request.prompt_id,
            group_id=request.group_id,
            branch_id=request.branch_id,
            media_layout="BCHW",
            camera_trajectory=None,
            context=request.context,
            selected_timestep_index=None,
            flash_coefficient=None,
            branch_step_index=None,
            trajectory_step_index=None,
            transition_std_dev=None,
            recompute_payload=payload,
            artifact_metadata={
                "adapter": "sd3_tempflow",
                "resolution": self.resolution,
            },
        )

    def _sample_branches(self, request: RolloutRequest) -> RolloutBatch:
        import torch

        if request.branch_step_index is None:
            raise RunError("SD3 branching request is missing branch indices")
        if len(set(request.branch_step_index)) != 1:
            raise RunError(
                "SD3 branching requires one shared global branch timestep"
            )
        parent_rows, row_to_parent, row_branch = _group_layout(request)
        parent_prompts = tuple(request.prompts[row] for row in parent_rows)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._prompt_payload(parent_prompts)
        generator = make_generator(self.device, request.context.seed)
        with torch.no_grad():
            result = self._run_branching_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_steps=request.num_steps,
                branch_count=request.group_size,
                generator=generator,
            )
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise RunError("SD3 branching pipeline must return exactly five values")
        branch_media, main_states_raw, branch_states, log_probs, _kls = result
        main_states = _state_sequence(main_states_raw)
        transition_count = len(log_probs)
        if transition_count != request.num_steps - 1:
            raise RunError(
                "SD3 branching must expose num_steps - 1 candidate transitions"
            )
        if (
            len(main_states) != request.num_steps
            or len(branch_states) != transition_count
            or len(branch_media) != transition_count
        ):
            raise RunError("SD3 branching outputs have inconsistent step counts")

        scheduler_values = _scheduler_values(
            self.pipeline.scheduler,
            expected=request.num_steps,
        )
        selected_sources = []
        selected_targets = []
        selected_logs = []
        selected_media = []
        selected_timesteps = []
        selected_std = []
        exploration_width: int | None = None
        parent_count = len(parent_rows)
        for row, (parent, branch) in enumerate(
            zip(row_to_parent, row_branch, strict=True)
        ):
            step = request.branch_step_index[row]
            source = main_states[step][parent]
            target_step = torch.as_tensor(branch_states[step])
            log_step = torch.as_tensor(log_probs[step]).reshape(-1)
            media_step = torch.as_tensor(branch_media[step])
            if target_step.shape[0] % parent_count:
                raise RunError("SD3 branch output is not grouped by parent")
            width = target_step.shape[0] // parent_count
            if exploration_width is None:
                exploration_width = width
            elif exploration_width != width:
                raise RunError("SD3 exploration width changed between steps")
            if branch >= width:
                raise RunError(
                    "SD3 reference pipeline returned fewer branches than requested"
                )
            index = parent * width + branch
            if log_step.numel() == parent_count:
                log_index = parent
            elif log_step.numel() == parent_count * width:
                log_index = index
            else:
                raise RunError("SD3 branch log-probs do not align with parents")
            selected_sources.append(source)
            selected_targets.append(target_step[index])
            selected_logs.append(log_step[log_index])
            selected_media.append(media_step[index])
            timestep = scheduler_values[step]
            selected_timesteps.append(timestep)
            selected_std.append(self._reference_transition_std(timestep))

        source_tensor = torch.stack(selected_sources)[:, None]
        target_tensor = torch.stack(selected_targets)[:, None]
        log_tensor = torch.stack(selected_logs).reshape(len(request.prompts), 1)
        media_tensor = torch.stack(selected_media)
        # FlowMatch schedulers can expose fractional timesteps (for example
        # 833.3333).  Recompute must receive the exact scheduler value: casting
        # it to int64 makes index_for_timestep() fail on a later training step.
        timestep_tensor = torch.stack(selected_timesteps).to(
            device=log_tensor.device,
        )[:, None]
        std_tensor = torch.stack(selected_std).to(
            device=log_tensor.device,
            dtype=log_tensor.dtype,
        )[:, None]
        parent_index = torch.tensor(
            row_to_parent,
            dtype=torch.int64,
            device=prompt_embeds.device,
        )
        payload = {
            "prompt_embeds": prompt_embeds.index_select(0, parent_index).detach(),
            "pooled_prompt_embeds": pooled_prompt_embeds.index_select(
                0,
                parent_index,
            ).detach(),
            "negative_prompt_embeds": negative_prompt_embeds.index_select(
                0,
                parent_index,
            ).detach(),
            "negative_pooled_prompt_embeds": (
                negative_pooled_prompt_embeds.index_select(
                    0,
                    parent_index,
                ).detach()
            ),
        }
        branch_tensor = torch.tensor(
            request.branch_step_index,
            dtype=torch.int64,
            device=log_tensor.device,
        )
        return RolloutBatch(
            prompts=request.prompts,
            metadata=request.metadata,
            media=media_tensor.detach(),
            latents=source_tensor.detach(),
            next_latents=target_tensor.detach(),
            timesteps=timestep_tensor.detach(),
            old_log_probs=log_tensor.detach(),
            transition_mask=torch.ones_like(log_tensor, dtype=torch.bool).detach(),
            sample_id=request.sample_id,
            prompt_id=request.prompt_id,
            group_id=request.group_id,
            branch_id=request.branch_id,
            media_layout="BCHW",
            camera_trajectory=None,
            context=request.context,
            selected_timestep_index=None,
            flash_coefficient=None,
            branch_step_index=branch_tensor,
            trajectory_step_index=branch_tensor[:1].clone(),
            transition_std_dev=std_tensor.detach(),
            recompute_payload=payload,
            artifact_metadata={
                "adapter": "sd3_tempflow",
                "resolution": self.resolution,
            },
        )

    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        self._ensure_loaded()
        self._offload_frozen_modules_for_update()
        self._activate_policy_module()
        payload = []
        for key in _PROMPT_PAYLOAD_KEYS:
            value = batch.recompute_payload.get(key)
            if value is None:
                raise RunError(f"SD3 recompute payload is missing {key}")
            payload.append(value.to(self.device))
        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = payload

        import torch

        was_training = self.train_module.training
        # Rollout and recompute must use the same deterministic module mode.
        # ``eval`` disables dropout but does not disable autograd, so LoRA
        # gradients still flow while old/new log-prob remain comparable.
        self.train_module.eval()
        try:
            new_log_probs = []
            current_means = []
            transition_stds = []
            for step in range(batch.transition_count):
                latent = batch.latents[:, step].to(self.device)
                timestep = batch.timesteps[:, step].to(self.device)
                next_latent = batch.next_latents[:, step].to(self.device)
                prediction = self._predict_noise(
                    latent,
                    timestep,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                )
                result = self._sde_step(
                    self.pipeline.scheduler,
                    prediction.float(),
                    timestep,
                    latent.float(),
                    prev_sample=next_latent.float(),
                )
                if not isinstance(result, (tuple, list)) or len(result) < 2:
                    raise RunError("SD3 transition helper returned an invalid result")
                new_log_probs.append(
                    _scalar_transition(
                        result[1],
                        batch.batch_size,
                        "new_log_probs",
                    )
                )
                if require_reference:
                    if len(result) < 4:
                        raise RunError(
                            "SD3 transition helper did not return mean/std"
                        )
                    current_means.append(
                        _transition_tensor(
                            result[2],
                            batch.batch_size,
                            "current_transition_mean",
                        )
                    )
                    transition_stds.append(
                        _transition_tensor(
                            result[3],
                            batch.batch_size,
                            "transition_std",
                        )
                    )
            value = torch.cat(new_log_probs, dim=1).to(
                device=batch.old_log_probs.device
            )
            if require_reference:
                reference_means = []
                with torch.no_grad(), self._disable_lora_reference():
                    for step in range(batch.transition_count):
                        latent = batch.latents[:, step].to(self.device)
                        timestep = batch.timesteps[:, step].to(self.device)
                        next_latent = batch.next_latents[:, step].to(self.device)
                        prediction = self._predict_noise(
                            latent,
                            timestep,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            negative_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        )
                        result = self._sde_step(
                            self.pipeline.scheduler,
                            prediction.float(),
                            timestep,
                            latent.float(),
                            prev_sample=next_latent.float(),
                        )
                        if (
                            not isinstance(result, (tuple, list))
                            or len(result) < 3
                        ):
                            raise RunError(
                                "SD3 reference transition helper did not "
                                "return a mean"
                            )
                        reference_means.append(
                            _transition_tensor(
                                result[2],
                                batch.batch_size,
                                "reference_transition_mean",
                            )
                        )
                output_device = batch.old_log_probs.device
                current_mean = torch.stack(current_means, dim=1).to(
                    device=output_device
                )
                reference_mean = (
                    torch.stack(reference_means, dim=1)
                    .to(device=output_device)
                    .detach()
                )
                transition_std = (
                    torch.stack(transition_stds, dim=1)
                    .to(device=output_device)
                    .detach()
                )
                stats = PolicyRecomputeStats(
                    new_log_probs=value,
                    current_transition_mean=current_mean,
                    transition_std=transition_std,
                    reference_transition_mean=reference_mean,
                )
            else:
                stats = PolicyRecomputeStats(new_log_probs=value)
            stats.validate_against(
                batch,
                require_reference=require_reference,
            )
            return stats
        finally:
            self.train_module.train(was_training)

    @contextmanager
    def _disable_lora_reference(self):
        disable_adapter = getattr(self.train_module, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RunError(
                "sd3_tempflow reference statistics require PEFT "
                "disable_adapter()"
            )
        context = disable_adapter()
        if not hasattr(context, "__enter__") or not hasattr(context, "__exit__"):
            raise RunError(
                "sd3_tempflow disable_adapter() must return a context manager"
            )
        with context:
            yield

    def _run_full_pipeline(
        self,
        *,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        num_steps: int,
        generator,
    ):
        timesteps, latents = self._prepare_native_sd3_rollout(
            prompt_embeds=prompt_embeds,
            num_steps=num_steps,
            generator=generator,
        )
        states = [latents]
        log_probs = []
        for timestep in timesteps:
            expanded = timestep.expand(latents.shape[0])
            prediction = self._predict_noise(
                latents,
                expanded,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            )
            latents, log_prob, _mean, _std = sd3_sde_step_with_logprob(
                self.pipeline.scheduler,
                prediction.float(),
                expanded,
                latents.float(),
                generator=generator,
            )
            states.append(latents)
            log_probs.append(log_prob)
        return self._decode_sd3_latents(latents), states, log_probs

    def _run_branching_pipeline(
        self,
        *,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        num_steps: int,
        branch_count: int,
        generator,
    ):
        import torch

        timesteps, latents = self._prepare_native_sd3_rollout(
            prompt_embeds=prompt_embeds,
            num_steps=num_steps,
            generator=generator,
        )
        main_states = [latents]
        branch_states = []
        branch_log_probs = []
        branch_final_latents = []
        parent_count = latents.shape[0]
        positive_branch = prompt_embeds.repeat_interleave(branch_count, dim=0)
        pooled_branch = pooled_prompt_embeds.repeat_interleave(
            branch_count,
            dim=0,
        )
        negative_branch = negative_prompt_embeds.repeat_interleave(
            branch_count,
            dim=0,
        )
        negative_pooled_branch = (
            negative_pooled_prompt_embeds.repeat_interleave(
                branch_count,
                dim=0,
            )
        )
        for index, timestep in enumerate(timesteps[:-1]):
            expanded = timestep.expand(parent_count)
            prediction = self._predict_noise(
                latents,
                expanded,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            )
            ode_latents = sd3_sde_step_with_logprob(
                self.pipeline.scheduler,
                prediction.float(),
                expanded,
                latents.float(),
                deterministic=True,
            )[0]
            branch_source = latents.repeat_interleave(branch_count, dim=0)
            branch_prediction = prediction.repeat_interleave(
                branch_count,
                dim=0,
            )
            branch_timestep = timestep.expand(parent_count * branch_count)
            sampled, log_prob, _mean, _std = sd3_sde_step_with_logprob(
                self.pipeline.scheduler,
                branch_prediction.float(),
                branch_timestep,
                branch_source.float(),
                generator=generator,
            )
            branch_states.append(sampled)
            branch_log_probs.append(log_prob)

            branch_latents = sampled
            for inner_timestep in timesteps[index + 1 :]:
                inner_expanded = inner_timestep.expand(branch_latents.shape[0])
                inner_prediction = self._predict_noise(
                    branch_latents,
                    inner_expanded,
                    positive_branch,
                    pooled_branch,
                    negative_branch,
                    negative_pooled_branch,
                )
                branch_latents = sd3_sde_step_with_logprob(
                    self.pipeline.scheduler,
                    inner_prediction.float(),
                    inner_expanded,
                    branch_latents.float(),
                    deterministic=True,
                )[0]
            branch_final_latents.append(branch_latents)
            latents = ode_latents
            main_states.append(latents)
        # Branching exposes one reward candidate for every possible branch
        # timestep.  Decode them in one VAE phase so CPU offload moves the
        # policy and VAE exactly once per training step rather than once per
        # candidate timestep.
        branch_media = self._decode_sd3_latent_sequence(branch_final_latents)
        zeros = [torch.zeros_like(value) for value in branch_log_probs]
        return (
            branch_media,
            main_states,
            branch_states,
            branch_log_probs,
            zeros,
        )

    def _prepare_native_sd3_rollout(
        self,
        *,
        prompt_embeds,
        num_steps: int,
        generator,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
            calculate_shift,
            retrieve_timesteps,
        )

        pipeline = self.pipeline
        pipeline._guidance_scale = self.guidance_scale
        pipeline._skip_layer_guidance_scale = 0.0
        pipeline._clip_skip = None
        pipeline._joint_attention_kwargs = None
        pipeline._interrupt = False
        channels = getattr(getattr(self.train_module, "config", None), "in_channels", None)
        if type(channels) is not int or channels <= 0:
            raise RunError("SD3 transformer.config.in_channels must be positive")
        latents = pipeline.prepare_latents(
            prompt_embeds.shape[0],
            channels,
            self.resolution,
            self.resolution,
            prompt_embeds.dtype,
            self.device,
            generator,
            None,
        ).float()
        scheduler_kwargs = {}
        scheduler_config = pipeline.scheduler.config
        if bool(scheduler_config.get("use_dynamic_shifting", False)):
            patch_size = int(self.train_module.config.patch_size)
            image_sequence_length = (
                latents.shape[-2] // patch_size
            ) * (latents.shape[-1] // patch_size)
            scheduler_kwargs["mu"] = calculate_shift(
                image_sequence_length,
                scheduler_config.get("base_image_seq_len", 256),
                scheduler_config.get("max_image_seq_len", 4096),
                scheduler_config.get("base_shift", 0.5),
                scheduler_config.get("max_shift", 1.16),
            )
        timesteps, _ = retrieve_timesteps(
            pipeline.scheduler,
            num_steps,
            self.device,
            **scheduler_kwargs,
        )
        pipeline._num_timesteps = len(timesteps)
        return timesteps, latents

    def _decode_sd3_latents(self, latents):
        return self._decode_sd3_latent_sequence((latents,))[0]

    def _decode_sd3_latent_sequence(self, latent_sequence):
        values = tuple(latent_sequence)
        if not values:
            raise RunError("SD3 decode sequence must not be empty")
        self._activate_vae_for_decode()
        try:
            return [
                self._decode_sd3_latents_with_active_vae(latents)
                for latents in values
            ]
        finally:
            self._offload_vae_after_decode()

    def _decode_sd3_latents_with_active_vae(self, latents):
        vae = self.pipeline.vae
        parameter = next(iter(vae.parameters()))
        value = (
            latents / vae.config.scaling_factor
        ) + vae.config.shift_factor
        value = value.to(device=parameter.device, dtype=parameter.dtype)
        image = vae.decode(value, return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(
            image,
            output_type="pt",
        )

    def _prompt_payload(self, prompts: tuple[str, ...]):
        if not callable(self._encode_prompt):
            raise RunError("SD3 prompt encoder is unavailable")
        self._activate_text_encoders_for_prompt()
        try:
            result = self._encode_prompt(
                prompt=list(prompts),
                prompt_2=None,
                prompt_3=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=[""] * len(prompts),
                negative_prompt_2=None,
                negative_prompt_3=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                clip_skip=None,
                max_sequence_length=self.max_sequence_length,
                lora_scale=None,
            )
        finally:
            self._offload_text_encoders()
            self._activate_policy_module()
        if not isinstance(result, tuple) or len(result) != 4:
            raise RunError("SD3 encode_prompt must return exactly four tensors")
        prompt_embeds, negative_embeds, pooled, negative_pooled = result
        return (
            prompt_embeds.to(device=self.device, dtype=self.dtype),
            pooled.to(device=self.device, dtype=self.dtype),
            negative_embeds.to(device=self.device, dtype=self.dtype),
            negative_pooled.to(device=self.device, dtype=self.dtype),
        )

    def _activate_text_encoders_for_prompt(self) -> None:
        if not self.offload_frozen_modules_during_update:
            return

        import torch

        if torch.device(self.device).type != "cuda":
            return
        if self._text_encoders_active:
            return
        if self._policy_active:
            self.train_module.to("cpu")
            self._policy_active = False
            torch.cuda.empty_cache()
        for module_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            getattr(self.pipeline, module_name).to(self.device, dtype=self.dtype)
        self._text_encoders_active = True
        self._frozen_modules_offloaded = False

    def _offload_text_encoders(self) -> None:
        if not self.offload_frozen_modules_during_update:
            return

        import torch

        if torch.device(self.device).type != "cuda":
            return
        if not self._text_encoders_active:
            return
        for module_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            getattr(self.pipeline, module_name).to("cpu", dtype=self.dtype)
        torch.cuda.empty_cache()
        self._text_encoders_active = False
        self._frozen_modules_offloaded = not self._vae_active

    def _activate_policy_module(self) -> None:
        if not self.offload_frozen_modules_during_update:
            return

        import torch

        if torch.device(self.device).type != "cuda" or self._policy_active:
            return
        if self._text_encoders_active:
            raise RunError(
                "SD3 policy cannot move to CUDA while text encoders are active"
            )
        self.train_module.to(self.device)
        self._policy_active = True

    def _activate_vae_for_decode(self) -> None:
        if not self.offload_frozen_modules_during_update:
            return

        import torch

        if torch.device(self.device).type != "cuda" or self._vae_active:
            return
        if self._text_encoders_active:
            raise RunError(
                "SD3 VAE cannot move to CUDA while text encoders are active"
            )
        if self._policy_active:
            self.train_module.to("cpu")
            self._policy_active = False
            torch.cuda.empty_cache()
        try:
            self.pipeline.vae.to(self.device, dtype=torch.float32)
        except BaseException:
            # Do not leave a partially moved VAE beside a disabled policy.
            self.pipeline.vae.to("cpu", dtype=torch.float32)
            torch.cuda.empty_cache()
            self._activate_policy_module()
            raise
        self._vae_active = True
        self._frozen_modules_offloaded = False

    def _offload_vae_after_decode(self) -> None:
        if not self.offload_frozen_modules_during_update:
            return

        import torch

        if torch.device(self.device).type != "cuda":
            return
        if self._vae_active:
            self.pipeline.vae.to("cpu", dtype=torch.float32)
            torch.cuda.empty_cache()
            self._vae_active = False
        self._frozen_modules_offloaded = not self._text_encoders_active
        self._activate_policy_module()

    def _offload_frozen_modules_for_update(self) -> None:
        self._offload_text_encoders()
        self._offload_vae_after_decode()

    def _install_policy_dtype_guard(self, transformer) -> None:
        """Keep reference-pipeline latent forwards at the policy precision."""

        import torch

        if self._policy_dtype_hook is not None:
            raise RuntimeError("SD3 policy dtype guard is already installed")

        def match_hidden_state_dtype(_module, args, kwargs):
            hidden_states = kwargs.get("hidden_states")
            if (
                not isinstance(hidden_states, torch.Tensor)
                or not hidden_states.is_floating_point()
                or hidden_states.dtype == self.dtype
            ):
                return None
            updated = dict(kwargs)
            updated["hidden_states"] = hidden_states.to(dtype=self.dtype)
            return args, updated

        self._policy_dtype_hook = transformer.register_forward_pre_hook(
            match_hidden_state_dtype,
            with_kwargs=True,
        )

    def _predict_noise(
        self,
        latent,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    ):
        import torch

        batch_size = int(latent.shape[0])
        microbatch_size = self.policy_forward_microbatch_size
        if microbatch_size is None or microbatch_size >= batch_size:
            return self._predict_noise_microbatch(
                latent,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            )
        outputs = []
        for start in range(0, batch_size, microbatch_size):
            stop = min(start + microbatch_size, batch_size)
            outputs.append(
                self._predict_noise_microbatch(
                    latent[start:stop],
                    timestep[start:stop],
                    prompt_embeds[start:stop],
                    pooled_prompt_embeds[start:stop],
                    negative_prompt_embeds[start:stop],
                    negative_pooled_prompt_embeds[start:stop],
                )
            )
        return torch.cat(outputs, dim=0)

    def _predict_noise_microbatch(
        self,
        latent,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    ):
        import torch

        if self.guidance_scale > 1.0:
            hidden_states = torch.cat((latent, latent))
            model_timestep = torch.cat((timestep, timestep))
            embeddings = torch.cat((negative_prompt_embeds, prompt_embeds))
            pooled = torch.cat(
                (negative_pooled_prompt_embeds, pooled_prompt_embeds)
            )
        else:
            hidden_states = latent
            model_timestep = timestep
            embeddings = prompt_embeds
            pooled = pooled_prompt_embeds
        result = self.train_module(
            hidden_states=hidden_states,
            timestep=model_timestep,
            encoder_hidden_states=embeddings,
            pooled_projections=pooled,
            return_dict=False,
        )[0]
        if self.guidance_scale > 1.0:
            negative, positive = result.chunk(2)
            result = negative + self.guidance_scale * (positive - negative)
        return result

    def _reference_transition_std(self, timestep):
        import torch

        scheduler = self.pipeline.scheduler
        try:
            index = int(scheduler.index_for_timestep(timestep))
        except Exception as exc:
            raise RunError("SD3 branch timestep is absent from scheduler") from exc
        sigmas = torch.as_tensor(scheduler.sigmas)
        if index + 1 >= sigmas.numel():
            raise RunError("SD3 branch timestep has no successor sigma")
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        sigma_max = sigmas[1].to(sigma)
        denominator = torch.where(sigma == 1, sigma_max, sigma)
        base_variance = sigma / (1 - denominator)
        delta = sigma - sigma_next
        if not bool((base_variance > 0).item()) or not bool((delta > 0).item()):
            raise RunError("SD3 branch variance must be positive")
        return (
            torch.sqrt(base_variance)
            * self.SD3_REFERENCE_NOISE_LEVEL
            * torch.sqrt(delta)
        )

    def _ensure_loaded(self) -> None:
        if (
            self.pipeline is None
            or self.transformer is None
            or not callable(self._sde_step)
        ):
            raise AdapterNotLoadedError("sd3_tempflow is not fully constructed")

    def close(self) -> None:
        dtype_hook = self._policy_dtype_hook
        self._policy_dtype_hook = None
        if dtype_hook is not None:
            dtype_hook.remove()
        self.pipeline = None
        self.transformer = None
        self._sde_step = None
        self._encode_prompt = None
        self._frozen_modules_offloaded = False
        self._text_encoders_active = False
        self._vae_active = False
        self._policy_active = False


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
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise ConfigError(
            "model.params.lora_target_modules must not contain duplicates",
            key="model.params.lora_target_modules",
        )
    return result


def _state_sequence(values: object) -> list[Any]:
    import torch

    if isinstance(values, torch.Tensor):
        if values.ndim < 2:
            raise RunError("SD3 state tensor has no step axis")
        return [values[:, index] for index in range(values.shape[1])]
    if not isinstance(values, (tuple, list)) or not values:
        raise RunError("SD3 state sequence must be non-empty")
    return [torch.as_tensor(value) for value in values]


def _stack_states(
    values: object,
    *,
    batch_size: int,
    expected: int,
    name: str,
):
    import torch

    states = _state_sequence(values)
    if len(states) != expected:
        raise RunError(f"SD3 {name} must contain {expected} states")
    if any(state.shape[0] != batch_size for state in states):
        raise RunError(f"SD3 {name} states must align with batch size")
    return torch.stack(states, dim=1)


def _scalar_transition(value: object, batch_size: int, name: str):
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.numel() != batch_size:
        raise RunError(f"SD3 {name} must contain one value per row")
    return tensor.reshape(batch_size, 1)


def _transition_tensor(value: object, batch_size: int, name: str):
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 0 or tensor.shape[0] != batch_size:
        raise RunError(f"SD3 {name} must have leading batch dimension B")
    if not tensor.is_floating_point():
        raise RunError(f"SD3 {name} must be floating point")
    return tensor


def _stack_scalars(
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
        raise RunError(f"SD3 {name} must have shape [B,{expected}]")
    if not isinstance(values, (tuple, list)) or len(values) != expected:
        raise RunError(f"SD3 {name} must contain {expected} transitions")
    return torch.cat(
        [_scalar_transition(value, batch_size, name) for value in values],
        dim=1,
    )


def _scheduler_values(scheduler: object, *, expected: int):
    import torch

    values = torch.as_tensor(getattr(scheduler, "timesteps", None)).reshape(-1)
    if values.numel() != expected:
        raise RunError(f"SD3 scheduler must expose {expected} timesteps")
    return values


def _scheduler_timesteps(
    scheduler: object,
    *,
    batch_size: int,
    expected: int,
    device: object,
):
    values = _scheduler_values(scheduler, expected=expected)
    return values.to(device=device)[None, :].expand(
        batch_size,
        expected,
    ).clone()


def _group_layout(
    request: RolloutRequest,
) -> tuple[list[int], list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for row, group_id in enumerate(request.group_id):
        groups.setdefault(group_id, []).append(row)
    parent_rows = [rows[0] for rows in groups.values()]
    group_to_parent = {
        group_id: index for index, group_id in enumerate(groups)
    }
    row_to_parent = [group_to_parent[group_id] for group_id in request.group_id]
    row_branch = [0] * len(request.prompts)
    for rows in groups.values():
        for branch, row in enumerate(rows):
            row_branch[row] = branch
    return parent_rows, row_to_parent, row_branch


__all__ = [
    "SD3_REFERENCE_NOISE_LEVEL",
    "SD3TempFlowAdapter",
]
