"""Final SD3/SD3.5 TempFlow model adapter."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import math
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
    reference_repo_import_path,
    resolve_torch_dtype,
    verify_gradient_checkpointing,
)

SD3_REFERENCE_NOISE_LEVEL = 0.7

_REQUIRED_PARAMS = frozenset(
    {
        "checkpoint",
        "reference_repo",
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
        reference_repo: Path,
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
    ) -> None:
        self.checkpoint = checkpoint
        self.reference_repo = reference_repo
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.gradient_checkpointing = gradient_checkpointing
        self.guidance_scale = guidance_scale
        self.resolution = resolution
        self.max_sequence_length = max_sequence_length
        self.local_files_only = local_files_only
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device = context.device
        self.dtype = resolve_torch_dtype(context.precision)
        self.pipeline = None
        self.transformer = None
        self._pipeline_full = None
        self._pipeline_branching = None
        self._sde_step = None
        self._encode_prompt = None
        self._gradient_checkpointing_state = None

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
        values["reference_repo"] = _resolve_path(
            values["reference_repo"],
            context,
            "reference_repo",
        )
        for key in ("lora_rank", "lora_alpha", "resolution", "max_sequence_length"):
            values[key] = _positive_int(values[key], key)
        values["lora_target_modules"] = _target_modules(
            values["lora_target_modules"]
        )
        for key in (
            "gradient_checkpointing",
            "local_files_only",
            "low_cpu_mem_usage",
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
        for key in ("checkpoint", "reference_repo"):
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
            reference_repo=_require_path(resolved, "reference_repo"),
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

        with reference_repo_import_path(self.reference_repo):
            from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import (
                pipeline_with_logprob,
            )
            from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_perstep import (
                pipeline_with_logprob as pipeline_with_logprob_perstep,
            )
            from flow_grpo.diffusers_patch.sd3_sde_with_logprob import (
                sde_step_with_logprob,
            )
            from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import (
                encode_prompt,
            )

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
        pipeline = pipeline.to(self.device)
        if torch.device(self.device).type == "cuda":
            for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                getattr(pipeline, name).to(self.device, dtype=self.dtype)
            pipeline.transformer.to(self.device)
        pipeline.vae.to(self.device, dtype=torch.float32)
        self.pipeline = pipeline
        self.transformer = pipeline.transformer
        self._pipeline_full = pipeline_with_logprob
        self._pipeline_branching = pipeline_with_logprob_perstep
        self._sde_step = sde_step_with_logprob
        self._encode_prompt = encode_prompt
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
            self.train_module.train(was_training)

    def _sample_full(self, request: RolloutRequest) -> RolloutBatch:
        import torch

        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._prompt_payload(request.prompts)
        generator = make_generator(self.device, request.context.seed)
        kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "num_inference_steps": request.num_steps,
            "guidance_scale": self.guidance_scale,
            "generator": generator,
            "output_type": "pt",
            "height": self.resolution,
            "width": self.resolution,
            "return_dict": False,
            "max_sequence_length": self.max_sequence_length,
            "kl_reward": 0.0,
        }
        with torch.no_grad(), self._bind_sde_generator(
            self._pipeline_full,
            generator,
        ):
            result = self._pipeline_full(self.pipeline, **kwargs)
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
        kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "num_inference_steps": request.num_steps,
            "guidance_scale": self.guidance_scale,
            "generator": generator,
            "output_type": "pt",
            "height": self.resolution,
            "width": self.resolution,
            "return_dict": False,
            "max_sequence_length": self.max_sequence_length,
            "kl_reward": 0.0,
        }
        with (
            torch.no_grad(),
            self._bind_branch_sde_generator(generator, request.group_size),
        ):
            result = self._pipeline_branching(self.pipeline, **kwargs)
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
        timestep_tensor = torch.stack(selected_timesteps).to(
            device=log_tensor.device,
            dtype=torch.int64,
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
        if require_reference:
            raise RunError("sd3_tempflow reference stats are added by stage 4")
        self._ensure_loaded()
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
        self.train_module.train(True)
        try:
            new_log_probs = []
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
            value = torch.cat(new_log_probs, dim=1).to(
                device=batch.old_log_probs.device
            )
            stats = PolicyRecomputeStats(new_log_probs=value)
            stats.validate_against(batch, require_reference=False)
            return stats
        finally:
            self.train_module.train(was_training)

    def _prompt_payload(self, prompts: tuple[str, ...]):
        if not callable(self._encode_prompt):
            raise RunError("SD3 prompt encoder is unavailable")
        encoders = [
            self.pipeline.text_encoder,
            self.pipeline.text_encoder_2,
            self.pipeline.text_encoder_3,
        ]
        tokenizers = [
            self.pipeline.tokenizer,
            self.pipeline.tokenizer_2,
            self.pipeline.tokenizer_3,
        ]
        prompt_embeds, pooled = self._encode_prompt(
            encoders,
            tokenizers,
            list(prompts),
            self.max_sequence_length,
        )
        negative_embeds, negative_pooled = self._encode_prompt(
            encoders,
            tokenizers,
            [""] * len(prompts),
            self.max_sequence_length,
        )
        return (
            prompt_embeds.to(self.device),
            pooled.to(self.device),
            negative_embeds.to(self.device),
            negative_pooled.to(self.device),
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

    @contextmanager
    def _bind_sde_generator(self, pipeline_function, generator):
        globals_mapping = _sde_globals(pipeline_function)
        original = (
            globals_mapping.get("sde_step_with_logprob")
            if globals_mapping is not None
            else None
        )
        if not callable(original):
            yield
            return

        def seeded(*args, **kwargs):
            deterministic = bool(kwargs.get("determistic", False))
            positional_previous = len(args) >= 5 and args[4] is not None
            positional_generator = len(args) >= 6 and args[5] is not None
            if (
                not deterministic
                and not positional_previous
                and not positional_generator
                and kwargs.get("prev_sample") is None
                and kwargs.get("generator") is None
            ):
                kwargs["generator"] = generator
            return original(*args, **kwargs)

        globals_mapping["sde_step_with_logprob"] = seeded
        try:
            yield
        finally:
            if globals_mapping.get("sde_step_with_logprob") is seeded:
                globals_mapping["sde_step_with_logprob"] = original

    @contextmanager
    def _bind_branch_sde_generator(self, generator, branch_count: int):
        globals_mapping = _sde_globals(self._pipeline_branching)
        original = (
            globals_mapping.get("sde_step_with_logprob")
            if globals_mapping is not None
            else None
        )
        if not callable(original) or not callable(self._sde_step):
            raise RunError("SD3 branching SDE helper cannot be bound")
        ordinary = self._sde_step
        bound_generator = generator

        def shared(
            scheduler,
            model_output,
            timestep,
            sample,
            prev_sample=None,
            generator=None,
            determistic=False,
        ):
            import torch

            if determistic or prev_sample is not None:
                return ordinary(
                    scheduler,
                    model_output,
                    timestep,
                    sample,
                    prev_sample=sample if determistic else prev_sample,
                    generator=None,
                    determistic=determistic,
                )
            parent_count = sample.shape[0]
            expanded_sample = sample.repeat_interleave(branch_count, dim=0)
            expanded_output = model_output.repeat_interleave(branch_count, dim=0)
            timestep_values = torch.as_tensor(
                timestep,
                device=sample.device,
            ).reshape(-1)
            if timestep_values.numel() == 1:
                expanded_timestep = timestep_values.expand(
                    parent_count * branch_count
                )
            elif timestep_values.numel() == parent_count:
                expanded_timestep = timestep_values.repeat_interleave(
                    branch_count
                )
            else:
                raise RunError("SD3 branch timestep does not align with parents")
            return ordinary(
                scheduler,
                expanded_output,
                expanded_timestep,
                expanded_sample,
                generator=(
                    generator if generator is not None else bound_generator
                ),
                determistic=False,
            )

        globals_mapping["sde_step_with_logprob"] = shared
        try:
            yield
        finally:
            if globals_mapping.get("sde_step_with_logprob") is shared:
                globals_mapping["sde_step_with_logprob"] = original

    def _ensure_loaded(self) -> None:
        if (
            self.pipeline is None
            or self.transformer is None
            or not callable(self._pipeline_full)
            or not callable(self._pipeline_branching)
            or not callable(self._sde_step)
        ):
            raise AdapterNotLoadedError("sd3_tempflow is not fully constructed")

    def close(self) -> None:
        self.pipeline = None
        self.transformer = None
        self._pipeline_full = None
        self._pipeline_branching = None
        self._sde_step = None
        self._encode_prompt = None


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
    import torch

    values = _scheduler_values(scheduler, expected=expected)
    return values.to(device=device, dtype=torch.int64)[None, :].expand(
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


def _sde_globals(function: object) -> dict[str, Any] | None:
    seen: set[int] = set()
    current = function
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        mapping = getattr(current, "__globals__", None)
        if isinstance(mapping, dict) and callable(
            mapping.get("sde_step_with_logprob")
        ):
            return mapping
        current = getattr(current, "__wrapped__", None)
    return None


__all__ = [
    "SD3_REFERENCE_NOISE_LEVEL",
    "SD3TempFlowAdapter",
]
