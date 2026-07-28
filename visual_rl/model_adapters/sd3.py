"""SD3/SD3.5 TempFlow adapter backed by the verified reference path."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.diffusers_common import (
    AdapterNotLoadedError,
    GradientCheckpointingState,
    apply_peft_lora,
    configure_gradient_checkpointing,
    gradient_checkpointing_metadata,
    gradient_checkpointing_request,
    make_generator,
    require_model_path,
    resolve_torch_dtype,
    stack_steps,
    trainable_parameters,
    validate_gradient_checkpointing_checkpoint_metadata,
    verify_gradient_checkpointing,
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

SD3_TRANSITION_CONTRACT_VERSION = "sd3_tempflow_v3"
SD3_REFERENCE_NOISE_LEVEL = 0.7
SD3_CHECKPOINT_FORMAT_VERSION = 1
SD3_CHECKPOINT_METADATA_NAME = "adapter_metadata.json"
SD3_LORA_CHECKPOINT_SUBDIR = "transformer"
SD3_LORA_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


class SD3TempFlowAdapter(ModelAdapter):
    """In-process wrapper around TempFlow-GRPO's SD3 patched pipeline."""

    name = "tempflow_sd3_legacy"
    media_type = "image"

    def __init__(self, config: dict[str, Any]):
        import torch

        self.config = config
        extra = dict(config.get("extra", {}))
        self.gradient_checkpointing_requested = gradient_checkpointing_request(config)
        self._gradient_checkpointing_state = GradientCheckpointingState(
            requested=self.gradient_checkpointing_requested,
            effective=None,
        )
        self.repo_root = resolve_legacy_repo(
            config.get(
                "repo_root", extra.get("repo_root", "reference_code/TempFlow-GRPO-main")
            )
        )
        self.device = torch.device(
            config.get(
                "device",
                extra.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            )
        )
        self.dtype = resolve_torch_dtype(
            config.get(
                "dtype",
                extra.get(
                    "dtype", "bfloat16" if self.device.type == "cuda" else "float32"
                ),
            )
        )
        self.resolution = int(config.get("resolution", extra.get("resolution", 256)))
        self.max_sequence_length = int(extra.get("max_sequence_length", 128))
        self.use_lora = bool(config.get("use_lora", extra.get("use_lora", True)))
        self.lora_rank = int(extra.get("lora_rank", 32))
        self.lora_alpha = int(extra.get("lora_alpha", self.lora_rank * 2))
        self.lora_path = config.get("lora_path", extra.get("lora_path"))
        self.lora_targets = list(
            extra.get("lora_target_modules", DEFAULT_SD3_LORA_TARGETS)
        )
        self.tempflow_reference_mode = bool(
            config.get(
                "tempflow_reference_mode",
                extra.get("tempflow_reference_mode", False),
            )
        )
        self.pipeline = None
        self.transformer = None
        self._pipeline_with_logprob = None
        self._pipeline_with_logprob_perstep = None
        self._sde_step_with_logprob = None
        self._encode_prompt = None
        if not extra.get("defer_load", False):
            self.load()

    def load(self) -> None:
        import torch

        try:
            from diffusers import StableDiffusion3Pipeline
        except ImportError as exc:  # pragma: no cover - optional train dependency
            raise ImportError(
                "Install visual-rl[train] to use SD3 TempFlow adapter."
            ) from exc

        model_path = require_model_path(self.config, self.name)
        with legacy_repo_path(self.repo_root):
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

        self._pipeline_with_logprob = pipeline_with_logprob
        self._pipeline_with_logprob_perstep = pipeline_with_logprob_perstep
        self._sde_step_with_logprob = sde_step_with_logprob
        self._encode_prompt = encode_prompt
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=self.dtype
        )
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        self.pipeline.text_encoder_3.requires_grad_(False)
        checkpointing_state = configure_gradient_checkpointing(
            self.pipeline.transformer,
            self.gradient_checkpointing_requested,
            context="SD3 base transformer",
        )
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
        self._gradient_checkpointing_state = verify_gradient_checkpointing(
            self.transformer,
            checkpointing_state,
            context="SD3 active transformer after PEFT attach",
        )
        self.pipeline.vae.eval()
        self.pipeline.text_encoder.eval()
        self.pipeline.text_encoder_2.eval()
        self.pipeline.text_encoder_3.eval()
        self.transformer.eval()

    def _ensure_loaded(self) -> None:
        if self.pipeline is None or self.transformer is None:
            raise AdapterNotLoadedError(
                "SD3 TempFlow adapter is not loaded. Provide model_path or call load()."
            )

    @property
    def train_module(self):
        if self.transformer is None:
            raise AdapterNotLoadedError(
                "SD3 TempFlow adapter has no train module before load()."
            )
        return self.transformer

    def parameters(self):
        params = trainable_parameters(self.train_module)
        if not params:
            raise AdapterNotLoadedError(
                "SD3 TempFlow adapter has no trainable parameters."
            )
        return params

    def named_parameters(self):
        return [
            (f"transformer.{name}", parameter)
            for name, parameter in self.train_module.named_parameters()
            if parameter.requires_grad
        ]

    def prepare_for_sampling(self) -> None:
        self.eval()

    def prepare_for_training(self) -> None:
        if self.tempflow_reference_mode:
            self.train()

    def branch_transition_count(self, rollout_config: dict[str, Any]) -> int:
        transition_count = int(rollout_config.get("num_steps", 3)) - 1
        if transition_count < 1:
            raise ValueError("SD3 TempFlow branching requires num_steps >= 2")
        return transition_count

    def sample(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> RolloutBatch:
        import torch

        self._ensure_loaded()
        self.prepare_for_sampling()
        if self.tempflow_reference_mode:
            self._validate_reference_noise_level(
                rollout_config.get("noise_level", SD3_REFERENCE_NOISE_LEVEL)
            )
        num_steps = int(rollout_config.get("num_steps", 3))
        guidance_scale = float(rollout_config.get("guidance_scale", 4.5))
        output_type = str(rollout_config.get("output_type", "pt"))
        generator = make_generator(self.device, rollout_config.get("seed"))

        with torch.no_grad(), self._full_sde_generator(generator):
            prompt_embeds, pooled_prompt_embeds = self._encode_text(prompts)
            neg_prompt_embeds, neg_pooled_prompt_embeds = self._encode_text(
                [""] * len(prompts)
            )
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
                raise ValueError(
                    f"Unexpected SD3 TempFlow pipeline result length: {len(result)}"
                )
            log_probs = stack_steps(log_probs, dim=1)
            kls = stack_steps(kls, dim=1) if kls else torch.zeros_like(log_probs)

            raw_states = self._step_sequence(latents)
            transition_count = int(log_probs.shape[1])
            if len(raw_states) != transition_count + 1:
                raise ValueError(
                    "SD3 full trajectory must return one more state than log-prob "
                    f"transitions: {len(raw_states)} != {transition_count + 1}"
                )
            source_dtype = raw_states[0].dtype
            source_latents = torch.stack(
                [state.to(dtype=source_dtype) for state in raw_states[:-1]],
                dim=1,
            )
            target_latents = torch.stack(raw_states[1:], dim=1)
            scheduler_context = self._scheduler_context_tensors()
            timesteps = self._expanded_transition_timesteps(
                scheduler_context["scheduler_timesteps"],
                batch_size=len(prompts),
                transition_count=transition_count,
                device=log_probs.device,
            )

        return RolloutBatch(
            prompts=list(prompts),
            metadata=[dict(item) for item in metadata],
            media=images.detach() if isinstance(images, torch.Tensor) else images,
            latents=source_latents.detach(),
            next_latents=target_latents.detach(),
            timesteps=timesteps,
            old_log_probs=log_probs.detach(),
            kl=kls.detach(),
            model_metadata={
                **self._gradient_checkpointing_metadata(),
                "adapter": self.name,
                "reference_repo": str(self.repo_root),
                "reference_pipeline": "sd3_pipeline_with_logprob",
                "resolution": self.resolution,
                "guidance_scale": guidance_scale,
                "trajectory_contract_version": SD3_TRANSITION_CONTRACT_VERSION,
                "trajectory_source_dtype": str(source_latents.dtype),
                "trajectory_target_dtype": str(target_latents.dtype),
                "transition_count": transition_count,
                "transition_kernel": self._transition_kernel_identity(),
                "scheduler_class": self._scheduler_identity(),
                "transformer_training": bool(self.transformer.training),
                "recompute_transformer_training": bool(self.tempflow_reference_mode),
                "tempflow_reference_mode": self.tempflow_reference_mode,
            },
            model_tensors={
                "prompt_embeds": prompt_embeds.detach(),
                "pooled_prompt_embeds": pooled_prompt_embeds.detach(),
                "negative_prompt_embeds": neg_prompt_embeds.detach(),
                "negative_pooled_prompt_embeds": neg_pooled_prompt_embeds.detach(),
                **scheduler_context,
            },
        )

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        import torch

        self._ensure_loaded()
        self.prepare_for_training()
        guidance_scale = float(batch.model_metadata.get("guidance_scale", 4.5))
        prompt_embeds, pooled_prompt_embeds = self._batch_embeddings(
            batch, positive=True
        )
        neg_prompt_embeds, neg_pooled_prompt_embeds = self._batch_embeddings(
            batch, positive=False
        )
        latents = batch.latents.to(self.device)
        next_latents = batch.next_latents.to(self.device)
        timesteps = batch.timesteps.to(self.device)
        self._validate_recompute_contract(batch, latents, next_latents)
        self._validate_scheduler_context(batch)
        if (
            batch.model_metadata.get("branching_mode") == "shared_prefix"
            and not self.tempflow_reference_mode
        ):
            return self._recompute_shared_prefix_log_probs(
                batch,
                latents=latents,
                next_latents=next_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                guidance_scale=guidance_scale,
            )

        log_probs = []
        transformer_dtype = self._transformer_dtype()
        legacy_contract = (
            batch.model_metadata.get("trajectory_contract_version")
            != SD3_TRANSITION_CONTRACT_VERSION
        )
        for index in range(latents.shape[1]):
            latent_step = latents[:, index]
            if legacy_contract:
                latent_step = latent_step.to(dtype=transformer_dtype)
            noise_pred = self._predict_noise(
                latent_step=latent_step,
                timestep=timesteps[:, index],
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                guidance_scale=guidance_scale,
            )
            _prev_sample, log_prob, _mean, _std = self._sde_step_with_logprob(
                self.pipeline.scheduler,
                noise_pred.float(),
                timesteps[:, index],
                latent_step.float(),
                prev_sample=next_latents[:, index].float(),
            )
            log_probs.append(log_prob)
        return torch.stack(log_probs, dim=1)

    def sample_branching(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> RolloutBatch:
        import torch

        self._ensure_loaded()
        self.prepare_for_sampling()
        if self._pipeline_with_logprob_perstep is None:
            raise RuntimeError(
                "TempFlow SD3 per-step branching pipeline is not available."
            )
        if bool(rollout_config.get("include_main", False)):
            raise ValueError("SD3 TempFlow branching does not expose a main sample")
        branch_step_index = int(rollout_config["branch_step_index"])
        branch_count = int(rollout_config["branch_count"])
        num_steps = int(rollout_config.get("num_steps", 3))
        guidance_scale = float(rollout_config.get("guidance_scale", 4.5))
        noise_level = self._validate_reference_noise_level(
            rollout_config.get("noise_level", SD3_REFERENCE_NOISE_LEVEL)
        )
        generator = make_generator(self.device, rollout_config.get("seed"))

        with (
            torch.no_grad(),
            self._perstep_sde_generator(
                generator,
                branch_count=branch_count,
            ),
            self._transformer_input_dtype(),
        ):
            prompt_embeds, pooled_prompt_embeds = self._encode_text(prompts)
            negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode_text(
                [""] * len(prompts)
            )
            result = self._call_pipeline_with_logprob_perstep(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type=str(rollout_config.get("output_type", "pt")),
                height=self.resolution,
                width=self.resolution,
                kl_reward=float(rollout_config.get("kl_reward", 0.0)),
            )
        if len(result) != 5:
            raise ValueError(
                f"SD3 per-step branching pipeline must return 5 items, got {len(result)}"
            )
        branch_media, main_latents, sde_latents, log_probs, kls = result
        if branch_step_index >= len(log_probs):
            raise ValueError(
                f"branch_step_index {branch_step_index} exceeds returned transitions {len(log_probs)}"
            )

        raw_main_states = self._step_sequence(main_latents)
        scheduler_context = self._scheduler_context_tensors()
        scheduler_timesteps = scheduler_context["scheduler_timesteps"]
        expected_transition_count = int(scheduler_timesteps.numel()) - 1
        if len(raw_main_states) != int(scheduler_timesteps.numel()):
            raise ValueError(
                "SD3 branching must return one main state per scheduler timestep: "
                f"{len(raw_main_states)} != {int(scheduler_timesteps.numel())}"
            )
        if not (
            len(sde_latents) == len(log_probs) == len(kls) == expected_transition_count
        ):
            raise ValueError(
                "SD3 branching transition outputs must all equal T-1: "
                f"sde={len(sde_latents)}, log_probs={len(log_probs)}, "
                f"kl={len(kls)}, expected={expected_transition_count}"
            )

        selected_next = torch.as_tensor(sde_latents[branch_step_index])
        exploration_k, gather_indices = self._branch_gather_indices(
            selected_next.shape[0],
            len(prompts),
            branch_count,
        )
        gather = torch.as_tensor(
            gather_indices,
            device=selected_next.device,
            dtype=torch.long,
        )
        parent_indices = [
            parent_index
            for parent_index in range(len(prompts))
            for _ in range(branch_count)
        ]
        branch_ids = list(range(branch_count)) * len(prompts)
        expanded_prompts = [prompts[index] for index in parent_indices]
        scheduler_timestep = scheduler_timesteps[branch_step_index].detach()
        timestep_value = scheduler_timestep.cpu().item()
        expanded_metadata = []
        for row, parent_index in enumerate(parent_indices):
            item = dict(metadata[parent_index])
            item.update(
                {
                    "parent_prompt_index": parent_index,
                    "branch_id": branch_ids[row],
                    "branch_step_index": branch_step_index,
                    "branch_timestep_value": timestep_value,
                    "is_main_branch": False,
                    "rollout_kind": "tempflow_branching",
                }
            )
            expanded_metadata.append(item)

        source_dtype = raw_main_states[0].dtype
        parent_latents = raw_main_states[branch_step_index].to(dtype=source_dtype)
        selected_latents = parent_latents.repeat_interleave(branch_count, dim=0)
        selected_next = selected_next.index_select(0, gather)
        selected_log_probs = torch.as_tensor(log_probs[branch_step_index]).index_select(
            0,
            gather.to(torch.as_tensor(log_probs[branch_step_index]).device),
        )
        selected_kl = self._select_branch_kl(
            torch.as_tensor(kls[branch_step_index]),
            gather_indices,
            len(prompts),
            branch_count,
        )
        selected_media = self._select_branch_media(
            branch_media[branch_step_index],
            gather_indices,
        )
        rows = len(expanded_prompts)
        transition_count = int(rollout_config.get("transition_count", len(log_probs)))
        positions = torch.arange(transition_count, dtype=torch.float32)
        global_weights = torch.sqrt(
            (transition_count - positions).clamp_min(1.0) / float(transition_count)
        )
        global_weights = global_weights / global_weights.mean().clamp_min(1e-6)
        noise_weight = float(global_weights[branch_step_index])
        transition_std_dev_t = float(
            self._reference_transition_std_dev_t(
                scheduler_timestep,
                noise_level=noise_level,
            )
            .detach()
            .cpu()
        )
        selected_timesteps = (
            scheduler_timestep.to(device=selected_log_probs.device)
            .reshape(1, 1)
            .expand(rows, 1)
            .clone()
        )
        return RolloutBatch(
            prompts=expanded_prompts,
            metadata=expanded_metadata,
            media=selected_media,
            latents=selected_latents[:, None].detach(),
            next_latents=selected_next[:, None].detach(),
            timesteps=selected_timesteps,
            old_log_probs=selected_log_probs.reshape(rows, 1).detach(),
            kl=selected_kl.reshape(rows, 1).detach(),
            branch_id=torch.as_tensor(branch_ids, dtype=torch.long),
            model_metadata={
                **self._gradient_checkpointing_metadata(),
                "adapter": self.name,
                "reference_repo": str(self.repo_root),
                "reference_pipeline": "sd3_pipeline_with_logprob_perstep",
                "resolution": self.resolution,
                "guidance_scale": guidance_scale,
                "branching_mode": "shared_prefix",
                "branch_step_index": branch_step_index,
                "branch_timestep_value": timestep_value,
                "trajectory_step_indices": [branch_step_index],
                "transition_count": transition_count,
                "noise_weights": [noise_weight],
                "transition_std_dev_t": [transition_std_dev_t],
                "sde_generator_bound": generator is not None,
                "reference_exploration_k": exploration_k,
                "branch_count": branch_count,
                "trajectory_contract_version": SD3_TRANSITION_CONTRACT_VERSION,
                "trajectory_source_dtype": str(selected_latents.dtype),
                "trajectory_target_dtype": str(selected_next.dtype),
                "transition_kernel": self._transition_kernel_identity(),
                "scheduler_class": self._scheduler_identity(),
                "transformer_training": bool(self.transformer.training),
                "recompute_transformer_training": bool(self.tempflow_reference_mode),
                "tempflow_reference_mode": self.tempflow_reference_mode,
                "branch_rng_consumption": "dynamic_branch_count",
            },
            model_tensors={
                "prompt_embeds": prompt_embeds.repeat_interleave(
                    branch_count, dim=0
                ).detach(),
                "pooled_prompt_embeds": pooled_prompt_embeds.repeat_interleave(
                    branch_count, dim=0
                ).detach(),
                "negative_prompt_embeds": negative_prompt_embeds.repeat_interleave(
                    branch_count, dim=0
                ).detach(),
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds.repeat_interleave(
                    branch_count,
                    dim=0,
                ).detach(),
                "parent_prompt_indices": torch.as_tensor(
                    parent_indices,
                    dtype=torch.long,
                    device=selected_latents.device,
                ),
                "initial_latents": raw_main_states[0].detach(),
                **scheduler_context,
            },
        )

    def save_pretrained(self, output_dir: str) -> None:
        self._ensure_loaded()
        checkpointing_metadata = self._gradient_checkpointing_metadata()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        from visual_rl.artifacts.checkpoint import save_json

        if self.use_lora:
            peft_config = getattr(self.transformer, "peft_config", None)
            save_adapter = getattr(self.transformer, "save_pretrained", None)
            if not peft_config or not callable(save_adapter):
                raise RuntimeError(
                    "SD3 use_lora=True requires a loaded PEFT transformer with "
                    "save_pretrained support"
                )
            adapter_path = path / SD3_LORA_CHECKPOINT_SUBDIR
            save_adapter(adapter_path, safe_serialization=True)
            weight_path = self._peft_adapter_weight_path(adapter_path)
            config_path = adapter_path / "adapter_config.json"
            if weight_path is None or not config_path.is_file():
                raise RuntimeError(
                    "SD3 PEFT save_pretrained did not write a complete adapter "
                    f"checkpoint under {adapter_path}"
                )
            save_json(
                path / SD3_CHECKPOINT_METADATA_NAME,
                {
                    **checkpointing_metadata,
                    "adapter": self.name,
                    "adapter_config": str(config_path.relative_to(path)),
                    "adapter_name": "default",
                    "format_version": SD3_CHECKPOINT_FORMAT_VERSION,
                    "save_kind": "peft_adapter",
                    "use_lora": True,
                    "weights": str(weight_path.relative_to(path)),
                },
            )
            return

        import torch

        if hasattr(self.transformer, "save_pretrained"):
            self.transformer.save_pretrained(path / SD3_LORA_CHECKPOINT_SUBDIR)
        state_path = path / "transformer_state.pt"
        torch.save(self.transformer.state_dict(), state_path)
        save_json(
            path / SD3_CHECKPOINT_METADATA_NAME,
            {
                **checkpointing_metadata,
                "adapter": self.name,
                "format_version": SD3_CHECKPOINT_FORMAT_VERSION,
                "save_kind": "full_transformer_state",
                "state": state_path.name,
                "use_lora": False,
            },
        )

    def _gradient_checkpointing_metadata(self) -> dict[str, bool | None]:
        (
            self._gradient_checkpointing_state,
            metadata,
        ) = gradient_checkpointing_metadata(
            self.train_module,
            self._gradient_checkpointing_state,
            context="SD3 active transformer",
        )
        return metadata

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        import torch

        self._ensure_loaded()
        path = Path(checkpoint_dir)
        metadata = self._checkpoint_metadata(path)
        self._gradient_checkpointing_metadata()
        validate_gradient_checkpointing_checkpoint_metadata(
            metadata,
            self._gradient_checkpointing_state,
            context="SD3 checkpoint",
        )
        state_path = path / "transformer_state.pt"
        if state_path.is_file():
            state = torch.load(
                state_path,
                map_location=self.device,
                weights_only=True,
            )
            self.transformer.load_state_dict(state)
            return

        if not self.use_lora:
            raise RuntimeError(
                "Missing SD3 full transformer checkpoint for use_lora=False: "
                f"expected {state_path}"
            )

        peft_config = getattr(self.transformer, "peft_config", None)
        if not peft_config:
            raise RuntimeError(
                "Cannot load an SD3 LoRA checkpoint into a non-PEFT transformer"
            )
        adapter_path = self._find_peft_adapter_path(path)
        if adapter_path is None:
            expected = path / SD3_LORA_CHECKPOINT_SUBDIR
            raise RuntimeError(
                "Missing SD3 checkpoint weights: expected legacy "
                f"{state_path} or a PEFT adapter under {expected} (or {path})"
            )

        try:
            from peft.utils.save_and_load import (
                get_peft_model_state_dict,
                load_peft_weights,
                set_peft_model_state_dict,
            )
        except ImportError as exc:  # pragma: no cover - optional train dependency
            raise RuntimeError(
                "Install visual-rl[train] with PEFT to load an SD3 LoRA checkpoint"
            ) from exc

        parameter_ids = {
            name: id(parameter)
            for name, parameter in self.transformer.named_parameters()
        }
        try:
            expected_state = self._get_default_peft_state(get_peft_model_state_dict)
            adapter_state = load_peft_weights(str(adapter_path), device="cpu")
            if not adapter_state:
                raise ValueError("adapter weight file contains no tensors")
            self._validate_peft_checkpoint_state(expected_state, adapter_state)
            load_result = set_peft_model_state_dict(
                self.transformer,
                adapter_state,
                adapter_name="default",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SD3 PEFT adapter checkpoint from {adapter_path}: {exc}"
            ) from exc

        current_parameter_ids = {
            name: id(parameter)
            for name, parameter in self.transformer.named_parameters()
        }
        if current_parameter_ids != parameter_ids:
            raise RuntimeError(
                "SD3 PEFT checkpoint loading replaced Parameter objects; "
                "optimizer references would be invalid"
            )
        restored_state = self._get_default_peft_state(get_peft_model_state_dict)
        self._validate_peft_checkpoint_state(adapter_state, restored_state)
        self._validate_peft_checkpoint_values(adapter_state, restored_state)
        unexpected = list(getattr(load_result, "unexpected_keys", ()) or ())
        if unexpected:
            raise RuntimeError(
                "SD3 PEFT checkpoint contains unexpected keys: "
                + ", ".join(str(key) for key in unexpected)
            )

    def _get_default_peft_state(self, get_peft_model_state_dict) -> Mapping[str, Any]:
        parameters = inspect.signature(get_peft_model_state_dict).parameters
        kwargs = {"adapter_name": "default"} if "adapter_name" in parameters else {}
        state = get_peft_model_state_dict(self.transformer, **kwargs)
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError("PEFT returned an empty SD3 default adapter state")
        return state

    @classmethod
    def _validate_peft_checkpoint_state(
        cls,
        expected_state: Any,
        checkpoint_state: Any,
    ) -> None:
        expected = cls._peft_state_shapes(expected_state, label="attached adapter")
        actual = cls._peft_state_shapes(checkpoint_state, label="checkpoint adapter")
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        mismatched = sorted(
            key
            for key in expected.keys() & actual.keys()
            if expected[key] != actual[key]
        )
        if missing:
            raise RuntimeError(
                "SD3 PEFT checkpoint is missing adapter keys: " + ", ".join(missing)
            )
        if extra:
            raise RuntimeError(
                "SD3 PEFT checkpoint contains extra adapter keys: " + ", ".join(extra)
            )
        if mismatched:
            details = ", ".join(
                f"{key}: expected {expected[key]}, got {actual[key]}"
                for key in mismatched
            )
            raise RuntimeError(f"SD3 PEFT checkpoint shape mismatch: {details}")

    @staticmethod
    def _peft_state_shapes(state: Any, *, label: str) -> dict[str, tuple[int, ...]]:
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(f"SD3 PEFT {label} state must be a non-empty mapping")
        shapes: dict[str, tuple[int, ...]] = {}
        for key, value in state.items():
            if not isinstance(key, str) or not key:
                raise RuntimeError(f"SD3 PEFT {label} state contains an invalid key")
            shape = getattr(value, "shape", None)
            if shape is None:
                raise RuntimeError(
                    f"SD3 PEFT {label} state value has no tensor shape: {key}"
                )
            shapes[key] = tuple(int(dimension) for dimension in shape)
        return shapes

    @staticmethod
    def _validate_peft_checkpoint_values(
        checkpoint_state: Mapping[str, Any],
        restored_state: Mapping[str, Any],
    ) -> None:
        import torch

        for key, expected in checkpoint_state.items():
            actual = restored_state[key]
            expected_tensor = torch.as_tensor(expected).detach().cpu()
            actual_tensor = (
                torch.as_tensor(actual).detach().cpu().to(dtype=expected_tensor.dtype)
            )
            if not torch.equal(actual_tensor, expected_tensor):
                raise RuntimeError(
                    f"SD3 PEFT checkpoint value was not restored exactly: {key}"
                )

    @staticmethod
    def _checkpoint_metadata(path: Path) -> dict[str, Any]:
        metadata_path = path / SD3_CHECKPOINT_METADATA_NAME
        try:
            metadata_path.lstat()
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise RuntimeError(
                f"Cannot inspect SD3 checkpoint metadata: {metadata_path}"
            ) from exc
        from visual_rl.artifacts.checkpoint import load_json, validated_checkpoint_file

        validated = validated_checkpoint_file(
            path,
            metadata_path,
            label="SD3 adapter_metadata.json",
        )
        metadata = load_json(validated)
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"SD3 checkpoint metadata must be an object: {metadata_path}"
            )
        return metadata

    @staticmethod
    def _peft_adapter_weight_path(path: Path) -> Path | None:
        for name in SD3_LORA_WEIGHT_NAMES:
            candidate = path / name
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def _find_peft_adapter_path(cls, checkpoint_path: Path) -> Path | None:
        for candidate in (
            checkpoint_path / SD3_LORA_CHECKPOINT_SUBDIR,
            checkpoint_path,
        ):
            if (
                candidate / "adapter_config.json"
            ).is_file() and cls._peft_adapter_weight_path(candidate) is not None:
                return candidate
        return None

    def _call_pipeline_with_logprob(self, **kwargs):
        signature = inspect.signature(self._pipeline_with_logprob)
        supported = set(signature.parameters)
        call_kwargs = {key: value for key, value in kwargs.items() if key in supported}
        if "return_dict" in supported:
            call_kwargs["return_dict"] = False
        return self._pipeline_with_logprob(self.pipeline, **call_kwargs)

    def _call_pipeline_with_logprob_perstep(self, **kwargs):
        signature = inspect.signature(self._pipeline_with_logprob_perstep)
        supported = set(signature.parameters)
        call_kwargs = {key: value for key, value in kwargs.items() if key in supported}
        if "return_dict" in supported:
            call_kwargs["return_dict"] = False
        return self._pipeline_with_logprob_perstep(self.pipeline, **call_kwargs)

    @staticmethod
    def _step_sequence(values):
        import torch

        if isinstance(values, torch.Tensor):
            if values.ndim < 2:
                raise ValueError(
                    "Trajectory tensor must expose a step dimension at axis 1"
                )
            return [values[:, index] for index in range(values.shape[1])]
        states = [torch.as_tensor(value) for value in values]
        if not states:
            raise ValueError("Trajectory must contain at least one state")
        return states

    def _scheduler_context_tensors(self) -> dict[str, Any]:
        import torch

        scheduler = self.pipeline.scheduler
        if not hasattr(scheduler, "timesteps"):
            raise ValueError("SD3 scheduler does not expose active timesteps")
        context = {
            "scheduler_timesteps": torch.as_tensor(scheduler.timesteps).detach().clone()
        }
        if hasattr(scheduler, "sigmas"):
            context["scheduler_sigmas"] = (
                torch.as_tensor(scheduler.sigmas).detach().clone()
            )
        return context

    @staticmethod
    def _expanded_transition_timesteps(
        scheduler_timesteps,
        *,
        batch_size: int,
        transition_count: int,
        device,
    ):
        import torch

        values = torch.as_tensor(scheduler_timesteps).reshape(-1)
        if values.numel() != transition_count:
            raise ValueError(
                "SD3 full trajectory timestep count must equal transition count: "
                f"{values.numel()} != {transition_count}"
            )
        return (
            values.to(device=device)
            .reshape(1, transition_count)
            .expand(batch_size, transition_count)
            .clone()
            .detach()
        )

    def _scheduler_identity(self) -> str:
        scheduler_type = type(self.pipeline.scheduler)
        return f"{scheduler_type.__module__}.{scheduler_type.__qualname__}"

    def _reference_transition_std_dev_t(self, timestep, *, noise_level: float):
        """Evaluate the standard deviation returned by TempFlow's frozen kernel."""

        import torch

        scheduler = self.pipeline.scheduler
        noise_level = self._validate_reference_noise_level(noise_level)
        requested_timestep = torch.as_tensor(timestep).reshape(-1)
        if requested_timestep.numel() != 1 or not bool(
            torch.isfinite(requested_timestep).all()
        ):
            raise ValueError(
                "TempFlow reference transition requires one finite scheduler timestep"
            )
        try:
            step_index = int(scheduler.index_for_timestep(timestep))
        except Exception as exc:
            raise ValueError(
                "TempFlow reference transition timestep is absent from the scheduler"
            ) from exc

        timesteps = torch.as_tensor(scheduler.timesteps).reshape(-1)
        sigmas = torch.as_tensor(scheduler.sigmas).reshape(-1)
        if step_index < 0 or step_index >= timesteps.numel() - 1:
            raise ValueError(
                "TempFlow reference transition timestep must identify a non-terminal "
                f"scheduler step, got index {step_index}"
            )
        if step_index + 1 >= sigmas.numel():
            raise ValueError("TempFlow reference transition is missing sigma_{t+1}")
        scheduled_timestep = torch.as_tensor(timesteps[step_index]).reshape(-1)
        if scheduled_timestep.numel() != 1 or float(requested_timestep.item()) != float(
            scheduled_timestep.item()
        ):
            raise ValueError(
                "TempFlow reference transition timestep does not match the indexed "
                "scheduler step"
            )
        if not sigmas.is_floating_point():
            sigmas = sigmas.to(torch.get_default_dtype())
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        sigma_max = sigmas[1].to(dtype=sigma.dtype, device=sigma.device)
        if not bool(torch.isfinite(torch.stack([sigma, sigma_next, sigma_max])).all()):
            raise ValueError("TempFlow reference transition sigmas must be finite")
        denominator_sigma = torch.where(sigma == 1, sigma_max, sigma)
        base_variance = sigma / (1 - denominator_sigma)
        sigma_delta = sigma - sigma_next
        if not bool((base_variance > 0).item()) or not bool((sigma_delta > 0).item()):
            raise ValueError(
                "TempFlow reference transition requires positive base variance and "
                "sigma_t - sigma_{t+1}"
            )
        base_std = torch.sqrt(base_variance) * noise_level
        return base_std * torch.sqrt(sigma_delta)

    def _validate_reference_noise_level(self, noise_level: float) -> float:
        try:
            value = float(noise_level)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "TempFlow noise_level must be a finite positive number"
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("TempFlow noise_level must be a finite positive number")
        if self.tempflow_reference_mode and value != SD3_REFERENCE_NOISE_LEVEL:
            raise ValueError(
                "TempFlow reference mode is pinned to the frozen SD3 kernel "
                f"noise_level={SD3_REFERENCE_NOISE_LEVEL}, got {value}"
            )
        return value

    def _transition_kernel_identity(self) -> str:
        kernel = self._sde_step_with_logprob
        if kernel is None:
            return "unavailable"
        module = getattr(kernel, "__module__", type(kernel).__module__)
        name = getattr(kernel, "__qualname__", type(kernel).__qualname__)
        return f"{module}.{name}"

    def _validate_recompute_contract(
        self,
        batch: RolloutBatch,
        latents,
        next_latents,
    ) -> None:
        version = batch.model_metadata.get("trajectory_contract_version")
        if version != SD3_TRANSITION_CONTRACT_VERSION:
            if self.tempflow_reference_mode:
                raise ValueError(
                    "SD3 TempFlow reference mode requires trajectory contract "
                    f"{SD3_TRANSITION_CONTRACT_VERSION!r}, got {version!r}"
                )
            return
        expected_dtype = batch.model_metadata.get("trajectory_source_dtype")
        if expected_dtype and str(latents.dtype) != expected_dtype:
            raise ValueError(
                "SD3 trajectory source dtype changed after rollout: "
                f"{latents.dtype} != {expected_dtype}"
            )
        expected_target_dtype = batch.model_metadata.get("trajectory_target_dtype")
        if expected_target_dtype and str(next_latents.dtype) != expected_target_dtype:
            raise ValueError(
                "SD3 trajectory target dtype changed after rollout: "
                f"{next_latents.dtype} != {expected_target_dtype}"
            )
        expected_kernel = batch.model_metadata.get("transition_kernel")
        current_kernel = self._transition_kernel_identity()
        if expected_kernel and expected_kernel != current_kernel:
            raise ValueError(
                "SD3 transition kernel changed after rollout: "
                f"{current_kernel} != {expected_kernel}"
            )
        expected_training = batch.model_metadata.get("transformer_training")
        recompute_training = batch.model_metadata.get(
            "recompute_transformer_training",
            expected_training,
        )
        if recompute_training is not None and bool(self.transformer.training) != bool(
            recompute_training
        ):
            raise ValueError(
                "SD3 transformer mode does not match the recorded recompute contract: "
                f"{self.transformer.training} != {recompute_training}"
            )

    def _validate_scheduler_context(self, batch: RolloutBatch) -> None:
        import torch

        saved_timesteps = batch.model_tensors.get("scheduler_timesteps")
        if saved_timesteps is None:
            return
        expected_class = batch.model_metadata.get("scheduler_class")
        if expected_class and expected_class != self._scheduler_identity():
            raise ValueError(
                "SD3 scheduler class changed after rollout: "
                f"{self._scheduler_identity()} != {expected_class}"
            )

        live_timesteps = torch.as_tensor(self.pipeline.scheduler.timesteps)
        if not torch.equal(
            live_timesteps.detach().cpu(),
            torch.as_tensor(saved_timesteps).detach().cpu(),
        ):
            raise ValueError("SD3 scheduler timesteps changed after rollout")

        saved_sigmas = batch.model_tensors.get("scheduler_sigmas")
        if saved_sigmas is None:
            return
        if not hasattr(self.pipeline.scheduler, "sigmas"):
            raise ValueError("SD3 scheduler sigmas disappeared after rollout")
        live_sigmas = torch.as_tensor(self.pipeline.scheduler.sigmas)
        if not torch.equal(
            live_sigmas.detach().cpu(),
            torch.as_tensor(saved_sigmas).detach().cpu(),
        ):
            raise ValueError("SD3 scheduler sigmas changed after rollout")

    def _predict_noise(
        self,
        *,
        latent_step,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        guidance_scale: float,
    ):
        import torch

        if guidance_scale > 1.0:
            hidden_states = torch.cat([latent_step, latent_step])
            model_timestep = torch.cat([timestep, timestep])
            embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        else:
            hidden_states = latent_step
            model_timestep = timestep
            embeds = prompt_embeds
            pooled = pooled_prompt_embeds

        transformer_kwargs = {
            "hidden_states": hidden_states,
            "timestep": model_timestep,
            "encoder_hidden_states": embeds,
            "pooled_projections": pooled,
            "return_dict": False,
        }
        joint_attention_kwargs = getattr(
            self.pipeline,
            "joint_attention_kwargs",
            None,
        )
        if joint_attention_kwargs is not None:
            transformer_kwargs["joint_attention_kwargs"] = joint_attention_kwargs
        noise_pred = self.transformer(**transformer_kwargs)[0]
        noise_pred = noise_pred.to(prompt_embeds.dtype)
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return noise_pred

    def _recompute_shared_prefix_log_probs(
        self,
        batch: RolloutBatch,
        *,
        latents,
        next_latents,
        timesteps,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        guidance_scale: float,
    ):
        import torch

        representative_rows, row_to_parent = self._shared_prefix_parent_layout(
            batch,
            tensors={
                "source latent": latents,
                "timestep": timesteps,
                "prompt embedding": prompt_embeds,
                "pooled prompt embedding": pooled_prompt_embeds,
                "negative prompt embedding": negative_prompt_embeds,
                "negative pooled prompt embedding": negative_pooled_prompt_embeds,
            },
            device=latents.device,
        )
        parent_prompt_embeds = prompt_embeds.index_select(0, representative_rows)
        parent_pooled_prompt_embeds = pooled_prompt_embeds.index_select(
            0,
            representative_rows,
        )
        parent_negative_prompt_embeds = negative_prompt_embeds.index_select(
            0,
            representative_rows,
        )
        parent_negative_pooled_prompt_embeds = (
            negative_pooled_prompt_embeds.index_select(0, representative_rows)
        )

        legacy_contract = (
            batch.model_metadata.get("trajectory_contract_version")
            != SD3_TRANSITION_CONTRACT_VERSION
        )
        transformer_dtype = self._transformer_dtype()
        log_probs = []
        for index in range(latents.shape[1]):
            parent_latent_step = latents[:, index].index_select(
                0,
                representative_rows,
            )
            if legacy_contract:
                parent_latent_step = parent_latent_step.to(dtype=transformer_dtype)
            parent_timestep = timesteps[:, index].index_select(
                0,
                representative_rows,
            )
            parent_noise_pred = self._predict_noise(
                latent_step=parent_latent_step,
                timestep=parent_timestep,
                prompt_embeds=parent_prompt_embeds,
                pooled_prompt_embeds=parent_pooled_prompt_embeds,
                negative_prompt_embeds=parent_negative_prompt_embeds,
                negative_pooled_prompt_embeds=parent_negative_pooled_prompt_embeds,
                guidance_scale=guidance_scale,
            )
            row_noise_pred = parent_noise_pred.index_select(0, row_to_parent)
            row_source = parent_latent_step.index_select(0, row_to_parent)
            _prev_sample, log_prob, _mean, _std = self._sde_step_with_logprob(
                self.pipeline.scheduler,
                row_noise_pred.float(),
                timesteps[:, index],
                row_source.float(),
                prev_sample=next_latents[:, index].float(),
            )
            log_probs.append(log_prob)
        return torch.stack(log_probs, dim=1)

    @staticmethod
    def _shared_prefix_parent_layout(
        batch: RolloutBatch,
        *,
        tensors: dict[str, Any],
        device,
    ):
        import torch

        parent_values = batch.model_tensors.get("parent_prompt_indices")
        if parent_values is None:
            parent_values = [item.get("parent_prompt_index") for item in batch.metadata]
        parent_ids = torch.as_tensor(parent_values, dtype=torch.long).reshape(-1)
        row_count = len(batch.prompts)
        if parent_ids.numel() != row_count:
            raise ValueError(
                "SD3 shared-prefix parent index count must match rollout rows"
            )

        ordered_parent_ids: list[int] = []
        rows_by_parent: dict[int, list[int]] = {}
        for row, parent_id in enumerate(parent_ids.tolist()):
            if parent_id not in rows_by_parent:
                ordered_parent_ids.append(parent_id)
                rows_by_parent[parent_id] = []
            rows_by_parent[parent_id].append(row)
        if not ordered_parent_ids:
            raise ValueError("SD3 shared-prefix rollout has no parent groups")
        group_sizes = {len(rows_by_parent[parent]) for parent in ordered_parent_ids}
        if len(group_sizes) != 1:
            raise ValueError("SD3 shared-prefix parents must have equal branch counts")
        branch_count = next(iter(group_sizes))
        expected_branch_count = batch.model_metadata.get("branch_count")
        if (
            expected_branch_count is not None
            and int(expected_branch_count) != branch_count
        ):
            raise ValueError(
                "SD3 shared-prefix branch count changed after rollout: "
                f"{branch_count} != {expected_branch_count}"
            )

        expected_parent_order = [
            parent for parent in ordered_parent_ids for _ in range(branch_count)
        ]
        if parent_ids.tolist() != expected_parent_order:
            raise ValueError("SD3 shared-prefix rows must use parent-major ordering")

        representative_rows = torch.as_tensor(
            [rows_by_parent[parent][0] for parent in ordered_parent_ids],
            dtype=torch.long,
            device=device,
        )
        row_to_parent = torch.arange(
            len(ordered_parent_ids),
            dtype=torch.long,
            device=device,
        ).repeat_interleave(branch_count)

        for name, value in tensors.items():
            tensor = torch.as_tensor(value)
            if tensor.shape[0] != row_count:
                raise ValueError(
                    f"SD3 shared-prefix {name} batch does not match rollout rows"
                )
            for parent in ordered_parent_ids:
                rows = torch.as_tensor(
                    rows_by_parent[parent],
                    dtype=torch.long,
                    device=tensor.device,
                )
                grouped = tensor.index_select(0, rows)
                reference = grouped[:1].expand_as(grouped)
                if not torch.equal(grouped, reference):
                    raise ValueError(
                        f"SD3 shared-prefix {name} differs within parent {parent}"
                    )
        return representative_rows, row_to_parent

    @contextmanager
    def _transformer_input_dtype(self):
        """Cast per-step hidden states without rounding the saved SDE target."""

        import torch

        dtype = self._transformer_dtype()

        def cast_hidden_states(_module, args, kwargs):
            hidden_states = kwargs.get("hidden_states")
            if isinstance(hidden_states, torch.Tensor):
                if hidden_states.dtype == dtype:
                    return None
                updated_kwargs = dict(kwargs)
                updated_kwargs["hidden_states"] = hidden_states.to(dtype=dtype)
                return args, updated_kwargs

            if args and isinstance(args[0], torch.Tensor):
                if args[0].dtype == dtype:
                    return None
                return (args[0].to(dtype=dtype), *args[1:]), kwargs
            return None

        handle = self.transformer.register_forward_pre_hook(
            cast_hidden_states,
            prepend=True,
            with_kwargs=True,
        )
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def _full_sde_generator(self, generator):
        """Bind the rollout generator to an upstream full SDE call that omits it."""

        function_globals = self._sde_function_globals(self._pipeline_with_logprob)
        original = (
            function_globals.get("sde_step_with_logprob")
            if isinstance(function_globals, dict)
            else None
        )
        if generator is None:
            yield
            return
        if not callable(original):
            if callable(self._sde_step_with_logprob):
                raise RuntimeError("Could not bind the SD3 full-pipeline SDE generator")
            yield
            return

        def seeded_sde_step(*args, **kwargs):
            deterministic = bool(kwargs.get("determistic", False))
            positional_prev_sample = len(args) >= 5 and args[4] is not None
            positional_generator = len(args) >= 6 and args[5] is not None
            if (
                not deterministic
                and not positional_prev_sample
                and not positional_generator
                and kwargs.get("generator") is None
                and kwargs.get("prev_sample") is None
            ):
                kwargs["generator"] = generator
            return original(*args, **kwargs)

        function_globals["sde_step_with_logprob"] = seeded_sde_step
        try:
            yield
        finally:
            if function_globals.get("sde_step_with_logprob") is seeded_sde_step:
                function_globals["sde_step_with_logprob"] = original

    @contextmanager
    def _perstep_sde_generator(self, generator, branch_count: int | None = None):
        """Bind RNG and optionally replace the fixed-six branch SDE kernel."""

        function_globals = self._sde_function_globals(
            self._pipeline_with_logprob_perstep
        )
        original = (
            function_globals.get("sde_step_with_logprob")
            if isinstance(function_globals, dict)
            else None
        )
        if not callable(original):
            if branch_count is not None and callable(self._sde_step_with_logprob):
                raise RuntimeError(
                    "Could not replace the SD3 per-step branching SDE kernel"
                )
            yield
            return

        if branch_count is not None:
            if branch_count < 1:
                raise ValueError("branch_count must be positive")
            ordinary_kernel = self._sde_step_with_logprob
            if not callable(ordinary_kernel):
                raise RuntimeError(
                    "SD3 ordinary transition kernel is required for branching"
                )

            def shared_sde_step(
                scheduler,
                model_output,
                timestep,
                sample,
                prev_sample=None,
                generator=None,
                determistic=False,
            ):
                import torch

                if determistic:
                    return ordinary_kernel(
                        scheduler,
                        model_output,
                        timestep,
                        sample,
                        prev_sample=sample,
                        generator=None,
                        determistic=True,
                    )
                if prev_sample is not None:
                    return ordinary_kernel(
                        scheduler,
                        model_output,
                        timestep,
                        sample,
                        prev_sample=prev_sample,
                        generator=None,
                        determistic=False,
                    )

                parent_count = int(sample.shape[0])
                expanded_sample = sample.repeat_interleave(branch_count, dim=0)
                expanded_output = model_output.repeat_interleave(
                    branch_count,
                    dim=0,
                )
                timestep_values = torch.as_tensor(
                    timestep,
                    device=sample.device,
                ).reshape(-1)
                if timestep_values.numel() == 1:
                    expanded_timestep = timestep_values.expand(
                        parent_count * branch_count
                    )
                elif timestep_values.numel() == parent_count:
                    expanded_timestep = timestep_values.repeat_interleave(branch_count)
                else:
                    raise ValueError(
                        "SD3 branch timestep must be scalar or parent-batched"
                    )
                active_generator = (
                    generator if generator is not None else generator_bound
                )
                return ordinary_kernel(
                    scheduler,
                    expanded_output,
                    expanded_timestep,
                    expanded_sample,
                    generator=active_generator,
                    determistic=False,
                )

            generator_bound = generator
            replacement = shared_sde_step
        else:
            if generator is None:
                yield
                return

            def seeded_sde_step(*args, **kwargs):
                deterministic = bool(kwargs.get("determistic", False))
                positional_prev_sample = len(args) >= 5 and args[4] is not None
                positional_generator = len(args) >= 6 and args[5] is not None
                if (
                    not deterministic
                    and not positional_prev_sample
                    and not positional_generator
                    and kwargs.get("generator") is None
                ):
                    kwargs["generator"] = generator
                return original(*args, **kwargs)

            replacement = seeded_sde_step

        function_globals["sde_step_with_logprob"] = replacement
        try:
            yield
        finally:
            if function_globals.get("sde_step_with_logprob") is replacement:
                function_globals["sde_step_with_logprob"] = original

    @staticmethod
    def _sde_function_globals(function):
        """Find the wrapped function globals that own the upstream SDE symbol."""

        seen: set[int] = set()
        current = function
        while callable(current) and id(current) not in seen:
            seen.add(id(current))
            function_globals = getattr(current, "__globals__", None)
            if isinstance(function_globals, dict) and callable(
                function_globals.get("sde_step_with_logprob")
            ):
                return function_globals
            current = getattr(current, "__wrapped__", None)
        return None

    @staticmethod
    def _branch_gather_indices(
        expanded_size: int,
        parent_count: int,
        branch_count: int,
    ) -> tuple[int, list[int]]:
        if parent_count < 1 or expanded_size % parent_count:
            raise ValueError(
                "Per-step branching output is not grouped by parent prompt"
            )
        exploration_k = expanded_size // parent_count
        if branch_count > exploration_k:
            raise ValueError(
                f"Requested {branch_count} branches but reference pipeline returned {exploration_k}"
            )
        indices = [
            parent * exploration_k + branch
            for parent in range(parent_count)
            for branch in range(branch_count)
        ]
        return exploration_k, indices

    @staticmethod
    def _select_branch_kl(
        values,
        gather_indices: list[int],
        parent_count: int,
        branch_count: int,
    ):
        import torch

        values = values.flatten()
        if values.numel() == parent_count:
            return values.repeat_interleave(branch_count)
        gather = torch.as_tensor(gather_indices, device=values.device, dtype=torch.long)
        return values.index_select(0, gather)

    @staticmethod
    def _select_branch_media(media, gather_indices: list[int]):
        import torch

        if isinstance(media, torch.Tensor):
            gather = torch.as_tensor(
                gather_indices,
                device=media.device,
                dtype=torch.long,
            )
            return media.index_select(0, gather).detach()
        return [media[index] for index in gather_indices]

    def _encode_text(self, prompts: list[str]):
        embeds, pooled = self._encode_prompt(
            [
                self.pipeline.text_encoder,
                self.pipeline.text_encoder_2,
                self.pipeline.text_encoder_3,
            ],
            [
                self.pipeline.tokenizer,
                self.pipeline.tokenizer_2,
                self.pipeline.tokenizer_3,
            ],
            prompts,
            self.max_sequence_length,
        )
        return embeds.to(self.device), pooled.to(self.device)

    def _transformer_dtype(self):
        import torch

        return next(
            self.transformer.parameters(), torch.empty((), dtype=torch.float32)
        ).dtype

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
