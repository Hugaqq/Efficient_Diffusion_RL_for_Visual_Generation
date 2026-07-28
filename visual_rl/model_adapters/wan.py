"""Wan/World-R1 adapter with lazy legacy imports."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import inspect
from pathlib import Path
import re
from typing import Any

from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.types import RolloutBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.model_adapters.diffusers_common import (
    AdapterNotLoadedError,
    GradientCheckpointingState,
    configure_gradient_checkpointing,
    gradient_checkpointing_metadata,
    gradient_checkpointing_request,
    make_generator,
    require_model_path,
    validate_gradient_checkpointing_checkpoint_metadata,
    verify_gradient_checkpointing,
)
from visual_rl.model_adapters.diffusers_common import resolve_torch_dtype
from visual_rl.third_party.legacy import legacy_repo_path, resolve_legacy_repo

# Wan2.1 Diffusers attention exposes these four projections. Architectures with
# additional cross-attention projections may opt into them explicitly through
# ``lora_target_modules``; defaults must remain valid for the pinned base model.
DEFAULT_WAN_LORA_TARGETS = [
    "to_k",
    "to_out.0",
    "to_q",
    "to_v",
]
WAN_LORA_CHECKPOINT_SUBDIR = "wan_lora_adapter"
WAN_CHECKPOINT_METADATA_NAME = "adapter_metadata.json"
WAN_LORA_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
WAN_CHECKPOINT_FORMAT_VERSION = 1
WAN_ADAPTER_NAME_MAX_LENGTH = 64
_WAN_ADAPTER_NAME_PATTERN = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9_.-]{{0,{WAN_ADAPTER_NAME_MAX_LENGTH - 1}}}$",
    re.ASCII,
)
_URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
WAN_BACKENDS = frozenset({"world_r1", "flash"})
_WORLD_PIPELINE_KWARGS = frozenset(
    {
        "prompt",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "num_videos_per_prompt",
        "generator",
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "output_type",
        "return_dict",
        "attention_kwargs",
        "max_sequence_length",
        "kl_reward",
        "use_camera_trajectory",
        "save_latents_vis",
    }
)
_FLASH_PIPELINE_KWARGS = frozenset(
    {
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "num_videos_per_prompt",
        "generator",
        "prompt_embeds",
        "negative_prompt_embeds",
        "output_type",
        "return_dict",
        "attention_kwargs",
        "max_sequence_length",
        "kl_reward",
        "index",
    }
)


class WorldR1WanLegacyAdapter(ModelAdapter):
    """Bounded bridge for Wan rollout/logprob behavior from the reference repos.

    The real heavy training path is intentionally lazy so `import visual_rl` and
    mock smoke tests do not require CUDA, diffusers, or Wan checkpoints.
    """

    name = "world_r1_wan_legacy"
    media_type = "video"

    def __init__(self, config: dict[str, Any]):
        gradient_checkpointing_requested = gradient_checkpointing_request(config)
        extra = dict(config.get("extra") or {})
        self.config = {
            **extra,
            **{key: value for key, value in config.items() if key != "extra"},
        }
        self.wan_backend = self._wan_backend()
        self.gradient_checkpointing_requested = gradient_checkpointing_requested
        self._gradient_checkpointing_state = GradientCheckpointingState(
            requested=self.gradient_checkpointing_requested,
            effective=None,
        )
        self.use_lora = self._required_bool("use_lora", True)
        self.lora_path = self._optional_local_absolute_path("lora_path")
        self.lora_rank = self._positive_int("lora_rank", 32)
        self.lora_alpha = self._positive_int("lora_alpha", 64)
        self.lora_targets = self._lora_targets()
        self._loaded_lora_targets: list[str] | None = None
        self.adapter_name = self._adapter_name()
        root_key = (
            "world_r1_root" if self.wan_backend == "world_r1" else "flash_grpo_root"
        )
        default_root = (
            "World-R1-main" if self.wan_backend == "world_r1" else "Flash-GRPO-main"
        )
        self.repo_root = resolve_legacy_repo(
            self.config.get(root_key) or self.config.get("repo_root") or default_root
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

    @property
    def flash_repo_root(self) -> Path:
        """Read-only compatibility view of the Flash reference root.

        The active backend still owns the single authoritative ``repo_root``.
        World-R1 adapters expose this lazy alias only for older inspection and
        staging code; it does not enable Flash sampling on the wrong backend.
        """

        if self.wan_backend == "flash":
            return self.repo_root
        return resolve_legacy_repo(
            self.config.get("flash_grpo_root") or "Flash-GRPO-main"
        )

    @property
    def train_module(self):
        if self.transformer is None:
            raise AdapterNotLoadedError(
                "WorldR1WanLegacyAdapter has no train module before load()."
            )
        return self.transformer

    def parameters(self):
        parameters = [
            parameter
            for parameter in self.train_module.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise AdapterNotLoadedError(
                "WorldR1WanLegacyAdapter has no trainable parameters."
            )
        return parameters

    def named_parameters(self):
        return [
            (f"transformer.{name}", parameter)
            for name, parameter in self.train_module.named_parameters()
            if parameter.requires_grad
        ]

    def load(self):
        model_path = require_model_path(self.config, self.name)
        from_pretrained_kwargs: dict[str, Any] = {
            "local_files_only": bool(self.config.get("local_files_only", True)),
        }
        dtype = resolve_torch_dtype(
            self.config.get("torch_dtype") or self.config.get("dtype")
        )
        if dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = dtype
        if "low_cpu_mem_usage" in self.config:
            from_pretrained_kwargs["low_cpu_mem_usage"] = bool(
                self.config["low_cpu_mem_usage"]
            )

        with legacy_repo_path(self.repo_root):
            from diffusers import WanPipeline

            self.pipeline = WanPipeline.from_pretrained(
                model_path, **from_pretrained_kwargs
            )
            device = str(self.config.get("device", "")).strip()
            if device:
                self.pipeline = self.pipeline.to(device)
            self._configure_inference_components(device=device, dtype=dtype)
            base_transformer = self.pipeline.transformer
            checkpointing_state = configure_gradient_checkpointing(
                base_transformer,
                self.gradient_checkpointing_requested,
                context="Wan base transformer",
            )
            if self.use_lora:
                self._validate_lora_targets(base_transformer)
                self.transformer = self._attach_lora(base_transformer)
                self.pipeline.transformer = self.transformer
                self._assert_lora_trainable()
            else:
                base_transformer.requires_grad_(True)
                self.transformer = base_transformer
            self.scheduler = getattr(self.pipeline, "scheduler", None)
            self.tokenizer = getattr(self.pipeline, "tokenizer", None)
            self.tokenizer_2 = getattr(self.pipeline, "tokenizer_2", None)
            self.text_encoder = getattr(self.pipeline, "text_encoder", None)
            self.text_encoder_2 = getattr(self.pipeline, "text_encoder_2", None)
            self.device = self._infer_device(device)
            self.dtype = dtype or getattr(self.transformer, "dtype", None)
            self.model_path = str(model_path)
            self._gradient_checkpointing_state = verify_gradient_checkpointing(
                self.transformer,
                checkpointing_state,
                context="Wan active transformer after PEFT attach",
            )
        return self

    def sample(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> RolloutBatch:
        if self.wan_backend != "world_r1":
            raise RuntimeError(
                "Wan backend 'flash' requires sample_single_step() with an aligned "
                "selected_timestep_indices contract."
            )
        self._ensure_loaded()
        self.prepare_for_sampling()
        self._assert_transformer_consistency()
        import torch

        batch_size = len(prompts)
        runtime = self._runtime_options(rollout_config)
        if runtime["use_camera_trajectory"] and runtime["num_videos_per_prompt"] != 1:
            raise ValueError(
                "Wan camera trajectory rollout currently requires num_videos_per_prompt=1"
            )
        generator = make_generator(runtime["device"], runtime["seed"])
        prompt_embeds, negative_prompt_embeds = self._encode_prompt_embeds(
            prompts,
            negative_prompts=[""] * batch_size,
            num_videos_per_prompt=runtime["num_videos_per_prompt"],
            max_sequence_length=runtime["max_sequence_length"],
            train_cfg=runtime["train_cfg"],
        )
        pipeline_with_logprob, reference_path, injected = (
            self._load_wan_pipeline_with_logprob()
        )
        batch_metadata = [dict(item) for item in metadata]
        camera_latents = None
        if runtime["use_camera_trajectory"]:
            camera_latents, batch_metadata = self._prepare_world_camera_latents(
                prompts,
                batch_metadata,
                runtime,
                generator,
            )

        with torch.no_grad():
            result = self._call_pipeline_with_logprob(
                pipeline_with_logprob,
                injected=injected,
                train_cfg=runtime["train_cfg"],
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
                latents=camera_latents,
                num_videos_per_prompt=runtime["num_videos_per_prompt"],
                max_sequence_length=runtime["max_sequence_length"],
                attention_kwargs=self.config.get("attention_kwargs"),
                use_camera_trajectory=False,
                save_latents_vis=False,
            )

        videos, all_latents, all_log_probs, all_kl, all_timesteps = (
            self._normalize_pipeline_result(result)
        )
        latents = torch.stack(list(all_latents), dim=1)
        old_log_probs = torch.stack(list(all_log_probs), dim=1)
        if all_kl:
            kl = torch.stack(list(all_kl), dim=1)
        else:
            kl = torch.zeros_like(old_log_probs)
        timesteps = (
            torch.stack(list(all_timesteps))
            .to(old_log_probs.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        batch = RolloutBatch(
            prompts=list(prompts),
            metadata=batch_metadata,
            media=videos,
            latents=latents[:, :-1],
            next_latents=latents[:, 1:],
            timesteps=timesteps,
            old_log_probs=old_log_probs,
            kl=kl,
            model_metadata={
                **self.runtime_metadata(),
                "adapter": self.name,
                "reference_path": str(reference_path),
                "wan_backend": self.wan_backend,
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

    def sample_single_step(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> RolloutBatch:
        if self.wan_backend != "flash":
            raise RuntimeError(
                "sample_single_step() is only available for wan_backend='flash'"
            )
        self._ensure_loaded()
        self.prepare_for_sampling()
        self._assert_transformer_consistency()
        import torch

        selected = rollout_config.get("selected_timestep_indices")
        if not isinstance(selected, (list, tuple)) or len(selected) != len(prompts):
            raise ValueError(
                "Flash sample_single_step requires one selected_timestep_indices value per batch item"
            )
        if any(
            isinstance(item, bool) or not isinstance(item, int) for item in selected
        ):
            raise ValueError("Flash selected_timestep_indices must contain integers")
        selected = [int(item) for item in selected]
        runtime = self._runtime_options(rollout_config)
        invalid_indices = sorted(
            {index for index in selected if index < 0 or index >= runtime["num_steps"]}
        )
        if invalid_indices:
            raise ValueError(
                "Flash selected timestep indices "
                f"{invalid_indices} are outside num_steps={runtime['num_steps']}"
            )

        prompt_embeds, negative_prompt_embeds = self._encode_prompt_embeds(
            prompts,
            negative_prompts=[""] * len(prompts),
            num_videos_per_prompt=1,
            max_sequence_length=runtime["max_sequence_length"],
            train_cfg=runtime["train_cfg"],
        )
        pipeline_with_logprob, reference_path, injected = (
            self._load_wan_pipeline_with_logprob()
        )
        grouped_positions: list[int] = []
        grouped_metadata: list[dict[str, Any]] = []
        chunks: dict[str, list[Any]] = {
            "media": [],
            "latents": [],
            "next_latents": [],
            "timesteps": [],
            "old_log_probs": [],
            "kl": [],
            "coefficient": [],
            "prompt_embeds": [],
            "negative_prompt_embeds": [],
        }
        actual_timestep_by_position = [0] * len(prompts)
        for reference_index in sorted(set(selected)):
            positions = [
                position
                for position, index in enumerate(selected)
                if index == reference_index
            ]
            grouped_positions.extend(positions)
            grouped_metadata.extend(dict(metadata[position]) for position in positions)
            prompt_group = self._select_flash_rows(prompt_embeds, positions)
            negative_group = (
                None
                if negative_prompt_embeds is None
                else self._select_flash_rows(negative_prompt_embeds, positions)
            )
            group_seed = self._flash_group_seed(runtime["seed"], reference_index)
            generator = make_generator(runtime["device"], group_seed)
            with self._fork_flash_group_rng(runtime["device"], group_seed):
                with torch.no_grad():
                    result = self._call_pipeline_with_logprob(
                        pipeline_with_logprob,
                        injected=injected,
                        train_cfg=runtime["train_cfg"],
                        prompt_embeds=prompt_group,
                        negative_prompt_embeds=negative_group,
                        num_inference_steps=runtime["num_steps"],
                        guidance_scale=runtime["guidance_scale"],
                        output_type="pt",
                        return_dict=False,
                        num_frames=runtime["frames"],
                        height=runtime["height"],
                        width=runtime["width"],
                        kl_reward=runtime["kl_reward"],
                        generator=generator,
                        num_videos_per_prompt=1,
                        max_sequence_length=runtime["max_sequence_length"],
                        attention_kwargs=self.config.get("attention_kwargs"),
                        index=reference_index,
                    )

            if not isinstance(result, (tuple, list)) or len(result) != 5:
                raise ValueError("Flash Wan pipeline must return exactly five items")
            videos, all_latents, all_log_probs, all_kl, returned_index = result
            if int(returned_index) != reference_index:
                raise ValueError(
                    "Flash Wan pipeline returned a selected index that differs "
                    "from the requested reference index"
                )
            if len(all_latents) != 2 or len(all_log_probs) != 1:
                raise ValueError(
                    "Flash Wan pipeline must return exactly one [latent, "
                    "next_latent] transition and one logprob"
                )
            group_size = len(positions)
            videos = self._as_flash_batch_tensor(videos, group_size, "videos")
            latents = self._as_flash_transition_tensor(
                all_latents[0], group_size, "latents"
            )
            next_latents = self._as_flash_transition_tensor(
                all_latents[1], group_size, "next_latents"
            )
            old_log_probs = self._as_flash_scalar_transition(
                all_log_probs[0], group_size, "old_log_probs"
            )
            scheduler_timesteps = getattr(self.scheduler, "timesteps", None)
            if (
                scheduler_timesteps is None
                or len(scheduler_timesteps) <= reference_index
            ):
                raise ValueError(
                    "Flash scheduler.timesteps does not contain the requested "
                    "selected timestep"
                )
            actual_timestep = torch.as_tensor(
                scheduler_timesteps[reference_index],
                device=old_log_probs.device,
            )
            timesteps = actual_timestep.reshape(1, 1).expand(group_size, 1).clone()
            kl = self._flash_selected_kl(
                all_kl,
                reference_index=reference_index,
                batch_size=group_size,
                like=old_log_probs,
            )
            coefficient = self._flash_reference_coefficient(
                latents=latents[:, 0],
                next_latents=next_latents[:, 0],
                timesteps=timesteps[:, 0],
            )
            actual_value = int(actual_timestep.item())
            for position in positions:
                actual_timestep_by_position[position] = actual_value
            for name, value in (
                ("media", videos),
                ("latents", latents),
                ("next_latents", next_latents),
                ("timesteps", timesteps),
                ("old_log_probs", old_log_probs),
                ("kl", kl),
                ("coefficient", coefficient),
                ("prompt_embeds", prompt_group),
            ):
                chunks[name].append(value)
            if negative_group is not None:
                chunks["negative_prompt_embeds"].append(negative_group)

        media = self._restore_flash_tensor_order(
            torch.cat(chunks["media"], dim=0), grouped_positions
        )
        latents = self._restore_flash_tensor_order(
            torch.cat(chunks["latents"], dim=0), grouped_positions
        )
        next_latents = self._restore_flash_tensor_order(
            torch.cat(chunks["next_latents"], dim=0), grouped_positions
        )
        timesteps = self._restore_flash_tensor_order(
            torch.cat(chunks["timesteps"], dim=0), grouped_positions
        )
        old_log_probs = self._restore_flash_tensor_order(
            torch.cat(chunks["old_log_probs"], dim=0), grouped_positions
        )
        kl = self._restore_flash_tensor_order(
            torch.cat(chunks["kl"], dim=0), grouped_positions
        )
        coefficient = self._restore_flash_tensor_order(
            torch.cat(chunks["coefficient"], dim=0), grouped_positions
        )
        prompt_embeds = self._restore_flash_tensor_order(
            torch.cat(chunks["prompt_embeds"], dim=0), grouped_positions
        )
        negative_prompt_embeds = (
            None
            if not chunks["negative_prompt_embeds"]
            else self._restore_flash_tensor_order(
                torch.cat(chunks["negative_prompt_embeds"], dim=0),
                grouped_positions,
            )
        )
        restored_metadata = self._restore_flash_sequence_order(
            grouped_metadata, grouped_positions
        )

        batch = RolloutBatch(
            prompts=list(prompts),
            metadata=restored_metadata,
            media=media,
            media_layout="BFCHW",
            latents=latents,
            next_latents=next_latents,
            timesteps=timesteps,
            old_log_probs=old_log_probs,
            kl=kl,
            model_metadata={
                **self.runtime_metadata(),
                "adapter": self.name,
                "wan_backend": self.wan_backend,
                "reference_path": str(reference_path),
                "selected_timestep_indices": list(selected),
                "actual_scheduler_timestep": (
                    actual_timestep_by_position[0]
                    if len(set(actual_timestep_by_position)) == 1
                    else list(actual_timestep_by_position)
                ),
                "actual_scheduler_timesteps": list(actual_timestep_by_position),
                "num_steps": runtime["num_steps"],
                "frame_shape": list(getattr(media, "shape", [])),
                "scheduler": self._scheduler_metadata(),
                "sample_config": self._serializable_runtime_options(runtime),
            },
            model_tensors={
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "coefficient": coefficient,
            },
        )
        batch.validate_lightweight(strict=True)
        return batch

    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        self._ensure_loaded()
        self.prepare_for_training()
        self._assert_transformer_consistency()
        import torch

        sde_step_with_logprob = self._load_sde_step_with_logprob()

        prompt_embeds = batch.model_tensors.get("prompt_embeds")
        negative_prompt_embeds = batch.model_tensors.get("negative_prompt_embeds")
        if prompt_embeds is None:
            raise ValueError(
                "RolloutBatch.model_tensors must contain prompt_embeds for Wan logprob recomputation."
            )

        runtime = dict(batch.model_metadata.get("sample_config") or {})
        guidance_scale = float(
            runtime.get("guidance_scale", self.config.get("guidance_scale", 5.0))
        )
        train_cfg = self._resolve_train_cfg(
            runtime,
            guidance_scale=guidance_scale,
        )
        attention_kwargs = self.config.get("attention_kwargs")
        transformer_device, transformer_dtype = self._transformer_device_dtype()
        prompt_embeds = prompt_embeds.to(
            device=transformer_device,
            dtype=transformer_dtype,
        )
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=transformer_device,
                dtype=transformer_dtype,
            )
        steps = int(batch.old_log_probs.shape[1])
        new_log_probs = []
        for j in range(steps):
            sample_j = batch.latents[:, j].to(device=transformer_device)
            model_input_j = sample_j.to(dtype=transformer_dtype)
            timesteps_j = batch.timesteps[:, j].to(device=transformer_device)
            next_sample_j = batch.next_latents[:, j].to(device=transformer_device)
            noise_pred_text = self.transformer(
                hidden_states=model_input_j,
                timestep=timesteps_j,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            if train_cfg:
                if negative_prompt_embeds is None:
                    raise ValueError(
                        "negative_prompt_embeds are required when train_cfg=True."
                    )
                noise_pred_uncond = self.transformer(
                    hidden_states=model_input_j,
                    timestep=timesteps_j,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_text

            sde_kwargs = {"prev_sample": next_sample_j.float()}
            if self.wan_backend == "flash":
                sde_kwargs["return_dt_and_std_dev_t"] = True
            step_result = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                timesteps_j,
                sample_j.float(),
                **sde_kwargs,
            )
            if not isinstance(step_result, (tuple, list)) or len(step_result) < 2:
                raise ValueError("Wan sde_step_with_logprob returned an invalid result")
            log_prob = step_result[1]
            if self.wan_backend == "flash":
                if len(step_result) != 6:
                    raise ValueError(
                        "Flash sde_step_with_logprob must return six items when "
                        "return_dt_and_std_dev_t=True"
                    )
                recomputed = self._normalize_flash_coefficient(
                    step_result[5],
                    batch_size=batch.batch_size,
                )
                stored = batch.model_tensors.get("coefficient")
                if stored is None:
                    raise ValueError(
                        "Flash reference_v1 rollout is missing coefficient"
                    )
                stored = self._normalize_flash_coefficient(
                    stored,
                    batch_size=batch.batch_size,
                ).to(recomputed.device, dtype=recomputed.dtype)
                if not torch.allclose(stored, recomputed, rtol=1e-5, atol=1e-7):
                    raise ValueError(
                        "Flash reference coefficient mismatch between rollout and recomputation"
                    )
            new_log_probs.append(log_prob)
        return torch.stack(new_log_probs, dim=1).to(device=batch.old_log_probs.device)

    def save_pretrained(self, output_dir: str) -> None:
        transformer = self.train_module
        runtime_metadata = self.runtime_metadata()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        from visual_rl.artifacts.checkpoint import (
            save_json,
            validated_checkpoint_directory,
        )

        path = validated_checkpoint_directory(path, path, label="Wan checkpoint root")

        if self.use_lora:
            peft_config = getattr(transformer, "peft_config", None)
            save_adapter = getattr(transformer, "save_pretrained", None)
            if not peft_config or not callable(save_adapter):
                raise RuntimeError(
                    "WorldR1 Wan use_lora=True requires a loaded PEFT transformer "
                    "with save_pretrained support"
                )
            adapter_path = path / WAN_LORA_CHECKPOINT_SUBDIR
            if adapter_path.is_symlink():
                raise RuntimeError(
                    f"Wan adapter output directory must not be a symlink: {adapter_path}"
                )
            save_adapter(
                adapter_path,
                safe_serialization=True,
                selected_adapters=[self.adapter_name],
            )
            selected_path = self._find_peft_adapter_path(
                adapter_path,
                include_stable_subdir=False,
            )
            if selected_path is None:
                raise RuntimeError(
                    "Wan PEFT save_pretrained did not write a complete adapter "
                    f"checkpoint under {adapter_path}"
                )
            weight_path = self._peft_adapter_weight_path(path, selected_path)
            config_path = self._validated_checkpoint_file(
                path,
                selected_path / "adapter_config.json",
                label="Wan adapter_config.json",
            )
            save_json(
                path / WAN_CHECKPOINT_METADATA_NAME,
                {
                    **runtime_metadata,
                    "adapter": self.name,
                    "adapter_config": str(config_path.relative_to(path)),
                    "format_version": WAN_CHECKPOINT_FORMAT_VERSION,
                    "save_kind": "peft_adapter",
                    "use_lora": True,
                    "weights": str(weight_path.relative_to(path)),
                },
            )
            return

        import torch

        if hasattr(transformer, "save_pretrained"):
            transformer.save_pretrained(path / "transformer")
            save_kind = "transformer_save_pretrained"
        else:
            save_kind = "transformer_state_dict"
        torch.save(transformer.state_dict(), path / "transformer_state.pt")

        save_json(
            path / WAN_CHECKPOINT_METADATA_NAME,
            {
                **runtime_metadata,
                "adapter": self.name,
                "format_version": WAN_CHECKPOINT_FORMAT_VERSION,
                "save_kind": save_kind,
                "resume_support": "full transformer state",
                "state": "transformer_state.pt",
                "use_lora": False,
            },
        )

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        transformer = self.train_module
        path = self._validated_checkpoint_directory(
            Path(checkpoint_dir),
            Path(checkpoint_dir),
            label="Wan checkpoint root",
        )
        metadata = self._checkpoint_metadata(path)
        self._refresh_gradient_checkpointing_state()
        validate_gradient_checkpointing_checkpoint_metadata(
            metadata,
            self._gradient_checkpointing_state,
            context="Wan checkpoint",
        )
        state_path = path / "transformer_state.pt"
        validated_state = self._optional_checkpoint_file(
            path,
            state_path,
            label="Wan transformer_state.pt",
        )
        if self.use_lora:
            if validated_state is not None or metadata.get("use_lora") is False:
                raise RuntimeError(
                    "Cannot load a full-transformer Wan checkpoint while use_lora=True"
                )
            adapter_path = self._find_peft_adapter_path(path)
            if adapter_path is None:
                raise RuntimeError(
                    "Missing Wan LoRA checkpoint weights: expected a complete adapter "
                    f"under {path / WAN_LORA_CHECKPOINT_SUBDIR} or {path}"
                )
            if metadata and metadata.get("use_lora") is not True:
                raise RuntimeError(
                    "Wan checkpoint metadata mode does not match use_lora=True"
                )
            self._load_peft_checkpoint_in_place(adapter_path)
            return

        if (
            metadata.get("use_lora") is True
            or self._find_peft_adapter_path(path) is not None
        ):
            raise RuntimeError(
                "Cannot load a LoRA-only Wan checkpoint while use_lora=False"
            )
        if validated_state is None:
            raise RuntimeError(
                "Missing Wan full transformer checkpoint for use_lora=False: "
                f"expected {state_path}"
            )
        import torch

        state = torch.load(
            validated_state,
            map_location=self.device or "cpu",
            weights_only=True,
        )
        transformer.load_state_dict(state)

    def runtime_metadata(self) -> dict[str, Any]:
        if self.transformer is None:
            checkpointing_metadata = {
                "gradient_checkpointing_requested": (
                    self._gradient_checkpointing_state.requested
                ),
                "gradient_checkpointing_effective": None,
            }
        else:
            checkpointing_metadata = self._refresh_gradient_checkpointing_state()
        return {
            **checkpointing_metadata,
            "model_path": self.model_path
            or self.config.get("model_path")
            or self.config.get("pretrained_model"),
            "repo_root": str(self.repo_root),
            "wan_backend": self.wan_backend,
            "reference_root": str(self.repo_root),
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
            "pipeline_class": type(self.pipeline).__name__
            if self.pipeline is not None
            else None,
            "transformer_class": type(self.transformer).__name__
            if self.transformer is not None
            else None,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_targets": list(self._loaded_lora_targets or self.lora_targets),
            "adapter_name": self.adapter_name,
            "checkpoint_format": (
                "wan_peft_adapter_v1" if self.use_lora else "wan_full_transformer_v1"
            ),
        }

    def _refresh_gradient_checkpointing_state(self) -> dict[str, bool | None]:
        (
            self._gradient_checkpointing_state,
            metadata,
        ) = gradient_checkpointing_metadata(
            self.train_module,
            self._gradient_checkpointing_state,
            context="Wan active transformer",
        )
        return metadata

    def _ensure_loaded(self) -> None:
        if self.pipeline is None or self.transformer is None or self.scheduler is None:
            raise AdapterNotLoadedError(
                "WorldR1WanLegacyAdapter.load() must run before Wan sampling/logprob."
            )

    def _required_bool(self, name: str, default: bool) -> bool:
        value = self.config.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"Wan {name} must be a bool")
        return value

    def _wan_backend(self) -> str:
        value = self.config.get("wan_backend", "world_r1")
        if not isinstance(value, str) or value not in WAN_BACKENDS:
            raise ValueError("Wan wan_backend must be one of: flash, world_r1")
        return value

    def _positive_int(self, name: str, default: int) -> int:
        value = self.config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Wan {name} must be a positive integer")
        return int(value)

    def _optional_path(self, name: str) -> str | None:
        value = self.config.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Wan {name} must be a non-empty string or None")
        return value.strip()

    def _optional_local_absolute_path(self, name: str) -> str | None:
        value = self._optional_path(name)
        if value is None:
            return None
        if _URL_PATTERN.match(value):
            raise ValueError(f"Wan {name} must be a local absolute path, not a URL")
        if not Path(value).is_absolute():
            raise ValueError(f"Wan {name} must be a local absolute path")
        return value

    def _lora_targets(self) -> list[str]:
        value = self.config.get("lora_target_modules", DEFAULT_WAN_LORA_TARGETS)
        if not isinstance(value, (list, tuple)):
            raise ValueError("Wan lora_target_modules must be a list of target strings")
        targets = list(value)
        if not targets or any(
            not isinstance(item, str) or not item.strip() for item in targets
        ):
            raise ValueError(
                "Wan lora_target_modules must contain non-empty target strings"
            )
        normalized = [item.strip() for item in targets]
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "Wan lora_target_modules must contain unique target strings"
            )
        return normalized

    def _adapter_name(self) -> str:
        value = self.config.get("adapter_name", "default")
        if (
            not isinstance(value, str)
            or value in {".", ".."}
            or not _WAN_ADAPTER_NAME_PATTERN.fullmatch(value)
        ):
            raise ValueError(
                "Wan adapter_name must be a safe ASCII identifier of 1-64 "
                "characters: alphanumeric first, then alphanumeric, '_', '.', or '-'"
            )
        return value

    def _validate_lora_targets(self, transformer) -> None:
        module_names = [name for name, _module in transformer.named_modules()]
        missing = [
            target
            for target in self.lora_targets
            if not any(
                name == target or name.endswith(f".{target}") for name in module_names
            )
        ]
        if missing:
            raise RuntimeError(
                "Wan LoRA target validation found missing targets: "
                + ", ".join(missing)
            )

    @staticmethod
    def _normalize_peft_target_modules(value: Any) -> frozenset[str]:
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (set, frozenset, list, tuple)):
            values = list(value)
        else:
            raise RuntimeError(
                "Wan PEFT adapter config target_modules must be a string, set, or list"
            )
        if not values or any(
            not isinstance(item, str) or not item.strip() for item in values
        ):
            raise RuntimeError(
                "Wan PEFT adapter config target_modules must contain non-empty strings"
            )
        return frozenset(item.strip() for item in values)

    def _verify_peft_target_modules(self, value: Any) -> None:
        actual = self._normalize_peft_target_modules(value)
        expected = frozenset(self.lora_targets)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing targets: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected targets: " + ", ".join(unexpected))
            raise RuntimeError(
                "Wan PEFT adapter target_modules do not exactly match configured "
                "lora_targets (" + "; ".join(details) + ")"
            )
        self._loaded_lora_targets = sorted(actual)

    def _verify_loaded_peft_config(self, transformer) -> None:
        peft_config = getattr(transformer, "peft_config", None)
        if not isinstance(peft_config, Mapping) or self.adapter_name not in peft_config:
            raise RuntimeError(
                f"Wan PEFT transformer has no config for adapter {self.adapter_name!r}"
            )
        adapter_config = peft_config[self.adapter_name]
        if not hasattr(adapter_config, "target_modules"):
            raise RuntimeError("Wan PEFT adapter config does not expose target_modules")
        self._verify_peft_target_modules(adapter_config.target_modules)

    def _verify_peft_checkpoint_config(self, adapter_path: Path) -> None:
        from visual_rl.artifacts.checkpoint import load_json

        config_path = self._validated_checkpoint_file(
            adapter_path,
            adapter_path / "adapter_config.json",
            label="Wan adapter_config.json",
        )
        adapter_config = load_json(config_path)
        if (
            not isinstance(adapter_config, dict)
            or "target_modules" not in adapter_config
        ):
            raise RuntimeError(
                "Wan PEFT adapter_config.json must contain target_modules"
            )
        self._verify_peft_target_modules(adapter_config["target_modules"])

    def _configure_inference_components(self, *, device: str, dtype: Any) -> None:
        import torch

        vae = getattr(self.pipeline, "vae", None)
        if vae is not None:
            vae.requires_grad_(False)
            vae_kwargs: dict[str, Any] = {"dtype": torch.float32}
            if device:
                vae_kwargs["device"] = device
            vae.to(**vae_kwargs)

        inference_kwargs: dict[str, Any] = {}
        if device:
            inference_kwargs["device"] = device
        if dtype is not None:
            inference_kwargs["dtype"] = dtype
        for name in ("text_encoder", "text_encoder_2"):
            text_encoder = getattr(self.pipeline, name, None)
            if text_encoder is None:
                continue
            text_encoder.requires_grad_(False)
            if inference_kwargs:
                text_encoder.to(**inference_kwargs)

        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is None:
            raise RuntimeError("Wan pipeline has no transformer")
        if inference_kwargs:
            transformer.to(**inference_kwargs)

    def _attach_lora(self, transformer):
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional train dependency
            raise ImportError(
                "Install visual-rl[train] to use Wan LoRA adapters."
            ) from exc

        transformer.requires_grad_(False)
        if self.lora_path:
            adapter_path = self._find_peft_adapter_path(Path(self.lora_path))
            if adapter_path is None:
                raise RuntimeError(
                    "Wan lora_path must contain adapter_config.json and "
                    "adapter_model.safetensors or adapter_model.bin"
                )
            wrapped = PeftModel.from_pretrained(
                transformer,
                str(adapter_path),
                adapter_name=self.adapter_name,
                is_trainable=True,
            )
        else:
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=self.lora_targets,
            )
            wrapped = get_peft_model(
                transformer,
                lora_config,
                adapter_name=self.adapter_name,
            )
        self._verify_loaded_peft_config(wrapped)
        set_adapter = getattr(wrapped, "set_adapter", None)
        if not callable(set_adapter):
            raise RuntimeError("Wan PEFT transformer does not support set_adapter")
        set_adapter(self.adapter_name)
        for name, parameter in wrapped.named_parameters():
            parameter.requires_grad_("lora" in name.lower())
        return wrapped

    def _assert_lora_trainable(self) -> None:
        active = [
            (name, parameter)
            for name, parameter in self.train_module.named_parameters()
            if parameter.requires_grad
        ]
        if not active:
            raise RuntimeError("Wan LoRA attach produced no trainable LoRA parameters")
        non_lora = [name for name, _parameter in active if "lora" not in name.lower()]
        if non_lora:
            raise RuntimeError(
                "Wan LoRA attach left non-LoRA parameters trainable: "
                + ", ".join(non_lora)
            )

    def _assert_transformer_consistency(self) -> None:
        if getattr(self.pipeline, "transformer", None) is not self.transformer:
            raise RuntimeError(
                "Wan pipeline.transformer and adapter transformer must reference "
                "the same wrapped module"
            )

    @staticmethod
    def _validated_checkpoint_directory(root: Path, path: Path, *, label: str) -> Path:
        from visual_rl.artifacts.checkpoint import validated_checkpoint_directory

        return validated_checkpoint_directory(root, path, label=label)

    @staticmethod
    def _validated_checkpoint_file(root: Path, path: Path, *, label: str) -> Path:
        from visual_rl.artifacts.checkpoint import validated_checkpoint_file

        return validated_checkpoint_file(root, path, label=label)

    @classmethod
    def _optional_checkpoint_file(
        cls,
        root: Path,
        path: Path,
        *,
        label: str,
    ) -> Path | None:
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect checkpoint {label}: {path}") from exc
        return cls._validated_checkpoint_file(root, path, label=label)

    @classmethod
    def _peft_adapter_weight_path(cls, root: Path, path: Path) -> Path | None:
        for name in WAN_LORA_WEIGHT_NAMES:
            candidate = path / name
            weight = cls._optional_checkpoint_file(
                root,
                candidate,
                label=f"Wan {name}",
            )
            if weight is not None:
                return weight
        return None

    def _find_peft_adapter_path(
        self,
        checkpoint_path: Path,
        *,
        include_stable_subdir: bool = True,
    ) -> Path | None:
        root = self._validated_checkpoint_directory(
            checkpoint_path,
            checkpoint_path,
            label="Wan adapter root",
        )
        bases = [checkpoint_path]
        if include_stable_subdir:
            bases.insert(0, checkpoint_path / WAN_LORA_CHECKPOINT_SUBDIR)
        candidates: list[Path] = []
        for base in bases:
            candidates.extend((base, base / self.adapter_name))
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                candidate.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot inspect Wan adapter directory: {candidate}"
                ) from exc
            adapter_dir = self._validated_checkpoint_directory(
                root,
                candidate,
                label="Wan adapter directory",
            )
            config_path = self._optional_checkpoint_file(
                root,
                adapter_dir / "adapter_config.json",
                label="Wan adapter_config.json",
            )
            weight_path = self._peft_adapter_weight_path(root, adapter_dir)
            if config_path is not None and weight_path is not None:
                return adapter_dir
        return None

    @classmethod
    def _checkpoint_metadata(cls, path: Path) -> dict[str, Any]:
        metadata_path = path / WAN_CHECKPOINT_METADATA_NAME
        metadata_path = cls._optional_checkpoint_file(
            path,
            metadata_path,
            label="Wan adapter_metadata.json",
        )
        if metadata_path is None:
            return {}
        from visual_rl.artifacts.checkpoint import load_json

        metadata = load_json(metadata_path)
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"Wan checkpoint metadata must be an object: {metadata_path}"
            )
        return metadata

    def _load_peft_checkpoint_in_place(self, adapter_path: Path) -> None:
        if not getattr(self.transformer, "peft_config", None):
            raise RuntimeError(
                "Cannot load a Wan LoRA checkpoint into a non-PEFT transformer"
            )
        self._verify_peft_checkpoint_config(adapter_path)
        try:
            from peft.utils.save_and_load import (
                get_peft_model_state_dict,
                load_peft_weights,
                set_peft_model_state_dict,
            )
        except ImportError as exc:  # pragma: no cover - optional train dependency
            raise RuntimeError(
                "Install visual-rl[train] with PEFT to load a Wan LoRA checkpoint"
            ) from exc

        parameter_ids = {
            name: id(parameter)
            for name, parameter in self.transformer.named_parameters()
        }
        expected_state = self._get_active_peft_state(get_peft_model_state_dict)
        adapter_state = load_peft_weights(str(adapter_path), device="cpu")
        if not adapter_state:
            raise RuntimeError("Wan PEFT adapter weight file contains no tensors")
        self._validate_peft_checkpoint_state(expected_state, adapter_state)
        load_result = set_peft_model_state_dict(
            self.transformer,
            adapter_state,
            adapter_name=self.adapter_name,
        )
        current_parameter_ids = {
            name: id(parameter)
            for name, parameter in self.transformer.named_parameters()
        }
        if current_parameter_ids != parameter_ids:
            raise RuntimeError(
                "Wan PEFT checkpoint loading replaced Parameter objects; "
                "optimizer references would be invalid"
            )
        missing, unexpected = self._peft_load_result_keys(load_result)
        active_missing = self._active_adapter_result_keys(missing, expected_state)
        active_unexpected = self._active_adapter_result_keys(
            unexpected,
            expected_state,
        )
        if active_missing:
            raise RuntimeError(
                "Wan PEFT checkpoint is missing keys required by the attached adapter: "
                + ", ".join(str(key) for key in active_missing)
            )
        if active_unexpected:
            raise RuntimeError(
                "Wan PEFT checkpoint reported unexpected active adapter keys: "
                + ", ".join(str(key) for key in active_unexpected)
            )
        restored_state = self._get_active_peft_state(get_peft_model_state_dict)
        self._validate_peft_checkpoint_state(adapter_state, restored_state)
        self._validate_peft_checkpoint_values(adapter_state, restored_state)

    def _get_active_peft_state(self, get_peft_model_state_dict) -> Mapping[str, Any]:
        parameters = inspect.signature(get_peft_model_state_dict).parameters
        kwargs: dict[str, Any] = {}
        if "adapter_name" in parameters:
            kwargs["adapter_name"] = self.adapter_name
        elif self.adapter_name != "default":
            raise RuntimeError(
                "Installed PEFT does not support selecting a non-default adapter state"
            )
        state = get_peft_model_state_dict(self.transformer, **kwargs)
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError("PEFT returned an empty active adapter state contract")
        return state

    @classmethod
    def _validate_peft_checkpoint_state(
        cls,
        expected_state: Mapping[str, Any],
        checkpoint_state: Any,
    ) -> None:
        expected = cls._peft_state_shapes(expected_state, label="active adapter")
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
                "Wan PEFT checkpoint is missing active adapter keys: "
                + ", ".join(missing)
            )
        if extra:
            raise RuntimeError(
                "Wan PEFT checkpoint contains extra adapter keys: " + ", ".join(extra)
            )
        if mismatched:
            details = ", ".join(
                f"{key}: expected {expected[key]}, got {actual[key]}"
                for key in mismatched
            )
            raise RuntimeError(f"Wan PEFT checkpoint shape mismatch: {details}")

    @staticmethod
    def _peft_state_shapes(state: Any, *, label: str) -> dict[str, tuple[int, ...]]:
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(f"Wan PEFT {label} state must be a non-empty mapping")
        shapes: dict[str, tuple[int, ...]] = {}
        for key, value in state.items():
            if not isinstance(key, str) or not key:
                raise RuntimeError(f"Wan PEFT {label} state contains an invalid key")
            shape = getattr(value, "shape", None)
            if shape is None:
                raise RuntimeError(
                    f"Wan PEFT {label} state value has no tensor shape: {key}"
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
                    f"Wan PEFT checkpoint value was not restored exactly: {key}"
                )

    def _active_adapter_result_keys(
        self,
        reported_keys: list[Any],
        expected_state: Mapping[str, Any],
    ) -> list[str]:
        expected = set(expected_state)
        adapter_segment = f".{self.adapter_name}."
        active: list[str] = []
        for raw_key in reported_keys:
            key = str(raw_key)
            normalized = key.replace(adapter_segment, ".")
            if key.endswith(f".{self.adapter_name}"):
                normalized = key[: -len(self.adapter_name) - 1]
            matches_expected = any(
                normalized == candidate
                or normalized.endswith(candidate)
                or candidate.endswith(normalized)
                for candidate in expected
            )
            explicitly_active = adapter_segment in key and (
                "lora_" in key or "modules_to_save" in key
            )
            if matches_expected or explicitly_active:
                active.append(key)
        return active

    @classmethod
    def _peft_load_result_keys(cls, load_result: Any) -> tuple[list[Any], list[Any]]:
        """Normalize PEFT's object, tuple, and legacy ``None`` results."""

        if load_result is None:
            return [], []
        if hasattr(load_result, "missing_keys") and hasattr(
            load_result, "unexpected_keys"
        ):
            missing = getattr(load_result, "missing_keys")
            unexpected = getattr(load_result, "unexpected_keys")
        elif isinstance(load_result, tuple) and len(load_result) == 2:
            missing, unexpected = load_result
        else:
            raise RuntimeError(
                "Wan PEFT checkpoint loader returned an unsupported compatibility result"
            )
        return (
            cls._normalize_peft_key_list(missing, label="missing_keys"),
            cls._normalize_peft_key_list(unexpected, label="unexpected_keys"),
        )

    @staticmethod
    def _normalize_peft_key_list(value: Any, *, label: str) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            return list(value)
        except TypeError as exc:
            raise RuntimeError(
                f"Wan PEFT checkpoint loader returned invalid {label}"
            ) from exc

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

    def _transformer_device_dtype(self):
        import torch

        transformer_dtype = getattr(self.transformer, "dtype", None)
        parameter = None
        try:
            parameter = next(self.transformer.parameters())
        except StopIteration:
            pass
        if not isinstance(transformer_dtype, torch.dtype):
            transformer_dtype = self.dtype
        if not isinstance(transformer_dtype, torch.dtype) and parameter is not None:
            transformer_dtype = parameter.dtype
        if not isinstance(transformer_dtype, torch.dtype):
            raise RuntimeError("Wan transformer dtype is unavailable for recomputation")
        transformer_device = (
            parameter.device
            if parameter is not None
            else torch.device(self.device or "cpu")
        )
        return transformer_device, transformer_dtype

    def _runtime_options(self, rollout_config: dict[str, Any]) -> dict[str, Any]:
        def pick(name: str, default: Any) -> Any:
            return rollout_config.get(name, self.config.get(name, default))

        sde_window_range = pick("sde_window_range", None)
        guidance_scale = self._resolve_guidance_scale(pick("guidance_scale", 5.0))
        num_videos_per_prompt = self._resolve_num_videos_per_prompt(
            pick(
                "num_videos_per_prompt",
                pick("num_video_per_prompt", 1),
            )
        )
        return {
            "num_steps": int(pick("num_steps", 2)),
            "guidance_scale": guidance_scale,
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
            "num_videos_per_prompt": num_videos_per_prompt,
            "train_cfg": self._resolve_train_cfg(
                rollout_config,
                guidance_scale=guidance_scale,
            ),
            "use_camera_trajectory": self._resolve_bool_option(
                "use_camera_trajectory", pick("use_camera_trajectory", False)
            ),
        }

    def _resolve_train_cfg(
        self,
        values: dict[str, Any],
        *,
        guidance_scale: float,
    ) -> bool:
        value = values.get("train_cfg", self.config.get("train_cfg", True))
        if not isinstance(value, bool):
            raise ValueError("Wan train_cfg must resolve to a bool")
        expected = guidance_scale > 1.0
        if value != expected:
            injected = self.config.get("wan_pipeline_with_logprob")
            if callable(injected):
                try:
                    parameters = inspect.signature(injected).parameters
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Cannot inspect the injected Wan pipeline train_cfg contract"
                    ) from exc
                if (
                    "train_cfg" in parameters
                    or "do_classifier_free_guidance" in parameters
                ):
                    return value
                raise RuntimeError(
                    "Injected Wan reference pipeline does not expose train_cfg; it "
                    "only supports the legacy train_cfg=True sampling contract"
                )
            raise ValueError("Wan train_cfg must equal (guidance_scale > 1.0)")
        return value

    @staticmethod
    def _resolve_guidance_scale(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Wan guidance_scale must resolve to a number")
        return float(value)

    @staticmethod
    def _resolve_num_videos_per_prompt(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Wan num_videos_per_prompt must resolve to an integer")
        if value != 1:
            raise ValueError("Wan v1 requires num_videos_per_prompt=1")
        return value

    @staticmethod
    def _resolve_bool_option(name: str, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"Wan {name} must resolve to a bool")
        return value

    @staticmethod
    def _serializable_runtime_options(runtime: dict[str, Any]) -> dict[str, Any]:
        serializable = dict(runtime)
        serializable["device"] = str(serializable.get("device"))
        if serializable.get("sde_window_range") is not None:
            serializable["sde_window_range"] = list(serializable["sde_window_range"])
        return serializable

    def _scheduler_metadata(self) -> dict[str, Any]:
        scheduler = self.scheduler
        metadata = {
            "class": type(scheduler).__name__ if scheduler is not None else None
        }
        timesteps = getattr(scheduler, "timesteps", None)
        if timesteps is not None:
            metadata["timesteps"] = [
                int(item) for item in timesteps.detach().cpu().flatten().tolist()
            ]
        return metadata

    def _encode_prompt_embeds(
        self,
        prompts: list[str],
        *,
        negative_prompts: list[str],
        num_videos_per_prompt: int,
        max_sequence_length: int,
        train_cfg: bool,
    ):
        if hasattr(self.pipeline, "encode_prompt"):
            encoded = self.pipeline.encode_prompt(
                prompt=prompts,
                negative_prompt=negative_prompts,
                do_classifier_free_guidance=train_cfg,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=max_sequence_length,
                device=self.device,
            )
            if isinstance(encoded, tuple):
                return encoded[0], encoded[1]
            raise TypeError(
                "WanPipeline.encode_prompt must return (prompt_embeds, negative_prompt_embeds)."
            )

        with legacy_repo_path(self.repo_root):
            from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt

        text_encoders = [
            item
            for item in [self.text_encoder, self.text_encoder_2]
            if item is not None
        ]
        tokenizers = [
            item for item in [self.tokenizer, self.tokenizer_2] if item is not None
        ]
        prompt_embeds = encode_prompt(
            text_encoders,
            tokenizers,
            prompts,
            max_sequence_length=max_sequence_length,
            num_videos_per_prompt=num_videos_per_prompt,
            device=self.device,
            dtype=self.dtype,
        )
        if not train_cfg:
            return prompt_embeds, None
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

    def _call_pipeline_with_logprob(
        self,
        pipeline_with_logprob,
        *,
        injected: bool,
        train_cfg: bool,
        **kwargs: Any,
    ):
        """Call one explicitly selected reference patch through its fixed allowlist."""

        allowlist = (
            _WORLD_PIPELINE_KWARGS
            if self.wan_backend == "world_r1"
            else _FLASH_PIPELINE_KWARGS
        )
        unsupported = sorted(set(kwargs).difference(allowlist))
        if unsupported:
            raise RuntimeError(
                f"Wan backend {self.wan_backend!r} does not allow pipeline kwargs: "
                + ", ".join(unsupported)
            )
        if injected:
            try:
                parameters = inspect.signature(pipeline_with_logprob).parameters
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Cannot inspect the injected Wan pipeline train_cfg contract"
                ) from exc
            if "train_cfg" in parameters:
                kwargs["train_cfg"] = train_cfg
            elif "do_classifier_free_guidance" in parameters:
                kwargs["do_classifier_free_guidance"] = train_cfg
            elif not train_cfg:
                raise RuntimeError(
                    "Injected Wan reference pipeline does not expose train_cfg; it "
                    "only supports the legacy train_cfg=True sampling contract"
                )
        else:
            guidance_scale = float(kwargs.get("guidance_scale", 1.0))
            if train_cfg != (guidance_scale > 1.0):
                raise RuntimeError(
                    "Wan reference pipeline derives CFG from guidance_scale; train_cfg "
                    "must equal (guidance_scale > 1.0)"
                )
        return pipeline_with_logprob(self.pipeline, **kwargs)

    def _load_wan_pipeline_with_logprob(self):
        injected = self.config.get("wan_pipeline_with_logprob")
        if injected is not None:
            if not callable(injected):
                raise TypeError(
                    "Wan wan_pipeline_with_logprob injection must be callable"
                )
            return injected, "<injected>", True
        with legacy_repo_path(self.repo_root):
            if self.wan_backend == "flash":
                from flow_grpo.diffusers_patch.wan2_1_pipeline_with_logprob_sample import (
                    wan_pipeline_with_logprob,
                )
            else:
                from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import (
                    wan_pipeline_with_logprob,
                )

        return wan_pipeline_with_logprob, self.repo_root, False

    def _load_sde_step_with_logprob(self):
        injected = self.config.get("sde_step_with_logprob")
        if injected is not None:
            if not callable(injected):
                raise TypeError("Wan sde_step_with_logprob injection must be callable")
            return injected
        with legacy_repo_path(self.repo_root):
            if self.wan_backend == "flash":
                from flow_grpo.diffusers_patch.wan2_1_pipeline_with_logprob_sample import (
                    sde_step_with_logprob,
                )
            else:
                from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import (
                    sde_step_with_logprob,
                )

        return sde_step_with_logprob

    def _load_camera_helpers(self):
        get_trajectories = self.config.get("get_camera_trajectories_for_batch")
        prepare_latents = self.config.get("prepare_latents_with_camera")
        if (get_trajectories is None) != (prepare_latents is None):
            raise ValueError(
                "Wan camera helper injection requires both get_camera_trajectories_for_batch "
                "and prepare_latents_with_camera"
            )
        if get_trajectories is not None:
            if not callable(get_trajectories) or not callable(prepare_latents):
                raise TypeError("Wan injected camera helpers must be callable")
            return get_trajectories, prepare_latents
        with legacy_repo_path(self.repo_root):
            from flow_grpo.diffusers_patch.camera_trajectory_utils import (
                get_camera_trajectories_for_batch,
                prepare_latents_with_camera,
            )

        return get_camera_trajectories_for_batch, prepare_latents_with_camera

    def _prepare_world_camera_latents(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        runtime: dict[str, Any],
        generator: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        if self.wan_backend != "world_r1":
            raise RuntimeError(
                "Camera trajectories are only supported by wan_backend='world_r1'"
            )
        get_trajectories, prepare_latents = self._load_camera_helpers()
        result = get_trajectories(
            prompts,
            batch_size=len(prompts),
            frames_per_trajectory=runtime["frames"],
            force_camera_movement=self.config.get("force_camera_movement"),
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise ValueError("get_camera_trajectories_for_batch must return four items")
        trajectories, detected_movements, _expanded_prompts, _motion_profiles = result
        if not isinstance(trajectories, list) or len(trajectories) != len(prompts):
            raise ValueError(
                "Wan camera trajectories must align one-to-one with prompts"
            )
        if not isinstance(detected_movements, list) or len(detected_movements) != len(
            prompts
        ):
            raise ValueError(
                "Wan detected camera movements must align one-to-one with prompts"
            )
        transformer_config = getattr(self.transformer, "config", None)
        num_channels = getattr(transformer_config, "in_channels", None)
        if (
            isinstance(num_channels, bool)
            or not isinstance(num_channels, int)
            or num_channels <= 0
        ):
            raise ValueError(
                "Wan transformer.config.in_channels is required for camera latents"
            )
        vae_scale_factor_temporal = getattr(
            self.pipeline, "vae_scale_factor_temporal", 4
        )
        latents = prepare_latents(
            prompt=prompts,
            batch_size=len(prompts),
            num_channels_latents=num_channels,
            height=runtime["height"],
            width=runtime["width"],
            num_frames=runtime["frames"],
            dtype=self.dtype,
            device=runtime["device"],
            generator=generator,
            latents=None,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            frames_per_trajectory=runtime["frames"],
            force_camera_movement=self.config.get("force_camera_movement"),
            camera_trajectories=trajectories,
            detected_movements_batch=detected_movements,
        )
        copied = [dict(item) for item in metadata]
        for item, trajectory in zip(copied, trajectories, strict=True):
            item["camera_trajectory"] = trajectory
        return latents, copied

    @staticmethod
    def _flash_group_seed(base_seed: int, reference_index: int) -> int:
        """Derive a stable per-index seed independent of grouping dict order."""

        return (int(base_seed) + 1_000_003 * (reference_index + 1)) % (2**63 - 1)

    @staticmethod
    @contextmanager
    def _fork_flash_group_rng(device: Any, seed: int):
        """Isolate the reference patch's implicit selected-step RNG."""

        import torch

        target = torch.device(device)
        cuda_devices: list[int] = []
        if target.type == "cuda":
            cuda_devices.append(
                target.index
                if target.index is not None
                else torch.cuda.current_device()
            )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            yield

    @staticmethod
    def _select_flash_rows(value: Any, positions: list[int]):
        import torch

        if not isinstance(value, torch.Tensor):
            raise ValueError("Flash prompt embeddings must be tensors")
        index = torch.as_tensor(positions, device=value.device, dtype=torch.long)
        return value.index_select(0, index)

    @staticmethod
    def _restore_flash_tensor_order(value: Any, grouped_positions: list[int]):
        import torch

        if not isinstance(value, torch.Tensor) or value.shape[0] != len(
            grouped_positions
        ):
            raise ValueError("Flash grouped tensor does not align with batch order")
        inverse = [0] * len(grouped_positions)
        for grouped_row, original_position in enumerate(grouped_positions):
            inverse[original_position] = grouped_row
        index = torch.as_tensor(inverse, device=value.device, dtype=torch.long)
        return value.index_select(0, index)

    @staticmethod
    def _restore_flash_sequence_order(
        values: list[Any], grouped_positions: list[int]
    ) -> list[Any]:
        if len(values) != len(grouped_positions):
            raise ValueError("Flash grouped metadata does not align with batch order")
        restored: list[Any] = [None] * len(values)
        for grouped_row, original_position in enumerate(grouped_positions):
            restored[original_position] = values[grouped_row]
        return restored

    @staticmethod
    def _as_flash_batch_tensor(value: Any, batch_size: int, name: str):
        import torch

        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 1
            or value.shape[0] != batch_size
        ):
            raise ValueError(
                f"Flash {name} must be a tensor with batch dimension {batch_size}"
            )
        return value

    @staticmethod
    def _as_flash_transition_tensor(value: Any, batch_size: int, name: str):
        return WorldR1WanLegacyAdapter._as_flash_batch_tensor(
            value, batch_size, name
        ).unsqueeze(1)

    @staticmethod
    def _as_flash_scalar_transition(value: Any, batch_size: int, name: str):
        import torch

        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.numel() != batch_size:
            raise ValueError(
                f"Flash {name} must contain exactly one value per batch item"
            )
        return tensor.reshape(batch_size, 1)

    @classmethod
    def _normalize_flash_coefficient(cls, value: Any, *, batch_size: int):
        import torch

        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.numel() % batch_size != 0:
            raise ValueError(
                "Flash reference coefficient does not align with batch size"
            )
        flattened = tensor.reshape(batch_size, -1)
        coefficient = flattened[:, :1]
        if not torch.allclose(flattened, coefficient.expand_as(flattened)):
            raise ValueError(
                "Flash reference coefficient must be scalar per batch item"
            )
        if not torch.isfinite(coefficient).all() or not (coefficient > 0).all():
            raise ValueError("Flash reference coefficient must be finite and positive")
        return coefficient

    def _flash_reference_coefficient(self, *, latents, next_latents, timesteps):
        import torch

        step_result = self._load_sde_step_with_logprob()(
            self.scheduler,
            torch.zeros_like(latents),
            timesteps,
            latents.float(),
            prev_sample=next_latents.float(),
            return_dt_and_std_dev_t=True,
        )
        if not isinstance(step_result, (tuple, list)) or len(step_result) != 6:
            raise ValueError(
                "Flash sde_step_with_logprob must return six items for coefficient sampling"
            )
        # Published Wan2.1 returns coe=1/denominator as item six; its t=999
        # value is 7.476982..., matching the training script's value_dict 7.4770.
        return self._normalize_flash_coefficient(
            step_result[5], batch_size=latents.shape[0]
        )

    @classmethod
    def _flash_selected_kl(
        cls,
        all_kl: Any,
        *,
        reference_index: int,
        batch_size: int,
        like: Any,
    ):
        import torch

        if not all_kl:
            return torch.zeros_like(like)
        if len(all_kl) == 1:
            value = all_kl[0]
        elif reference_index < len(all_kl):
            value = all_kl[reference_index]
        else:
            raise ValueError("Flash KL values do not contain the selected timestep")
        return cls._as_flash_scalar_transition(value, batch_size, "kl").to(
            device=like.device,
            dtype=like.dtype,
        )

    def _normalize_pipeline_result(self, result: Any):
        if len(result) == 5:
            return result
        if len(result) == 4:
            videos, latents, log_probs, kl = result
            timesteps = getattr(self.scheduler, "timesteps", None)
            if timesteps is None:
                raise ValueError(
                    "4-item Wan pipeline result requires scheduler.timesteps to reconstruct timesteps."
                )
            return videos, latents, log_probs, kl, list(timesteps[: len(log_probs)])
        raise ValueError(
            f"Wan pipeline result must have 4 or 5 items, got {len(result)}."
        )


MODEL_ADAPTERS.register("world_r1_wan_legacy", WorldR1WanLegacyAdapter)
