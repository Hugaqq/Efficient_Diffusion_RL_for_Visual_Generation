"""SD3/SD3.5 TempFlow adapter backed by the verified reference path."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
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
    media_type = "image"

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
            raise ImportError("Install visual-rl[train] to use SD3 TempFlow adapter.") from exc

        model_path = require_model_path(self.config, self.name)
        with legacy_repo_path(self.repo_root):
            from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
            from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_perstep import (
                pipeline_with_logprob as pipeline_with_logprob_perstep,
            )
            from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
            from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

        self._pipeline_with_logprob = pipeline_with_logprob
        self._pipeline_with_logprob_perstep = pipeline_with_logprob_perstep
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

    def named_parameters(self):
        self._ensure_loaded()
        return [
            (f"transformer.{name}", parameter)
            for name, parameter in self.transformer.named_parameters()
            if parameter.requires_grad
        ]

    def branch_transition_count(self, rollout_config: dict[str, Any]) -> int:
        transition_count = int(rollout_config.get("num_steps", 3)) - 1
        if transition_count < 1:
            raise ValueError("SD3 TempFlow branching requires num_steps >= 2")
        return transition_count

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

    def sample_branching(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        rollout_config: dict[str, Any],
    ) -> RolloutBatch:
        import torch

        self._ensure_loaded()
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
        generator = make_generator(self.device, rollout_config.get("seed"))

        with torch.no_grad(), self._perstep_sde_generator(generator):
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
        scheduler_timestep = torch.as_tensor(
            self.pipeline.scheduler.timesteps
        )[branch_step_index].detach()
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

        selected_latents = torch.as_tensor(main_latents[branch_step_index]).repeat_interleave(
            branch_count,
            dim=0,
        )
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
        transition_count = int(
            rollout_config.get("transition_count", len(log_probs))
        )
        positions = torch.arange(transition_count, dtype=torch.float32)
        global_weights = torch.sqrt(
            (transition_count - positions).clamp_min(1.0) / float(transition_count)
        )
        global_weights = global_weights / global_weights.mean().clamp_min(1e-6)
        noise_weight = float(global_weights[branch_step_index])
        selected_timesteps = scheduler_timestep.to(
            device=selected_log_probs.device
        ).reshape(1, 1).expand(rows, 1).clone()
        return RolloutBatch(
            prompts=expanded_prompts,
            metadata=expanded_metadata,
            media=selected_media,
            latents=selected_latents[:, None].detach(),
            next_latents=selected_next[:, None].detach(),
            timesteps=selected_timesteps,
            old_log_probs=selected_log_probs.reshape(rows, 1).detach(),
            kl=selected_kl.reshape(rows, 1).detach(),
            branch_ids=torch.as_tensor(branch_ids, dtype=torch.long),
            epoch_tag=rollout_config.get("epoch_tag"),
            seed=rollout_config.get("seed"),
            model_metadata={
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
                "sde_generator_bound": generator is not None,
                "reference_exploration_k": exploration_k,
            },
            model_tensors={
                "prompt_embeds": prompt_embeds.repeat_interleave(branch_count, dim=0).detach(),
                "pooled_prompt_embeds": pooled_prompt_embeds.repeat_interleave(branch_count, dim=0).detach(),
                "negative_prompt_embeds": negative_prompt_embeds.repeat_interleave(branch_count, dim=0).detach(),
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds.repeat_interleave(
                    branch_count,
                    dim=0,
                ).detach(),
            },
        )

    def save_pretrained(self, output_dir: str) -> None:
        self._ensure_loaded()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        import torch

        if hasattr(self.transformer, "save_pretrained"):
            self.transformer.save_pretrained(path / "transformer")
        torch.save(self.transformer.state_dict(), path / "transformer_state.pt")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        import torch

        self._ensure_loaded()
        state_path = Path(checkpoint_dir) / "transformer_state.pt"
        if not state_path.exists():
            raise RuntimeError(f"Missing SD3 transformer state: {state_path}")
        state = torch.load(
            state_path,
            map_location=self.device,
            weights_only=False,
        )
        self.transformer.load_state_dict(state)

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

    @contextmanager
    def _perstep_sde_generator(self, generator):
        """Bind our generator through an upstream call site that omits it."""

        function_globals = getattr(
            self._pipeline_with_logprob_perstep,
            "__globals__",
            None,
        )
        original = (
            function_globals.get("sde_step_with_logprob")
            if isinstance(function_globals, dict)
            else None
        )
        if generator is None or not callable(original):
            yield
            return

        def seeded_sde_step(*args, **kwargs):
            deterministic = bool(kwargs.get("determistic", False))
            positional_generator = len(args) >= 6 and args[5] is not None
            if (
                not deterministic
                and not positional_generator
                and kwargs.get("generator") is None
            ):
                kwargs["generator"] = generator
            return original(*args, **kwargs)

        function_globals["sde_step_with_logprob"] = seeded_sde_step
        try:
            yield
        finally:
            if function_globals.get("sde_step_with_logprob") is seeded_sde_step:
                function_globals["sde_step_with_logprob"] = original

    @staticmethod
    def _branch_gather_indices(
        expanded_size: int,
        parent_count: int,
        branch_count: int,
    ) -> tuple[int, list[int]]:
        if parent_count < 1 or expanded_size % parent_count:
            raise ValueError("Per-step branching output is not grouped by parent prompt")
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
