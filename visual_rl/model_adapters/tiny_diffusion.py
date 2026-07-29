"""Tiny trainable diffusion-like image adapter used by CPU contract tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Literal

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


class TinyDiffusionAdapter(ModelAdapter):
    """The sole lightweight builtin model.

    The rollout engine has already expanded prompts and assigned row identities
    before this adapter is called.  This implementation therefore samples one
    tensor row for every request row and echoes the request identity unchanged.
    """

    MEDIA_TYPE: ClassVar[Literal["image"]] = "image"
    CHANNELS: ClassVar[int] = 3

    def __init__(self, *, image_size: int, device: object) -> None:
        import torch

        if type(image_size) is not int or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        self.image_size = image_size
        self.device = torch.device(device)
        self._train_module = torch.nn.Module()
        self._train_module.register_buffer(
            "base_color_bias",
            torch.zeros(self.CHANNELS, device=self.device),
            persistent=False,
        )
        self._train_module.register_parameter(
            "color_bias",
            torch.nn.Parameter(torch.zeros(self.CHANNELS, device=self.device)),
        )

    @property
    def train_module(self):
        return self._train_module

    @property
    def color_bias(self):
        return self._train_module.color_bias

    @property
    def base_color_bias(self):
        return self._train_module.base_color_bias

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, object],
        context: ResolutionContext,
    ) -> Mapping[str, object]:
        del context
        if not isinstance(raw, Mapping):
            raise ConfigError("model.params must be a mapping", key="model.params")
        if set(raw) != {"image_size"}:
            unknown = sorted(set(raw) - {"image_size"})
            missing = sorted({"image_size"} - set(raw))
            if unknown:
                raise ConfigError(
                    f"Unknown model.params: {unknown}",
                    key="model.params",
                )
            raise ConfigError(
                f"Missing model.params: {missing}",
                key="model.params",
            )
        image_size = raw["image_size"]
        if type(image_size) is not int or image_size <= 0:
            raise ConfigError(
                "model.params.image_size must be a positive integer",
                key="model.params.image_size",
            )
        return FrozenMapping({"image_size": image_size})

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, object],
        context: ValidationContext,
    ) -> tuple[ValidationCheck, ...]:
        del cls, resolved, context
        return ()

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, object],
        context: RuntimeBuildContext,
    ) -> TinyDiffusionAdapter:
        if set(resolved) != {"image_size"}:
            raise ConfigError(
                "resolved tiny_diffusion params must contain exactly image_size",
                key="model.params",
            )
        return cls(image_size=int(resolved["image_size"]), device=context.device)

    def sample(self, request: RolloutRequest) -> RolloutBatch:
        import torch

        was_training = self.train_module.training
        self.train_module.eval()
        try:
            with torch.no_grad():
                if request.kind == "full_trajectory":
                    batch = self._sample_full(request)
                elif request.kind == "single_step":
                    batch = self._sample_selected(request)
                elif request.kind == "branching":
                    batch = self._sample_branches(request)
                else:  # RolloutRequest validates this; keep the boundary closed.
                    raise RunError(f"unsupported rollout kind: {request.kind!r}")
            batch.validate_against(request)
            return batch
        finally:
            self.train_module.train(was_training)

    def _sample_full(self, request: RolloutRequest) -> RolloutBatch:
        import torch

        batch_size = len(request.prompts)
        generator = self._generator(request.context.seed)
        shape = (
            batch_size,
            request.num_steps,
            self.CHANNELS,
            self.image_size,
            self.image_size,
        )
        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
        ) * 0.25
        noise = torch.randn(
            shape,
            generator=generator,
            device=self.device,
        ) * 0.05
        bias = (
            self.base_color_bias + self.color_bias.detach()
        ).view(1, 1, self.CHANNELS, 1, 1)
        next_latents = latents + noise + bias
        timesteps = torch.arange(
            request.num_steps,
            dtype=torch.int64,
            device=self.device,
        ).repeat(batch_size, 1)
        return self._make_batch(
            request,
            latents=latents,
            next_latents=next_latents,
            timesteps=timesteps,
            selected_timestep_index=None,
            branch_step_index=None,
            trajectory_step_index=None,
            transition_std_dev=None,
            flash_coefficient=None,
        )

    def _sample_selected(self, request: RolloutRequest) -> RolloutBatch:
        import torch

        if request.selected_timestep_index is None:
            raise RunError("single_step request is missing selected timestep indices")
        batch_size = len(request.prompts)
        generator = self._generator(request.context.seed)
        shape = (
            batch_size,
            1,
            self.CHANNELS,
            self.image_size,
            self.image_size,
        )
        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
        ) * 0.25
        noise = torch.randn(
            shape,
            generator=generator,
            device=self.device,
        ) * 0.05
        bias = (
            self.base_color_bias + self.color_bias.detach()
        ).view(1, 1, self.CHANNELS, 1, 1)
        next_latents = latents + noise + bias
        selected = torch.tensor(
            request.selected_timestep_index,
            dtype=torch.int64,
            device=self.device,
        )
        return self._make_batch(
            request,
            latents=latents,
            next_latents=next_latents,
            timesteps=selected[:, None],
            selected_timestep_index=selected,
            branch_step_index=None,
            trajectory_step_index=None,
            transition_std_dev=None,
            flash_coefficient=torch.ones(
                (batch_size, 1),
                dtype=latents.dtype,
                device=self.device,
            ),
        )

    def _sample_branches(self, request: RolloutRequest) -> RolloutBatch:
        import torch

        if request.branch_step_index is None:
            raise RunError("branching request is missing branch timestep indices")
        if len(set(request.branch_step_index)) != 1:
            raise RunError(
                "branching requires one shared global branch timestep"
            )
        batch_size = len(request.prompts)
        sample_shape = (
            self.CHANNELS,
            self.image_size,
            self.image_size,
        )
        generator = self._generator(request.context.seed)
        bias = (
            self.base_color_bias + self.color_bias.detach()
        ).view(self.CHANNELS, 1, 1)
        selected_latents = torch.empty(
            (batch_size, *sample_shape),
            device=self.device,
        )

        group_rows: dict[str, list[int]] = {}
        for row, group_id in enumerate(request.group_id):
            group_rows.setdefault(group_id, []).append(row)
        for rows in group_rows.values():
            state = torch.randn(
                sample_shape,
                generator=generator,
                device=self.device,
            ) * 0.25
            wanted = {
                row: request.branch_step_index[row]
                for row in rows
            }
            for step in range(max(wanted.values()) + 1):
                for row, selected_step in wanted.items():
                    if selected_step == step:
                        selected_latents[row].copy_(state)
                if step < max(wanted.values()):
                    shared_noise = torch.randn(
                        sample_shape,
                        generator=generator,
                        device=self.device,
                    ) * 0.05
                    state = state + shared_noise + bias

        next_latents = torch.empty_like(selected_latents)
        for row in range(batch_size):
            branch_noise = torch.randn(
                sample_shape,
                generator=generator,
                device=self.device,
            ) * 0.05
            next_latents[row] = selected_latents[row] + branch_noise + bias
        branch_steps = torch.tensor(
            request.branch_step_index,
            dtype=torch.int64,
            device=self.device,
        )
        return self._make_batch(
            request,
            latents=selected_latents[:, None],
            next_latents=next_latents[:, None],
            timesteps=branch_steps[:, None],
            selected_timestep_index=None,
            branch_step_index=branch_steps,
            trajectory_step_index=branch_steps[:1].clone(),
            transition_std_dev=torch.full(
                (batch_size, 1),
                0.05,
                dtype=selected_latents.dtype,
                device=self.device,
            ),
            flash_coefficient=None,
        )

    def _make_batch(
        self,
        request: RolloutRequest,
        *,
        latents,
        next_latents,
        timesteps,
        selected_timestep_index,
        branch_step_index,
        trajectory_step_index,
        transition_std_dev,
        flash_coefficient,
    ) -> RolloutBatch:
        import torch

        bias = (
            self.base_color_bias + self.color_bias.detach()
        ).view(1, 1, self.CHANNELS, 1, 1)
        old_log_probs = -(
            (next_latents - latents - bias) ** 2
        ).mean(dim=(2, 3, 4))
        transition_mask = torch.ones_like(old_log_probs, dtype=torch.bool)
        media = torch.sigmoid(next_latents[:, -1])
        return RolloutBatch(
            prompts=request.prompts,
            metadata=request.metadata,
            media=media.detach(),
            latents=latents.detach(),
            next_latents=next_latents.detach(),
            timesteps=timesteps.detach(),
            old_log_probs=old_log_probs.detach(),
            transition_mask=transition_mask.detach(),
            sample_id=request.sample_id,
            prompt_id=request.prompt_id,
            group_id=request.group_id,
            branch_id=request.branch_id,
            media_layout="BCHW",
            camera_trajectory=None,
            context=request.context,
            selected_timestep_index=selected_timestep_index,
            flash_coefficient=flash_coefficient,
            branch_step_index=branch_step_index,
            trajectory_step_index=trajectory_step_index,
            transition_std_dev=transition_std_dev,
            recompute_payload={},
            artifact_metadata={
                "adapter": "tiny_diffusion",
                "image_size": self.image_size,
            },
        )

    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ) -> PolicyRecomputeStats:
        import torch

        was_training = self.train_module.training
        self.train_module.train(True)
        try:
            base_bias = self.base_color_bias.view(
                1,
                1,
                self.CHANNELS,
                1,
                1,
            )
            adapter_delta = self.color_bias.view(
                1,
                1,
                self.CHANNELS,
                1,
                1,
            )
            current_mean = batch.latents + base_bias + adapter_delta
            new_log_probs = -(
                (batch.next_latents - current_mean) ** 2
            ).mean(dim=(2, 3, 4))
            if require_reference:
                reference_mean = self._reference_transition_mean(
                    batch,
                    base_bias,
                )
                transition_std = torch.full(
                    tuple(batch.old_log_probs.shape),
                    0.05,
                    dtype=current_mean.dtype,
                    device=current_mean.device,
                )
                result = PolicyRecomputeStats(
                    new_log_probs=new_log_probs,
                    current_transition_mean=current_mean,
                    transition_std=transition_std,
                    reference_transition_mean=reference_mean,
                )
            else:
                result = PolicyRecomputeStats(new_log_probs=new_log_probs)
            result.validate_against(
                batch,
                require_reference=require_reference,
            )
            return result
        finally:
            self.train_module.train(was_training)

    def _reference_transition_mean(self, batch: RolloutBatch, base_bias):
        import torch

        with torch.no_grad():
            return (batch.latents + base_bias).detach()

    def _generator(self, seed: int):
        import torch

        generator_device = self.device if self.device.type == "cuda" else "cpu"
        return torch.Generator(device=generator_device).manual_seed(seed)

    def close(self) -> None:
        return None
