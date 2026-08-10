"""Canonical World-R1 camera initialization and early-step conditioner."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from visual_rl.algorithms.conditioning.config import WorldR1CameraConfig
from visual_rl.algorithms.conditioning.interface import (
    ConditionInitialization,
    LatentConditioner,
    LatentSpec,
)
from visual_rl.core.contracts import DeclaredContract, LatentLayout
from visual_rl.data.samples.items import (
    CameraConditionPayload,
    camera_condition_batch_identity,
    camera_condition_identity,
)

__all__ = (
    "CameraConditionState",
    "WorldR1CameraConditioner",
    "WorldR1CameraConfig",
)


@dataclass(frozen=True, slots=True)
class CameraConditionState:
    latent_spec: LatentSpec
    prompts: tuple[str, ...]
    camera_trajectory: Any
    trajectory_strings: tuple[tuple[tuple[str, str], ...], ...]
    movement_names: tuple[tuple[str, ...], ...]
    conditioner_config_identity: str
    row_condition_identities: tuple[str, ...]
    condition_identity: str
    active_steps: int
    wrap_strength: float
    camera_delta: Any | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        import torch

        if not isinstance(self.latent_spec, LatentSpec):
            raise TypeError("latent_spec must be a LatentSpec")
        batch_size = self.latent_spec.batch_size
        if type(self.prompts) is not tuple or len(self.prompts) != batch_size:
            raise ValueError("prompts must contain B entries")
        if any(not isinstance(item, str) or not item for item in self.prompts):
            raise ValueError("prompts must be non-empty strings")
        if not isinstance(self.camera_trajectory, torch.Tensor):
            raise TypeError("camera_trajectory must be a torch.Tensor")
        if tuple(self.camera_trajectory.shape) != (
            batch_size,
            self.latent_spec.output_frames,
            4,
            4,
        ):
            raise ValueError("camera_trajectory must have shape [B,F,4,4]")
        if self.camera_trajectory.dtype != torch.float32:
            raise TypeError("camera trajectory must be stored in FP32")
        if self.camera_trajectory.requires_grad or not bool(
            torch.isfinite(self.camera_trajectory).all()
        ):
            raise ValueError("camera trajectory must be detached and finite")
        for name in ("trajectory_strings", "movement_names"):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != batch_size:
                raise ValueError(f"{name} must contain B entries")
        for name in ("conditioner_config_identity", "condition_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 identity")
        row_identities = self.row_condition_identities
        if type(row_identities) is not tuple or len(row_identities) != batch_size:
            raise ValueError("row_condition_identities must contain B entries")
        expected_row_identities = tuple(
            camera_condition_identity(
                self.camera_trajectory[index],
                self.conditioner_config_identity,
            )
            for index in range(batch_size)
        )
        if row_identities != expected_row_identities:
            raise ValueError(
                "row_condition_identities must match camera trajectory content"
            )
        if self.condition_identity != camera_condition_batch_identity(row_identities):
            raise ValueError("condition_identity must match the ordered row identities")
        if type(self.active_steps) is not int or self.active_steps < 0:
            raise ValueError("active_steps must be non-negative")
        if self.camera_delta is not None:
            if not isinstance(self.camera_delta, torch.Tensor):
                raise TypeError("camera_delta must be a torch.Tensor")
            expected = (
                batch_size,
                self.latent_spec.channels,
                self.latent_spec.latent_frames,
                self.latent_spec.latent_height,
                self.latent_spec.latent_width,
            )
            if tuple(self.camera_delta.shape) != expected:
                raise ValueError("camera_delta shape does not match latent geometry")
            if self.camera_delta.dtype != torch.float32:
                raise TypeError("camera_delta must be stored in FP32")
            if self.camera_delta.requires_grad or not bool(
                torch.isfinite(self.camera_delta).all()
            ):
                raise ValueError("camera_delta must be detached and finite")


class WorldR1CameraConditioner(LatentConditioner):
    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = "visual_rl.algorithms.conditioning.config:WorldR1CameraConfig"

    def __init__(self, config: WorldR1CameraConfig) -> None:
        if not isinstance(config, WorldR1CameraConfig):
            raise TypeError("config must be WorldR1CameraConfig")
        self.config = config

    @classmethod
    def describe(cls, config: object) -> DeclaredContract:
        if not isinstance(config, WorldR1CameraConfig):
            raise TypeError("config must be WorldR1CameraConfig")
        return config.describe_contract()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> WorldR1CameraConditioner:
        del runtime_context
        if not isinstance(config, WorldR1CameraConfig):
            raise TypeError("config must be WorldR1CameraConfig")
        return cls(config)

    def bind_model_geometry(self, geometry: object) -> LatentSpec:
        """Bind only generic video geometry; the model remains conditioner-blind."""

        from visual_rl.models.scheduler import ModelScheduleContext

        if not isinstance(geometry, ModelScheduleContext):
            raise TypeError(
                "camera conditioner geometry must implement ModelScheduleContext"
            )
        if geometry.layout is not LatentLayout.BCTHW:
            raise ValueError("camera conditioner requires BCTHW model geometry")
        spec = super().bind_model_geometry(geometry)
        if spec.output_frames != self.config.frames_per_trajectory:
            raise ValueError(
                "camera trajectory length must match model output-frame geometry"
            )
        return spec

    def prepare(
        self,
        prompts: tuple[str, ...],
        latent_spec: LatentSpec,
        *,
        generator: Any,
    ) -> CameraConditionState:
        import torch

        from visual_rl.algorithms.conditioning.camera_math import (
            TrajectoryGenerator,
            get_camera_trajectories_for_batch,
            parse_camera_matrix_torch,
        )

        _generator(generator, latent_spec.device)
        if type(prompts) is not tuple or len(prompts) != latent_spec.batch_size:
            raise ValueError("prompts must contain one entry per latent row")
        trajectories, movements, _, _ = get_camera_trajectories_for_batch(
            list(prompts),
            batch_size=latent_spec.batch_size,
            frames_per_trajectory=self.config.frames_per_trajectory,
            force_camera_movement=self.config.force_camera_movement,
        )
        frozen: list[tuple[tuple[str, str], ...]] = []
        matrices = []
        for trajectory in trajectories:
            if trajectory is None:
                trajectory = TrajectoryGenerator(
                    [3390, 1380, 240],
                    num_frames=latent_spec.output_frames,
                ).generate("fixed")
            items = tuple(
                sorted(
                    trajectory.items(),
                    key=lambda item: int(item[0].replace("frame", "")),
                )
            )
            frozen.append(items)
            selected = _uniform_indices(len(items), latent_spec.output_frames)
            matrices.append(
                torch.stack(
                    [
                        parse_camera_matrix_torch(
                            items[index][1],
                            device=latent_spec.device,
                            dtype=torch.float32,
                        )
                        for index in selected
                    ]
                )
            )
        camera = torch.stack(matrices).detach()
        row_identities = tuple(
            camera_condition_identity(
                camera[index],
                self.config.config_identity,
            )
            for index in range(latent_spec.batch_size)
        )
        identity = camera_condition_batch_identity(row_identities)
        return CameraConditionState(
            latent_spec=latent_spec,
            prompts=prompts,
            camera_trajectory=camera,
            trajectory_strings=tuple(frozen),
            movement_names=tuple(tuple(item) for item in movements),
            conditioner_config_identity=self.config.config_identity,
            row_condition_identities=row_identities,
            condition_identity=identity,
            active_steps=self.config.guidance_steps,
            wrap_strength=float(self.config.wrap_strength),
        )

    def initialize_latents(
        self,
        base_latents: Any,
        state: object,
        *,
        generator: Any,
    ) -> ConditionInitialization:
        import torch

        from visual_rl.algorithms.conditioning.camera_math import (
            apply_wrap_strength_to_latents,
            generate_camera_warped_latents,
            lowpass_latent_delta,
            per_sample_mean_std,
        )

        if not isinstance(state, CameraConditionState):
            raise TypeError("state must be CameraConditionState")
        state.validate()
        _generator(generator, base_latents.device)
        expected = (
            state.latent_spec.batch_size,
            state.latent_spec.channels,
            state.latent_spec.latent_frames,
            state.latent_spec.latent_height,
            state.latent_spec.latent_width,
        )
        if (
            not isinstance(base_latents, torch.Tensor)
            or tuple(base_latents.shape) != expected
        ):
            raise ValueError("base_latents do not match the prepared latent geometry")
        if base_latents.requires_grad or not bool(torch.isfinite(base_latents).all()):
            raise ValueError("base_latents must be detached and finite")
        original_dtype = base_latents.dtype
        base_fp32 = base_latents.float()
        initialized = []
        for row, (trajectory_items, movement_names) in enumerate(
            zip(state.trajectory_strings, state.movement_names, strict=True)
        ):
            if not movement_names or self.config.wrap_strength == 0:
                initialized.append(base_fp32[row : row + 1])
                continue
            with _generator_scope(generator, base_latents.device):
                wrapped = generate_camera_warped_latents(
                    dict(trajectory_items),
                    batch_size=1,
                    num_channels_latents=state.latent_spec.channels,
                    height=state.latent_spec.output_height,
                    width=state.latent_spec.output_width,
                    num_frames=state.latent_spec.output_frames,
                    dtype=torch.float32,
                    device=base_latents.device,
                    spatial_compression=state.latent_spec.spatial_compression,
                    temporal_compression=state.latent_spec.temporal_compression,
                    noise_downtemp_interp=self.config.noise_downtemp_interp,
                    noise_downspatial_mode=self.config.noise_downspatial_mode,
                    noise_degradation=self.config.noise_degradation,
                    flow_scale=self.config.flow_scale,
                )
            base_row = base_fp32[row : row + 1]
            candidate = apply_wrap_strength_to_latents(
                base_row,
                wrapped.float(),
                float(self.config.wrap_strength),
                injection_mode=self.config.injection_mode,
                delta_lowpass_kernel=self.config.delta_lowpass_kernel,
            )
            base_mean, base_std = per_sample_mean_std(base_row)
            candidate_mean, candidate_std = per_sample_mean_std(candidate)
            candidate = (candidate - candidate_mean) / (candidate_std + 1e-6)
            initialized.append(candidate * base_std + base_mean)
        initialized_fp32 = torch.cat(initialized, dim=0)
        delta = lowpass_latent_delta(
            initialized_fp32 - base_fp32,
            self.config.delta_lowpass_kernel,
        ).detach()
        initialized_value = initialized_fp32.to(dtype=original_dtype).detach()
        updated = replace(state, camera_delta=delta)
        updated.validate()
        payloads = tuple(
            CameraConditionPayload(
                camera_trajectory=updated.camera_trajectory[index].detach(),
                conditioner_config_identity=(updated.conditioner_config_identity),
            )
            for index in range(updated.latent_spec.batch_size)
        )
        if tuple(payload.condition_identity for payload in payloads) != (
            updated.row_condition_identities
        ):
            raise RuntimeError(
                "camera conditioner payload identities drifted from prepared state"
            )
        return ConditionInitialization(initialized_value, updated, payloads)

    def after_step(
        self,
        step_index: int,
        timestep: Any,
        next_latents: Any,
        state: object,
    ) -> Any:
        import torch

        from visual_rl.algorithms.conditioning.camera_math import (
            build_stepwise_delta_callback,
        )

        if type(step_index) is not int or step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not isinstance(state, CameraConditionState):
            raise TypeError("state must be CameraConditionState")
        state.validate()
        if state.camera_delta is None:
            raise RuntimeError("initialize_latents() must run before after_step()")
        if step_index >= state.active_steps or state.wrap_strength <= 0:
            return next_latents
        callback = build_stepwise_delta_callback(
            state.camera_delta.to(device=next_latents.device, dtype=torch.float32),
            state.wrap_strength,
            state.active_steps,
        )
        if callback is None:
            return next_latents
        result = callback(
            None,
            step_index,
            timestep,
            {"latents": next_latents.float()},
        )["latents"]
        return result.to(dtype=next_latents.dtype)


def _uniform_indices(length: int, target: int) -> tuple[int, ...]:
    if length < 1 or target < 1:
        raise ValueError("trajectory lengths must be positive")
    if target == 1:
        return (0,)
    return tuple(round(index * (length - 1) / (target - 1)) for index in range(target))


def _generator(value: Any, device: Any) -> None:
    import torch

    if not isinstance(value, torch.Generator):
        raise TypeError("camera conditioner requires an explicit torch.Generator")
    generator_device = torch.device(value.device)
    target_device = torch.device(device)
    if generator_device.type != target_device.type:
        raise ValueError("generator device type must match latent device type")


@contextmanager
def _generator_scope(generator: Any, device: Any):
    """Bridge legacy global-RNG camera math without leaking global RNG state."""

    import torch

    _generator(generator, device)
    target = torch.device(device)
    if target.type == "cuda":
        index = (
            target.index if target.index is not None else torch.cuda.current_device()
        )
        original = torch.cuda.get_rng_state(index)
        torch.cuda.set_rng_state(generator.get_state(), index)
        try:
            yield
        finally:
            generator.set_state(torch.cuda.get_rng_state(index))
            torch.cuda.set_rng_state(original, index)
    else:
        original = torch.random.get_rng_state()
        torch.random.set_rng_state(generator.get_state())
        try:
            yield
        finally:
            generator.set_state(torch.random.get_rng_state())
            torch.random.set_rng_state(original)
