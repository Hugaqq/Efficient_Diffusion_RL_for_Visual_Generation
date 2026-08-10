"""Canonical algorithm-owned latent-conditioning runtime boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from visual_rl.core.contracts import LatentLayout
from visual_rl.data.samples.items import ConditionPayload

__all__ = ("ConditionInitialization", "LatentConditioner", "LatentSpec")


@dataclass(frozen=True, slots=True)
class LatentSpec:
    batch_size: int
    channels: int
    latent_frames: int
    latent_height: int
    latent_width: int
    output_frames: int
    output_height: int
    output_width: int
    temporal_compression: int
    spatial_compression: int
    device: Any
    dtype: Any

    @classmethod
    def from_model_geometry(cls, geometry: object) -> LatentSpec:
        """Project a generic model schedule context into conditioner input.

        The dependency points from conditioning to the import-safe model port;
        a model never imports or constructs this conditioner-owned type.
        """

        from visual_rl.models.scheduler import ModelScheduleContext

        if not isinstance(geometry, ModelScheduleContext):
            raise TypeError("geometry must implement ModelScheduleContext")
        if geometry.layout not in {LatentLayout.BCHW, LatentLayout.BCTHW}:
            raise ValueError(
                "packed_sequence model geometry cannot bind a spatial conditioner"
            )
        spatial_stride = geometry.spatial_stride
        if spatial_stride is None:
            raise ValueError("conditioner binding requires a spatial stride")
        height_stride, width_stride = spatial_stride
        if height_stride != width_stride:
            raise ValueError(
                "conditioner binding requires equal height and width strides"
            )
        if geometry.layout is LatentLayout.BCHW:
            batch_size, channels, latent_height, latent_width = geometry.shape
            latent_frames = 1
            temporal_stride = 1
        elif geometry.layout is LatentLayout.BCTHW:
            (
                batch_size,
                channels,
                latent_frames,
                latent_height,
                latent_width,
            ) = geometry.shape
            temporal_stride = geometry.temporal_stride
            if temporal_stride is None:
                raise ValueError("video conditioner binding requires a temporal stride")
        return cls(
            batch_size=batch_size,
            channels=channels,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            output_frames=(latent_frames - 1) * temporal_stride + 1,
            output_height=latent_height * height_stride,
            output_width=latent_width * width_stride,
            temporal_compression=temporal_stride,
            spatial_compression=height_stride,
            device=geometry.device,
            dtype=geometry.dtype,
        )

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "channels",
            "latent_frames",
            "latent_height",
            "latent_width",
            "output_frames",
            "output_height",
            "output_width",
            "temporal_compression",
            "spatial_compression",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        expected_frames = (self.output_frames - 1) // self.temporal_compression + 1
        if expected_frames != self.latent_frames:
            raise ValueError("output/latent frame geometry is inconsistent")
        if self.output_height // self.spatial_compression != self.latent_height:
            raise ValueError("output/latent height geometry is inconsistent")
        if self.output_width // self.spatial_compression != self.latent_width:
            raise ValueError("output/latent width geometry is inconsistent")


@dataclass(frozen=True, slots=True)
class ConditionInitialization:
    latents: Any
    state: object
    condition_payloads: tuple[ConditionPayload, ...] = ()

    def __post_init__(self) -> None:
        if type(self.condition_payloads) is not tuple:
            raise TypeError("condition_payloads must be a tuple")
        if any(
            not isinstance(payload, ConditionPayload)
            for payload in self.condition_payloads
        ):
            raise TypeError(
                "condition_payloads must contain only ConditionPayload values"
            )
        for payload in self.condition_payloads:
            payload.validate()


class LatentConditioner(ABC):
    """Prepare payload once and apply deterministic initialization/step hooks."""

    INTERFACE_VERSION = "1.0"

    @classmethod
    @abstractmethod
    def describe(cls, config: object) -> object:
        """Return the import-safe static contract for ``config``."""

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> LatentConditioner:
        """Construct only after static/environment/artifact gates pass."""

        raise NotImplementedError

    def bind_model_geometry(self, geometry: object) -> LatentSpec:
        """Own the projection from generic model geometry to conditioner input."""

        return LatentSpec.from_model_geometry(geometry)

    @abstractmethod
    def prepare(
        self,
        prompts: tuple[str, ...],
        latent_spec: LatentSpec,
        *,
        generator: Any,
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def initialize_latents(
        self,
        base_latents: Any,
        state: object,
        *,
        generator: Any,
    ) -> ConditionInitialization:
        raise NotImplementedError

    @abstractmethod
    def after_step(
        self,
        step_index: int,
        timestep: Any,
        next_latents: Any,
        state: object,
    ) -> Any:
        raise NotImplementedError
