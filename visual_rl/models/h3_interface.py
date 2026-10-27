"""Closed B1 H3 prompt, conditioning, generator and independent-sigma ports."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from visual_rl.core.contracts import LatentLayout
from visual_rl.models.interface import ModelInput, ModelPortError

H3_GUIDANCE_ID = "h3.guidance1.v1"
H3_INITIAL_STATE_VERSION = "h3.av-normal-fp32-to-fp16.v1"


@dataclass(frozen=True)
class H3PromptInput:
    prompt: str

    def __post_init__(self):
        if not isinstance(self.prompt, str):
            raise TypeError("H3PromptInput requires an exact prompt string")


@dataclass(frozen=True, init=False)
class H3Conditioning:
    prompt: str
    cache_identity: str
    _embeddings: Any = field(repr=False)

    def __init__(self, prompt: str, cache_identity: str, embeddings):
        import torch

        H3PromptInput(prompt)
        if (not isinstance(cache_identity, str) or len(cache_identity) != 64
                or any(c not in "0123456789abcdef" for c in cache_identity)):
            raise ValueError("H3 conditioning requires full cache SHA256")
        if (not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3
                or embeddings.shape[0] != 1 or embeddings.shape[1] < 1 or embeddings.shape[2] != 5120
                or embeddings.device.type != "cpu" or embeddings.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or embeddings.requires_grad or embeddings.grad_fn is not None or not bool(torch.isfinite(embeddings).all())):
            raise ModelPortError("H3 conditioning requires detached finite CPU [1,L,5120] embeddings")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "cache_identity", cache_identity)
        object.__setattr__(self, "_embeddings", embeddings.clone())

    @property
    def embeddings(self):
        return self._embeddings.clone()

    @property
    def condition_identity(self) -> tuple[str, ...]:
        key = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        return (f"h3-text:{self.cache_identity}:{key}",)


@dataclass(frozen=True)
class H3Generators:
    video: Any
    audio: Any

    def __post_init__(self):
        import torch

        if not isinstance(self.video, torch.Generator) or not isinstance(self.audio, torch.Generator):
            raise TypeError("H3 requires explicit video/audio generators")
        if self.video is self.audio:
            raise ValueError("H3 initial state requires independent modality generator objects")


@dataclass(frozen=True)
class H3ModelInput(ModelInput):
    """Here timestep explicitly means video_sigma, NEVER schedule index or t.

    Both sigmas must be scalar FP32 on the latent device. Algorithm/runtime
    resolves schedules; this model-domain type owns no schedule or CPS logic.
    """
    audio_sigma: Any

    def __post_init__(self):
        import torch

        super().__post_init__()
        if (self.latent_spec.layout is not LatentLayout.JOINT_AV_FLAT or self.latents.shape[0] != 1
                or self.latents.dtype != torch.float16 or self.latents.device.type not in ("cpu", "cuda")):
            raise ModelPortError("H3 input requires B1 FP16 joint_av_flat on CPU/CUDA")
        for name, sigma in (("video_sigma", self.timestep), ("audio_sigma", self.audio_sigma)):
            if (not isinstance(sigma, torch.Tensor) or sigma.shape != () or sigma.dtype != torch.float32
                    or sigma.device != self.latents.device or sigma.requires_grad or sigma.grad_fn is not None
                    or not bool(torch.isfinite(sigma)) or not 0 <= sigma.item() <= 1):
                raise ModelPortError(f"{name} requires a detached FP32 scalar in [0,1] on latent device")
        if not isinstance(self.conditioning, H3Conditioning) or self.condition_identity != self.conditioning.condition_identity:
            raise ModelPortError("H3 conditioning/identity mismatch")
        if self.guidance is not None or self.guidance_identity != (H3_GUIDANCE_ID,):
            raise ModelPortError("H3 supports only guidance1 without negative conditioning")
