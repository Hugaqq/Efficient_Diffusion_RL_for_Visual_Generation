"""One caller-owned prepared H3 forward, with no algorithm or loading policy.

Signature and data-ward convention checked against UniRL
8cfbac17765004afb68fef08a4e011a4c8daf555 minimax_h3/diffusion.py and
vendor/transformer_minimax_h3.py (Apache-2.0). Independently implemented.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry

if TYPE_CHECKING:
    import torch


def h3_t2va_forward(
    prepared_forward: Callable, *, x: torch.Tensor, geometry: H3T2VAGeometry,
    text_embeddings: torch.Tensor, video_sigma: torch.Tensor, audio_sigma: torch.Tensor,
) -> torch.Tensor:
    """Return noise-ward joint prediction, preserving both output graphs.

    The caller supplies its sole prepared/guarded callable, and owns grad mode.
    No unwrapping, parameter casting, autocast, CFG, or scheduler lookup here.
    State is FP16 B1; projection dtype alignment belongs to the H3 model.
    """
    import torch

    if not callable(prepared_forward):
        raise TypeError("prepared_forward must be callable")
    if not isinstance(geometry, H3T2VAGeometry):
        raise TypeError("geometry must be H3T2VAGeometry")
    layout = geometry.joint_layout
    video, audio = layout.unpack(x)
    if x.shape[0] != 1 or x.dtype != torch.float16:
        raise ValueError("x must be FP16 microbatch B1")
    if x.device.type not in ("cpu", "cuda") or not bool(torch.isfinite(x).all()):
        raise ValueError("x must be finite on CPU or CUDA")
    if not isinstance(text_embeddings, torch.Tensor):
        raise TypeError("text_embeddings must be a tensor")
    if (text_embeddings.ndim != 3 or text_embeddings.shape[0] != 1
            or min(text_embeddings.shape[1:]) < 1):
        raise ValueError("trimmed text_embeddings must have shape [1,L,D], L,D > 0")
    if (text_embeddings.device != x.device or not text_embeddings.is_floating_point()
            or not bool(torch.isfinite(text_embeddings).all())):
        raise ValueError("text_embeddings must be finite floating point on x device")
    for name, sigma in (("video_sigma", video_sigma), ("audio_sigma", audio_sigma)):
        if not isinstance(sigma, torch.Tensor):
            raise TypeError(f"{name} must be an explicit FP32 scalar tensor")
        if (sigma.shape != () or sigma.dtype != torch.float32 or sigma.requires_grad
                or sigma.device.type not in ("cpu", "cuda")):
            raise ValueError(f"{name} must be a detached FP32 scalar on CPU or CUDA")
        if not bool(torch.isfinite(sigma)) or not 0 <= sigma.item() <= 1:
            raise ValueError(f"{name} must be finite in [0,1]")

    attention = geometry.attention_inputs(text_embeddings.shape[1], device=x.device)
    rows = torch.full(attention["token_tags"].shape, 1 - video_sigma.item(),
                      dtype=torch.float32, device=x.device)
    rows[attention["audio_indices"]] = 1 - audio_sigma.item()
    timestep, indices = torch.unique(rows, sorted=True, return_inverse=True)
    outputs = prepared_forward(
        hidden_states=video, audio_hidden_states=audio,
        encoder_hidden_states=text_embeddings, timestep=timestep,
        timestep_indices=indices, **attention, return_dict=False,
    )
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise ValueError("H3 forward must return (video_velocity, audio_velocity)")
    for name, output, state in zip(("video", "audio"), outputs, (video, audio)):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{name} velocity must be a tensor")
        if output.shape != state.shape or output.device != x.device:
            raise ValueError(f"{name} velocity shape/device must match generated state")
        if not output.is_floating_point() or not bool(torch.isfinite(output).all()):
            raise ValueError(f"{name} velocity must be finite floating point")
    # H3JointLayout also rejects accidental mixed AV output dtypes.
    return layout.pack(-outputs[0], -outputs[1])
