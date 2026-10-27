"""Generated T2VA trajectory storage, NOT H3's attention-token layout.

Semantic reference: UniRL 8cfbac17765004afb68fef08a4e011a4c8daf555,
minimax_h3/diffusion.py (video-first dual streams), Apache-2.0.
This narrow shape contract is independently implemented; no vendor dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class H3JointLayout:
    """Only generated rows: no text, conditioning rows, padding or masks."""

    video_rows: int
    video_channels: int
    audio_rows: int
    audio_channels: int

    def __post_init__(self) -> None:
        for name in ("video_rows", "video_channels", "audio_rows", "audio_channels"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def video_numel(self) -> int:
        return self.video_rows * self.video_channels

    @property
    def audio_numel(self) -> int:
        return self.audio_rows * self.audio_channels

    @property
    def packed_width(self) -> int:
        return self.video_numel + self.audio_numel

    def pack(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Pack [B,V,Cv] and [B,A,Ca] without casting or detaching either."""
        import torch

        if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
            raise TypeError("video and audio must be tensors")
        if video.ndim != 3 or video.shape[0] < 1 or tuple(video.shape[1:]) != (
            self.video_rows, self.video_channels
        ):
            raise ValueError("video must have shape [B,video_rows,video_channels], B > 0")
        if tuple(audio.shape) != (video.shape[0], self.audio_rows, self.audio_channels):
            raise ValueError("audio must have matching B and shape [B,audio_rows,audio_channels]")
        if video.dtype != audio.dtype or video.device != audio.device:
            raise ValueError("video and audio must have identical dtype and device")
        return torch.cat((video.flatten(1), audio.flatten(1)), dim=1)

    def unpack(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two differentiable views; their storage aliases packed."""
        import torch

        if not isinstance(packed, torch.Tensor):
            raise TypeError("packed must be a tensor")
        if packed.ndim != 2 or packed.shape[0] < 1 or packed.shape[1] != self.packed_width:
            raise ValueError("packed must have shape [B,packed_width], B > 0")
        batch = packed.shape[0]
        return (
            packed[:, :self.video_numel].reshape(batch, self.video_rows, self.video_channels),
            packed[:, self.video_numel:].reshape(batch, self.audio_rows, self.audio_channels),
        )
