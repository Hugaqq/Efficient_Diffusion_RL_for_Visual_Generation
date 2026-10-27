"""Narrow, padless H3 T2VA geometry; trajectory packing stays in H3JointLayout.

Independent implementation of the mathematical geometry described in UniRL
8cfbac17765004afb68fef08a4e011a4c8daf555, minimax_h3/{config,packing}.py
and vendor/packing.py (Apache-2.0, Tencent / MiniMax and HuggingFace).
No reference imports or copied vendor implementation; no conditioning rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from visual_rl.models.h3_joint import H3JointLayout

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class H3T2VAGeometry:
    height: int
    width: int
    num_frames: int
    fps: int = 24
    task: str = "t2va"
    guidance: float = 1.0

    def __post_init__(self) -> None:
        for name in ("height", "width", "num_frames"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.height % 32 or self.width % 32:
            raise ValueError("canvas axes must be multiples of 32")
        if not .25 <= self.width / self.height <= 4 or self.height * self.width > 768 * 1344:
            raise ValueError("canvas exceeds H3 aspect ratio or area limits")
        if type(self.fps) is not int or self.fps != 24:
            raise ValueError("H3 requires 24 fps")
        if self.num_frames % 17 != 5:
            raise ValueError("num_frames must be 17n+5; no implicit alignment")
        if not 5 <= self.num_frames / 24 <= 15:
            raise ValueError("H3 duration must be 5..15 seconds")
        if self.task != "t2va":
            raise ValueError("only t2va is supported")
        if type(self.guidance) not in (int, float) or self.guidance != 1:
            raise ValueError("only guidance=1 is supported")

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """Unpatchified video C,T,H,W; patch is fixed at (1,2,2)."""
        return 24, (self.num_frames - 5) // 17 * 5 + 2, self.height // 16, self.width // 16

    @property
    def audio_latents(self) -> int:
        return round(self.num_frames / 24 * 40)

    @property
    def joint_layout(self) -> H3JointLayout:
        _, frames, height, width = self.latent_shape
        return H3JointLayout(frames * (height // 2) * (width // 2), 96, 2 * self.audio_latents, 32)

    def attention_inputs(self, num_text_tokens: int, *, device: torch.device) -> dict[str, torch.Tensor]:
        """Fresh owned tensors in [text|audio|video] order, not joint-state order.

        Build FP64 coordinates on CPU, then transfer: identical rotary geometry
        across devices. Text must already be trimmed; no mask/padding inference.
        """
        import torch

        if type(num_text_tokens) is not int or num_text_tokens < 1:
            raise ValueError("num_text_tokens must be a positive integer")
        _, frames, height, width = self.latent_shape
        layout = self.joint_layout
        audio_start = num_text_tokens
        video_start = audio_start + layout.audio_rows
        length = video_start + layout.video_rows
        text = torch.arange(audio_start)
        audio = torch.arange(audio_start, video_start)
        video = torch.arange(video_start, length)
        tags = torch.cat((torch.ones(audio_start, dtype=torch.long),
                          torch.full((layout.audio_rows,), 2, dtype=torch.long),
                          torch.zeros(layout.video_rows, dtype=torch.long)))
        positions = torch.zeros(length, 3, dtype=torch.float64)
        positions[text, 0] = text.double()

        axes = []
        for dim in (height, width):
            ratio = dim / math.sqrt(height * width)
            left = (1 - ratio) / 2
            # endpoint=False arithmetic, including stop-start rounding.
            step = ((left + ratio) - left) / (dim // 2)
            axes.append((left + torch.arange(dim // 2, dtype=torch.float64) * step) * 32)
        h, w = axes
        positions[audio, 0] = (num_text_tokens + torch.arange(self.audio_latents, dtype=torch.float64)).repeat(2)
        positions[audio[:self.audio_latents], 2] = w[0]
        positions[audio[self.audio_latents:], 2] = w[-1]

        spans = torch.tensor([(5 / 3) * (1 if i % 5 == 0 else 4) for i in range(frames - 1)],
                             dtype=torch.float64)
        times = num_text_tokens + torch.cat((torch.zeros(1, dtype=torch.float64), spans.cumsum(0)))
        positions[video, 0] = times.repeat_interleave((height // 2) * (width // 2))
        positions[video, 1] = h.repeat_interleave(width // 2).repeat(frames)
        positions[video, 2] = w.repeat(frames * (height // 2))
        return {name: value.to(device) for name, value in dict(
            position_ids=positions, token_tags=tags, text_indices=text,
            audio_indices=audio, video_indices=video,
        ).items()}
