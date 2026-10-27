"""Exact independent AV sigma grids, separate from the Gaussian dynamics ABI.

Semantic reference: UniRL 8cfbac17765004afb68fef08a4e011a4c8daf555,
minimax_h3/diffusion.py and vendor/scheduling_minimax_h3.py (Apache-2.0).
Independently implemented: N evaluations mean N+1 points; FP32 collisions
are rejected rather than deduplicated. Sigma is NOT model time (1-sigma).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, init=False)
class H3ScheduleSnapshot:
    """Immutable FP32 values; tensor reads are fresh CPU copies, never aliases.

    Explicit grids must already be CPU float32 tensors: no silent rounding,
    reshaping, device transfer, shifting or endpoint repair on restoration.
    """

    _video: tuple[float, ...]
    _audio: tuple[float, ...]

    def __init__(self, video_sigmas: torch.Tensor, audio_sigmas: torch.Tensor) -> None:
        import torch

        for name, grid in (("video", video_sigmas), ("audio", audio_sigmas)):
            if not isinstance(grid, torch.Tensor):
                raise TypeError(f"{name} sigmas must be a tensor")
            if grid.dtype != torch.float32 or grid.device.type != "cpu":
                raise ValueError(f"{name} sigmas must be CPU float32")
            if grid.ndim != 1 or grid.numel() < 2:
                raise ValueError(f"{name} sigmas must be 1-D with at least two points")
            if not bool(torch.isfinite(grid).all()) or not bool(((grid >= 0) & (grid <= 1)).all()):
                raise ValueError(f"{name} sigmas must be finite and in [0,1]")
            if grid[0].item() != 1.0 or grid[-1].item() != 0.0:
                raise ValueError(f"{name} sigmas must start at 1 and end at 0")
            if not bool((grid[:-1] > grid[1:]).all()):
                raise ValueError(f"{name} sigmas must be strictly decreasing")
        if video_sigmas.shape != audio_sigmas.shape:
            raise ValueError("video and audio grids must have equal length")
        object.__setattr__(self, "_video", tuple(video_sigmas.detach().tolist()))
        object.__setattr__(self, "_audio", tuple(audio_sigmas.detach().tolist()))

    @classmethod
    def build(
        cls, num_steps: int, *, video_shift: float = 12.0, audio_shift: float = 3.0
    ) -> H3ScheduleSnapshot:
        import torch

        if type(num_steps) is not int or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        for shift in (video_shift, audio_shift):
            if type(shift) not in (int, float) or not math.isfinite(shift) or shift <= 0:
                raise ValueError("shifts must be finite positive numbers")
        base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32, device="cpu")
        video = video_shift * base / (1 + (video_shift - 1) * base)
        audio = audio_shift * base / (1 + (audio_shift - 1) * base)
        return cls(video, audio)

    @property
    def num_steps(self) -> int:
        return len(self._video) - 1

    @property
    def video_sigmas(self) -> torch.Tensor:
        import torch

        return torch.tensor(self._video, dtype=torch.float32, device="cpu")

    @property
    def audio_sigmas(self) -> torch.Tensor:
        import torch

        return torch.tensor(self._audio, dtype=torch.float32, device="cpu")

    def transition(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scalar FP32 (video_s, video_u, audio_s, audio_u), no cursor."""
        import torch

        if type(index) is not int:
            raise TypeError("transition index must be an integer, not sigma or model time")
        if not 0 <= index < self.num_steps:
            raise IndexError("transition index outside schedule")
        values = (self._video[index], self._video[index + 1], self._audio[index], self._audio[index + 1])
        return tuple(torch.tensor(value, dtype=torch.float32, device="cpu") for value in values)
