"""Cheap image rewards for local visual RL tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from visual_rl.core.registry import REWARD_CLIENTS


COLOR_TO_INDEX = {"red": 0, "green": 1, "blue": 2}


@dataclass
class PromptColorRewardClient:
    name: str = "prompt_color"
    default_color: str = "red"

    def score(self, media: Any, prompts: list[str], metadata: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        images = self._to_numpy_images(media)
        scores = []
        targets = []
        for index, prompt in enumerate(prompts):
            color = str(metadata[index].get("target_color") or self._color_from_prompt(prompt))
            target = COLOR_TO_INDEX.get(color, COLOR_TO_INDEX[self.default_color])
            channel_means = images[index].reshape(3, -1).mean(axis=1)
            target_score = channel_means[target]
            distractor_score = np.delete(channel_means, target).mean()
            scores.append(float(np.clip(0.5 + target_score - distractor_score, 0.0, 1.0)))
            targets.append(color)
        return np.asarray(scores, dtype=np.float32), {"targets": targets}

    def _color_from_prompt(self, prompt: str) -> str:
        lower = prompt.lower()
        for color in COLOR_TO_INDEX:
            if color in lower:
                return color
        return self.default_color

    @staticmethod
    def _to_numpy_images(media: Any) -> np.ndarray:
        try:
            import torch

            if isinstance(media, torch.Tensor):
                media = media.detach().cpu().float().numpy()
        except Exception:  # noqa: BLE001 - numpy fallback below reports shape issues
            pass
        images = np.asarray(media, dtype=np.float32)
        if images.ndim == 5:
            images = images[:, -1]
        if images.ndim != 4:
            raise ValueError(f"prompt_color expects [B, C, H, W] or [B, T, C, H, W], got {images.shape}")
        if images.shape[1] == 3:
            return images
        if images.shape[-1] == 3:
            return np.moveaxis(images, -1, 1)
        raise ValueError(f"prompt_color cannot locate RGB channel in shape {images.shape}")


@dataclass
class PromptColorMarginRewardClient(PromptColorRewardClient):
    """Unclipped RGB margin for branch-relative optimization."""

    name: str = "prompt_color_margin"

    def score(
        self,
        media: Any,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        images = self._to_numpy_images(media)
        scores = []
        targets = []
        for index, prompt in enumerate(prompts):
            color = str(
                metadata[index].get("target_color")
                or self._color_from_prompt(prompt)
            )
            target = COLOR_TO_INDEX.get(color, COLOR_TO_INDEX[self.default_color])
            channel_means = images[index].reshape(3, -1).mean(axis=1)
            target_score = channel_means[target]
            distractor_score = np.delete(channel_means, target).mean()
            scores.append(float(target_score - distractor_score))
            targets.append(color)
        return np.asarray(scores, dtype=np.float32), {
            "targets": targets,
            "score_kind": "unclipped_target_channel_margin",
        }


REWARD_CLIENTS.register("prompt_color", PromptColorRewardClient)
REWARD_CLIENTS.register("prompt_color_margin", PromptColorMarginRewardClient)
