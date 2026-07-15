"""Cheap image rewards for local visual RL tests."""

from __future__ import annotations

from dataclasses import dataclass
import re
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
        tokens = set(re.findall(r"[a-z]+", prompt.lower()))
        matches = [color for color in COLOR_TO_INDEX if color in tokens]
        if len(matches) == 1:
            return matches[0]
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


@dataclass
class PromptColorGuardedRewardClient(PromptColorRewardClient):
    """Bounded color reward with cheap anti-saturation image guardrails.

    This remains a lightweight diagnostic reward rather than a semantic image
    quality model.  Its purpose is to make the RGB-control experiment harder to
    solve by globally saturating, whitening, darkening, or flattening an image.
    """

    name: str = "prompt_color_guarded"
    margin_clip: float = 0.35
    saturation_max: float = 0.60
    luminance_min: float = 0.12
    luminance_max: float = 0.88
    spatial_std_min: float = 0.08
    spatial_std_max: float = 0.38
    saturation_penalty_weight: float = 2.0
    luminance_penalty_weight: float = 1.0
    spatial_penalty_weight: float = 1.0

    def score(
        self,
        media: Any,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        images = np.clip(self._to_numpy_images(media), 0.0, 1.0)
        scores: list[float] = []
        targets: list[str] = []
        details: list[dict[str, float]] = []
        for index, prompt in enumerate(prompts):
            color = str(
                metadata[index].get("target_color")
                or self._color_from_prompt(prompt)
            )
            target = COLOR_TO_INDEX.get(color, COLOR_TO_INDEX[self.default_color])
            image = images[index]
            channel_means = image.reshape(3, -1).mean(axis=1)
            raw_margin = float(
                channel_means[target] - np.delete(channel_means, target).mean()
            )
            bounded_margin = float(
                np.clip(raw_margin, -self.margin_clip, self.margin_clip)
            )
            maximum = image.max(axis=0)
            minimum = image.min(axis=0)
            saturation = float(
                np.mean(np.where(maximum > 1e-6, (maximum - minimum) / maximum, 0.0))
            )
            luminance = float(
                np.mean(
                    0.2126 * image[0]
                    + 0.7152 * image[1]
                    + 0.0722 * image[2]
                )
            )
            spatial_std = float(np.std(image, axis=(1, 2)).mean())
            saturation_penalty = max(0.0, saturation - self.saturation_max)
            luminance_penalty = max(
                0.0,
                self.luminance_min - luminance,
                luminance - self.luminance_max,
            )
            spatial_penalty = max(
                0.0,
                self.spatial_std_min - spatial_std,
                spatial_std - self.spatial_std_max,
            )
            score = (
                bounded_margin
                - self.saturation_penalty_weight * saturation_penalty
                - self.luminance_penalty_weight * luminance_penalty
                - self.spatial_penalty_weight * spatial_penalty
            )
            scores.append(float(score))
            targets.append(color)
            details.append(
                {
                    "raw_margin": raw_margin,
                    "bounded_margin": bounded_margin,
                    "saturation_mean": saturation,
                    "luminance_mean": luminance,
                    "spatial_std": spatial_std,
                    "penalty_total": float(bounded_margin - score),
                }
            )
        return np.asarray(scores, dtype=np.float32), {
            "targets": targets,
            "score_kind": "bounded_color_margin_with_pixel_guardrails",
            "thresholds": {
                "margin_clip": self.margin_clip,
                "saturation_max": self.saturation_max,
                "luminance_min": self.luminance_min,
                "luminance_max": self.luminance_max,
                "spatial_std_min": self.spatial_std_min,
                "spatial_std_max": self.spatial_std_max,
            },
            "records": details,
        }


REWARD_CLIENTS.register("prompt_color", PromptColorRewardClient)
REWARD_CLIENTS.register("prompt_color_margin", PromptColorMarginRewardClient)
REWARD_CLIENTS.register("prompt_color_guarded", PromptColorGuardedRewardClient)
