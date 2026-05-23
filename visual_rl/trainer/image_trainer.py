"""Image diffusion trainer entry point."""

from __future__ import annotations

from visual_rl.configs.schema import VisualRLConfig
from visual_rl.trainer.trainer import VisualRLTrainer


class ImageRLTrainer(VisualRLTrainer):
    """VisualRL trainer with strict rollout validation enabled for image models."""

    def __init__(self, config: VisualRLConfig):
        config.trainer.setdefault("strict_rollout_validation", True)
        super().__init__(config)
        media_family = config.model.model_family.lower()
        if media_family not in {"image", "sd", "sd15", "sd3", "flux", "qwenimage"}:
            raise ValueError(f"ImageRLTrainer expects an image model_family, got {config.model.model_family!r}")
