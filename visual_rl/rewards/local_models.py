"""Reward model offload helpers inspired by GenRL."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


class RewardModelOffloader:
    """Move reward models onto a device only for scoring, then back to CPU.

    v0.2 provides the contract; concrete local reward models can register
    `to(device)` capable objects here in later phases.
    """

    def __init__(self, models: dict[str, Any] | None = None):
        self.models = models or {}

    @contextmanager
    def on_device(self, device: str):
        moved = []
        for model in self.models.values():
            if hasattr(model, "to"):
                model.to(device)
                moved.append(model)
        try:
            yield
        finally:
            for model in moved:
                model.to("cpu")
            try:
                import torch

                if str(device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
