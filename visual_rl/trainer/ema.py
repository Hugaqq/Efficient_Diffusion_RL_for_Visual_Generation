"""Small EMA wrapper used by future real trainers."""

from __future__ import annotations


class EMA:
    def __init__(self, parameters, decay: float = 0.999):
        self.decay = decay
        self.shadow = [param.detach().clone() for param in parameters]

    def update(self, parameters) -> None:
        for shadow, param in zip(self.shadow, parameters, strict=True):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

