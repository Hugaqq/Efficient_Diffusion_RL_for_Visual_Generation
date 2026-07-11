"""Feedback provider interface."""

from abc import ABC, abstractmethod

from visual_rl.core.types import RolloutBatch, RewardBatch


class FeedbackProvider(ABC):
    @abstractmethod
    def score(self, batch: RolloutBatch) -> RewardBatch:
        raise NotImplementedError
