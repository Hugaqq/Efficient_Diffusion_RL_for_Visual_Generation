"""Feedback provider interface."""

from abc import ABC, abstractmethod

from visual_rl.core.types import RolloutBatch, RewardBatch


class FeedbackProvider(ABC):
    """Score rollout batches with conservative in-process capabilities.

    Subclasses may opt into concurrent calls only after auditing all shared
    clients, sessions, and caches. Executor-level retries require both an
    idempotency declaration and an explicit maximum number of attempts made by
    one ``score`` call.
    """

    supports_concurrent_score = False
    executor_retry_safe = False
    requires_hard_timeout = False
    max_attempts_per_score: int

    @abstractmethod
    def score(self, batch: RolloutBatch) -> RewardBatch:
        raise NotImplementedError
