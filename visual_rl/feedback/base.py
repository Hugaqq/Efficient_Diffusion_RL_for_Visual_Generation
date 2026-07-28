"""Feedback provider and reward client interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from visual_rl.core.types import FrozenMapping, RewardBatch, RolloutBatch, StepContext


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


class RewardClient(ABC):
    """Base contract for one builtin reward component (v0.7 direction).

    Frozen by the master plan (stage 2.1/2.2); wired up by the atomic
    cutover. A ``RewardClient`` is not a second feedback provider: its only
    base entry is the synchronous ``score(batch, context) -> RewardVector``
    below, where ``RewardVector`` is the provider-internal frozen shard
    result. The fixed ``RewardFeedbackProvider`` weights one or more
    ``RewardVector`` into an internal shard and only
    ``RewardExecutor.score()`` merges shards into the cross-component
    ``RewardBatch``.

    Base inputs are limited to ``batch.media``/``prompts``/``metadata``/
    ``sample_id``/``context``; only a client whose static requirements
    include ``conditioning.camera`` may additionally read
    ``batch.camera_trajectory``. Latents, log-probs and the
    recompute/artifact payloads are never reward inputs. Reward forward runs
    inside an inference/no-grad boundary and every returned tensor is
    detached.

    Declared non-abstract so the existing concrete clients keep working
    until the cutover moves them onto this base.
    """

    name: str

    def score(self, batch: RolloutBatch, context: StepContext):
        """Score one batch and return a frozen ``RewardVector``."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement the v0.7 "
            "score(batch, context) contract yet"
        )

    # ------------------------------------------------------------------
    # Unified component factory protocol (plan stage 2.2); see
    # ``ModelAdapter`` for the shared contract.
    # ------------------------------------------------------------------

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        """Whitelist/default/validate/canonicalize component params."""

        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__}.resolve_params() requires a mapping, "
                f"got {type(raw).__name__}"
            )
        return FrozenMapping(raw)

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, Any],
        context: Any,
    ) -> tuple:
        """Bounded, read-only environment checks; default is no checks."""

        return ()

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, Any],
        context: Any,
    ):
        """Construct the runtime component from resolved params."""

        raise NotImplementedError(
            f"{cls.__name__} does not implement from_config() yet"
        )

    @classmethod
    def required_capabilities(cls, resolved_params: Mapping[str, Any]) -> frozenset:
        """Conditional capabilities implied by the component's own params."""

        return frozenset()
