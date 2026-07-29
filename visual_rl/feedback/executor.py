"""The single synchronous reward execution facade."""

from __future__ import annotations

from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.feedback.provider import RewardFeedbackProvider, RewardShard

__all__ = ["RewardExecutionError", "RewardExecutor"]


class RewardExecutionError(RuntimeError):
    """A reward shard failed before a complete RewardBatch was produced."""

    def __init__(self, message: str, *, shard_index: int, attempts: int) -> None:
        super().__init__(message)
        self.shard_index = shard_index
        self.attempts = attempts


class RewardExecutor:
    """Serially shard, retry, and assemble exactly one final RewardBatch."""

    def __init__(
        self,
        *,
        provider: RewardFeedbackProvider,
        microbatch_size: int | None,
        max_retries: int,
    ) -> None:
        if not isinstance(provider, RewardFeedbackProvider):
            raise TypeError("provider must be a RewardFeedbackProvider")
        if microbatch_size is not None and (
            type(microbatch_size) is not int or microbatch_size <= 0
        ):
            raise ValueError("microbatch_size must be a positive integer or None")
        if type(max_retries) is not int or not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be an integer in [0, 10]")
        self._provider = provider
        self._microbatch_size = microbatch_size
        self._max_retries = max_retries
        self._closed = False

    def score(
        self,
        batch: RolloutBatch,
        context: StepContext,
    ) -> RewardBatch:
        if self._closed:
            raise RewardExecutionError(
                "RewardExecutor is closed",
                shard_index=-1,
                attempts=0,
            )
        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if not isinstance(context, StepContext):
            raise TypeError("context must be a StepContext")
        if batch.context is not context:
            raise ValueError("batch.context must be the identical StepContext")
        if batch.batch_size < 1:
            raise ValueError("reward execution rejects an empty batch")

        shard_size = self._microbatch_size or batch.batch_size
        shards: list[RewardShard] = []
        for shard_index, start in enumerate(
            range(0, batch.batch_size, shard_size)
        ):
            stop = min(start + shard_size, batch.batch_size)
            shard_batch = batch.slice(tuple(range(start, stop)))
            shard = self._score_with_retry(
                shard_batch,
                context,
                shard_index=shard_index,
            )
            expected = batch.sample_id[start:stop]
            if shard.sample_id != expected:
                raise RewardExecutionError(
                    f"reward shard {shard_index} did not cover its contiguous "
                    "sample_id slice",
                    shard_index=shard_index,
                    attempts=1,
                )
            shards.append(shard)

        result = _assemble_reward_batch(batch, tuple(shards))
        result.validate_against(batch)
        return result

    def close(self) -> None:
        """Idempotently close only this facade, never the injected Provider."""

        self._closed = True

    def _score_with_retry(
        self,
        shard: RolloutBatch,
        context: StepContext,
        *,
        shard_index: int,
    ) -> RewardShard:
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                result = self._provider.score(shard, context)
                if not isinstance(result, RewardShard):
                    raise TypeError(
                        "RewardFeedbackProvider.score() must return RewardShard"
                    )
                return result
            except Exception as exc:
                if attempt == attempts:
                    raise RewardExecutionError(
                        f"reward shard {shard_index} failed after {attempt} "
                        f"attempt(s): {type(exc).__name__}: {exc}",
                        shard_index=shard_index,
                        attempts=attempt,
                    ) from exc
        raise AssertionError("unreachable retry state")


def _assemble_reward_batch(
    batch: RolloutBatch,
    shards: tuple[RewardShard, ...],
) -> RewardBatch:
    if not shards:
        raise ValueError("reward execution produced no shards")
    import torch

    names = tuple(shards[0].raw)
    if not names:
        raise ValueError("reward shards contain no reward components")
    sample_id: list[str] = []
    raw_parts: dict[str, list] = {name: [] for name in names}
    weighted_parts: dict[str, list] = {name: [] for name in names}
    total_parts = []
    valid_parts = []
    shared = {name: shards[0].shared_metadata[name] for name in names}
    samples: dict[str, list] = {name: [] for name in names}

    for shard in shards:
        if tuple(shard.raw) != names or tuple(shard.weighted) != names:
            raise ValueError("reward shard component order is inconsistent")
        if any(
            shard.shared_metadata[name] != shared[name] for name in names
        ):
            raise ValueError("reward shard shared_metadata is inconsistent")
        sample_id.extend(shard.sample_id)
        for name in names:
            raw_parts[name].append(shard.raw[name])
            weighted_parts[name].append(shard.weighted[name])
            samples[name].extend(shard.sample_metadata[name])
        total_parts.append(shard.weighted_total)
        valid_parts.append(shard.valid_mask)

    final_sample_id = tuple(sample_id)
    if final_sample_id != batch.sample_id:
        raise ValueError(
            "reward shards have missing, duplicate, overlapping, or reordered rows"
        )
    return RewardBatch(
        sample_id=final_sample_id,
        raw={
            name: torch.cat(tuple(raw_parts[name]), dim=0).contiguous()
            for name in names
        },
        weighted={
            name: torch.cat(tuple(weighted_parts[name]), dim=0).contiguous()
            for name in names
        },
        weighted_total=torch.cat(tuple(total_parts), dim=0).contiguous(),
        valid_mask=torch.cat(tuple(valid_parts), dim=0).contiguous(),
        shared_metadata=shared,
        sample_metadata={
            name: tuple(samples[name])
            for name in names
        },
    )
