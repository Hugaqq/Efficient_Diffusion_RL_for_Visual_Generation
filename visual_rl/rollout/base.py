"""Rollout engine interface and shared batch finalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from typing import Any

from visual_rl.core.types import RolloutBatch, StepContext
from visual_rl.model_adapters.base import ModelAdapter


class RolloutEngine(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def sample(
        self,
        adapter: ModelAdapter,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        context: StepContext | None = None,
    ) -> RolloutBatch:
        raise NotImplementedError

    def resolve_context(self, context: StepContext | None) -> StepContext:
        """Resolve legacy config-carried runtime values without mutating config."""

        if context is not None:
            return context
        epoch_tag = int(self.config.get("epoch_tag") or 0)
        return StepContext(
            step=int(self.config.get("step", epoch_tag)),
            seed=int(self.config.get("seed") or 0),
            epoch_tag=epoch_tag,
            rank=int(self.config.get("rank") or 0),
            world_size=int(self.config.get("world_size") or 1),
            policy_version=int(self.config.get("policy_version") or 0),
        )

    def runtime_config(
        self,
        context: StepContext,
        **updates: Any,
    ) -> dict[str, Any]:
        """Build an isolated config for adapters that still accept a dict."""

        runtime = deepcopy(self.config)
        runtime.update(asdict(context))
        runtime.update(updates)
        return runtime

    def finalize_batch(
        self,
        batch: RolloutBatch,
        context: StepContext,
        *,
        media_type: str | None = None,
    ) -> RolloutBatch:
        """Attach canonical identity/layout/context and validate one rollout."""

        batch_size = batch.batch_size
        prompt_ids = [
            str(item.get("prompt_id") or batch.prompt_id[index])
            for index, item in enumerate(batch.metadata)
        ]
        group_ids = [
            str(item.get("group_id") or batch.group_id[index])
            for index, item in enumerate(batch.metadata)
        ]
        current_branch_ids = _as_list(batch.branch_id, batch_size)
        branch_ids = [
            item.get(
                "branch_id",
                item.get("sample_index", current_branch_ids[index]),
            )
            for index, item in enumerate(batch.metadata)
        ]
        sample_ids = [
            _sample_id(
                context=context,
                prompt_id=prompt_ids[index],
                group_id=group_ids[index],
                branch_id=branch_ids[index],
                row=index,
            )
            for index in range(batch_size)
        ]
        media_layout = batch.media_layout or {
            "image": "BCHW",
            "video": "BFCHW",
        }.get(str(media_type).lower())
        finalized = batch.replace(
            sample_id=sample_ids,
            prompt_id=prompt_ids,
            group_id=group_ids,
            branch_id=branch_ids,
            media_layout=media_layout,
            context=context,
        )
        finalized.validate_lightweight(strict=True)
        return finalized


def _as_list(value: Any, expected: int) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    result = list(value)
    if len(result) != expected:
        raise ValueError(
            f"branch_id length must be {expected}, got {len(result)}"
        )
    return result


def _sample_id(
    *,
    context: StepContext,
    prompt_id: str,
    group_id: str,
    branch_id: Any,
    row: int,
) -> str:
    payload = json.dumps(
        {
            "step": context.step,
            "seed": context.seed,
            "rank": context.rank,
            "policy_version": context.policy_version,
            "prompt_id": prompt_id,
            "group_id": group_id,
            "branch_id": branch_id,
            "row": row,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"step-{context.step:06d}-rank-{context.rank:04d}-{digest}"
