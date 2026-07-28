"""Fixed reward aggregation between builtin clients and the executor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from visual_rl.core.types import (
    FrozenMapping,
    RewardVector,
    RolloutBatch,
    StepContext,
)
from visual_rl.feedback.base import RewardClient
from visual_rl.feedback.cache import RewardCache, reward_cache_key

__all__ = ["RewardClientBinding", "RewardFeedbackProvider"]


class _TensorMapping(Mapping[str, Any]):
    __slots__ = ("_items",)

    def __init__(self, source: Mapping[str, Any]) -> None:
        items: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for key, value in source.items():
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError("reward tensor keys must be unique non-empty strings")
            seen.add(key)
            items.append((key, value))
        object.__setattr__(self, "_items", tuple(items))

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("reward tensor mappings are immutable")


@dataclass(frozen=True)
class RewardClientBinding:
    """One canonical selected reward component and its non-owning client."""

    name: str
    client: RewardClient
    weight: float
    resolved_params: FrozenMapping

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("reward binding name must be a non-empty string")
        if not isinstance(self.client, RewardClient):
            raise TypeError("reward binding client must be a RewardClient")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(float(self.weight))
        ):
            raise ValueError("reward binding weight must be finite")
        object.__setattr__(self, "weight", float(self.weight))
        if not isinstance(self.resolved_params, FrozenMapping):
            object.__setattr__(
                self,
                "resolved_params",
                FrozenMapping(self.resolved_params),
            )


@dataclass(frozen=True)
class RewardShard:
    """Provider-internal result; only RewardExecutor may consume it."""

    sample_id: tuple[str, ...]
    raw: Mapping[str, Any]
    weighted: Mapping[str, Any]
    weighted_total: Any
    valid_mask: Any
    shared_metadata: Mapping[str, Mapping[str, Any]]
    sample_metadata: Mapping[str, tuple[Mapping[str, Any], ...]]

    def __post_init__(self) -> None:
        import torch

        if type(self.sample_id) is not tuple or not self.sample_id:
            raise ValueError("RewardShard sample_id must be a non-empty tuple")
        if len(set(self.sample_id)) != len(self.sample_id):
            raise ValueError("RewardShard sample_id entries must be unique")
        size = len(self.sample_id)
        raw = _TensorMapping(self.raw)
        weighted = _TensorMapping(self.weighted)
        if not raw or tuple(raw) != tuple(weighted):
            raise ValueError("RewardShard raw/weighted keys must match in order")
        for group, mapping in (("raw", raw), ("weighted", weighted)):
            for name, value in mapping.items():
                _validate_tensor(f"{group}.{name}", value, size, torch.float32)
        _validate_tensor("weighted_total", self.weighted_total, size, torch.float32)
        _validate_tensor("valid_mask", self.valid_mask, size, torch.bool)
        if not bool(self.valid_mask.all()):
            raise ValueError("RewardShard rows must all be valid")
        total = torch.zeros(size, dtype=torch.float32)
        for value in weighted.values():
            total.add_(value)
        if not torch.allclose(total, self.weighted_total, rtol=1e-6, atol=1e-7):
            raise ValueError("RewardShard weighted_total is inconsistent")

        shared = FrozenMapping(self.shared_metadata)
        samples = FrozenMapping(self.sample_metadata)
        if tuple(shared) != tuple(raw) or tuple(samples) != tuple(raw):
            raise ValueError("RewardShard metadata keys must match reward keys")
        for name in raw:
            rows = samples[name]
            if type(rows) is not tuple or len(rows) != size:
                raise ValueError(
                    f"sample_metadata[{name!r}] must contain one row per sample"
                )
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "weighted", weighted)
        object.__setattr__(self, "shared_metadata", shared)
        object.__setattr__(self, "sample_metadata", samples)


class RewardFeedbackProvider:
    """Call selected clients in canonical order and weight one shard."""

    def __init__(
        self,
        *,
        clients: tuple[RewardClientBinding, ...],
        cache: RewardCache | None,
    ) -> None:
        if type(clients) is not tuple or not clients:
            raise ValueError("RewardFeedbackProvider requires a non-empty client tuple")
        if any(not isinstance(item, RewardClientBinding) for item in clients):
            raise TypeError("clients must contain RewardClientBinding values")
        names = tuple(item.name for item in clients)
        if len(set(names)) != len(names):
            raise ValueError("reward binding names must be unique")
        if not any(item.weight != 0.0 for item in clients):
            raise ValueError("at least one reward binding weight must be non-zero")
        if cache is not None and not isinstance(cache, RewardCache):
            raise TypeError("cache must be a RewardCache or None")
        self._clients = clients
        self._cache = cache
        self._closed = False

    def score(self, batch: RolloutBatch, context: StepContext) -> RewardShard:
        if self._closed:
            raise RuntimeError("RewardFeedbackProvider is closed")
        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if not isinstance(context, StepContext):
            raise TypeError("context must be a StepContext")
        if batch.context is not context:
            raise ValueError("batch.context must be the identical StepContext")

        cache_keys_by_name: dict[str, tuple[str, ...]] = {}
        if self._cache is not None:
            for binding in self._clients:
                cache_keys_by_name[binding.name] = tuple(
                    reward_cache_key(
                        component_name=binding.name,
                        resolved_params=binding.resolved_params,
                        batch=batch,
                        context=context,
                        row=row,
                    )
                    for row in range(batch.batch_size)
                )

        raw: dict[str, Any] = {}
        weighted: dict[str, Any] = {}
        shared_metadata: dict[str, Mapping[str, Any]] = {}
        sample_metadata: dict[str, tuple[Mapping[str, Any], ...]] = {}
        total = None
        for binding in self._clients:
            vector = self._score_binding(
                binding,
                batch,
                context,
                cache_keys=cache_keys_by_name.get(binding.name),
            )
            values = vector.values
            weighted_values = (values * binding.weight).contiguous()
            raw[binding.name] = values
            weighted[binding.name] = weighted_values
            shared_metadata[binding.name] = vector.shared_metadata
            sample_metadata[binding.name] = vector.sample_metadata
            total = (
                weighted_values.clone()
                if total is None
                else total.add(weighted_values)
            )

        import torch

        assert total is not None
        return RewardShard(
            sample_id=batch.sample_id,
            raw=raw,
            weighted=weighted,
            weighted_total=total.contiguous(),
            valid_mask=torch.ones(batch.batch_size, dtype=torch.bool),
            shared_metadata=shared_metadata,
            sample_metadata=sample_metadata,
        )

    def close(self) -> None:
        """Idempotently close only this facade, never injected resources."""

        self._closed = True

    def _score_binding(
        self,
        binding: RewardClientBinding,
        batch: RolloutBatch,
        context: StepContext,
        *,
        cache_keys: tuple[str, ...] | None,
    ) -> RewardVector:
        cached: list[RewardVector | None] = [None] * batch.batch_size
        if self._cache is not None:
            if cache_keys is None or len(cache_keys) != batch.batch_size:
                raise RuntimeError("reward cache keys were not precomputed")
            cached = [self._cache.get(key) for key in cache_keys]

        missing = tuple(
            row for row, vector in enumerate(cached) if vector is None
        )
        if missing:
            shard = batch.slice(missing)
            import torch

            with torch.inference_mode():
                produced = binding.client.score(shard, context)
            produced = _normalize_vector(
                produced,
                expected_sample_id=shard.sample_id,
            )
            for offset, row in enumerate(missing):
                single = RewardVector(
                    sample_id=(produced.sample_id[offset],),
                    values=produced.values[offset : offset + 1].contiguous(),
                    shared_metadata=produced.shared_metadata,
                    sample_metadata=(produced.sample_metadata[offset],),
                )
                cached[row] = single
                if self._cache is not None:
                    assert cache_keys is not None
                    self._cache.set(cache_keys[row], single)

        vectors = tuple(cached)
        if any(vector is None for vector in vectors):
            raise RuntimeError("reward cache/client merge left an uncovered row")
        resolved = tuple(vector for vector in vectors if vector is not None)
        first_shared = resolved[0].shared_metadata
        if any(vector.shared_metadata != first_shared for vector in resolved[1:]):
            raise ValueError(
                f"reward component {binding.name!r} returned conflicting "
                "shared_metadata across samples"
            )
        import torch

        return RewardVector(
            sample_id=batch.sample_id,
            values=torch.cat(
                tuple(vector.values for vector in resolved),
                dim=0,
            ).contiguous(),
            shared_metadata=first_shared,
            sample_metadata=tuple(
                vector.sample_metadata[0] for vector in resolved
            ),
        )


def _normalize_vector(
    vector: RewardVector,
    *,
    expected_sample_id: tuple[str, ...],
) -> RewardVector:
    if not isinstance(vector, RewardVector):
        raise TypeError("RewardClient.score() must return RewardVector")
    if vector.sample_id != expected_sample_id:
        raise ValueError("RewardClient sample_id order does not match its input shard")
    values = vector.values.detach().to(device="cpu", dtype=None)
    import torch

    values = values.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("RewardClient values must be finite after normalization")
    return RewardVector(
        sample_id=vector.sample_id,
        values=values,
        shared_metadata=vector.shared_metadata,
        sample_metadata=vector.sample_metadata,
    )


def _validate_tensor(name: str, value: Any, size: int, dtype: Any) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        value.device.type != "cpu"
        or value.dtype != dtype
        or tuple(value.shape) != (size,)
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous CPU {dtype} with shape [{size}]"
        )
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} must be detached")
    if dtype != torch.bool and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
