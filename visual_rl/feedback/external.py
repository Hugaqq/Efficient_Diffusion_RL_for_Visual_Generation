"""Runtime adapter for trusted external feedback callables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
import math
from typing import Any

from visual_rl.core.types import RolloutBatch, RewardBatch
from visual_rl.feedback.base import FeedbackProvider
from visual_rl.feedback.cache import RewardCache, stable_hash_json, stable_hash_media


_CACHE_SCHEMA_VERSION = 2
_CACHE_KIND = "__visual_rl_cache_kind__"


class CallableFeedbackProvider(FeedbackProvider):
    """Adapt a trusted batch scoring callable to the feedback contract."""

    def __init__(
        self,
        component,
        *,
        name,
        version,
        params=None,
        weight=1.0,
        cache_dir=None,
        target=None,
        source_sha256=None,
    ):
        if not callable(component) and not isinstance(component, FeedbackProvider):
            raise TypeError(
                "External feedback component must be callable or a FeedbackProvider"
            )
        if not isinstance(name, str) or not name.strip():
            raise ValueError("External feedback name must be a non-empty string")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("External feedback version must be a non-empty string")
        resolved_params = {} if params is None else params
        if not isinstance(resolved_params, Mapping):
            raise TypeError("External feedback params must be a mapping")
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            raise TypeError("External feedback weight must be a finite number")
        if not math.isfinite(float(weight)):
            raise ValueError("External feedback weight must be finite")

        self.component = component
        self.name = name.strip()
        self.version = version.strip()
        self.params = _json_copy(dict(resolved_params), "params")
        self.weight = float(weight)
        self.target = target or _component_label(component)
        self.source_sha256 = source_sha256
        self.cache = RewardCache(cache_dir)
        self._visual_rl_identity = _json_copy(
            {
                "name": self.name,
                "target": self.target,
                "version": self.version,
                "source_sha256": self.source_sha256,
                "params": self.params,
                "weight": self.weight,
            },
            "provider identity",
        )

    def score(self, batch: RolloutBatch) -> RewardBatch:
        batch.validate_lightweight()
        input_identity = None
        cache_key = None
        if self.cache.root is not None:
            input_identity = self._input_identity(batch)
            cache_key = stable_hash_json(
                {
                    "provider": self._visual_rl_identity,
                    "input": input_identity,
                }
            )
            cached = self._load_cached(cache_key, input_identity, batch)
            if cached is not None:
                return cached

        if isinstance(self.component, FeedbackProvider):
            result = self.component.score(batch)
        else:
            result = self.component(batch, **self.params)
        rewards = self._normalize_result(result, batch)
        rewards.validate_against(batch)
        if cache_key is not None and input_identity is not None:
            self._store_cached(cache_key, input_identity, rewards)
        return rewards

    def _input_identity(self, batch: RolloutBatch) -> dict[str, Any]:
        return _json_copy(
            {
                "ordered_sample_id": list(batch.sample_id),
                "target": self.target,
                "version": self.version,
                "params": self.params,
                "sample_id_sha256": stable_hash_media(batch.sample_id),
                "prompt_id_sha256": stable_hash_media(batch.prompt_id),
                "group_id_sha256": stable_hash_media(batch.group_id),
                "branch_id_sha256": stable_hash_media(batch.branch_id),
                "context": None if batch.context is None else asdict(batch.context),
                "media_layout": batch.media_layout,
                "prompts_sha256": stable_hash_media(batch.prompts),
                "metadata_sha256": stable_hash_media(batch.metadata),
                "media_sha256": stable_hash_media(batch.media),
                "latents_sha256": stable_hash_media(batch.latents),
                "next_latents_sha256": stable_hash_media(batch.next_latents),
                "timesteps_sha256": stable_hash_media(batch.timesteps),
                "old_log_probs_sha256": stable_hash_media(batch.old_log_probs),
                "kl_sha256": stable_hash_media(batch.kl),
                "transition_mask_sha256": stable_hash_media(
                    batch.transition_mask
                ),
                "model_metadata_sha256": stable_hash_media(
                    batch.model_metadata
                ),
                "model_tensors_sha256": stable_hash_media(batch.model_tensors),
            },
            "feedback cache input identity",
        )

    def _normalize_result(
        self, result: Any, batch: RolloutBatch
    ) -> RewardBatch:
        if isinstance(result, RewardBatch):
            return self._normalize_reward_batch(result, batch)

        result_metadata: dict[str, Any] = {}
        values = result
        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[1], Mapping):
                raise TypeError(
                    "External feedback tuple result must be (values, metadata)"
                )
            values, metadata = result
            result_metadata = _json_copy(dict(metadata), "reward metadata")
        if isinstance(values, RewardBatch):
            raise TypeError(
                "A RewardBatch must be returned directly, not inside a tuple"
            )

        vector = _as_vector(values, batch.batch_size)
        weighted = vector * self.weight
        valid_mask = _true_mask(batch.batch_size, vector)
        result_metadata.update(
            {
                "provider_identity": dict(self._visual_rl_identity),
                "sample_id_provenance": "trusted_input_order_callable",
                "trusted_input_order_callable": True,
                "weight_source": f"rewards.weights[{self.name!r}]",
                "configured_weight": self.weight,
            }
        )
        rewards = RewardBatch(
            raw={self.name: vector},
            weighted={self.name: weighted},
            weighted_total=weighted,
            valid_mask=valid_mask,
            metadata=result_metadata,
            sample_id=list(batch.sample_id),
        ).canonical()
        rewards.validate_against(batch)
        return rewards

    def _normalize_reward_batch(
        self, result: RewardBatch, batch: RolloutBatch
    ) -> RewardBatch:
        if not isinstance(result.raw, Mapping):
            raise TypeError("External RewardBatch.raw must be a mapping")
        if len(result.raw) != 1:
            raise ValueError(
                "External RewardBatch must contain exactly one raw reward; "
                "multi-reward providers require a full feedback plugin"
            )
        source_name, raw_values = next(iter(result.raw.items()))
        source = RewardBatch(
            raw={str(source_name): raw_values},
            weighted={str(source_name): raw_values},
            weighted_total=raw_values,
            valid_mask=result.valid_mask,
            metadata=dict(result.metadata),
            sample_id=result.sample_id,
        ).canonical()
        raw = source.raw[str(source_name)]
        metadata = dict(source.metadata)
        metadata.update(
            {
                "provider_identity": dict(self._visual_rl_identity),
                "source_reward_name": str(source_name),
                "weight_source": f"rewards.weights[{self.name!r}]",
                "configured_weight": self.weight,
            }
        )
        rewards = RewardBatch(
            raw={self.name: raw},
            weighted={self.name: raw * self.weight},
            weighted_total=raw * self.weight,
            valid_mask=source.valid_mask,
            metadata=metadata,
            sample_id=source.sample_id,
        ).canonical()
        rewards.validate_against(batch)
        return rewards

    def _load_cached(
        self,
        cache_key: str,
        input_identity: dict[str, Any],
        batch: RolloutBatch,
    ) -> RewardBatch | None:
        try:
            payload = self.cache.get(cache_key)
        except Exception as exc:  # noqa: BLE001 - corrupt cache must fail closed
            raise ValueError("External feedback cache is not valid JSON") from exc
        if payload is None:
            return None
        try:
            if not isinstance(payload, dict):
                raise TypeError("cache payload must be an object")
            expected_fields = {
                "schema_version",
                "provider_identity",
                "input_identity",
                "reward_batch",
                "reward_batch_sha256",
            }
            if set(payload) != expected_fields:
                raise ValueError("cache payload fields are invalid")
            if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
                raise ValueError("cache schema version mismatch")
            if payload.get("provider_identity") != self._visual_rl_identity:
                raise ValueError("cache provider identity mismatch")
            if payload.get("input_identity") != input_identity:
                raise ValueError("cache input identity mismatch")
            if payload["reward_batch_sha256"] != stable_hash_json(
                payload["reward_batch"]
            ):
                raise ValueError("cache reward payload digest mismatch")
            rewards = _reward_batch_from_payload(payload["reward_batch"]).canonical()
            rewards.validate_against(batch)
        except Exception as exc:  # noqa: BLE001 - every bad payload is rejected
            raise ValueError("External feedback cache payload failed validation") from exc
        return rewards

    def _store_cached(
        self,
        cache_key: str,
        input_identity: dict[str, Any],
        rewards: RewardBatch,
    ) -> None:
        reward_payload = _reward_batch_to_payload(rewards)
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "provider_identity": self._visual_rl_identity,
            "input_identity": input_identity,
            "reward_batch": reward_payload,
            "reward_batch_sha256": stable_hash_json(reward_payload),
        }
        _json_copy(payload, "feedback cache payload")
        self.cache.set(cache_key, payload)


def _component_label(component: Any) -> str:
    module = getattr(component, "__module__", type(component).__module__)
    qualname = getattr(component, "__qualname__", type(component).__qualname__)
    return f"{module}:{qualname}"


def _as_vector(values: Any, batch_size: int) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover - training environments include torch
        import numpy as np

        vector = np.asarray(values)
        if vector.dtype.kind == "b":
            raise TypeError("External feedback values must be numeric, not bool")
        if vector.ndim != 1 or vector.shape[0] != batch_size:
            raise ValueError(
                "External feedback values must be a 1D vector with "
                f"length {batch_size}"
            )
        return vector

    if isinstance(values, torch.Tensor):
        vector = values.detach()
    else:
        try:
            vector = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(
                "External feedback values must be a tensor, list, or ndarray"
            ) from exc
    if vector.dtype == torch.bool:
        raise TypeError("External feedback values must be numeric, not bool")
    if vector.ndim != 1 or vector.shape[0] != batch_size:
        raise ValueError(
            "External feedback values must be a 1D vector with "
            f"length {batch_size}"
        )
    return vector


def _true_mask(batch_size: int, vector: Any) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover - training environments include torch
        import numpy as np

        return np.ones(batch_size, dtype=np.bool_)
    device = vector.device if isinstance(vector, torch.Tensor) else None
    return torch.ones(batch_size, dtype=torch.bool, device=device)


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"External feedback {label} must be JSON-safe") from exc


def _reward_batch_to_payload(rewards: RewardBatch) -> dict[str, Any]:
    return {
        "raw": _encode_value(rewards.raw),
        "weighted": _encode_value(rewards.weighted),
        "weighted_total": _encode_value(rewards.weighted_total),
        "valid_mask": _encode_value(rewards.valid_mask),
        "metadata": _encode_value(rewards.metadata),
        "sample_id": _encode_value(rewards.sample_id),
    }


def _reward_batch_from_payload(payload: Any) -> RewardBatch:
    if not isinstance(payload, dict):
        raise TypeError("cached RewardBatch must be an object")
    expected = {
        "raw",
        "weighted",
        "weighted_total",
        "valid_mask",
        "metadata",
        "sample_id",
    }
    if set(payload) != expected:
        raise ValueError("cached RewardBatch fields are incomplete")
    raw = _decode_value(payload["raw"])
    weighted = _decode_value(payload["weighted"])
    metadata = _decode_value(payload["metadata"])
    if not isinstance(raw, dict) or not isinstance(weighted, dict):
        raise TypeError("cached reward mappings are invalid")
    if not isinstance(metadata, dict):
        raise TypeError("cached reward metadata is invalid")
    return RewardBatch(
        raw=raw,
        weighted=weighted,
        weighted_total=_decode_value(payload["weighted_total"]),
        valid_mask=_decode_value(payload["valid_mask"]),
        metadata=metadata,
        sample_id=_decode_value(payload["sample_id"]),
    )


def _encode_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {_CACHE_KIND: "none"}
    if isinstance(value, bool):
        return {_CACHE_KIND: "bool", "value": value}
    if isinstance(value, int):
        return {_CACHE_KIND: "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("External feedback cache values must be finite")
        return {_CACHE_KIND: "float", "value": value}
    if isinstance(value, str):
        return {_CACHE_KIND: "str", "value": value}
    if isinstance(value, Mapping):
        items = [
            [_encode_value(key), _encode_value(item)]
            for key, item in value.items()
        ]
        return {
            _CACHE_KIND: "dict",
            "items": sorted(items, key=_encoded_mapping_key),
        }
    if isinstance(value, list | tuple):
        kind = "tuple" if isinstance(value, tuple) else "list"
        return {_CACHE_KIND: kind, "items": [_encode_value(item) for item in value]}

    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.is_complex():
                raise TypeError("Complex tensors are not supported in reward cache")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("External feedback cache values must be finite")
            return {
                _CACHE_KIND: "torch",
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "data": tensor.tolist(),
            }
    except ImportError:  # pragma: no cover - torch is optional for cache metadata
        pass

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _encode_value(value.item())
        if isinstance(value, np.ndarray):
            if value.dtype.kind in {"O", "c"}:
                raise TypeError("Object and complex arrays are not supported")
            if value.dtype.kind == "f" and not bool(np.isfinite(value).all()):
                raise ValueError("External feedback cache values must be finite")
            return {
                _CACHE_KIND: "numpy",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    raise TypeError(f"Unsupported reward cache value: {type(value).__name__}")


def _decode_value(payload: Any) -> Any:
    if not isinstance(payload, dict) or _CACHE_KIND not in payload:
        raise TypeError("Invalid tagged reward cache value")
    kind = payload[_CACHE_KIND]
    if not isinstance(kind, str):
        raise TypeError("Reward cache value kind must be a string")
    if kind == "none":
        _require_tag_fields(payload, kind, {_CACHE_KIND})
        return None
    if kind == "bool":
        _require_tag_fields(payload, kind, {_CACHE_KIND, "value"})
        value = payload["value"]
        if not isinstance(value, bool):
            raise TypeError("Tagged bool value must contain a bool")
        return value
    if kind == "int":
        _require_tag_fields(payload, kind, {_CACHE_KIND, "value"})
        value = payload["value"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Tagged int value must contain an int")
        return value
    if kind == "float":
        _require_tag_fields(payload, kind, {_CACHE_KIND, "value"})
        value = payload["value"]
        if not isinstance(value, float) or not math.isfinite(value):
            raise TypeError("Tagged float value must contain a finite float")
        return value
    if kind == "str":
        _require_tag_fields(payload, kind, {_CACHE_KIND, "value"})
        value = payload["value"]
        if not isinstance(value, str):
            raise TypeError("Tagged str value must contain a string")
        return value
    if kind == "dict":
        _require_tag_fields(payload, kind, {_CACHE_KIND, "items"})
        items = payload["items"]
        if not isinstance(items, list):
            raise TypeError("Tagged dict items must be a list")
        result = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2:
                raise TypeError("Tagged dict entries must be [key, value] pairs")
            key, value = (_decode_value(part) for part in item)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise TypeError("Tagged dict key must be hashable") from exc
            if duplicate:
                raise ValueError(f"Duplicate tagged dict key: {key!r}")
            result[key] = value
        return result
    if kind in {"list", "tuple"}:
        _require_tag_fields(payload, kind, {_CACHE_KIND, "items"})
        encoded_items = payload["items"]
        if not isinstance(encoded_items, list):
            raise TypeError(f"Tagged {kind} items must be a list")
        items = [_decode_value(item) for item in encoded_items]
        return tuple(items) if kind == "tuple" else items
    if kind == "torch":
        import torch

        _require_tag_fields(
            payload, kind, {_CACHE_KIND, "dtype", "shape", "data"}
        )
        if not isinstance(payload["dtype"], str):
            raise TypeError("Torch reward cache dtype must be a string")
        shape = _decode_shape(payload["shape"], "Torch")
        dtype = getattr(torch, payload["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError("Unknown torch dtype in reward cache")
        probe = torch.empty((), dtype=dtype)
        if probe.is_complex() or not (
            probe.is_floating_point()
            or dtype == torch.bool
            or _is_torch_integer_dtype(torch, dtype)
        ):
            raise TypeError("Unsupported torch dtype in reward cache")
        value = torch.tensor(payload["data"], dtype=dtype)
        if tuple(value.shape) != shape:
            raise ValueError("Torch reward cache shape mismatch")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("Torch reward cache values must be finite")
        return value
    if kind == "numpy":
        import numpy as np

        _require_tag_fields(
            payload, kind, {_CACHE_KIND, "dtype", "shape", "data"}
        )
        if not isinstance(payload["dtype"], str):
            raise TypeError("Numpy reward cache dtype must be a string")
        shape = _decode_shape(payload["shape"], "Numpy")
        dtype = np.dtype(payload["dtype"])
        if dtype.kind not in {"b", "i", "u", "f"}:
            raise TypeError("Unsafe numpy dtype in reward cache")
        value = np.asarray(payload["data"], dtype=dtype)
        if value.shape != shape:
            raise ValueError("Numpy reward cache shape mismatch")
        if dtype.kind == "f" and not bool(np.isfinite(value).all()):
            raise ValueError("Numpy reward cache values must be finite")
        return value
    raise ValueError(f"Unknown reward cache value kind: {kind!r}")


def _require_tag_fields(
    payload: dict[str, Any], kind: str, expected: set[str]
) -> None:
    if set(payload) != expected:
        raise ValueError(f"Tagged {kind} value has unexpected fields")


def _encoded_mapping_key(item: list[dict[str, Any]]) -> str:
    return json.dumps(item[0], sort_keys=True, separators=(",", ":"))


def _decode_shape(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise TypeError(f"{label} reward cache shape must be non-negative integers")
    return tuple(value)


def _is_torch_integer_dtype(torch: Any, dtype: Any) -> bool:
    try:
        torch.iinfo(dtype)
    except TypeError:
        return False
    return True


__all__ = ["CallableFeedbackProvider"]
