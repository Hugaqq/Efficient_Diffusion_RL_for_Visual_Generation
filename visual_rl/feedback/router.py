"""Unified reward routing, weighting, validity, normalization, and caching."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
import time
from typing import Any

import numpy as np

from visual_rl.configs.schema import normalize_reward_schedule
from visual_rl.core.registry import REWARD_CLIENTS
from visual_rl.core.types import RewardBatch
import visual_rl.feedback.world_r1_rewards  # noqa: F401 - registers World-R1 reward clients
from visual_rl.feedback.cache import RewardCache, stable_hash_json, stable_hash_media
from visual_rl.feedback.clients import RewardProtocolError, redact_error_text


class _UnstableConstructorValue(Exception):
    pass


_MISSING_CACHE_FINGERPRINT = object()


class RewardRouter:
    def __init__(self, config: dict[str, Any], cache_dir: str | Path | None = None):
        if not isinstance(config, dict):
            from dataclasses import asdict

            config = asdict(config)
        self.config = dict(config)
        if "normalize" in config:
            raise ValueError(
                "Reward normalization belongs to AdvantageComputer; remove rewards.normalize"
            )
        self.weights = dict(config.get("weights", {}))
        self.schedule = normalize_reward_schedule(
            config.get("schedule", []),
            weights=self.weights,
            clients=config.get("clients", {}),
        )
        self.fail_policy = config.get("fail_policy", "invalid")
        self.cache = RewardCache(cache_dir)
        self.clients = {}
        self.client_versions = {}
        self.client_fingerprints = {}
        for key, client_config in config.get("clients", {}).items():
            name = client_config.get("name", key)
            cls = REWARD_CLIENTS.get(name)
            params = dict(client_config.get("params", {}))
            kwargs = {
                k: v
                for k, v in client_config.items()
                if k not in {"name", "version", "params", "target"}
            }
            kwargs.update(params)
            client = cls(**kwargs)
            self.clients[key] = client
            self.client_versions[key] = client_config.get("version", "v1")
            self.client_fingerprints[key] = self._client_fingerprint(
                registry_name=name,
                registry_component=cls,
                constructor_config=kwargs,
                client=client,
            )

        if not self.clients and self.weights:
            for key in self.weights:
                cls = REWARD_CLIENTS.get(key)
                client = cls()
                self.clients[key] = client
                self.client_versions[key] = "v1"
                self.client_fingerprints[key] = self._client_fingerprint(
                    registry_name=key,
                    registry_component=cls,
                    constructor_config={},
                    client=client,
                )

    def score(
        self,
        media: Any,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        sample_id: Any = None,
        *,
        step: int | None = None,
    ) -> RewardBatch:
        import torch

        schedule_phase, effective_weights = self._effective_weights(step)
        resolved_sample_id = self._coerce_sample_id(sample_id, expected=len(prompts))
        raw_numpy: dict[str, np.ndarray] = {}
        weighted_numpy: dict[str, np.ndarray] = {}
        merged_metadata: dict[str, Any] = {}
        valid = np.ones(len(prompts), dtype=bool)
        cache_hits = 0
        cache_misses = 0
        client_latencies: list[float] = []
        reward_latencies: list[float] = []
        cache_status: dict[str, bool] = {}

        for reward_name, weight in effective_weights.items():
            reward_started = time.perf_counter()
            if reward_name not in self.clients:
                raise KeyError(f"No reward client configured for {reward_name!r}")
            client_fingerprint = self.client_fingerprints.get(reward_name)
            cache_key = None
            if client_fingerprint is not None:
                cache_key = self._cache_key(
                    reward_name,
                    self.client_versions.get(reward_name, "v1"),
                    prompts,
                    metadata,
                    media,
                    resolved_sample_id,
                    client_fingerprint,
                )
            cached = self.cache.get(cache_key) if cache_key is not None else None
            if cached is not None:
                cache_hits += 1
                cache_status[reward_name] = True
                self._validate_cached_sample_id(
                    reward_name,
                    cached,
                    resolved_sample_id,
                )
                values = self._coerce_reward_values(
                    reward_name,
                    cached["values"],
                    expected=len(prompts),
                )
                reward_meta = self._prepare_reward_metadata(
                    reward_name,
                    cached.get("metadata", {}),
                    resolved_sample_id,
                )
                if "valid_mask" not in cached:
                    raise ValueError(
                        f"Cached reward {reward_name!r} is missing valid_mask semantics."
                    )
                reward_valid = self._coerce_valid_mask(
                    reward_name,
                    cached["valid_mask"],
                    expected=len(prompts),
                )
            else:
                cache_misses += 1
                cache_status[reward_name] = False
                cache_result = False
                client_started = time.perf_counter()
                try:
                    values, reward_meta = self._score_client(
                        self.clients[reward_name],
                        media,
                        prompts,
                        metadata,
                        resolved_sample_id,
                    )
                    values = self._coerce_reward_values(
                        reward_name,
                        values,
                        expected=len(prompts),
                    )
                    cache_result = True
                except RewardProtocolError:
                    raise
                except Exception as exc:  # noqa: BLE001 - failure is represented in valid_mask
                    values = np.zeros(len(prompts), dtype=np.float32)
                    reward_valid = np.zeros(len(prompts), dtype=bool)
                    reward_meta = {
                        "error_type": type(exc).__name__,
                        "error": redact_error_text(exc),
                    }
                    if self.fail_policy == "raise":
                        raise
                finally:
                    client_latencies.append(time.perf_counter() - client_started)
                reward_meta = self._prepare_reward_metadata(
                    reward_name,
                    reward_meta,
                    resolved_sample_id,
                )
                if cache_result:
                    reward_valid = self._coerce_valid_mask(
                        reward_name,
                        reward_meta.get("valid_mask", np.ones(len(prompts), dtype=bool)),
                        expected=len(prompts),
                    )
                reward_meta["valid_mask"] = reward_valid.tolist()
                if cache_result and cache_key is not None:
                    self.cache.set(
                        cache_key,
                        {
                            "values": values.tolist(),
                            "metadata": reward_meta,
                            "sample_id": resolved_sample_id,
                            "valid_mask": reward_valid.tolist(),
                        },
                    )
            valid &= reward_valid
            raw_numpy[reward_name] = values.astype(np.float32)
            weighted_numpy[reward_name] = raw_numpy[reward_name] * float(weight)
            merged_metadata[reward_name] = reward_meta
            reward_latencies.append(time.perf_counter() - reward_started)

        request_count = cache_hits + cache_misses
        merged_metadata["_runtime"] = {
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": (
                float(cache_hits / request_count) if request_count else 0.0
            ),
            "cache_status": cache_status,
            "client_latencies_s": client_latencies,
            "reward_latencies_s": reward_latencies,
        }
        if schedule_phase is not None:
            merged_metadata["_schedule"] = {
                "name": schedule_phase["name"],
                "step": step,
                "start_step": schedule_phase["start_step"],
                "end_step": schedule_phase["end_step"],
                "effective_weights": dict(effective_weights),
            }

        if weighted_numpy:
            total_np = sum(weighted_numpy.values())
        else:
            total_np = np.zeros(len(prompts), dtype=np.float32)

        weighted_total = torch.as_tensor(total_np, dtype=torch.float32)
        return RewardBatch(
            raw={key: torch.as_tensor(value, dtype=torch.float32) for key, value in raw_numpy.items()},
            weighted={key: torch.as_tensor(value, dtype=torch.float32) for key, value in weighted_numpy.items()},
            weighted_total=weighted_total,
            valid_mask=torch.as_tensor(valid, dtype=torch.bool),
            metadata=merged_metadata,
            sample_id=resolved_sample_id,
        )

    def _effective_weights(
        self, step: int | None
    ) -> tuple[dict[str, Any] | None, dict[str, float]]:
        if not self.schedule:
            return None, dict(self.weights)
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(
                "RewardRouter requires an integer step when rewards.schedule is set"
            )
        for phase in self.schedule:
            if phase["start_step"] <= step < phase["end_step"]:
                return phase, dict(phase["weights"])
        raise ValueError(
            f"Reward step {step} is outside the configured rewards.schedule"
        )

    @staticmethod
    def _cache_key(
        reward_name: str,
        reward_version: str,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        media: Any,
        sample_id: list[str] | None,
        client_fingerprint: Any = None,
    ) -> str:
        payload_hash = stable_hash_json(
            {
                "prompts": prompts,
                "metadata": metadata,
                "sample_id": sample_id,
                "client_fingerprint": client_fingerprint,
            }
        )
        media_hash = stable_hash_media(media)
        return f"{reward_name}-{reward_version}-{payload_hash}-{media_hash}"

    @staticmethod
    def _coerce_sample_id(
        sample_id: Any,
        *,
        expected: int,
    ) -> list[str] | None:
        if sample_id is None:
            return None
        if isinstance(sample_id, (str, bytes)) or not hasattr(sample_id, "__len__"):
            raise ValueError(
                f"RewardRouter sample_id must be a sequence of length {expected}."
            )
        resolved = list(sample_id)
        if len(resolved) != expected:
            raise ValueError(
                f"RewardRouter sample_id length must be {expected}, got {len(resolved)}."
            )
        if any(not isinstance(item, str) or not item.strip() for item in resolved):
            raise ValueError("RewardRouter sample_id values must be non-empty strings.")
        if len(set(resolved)) != len(resolved):
            raise ValueError("RewardRouter sample_id values must be unique.")
        return resolved

    @classmethod
    def _prepare_reward_metadata(
        cls,
        reward_name: str,
        metadata: Any,
        sample_id: list[str] | None,
    ) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Reward client {reward_name!r} metadata must be a mapping."
            )
        result = dict(metadata)
        declared_sample_id = result.get("sample_id")
        if declared_sample_id is not None:
            if sample_id is None:
                raise ValueError(
                    f"Reward client {reward_name!r} returned sample_id, but the "
                    "legacy RewardRouter.score call supplied no sample_id to validate."
                )
            declared = cls._coerce_sample_id(
                declared_sample_id,
                expected=len(sample_id),
            )
            if declared != sample_id:
                raise ValueError(
                    f"Reward client {reward_name!r} sample_id order does not match "
                    "the requested rollout batch."
                )
            result["sample_id"] = declared
            if "sample_id_mode" not in result:
                result["sample_id_mode"] = "explicit"
        elif "sample_id_mode" not in result:
            result["sample_id_mode"] = "trusted_input_order_legacy"
        return result

    @classmethod
    def _client_fingerprint(
        cls,
        *,
        registry_name: str,
        registry_component: Any,
        constructor_config: Mapping[str, Any],
        client: Any,
    ) -> dict[str, Any] | None:
        try:
            normalized_constructor = cls._normalize_constructor_value(
                constructor_config
            )
        except _UnstableConstructorValue:
            return None
        identity = {
            "registry_name": registry_name,
            "registry_component": cls._callable_identity(registry_component),
            "client_class": cls._type_identity(client),
            "constructor_sha256": stable_hash_json(normalized_constructor),
        }
        fingerprint = getattr(
            client,
            "cache_fingerprint",
            _MISSING_CACHE_FINGERPRINT,
        )
        if fingerprint is not _MISSING_CACHE_FINGERPRINT:
            value = fingerprint() if callable(fingerprint) else fingerprint
            if value is None:
                return None
            identity["client_reported_sha256"] = stable_hash_json(
                cls._normalize_explicit_fingerprint(value)
            )
        return identity

    @classmethod
    def _normalize_constructor_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, Path):
            return {"kind": "path", "value": str(value)}
        if isinstance(value, Mapping):
            return {
                key: cls._normalize_constructor_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return {
                "kind": "list",
                "items": [cls._normalize_constructor_value(item) for item in value],
            }
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [cls._normalize_constructor_value(item) for item in value],
            }
        fingerprint = getattr(value, "cache_fingerprint", None)
        if fingerprint is None:
            raise _UnstableConstructorValue
        declared = fingerprint() if callable(fingerprint) else fingerprint
        if declared is None:
            raise _UnstableConstructorValue
        normalized = cls._normalize_explicit_fingerprint(declared)
        return {
            "kind": "injected_callable" if callable(value) else "injected_object",
            "cache_fingerprint_sha256": stable_hash_json(normalized),
        }

    @classmethod
    def _normalize_explicit_fingerprint(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, Path):
            return {"kind": "path", "value": str(value)}
        if isinstance(value, Mapping):
            return {
                key: cls._normalize_explicit_fingerprint(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._normalize_explicit_fingerprint(item) for item in value]
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [
                    cls._normalize_explicit_fingerprint(item) for item in value
                ],
            }
        raise TypeError(
            "cache_fingerprint values must contain only stable JSON-like data."
        )

    @staticmethod
    def _type_identity(value: Any) -> str:
        value_type = value if inspect.isclass(value) else type(value)
        return f"{value_type.__module__}:{value_type.__qualname__}"

    @classmethod
    def _callable_identity(cls, value: Any) -> str:
        if inspect.ismethod(value):
            function = value.__func__
            owner = cls._type_identity(value.__self__)
            return f"{function.__module__}:{function.__qualname__}@{owner}"
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str):
            return f"{module}:{qualname}"
        return cls._type_identity(value)

    @staticmethod
    def _score_client(
        client: Any,
        media: Any,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        sample_id: list[str] | None,
    ) -> tuple[Any, Any]:
        score = client.score
        try:
            signature = inspect.signature(score)
        except (TypeError, ValueError):
            accepts_sample_id = False
        else:
            parameter = signature.parameters.get("sample_id")
            if (
                parameter is not None
                and parameter.kind is inspect.Parameter.POSITIONAL_ONLY
            ):
                return score(media, prompts, metadata, sample_id)
            accepts_sample_id = (
                parameter is not None
                and parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            ) or any(
                item.kind is inspect.Parameter.VAR_KEYWORD
                for item in signature.parameters.values()
            )
        if accepts_sample_id:
            return score(media, prompts, metadata, sample_id=sample_id)
        return score(media, prompts, metadata)

    @staticmethod
    def _validate_cached_sample_id(
        reward_name: str,
        cached: dict[str, Any],
        sample_id: list[str] | None,
    ) -> None:
        if "sample_id" not in cached:
            raise ValueError(
                f"Cached reward {reward_name!r} is missing sample_id identity."
            )
        if cached["sample_id"] != sample_id:
            raise ValueError(
                f"Cached reward {reward_name!r} sample_id identity mismatch."
            )

    @staticmethod
    def _coerce_reward_values(reward_name: str, values: Any, *, expected: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (expected,):
            raise ValueError(
                f"Reward client {reward_name!r} returned shape {array.shape}; expected shape ({expected},)."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Reward client {reward_name!r} returned non-finite values.")
        return array

    @staticmethod
    def _coerce_valid_mask(reward_name: str, mask: Any, *, expected: int) -> np.ndarray:
        try:
            import torch
        except ModuleNotFoundError:  # pragma: no cover - torch is a runtime dependency
            torch = None
        if torch is not None and isinstance(mask, torch.Tensor):
            if mask.dtype is not torch.bool:
                raise ValueError(
                    f"Reward client {reward_name!r} valid_mask must have bool dtype."
                )
            mask = mask.detach().cpu().numpy()
        array = np.asarray(mask)
        if array.shape != (expected,):
            raise ValueError(
                f"Reward client {reward_name!r} valid_mask shape must be "
                f"({expected},), got {array.shape}."
            )
        if array.dtype.kind != "b":
            raise ValueError(
                f"Reward client {reward_name!r} valid_mask must contain booleans."
            )
        return array.astype(bool, copy=False)
