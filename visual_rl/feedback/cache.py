"""Strict per-client, per-sample cache for unweighted reward vectors."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from visual_rl.core.types import (
    FrozenMapping,
    RewardVector,
    RolloutBatch,
    StepContext,
    to_plain_dict,
)

CACHE_VERSION = 1
_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CACHE_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "value",
        "shared_metadata",
        "sample_metadata",
    }
)


def stable_hash_bytes(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("stable_hash_bytes requires bytes")
    return hashlib.sha256(data).hexdigest()


def stable_hash_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("stable_hash_text requires str")
    return stable_hash_bytes(text.encode("utf-8"))


def stable_hash_json(data: Any) -> str:
    """Hash only strict finite JSON values; never inspect arbitrary objects."""

    _require_json_value(data)
    encoded = json.dumps(
        data,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return stable_hash_bytes(encoded)


def reward_cache_key(
    *,
    component_name: str,
    resolved_params: FrozenMapping,
    batch: RolloutBatch,
    context: StepContext,
    row: int,
) -> str:
    """Return the one canonical cache key for one component/sample pair."""

    if not isinstance(component_name, str) or not component_name:
        raise TypeError("component_name must be a non-empty string")
    if not isinstance(resolved_params, FrozenMapping):
        raise TypeError("resolved_params must be a FrozenMapping")
    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    if not isinstance(context, StepContext):
        raise TypeError("context must be a StepContext")
    if batch.context is not context:
        raise ValueError("batch.context must be the identical StepContext")
    if type(row) is not int or not 0 <= row < batch.batch_size:
        raise IndexError("reward cache row is out of bounds")

    payload: dict[str, Any] = {
        "schema_version": CACHE_VERSION,
        "component": component_name,
        "resolved_params": to_plain_dict(resolved_params),
        "sample_id": batch.sample_id[row],
        "context": {
            "step": context.step,
            "seed": context.seed,
            "rank": context.rank,
            "world_size": context.world_size,
        },
        "prompt": batch.prompts[row],
        "prompt_metadata": to_plain_dict(batch.metadata[row]),
        "media": _tensor_identity(batch.media[row], name="media"),
    }
    if component_name == "reward_3d":
        if batch.camera_trajectory is None:
            raise ValueError("reward_3d cache key requires camera_trajectory")
        payload["camera_trajectory"] = _tensor_identity(
            batch.camera_trajectory[row],
            name="camera_trajectory",
            required_dtype="torch.float64",
        )
    return stable_hash_json(payload)


class RewardCache:
    """Rank-local cache owned by the runtime factory reward ExitStack."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> RewardVector | None:
        path = self._path(key)
        with self._locked(path):
            try:
                payload_bytes = path.read_bytes()
            except FileNotFoundError:
                return None
            try:
                payload = json.loads(
                    payload_bytes.decode("utf-8"),
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_keys,
                )
                _require_json_value(payload)
                if not isinstance(payload, dict) or set(payload) != _CACHE_FIELDS:
                    raise ValueError("reward cache entry has an invalid field set")
                if payload["schema_version"] != CACHE_VERSION:
                    raise ValueError("reward cache entry has an unsupported version")
                sample_id = payload["sample_id"]
                value = payload["value"]
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("reward cache sample_id is invalid")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError("reward cache value must be finite numeric")
                import torch

                return RewardVector(
                    sample_id=(sample_id,),
                    values=torch.tensor([float(value)], dtype=torch.float32),
                    shared_metadata=payload["shared_metadata"],
                    sample_metadata=(payload["sample_metadata"],),
                )
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                self._quarantine(path, payload_bytes)
                raise

    def set(self, key: str, vector: RewardVector) -> None:
        path = self._path(key)
        if not isinstance(vector, RewardVector):
            raise TypeError("RewardCache.set requires a RewardVector")
        if len(vector.sample_id) != 1:
            raise ValueError("RewardCache stores exactly one sample per entry")
        value = vector.values
        if (
            value.device.type != "cpu"
            or str(value.dtype) != "torch.float32"
            or not value.is_contiguous()
        ):
            raise ValueError(
                "cached RewardVector values must be contiguous CPU float32"
            )
        payload = {
            "schema_version": CACHE_VERSION,
            "sample_id": vector.sample_id[0],
            "value": float(value[0].item()),
            "shared_metadata": to_plain_dict(vector.shared_metadata),
            "sample_metadata": to_plain_dict(vector.sample_metadata[0]),
        }
        _require_json_value(payload)
        with self._locked(path):
            self._write_atomic(path, payload)

    def close(self) -> None:
        """Idempotent no-op; the cache owns no recursive resources."""

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or _KEY_PATTERN.fullmatch(key) is None:
            raise ValueError("reward cache key must be a lowercase SHA-256 digest")
        return self.root / f"{key}.json"

    @contextmanager
    def _locked(self, path: Path):
        lock_path = path.with_name(f"{path.name}.lock")
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    payload,
                    handle,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @staticmethod
    def _quarantine(path: Path, payload: bytes) -> None:
        suffix = stable_hash_bytes(payload)[:16]
        os.replace(path, path.with_name(f"{path.stem}.corrupt-{suffix}.json"))


def _tensor_identity(
    value: Any,
    *,
    name: str,
    required_dtype: str | None = None,
) -> dict[str, Any]:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError(f"{name} must be detached")
            tensor = value.detach().to(device="cpu").contiguous()
            dtype = str(tensor.dtype)
            if required_dtype is not None and dtype != required_dtype:
                raise TypeError(f"{name} must have dtype {required_dtype}")
            raw = tensor.view(torch.uint8).numpy().tobytes()
            return {
                "kind": "torch",
                "dtype": dtype,
                "shape": list(tensor.shape),
                "sha256": stable_hash_bytes(raw),
            }
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            dtype = str(array.dtype)
            if required_dtype is not None:
                normalized_required = required_dtype.removeprefix("torch.")
                if dtype != normalized_required:
                    raise TypeError(f"{name} must have dtype {normalized_required}")
            return {
                "kind": "numpy",
                "dtype": dtype,
                "shape": list(array.shape),
                "sha256": stable_hash_bytes(array.tobytes()),
            }
    except ImportError:
        pass
    raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _require_json_value(value: Any) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is not allowed")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _require_json_value(item)
        return
    raise TypeError(f"value is not strict JSON: {type(value).__name__}")
