"""Disk cache for expensive reward calls."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
import tempfile
from typing import Any


def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash_text(text: str) -> str:
    return stable_hash_bytes(text.encode("utf-8"))


def stable_hash_json(data: Any) -> str:
    return stable_hash_text(
        json.dumps(
            _hashable_value(data),
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    )


def stable_hash_media(media: Any) -> str:
    return stable_hash_json({"media": media})


def _hashable_value(value: Any) -> Any:
    """Encode provider-visible values without device state or ``repr`` fallback."""

    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Cache hashes reject NaN and infinity")
        return {"kind": "float", "value": value}
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("Cache hashes reject NaN and infinity")
        return {
            "kind": "complex",
            "real": value.real,
            "imag": value.imag,
        }
    if isinstance(value, bytes | bytearray):
        return {"kind": "bytes", "sha256": stable_hash_bytes(bytes(value))}
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.is_sparse:
                tensor = tensor.to_dense()
            tensor = tensor.contiguous()
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                torch.isfinite(tensor).all()
            ):
                raise ValueError("Cache hashes reject NaN and infinity")
            raw_bytes = tensor.view(torch.uint8).numpy().tobytes()
            return {
                "kind": "torch",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": stable_hash_bytes(raw_bytes),
            }
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        pass
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _hashable_value(value.item())
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {
                    "kind": "numpy-object",
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "items": _hashable_value(value.tolist()),
                }
            if not _numpy_array_is_finite(value, np):
                raise ValueError("Cache hashes reject NaN and infinity")
            array = np.ascontiguousarray(value)
            return {
                "kind": "numpy",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "bytes": stable_hash_bytes(array.tobytes()),
            }
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            image = value.convert(value.mode)
            return {
                "kind": "pil",
                "mode": image.mode,
                "size": image.size,
                "bytes": stable_hash_bytes(image.tobytes()),
            }
    except Exception:  # noqa: BLE001 - pillow is optional
        pass
    if isinstance(value, Mapping):
        items = [
            [_hashable_value(key), _hashable_value(item)]
            for key, item in value.items()
        ]
        return {
            "kind": "mapping",
            "items": sorted(
                items,
                key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":")),
            ),
        }
    if isinstance(value, list):
        return {"kind": "list", "items": [_hashable_value(item) for item in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_hashable_value(item) for item in value]}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "type": f"{type(value).__module__}:{type(value).__qualname__}",
            "fields": {
                item.name: _hashable_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if hasattr(value, "__dict__"):
        return {
            "kind": "object",
            "type": f"{type(value).__module__}:{type(value).__qualname__}",
            "attributes": _hashable_value(vars(value)),
        }
    raise TypeError(f"Cannot deterministically hash {type(value).__name__}")


def _numpy_array_is_finite(value: Any, np: Any) -> bool:
    """Check floating leaves, including structured arrays, before hashing."""

    if value.dtype.fields:
        return all(
            _numpy_array_is_finite(value[field], np)
            for field in value.dtype.fields
        )
    if value.dtype.kind in {"f", "c"}:
        return bool(np.isfinite(value).all())
    return True


class RewardCache:
    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if self.root is None:
            raise RuntimeError("RewardCache has no root")
        return self.root / f"{key}.json"

    @contextmanager
    def _locked(self, key: str):
        path = self._path(key)
        lock_path = path.with_name(f"{path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _quarantine(path: Path, payload: bytes) -> Path:
        suffix = stable_hash_bytes(payload)[:16]
        quarantine = path.with_name(f"{path.stem}.corrupt-{suffix}.json")
        os.replace(path, quarantine)
        return quarantine

    def get(self, key: str) -> dict[str, Any] | None:
        if self.root is None:
            return None
        path = self._path(key)
        with self._locked(key):
            if not path.exists():
                return None
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
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._quarantine(path, payload_bytes)
                return None
            except ValueError:
                self._quarantine(path, payload_bytes)
                raise
            try:
                _require_finite_json(payload)
            except ValueError:
                self._quarantine(path, payload_bytes)
                raise
            if not isinstance(payload, dict):
                self._quarantine(path, payload_bytes)
                return None
            return payload

    def set(self, key: str, value: dict[str, Any]) -> None:
        if self.root is None:
            return
        if not isinstance(value, dict):
            raise TypeError("RewardCache values must be dictionaries")
        _require_json_value(value)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(key):
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    tmp_path = Path(handle.name)
                    json.dump(
                        value,
                        handle,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, path)
                tmp_path = None
            finally:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _require_finite_json(value: Any) -> None:
    if isinstance(value, float):
        import math

        if not math.isfinite(value):
            raise ValueError("Non-finite JSON number is not allowed")
    elif isinstance(value, dict):
        for item in value.values():
            _require_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _require_finite_json(item)


def _require_json_value(value: Any) -> None:
    if value is None or isinstance(value, bool | str | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite JSON number is not allowed")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Reward cache JSON object keys must be strings")
            _require_json_value(item)
        return
    raise TypeError(f"Reward cache values must be JSON-safe, got {type(value).__name__}")
