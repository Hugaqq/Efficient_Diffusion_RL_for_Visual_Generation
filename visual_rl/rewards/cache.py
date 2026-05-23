"""Disk cache for expensive reward calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash_text(text: str) -> str:
    return stable_hash_bytes(text.encode("utf-8"))


def stable_hash_json(data: Any) -> str:
    return stable_hash_text(json.dumps(data, sort_keys=True, default=str))


def stable_hash_media(media: Any) -> str:
    if media is None:
        return stable_hash_text("none")
    if isinstance(media, bytes | bytearray):
        return stable_hash_bytes(bytes(media))
    if isinstance(media, str):
        return stable_hash_text(media)
    if isinstance(media, Path):
        return stable_hash_text(str(media))
    if isinstance(media, list | tuple):
        return stable_hash_json([stable_hash_media(item) for item in media])
    if isinstance(media, dict):
        return stable_hash_json({str(key): stable_hash_media(value) for key, value in sorted(media.items())})
    try:
        import torch

        if isinstance(media, torch.Tensor):
            tensor = media.detach().cpu().contiguous()
            return stable_hash_json(
                {
                    "kind": "torch",
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "bytes": stable_hash_bytes(tensor.numpy().tobytes()),
                }
            )
    except Exception:  # noqa: BLE001 - hashing should degrade gracefully
        pass
    try:
        import numpy as np

        if isinstance(media, np.ndarray):
            array = np.ascontiguousarray(media)
            return stable_hash_json(
                {
                    "kind": "numpy",
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "bytes": stable_hash_bytes(array.tobytes()),
                }
            )
    except Exception:  # noqa: BLE001 - numpy is optional
        pass
    try:
        from PIL import Image

        if isinstance(media, Image.Image):
            image = media.convert(media.mode)
            return stable_hash_json(
                {
                    "kind": "pil",
                    "mode": image.mode,
                    "size": image.size,
                    "bytes": stable_hash_bytes(image.tobytes()),
                }
            )
    except Exception:  # noqa: BLE001 - pillow is optional
        pass
    if hasattr(media, "tobytes"):
        try:
            return stable_hash_json(
                {
                    "kind": type(media).__name__,
                    "shape": list(getattr(media, "shape", [])),
                    "bytes": stable_hash_bytes(media.tobytes()),
                }
            )
        except Exception:  # noqa: BLE001 - fall back below
            pass
    return stable_hash_text(repr(type(media)) + repr(getattr(media, "shape", "")) + repr(media))


class RewardCache:
    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if self.root is None:
            raise RuntimeError("RewardCache has no root")
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if self.root is None:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def set(self, key: str, value: dict[str, Any]) -> None:
        if self.root is None:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)
