"""JSON-safe conversion for experiment artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Detach tensor-like values and recursively convert them to JSON data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", type(value).__qualname__),
        )
        return {"callable": f"{module}.{qualname}"}
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "tolist"):
        return to_jsonable(converted.tolist())
    if hasattr(converted, "item"):
        try:
            return to_jsonable(converted.item())
        except (TypeError, ValueError):
            pass

    raise TypeError(f"Unsupported artifact value type: {type(value).__name__}")
