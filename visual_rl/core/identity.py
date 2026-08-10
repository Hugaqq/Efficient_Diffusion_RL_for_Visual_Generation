"""Canonical, standard-library-only identity projection for core contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum

__all__ = ("canonical_identity", "to_identity_value")


def to_identity_value(value: object) -> object:
    """Project an immutable contract value to a canonical JSON value."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_identity_value(getattr(value, item.name))
            for item in fields(value)
            if item.init
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): to_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [to_identity_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"identity value is not serializable: {type(value).__name__}")


def canonical_identity(namespace: str, value: object) -> str:
    """Return a stable namespaced SHA-256 identity for one core value."""

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("identity namespace must be non-empty")
    encoded = json.dumps(
        {"namespace": namespace, "value": to_identity_value(value)},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{namespace}:{hashlib.sha256(encoded).hexdigest()}"
