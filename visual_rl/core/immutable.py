"""Pickle-safe immutable containers shared by import-safe contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ("FrozenMapping",)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("FrozenMapping does not accept non-finite floats")
    if isinstance(value, (str, int, float, bool, type(None), Path)):
        return value
    raise TypeError(
        "FrozenMapping values must be JSON-safe scalars/containers or Path, "
        f"got {type(value).__name__}"
    )


class FrozenMapping(Mapping):
    """Recursively copied, pickle-safe, read-only string mapping."""

    __slots__ = ("_items",)

    def __init__(self, source: Any = ()) -> None:
        raw_items = source.items() if isinstance(source, Mapping) else source
        items = []
        keys = set()
        for key, value in raw_items:
            if not isinstance(key, str):
                raise TypeError("FrozenMapping keys must be strings")
            if key in keys:
                raise ValueError(f"FrozenMapping contains duplicate key {key!r}")
            keys.add(key)
            items.append((key, _freeze_value(value)))
        object.__setattr__(self, "_items", tuple(items))

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("FrozenMapping is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("FrozenMapping is immutable")

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def __hash__(self) -> int:
        return hash(frozenset(self._items))

    def __reduce__(self):
        return type(self), (self._items,)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._items)!r})"
