"""Small explicit registry for adapters, algorithms, rewards, and backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, object] = {}

    def register(self, key: str, value: T | None = None) -> Callable[[T], T] | T:
        def _decorator(obj: T) -> T:
            if key in self._items:
                raise KeyError(f"{self.name} registry already has key {key!r}")
            self._items[key] = obj
            return obj

        if value is not None:
            return _decorator(value)
        return _decorator

    def get(self, key: str) -> object:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name} key {key!r}. Available: {available}") from exc

    def keys(self) -> list[str]:
        return sorted(self._items)


MODEL_ADAPTERS = Registry("model_adapter")
ALGORITHMS = Registry("algorithm")
REWARD_CLIENTS = Registry("reward_client")
FEEDBACK_PROVIDERS = Registry("feedback_provider")
OPTIMIZER_PLUGINS = Registry("optimizer_plugin")
ROLLOUT_ENGINES = Registry("rollout_engine")
