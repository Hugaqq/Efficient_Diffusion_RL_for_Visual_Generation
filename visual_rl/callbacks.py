"""Small, ordered lifecycle callbacks for a VisualRL run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class CallbackError(RuntimeError):
    """A lifecycle callback failed and stopped the run."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _readonly_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _freeze(values or {})


@dataclass(frozen=True)
class CallbackContext:
    """Read-only run state exposed to lifecycle callbacks.

    This deliberately contains run bookkeeping only. Training batches, rewards,
    gradients, models, and optimizers remain inside the runner.
    """

    run_id: str
    output_dir: Path
    global_step: int
    rank: int
    world_size: int
    metrics: Mapping[str, Any]
    artifacts: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _readonly_mapping(self.metrics))
        object.__setattr__(self, "artifacts", _readonly_mapping(self.artifacts))


class RunCallback:
    """Inherit and override any of the four stable lifecycle hooks."""

    def on_run_start(self, context: CallbackContext) -> None:
        pass

    def on_step_end(self, context: CallbackContext) -> None:
        pass

    def on_checkpoint(self, context: CallbackContext) -> None:
        pass

    def on_run_end(self, context: CallbackContext) -> None:
        pass


__all__ = ["CallbackContext", "CallbackError", "RunCallback"]
