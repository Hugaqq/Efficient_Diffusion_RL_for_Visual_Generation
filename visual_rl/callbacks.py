"""Torch-free public contracts for VisualRL's minimal read-only callbacks."""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from visual_rl.core.types import FrozenMapping

__all__ = ("Callback", "CallbackEvent")


_CallbackKind = Literal["run_start", "step_end", "commit", "run_end"]
_CallbackHook = Literal[
    "on_run_start",
    "on_step_end",
    "on_commit",
    "on_run_end",
]
_CALLBACK_KINDS = frozenset(("run_start", "step_end", "commit", "run_end"))
_HOOK_KIND = {
    "on_run_start": "run_start",
    "on_step_end": "step_end",
    "on_commit": "commit",
    "on_run_end": "run_end",
}
_METRIC_INTEGER_KEYS = frozenset(("step", "sample_count", "active_transition_count"))
_ARTIFACT_KEYS = frozenset(
    (
        "authoritative_checkpoint",
        "resolved_config_path",
        "manifest_path",
        "metrics_path",
        "marker_path",
    )
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallbackEvent:
    """One immutable, tensor-free observation of the shared Runner lifecycle."""

    kind: _CallbackKind
    run_id: str
    output_dir: Path
    step: int | None
    target_steps: int
    committed_steps: int
    metrics: FrozenMapping
    artifacts: FrozenMapping

    def __post_init__(self) -> None:
        if self.kind not in _CALLBACK_KINDS:
            raise ValueError(f"unknown callback event kind {self.kind!r}")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        if not self.output_dir.is_absolute():
            raise ValueError("output_dir must be absolute")
        if not self.output_dir.is_dir():
            raise ValueError("output_dir must be an existing directory")
        if type(self.target_steps) is not int or self.target_steps <= 0:
            raise ValueError("target_steps must be a positive integer")
        if (
            type(self.committed_steps) is not int
            or not 0 <= self.committed_steps <= self.target_steps
        ):
            raise ValueError("committed_steps must be an integer in [0, target_steps]")

        metrics = _frozen_mapping("metrics", self.metrics)
        artifacts = _frozen_mapping("artifacts", self.artifacts)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "artifacts", artifacts)

        if self.kind == "run_start":
            if self.step is not None:
                raise ValueError("run_start step must be None")
            if metrics:
                raise ValueError("run_start metrics must be empty")
            if self.committed_steps == 0:
                if artifacts:
                    raise ValueError("fresh run_start artifacts must be empty")
            else:
                _validate_artifacts(artifacts, self.output_dir)
            return

        if type(self.step) is not int or not 0 <= self.step < self.target_steps:
            raise ValueError(
                "non-start callback step must be an integer in [0, target_steps)"
            )
        _validate_metrics(metrics, expected_step=self.step)

        if self.kind == "step_end":
            if self.committed_steps > self.step + 1:
                raise ValueError("step_end committed_steps cannot exceed step + 1")
            if artifacts:
                raise ValueError("step_end artifacts must be empty")
            return

        if self.committed_steps != self.step + 1:
            raise ValueError(f"{self.kind} committed_steps must equal step + 1")
        _validate_artifacts(artifacts, self.output_dir)


class Callback:
    """Read-only observer base class; override only the hooks you need."""

    def on_run_start(self, event: CallbackEvent) -> None:
        del event

    def on_step_end(self, event: CallbackEvent) -> None:
        del event

    def on_commit(self, event: CallbackEvent) -> None:
        del event

    def on_run_end(self, event: CallbackEvent) -> None:
        del event


def _normalize_callbacks(callbacks: object) -> tuple[Callback, ...]:
    if isinstance(callbacks, (str, bytes, bytearray)) or not isinstance(
        callbacks, Sequence
    ):
        raise TypeError("callbacks must be a non-string sequence of Callback instances")
    normalized = tuple(callbacks)
    for index, callback in enumerate(normalized):
        if not isinstance(callback, Callback):
            raise TypeError(
                f"callbacks[{index}] must be a constructed Callback instance"
            )
    return normalized


def _dispatch_callbacks(
    callbacks: tuple[Callback, ...],
    hook: _CallbackHook,
    event: CallbackEvent,
) -> None:
    if type(callbacks) is not tuple or any(
        not isinstance(callback, Callback) for callback in callbacks
    ):
        raise TypeError("Runner callbacks must be a tuple of Callback instances")
    expected_kind = _HOOK_KIND.get(hook)
    if expected_kind is None:
        raise ValueError(f"unknown callback hook {hook!r}")
    if not isinstance(event, CallbackEvent):
        raise TypeError("callback event must be CallbackEvent")
    if event.kind != expected_kind:
        raise ValueError(
            f"callback hook {hook!r} cannot receive event kind {event.kind!r}"
        )
    if not callbacks:
        return

    with _preserve_rng_state():
        for index, callback in enumerate(callbacks):
            try:
                result = getattr(callback, hook)(event)
            except Exception as exc:  # callback failures are deliberately fail-open
                _LOGGER.warning(
                    "callback failure index=%d class=%s hook=%s exception=%s",
                    index,
                    type(callback).__qualname__,
                    hook,
                    type(exc).__name__,
                    exc_info=True,
                )
                continue
            if result is not None:
                _LOGGER.warning(
                    "callback misuse index=%d class=%s hook=%s return_type=%s; "
                    "return values are ignored",
                    index,
                    type(callback).__qualname__,
                    hook,
                    type(result).__qualname__,
                )


def _frozen_mapping(name: str, value: Any) -> FrozenMapping:
    if isinstance(value, FrozenMapping):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return FrozenMapping(value)


def _validate_metrics(metrics: FrozenMapping, *, expected_step: int) -> None:
    missing = _METRIC_INTEGER_KEYS.difference(metrics)
    if missing:
        raise ValueError(
            f"callback metrics are missing required keys: {sorted(missing)}"
        )
    for key, value in metrics.items():
        if not key:
            raise ValueError("callback metric keys must be non-empty strings")
        if key in _METRIC_INTEGER_KEYS:
            if type(value) is not int:
                raise TypeError(f"callback metric {key!r} must be an integer")
            if key == "step" and value != expected_step:
                raise ValueError("callback metric step does not match event step")
            if key != "step" and value <= 0:
                raise ValueError(f"callback metric {key!r} must be positive")
            continue
        if type(value) is not float or not math.isfinite(value):
            raise TypeError(f"callback metric {key!r} must be a finite Python float")


def _validate_artifacts(
    artifacts: FrozenMapping,
    output_dir: Path,
) -> None:
    if set(artifacts) != _ARTIFACT_KEYS:
        raise ValueError(
            "callback artifacts must contain exactly the authoritative path keys"
        )
    for key, value in artifacts.items():
        if not isinstance(value, Path):
            raise TypeError(f"callback artifact {key!r} must be a pathlib.Path")
        if not value.is_absolute():
            raise ValueError(f"callback artifact {key!r} must be absolute")
        if value != output_dir and output_dir not in value.parents:
            raise ValueError(
                f"callback artifact {key!r} must be located inside output_dir"
            )
        if not value.exists():
            raise ValueError(f"callback artifact {key!r} must exist")


@contextmanager
def _preserve_rng_state():
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state().clone()
    torch_cuda_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_initialized()
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu_state)
        if torch_cuda_states is not None:
            torch.cuda.set_rng_state_all(list(torch_cuda_states))
