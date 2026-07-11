"""Simple JSONL metric logging."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal editable envs
    tqdm = None


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class TrainProgressPrinter:
    """Small tqdm wrapper for the single VisualRL training loop."""

    def __init__(self, enabled: bool = True, interval: int = 1, leave: bool = False):
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.leave = leave
        self._bar = None
        self._last_step = 0

    def start(self, total_steps: int, initial_step: int = 0) -> None:
        if not self.enabled or self._bar is not None:
            return
        if tqdm is None:
            return
        self._bar = tqdm(
            total=max(0, int(total_steps)),
            initial=max(0, int(initial_step)),
            desc="train",
            unit="step",
            dynamic_ncols=True,
            leave=self.leave,
            file=sys.stderr,
        )
        self._last_step = max(0, int(initial_step))

    def log_step(self, payload: dict[str, Any], total_steps: int) -> None:
        if not self.enabled:
            return
        step = int(payload.get("step", 0)) + 1
        self.start(total_steps)
        if self._bar is None:
            if step == total_steps or step % self.interval == 0:
                print(_fallback_progress_line(payload, step, total_steps), file=sys.stderr, flush=True)
            return

        delta = step - self._last_step
        if delta > 0:
            self._bar.update(delta)
            self._last_step = step
        if step == total_steps or step % self.interval == 0:
            self._bar.set_postfix(_progress_metrics(payload), refresh=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _format_metric(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _progress_metrics(payload: dict[str, Any]) -> dict[str, str]:
    keys = (
        "loss",
        "reward_mean",
        "reward_std",
        "approx_kl",
        "clipfrac",
        "logprob_delta_abs_max",
    )
    return {key: _format_metric(payload[key]) for key in keys if key in payload}


def _fallback_progress_line(payload: dict[str, Any], step: int, total_steps: int) -> str:
    metrics = " ".join(f"{key}={value}" for key, value in _progress_metrics(payload).items())
    return f"train step={step}/{total_steps} {metrics}".rstrip()
