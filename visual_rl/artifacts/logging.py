"""Best-effort terminal progress for the sole training loop."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visual_rl.runner import StepMetrics

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - minimal runtime
    tqdm = None


class TrainProgressPrinter:
    """No-throw terminal helper; it never persists training metrics."""

    def __init__(
        self,
        enabled: bool = True,
        interval: int = 1,
        leave: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.interval = max(1, int(interval))
        self.leave = bool(leave)
        self._bar: Any | None = None
        self._target_steps = 0
        self._last_step = 0

    def start(self, target_steps: int, *, initial_step: int) -> None:
        try:
            if (
                not self.enabled
                or self._bar is not None
                or tqdm is None
            ):
                return
            self._target_steps = max(0, int(target_steps))
            self._last_step = max(0, int(initial_step))
            self._bar = tqdm(
                total=self._target_steps,
                initial=self._last_step,
                desc="train",
                unit="step",
                dynamic_ncols=True,
                leave=self.leave,
                file=sys.stderr,
            )
        except Exception:
            self._bar = None

    def update(
        self,
        completed_steps: int,
        metrics: StepMetrics,
    ) -> None:
        try:
            if not self.enabled:
                return
            completed = int(completed_steps)
            values = dict(metrics.values)
            if self._bar is None:
                if (
                    completed == self._target_steps
                    or completed % self.interval == 0
                ):
                    line = _fallback_line(
                        values,
                        completed,
                        self._target_steps,
                    )
                    print(line, file=sys.stderr, flush=True)
                return
            delta = completed - self._last_step
            if delta > 0:
                self._bar.update(delta)
                self._last_step = completed
            if (
                completed == self._target_steps
                or completed % self.interval == 0
            ):
                self._bar.set_postfix(
                    _progress_metrics(values),
                    refresh=True,
                )
        except Exception:
            return

    def close(self) -> None:
        try:
            if self._bar is not None:
                self._bar.close()
        except Exception:
            pass
        finally:
            self._bar = None


def _format_metric(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _progress_metrics(values: dict[str, Any]) -> dict[str, str]:
    names = (
        "loss",
        "reward_mean",
        "reward_std",
        "approx_kl",
        "clipfrac",
        "logprob_delta_abs_max",
    )
    return {
        name: _format_metric(values[name]) for name in names if name in values
    }


def _fallback_line(
    values: dict[str, Any],
    completed_steps: int,
    target_steps: int,
) -> str:
    metrics = " ".join(
        f"{name}={value}"
        for name, value in _progress_metrics(values).items()
    )
    return (
        f"train step={completed_steps}/{target_steps} {metrics}"
    ).rstrip()
