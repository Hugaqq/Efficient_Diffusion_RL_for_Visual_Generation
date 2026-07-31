"""Linux process supervision primitives for strict reward workers.

Gunicorn owns the service worker while each reward manager owns one or more
``multiprocessing`` children. A native extension may terminate the Gunicorn
worker before Python cleanup runs, so the master is made a child subreaper and
manager children arm a parent-death signal before importing model code.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import signal
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36


def _prctl(option: int, value: int) -> None:
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    if prctl(option, value, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def configure_child_subreaper() -> None:
    """Make the Gunicorn master reap descendants of an abruptly lost worker."""

    _prctl(_PR_SET_CHILD_SUBREAPER, 1)


def arm_parent_death_signal(expected_parent_pid: int) -> None:
    """Kill this process if its exact manager parent disappears."""

    if type(expected_parent_pid) is not int or expected_parent_pid <= 1:
        raise ValueError("expected_parent_pid must be an integer greater than one")
    _prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    if sys.platform == "linux" and os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def supervised_worker_entry(
    expected_parent_pid: int,
    module_name: str,
    function_name: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Arm supervision before importing and calling a native worker target."""

    arm_parent_death_signal(expected_parent_pid)
    module = importlib.import_module(module_name)
    target = getattr(module, function_name)
    if not callable(target):
        raise TypeError(
            f"supervised worker target {module_name}.{function_name} is not callable"
        )
    target(*tuple(args), **dict(kwargs or {}))


__all__ = (
    "arm_parent_death_signal",
    "configure_child_subreaper",
    "supervised_worker_entry",
)
