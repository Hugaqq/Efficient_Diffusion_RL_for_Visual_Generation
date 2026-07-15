"""Validated runtime controls for exact single-process reproducibility."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import os
import platform
import sys
from typing import Any

import numpy as np


_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_NVIDIA_TF32_OVERRIDE = "0"
_RUNTIME_ASSERT_KEYS = (
    "pythonhashseed",
    "cublas_workspace_config",
    "nvidia_tf32_override",
    "deterministic_algorithms",
    "deterministic_algorithms_warn_only",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "matmul_allow_tf32",
    "cudnn_allow_tf32",
    "float32_matmul_precision",
)


def configure_runtime(*, enabled: bool, seed: int) -> dict[str, Any]:
    """Configure and report the runtime used by checkpoint identity checks.

    Python hash randomization is fixed at interpreter startup, so deterministic
    mode refuses to pretend that setting ``PYTHONHASHSEED`` inside this process
    would be sufficient.
    """

    expected_hash_seed = str(int(seed))
    if enabled and os.environ.get("PYTHONHASHSEED") != expected_hash_seed:
        raise RuntimeError(
            "runner.deterministic_runtime requires PYTHONHASHSEED to be set "
            "before process start: "
            f"PYTHONHASHSEED={expected_hash_seed}"
        )

    original_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    original_tf32_override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    torch_already_loaded = "torch" in sys.modules
    cuda_already_initialized = bool(
        torch_already_loaded
        and getattr(sys.modules["torch"], "cuda", None) is not None
        and sys.modules["torch"].cuda.is_initialized()
    )

    if enabled:
        _require_compatible_env(
            "CUBLAS_WORKSPACE_CONFIG",
            original_cublas,
            _CUBLAS_WORKSPACE_CONFIG,
        )
        _require_compatible_env(
            "NVIDIA_TF32_OVERRIDE",
            original_tf32_override,
            _NVIDIA_TF32_OVERRIDE,
        )
        if cuda_already_initialized and (
            original_cublas != _CUBLAS_WORKSPACE_CONFIG
            or original_tf32_override != _NVIDIA_TF32_OVERRIDE
        ):
            raise RuntimeError(
                "deterministic runtime must be configured before CUDA is "
                "initialized"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
        os.environ["NVIDIA_TF32_OVERRIDE"] = _NVIDIA_TF32_OVERRIDE

    import torch

    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    return runtime_snapshot(enabled=enabled, seed=seed)


def runtime_snapshot(*, enabled: bool, seed: int) -> dict[str, Any]:
    """Read the current runtime identity without changing process state."""

    import torch

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": [int(capability[0]), int(capability[1])],
                }
            )

    return {
        "enabled": bool(enabled),
        "seed": int(seed),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch": str(torch.__version__),
        "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "numpy": str(np.__version__),
        "packages": {
            name: _installed_version(name)
            for name in ("diffusers", "peft", "transformers")
        },
        "cuda_devices": cuda_devices,
    }


def assert_runtime(
    expected: dict[str, Any],
    *,
    context: str,
) -> None:
    """Fail when deterministic torch flags or environment variables drift."""

    if not bool(expected.get("enabled")):
        return
    current = runtime_snapshot(
        enabled=True,
        seed=int(expected["seed"]),
    )
    differences = []
    for key in _RUNTIME_ASSERT_KEYS:
        if key not in expected or current.get(key) != expected.get(key):
            differences.append(
                f"{key}: expected {expected.get(key)!r}, "
                f"actual {current.get(key)!r}"
            )
    if differences:
        raise RuntimeError(
            f"Deterministic runtime drift detected {context}: "
            + "; ".join(differences)
        )


def _require_compatible_env(
    name: str,
    current: str | None,
    required: str,
) -> None:
    if current not in {None, required}:
        raise RuntimeError(
            f"{name}={current!r} conflicts with deterministic runtime; "
            f"expected {required!r}"
        )


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None
