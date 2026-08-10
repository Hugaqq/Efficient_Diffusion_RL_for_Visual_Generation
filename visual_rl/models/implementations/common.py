"""Runtime artifact helpers shared by concrete model implementations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from visual_rl.core.contracts import ComputePrecision

__all__ = (
    "local_model_artifact",
    "runtime_model_loader",
    "runtime_precision",
)


def runtime_precision(runtime_context: Mapping[str, Any]) -> ComputePrecision:
    if not isinstance(runtime_context, Mapping):
        raise TypeError("runtime_context must be a mapping")
    value = runtime_context.get("precision")
    try:
        return ComputePrecision(value)
    except (TypeError, ValueError):
        raise ValueError(
            "runtime_context.precision must be fp32, fp16, or bf16"
        ) from None


def local_model_artifact(
    runtime_context: Mapping[str, Any],
    artifact_ref: str,
) -> Path:
    """Resolve one logical recipe artifact to a local directory only."""

    artifacts = runtime_context.get("model_artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError(
            "runtime_context.model_artifacts must map artifact refs to local paths"
        )
    if artifact_ref not in artifacts:
        raise ValueError(f"model artifact ref {artifact_ref!r} is not runtime-bound")
    raw = artifacts[artifact_ref]
    if not isinstance(raw, (str, Path)):
        raise TypeError("runtime-bound model artifact must be a local path")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"runtime-bound model artifact is not a directory: {path}")
    return path


def runtime_model_loader(runtime_context: Mapping[str, Any]) -> object | None:
    loader = runtime_context.get("model_loader")
    if loader is not None and not callable(loader):
        raise TypeError("runtime_context.model_loader must be callable")
    return loader
