"""Strict, explicit-device Qwen model loading for the World-R1 service."""

from __future__ import annotations

import os
from typing import Any


def load_qwen_model_on_cuda_device(
    *,
    model_class: Any,
    model_path: str | os.PathLike[str],
    device: Any,
    dtype: Any,
) -> Any:
    """Stream local safetensor shards directly onto one explicit CUDA device.

    ``device_map`` is deliberately a root-only mapping, not ``"auto"``.  It
    therefore keeps placement owned by the worker's logical CUDA-device
    selection while allowing Transformers/Accelerate to materialize each
    checkpoint shard directly at its destination.  A trailing ``model.to`` is
    intentionally absent because it would first stage the full model on CPU.
    """

    if getattr(device, "type", None) != "cuda":
        raise ValueError("Qwen scorer device must be an explicit CUDA device")
    if getattr(device, "index", None) is None:
        raise ValueError("Qwen scorer CUDA device must include a logical index")

    model = model_class.from_pretrained(
        os.fspath(model_path),
        dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_safetensors=True,
    )
    model.requires_grad_(False)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


__all__ = ("load_qwen_model_on_cuda_device",)
