"""The sole deterministic-runtime setup switch."""

from __future__ import annotations

import os


def configure_deterministic_runtime(enabled: bool) -> None:
    """Configure Torch deterministic flags before model construction."""

    if type(enabled) is not bool:
        raise TypeError("enabled must be a bool")
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(enabled, warn_only=False)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = not enabled
    torch.backends.cudnn.allow_tf32 = not enabled
    if enabled:
        torch.set_float32_matmul_precision("highest")
