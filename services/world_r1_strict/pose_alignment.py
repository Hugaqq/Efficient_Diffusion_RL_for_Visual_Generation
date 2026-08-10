"""Failure policy for World-R1 camera-trajectory alignment.

The 3D reward is fail-closed for implementation and model failures.  One
geometric condition is different: Umeyama alignment is undefined when the
predicted camera path has degenerate covariance.  That is a valid (but poor)
model output, so it receives zero camera-motion credit instead of poisoning
the reward worker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def is_degenerate_umeyama_error(exc: BaseException) -> bool:
    """Return whether *exc* is evo's known degenerate-path alignment error."""

    error_type = type(exc)
    return (
        error_type.__module__ == "evo.core.geometry"
        and error_type.__name__ == "GeometryException"
        and "Degenerate covariance rank" in str(exc)
    )


def align_camera_extrinsics(
    align_fn: Callable[..., tuple[Any, Any, Any, Any]],
    target_extrinsics: Any,
    predicted_extrinsics: Any,
    *,
    ransac: bool,
) -> tuple[Any, bool]:
    """Align predicted poses, marking only known geometric degeneracy.

    Returns ``(extrinsics, is_degenerate)``.  All errors other than evo's
    explicit degenerate-covariance condition are re-raised for the caller's
    fail-closed handling.
    """

    try:
        _, _, _, aligned_extrinsics = align_fn(
            target_extrinsics,
            predicted_extrinsics,
            return_aligned=True,
            ransac=ransac,
        )
    except Exception as exc:
        if not is_degenerate_umeyama_error(exc):
            raise
        return predicted_extrinsics, True
    return aligned_extrinsics, False
