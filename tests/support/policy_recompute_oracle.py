"""Full-grid recompute oracle assembled only from canonical slot APIs.

The production runtime intentionally cannot build all ``T`` current-policy
graphs at once.  A few numerical parity tests need a monolithic reference;
this helper makes that memory-unbounded behavior visibly test-only while
reusing the exact production replay implementation.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from visual_rl.algorithms.optimization.recompute import (
    PolicyRecomputer,
    PolicyRecomputeRequest,
    PolicyStats,
)
from visual_rl.algorithms.optimization.slots import UpdateSlot


def _execution_context(factory: Any, *, name: str) -> Any:
    context = nullcontext() if factory is None else factory()
    if not callable(getattr(context, "__enter__", None)) or not callable(
        getattr(context, "__exit__", None)
    ):
        raise TypeError(f"{name} must return a context manager")
    return context


def compute_full_policy_stats_oracle(
    request: PolicyRecomputeRequest,
    *,
    recomputer: PolicyRecomputer | None = None,
) -> PolicyStats:
    """Materialize one full ``[B,T]`` graph for tests, never production."""

    if not isinstance(request, PolicyRecomputeRequest):
        raise TypeError("request must be a PolicyRecomputeRequest")
    if recomputer is None:
        recomputer = PolicyRecomputer()
    if not isinstance(recomputer, PolicyRecomputer):
        raise TypeError("recomputer must be a PolicyRecomputer")
    trajectory = request.rollout.trajectory
    active_count = int(trajectory.transition_mask.sum().item())
    slot = UpdateSlot(
        slot_index=0,
        row_indices=tuple(range(trajectory.batch_size)),
        transition_start=0,
        transition_stop=trajectory.transition_count,
        active_count=active_count,
        global_active_count=active_count,
    )

    reference_stats = None
    if request.require_reference_statistics:
        with _execution_context(
            request.reference_context,
            name="reference_context",
        ):
            reference_stats = recomputer.compute_reference_slot(request, slot)
    with _execution_context(request.current_context, name="current_context"):
        result = recomputer.compute_current_slot(
            request,
            slot,
            reference_stats=reference_stats,
        )
    result.validate_against_trajectory(trajectory)
    return result


__all__ = ("compute_full_policy_stats_oracle",)
