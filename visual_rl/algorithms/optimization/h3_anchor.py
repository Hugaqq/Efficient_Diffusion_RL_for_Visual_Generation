"""One-shot rank-local CPS replay anchor, not a rollout-score alias.

The caller must finish credit assignment, keep the model in EVAL with fixed
behavior parameters, and provide a version that changes on every policy update.
This owner neither enforces a world barrier nor owns a model/optimizer/pool.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics, H3CPSRecord

if TYPE_CHECKING:
    import torch


def _identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


class H3ReplayAnchor:
    """One attempt per collection/behavior, with atomic publication of scores.

    Expected sample IDs are explicit, so even omission of an entire trajectory
    is rejected. IDs are not prompt text and are never grouped by prompt.
    Only keys, the immutable numerical contract, and small CPU scores survive
    capture. Failed owners cannot be retried; public score reads are copies.
    """

    def __init__(
        self, *, collection_id: str, behavior_id: str,
        sample_ids: tuple[str, ...], dynamics: H3CPSDynamics,
        max_cells: int, max_trajectory_bytes: int,
    ) -> None:
        _identifier("collection_id", collection_id)
        _identifier("behavior_id", behavior_id)
        if type(sample_ids) is not tuple or not sample_ids:
            raise ValueError("sample_ids must be a nonempty tuple")
        for sample_id in sample_ids:
            _identifier("sample_id", sample_id)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        if not isinstance(dynamics, H3CPSDynamics):
            raise TypeError("dynamics must be H3CPSDynamics")
        for name, value in (("max_cells", max_cells), ("max_trajectory_bytes", max_trajectory_bytes)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._collection_id = collection_id
        self._behavior_id = behavior_id
        self._sample_ids = sample_ids
        self._dynamics = dynamics
        self._max_cells = max_cells
        self._max_trajectory_bytes = max_trajectory_bytes
        self._attempted = False
        self._scores: dict[tuple[str, int], torch.Tensor] | None = None

    def _identity(self, collection_id: str, behavior_id: str) -> None:
        if collection_id != self._collection_id or behavior_id != self._behavior_id:
            raise ValueError("collection/behavior identity mismatch")

    def _preflight(self, records: tuple[tuple[str, H3CPSRecord], ...]) -> None:
        if type(records) is not tuple or not records:
            raise ValueError("records must be a nonempty ordered tuple")
        if len(records) > self._max_cells:
            raise ValueError("collection exceeds max_cells")
        # B1 FP16 x + action, plus the FP32 rollout diagnostic. The schedule is
        # immutable shared metadata, not trajectory tensor payload. Check this
        # exact admitted payload bound before reading even one tensor copy.
        cell_bytes = 4 * self._dynamics.layout.packed_width + 4
        if len(records) * cell_bytes > self._max_trajectory_bytes:
            raise ValueError("collection exceeds max_trajectory_bytes")
        steps = self._dynamics.schedule.num_steps
        if len(records) != len(self._sample_ids) * steps:
            raise ValueError("incomplete collection coverage")
        sample_ids = set(self._sample_ids)
        seen = set()
        for item in records:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("each cell must be (sample_id, H3CPSRecord)")
            sample_id, record = item
            _identifier("sample_id", sample_id)
            if sample_id not in sample_ids:
                raise ValueError("unexpected sample_id")
            if type(record) is not H3CPSRecord:
                raise TypeError("each cell requires H3CPSRecord")
            if (record.dynamics != self._dynamics
                    or record.score_family != "cps_proxy_negative_mse"
                    or record.quantization != "fp16"):
                raise ValueError("record numerical contract mismatch")
            self._dynamics.schedule.transition(record.index)
            key = (sample_id, record.index)
            if key in seen:
                raise ValueError("duplicate sample/index cell")
            seen.add(key)
            # H3CPSRecord already validates/owns x, action and diagnostic with
            # matching batch/device. Read only its public copying interface.
            x = record.x
            if x.shape[0] != 1 or x.device.type != "cpu":
                raise ValueError("input records must be CPU B1")
            del x
        # Exact size, unique keys, known IDs and bounded indices imply complete
        # sample x transition coverage, including the last q=0 transition.

    def capture(
        self, records: tuple[tuple[str, H3CPSRecord], ...], *,
        collection_id: str, behavior_id: str, device: torch.device | str,
        predict: Callable[[H3CPSRecord], torch.Tensor],
        read_behavior_id: Callable[[], str],
        status_hook: Callable[..., None] | None = None,
    ) -> None:
        """Replay every cell in the supplied order; publish only after success.

        Batch metadata labels all input cells with collection/behavior identity;
        the future pool is responsible for that provenance. The callback must
        not retain device records. No input records or callbacks are retained
        here, and any failure (including preflight) consumes this owner.
        """
        import torch

        if self._attempted:
            raise RuntimeError("anchor capture already attempted")
        self._attempted = True
        self._identity(collection_id, behavior_id)
        self._preflight(records)
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("capture device must be CPU or CUDA")
        if not callable(predict) or not callable(read_behavior_id):
            raise TypeError("predict and read_behavior_id must be callable")
        pending = {}
        with torch.no_grad():
            if read_behavior_id() != self._behavior_id:
                raise ValueError("behavior version changed before capture")
            for ordinal, (sample_id, record) in enumerate(records):
                device_record = H3CPSRecord(
                    record.dynamics, record.index, record.x.to(device),
                    record.action.to(device), record.rollout_score.to(device),
                )
                if status_hook is not None:
                    status_hook(f"cell/{ordinal}/forward", local_error=False)
                # Do not add collectives after a raw forward/AC failure.
                prediction = predict(device_record)
                try:
                    score = self._dynamics.evaluate(device_record, prediction)
                    if score.shape != (1,) or score.dtype != torch.float32 or not bool(torch.isfinite(score).all()):
                        raise ValueError("replay score must be finite FP32 [1]")
                    pending[(sample_id, record.index)] = score.detach().cpu().clone()
                except Exception:
                    if status_hook is not None:
                        status_hook(f"cell/{ordinal}/result", local_error=True)
                    raise
                if status_hook is not None:
                    status_hook(f"cell/{ordinal}/result", local_error=False)
                # Free each temporary record before allocating the next one.
                del device_record, prediction, score
            if read_behavior_id() != self._behavior_id:
                raise ValueError("behavior version changed during capture")
        self._scores = pending

    def score(
        self, *, collection_id: str, behavior_id: str, sample_id: str, index: int,
    ) -> torch.Tensor:
        """Return the frozen detached CPU FP32 [1] score for one exact key."""
        self._identity(collection_id, behavior_id)
        _identifier("sample_id", sample_id)
        self._dynamics.schedule.transition(index)
        if self._scores is None:
            raise RuntimeError("anchor has no complete captured scores")
        return self._scores[(sample_id, index)].clone()
