"""Import-safe decoded-media values crossing the model/data boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ("DecodedMediaBatch", "DecodedMediaLayout")


DecodedMediaLayout = Literal["BCHW", "BFCHW", "BFHWC"]

_RANK_BY_LAYOUT: dict[str, int] = {
    "BCHW": 4,
    "BFCHW": 5,
    "BFHWC": 5,
}


def _tensor_shape(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("decoded media tensor must expose a shape")
    try:
        result = tuple(shape)
    except TypeError as exc:
        raise TypeError("decoded media tensor shape must be iterable") from exc
    if not result or any(type(dimension) is not int for dimension in result):
        raise TypeError("decoded media tensor shape must contain integer dimensions")
    if any(dimension < 1 for dimension in result):
        raise ValueError("decoded media tensor dimensions must be positive")
    return result


@dataclass(frozen=True, slots=True)
class DecodedMediaBatch:
    """One opaque decoded tensor paired with its model-owned batch layout.

    The data contract deliberately does not import a tensor framework.  Model
    adapters own the concrete tensor type; consumers may enforce device, dtype,
    gradient, and finite-value requirements at their runtime boundary.
    """

    tensor: object = field(repr=False, compare=False)
    layout: DecodedMediaLayout
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        expected_rank = _RANK_BY_LAYOUT.get(self.layout)
        if expected_rank is None:
            raise ValueError("decoded media layout must be BCHW, BFCHW, or BFHWC")
        shape = _tensor_shape(self.tensor)
        if len(shape) != expected_rank:
            raise ValueError(
                f"decoded media layout {self.layout} requires rank {expected_rank}"
            )
        object.__setattr__(self, "shape", shape)

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def rank(self) -> int:
        return len(self.shape)

    def assert_integrity(self) -> None:
        """Reject an opaque tensor whose shape changed after construction."""

        if _tensor_shape(self.tensor) != self.shape:
            raise ValueError("decoded media tensor shape changed after construction")
