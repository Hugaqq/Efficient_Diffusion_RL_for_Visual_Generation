"""Composition-owned capability resolution failures."""

from __future__ import annotations

from visual_rl.core.contracts.composition import CapabilityMismatch

__all__ = ("ModelAlgorithmMismatch",)


class ModelAlgorithmMismatch(ValueError):
    """Fail-closed result of model/algorithm capability unification."""

    def __init__(self, mismatches: tuple[CapabilityMismatch, ...]) -> None:
        if type(mismatches) is not tuple or not mismatches:
            raise ValueError("mismatches must be a non-empty tuple")
        if any(not isinstance(item, CapabilityMismatch) for item in mismatches):
            raise TypeError("mismatches must contain CapabilityMismatch values")
        self.mismatches = mismatches
        summary = ", ".join(item.code for item in mismatches)
        super().__init__(f"model/algorithm binding is incompatible: {summary}")
