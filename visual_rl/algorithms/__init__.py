"""Import-safe algorithm domain root.

During M2.1 the existing production exports remain available through lazy
resolution so importing :mod:`visual_rl.algorithms.catalog` does not first
load credit, objective, Torch, or any runtime implementation.  The legacy
export map is removed when the canonical M2 paths become the production
surface.
"""

from __future__ import annotations

import importlib

_LEGACY_EXPORTS = {
    "ClippedSurrogateObjective": (
        "visual_rl.algorithms.optimization.objective",
        "ClippedSurrogateObjective",
    ),
    "CreditStrategy": (
        "visual_rl.algorithms.optimization.credit",
        "CreditStrategy",
    ),
    "FlashGRPOCreditStrategy": (
        "visual_rl.algorithms.optimization.credit",
        "FlashGRPOCreditStrategy",
    ),
    "FlashCreditConfig": (
        "visual_rl.algorithms.optimization.config",
        "FlashCreditConfig",
    ),
    "GRPOCreditStrategy": (
        "visual_rl.algorithms.optimization.credit",
        "GRPOCreditStrategy",
    ),
    "GRPOCreditConfig": (
        "visual_rl.algorithms.optimization.config",
        "GRPOCreditConfig",
    ),
    "LossOutput": ("visual_rl.algorithms.optimization.objective", "LossOutput"),
    "PolicyLossInputs": (
        "visual_rl.algorithms.optimization.objective",
        "PolicyLossInputs",
    ),
    "PolicyStats": ("visual_rl.algorithms.optimization.recompute", "PolicyStats"),
    "RegisteredFlashCredit": (
        "visual_rl.algorithms.optimization.credit",
        "RegisteredFlashCredit",
    ),
    "RegisteredGRPOCredit": (
        "visual_rl.algorithms.optimization.credit",
        "RegisteredGRPOCredit",
    ),
    "RegisteredTempFlowCredit": (
        "visual_rl.algorithms.optimization.credit",
        "RegisteredTempFlowCredit",
    ),
    "TempFlowGRPOCreditStrategy": (
        "visual_rl.algorithms.optimization.credit",
        "TempFlowGRPOCreditStrategy",
    ),
    "TempFlowCreditConfig": (
        "visual_rl.algorithms.optimization.config",
        "TempFlowCreditConfig",
    ),
}

__all__ = tuple(sorted(_LEGACY_EXPORTS))


def __getattr__(name: str) -> object:
    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_LEGACY_EXPORTS))
