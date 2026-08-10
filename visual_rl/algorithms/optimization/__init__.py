"""Import-safe optimization declarations.

Runtime optimization implementations are intentionally absent from this
surface until the M2 production identity release cut.
"""

from visual_rl.algorithms.optimization.config import (
    CREDIT_CATALOG_FRAGMENT,
    FlashCreditConfig,
    FlashCreditDeclarationProvider,
    GRPOCreditConfig,
    GRPOCreditDeclarationProvider,
    TempFlowCreditConfig,
    TempFlowCreditDeclarationProvider,
    credit_catalog_fragment,
)

__all__ = (
    "CREDIT_CATALOG_FRAGMENT",
    "FlashCreditConfig",
    "FlashCreditDeclarationProvider",
    "GRPOCreditConfig",
    "GRPOCreditDeclarationProvider",
    "TempFlowCreditConfig",
    "TempFlowCreditDeclarationProvider",
    "credit_catalog_fragment",
)
