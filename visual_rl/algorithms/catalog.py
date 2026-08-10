"""Import-safe catalog fragments owned by the algorithm domain.

This module contains descriptor strings and immutable declaration providers
only.  Registry state, alias conflict handling, implementation imports, and
component construction remain owned by composition/runtime.
"""

from __future__ import annotations

from visual_rl.algorithms.conditioning.config import CONDITIONING_CATALOG_FRAGMENT
from visual_rl.algorithms.dynamics.config import DYNAMICS_CATALOG_FRAGMENT
from visual_rl.algorithms.modules.declarations import (
    ALGORITHM_DECLARATION_PROVIDER_ABI,
)
from visual_rl.algorithms.optimization.config import CREDIT_CATALOG_FRAGMENT
from visual_rl.algorithms.rewards.config import REWARD_CATALOG_FRAGMENT
from visual_rl.algorithms.rollout.config import ROLLOUT_CATALOG_FRAGMENT
from visual_rl.algorithms.trainer.config import TRAINER_CATALOG_FRAGMENT
from visual_rl.core.contracts import CatalogFragment, ComponentDescriptor

__all__ = (
    "ALGORITHM_CATALOG_FRAGMENT",
    "ALGORITHM_DOMAIN_CATALOG_FRAGMENTS",
    "algorithm_catalog_fragment",
    "algorithm_domain_catalog_fragments",
)


ALGORITHM_CATALOG_FRAGMENT = CatalogFragment(
    owner="algorithms.modules",
    kind="algorithm",
    descriptors=(
        ComponentDescriptor(
            alias="flow-grpo",
            implementation_class_path=(
                "visual_rl.algorithms.modules.flow_grpo:FlowGRPOAlgorithmModule"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.modules.declarations:FlowGRPODeclarationProvider"
            ),
            declaration_provider_abi=ALGORITHM_DECLARATION_PROVIDER_ABI,
        ),
        ComponentDescriptor(
            alias="tempflow-grpo",
            implementation_class_path=(
                "visual_rl.algorithms.modules.tempflow_grpo:TempFlowGRPOAlgorithmModule"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.modules.declarations:"
                "TempFlowGRPODeclarationProvider"
            ),
            declaration_provider_abi=ALGORITHM_DECLARATION_PROVIDER_ABI,
        ),
        ComponentDescriptor(
            alias="flash-grpo",
            implementation_class_path=(
                "visual_rl.algorithms.modules.flash_grpo:FlashGRPOAlgorithmModule"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.modules.declarations:FlashGRPODeclarationProvider"
            ),
            declaration_provider_abi=ALGORITHM_DECLARATION_PROVIDER_ABI,
        ),
    ),
)

# The order follows core.contracts.COMPONENT_KINDS with kinds absent from this
# domain omitted.  Keeping this a value tuple prevents a hidden global Registry.
ALGORITHM_DOMAIN_CATALOG_FRAGMENTS = (
    ALGORITHM_CATALOG_FRAGMENT,
    TRAINER_CATALOG_FRAGMENT,
    DYNAMICS_CATALOG_FRAGMENT,
    ROLLOUT_CATALOG_FRAGMENT,
    REWARD_CATALOG_FRAGMENT,
    CONDITIONING_CATALOG_FRAGMENT,
    CREDIT_CATALOG_FRAGMENT,
)


def algorithm_catalog_fragment() -> CatalogFragment:
    """Return the public coarse-algorithm descriptor contribution."""

    return ALGORITHM_CATALOG_FRAGMENT


def algorithm_domain_catalog_fragments() -> tuple[CatalogFragment, ...]:
    """Return all immutable fragments owned by the algorithm domain."""

    return ALGORITHM_DOMAIN_CATALOG_FRAGMENTS
