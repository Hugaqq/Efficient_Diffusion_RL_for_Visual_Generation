"""Canonical runtime boundary for algorithm-owned reward components.

The interface is intentionally independent of registries, concrete reward
clients, NumPy, Torch, and runtime resource management.  Static declaration
can therefore validate an implementation type without importing its scoring
kernel or acquiring a physical reward resource.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

__all__ = ("RewardComponent",)


class RewardComponent(ABC):
    """Runtime construction boundary for one declared reward component.

    Static configuration and capability declaration belong exclusively to the
    import-safe declaration provider.  A runtime implementation may retain a
    transitional ``describe()`` method for the legacy resolver, but that
    method is deliberately not part of this canonical ABI.
    """

    INTERFACE_VERSION = "1.0"

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> RewardComponent:
        """Construct only after static, environment, and artifact gates."""

        raise NotImplementedError
