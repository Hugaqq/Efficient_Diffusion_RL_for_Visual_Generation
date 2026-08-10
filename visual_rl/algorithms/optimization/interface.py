"""Import-safe runtime boundary for detached credit planning.

The optimization domain owns this interface.  Registries may point at it, but
canonical algorithm code never imports the transitional registry hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

__all__ = ("CreditComponent", "CreditPlanningPort")


class CreditPlanningPort(ABC):
    """Executable, model-independent credit planning boundary."""

    REQUIRED_RUNTIME_METHODS = ("plan",)

    @property
    @abstractmethod
    def advantage_epsilon(self) -> float:
        """Numerical stabilizer used by the shared advantage processor."""

        raise NotImplementedError

    @property
    @abstractmethod
    def advantage_std_domain(self) -> str:
        """Normalization domain consumed by the shared advantage processor."""

        raise NotImplementedError

    @abstractmethod
    def plan(
        self,
        *,
        trajectory: Any,
        advantage: Any,
        coefficient_mean_reducer: object | None = None,
    ) -> Any:
        """Build detached objective inputs without a current-policy graph."""

        raise NotImplementedError


class CreditComponent(CreditPlanningPort):
    """Registry-loadable credit component with one canonical runtime ABI."""

    INTERFACE_VERSION = "1.0"

    @classmethod
    @abstractmethod
    def describe(cls, config: object) -> object:
        """Return the exact import-safe contract implemented by ``config``."""

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> CreditComponent:
        """Construct only after static, environment, and artifact gates."""

        raise NotImplementedError
