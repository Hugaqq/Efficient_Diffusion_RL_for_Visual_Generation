"""Logical bind port for externally owned physical reward resources."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from visual_rl.core.contracts import RewardPlanSpec

__all__ = (
    "RewardResourceHandle",
    "RewardResourcePoolView",
    "RewardResourcePort",
    "RewardResourceState",
)


class RewardResourceState(str, Enum):
    """Lifecycle observed by logical ports and owned by the runtime pool."""

    DECLARED = "declared"
    ACQUIRED = "acquired"
    ACTIVE = "active"
    CLOSED = "closed"


@runtime_checkable
class RewardResourceHandle(Protocol):
    """Non-owning capability exposed by the runtime to one logical reward."""

    @property
    def resource_identity(self) -> str: ...

    @property
    def state(self) -> RewardResourceState: ...

    def require_method(self, name: str) -> None: ...

    def resource_for_execution(self) -> object: ...


@runtime_checkable
class RewardResourcePoolView(Protocol):
    """Read-only view used by reward execution without lifecycle ownership."""

    plan: RewardPlanSpec

    def handle(self, resource_identity: str) -> RewardResourceHandle: ...

    def get(self, resource_identity: str) -> object: ...


class RewardResourcePort:
    """One logical component's strict bind-once borrowed-resource port."""

    __slots__ = ("_closed", "_handle")

    def __init__(self) -> None:
        self._handle: RewardResourceHandle | None = None
        self._closed = False

    @property
    def state(self) -> RewardResourceState:
        if self._closed:
            return RewardResourceState.CLOSED
        if self._handle is None:
            return RewardResourceState.DECLARED
        return self._handle.state

    @property
    def resource_identity(self) -> str | None:
        return None if self._handle is None else self._handle.resource_identity

    def is_bound_to(self, handle: RewardResourceHandle) -> bool:
        if not isinstance(handle, RewardResourceHandle):
            raise TypeError("handle must implement RewardResourceHandle")
        return not self._closed and self._handle is handle

    def bind(
        self,
        handle: RewardResourceHandle,
        *,
        required_method: str,
    ) -> None:
        if self._closed:
            raise RuntimeError("closed reward resource port cannot bind")
        if self._handle is not None:
            raise RuntimeError("reward resource port is bind-once")
        if not isinstance(handle, RewardResourceHandle):
            raise TypeError("handle must implement RewardResourceHandle")
        if handle.state is not RewardResourceState.ACQUIRED:
            raise RuntimeError("reward resource binding requires an ACQUIRED handle")
        handle.require_method(required_method)
        self._handle = handle

    def resource_for_execution(self) -> object:
        if self._closed:
            raise RuntimeError("closed reward resource port cannot execute")
        if self._handle is None:
            raise RuntimeError("reward resource port is still DECLARED and unbound")
        return self._handle.resource_for_execution()

    def close(self) -> None:
        """Close only the logical port; the borrowed resource remains owned."""

        self._closed = True
