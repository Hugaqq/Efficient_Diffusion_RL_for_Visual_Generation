"""Frozen descriptions for VisualRL's four builtin component kinds.

This module deliberately contains no builtin factories, manifest tuple, lookup
function, or mutable registration state.  The one manifest and name lookup live
in :mod:`visual_rl.builtins`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CAPABILITY_OWNER",
    "CAPABILITY_VOCABULARY",
    "COMPONENT_KINDS",
    "ComponentKind",
    "ComponentSpec",
]

ComponentKind = Literal["model", "rollout", "reward", "algorithm"]

COMPONENT_KINDS: tuple[ComponentKind, ...] = (
    "model",
    "rollout",
    "reward",
    "algorithm",
)

CAPABILITY_VOCABULARY = frozenset(
    {
        "media.image",
        "media.video",
        "sampling.full_trajectory",
        "sampling.single_step",
        "sampling.branching",
        "rollout.full_trajectory",
        "rollout.single_step",
        "rollout.branching",
        "policy.reference_stats",
        "conditioning.camera",
    }
)

CAPABILITY_OWNER: dict[str, ComponentKind] = {
    "media.image": "model",
    "media.video": "model",
    "sampling.full_trajectory": "model",
    "sampling.single_step": "model",
    "sampling.branching": "model",
    "rollout.full_trajectory": "rollout",
    "rollout.single_step": "rollout",
    "rollout.branching": "rollout",
    "policy.reference_stats": "model",
    "conditioning.camera": "model",
}


@dataclass(frozen=True)
class ComponentSpec:
    """An immutable description of one repository-owned component."""

    kind: ComponentKind
    name: str
    factory: type
    provides: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in COMPONENT_KINDS:
            raise ValueError(f"unsupported component kind: {self.kind!r}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("component name must be a non-empty string")
        if not isinstance(self.factory, type):
            raise TypeError("component factory must be a concrete class")
        if not isinstance(self.provides, frozenset):
            raise TypeError("provides must be a frozenset")
        if not isinstance(self.requires, frozenset):
            raise TypeError("requires must be a frozenset")
        if not isinstance(self.dependencies, tuple):
            raise TypeError("dependencies must be a tuple")
        unknown = (self.provides | self.requires) - CAPABILITY_VOCABULARY
        if unknown:
            raise ValueError(f"unknown component capabilities: {sorted(unknown)}")
        wrong_owner = sorted(
            capability
            for capability in self.provides
            if CAPABILITY_OWNER[capability] != self.kind
        )
        if wrong_owner:
            raise ValueError(
                f"{self.kind} cannot provide capabilities owned by another kind: "
                f"{wrong_owner}"
            )
