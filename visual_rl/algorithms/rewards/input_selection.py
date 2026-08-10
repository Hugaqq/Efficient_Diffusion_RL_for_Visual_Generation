"""Typed, replayable reward-input selection policies.

Input selection is part of reward semantics, not an implementation detail of a
transport client.  The first contract is intentionally narrow: select one
source video frame for an entire homogeneous reward invocation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal

from visual_rl.core.types import StepContext

__all__ = (
    "RewardInputSelection",
    "RewardInputSelectionPolicy",
)


_IDENTITY_DOMAIN = b"visual-rl.reward-input-selection-policy.v1\0"
_KEY_DOMAIN = b"visual-rl.reward-input-selection-key.v1\0"
_DRAW_DOMAIN = b"visual-rl.reward-input-selection-draw.v1\0"


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RewardInputSelection:
    """One immutable selection decision and its replay identity."""

    frame_count: int
    selected_frame_index: int
    policy_id: str
    selection_key_id: str

    def __post_init__(self) -> None:
        if type(self.frame_count) is not int or self.frame_count < 1:
            raise ValueError("frame_count must be a positive integer")
        if (
            type(self.selected_frame_index) is not int
            or not 0 <= self.selected_frame_index < self.frame_count
        ):
            raise ValueError("selected_frame_index must be inside the frame range")
        for name in ("policy_id", "selection_key_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a sha256 hex identity")

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "selected_frame_index": self.selected_frame_index,
            "policy_id": self.policy_id,
            "selection_key_id": self.selection_key_id,
        }


@dataclass(frozen=True, slots=True)
class RewardInputSelectionPolicy:
    """Recipe-owned policy for selecting one frame shared by a whole batch."""

    domain: Literal["video_frame"]
    candidate_indices: Literal["all"]
    selection: Literal["keyed_uniform", "fixed_middle"]
    sharing: Literal["batch"]
    seed_derivation_schema: Literal["sha256_rejection_v1", "none"]
    schema_version: int = 1
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.domain != "video_frame":
            raise ValueError("reward input domain must be video_frame")
        if self.candidate_indices != "all":
            raise ValueError("v1 reward input candidates must be all frames")
        if self.selection not in {"keyed_uniform", "fixed_middle"}:
            raise ValueError("unsupported reward input selection mode")
        if self.sharing != "batch":
            raise ValueError("v1 reward input selection must be batch-shared")
        expected_seed_schema = (
            "sha256_rejection_v1" if self.selection == "keyed_uniform" else "none"
        )
        if self.seed_derivation_schema != expected_seed_schema:
            raise ValueError("seed_derivation_schema does not match the selection mode")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("reward input selection schema_version must equal 1")
        object.__setattr__(
            self,
            "policy_id",
            hashlib.sha256(
                _IDENTITY_DOMAIN + _canonical_bytes(self.to_payload())
            ).hexdigest(),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> RewardInputSelectionPolicy:
        if not isinstance(values, Mapping):
            raise TypeError("reward input selection policy must be a mapping")
        expected = {
            "schema_version",
            "domain",
            "candidate_indices",
            "selection",
            "sharing",
            "seed_derivation_schema",
        }
        if set(values) != expected:
            raise ValueError(
                "reward input selection policy has an invalid exact key set: "
                f"missing={sorted(expected - set(values))}, "
                f"unknown={sorted(set(values) - expected)}"
            )
        return cls(**values)

    @classmethod
    def release_world_r1(cls) -> RewardInputSelectionPolicy:
        return cls(
            domain="video_frame",
            candidate_indices="all",
            selection="keyed_uniform",
            sharing="batch",
            seed_derivation_schema="sha256_rejection_v1",
        )

    @classmethod
    def fixed_middle_extension(cls) -> RewardInputSelectionPolicy:
        return cls(
            domain="video_frame",
            candidate_indices="all",
            selection="fixed_middle",
            sharing="batch",
            seed_derivation_schema="none",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "candidate_indices": self.candidate_indices,
            "selection": self.selection,
            "sharing": self.sharing,
            "seed_derivation_schema": self.seed_derivation_schema,
        }

    def select(
        self,
        *,
        frame_count: int,
        context: StepContext,
        sample_ids: Sequence[str],
        invocation_identity: str,
    ) -> RewardInputSelection:
        if type(frame_count) is not int or frame_count < 1:
            raise ValueError("frame_count must be a positive integer")
        if not isinstance(context, StepContext):
            raise TypeError("context must be a StepContext")
        if isinstance(sample_ids, (str, bytes)):
            raise TypeError("sample_ids must be an ordered sequence")
        ordered_ids = tuple(sample_ids)
        if not ordered_ids or any(
            not isinstance(item, str) or not item for item in ordered_ids
        ):
            raise ValueError("sample_ids must contain non-empty strings")
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("sample_ids must be unique within one invocation")
        if not isinstance(invocation_identity, str) or not invocation_identity:
            raise ValueError("invocation_identity must be a non-empty string")

        key_payload = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "frame_count": frame_count,
            "step": context.step,
            "seed": context.seed,
            "rank": context.rank,
            "world_size": context.world_size,
            "sample_ids": list(ordered_ids),
            "invocation_identity": invocation_identity,
        }
        key_digest = hashlib.sha256(
            _KEY_DOMAIN + _canonical_bytes(key_payload)
        ).digest()
        key_id = key_digest.hex()
        if self.selection == "fixed_middle":
            selected = frame_count // 2
        else:
            selected = _uniform_index(key_digest, upper_bound=frame_count)
        return RewardInputSelection(
            frame_count=frame_count,
            selected_frame_index=selected,
            policy_id=self.policy_id,
            selection_key_id=key_id,
        )


def _uniform_index(key_digest: bytes, *, upper_bound: int) -> int:
    """Use rejection sampling so modulo reduction has no finite-range bias."""

    if type(upper_bound) is not int or upper_bound < 1:
        raise ValueError("upper_bound must be a positive integer")
    modulus = 1 << 256
    limit = modulus - modulus % upper_bound
    counter = 0
    while True:
        counter_bytes = counter.to_bytes(8, byteorder="big", signed=False)
        candidate = int.from_bytes(
            hashlib.sha256(_DRAW_DOMAIN + key_digest + counter_bytes).digest(),
            byteorder="big",
            signed=False,
        )
        if candidate < limit:
            return candidate % upper_bound
        counter += 1
