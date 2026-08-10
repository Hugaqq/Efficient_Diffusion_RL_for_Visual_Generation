"""Explicit K-repeat placement geometry for distributed GRPO batches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import re

from visual_rl.core.serialization import canonical_json_text
from visual_rl.data.samples import BatchRowContext

__all__ = (
    "BoundGroupPlacementLayout",
    "CollectiveDomain",
    "GroupMemberPlacement",
    "GroupPlacementContract",
    "GroupPlacementError",
    "GroupPlacementKind",
    "GroupPlacementLayout",
    "PlacedBatchRow",
)


_SHA256 = re.compile(r"[0-9a-f]{64}")


class GroupPlacementError(ValueError):
    """Raised when K-repeat groups cannot be placed without fragmentation."""


class GroupPlacementKind(str, Enum):
    """The two group-complete layouts supported by the v0.8 data plane."""

    LOCAL_COMPLETE = "local_complete"
    SHARDED_COMPLETE = "sharded_complete"


class CollectiveDomain(str, Enum):
    """Where groupwise reward/scatter/normalization must execute."""

    LOCAL_RANK = "local_rank"
    WORLD = "world"


def _positive(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise GroupPlacementError(f"{field_name} must be a positive integer")
    return value


def _non_negative(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise GroupPlacementError(f"{field_name} must be a non-negative integer")
    return value


def _group_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\r" in value
        or "\n" in value
    ):
        raise GroupPlacementError("group ids must be canonical non-empty strings")
    return value


@dataclass(frozen=True, slots=True)
class GroupMemberPlacement:
    """One globally unique group member at one optimizer accumulation index."""

    group_id: str
    member_id: int
    rank: int
    accumulation_index: int
    optimizer_step: int

    def __post_init__(self) -> None:
        _group_id(self.group_id)
        _non_negative(self.member_id, field_name="member_id")
        _non_negative(self.rank, field_name="rank")
        _non_negative(self.accumulation_index, field_name="accumulation_index")
        _non_negative(self.optimizer_step, field_name="optimizer_step")

    def to_payload(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "member_id": self.member_id,
            "rank": self.rank,
            "accumulation_index": self.accumulation_index,
            "optimizer_step": self.optimizer_step,
        }


@dataclass(frozen=True, slots=True)
class GroupPlacementContract:
    """Sole owner of distributed K-repeat and accumulation geometry.

    ``global_prompt_batch_size`` is the number of unique prompt groups in one
    optimizer update.  The equality below must hold exactly::

        prompts * K == world_size * per_rank_microbatch_rows * accumulation

    This makes the logical optimizer batch independent of tensor-shape guesses.
    """

    placement: GroupPlacementKind
    global_prompt_batch_size: int
    group_size: int
    world_size: int
    per_rank_microbatch_rows: int
    gradient_accumulation_steps: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        try:
            placement = GroupPlacementKind(self.placement)
        except (TypeError, ValueError):
            raise GroupPlacementError(
                f"unknown group placement: {self.placement!r}"
            ) from None
        object.__setattr__(self, "placement", placement)
        for name in (
            "global_prompt_batch_size",
            "group_size",
            "world_size",
            "per_rank_microbatch_rows",
            "gradient_accumulation_steps",
        ):
            _positive(getattr(self, name), field_name=name)
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise GroupPlacementError("unsupported group placement schema_version")

        logical_rows = self.global_prompt_batch_size * self.group_size
        physical_rows = (
            self.world_size
            * self.per_rank_microbatch_rows
            * self.gradient_accumulation_steps
        )
        if logical_rows != physical_rows:
            raise GroupPlacementError(
                "global prompt/group batch is inconsistent with world size, "
                "per-rank microbatch rows, and gradient accumulation"
            )

        if self.placement is GroupPlacementKind.LOCAL_COMPLETE:
            if self.global_prompt_batch_size % self.world_size:
                raise GroupPlacementError(
                    "local_complete requires global_prompt_batch_size divisible "
                    "by world_size"
                )
            if self.per_rank_microbatch_rows % self.group_size:
                raise GroupPlacementError(
                    "local_complete requires every per-rank microbatch to contain "
                    "only complete groups"
                )
        else:
            if self.group_size % self.world_size:
                raise GroupPlacementError(
                    "sharded_complete requires group_size divisible by world_size"
                )
            if self.global_prompt_batch_size % self.gradient_accumulation_steps:
                raise GroupPlacementError(
                    "sharded_complete cannot split a group across accumulation indices"
                )
            if self.per_rank_microbatch_rows % self.copies_per_rank:
                raise GroupPlacementError(
                    "sharded_complete per-rank microbatch does not tile complete "
                    "global groups"
                )

    @property
    def global_row_count(self) -> int:
        return self.global_prompt_batch_size * self.group_size

    @property
    def per_rank_update_rows(self) -> int:
        return self.per_rank_microbatch_rows * self.gradient_accumulation_steps

    @property
    def copies_per_rank(self) -> int:
        if self.placement is GroupPlacementKind.LOCAL_COMPLETE:
            return self.group_size
        return self.group_size // self.world_size

    @property
    def groups_per_rank_microbatch(self) -> int:
        return self.per_rank_microbatch_rows // self.copies_per_rank

    @property
    def reward_gather_domain(self) -> CollectiveDomain:
        if self.placement is GroupPlacementKind.LOCAL_COMPLETE:
            return CollectiveDomain.LOCAL_RANK
        return CollectiveDomain.WORLD

    @property
    def reward_scatter_domain(self) -> CollectiveDomain:
        return self.reward_gather_domain

    @property
    def advantage_normalization_domain(self) -> CollectiveDomain:
        return self.reward_gather_domain

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "group_placement_contract",
            "placement": self.placement.value,
            "global_prompt_batch_size": self.global_prompt_batch_size,
            "group_size": self.group_size,
            "world_size": self.world_size,
            "per_rank_microbatch_rows": self.per_rank_microbatch_rows,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "reward_gather_domain": self.reward_gather_domain.value,
            "reward_scatter_domain": self.reward_scatter_domain.value,
            "advantage_normalization_domain": (
                self.advantage_normalization_domain.value
            ),
        }

    @property
    def contract_id(self) -> str:
        return hashlib.sha256(
            canonical_json_text(self.to_payload()).encode("utf-8")
        ).hexdigest()

    def place(
        self,
        group_ids: tuple[str, ...],
        *,
        optimizer_step: int,
    ) -> GroupPlacementLayout:
        if type(group_ids) is not tuple:
            raise TypeError("group_ids must be a tuple")
        if len(group_ids) != self.global_prompt_batch_size:
            raise GroupPlacementError(
                "group_ids must contain global_prompt_batch_size entries"
            )
        for value in group_ids:
            _group_id(value)
        if len(group_ids) != len(set(group_ids)):
            raise GroupPlacementError("group_ids must be unique within an update")
        _non_negative(optimizer_step, field_name="optimizer_step")
        rows = self._expected_rows(group_ids, optimizer_step=optimizer_step)
        return GroupPlacementLayout(
            contract_id=self.contract_id,
            placement=self.placement,
            group_size=self.group_size,
            world_size=self.world_size,
            per_rank_microbatch_rows=self.per_rank_microbatch_rows,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optimizer_step=optimizer_step,
            group_ids=group_ids,
            rows=rows,
        )

    def _expected_rows(
        self,
        group_ids: tuple[str, ...],
        *,
        optimizer_step: int,
    ) -> tuple[GroupMemberPlacement, ...]:
        rows: list[GroupMemberPlacement] = []
        groups_per_microbatch = (
            self.groups_per_rank_microbatch * self.world_size
            if self.placement is GroupPlacementKind.LOCAL_COMPLETE
            else self.groups_per_rank_microbatch
        )
        for accumulation_index in range(self.gradient_accumulation_steps):
            start = accumulation_index * groups_per_microbatch
            microbatch_groups = group_ids[start : start + groups_per_microbatch]
            for rank in range(self.world_size):
                if self.placement is GroupPlacementKind.LOCAL_COMPLETE:
                    rank_start = rank * self.groups_per_rank_microbatch
                    rank_groups = microbatch_groups[
                        rank_start : rank_start + self.groups_per_rank_microbatch
                    ]
                    member_ids = range(self.group_size)
                else:
                    rank_groups = microbatch_groups
                    member_start = rank * self.copies_per_rank
                    member_ids = range(
                        member_start,
                        member_start + self.copies_per_rank,
                    )
                rows.extend(
                    GroupMemberPlacement(
                        group_id=group_id,
                        member_id=member_id,
                        rank=rank,
                        accumulation_index=accumulation_index,
                        optimizer_step=optimizer_step,
                    )
                    for group_id in rank_groups
                    for member_id in member_ids
                )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class GroupPlacementLayout:
    """Materialized placement for one optimizer step and ordered group batch."""

    contract_id: str
    placement: GroupPlacementKind
    group_size: int
    world_size: int
    per_rank_microbatch_rows: int
    gradient_accumulation_steps: int
    optimizer_step: int
    group_ids: tuple[str, ...]
    rows: tuple[GroupMemberPlacement, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.contract_id, str)
            or _SHA256.fullmatch(self.contract_id) is None
        ):
            raise GroupPlacementError("contract_id must be a lowercase SHA-256 digest")
        if type(self.group_ids) is not tuple or not self.group_ids:
            raise GroupPlacementError("group_ids must be a non-empty tuple")
        if len(self.group_ids) != len(set(self.group_ids)):
            raise GroupPlacementError("group_ids must be unique")
        for value in self.group_ids:
            _group_id(value)
        if type(self.rows) is not tuple or any(
            not isinstance(item, GroupMemberPlacement) for item in self.rows
        ):
            raise TypeError("rows must be a tuple of GroupMemberPlacement values")
        _non_negative(self.optimizer_step, field_name="optimizer_step")

        contract = GroupPlacementContract(
            placement=self.placement,
            global_prompt_batch_size=len(self.group_ids),
            group_size=self.group_size,
            world_size=self.world_size,
            per_rank_microbatch_rows=self.per_rank_microbatch_rows,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        object.__setattr__(self, "placement", contract.placement)
        if self.contract_id != contract.contract_id:
            raise GroupPlacementError(
                "layout geometry does not match its GroupPlacementContract identity"
            )
        expected = contract._expected_rows(
            self.group_ids,
            optimizer_step=self.optimizer_step,
        )
        if self.rows != expected:
            raise GroupPlacementError(
                "layout rows are incomplete, duplicated, cross-rank, or cross-step "
                "relative to the declared placement contract"
            )

    def rows_for_rank(
        self,
        rank: int,
        *,
        accumulation_index: int | None = None,
    ) -> tuple[GroupMemberPlacement, ...]:
        _non_negative(rank, field_name="rank")
        if rank >= self.world_size:
            raise GroupPlacementError("rank must be smaller than world_size")
        if accumulation_index is not None:
            _non_negative(
                accumulation_index,
                field_name="accumulation_index",
            )
            if accumulation_index >= self.gradient_accumulation_steps:
                raise GroupPlacementError(
                    "accumulation_index must be smaller than accumulation steps"
                )
        return tuple(
            row
            for row in self.rows
            if row.rank == rank
            and (
                accumulation_index is None
                or row.accumulation_index == accumulation_index
            )
        )

    def bind_batch_rows(
        self,
        batch_rows: tuple[BatchRowContext, ...],
    ) -> BoundGroupPlacementLayout:
        """Bind the placement to real row identities before reward execution."""

        if type(batch_rows) is not tuple or not batch_rows:
            raise GroupPlacementError("batch_rows must be a non-empty tuple")
        if any(not isinstance(item, BatchRowContext) for item in batch_rows):
            raise TypeError("batch_rows must contain BatchRowContext values")
        by_key: dict[tuple[str, int], BatchRowContext] = {}
        for row in batch_rows:
            key = (row.group_id, row.member_id)
            if key in by_key:
                raise GroupPlacementError(
                    "batch_rows contain duplicate group/member identity"
                )
            by_key[key] = row
        expected_keys = tuple((row.group_id, row.member_id) for row in self.rows)
        if set(by_key) != set(expected_keys) or len(by_key) != len(expected_keys):
            raise GroupPlacementError(
                "batch_rows are missing or unexpected for the placement layout"
            )
        return BoundGroupPlacementLayout(
            layout=self,
            rows=tuple(
                PlacedBatchRow(
                    placement=placement,
                    batch_row=by_key[(placement.group_id, placement.member_id)],
                )
                for placement in self.rows
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "contract_id": self.contract_id,
            "placement": self.placement.value,
            "group_size": self.group_size,
            "world_size": self.world_size,
            "per_rank_microbatch_rows": self.per_rank_microbatch_rows,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_step": self.optimizer_step,
            "group_ids": list(self.group_ids),
            "rows": [item.to_payload() for item in self.rows],
        }


@dataclass(frozen=True, slots=True)
class PlacedBatchRow:
    """One canonical placement bound to its persistent iteration row identity."""

    placement: GroupMemberPlacement
    batch_row: BatchRowContext

    def __post_init__(self) -> None:
        if not isinstance(self.placement, GroupMemberPlacement):
            raise TypeError("placement must be a GroupMemberPlacement")
        if not isinstance(self.batch_row, BatchRowContext):
            raise TypeError("batch_row must be a BatchRowContext")
        self.batch_row.validate()
        if (
            self.batch_row.group_id != self.placement.group_id
            or self.batch_row.member_id != self.placement.member_id
            or self.batch_row.optimizer_step != self.placement.optimizer_step
        ):
            raise GroupPlacementError(
                "BatchRowContext group/member/step does not match placement"
            )

    @property
    def batch_row_identity(self) -> str:
        return self.batch_row.identity

    def to_payload(self) -> dict[str, object]:
        return {
            "placement": self.placement.to_payload(),
            "batch_row_identity": self.batch_row_identity,
            "batch_row": self.batch_row.serialize(),
        }


@dataclass(frozen=True, slots=True)
class BoundGroupPlacementLayout:
    """Group-complete layout with one phase and reward-scatter row identities."""

    layout: GroupPlacementLayout
    rows: tuple[PlacedBatchRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout, GroupPlacementLayout):
            raise TypeError("layout must be a GroupPlacementLayout")
        if type(self.rows) is not tuple or any(
            not isinstance(item, PlacedBatchRow) for item in self.rows
        ):
            raise TypeError("rows must be a tuple of PlacedBatchRow values")
        if tuple(item.placement for item in self.rows) != self.layout.rows:
            raise GroupPlacementError(
                "bound rows must preserve the placement layout's canonical order"
            )
        row_ids = tuple(item.batch_row_identity for item in self.rows)
        if len(row_ids) != len(set(row_ids)):
            raise GroupPlacementError("bound batch row identities must be unique")
        occurrence_ids = tuple(item.batch_row.occurrence_id for item in self.rows)
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise GroupPlacementError(
                "K-repeat members must not reuse occurrence identities"
            )
        phases = {item.batch_row.phase for item in self.rows}
        if len(phases) != 1:
            raise GroupPlacementError(
                "one bound group placement cannot contain multiple phases"
            )
        for group_id in self.layout.group_ids:
            source_ids = {
                item.batch_row.source_item_id
                for item in self.rows
                if item.batch_row.group_id == group_id
            }
            if len(source_ids) != 1:
                raise GroupPlacementError(
                    "all K-repeat members in a group must share one source item"
                )

    @property
    def phase(self) -> str:
        return self.rows[0].batch_row.phase

    @property
    def scatter_identity(self) -> tuple[tuple[str, int], ...]:
        """Canonical row-id to global row-index mapping for reward scatter."""

        return tuple(
            (item.batch_row_identity, index) for index, item in enumerate(self.rows)
        )

    def rows_for_rank(
        self,
        rank: int,
        *,
        accumulation_index: int | None = None,
    ) -> tuple[PlacedBatchRow, ...]:
        placements = self.layout.rows_for_rank(
            rank,
            accumulation_index=accumulation_index,
        )
        selected = set(placements)
        return tuple(item for item in self.rows if item.placement in selected)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "layout": self.layout.to_payload(),
            "phase": self.phase,
            "scatter_identity": [
                {"batch_row_identity": row_id, "row_index": index}
                for row_id, index in self.scatter_identity
            ],
            "rows": [item.to_payload() for item in self.rows],
        }
