"""Resolved reward execution and strict row-aligned aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from visual_rl.core.contracts import (
    LogicalRewardSpec,
    RewardGranularity,
    RewardPlanSpec,
    RewardRouteBinding,
    RewardRouteSpec,
)
from visual_rl.core.identity import to_identity_value
from visual_rl.core.types import FrozenMapping
from visual_rl.algorithms.rewards.resource_port import (
    RewardResourceHandle,
    RewardResourcePoolView,
)
from visual_rl.algorithms.rewards.types import (
    GroupwiseReward,
    GroupwiseRewardOutput,
    PointwiseReward,
    PointwiseRewardOutput,
    RewardBatchView,
    RewardResult,
)


class RewardProcessor:
    """Execute an explicit resolved route without source/phase heuristics."""

    def __init__(
        self,
        *,
        plan: RewardPlanSpec,
        pool: RewardResourcePoolView,
        logical_rewards: Mapping[str, PointwiseReward | GroupwiseReward],
    ) -> None:
        if not isinstance(plan, RewardPlanSpec):
            raise TypeError("plan must be a RewardPlanSpec")
        if not plan.materialized:
            raise ValueError("reward execution requires a materialized RewardPlanSpec")
        if not isinstance(pool, RewardResourcePoolView):
            raise TypeError("pool must be a non-owning RewardPoolView")
        if pool.plan != plan:
            raise ValueError("RewardPoolView belongs to a different reward plan")
        if not isinstance(logical_rewards, Mapping):
            raise TypeError("logical_rewards must be a mapping")
        copied = dict(logical_rewards)
        if tuple(copied) != plan.logical_reward_ids:
            raise ValueError(
                "logical_rewards must match the resolved plan ids and order exactly"
            )
        for logical_id, reward in copied.items():
            granularity = plan.logical_reward(logical_id).contract.granularity
            if (
                granularity is RewardGranularity.POINTWISE
                and not isinstance(reward, PointwiseReward)
            ):
                raise TypeError(
                    f"logical reward {logical_id!r} must be PointwiseReward"
                )
            if (
                granularity is RewardGranularity.GROUPWISE
                and not isinstance(reward, GroupwiseReward)
            ):
                raise TypeError(
                    f"logical reward {logical_id!r} must be GroupwiseReward"
                )
        handles: dict[str, RewardResourceHandle] = {}
        for logical_id, reward in copied.items():
            logical = plan.logical_reward(logical_id)
            handle = pool.handle(logical.resource_identity)
            handles[logical_id] = handle
            bind = getattr(reward, "bind_resource", None)
            if callable(bind):
                is_bound_to = getattr(reward, "is_bound_to", None)
                if callable(is_bound_to) and is_bound_to(handle):
                    continue
                bind(handle)
        self.plan = plan
        self.pool = pool
        self.logical_rewards = MappingProxyType(copied)
        self.resource_handles = MappingProxyType(handles)

    def process(
        self,
        *,
        batch: RewardBatchView,
        route: RewardRouteSpec,
    ) -> RewardResult:
        if not isinstance(batch, RewardBatchView):
            raise TypeError("batch must be a RewardBatchView")
        if not isinstance(route, RewardRouteSpec):
            raise TypeError("route must be a RewardRouteSpec")
        canonical_route = self.plan.route_for(
            source_id=route.source_id,
            phase_id=route.phase_id,
        )
        if canonical_route != route:
            raise ValueError("route is not the canonical route from this reward plan")
        if batch.source_id != route.source_id or batch.phase_id != route.phase_id:
            raise ValueError(
                "homogeneous batch source/phase does not match reward route"
            )
        if batch.active_reward_ids != route.logical_reward_ids:
            raise ValueError("batch active rewards do not match the resolved route")

        component_scores: dict[str, np.ndarray] = {}
        weighted_scores: dict[str, np.ndarray] = {}
        component_masks: dict[str, np.ndarray] = {}
        active_ids = set(route.logical_reward_ids)
        row_applicable_masks = {
            logical_id: np.full(
                batch.batch_size,
                logical_id in active_ids,
                dtype=np.bool_,
            )
            for logical_id in self.plan.logical_reward_ids
        }
        applicable_masks = {
            logical_id: np.array(
                np.broadcast_to(
                    row_mask.reshape(
                        (batch.batch_size, *([1] * len(batch.score_axis_names)))
                    ),
                    batch.score_shape,
                ),
                dtype=np.bool_,
                copy=True,
            )
            for logical_id, row_mask in row_applicable_masks.items()
        }
        resource_identities: dict[str, str] = {}
        logical_weights: dict[str, float] = {}
        logical_provenance = {}
        logical_execution_provenance: dict[str, FrozenMapping] = {}
        total = np.zeros(batch.score_shape, dtype=np.float64)
        combined_mask = np.ones(batch.score_shape, dtype=np.bool_)
        expected_groups = tuple(dict.fromkeys(batch.identity.group_ids))

        for binding in route.rewards:
            logical = self.plan.logical_reward(binding.logical_reward_id)
            values, valid_mask, score_axes, execution_provenance = self._score_binding(
                binding,
                logical=logical,
                batch=batch,
                expected_groups=expected_groups,
            )
            logical_id = binding.logical_reward_id
            if score_axes != batch.score_axis_names:
                raise ValueError(
                    f"reward {logical_id!r} score axes do not match the batch"
                )
            if (
                values.shape != batch.score_shape
                or valid_mask.shape != batch.score_shape
            ):
                raise ValueError(
                    f"reward {logical_id!r} score shape does not match the batch"
                )
            applicable = applicable_masks[logical_id]
            values = np.where(applicable, values, 0.0)
            valid_mask = np.asarray(valid_mask & applicable, dtype=np.bool_)
            weighted = np.where(applicable, values * binding.weight, 0.0)
            weighted = np.asarray(weighted, dtype=np.float64)
            if not bool(np.isfinite(weighted).all()):
                raise ValueError(f"weighted reward {logical_id!r} is non-finite")
            component_scores[logical_id] = values
            weighted_scores[logical_id] = weighted
            component_masks[logical_id] = valid_mask
            resource_identities[logical_id] = logical.resource_identity
            logical_weights[logical_id] = binding.weight
            logical_provenance[logical_id] = _logical_provenance(logical)
            logical_execution_provenance[logical_id] = execution_provenance
            total += weighted
            combined_mask &= ~applicable | valid_mask

        for logical_id in self.plan.logical_reward_ids:
            if logical_id in component_scores:
                continue
            logical = self.plan.logical_reward(logical_id)
            component_scores[logical_id] = np.zeros(
                batch.score_shape,
                dtype=np.float64,
            )
            weighted_scores[logical_id] = np.zeros(
                batch.score_shape,
                dtype=np.float64,
            )
            component_masks[logical_id] = np.zeros(
                batch.score_shape,
                dtype=np.bool_,
            )
            resource_identities[logical_id] = logical.resource_identity
            logical_weights[logical_id] = 0.0
            logical_provenance[logical_id] = _logical_provenance(logical)
            logical_execution_provenance[logical_id] = FrozenMapping()

        component_scores = {
            logical_id: component_scores[logical_id]
            for logical_id in self.plan.logical_reward_ids
        }
        weighted_scores = {
            logical_id: weighted_scores[logical_id]
            for logical_id in self.plan.logical_reward_ids
        }
        component_masks = {
            logical_id: component_masks[logical_id]
            for logical_id in self.plan.logical_reward_ids
        }
        resource_identities = {
            logical_id: resource_identities[logical_id]
            for logical_id in self.plan.logical_reward_ids
        }
        logical_weights = {
            logical_id: logical_weights[logical_id]
            for logical_id in self.plan.logical_reward_ids
        }
        logical_provenance = {
            logical_id: logical_provenance[logical_id]
            for logical_id in self.plan.logical_reward_ids
        }
        logical_execution_provenance = {
            logical_id: logical_execution_provenance[logical_id]
            for logical_id in self.plan.logical_reward_ids
        }

        return RewardResult(
            identity=batch.identity,
            component_scores=component_scores,
            weighted_scores=weighted_scores,
            component_valid_masks=component_masks,
            weighted_total=total,
            valid_mask=combined_mask,
            resource_identities=resource_identities,
            score_axis_names=batch.score_axis_names,
            component_applicable_masks=applicable_masks,
            logical_weights=logical_weights,
            logical_provenance=logical_provenance,
            logical_execution_provenance=logical_execution_provenance,
        )

    def _score_binding(
        self,
        binding: RewardRouteBinding,
        *,
        logical: LogicalRewardSpec,
        batch: RewardBatchView,
        expected_groups: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], FrozenMapping]:
        logical_id = binding.logical_reward_id
        reward = self.logical_rewards[logical_id]
        handle = self.resource_handles[logical_id]
        if handle.resource_identity != logical.resource_identity:
            raise RuntimeError("logical reward handle identity drifted after bind")
        resource = handle.resource_for_execution()
        if logical.contract.granularity is RewardGranularity.POINTWISE:
            assert isinstance(reward, PointwiseReward)
            output = reward.score(
                logical_reward_id=logical_id,
                resource=resource,
                batch=batch,
            )
            if not isinstance(output, PointwiseRewardOutput):
                raise TypeError(
                    f"pointwise reward {logical_id!r} must return PointwiseRewardOutput"
                )
            self._validate_identity(output.identity, batch=batch, logical_id=logical_id)
            return (
                output.values,
                output.valid_mask,
                output.score_axis_names,
                output.execution_provenance,
            )

        if batch.score_axis_names:
            raise NotImplementedError(
                "groupwise rewards do not yet support additional score axes"
            )
        assert isinstance(reward, GroupwiseReward)
        output = reward.score_groups(
            logical_reward_id=logical_id,
            resource=resource,
            batch=batch,
            group_ids=expected_groups,
        )
        if not isinstance(output, GroupwiseRewardOutput):
            raise TypeError(
                f"groupwise reward {logical_id!r} must return GroupwiseRewardOutput"
            )
        self._validate_identity(output.identity, batch=batch, logical_id=logical_id)
        if output.group_ids != expected_groups:
            raise ValueError(
                f"groupwise reward {logical_id!r} group identity/order mismatch"
            )
        group_index = {
            group_id: index for index, group_id in enumerate(output.group_ids)
        }
        row_indices = np.fromiter(
            (group_index[group_id] for group_id in batch.identity.group_ids),
            dtype=np.int64,
            count=batch.batch_size,
        )
        return (
            output.values[row_indices],
            output.valid_mask[row_indices],
            (),
            FrozenMapping(),
        )

    @staticmethod
    def _validate_identity(
        identity: Any,
        *,
        batch: RewardBatchView,
        logical_id: str,
    ) -> None:
        if identity != batch.identity:
            raise ValueError(
                f"reward {logical_id!r} batch row/sample/trajectory/condition "
                "payload identity mismatch"
            )


def _logical_provenance(logical: LogicalRewardSpec) -> FrozenMapping:
    return FrozenMapping(
        {
            "component_declaration_id": logical.component_declaration_id,
            "resource_identity": logical.resource_identity,
            "contract": to_identity_value(logical.contract),
        }
    )


__all__ = ("RewardProcessor",)
