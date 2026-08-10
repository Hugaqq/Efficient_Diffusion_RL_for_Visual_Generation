"""Contracts for recipe-owned, replayable reward-input selection."""

from __future__ import annotations

import pytest

from visual_rl.core.types import StepContext
from visual_rl.algorithms.rewards import RewardInputSelectionPolicy


def _context(*, seed: int = 19) -> StepContext:
    return StepContext(step=3, seed=seed, rank=1, world_size=2)


def test_world_r1_release_policy_has_stable_identity_and_vector() -> None:
    policy = RewardInputSelectionPolicy.release_world_r1()

    selection = policy.select(
        frame_count=17,
        context=_context(),
        sample_ids=("a", "b"),
        invocation_identity="reward_general",
    )

    assert policy.to_payload() == {
        "schema_version": 1,
        "domain": "video_frame",
        "candidate_indices": "all",
        "selection": "keyed_uniform",
        "sharing": "batch",
        "seed_derivation_schema": "sha256_rejection_v1",
    }
    assert policy.policy_id == (
        "4cfef0ae44c1003304943a706215894597b5b9ab0d04bd99f13ff701bb1b1e24"
    )
    assert selection.to_payload() == {
        "frame_count": 17,
        "selected_frame_index": 4,
        "policy_id": policy.policy_id,
        "selection_key_id": (
            "9da9b31e52cf4c3da0e7c3847fa3c69cace2af994501a7c0fe4d1cf53ce80cdd"
        ),
    }


def test_keyed_selection_is_exactly_replayable_and_key_sensitive() -> None:
    policy = RewardInputSelectionPolicy.release_world_r1()
    kwargs = {
        "frame_count": 9,
        "context": _context(),
        "sample_ids": ("sample-a", "sample-b"),
        "invocation_identity": "reward_general",
    }

    first = policy.select(**kwargs)
    replay = policy.select(**kwargs)
    changed_seed = policy.select(**{**kwargs, "context": _context(seed=20)})
    changed_rows = policy.select(
        **{**kwargs, "sample_ids": ("sample-b", "sample-a")}
    )

    assert first == replay
    assert first.selection_key_id != changed_seed.selection_key_id
    assert first.selection_key_id != changed_rows.selection_key_id
    assert 0 <= first.selected_frame_index < first.frame_count


def test_fixed_middle_is_a_distinct_explicit_extension() -> None:
    policy = RewardInputSelectionPolicy.fixed_middle_extension()

    selection = policy.select(
        frame_count=4,
        context=_context(),
        sample_ids=("a", "b"),
        invocation_identity="reward_general",
    )

    assert policy.seed_derivation_schema == "none"
    assert selection.selected_frame_index == 2
    assert policy.policy_id != RewardInputSelectionPolicy.release_world_r1().policy_id


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"selection": "middle"}, "unsupported"),
        ({"sharing": "sample"}, "batch-shared"),
        ({"candidate_indices": "first_half"}, "all frames"),
        ({"seed_derivation_schema": "none"}, "does not match"),
    ],
)
def test_policy_rejects_silent_semantic_variants(
    mutation: dict[str, object],
    error: str,
) -> None:
    payload = RewardInputSelectionPolicy.release_world_r1().to_payload()
    payload.update(mutation)

    with pytest.raises(ValueError, match=error):
        RewardInputSelectionPolicy.from_mapping(payload)


def test_selection_rejects_ambiguous_row_identity() -> None:
    policy = RewardInputSelectionPolicy.release_world_r1()

    with pytest.raises(ValueError, match="unique"):
        policy.select(
            frame_count=3,
            context=_context(),
            sample_ids=("same", "same"),
            invocation_identity="reward_general",
        )
