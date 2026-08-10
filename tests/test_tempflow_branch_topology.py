"""Typed TempFlow topology identity and recipe fidelity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from visual_rl.algorithms.rollout.branching import BranchingRollout
from visual_rl.algorithms.rollout.config import BranchingRolloutConfig
from visual_rl.composition.config import compile_recipe_v2, load_source_recipe
from visual_rl.data.samples import BranchTopology

ROOT = Path(__file__).resolve().parents[1]


def test_branch_topology_payload_round_trip_and_identity_are_exact() -> None:
    paper = BranchTopology.every_policy_timestep(6)
    ablation = BranchTopology.single_point_branch_ablation(6)

    assert paper.topology_identity != ablation.topology_identity
    assert BranchTopology.from_payload(paper.to_payload()) == paper
    assert paper.stored_policy_axes == (
        "prompt_group",
        "exploration_member",
        "policy_timestep",
    )
    assert paper.reward_media_axes == paper.stored_policy_axes
    assert ablation.stored_policy_axes == (
        "prompt_group",
        "exploration_member",
    )

    tampered = paper.to_payload()
    tampered["exploration_count"] = 7
    with pytest.raises(ValueError, match="identity mismatch"):
        BranchTopology.from_payload(tampered)


def test_branching_config_separates_paper_topology_from_single_point_ablation() -> None:
    paper = BranchingRolloutConfig.from_mapping(
        {
            "num_steps": 28,
            "branch_count": 6,
            "branch_topology": BranchTopology.every_policy_timestep(6).to_payload(),
        },
        context=None,
    )
    assert paper.branch_topology == BranchTopology.every_policy_timestep(6)
    assert paper.branch_step_policy is None
    contract = BranchingRollout.describe(paper).rollout
    assert contract.schedule_step_count == (28, 28)
    assert contract.physical_transition_count == (432, 432)
    assert contract.stored_policy_transition_count == (27, 27)

    ablation = BranchingRolloutConfig(
        num_steps=28,
        branch_count=6,
        branch_topology=BranchTopology.single_point_branch_ablation(6),
        branch_step_policy="uniform_intermediate",
        branch_step_index=3,
    )
    assert ablation.branch_topology == BranchTopology.single_point_branch_ablation(6)
    assert paper.selection_contract_identity != ablation.selection_contract_identity
    assert (
        paper.selection_contract_identity
        != BranchingRolloutConfig(
            num_steps=27,
            branch_count=6,
            branch_topology=BranchTopology.every_policy_timestep(6),
        ).selection_contract_identity
    )
    assert (
        ablation.selection_contract_identity
        != BranchingRolloutConfig(
            num_steps=28,
            branch_count=6,
            branch_topology=BranchTopology.single_point_branch_ablation(6),
            branch_step_policy="uniform_intermediate",
            branch_step_index=4,
        ).selection_contract_identity
    )
    assert BranchingRollout.describe(
        ablation
    ).rollout.stored_policy_transition_count == (1, 1)

    with pytest.raises(ValueError, match="missing branching rollout params"):
        BranchingRolloutConfig.from_mapping(
            {
                "num_steps": 28,
                "branch_count": 6,
                "branch_step_policy": "uniform_intermediate",
            },
            context=None,
        )

    with pytest.raises(ValueError, match="does not accept"):
        BranchingRolloutConfig(
            num_steps=28,
            branch_count=6,
            branch_step_policy="uniform_intermediate",
            branch_topology=BranchTopology.every_policy_timestep(6),
        )
    with pytest.raises(ValueError, match="must equal branch_count"):
        BranchingRolloutConfig(
            num_steps=28,
            branch_count=6,
            branch_topology=BranchTopology.every_policy_timestep(5),
        )


@pytest.mark.parametrize(
    ("num_steps", "expected_physical_count"),
    ((2, 3), (3, 7), (4, 12), (28, 432)),
)
def test_paper_transition_work_formula_is_not_schedule_depth(
    num_steps: int,
    expected_physical_count: int,
) -> None:
    config = BranchingRolloutConfig(
        num_steps=num_steps,
        branch_count=2,
        branch_topology=BranchTopology.every_policy_timestep(2),
    )
    contract = BranchingRollout.describe(config).rollout

    policy_action_count = num_steps - 1
    continuation_count = num_steps * (num_steps - 1) // 2
    expected_from_loop = policy_action_count + policy_action_count + continuation_count
    assert expected_from_loop == expected_physical_count
    assert contract.schedule_step_count == (num_steps, num_steps)
    assert contract.physical_transition_count == (
        expected_physical_count,
        expected_physical_count,
    )
    assert contract.stored_policy_transition_count == (
        policy_action_count,
        policy_action_count,
    )


def test_tempflow_recipe_selects_paper_topology_not_ablation() -> None:
    recipe = compile_recipe_v2(
        load_source_recipe(ROOT / "configs/v2/tempflow_sd3.yaml")
    )
    rollout = recipe.component("rollout").declaration

    assert rollout.alias == "branching"
    assert isinstance(rollout.config, BranchingRolloutConfig)
    topology = rollout.config.branch_topology
    assert topology.kind == "every_policy_timestep"
    assert topology.exploration_count == 6
    assert rollout.config.branch_step_policy is None
