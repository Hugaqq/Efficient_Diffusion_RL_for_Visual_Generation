"""Immutable DynamicsSession schedule/replay and selection-state contracts."""

from __future__ import annotations

import json

import pytest
import torch

from visual_rl.algorithms.dynamics.config import WanFlowSDEConfig
from visual_rl.algorithms.dynamics.interface import (
    DeterministicTransitionOutput,
    DynamicsContractError,
    TransitionInput,
)
from visual_rl.algorithms.dynamics.selection import DynamicsSelectionPolicyState
from visual_rl.algorithms.dynamics.session import (
    DynamicsSelectionState,
    DynamicsSession,
    PolicyStepSelection,
    ScheduleSnapshot,
)
from visual_rl.algorithms.dynamics.wan_flow_sde import (
    WanFlowSDEDynamics,
    WanScheduleReplayState,
)
from visual_rl.algorithms.rollout.config import SingleStepRolloutConfig
from visual_rl.core.contracts import LikelihoodSemantics


def _wan_config(profile: str) -> WanFlowSDEConfig:
    return WanFlowSDEConfig.from_mapping(
        {
            "profile": profile,
            "likelihood_semantics": "exact_env_action",
            "replay_target": "sampled_action",
        },
        context=None,
    )


def _dynamics(*, profile: str = "flash") -> WanFlowSDEDynamics:
    return WanFlowSDEDynamics(
        WanScheduleReplayState(
            torch.tensor([900.5, 400.25], dtype=torch.float32),
            torch.tensor([1.0, 0.6, 0.1], dtype=torch.float32),
            stochastic_sampling=True,
            scheduler_identity="test.wan-scheduler@1",
        ),
        config=_wan_config(profile),
    )


def _selection(
    indices: tuple[int, ...],
    *,
    seed: int = 7,
) -> PolicyStepSelection:
    return PolicyStepSelection.fixed(
        indices,
        num_steps=2,
        generator=torch.Generator().manual_seed(seed),
        policy="test.fixed",
    )


def _session(
    dynamics: WanFlowSDEDynamics,
    indices: tuple[int, ...],
) -> DynamicsSession:
    return DynamicsSession.create(
        dynamics,
        num_steps=2,
        device="cpu",
        selection=_selection(indices),
    )


def _transition(
    session: DynamicsSession,
    indices: tuple[int, ...],
    *,
    prediction_requires_grad: bool = True,
) -> TransitionInput:
    schedule = session.transition_schedule(device="cpu")
    index = torch.tensor(indices, dtype=torch.int64)
    batch_size = len(indices)
    x_t = torch.linspace(
        -0.5,
        0.75,
        batch_size * 4,
        dtype=torch.float32,
    ).reshape(batch_size, 1, 2, 2)
    prediction = torch.full_like(
        x_t,
        0.125,
        requires_grad=prediction_requires_grad,
    )
    return TransitionInput(
        x_t=x_t,
        model_prediction=prediction,
        t=schedule.timesteps.index_select(0, index),
        t_next=schedule.next_timesteps.index_select(0, index),
        mask=torch.ones(batch_size, dtype=torch.bool),
        transition_index=index,
        condition_identity=("none",) * batch_size,
        guidance_identity=("cfg:1",) * batch_size,
        storage_dtype_identity=("torch.float32",) * batch_size,
        quantization_identity=("none",) * batch_size,
    )


def test_snapshot_round_trip_is_bit_exact_owned_and_identity_checked() -> None:
    dynamics = _dynamics()
    first = _session(dynamics, (0, 1))
    second = _session(dynamics, (1, 0))

    assert first.snapshot.schedule_identity == second.snapshot.schedule_identity
    assert first.snapshot.snapshot_identity != second.snapshot.snapshot_identity
    assert torch.equal(
        first.snapshot.sigmas,
        torch.tensor([1.0, 0.6, 0.1]),
    )
    torch.testing.assert_close(first.snapshot.dt, torch.tensor([-0.4, -0.5]))

    payload = first.snapshot.to_payload()
    json.dumps(payload)
    restored = ScheduleSnapshot.from_payload(payload)
    assert restored == first.snapshot
    assert restored.to_payload() == payload

    exposed = restored.timesteps
    exposed[0] = -999
    assert restored.timesteps[0].item() == pytest.approx(900.5)

    corrupted = dict(payload)
    corrupted["snapshot_identity"] = "0" * 64
    with pytest.raises(DynamicsContractError, match="snapshot identity mismatch"):
        ScheduleSnapshot.from_payload(corrupted)


def test_sample_and_recompute_use_the_same_snapshot_sigma_and_dt() -> None:
    session = _session(_dynamics(), (0, 1))
    transition = _transition(session, (0, 1))
    output = session.sample_transition(
        transition,
        generator=torch.Generator().manual_seed(19),
    )
    record = session.make_record(
        transition,
        output,
        conditioned_next=output.sampled_next,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
    )
    recompute_prediction = transition.model_prediction.detach().clone().requires_grad_()

    recomputed = session.recompute_log_prob(record, recompute_prediction)

    torch.testing.assert_close(recomputed, record.old_log_prob)
    torch.testing.assert_close(output.dt, session.snapshot.dt)
    recomputed.sum().backward()
    assert recompute_prediction.grad is not None


def test_ode_port_uses_terminal_snapshot_dt_and_rejects_dt_drift() -> None:
    session = _session(_dynamics(), (1, 1))
    transition = _transition(session, (1, 1))
    output = session.deterministic_ode_step(transition)

    torch.testing.assert_close(
        output.dt,
        session.snapshot.dt.index_select(0, transition.transition_index),
        rtol=0,
        atol=0,
    )
    assert bool((transition.t_next == 0).all())

    class _WrongDtWan(WanFlowSDEDynamics):
        def _deterministic_ode_step(self, request):
            result = super()._deterministic_ode_step(request)
            return DeterministicTransitionOutput(
                next_state=result.next_state,
                dt=result.dt * 0.5,
            )

    wrong = _WrongDtWan(
        WanScheduleReplayState(
            torch.tensor([900.5, 400.25], dtype=torch.float32),
            torch.tensor([1.0, 0.6, 0.1], dtype=torch.float32),
            stochastic_sampling=True,
            scheduler_identity="test.wan-scheduler@1",
        ),
        config=_wan_config("flash"),
    )
    wrong_session = _session(wrong, (1, 1))
    with pytest.raises(DynamicsContractError, match="dt does not match"):
        wrong_session.deterministic_ode_step(
            _transition(wrong_session, (1, 1), prediction_requires_grad=False)
        )


def test_interleaved_sessions_have_no_shared_schedule_cursor() -> None:
    dynamics = _dynamics()
    first = _session(dynamics, (0, 0))
    second = _session(dynamics, (1, 1))
    first_transition = _transition(first, (0, 0), prediction_requires_grad=False)
    second_transition = _transition(second, (1, 1), prediction_requires_grad=False)

    first_output = first.sample_transition(
        first_transition,
        generator=torch.Generator().manual_seed(31),
    )
    second.sample_transition(
        second_transition,
        generator=torch.Generator().manual_seed(41),
    )
    first_replay = first.transition_log_prob(
        first_transition,
        first_output.sampled_next,
    )

    isolated = DynamicsSession.from_snapshot(dynamics, first.snapshot)
    isolated_output = isolated.sample_transition(
        first_transition,
        generator=torch.Generator().manual_seed(31),
    )
    torch.testing.assert_close(first_output.sampled_next, isolated_output.sampled_next)
    torch.testing.assert_close(first_replay, isolated_output.log_prob)
    assert not hasattr(dynamics, "step_index")
    assert not hasattr(dynamics, "current_timestep")


def test_selection_state_resume_reproduces_the_next_policy_step_set() -> None:
    continuous = torch.Generator().manual_seed(2028)
    first = PolicyStepSelection.uniform(
        num_steps=40,
        cardinality=6,
        generator=continuous,
        policy="uniform",
    )
    expected_next = PolicyStepSelection.uniform(
        num_steps=40,
        cardinality=6,
        generator=continuous,
        policy="uniform",
    )

    checkpoint_payload = first.next_state.to_checkpoint_payload()
    restored_state = DynamicsSelectionState.from_checkpoint_payload(checkpoint_payload)
    resumed_next = PolicyStepSelection.uniform(
        num_steps=40,
        cardinality=6,
        selection_state=restored_state,
        policy="uniform",
    )

    assert resumed_next.indices == expected_next.indices
    assert resumed_next.next_state == expected_next.next_state


def test_checkpointed_flash_contract_reconstructs_prompt_shared_mapping() -> None:
    config = SingleStepRolloutConfig(
        selected_timestep_policy="uniform",
        num_steps=40,
        candidate_timestep_window=(0, 10),
        selection_key="prompt",
        selection_domain="single_process",
    )
    original = DynamicsSelectionPolicyState(
        base_seed=2029,
        selection_contract_identity=config.selection_contract_identity,
    )
    restored = DynamicsSelectionPolicyState.from_checkpoint_payload(
        original.to_checkpoint_payload()
    )
    rollout_identity = "iteration-rollout.v1:test-flash-resume"
    keys = ("prompt-a", "prompt-a", "prompt-b", "prompt-b")

    def select(policy: DynamicsSelectionPolicyState) -> PolicyStepSelection:
        seed = policy.derive_stream_seed(
            rollout_identity=rollout_identity,
            stream="selection",
        )
        return PolicyStepSelection.uniform_from_candidates_by_key(
            num_steps=config.num_steps,
            candidate_indices=config.candidate_indices,
            keys=keys,
            generator=torch.Generator().manual_seed(seed),
            policy=policy.selection_contract_identity,
        )

    expected = select(original)
    resumed = select(restored)

    assert restored == original
    assert expected.indices == resumed.indices
    assert expected.selection_mapping_identity == (resumed.selection_mapping_identity)
    assert expected.indices[0] == expected.indices[1]
    assert expected.indices[2] == expected.indices[3]
    assert all(index < 10 for index in expected.indices)


def test_snapshot_refuses_a_different_dynamics_equation() -> None:
    flash = _dynamics(profile="flash")
    snapshot = _session(flash, (0, 1)).snapshot
    conditioned = _dynamics(profile="conditioned")

    with pytest.raises(DynamicsContractError, match="does not match"):
        DynamicsSession.from_snapshot(conditioned, snapshot)
