"""Parity and replay-state contracts for the SD3 Dynamics wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from visual_rl.algorithms.dynamics.config import FlowSDEConfig
from visual_rl.algorithms.dynamics.interface import (
    DynamicsContractError,
    TransitionInput,
)
from visual_rl.algorithms.dynamics.sd3_flow_sde import (
    SD3FlowSDEDynamics,
    SD3ScheduleReplayState,
)
from visual_rl.algorithms.dynamics.transition import (
    sd3_sde_step_with_logprob,
)
from visual_rl.core.contracts import LikelihoodSemantics


@dataclass
class _Config:
    stochastic_sampling: bool = True


class _Scheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor(
            [999.0, 833.3333, 0.25],
            dtype=torch.float32,
        )
        self.sigmas = torch.tensor(
            [1.0, 0.75, 0.25, 0.05],
            dtype=torch.float32,
        )
        self.config = _Config()

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        matches = (self.timesteps == timestep).nonzero().reshape(-1)
        if matches.numel() != 1:
            raise ValueError("unknown timestep")
        return int(matches.item())


def _transition(
    scheduler: _Scheduler,
    *,
    dtype: torch.dtype = torch.float32,
) -> TransitionInput:
    x_t = torch.tensor(
        [
            [[0.25, -0.5], [0.75, 0.125]],
            [[1.0, 0.75], [-0.25, 0.5]],
        ],
        dtype=dtype,
    )
    prediction = torch.tensor(
        [
            [[0.1, 0.2], [-0.3, 0.4]],
            [[-0.2, 0.05], [0.3, -0.1]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    indices = torch.tensor([0, 1], dtype=torch.int64)
    return TransitionInput(
        x_t=x_t,
        model_prediction=prediction,
        t=scheduler.timesteps.index_select(0, indices),
        t_next=scheduler.timesteps.index_select(
            0,
            torch.tensor([1, 2], dtype=torch.int64),
        ),
        mask=torch.tensor([True, True]),
        transition_index=indices,
        condition_identity=("none", "none"),
        guidance_identity=("cfg:1", "cfg:1"),
        storage_dtype_identity=(str(dtype), str(dtype)),
        quantization_identity=("none", "none"),
    )


def test_sd3_fp32_sample_and_arbitrary_action_match_legacy_kernel() -> None:
    scheduler = _Scheduler()
    transition = _transition(scheduler)
    dynamics = SD3FlowSDEDynamics(
        SD3ScheduleReplayState.from_scheduler(scheduler),
    )

    legacy_generator = torch.Generator().manual_seed(1729)
    dynamics_generator = torch.Generator().manual_seed(1729)
    legacy_sample, legacy_lp, legacy_mean, legacy_std = sd3_sde_step_with_logprob(
        scheduler,
        transition.model_prediction,
        transition.t,
        transition.x_t,
        generator=legacy_generator,
    )
    output = dynamics.sample_transition(
        transition,
        generator=dynamics_generator,
    )

    torch.testing.assert_close(output.mean, legacy_mean, rtol=0, atol=0)
    torch.testing.assert_close(output.std, legacy_std, rtol=0, atol=0)
    torch.testing.assert_close(output.sampled_next, legacy_sample, rtol=0, atol=0)
    torch.testing.assert_close(output.log_prob, legacy_lp, rtol=0, atol=0)

    arbitrary_action = torch.tensor(
        [
            [[-0.25, 0.0], [0.5, 0.75]],
            [[0.125, -0.5], [0.375, 0.25]],
        ],
        dtype=torch.float32,
    )
    _legacy_action, legacy_action_lp, action_mean, action_std = (
        sd3_sde_step_with_logprob(
            scheduler,
            transition.model_prediction,
            transition.t,
            transition.x_t,
            prev_sample=arbitrary_action,
        )
    )
    replay_lp = dynamics.transition_log_prob(transition, arbitrary_action)
    torch.testing.assert_close(action_mean, legacy_mean, rtol=0, atol=0)
    torch.testing.assert_close(action_std, legacy_std, rtol=0, atol=0)
    torch.testing.assert_close(replay_lp, legacy_action_lp, rtol=0, atol=0)

    replay_lp.sum().backward()
    assert transition.model_prediction.grad is not None
    assert bool(torch.isfinite(transition.model_prediction.grad).all())


def test_sd3_policy_metadata_exposes_the_exact_row_scalar_transition_std() -> None:
    scheduler = _Scheduler()
    transition = _transition(scheduler)
    dynamics = SD3FlowSDEDynamics(
        SD3ScheduleReplayState.from_scheduler(scheduler),
    )

    stats = dynamics.transition_mean_std(transition)
    metadata = dynamics.policy_metadata(transition, stats)

    assert metadata.rectification_coefficient is None
    assert metadata.transition_std_dev is not None
    torch.testing.assert_close(
        metadata.transition_std_dev,
        stats.std.reshape(transition.batch_size, -1)[:, 0],
        rtol=0,
        atol=0,
    )
    assert not metadata.transition_std_dev.requires_grad
    assert metadata.transition_std_dev.grad_fn is None


def test_sd3_explicit_ode_uses_frozen_sigma_dt_including_terminal_pair() -> None:
    scheduler = _Scheduler()
    state = SD3ScheduleReplayState.from_scheduler(scheduler)
    dynamics = SD3FlowSDEDynamics(state)
    original = _transition(scheduler)
    indices = torch.tensor([0, 2], dtype=torch.int64)
    transition = TransitionInput(
        x_t=original.x_t,
        model_prediction=original.model_prediction,
        t=scheduler.timesteps.index_select(0, indices),
        t_next=torch.stack((scheduler.timesteps[1], scheduler.timesteps.new_zeros(()))),
        mask=original.mask,
        transition_index=indices,
        condition_identity=original.condition_identity,
        guidance_identity=original.guidance_identity,
        storage_dtype_identity=original.storage_dtype_identity,
        quantization_identity=original.quantization_identity,
    )

    output = dynamics.deterministic_ode_step(transition)
    expected_dt = scheduler.sigmas.index_select(0, indices + 1) - (
        scheduler.sigmas.index_select(0, indices)
    )
    expected = transition.x_t + transition.model_prediction.detach() * (
        expected_dt.reshape(transition.batch_size, 1, 1)
    )

    assert transition.t_next[-1].item() == pytest.approx(0.0)
    torch.testing.assert_close(output.dt, expected_dt, rtol=0, atol=0)
    torch.testing.assert_close(output.next_state, expected, rtol=0, atol=0)
    assert output.next_state.shape == transition.x_t.shape
    assert output.next_state.dtype == transition.x_t.dtype
    assert output.next_state.device == transition.x_t.device
    assert not output.next_state.requires_grad
    assert output.next_state.grad_fn is None
    assert not torch.equal(
        output.next_state,
        dynamics.transition_mean_std(transition).mean,
    )


def test_sd3_replay_state_preserves_fractional_values_and_legacy_payload() -> None:
    scheduler = _Scheduler()
    original_timesteps = scheduler.timesteps.clone()
    original_sigmas = scheduler.sigmas.clone()
    state = SD3ScheduleReplayState.from_scheduler(
        scheduler,
        expected_steps=3,
    )
    scheduler.timesteps.zero_()
    scheduler.sigmas.zero_()

    assert state.timesteps.dtype == torch.float32
    assert state.sigmas.dtype == torch.float32
    assert torch.equal(state.timesteps, original_timesteps)
    assert torch.equal(state.sigmas, original_sigmas)
    assert state.timesteps[1].item() == original_timesteps[1].item()

    exposed = state.timesteps
    exposed.zero_()
    assert torch.equal(state.timesteps, original_timesteps)

    payload = state.to_recompute_payload(
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert set(payload) == {
        "sd3_scheduler_timesteps",
        "sd3_scheduler_sigmas",
    }
    assert payload["sd3_scheduler_timesteps"].dtype == torch.float32
    assert torch.equal(payload["sd3_scheduler_timesteps"][1], original_timesteps)

    restored = SD3ScheduleReplayState.from_recompute_payload(
        payload,
        batch_size=2,
    )
    assert restored.timesteps.dtype == torch.float32
    assert torch.equal(restored.timesteps, original_timesteps)
    assert torch.equal(restored.sigmas, original_sigmas)

    drifted = {name: value.clone() for name, value in payload.items()}
    drifted["sd3_scheduler_sigmas"][1, 1] += 0.01
    with pytest.raises(DynamicsContractError, match="rows must share"):
        SD3ScheduleReplayState.from_recompute_payload(
            drifted,
            batch_size=2,
        )
    with pytest.raises(DynamicsContractError, match="incomplete"):
        SD3ScheduleReplayState.from_recompute_payload(
            {"sd3_scheduler_timesteps": payload["sd3_scheduler_timesteps"]},
            batch_size=2,
        )


def test_sd3_record_storage_dtype_round_trip_keeps_replay_timestep_dtype() -> None:
    scheduler = _Scheduler()
    transition = _transition(scheduler)
    dynamics = SD3FlowSDEDynamics(
        SD3ScheduleReplayState.from_scheduler(scheduler),
    )
    output = dynamics.sample_transition(
        transition,
        generator=torch.Generator().manual_seed(11),
    )
    record = dynamics.make_record(
        transition,
        output,
        conditioned_next=output.sampled_next,
        likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
    )

    widened = record.to("cpu", dtype=torch.float64)
    restored = widened.to("cpu", dtype=torch.float32)
    assert widened.storage_dtype_identity == ("torch.float64", "torch.float64")
    assert restored.storage_dtype_identity == ("torch.float32", "torch.float32")
    assert widened.t.dtype == torch.float32
    assert restored.t.dtype == torch.float32
    torch.testing.assert_close(restored.x_t, record.x_t, rtol=0, atol=0)
    torch.testing.assert_close(
        restored.sampled_action,
        record.sampled_action,
        rtol=0,
        atol=0,
    )


def test_sd3_invalid_dtype_schedule_and_diffusion_inputs_fail_fast() -> None:
    scheduler = _Scheduler()
    dynamics = SD3FlowSDEDynamics(
        SD3ScheduleReplayState.from_scheduler(scheduler),
    )
    with pytest.raises(DynamicsContractError, match="requires FP32"):
        dynamics.sample_transition(
            _transition(scheduler, dtype=torch.float64),
            generator=torch.Generator().manual_seed(1),
        )

    with pytest.raises(DynamicsContractError, match="dt is negative"):
        SD3ScheduleReplayState(
            torch.tensor([9.5, 4.25]),
            torch.tensor([1.0, 1.0, 0.0]),
        )
    with pytest.raises(DynamicsContractError, match="non-negative"):
        SD3ScheduleReplayState(
            torch.tensor([9.5, 4.25]),
            torch.tensor([1.0, 0.5, -0.1]),
        )
    bad_denominator = SD3ScheduleReplayState(
        scheduler.timesteps,
        torch.tensor([1.2, 1.0, 0.5, 0.0]),
    )
    bad_scheduler = _Scheduler()
    bad_scheduler.timesteps = bad_denominator.timesteps
    bad_scheduler.sigmas = bad_denominator.sigmas
    with pytest.raises(DynamicsContractError, match="denominator"):
        SD3FlowSDEDynamics(bad_denominator).sample_transition(
            _transition(bad_scheduler),
            generator=torch.Generator().manual_seed(1),
        )
    with pytest.raises(ValueError, match="noise_level"):
        FlowSDEConfig(noise_level=0.0)
    with pytest.raises(DynamicsContractError, match="std must be strictly positive"):
        SD3FlowSDEDynamics(
            SD3ScheduleReplayState.from_scheduler(scheduler),
            config=FlowSDEConfig(noise_level=1.0e-45),
        ).sample_transition(
            _transition(scheduler),
            generator=torch.Generator().manual_seed(1),
        )
