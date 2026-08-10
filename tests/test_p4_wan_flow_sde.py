"""Parity and typed schedule tests for Wan Flash/World-R1 Dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from visual_rl.algorithms.dynamics.config import (
    WanFlowSDEConfig,
    WanFlowSDEProfile,
)
from visual_rl.algorithms.dynamics.interface import (
    DynamicsContractError,
    TransitionInput,
)
from visual_rl.algorithms.dynamics.transition import (
    wan_sde_step_with_logprob,
)
from visual_rl.algorithms.dynamics.wan_flow_sde import (
    WanFlowSDEDynamics,
    WanScheduleReplayState,
)


@dataclass
class _Config:
    stochastic_sampling: bool


class _Scheduler:
    def __init__(self, *, stochastic_sampling: bool) -> None:
        self.timesteps = torch.tensor(
            [900.5, 400.25],
            dtype=torch.float32,
        )
        self.sigmas = torch.tensor(
            [1.0, 0.6, 0.1],
            dtype=torch.float32,
        )
        self.config = _Config(stochastic_sampling=stochastic_sampling)

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        matches = (self.timesteps == timestep).nonzero().reshape(-1)
        if matches.numel() != 1:
            raise ValueError("unknown timestep")
        return int(matches.item())


def _wan_config(
    profile: WanFlowSDEProfile,
    *,
    stochastic_sampling: bool = True,
) -> WanFlowSDEConfig:
    return WanFlowSDEConfig.from_mapping(
        {
            "profile": profile.value,
            "likelihood_semantics": "exact_env_action",
            "replay_target": "sampled_action",
            "stochastic_sampling": stochastic_sampling,
        },
        context=None,
    )


def _transition(
    scheduler: _Scheduler,
    *,
    dtype: torch.dtype = torch.float32,
) -> TransitionInput:
    x_t = torch.tensor(
        [
            [[[0.25, -0.5], [0.75, 0.125]]],
            [[[1.0, 0.75], [-0.25, 0.5]]],
        ],
        dtype=dtype,
    )
    prediction = torch.tensor(
        [
            [[[0.1, 0.2], [-0.3, 0.4]]],
            [[[-0.2, 0.05], [0.3, -0.1]]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    indices = torch.tensor([0, 1], dtype=torch.int64)
    return TransitionInput(
        x_t=x_t,
        model_prediction=prediction,
        t=scheduler.timesteps.index_select(0, indices),
        t_next=torch.stack((scheduler.timesteps[1], scheduler.timesteps.new_zeros(()))),
        mask=torch.tensor([True, True]),
        transition_index=indices,
        condition_identity=("none", "camera-v1"),
        guidance_identity=("cfg:1", "cfg:5"),
        storage_dtype_identity=(str(dtype), str(dtype)),
        quantization_identity=("none", "none"),
    )


@pytest.mark.parametrize(
    ("profile", "stochastic_sampling"),
    [
        (WanFlowSDEProfile.FLASH, True),
        (WanFlowSDEProfile.STANDARD, True),
        (WanFlowSDEProfile.CONDITIONED, True),
    ],
)
def test_wan_stochastic_sample_and_action_replay_match_legacy_kernel(
    profile: WanFlowSDEProfile,
    stochastic_sampling: bool,
) -> None:
    scheduler = _Scheduler(stochastic_sampling=stochastic_sampling)
    transition = _transition(scheduler)
    dynamics = WanFlowSDEDynamics(
        WanScheduleReplayState.from_scheduler(scheduler),
        config=_wan_config(profile),
    )

    legacy_generator = torch.Generator().manual_seed(2027)
    dynamics_generator = torch.Generator().manual_seed(2027)
    legacy_sample, legacy_lp, legacy_mean, legacy_std = wan_sde_step_with_logprob(
        scheduler,
        transition.model_prediction,
        transition.t,
        transition.x_t,
        profile=profile,
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
            [[[0.0, 0.25], [-0.125, 0.5]]],
            [[[0.75, -0.25], [0.125, 0.375]]],
        ],
        dtype=torch.float32,
    )
    _legacy_action, legacy_action_lp, action_mean, action_std = (
        wan_sde_step_with_logprob(
            scheduler,
            transition.model_prediction,
            transition.t,
            transition.x_t,
            profile=profile,
            prev_sample=arbitrary_action,
        )
    )
    replay_lp = dynamics.transition_log_prob(transition, arbitrary_action)
    torch.testing.assert_close(action_mean, legacy_mean, rtol=0, atol=0)
    torch.testing.assert_close(action_std, legacy_std, rtol=0, atol=0)
    torch.testing.assert_close(replay_lp, legacy_action_lp, rtol=0, atol=0)


def test_flash_policy_metadata_matches_the_frozen_kernel_coefficient() -> None:
    scheduler = _Scheduler(stochastic_sampling=True)
    transition = _transition(scheduler)
    dynamics = WanFlowSDEDynamics(
        WanScheduleReplayState.from_scheduler(scheduler),
        config=_wan_config(WanFlowSDEProfile.FLASH),
    )

    stats = dynamics.transition_mean_std(transition)
    metadata = dynamics.policy_metadata(transition, stats)
    legacy = wan_sde_step_with_logprob(
        scheduler,
        transition.model_prediction,
        transition.t,
        transition.x_t,
        profile=WanFlowSDEProfile.FLASH,
        prev_sample=transition.x_t,
        return_flash_coefficient=True,
    )
    legacy_coefficient = legacy[5].reshape(transition.batch_size, -1)[:, 0]

    assert metadata.transition_std_dev is None
    assert metadata.rectification_coefficient is not None
    torch.testing.assert_close(
        metadata.rectification_coefficient,
        legacy_coefficient,
        rtol=0,
        atol=0,
    )
    assert not metadata.rectification_coefficient.requires_grad
    assert metadata.rectification_coefficient.grad_fn is None


def test_conditioned_profile_does_not_invent_flash_statistics() -> None:
    scheduler = _Scheduler(stochastic_sampling=True)
    transition = _transition(scheduler)
    dynamics = WanFlowSDEDynamics(
        WanScheduleReplayState.from_scheduler(scheduler),
        config=_wan_config(WanFlowSDEProfile.CONDITIONED),
    )

    metadata = dynamics.policy_metadata(
        transition,
        dynamics.transition_mean_std(transition),
    )

    assert metadata.transition_std_dev is None
    assert metadata.rectification_coefficient is None


def test_standard_and_conditioned_profiles_share_math_but_not_identity() -> None:
    scheduler = _Scheduler(stochastic_sampling=True)
    transition = _transition(scheduler)
    state = WanScheduleReplayState.from_scheduler(scheduler)
    standard = WanFlowSDEDynamics(
        state,
        config=_wan_config(WanFlowSDEProfile.STANDARD),
    )
    conditioned = WanFlowSDEDynamics(
        state,
        config=_wan_config(WanFlowSDEProfile.CONDITIONED),
    )

    standard_output = standard.sample_transition(
        transition,
        generator=torch.Generator().manual_seed(901),
    )
    conditioned_output = conditioned.sample_transition(
        transition,
        generator=torch.Generator().manual_seed(901),
    )

    torch.testing.assert_close(
        standard_output.sampled_next,
        conditioned_output.sampled_next,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        standard_output.log_prob,
        conditioned_output.log_prob,
        rtol=0,
        atol=0,
    )
    assert standard.dynamics_config_identity != conditioned.dynamics_config_identity


@pytest.mark.parametrize("profile", tuple(WanFlowSDEProfile))
def test_wan_explicit_ode_uses_frozen_sigma_dt_including_terminal_pair(
    profile: WanFlowSDEProfile,
) -> None:
    scheduler = _Scheduler(stochastic_sampling=True)
    transition = _transition(scheduler)
    dynamics = WanFlowSDEDynamics(
        WanScheduleReplayState.from_scheduler(scheduler),
        config=_wan_config(profile),
    )

    output = dynamics.deterministic_ode_step(transition)
    expected_dt = scheduler.sigmas[1:] - scheduler.sigmas[:-1]
    expected = transition.x_t + transition.model_prediction.detach() * (
        expected_dt.reshape(transition.batch_size, 1, 1, 1)
    )

    assert transition.t_next[-1].item() == pytest.approx(0.0)
    torch.testing.assert_close(output.dt, expected_dt, rtol=0, atol=0)
    torch.testing.assert_close(output.next_state, expected, rtol=0, atol=0)
    assert output.next_state.shape == transition.x_t.shape
    assert output.next_state.dtype == transition.x_t.dtype
    assert output.next_state.device == transition.x_t.device
    assert not output.next_state.requires_grad
    assert output.next_state.grad_fn is None
    if profile is WanFlowSDEProfile.FLASH:
        assert not torch.equal(
            output.next_state,
            dynamics.transition_mean_std(transition).mean,
        )


@pytest.mark.parametrize(
    "profile",
    (WanFlowSDEProfile.STANDARD, WanFlowSDEProfile.CONDITIONED),
)
def test_non_flash_deterministic_scheduler_matches_mean_sample_and_log_prob(
    profile: WanFlowSDEProfile,
) -> None:
    scheduler = _Scheduler(stochastic_sampling=False)
    transition = _transition(scheduler)
    dynamics = WanFlowSDEDynamics(
        WanScheduleReplayState.from_scheduler(scheduler),
        config=_wan_config(profile, stochastic_sampling=False),
    )

    legacy_sample, legacy_lp, legacy_mean, legacy_std = wan_sde_step_with_logprob(
        scheduler,
        transition.model_prediction,
        transition.t,
        transition.x_t,
        profile=profile,
        generator=torch.Generator().manual_seed(41),
    )
    output = dynamics.sample_transition(
        transition,
        generator=torch.Generator().manual_seed(41),
    )
    assert dynamics.scheduler_is_deterministic
    torch.testing.assert_close(output.mean, legacy_mean, rtol=0, atol=0)
    torch.testing.assert_close(output.std, legacy_std, rtol=0, atol=0)
    torch.testing.assert_close(output.sampled_next, legacy_sample, rtol=0, atol=0)
    torch.testing.assert_close(output.sampled_next, output.mean, rtol=0, atol=0)
    torch.testing.assert_close(output.log_prob, legacy_lp, rtol=0, atol=0)

    arbitrary_action = output.mean + 0.125
    _legacy_action, legacy_action_lp, _mean, _std = wan_sde_step_with_logprob(
        scheduler,
        transition.model_prediction,
        transition.t,
        transition.x_t,
        profile=profile,
        prev_sample=arbitrary_action,
    )
    torch.testing.assert_close(
        dynamics.transition_log_prob(transition, arbitrary_action),
        legacy_action_lp,
        rtol=0,
        atol=0,
    )


def test_wan_schedule_state_round_trip_never_casts_fractional_timesteps() -> None:
    scheduler = _Scheduler(stochastic_sampling=True)
    state = WanScheduleReplayState.from_scheduler(
        scheduler,
        expected_steps=2,
    )

    assert state.timesteps.dtype == torch.float32
    assert state.timesteps.dtype != torch.int64
    assert torch.equal(state.timesteps, scheduler.timesteps)
    assert state.timesteps[0].item() == scheduler.timesteps[0].item()
    payload = state.to_recompute_payload(batch_size=3, device="cpu")
    assert payload["wan_scheduler_timesteps"].dtype == torch.float32
    assert payload["wan_scheduler_stochastic_sampling"].dtype == torch.bool
    assert tuple(payload["wan_scheduler_stochastic_sampling"].shape) == (3,)

    restored = WanScheduleReplayState.from_recompute_payload(
        payload,
        batch_size=3,
    )
    assert restored.stochastic_sampling is True
    assert restored.timesteps.dtype == torch.float32
    assert torch.equal(restored.timesteps, scheduler.timesteps)
    assert torch.equal(restored.sigmas, scheduler.sigmas)

    drifted = {name: value.clone() for name, value in payload.items()}
    drifted["wan_scheduler_stochastic_sampling"][1] = False
    with pytest.raises(DynamicsContractError, match="rows must share"):
        WanScheduleReplayState.from_recompute_payload(
            drifted,
            batch_size=3,
        )


def test_wan_invalid_dtype_dt_and_schedule_identity_fail_fast() -> None:
    scheduler = _Scheduler(stochastic_sampling=True)
    dynamics = WanFlowSDEDynamics(
        WanScheduleReplayState.from_scheduler(scheduler),
        config=_wan_config(WanFlowSDEProfile.FLASH),
    )
    with pytest.raises(DynamicsContractError, match="requires FP32"):
        dynamics.sample_transition(
            _transition(scheduler, dtype=torch.float64),
            generator=torch.Generator().manual_seed(3),
        )
    with pytest.raises(DynamicsContractError, match="dt is negative"):
        WanScheduleReplayState(
            torch.tensor([900.5, 400.25]),
            torch.tensor([1.0, 1.0, 0.0]),
            stochastic_sampling=True,
        )
    with pytest.raises(DynamicsContractError, match="scheduler_identity"):
        WanScheduleReplayState(
            scheduler.timesteps,
            scheduler.sigmas,
            stochastic_sampling=True,
            scheduler_identity="",
        )
    with pytest.raises(TypeError, match="WanFlowSDEConfig"):
        WanFlowSDEDynamics(
            WanScheduleReplayState.from_scheduler(scheduler),
            config=object(),  # type: ignore[arg-type]
        )
