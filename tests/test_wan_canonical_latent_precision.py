"""Canonical Wan policy-replay coverage for the FP32 latent boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tests.support.policy_recompute_oracle import compute_full_policy_stats_oracle
from visual_rl.algorithms.dynamics.config import WanFlowSDEConfig
from visual_rl.algorithms.dynamics.session import DynamicsSession, PolicyStepSelection
from visual_rl.algorithms.dynamics.wan_flow_sde import (
    WanFlowSDEDynamics,
    WanScheduleReplayState,
)
from visual_rl.algorithms.optimization.recompute import PolicyRecomputeRequest
from visual_rl.algorithms.rollout.interface import RolloutExecution
from visual_rl.core.contracts import (
    ComputePrecision,
    LatentLayout,
    LikelihoodSemantics,
)
from visual_rl.models import ModelLatentSpec
from visual_rl.models.implementations.wan import (
    WanConditioning,
    WanConfig,
    WanT2VAdapter,
)
from visual_rl.data.samples import (
    BatchRowContext,
    ExplicitCollator,
    FullTrajectoryItem,
    TrajectoryContext,
    TrajectoryStep,
)


class _RecordingTransformer(torch.nn.Module):
    def __init__(self, received: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.received = received
        # PEFT may enumerate an FP32 LoRA parameter before Wan's frozen BF16
        # patch embedding. That ordering must not choose the activation dtype.
        self.lora_anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.patch_embedding = torch.nn.Conv3d(
            1,
            1,
            kernel_size=1,
            dtype=torch.bfloat16,
        )

    def forward(self, *, hidden_states: torch.Tensor, **_kwargs: Any):
        self.received["transformer_hidden_states"] = hidden_states.detach().clone()
        return (hidden_states + self.lora_anchor.to(hidden_states.dtype),)


class _RecordingWanDynamics(WanFlowSDEDynamics):
    def __init__(self, replay_state: WanScheduleReplayState) -> None:
        super().__init__(
            replay_state,
            config=WanFlowSDEConfig.from_mapping(
                {
                    "profile": "standard",
                    "likelihood_semantics": "exact_env_action",
                    "replay_target": "sampled_action",
                },
                context=None,
            ),
        )
        self.current_latents: list[torch.Tensor] = []
        self.action_latents: list[torch.Tensor] = []

    def transition_mean_std(self, transition):
        self.current_latents.append(transition.x_t.detach().clone())
        return super().transition_mean_std(transition)

    def _log_prob(self, transition, action_latent, stats):
        self.action_latents.append(action_latent.detach().clone())
        return super()._log_prob(transition, action_latent, stats)


def test_wan_policy_recompute_casts_only_model_activation_to_bf16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One stored FP32 transition crosses Wan predict and real Dynamics intact."""

    current = torch.tensor(
        [[[[1.0001, -2.0003]]]],
        dtype=torch.float32,
    )
    next_ = torch.tensor(
        [[[[3.0007, -4.0013]]]],
        dtype=torch.float32,
    )
    assert not torch.equal(current.to(torch.bfloat16).float(), current)
    assert not torch.equal(next_.to(torch.bfloat16).float(), next_)

    row = BatchRowContext(
        occurrence_id="occurrence-0",
        group_id="group-0",
        member_id=0,
        phase="main",
        optimizer_step=0,
        source_item_id="source-0",
    )
    trajectory = ExplicitCollator().collate_trajectories(
        (
            FullTrajectoryItem(
                context=TrajectoryContext(
                    sample_id="sample-0",
                    trajectory_id="trajectory-0",
                    batch_row=row,
                ),
                steps=(
                    TrajectoryStep(
                        x_t=current,
                        sampled_action=next_,
                        conditioned_next=next_.clone(),
                        t=torch.tensor(900.5, dtype=torch.float32),
                        t_next=torch.tensor(0.0, dtype=torch.float32),
                        old_log_prob=torch.tensor(0.0, dtype=torch.float32),
                        likelihood_semantics=(LikelihoodSemantics.EXACT_ENV_ACTION),
                        condition_identity="none",
                        guidance_identity="cfg:1.0",
                        transition_index=0,
                        storage_dtype_identity=str(torch.float32),
                    ),
                ),
                media=torch.zeros((1, 3, 1, 2), dtype=torch.float32),
                media_layout="FCHW",
            ),
        )
    )

    received: dict[str, torch.Tensor] = {}
    transformer = _RecordingTransformer(received)
    adapter = WanT2VAdapter(
        WanConfig(
            artifact_ref="test-wan",
            gradient_checkpointing=False,
            guidance_scale=1.0,
            height=8,
            width=16,
            frames=1,
            vae_tiling=False,
        ),
        artifact_path=tmp_path,
        precision=ComputePrecision.BF16,
        model_loader=None,
    )
    adapter._transformer = transformer
    monkeypatch.setattr(
        adapter,
        "_forward_prepared",
        lambda _component_name, **kwargs: transformer(**kwargs),
    )

    replay_state = WanScheduleReplayState(
        torch.tensor([900.5], dtype=torch.float32),
        torch.tensor([1.0, 0.1], dtype=torch.float32),
        stochastic_sampling=True,
        terminal_timestep=torch.tensor(0.0, dtype=torch.float32),
    )
    dynamics = _RecordingWanDynamics(replay_state)
    schedule = DynamicsSession.create(
        dynamics,
        num_steps=1,
        device="cpu",
        selection=PolicyStepSelection.all_steps(
            num_steps=1,
            generator=torch.Generator().manual_seed(7),
        ),
    ).snapshot
    model_identity = (row.identity,)
    conditioning = WanConditioning(
        prompt_embeds=torch.ones((1, 1, 1), dtype=torch.bfloat16),
        negative_prompt_embeds=None,
        condition_identity=model_identity,
    )
    rollout = RolloutExecution(
        trajectory=trajectory,
        schedule_snapshot=schedule,
        encoded_conditioning=conditioning,
        model_condition_identity=model_identity,
    )
    latent_spec = ModelLatentSpec(
        shape=(1, 1, 1, 1, 2),
        layout=LatentLayout.BCTHW,
        axis_semantics=("batch", "channel", "time", "height", "width"),
        device="cpu",
        dtype=torch.float32,
        spatial_stride=(8, 8),
        temporal_stride=4,
    )

    stats = compute_full_policy_stats_oracle(
        PolicyRecomputeRequest(
            adapter=adapter,
            dynamics=dynamics,
            rollout=rollout,
            latent_spec=latent_spec,
        )
    )
    stats.current_log_probs.sum().backward()

    hidden_states = received["transformer_hidden_states"]
    assert hidden_states.dtype is torch.bfloat16
    assert not torch.equal(hidden_states.float(), trajectory.x_t[:, 0])
    assert transformer.lora_anchor.grad is not None

    assert len(dynamics.current_latents) == 1
    assert len(dynamics.action_latents) == 1
    current_at_dynamics = dynamics.current_latents[0]
    next_at_dynamics = dynamics.action_latents[0]
    assert current_at_dynamics.dtype is torch.float32
    assert next_at_dynamics.dtype is torch.float32
    assert torch.equal(current_at_dynamics, trajectory.x_t[:, 0])
    assert torch.equal(next_at_dynamics, trajectory.scoring_target[:, 0])
    assert not torch.equal(
        current_at_dynamics.to(torch.bfloat16).float(),
        current_at_dynamics,
    )
    assert not torch.equal(
        next_at_dynamics.to(torch.bfloat16).float(),
        next_at_dynamics,
    )
