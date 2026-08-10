"""Full-trajectory control flow: store every physical policy transition."""

from __future__ import annotations

from visual_rl.algorithms.dynamics.session import DynamicsSession, PolicyStepSelection
from visual_rl.algorithms.rollout.builder import (
    RolloutStrategyResult,
    TrajectoryRowBuilder,
)
from visual_rl.algorithms.rollout.collector import RolloutCollector
from visual_rl.algorithms.rollout.config import FullTrajectoryRolloutConfig
from visual_rl.core.contracts import DeclaredContract
from visual_rl.core.contracts.runtime import ExecutionPolicyReceipt

__all__ = ("FullTrajectoryRollout",)


class FullTrajectoryRollout(RolloutCollector):
    CONFIG_TYPE = "visual_rl.algorithms.rollout.config:FullTrajectoryRolloutConfig"

    def __init__(
        self,
        config: FullTrajectoryRolloutConfig,
        *,
        execution_policy: ExecutionPolicyReceipt,
        expected_policy_id: str,
    ) -> None:
        if not isinstance(config, FullTrajectoryRolloutConfig):
            raise TypeError("config must be FullTrajectoryRolloutConfig")
        super().__init__(
            config,
            execution_policy=execution_policy,
            expected_policy_id=expected_policy_id,
        )

    @classmethod
    def describe(cls, config: object) -> DeclaredContract:
        if not isinstance(config, FullTrajectoryRolloutConfig):
            raise TypeError("config must be FullTrajectoryRolloutConfig")
        return config.describe_contract()

    @property
    def num_steps(self) -> int:
        return self.config.num_steps

    def _policy_step_selection(self, episode) -> PolicyStepSelection:
        return PolicyStepSelection.all_steps(
            num_steps=self.num_steps,
            generator=episode.request.selection_generator,
        )

    def _validate_session(self, episode, session: DynamicsSession) -> None:
        super()._validate_session(episode, session)
        if session.snapshot.selection_policy != "all" or (
            session.snapshot.selected_policy_step_indices
            != tuple(range(self.num_steps))
        ):
            raise ValueError("full-trajectory session must select every schedule step")

    def _execute(
        self,
        episode,
        session: DynamicsSession,
    ) -> RolloutStrategyResult:
        import torch

        schedule = session.transition_schedule(device=episode.latents.device)
        mask = torch.ones(
            episode.request.samples.batch_size,
            dtype=torch.bool,
            device=episode.latents.device,
        )
        rows = TrajectoryRowBuilder(episode.request.samples.batch_size)
        for step_index in range(schedule.num_steps):
            record = self._stochastic_step(
                episode,
                session,
                schedule,
                step_index,
                mask=mask,
            )
            rows.append_record(
                record,
                storage_device=(
                    self.rollout_execution_policy.trajectory_storage_device
                ),
            )
        return RolloutStrategyResult(
            strategy="full-trajectory",
            strategy_identity=str(self.num_steps),
            steps_by_row=rows.freeze(expected_steps_per_row=self.num_steps),
            final_latents=episode.latents.detach(),
        )
