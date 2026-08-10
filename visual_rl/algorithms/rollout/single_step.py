"""Single-step control flow with explicit deterministic continuations."""

from __future__ import annotations

from visual_rl.algorithms.dynamics.session import DynamicsSession, PolicyStepSelection
from visual_rl.algorithms.rollout.builder import (
    RolloutStrategyResult,
    TrajectoryRowBuilder,
)
from visual_rl.algorithms.rollout.collector import RolloutCollector
from visual_rl.algorithms.rollout.config import SingleStepRolloutConfig
from visual_rl.algorithms.rollout.interface import RolloutContractError
from visual_rl.core.contracts import DeclaredContract
from visual_rl.core.contracts.runtime import ExecutionPolicyReceipt

__all__ = ("SingleStepRollout",)


class SingleStepRollout(RolloutCollector):
    CONFIG_TYPE = "visual_rl.algorithms.rollout.config:SingleStepRolloutConfig"

    def __init__(
        self,
        config: SingleStepRolloutConfig,
        *,
        execution_policy: ExecutionPolicyReceipt,
        expected_policy_id: str,
    ) -> None:
        if not isinstance(config, SingleStepRolloutConfig):
            raise TypeError("config must be SingleStepRolloutConfig")
        super().__init__(
            config,
            execution_policy=execution_policy,
            expected_policy_id=expected_policy_id,
        )

    @classmethod
    def describe(cls, config: object) -> DeclaredContract:
        if not isinstance(config, SingleStepRolloutConfig):
            raise TypeError("config must be SingleStepRolloutConfig")
        return config.describe_contract()

    @property
    def num_steps(self) -> int:
        return self.config.num_steps

    @property
    def selection_contract_identity(self) -> str:
        return self.config.selection_contract_identity

    def _require_supported_selection_domain(self) -> None:
        if self.config.selection_domain != "single_process":
            raise RolloutContractError(
                "global_rank_broadcast selection requires a real distributed "
                "prompt-mapping broadcast port; this runtime only implements "
                "single_process"
            )

    def _selection_keys(self, episode) -> tuple[str, ...]:
        if self.config.selection_key == "prompt":
            return tuple(episode.request.samples.prompts)
        return tuple(row.identity for row in episode.request.samples.rows)

    def _policy_step_selection(self, episode) -> PolicyStepSelection:
        self._require_supported_selection_domain()
        request_identity = episode.request.selection_contract_identity
        if request_identity and request_identity != self.selection_contract_identity:
            raise RolloutContractError(
                "rollout request selection contract does not match SingleStepRollout"
            )
        batch_size = episode.request.samples.batch_size
        selected = self.config.selected_timestep_index
        if selected is not None:
            return PolicyStepSelection.fixed(
                (selected,) * batch_size,
                num_steps=self.num_steps,
                generator=episode.request.selection_generator,
                policy=self.selection_contract_identity,
            )
        return PolicyStepSelection.uniform_from_candidates_by_key(
            num_steps=self.num_steps,
            candidate_indices=self.config.candidate_indices,
            keys=self._selection_keys(episode),
            generator=episode.request.selection_generator,
            policy=self.selection_contract_identity,
        )

    def _validate_session(self, episode, session: DynamicsSession) -> None:
        self._require_supported_selection_domain()
        super()._validate_session(episode, session)
        request_identity = episode.request.selection_contract_identity
        if request_identity and request_identity != self.selection_contract_identity:
            raise RolloutContractError(
                "rollout request selection contract does not match SingleStepRollout"
            )
        if session.snapshot.selection_policy != self.selection_contract_identity:
            raise ValueError(
                "single-step session selection policy identity does not match config"
            )
        selected = session.snapshot.selected_policy_step_indices
        if len(selected) != episode.request.samples.batch_size:
            raise ValueError("single-step session must select one step per batch row")
        candidates = frozenset(self.config.candidate_indices)
        if any(item not in candidates for item in selected):
            raise ValueError("single-step session selected a non-candidate step")
        if self.config.selection_key == "prompt":
            selected_by_prompt: dict[str, int] = {}
            for prompt, step in zip(
                episode.request.samples.prompts,
                selected,
                strict=True,
            ):
                previous = selected_by_prompt.setdefault(prompt, step)
                if previous != step:
                    raise ValueError(
                        "single-step session must share one step for equal prompts"
                    )
        if self.config.selected_timestep_index is not None and any(
            item != self.config.selected_timestep_index for item in selected
        ):
            raise ValueError(
                "single-step session conflicts with fixed selected_timestep_index"
            )

    def _execute(
        self,
        episode,
        session: DynamicsSession,
    ) -> RolloutStrategyResult:
        import torch

        schedule = session.transition_schedule(device=episode.latents.device)
        selected = torch.tensor(
            session.snapshot.selected_policy_step_indices,
            dtype=torch.int64,
            device=episode.latents.device,
        )
        rows = TrajectoryRowBuilder(episode.request.samples.batch_size)
        for step_index in range(schedule.num_steps):
            active = selected == step_index
            if bool(active.any()):
                record = self._stochastic_step(
                    episode,
                    session,
                    schedule,
                    step_index,
                    mask=active,
                    inactive_deterministic_path=True,
                )
                active_rows = tuple(
                    int(row_index)
                    for row_index in active.nonzero().reshape(-1).tolist()
                )
                rows.append_record(
                    record,
                    row_indices=active_rows,
                    storage_device=(
                        self.rollout_execution_policy.trajectory_storage_device
                    ),
                )
            else:
                self._deterministic_ode_step(
                    episode,
                    session,
                    schedule,
                    step_index,
                )
        return RolloutStrategyResult(
            strategy="single-step",
            steps_by_row=rows.freeze(expected_steps_per_row=1),
            final_latents=episode.latents.detach(),
            selected_timestep_index=tuple(
                int(item) for item in selected.to(device="cpu").tolist()
            ),
            selection_policy_identity=session.snapshot.selection_policy,
            selection_mapping_identity=session.snapshot.randomness_identity,
        )
