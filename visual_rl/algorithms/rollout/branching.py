"""TempFlow branching control flow plus a single-point branch ablation."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from visual_rl.algorithms.dynamics.interface import TransitionInput
from visual_rl.algorithms.dynamics.session import (
    DynamicsSession,
    PolicyStepSelection,
)
from visual_rl.algorithms.rollout.builder import (
    RolloutStrategyResult,
    TrajectoryRowBuilder,
    resolve_item_media_layout,
)
from visual_rl.algorithms.rollout.collector import (
    RolloutCollector,
    RolloutEpisode,
    validate_decoded_media,
    validate_latents,
)
from visual_rl.algorithms.rollout.config import BranchingRolloutConfig
from visual_rl.algorithms.rollout.interface import (
    ModelForwardReplayPlan,
    RolloutContractError,
    project_model_payload_rows,
)
from visual_rl.core.contracts import DeclaredContract
from visual_rl.core.contracts.runtime import (
    ExecutionPolicyReceipt,
    PolicyTransitionRequest,
)
from visual_rl.data.media import DecodedMediaBatch
from visual_rl.models.interface import BatchProjectableModelPayload, ModelInput

__all__ = ("BranchingRollout",)


class BranchingRollout(RolloutCollector):
    CONFIG_TYPE = "visual_rl.algorithms.rollout.config:BranchingRolloutConfig"

    def __init__(
        self,
        config: BranchingRolloutConfig,
        *,
        execution_policy: ExecutionPolicyReceipt,
        expected_policy_id: str,
    ) -> None:
        if not isinstance(config, BranchingRolloutConfig):
            raise TypeError("config must be BranchingRolloutConfig")
        super().__init__(
            config,
            execution_policy=execution_policy,
            expected_policy_id=expected_policy_id,
        )

    @classmethod
    def describe(cls, config: object) -> DeclaredContract:
        if not isinstance(config, BranchingRolloutConfig):
            raise TypeError("config must be BranchingRolloutConfig")
        return config.describe_contract()

    @property
    def selection_contract_identity(self) -> str:
        return self.config.selection_contract_identity

    @property
    def num_steps(self) -> int:
        return self.config.num_steps

    def _prepare_initial_latents(self, request) -> Any:
        """Draw ``B0`` prompt latents before expanding the K exploration axis."""

        import torch

        groups = self._group_row_indices(request.samples)
        base_spec = replace(
            request.latent_spec,
            shape=(len(groups), *request.latent_spec.shape[1:]),
        )
        base_latents = request.adapter.prepare_latents(
            base_spec,
            generator=request.generator,
        )
        validate_latents(
            "branch base-prompt latents",
            base_latents,
            base_spec,
        )
        group_position = {
            group_id: position for position, group_id in enumerate(groups)
        }
        row_to_group = torch.tensor(
            [group_position[row.group_id] for row in request.samples.rows],
            dtype=torch.int64,
            device=base_latents.device,
        )
        return base_latents.index_select(0, row_to_group).clone().detach()

    def _policy_step_selection(self, episode) -> PolicyStepSelection:
        self._validate_branch_rows(episode)
        request_identity = episode.request.selection_contract_identity
        if request_identity and request_identity != self.selection_contract_identity:
            raise RolloutContractError(
                "rollout request selection contract does not match BranchingRollout"
            )
        assert self.config.branch_topology is not None
        if self.config.branch_topology.kind == "every_policy_timestep":
            return PolicyStepSelection.fixed(
                tuple(range(self.num_steps - 1)),
                num_steps=self.num_steps,
                generator=episode.request.selection_generator,
                policy=self.selection_contract_identity,
            )
        batch_size = episode.request.samples.batch_size
        branch_step = self.config.branch_step_index
        if branch_step is not None:
            return PolicyStepSelection.fixed(
                (branch_step,) * batch_size,
                num_steps=self.num_steps,
                generator=episode.request.selection_generator,
                policy=self.selection_contract_identity,
            )
        return PolicyStepSelection.uniform(
            num_steps=self.num_steps,
            cardinality=batch_size,
            generator=episode.request.selection_generator,
            shared=True,
            include_final=False,
            policy=self.selection_contract_identity,
        )

    def _validate_session(self, episode, session: DynamicsSession) -> None:
        super()._validate_session(episode, session)
        request_identity = episode.request.selection_contract_identity
        if request_identity and request_identity != self.selection_contract_identity:
            raise RolloutContractError(
                "rollout request selection contract does not match BranchingRollout"
            )
        selected = session.snapshot.selected_policy_step_indices
        assert self.config.branch_topology is not None
        if self.config.branch_topology.kind == "every_policy_timestep":
            if (
                session.snapshot.selection_policy != self.selection_contract_identity
                or selected != tuple(range(self.num_steps - 1))
            ):
                raise ValueError(
                    "every_policy_timestep session must select every nonterminal "
                    "policy timestep"
                )
            return
        if session.snapshot.selection_policy != self.selection_contract_identity:
            raise ValueError(
                "branching session selection policy identity does not match config"
            )
        if len(selected) != episode.request.samples.batch_size:
            raise ValueError("branching session must select one step per batch row")
        if len(set(selected)) != 1 or selected[0] >= self.num_steps - 1:
            raise ValueError("branching session must share one non-final branch step")
        if (
            self.config.branch_step_index is not None
            and selected[0] != self.config.branch_step_index
        ):
            raise ValueError("branching session conflicts with fixed branch_step_index")

    def _execute(self, episode, session: DynamicsSession) -> RolloutStrategyResult:
        assert self.config.branch_topology is not None
        if self.config.branch_topology.kind == "every_policy_timestep":
            return self._execute_every_policy_timestep(episode, session)
        return self._execute_single_point_branch_ablation(episode, session)

    def _execute_single_point_branch_ablation(
        self,
        episode: RolloutEpisode,
        session: DynamicsSession,
    ) -> RolloutStrategyResult:
        import torch

        schedule = session.transition_schedule(device=episode.latents.device)
        groups = self._validate_branch_rows(episode)
        branch_step = session.snapshot.selected_policy_step_indices[0]

        leader_for_row = [0] * episode.request.samples.batch_size
        for indices in groups.values():
            leader = indices[0]
            for row_index in indices:
                leader_for_row[row_index] = leader
        leader_index = torch.tensor(
            leader_for_row,
            dtype=torch.int64,
            device=episode.latents.device,
        )
        episode.latents = episode.latents.index_select(0, leader_index).clone()

        # Prefix advancement is explicitly a deterministic ODE path.  It is
        # not a policy action and is therefore absent from replay records.
        for step_index in range(branch_step):
            self._deterministic_ode_step(
                episode,
                session,
                schedule,
                step_index,
            )
            episode.latents = episode.latents.index_select(
                0,
                leader_index,
            ).clone()

        prefix_by_group = {
            group_id: _tensor_identity(
                episode.latents[indices[0]],
                "shared-prefix",
                group_id,
                branch_step,
            )
            for group_id, indices in groups.items()
        }
        step_by_group = {
            group_id: hashlib.sha256(
                f"{prefix_id}\0{branch_step}\0"
                f"{schedule.timesteps[branch_step].item()}".encode()
            ).hexdigest()
            for group_id, prefix_id in prefix_by_group.items()
        }
        shared_prefix_id = tuple(
            prefix_by_group[row.group_id] for row in episode.request.samples.rows
        )
        branch_step_identity = tuple(
            step_by_group[row.group_id] for row in episode.request.samples.rows
        )

        mask = torch.ones(
            episode.request.samples.batch_size,
            dtype=torch.bool,
            device=episode.latents.device,
        )
        record = self._stochastic_step(
            episode,
            session,
            schedule,
            branch_step,
            mask=mask,
        )
        rows = TrajectoryRowBuilder(episode.request.samples.batch_size)
        rows.append_record(
            record,
            storage_device=self.rollout_execution_policy.trajectory_storage_device,
        )
        # TempFlow assigns policy credit only to the sampled branch action.
        # Media generation after it is an explicit deterministic ODE path.
        for step_index in range(branch_step + 1, schedule.num_steps):
            self._deterministic_ode_step(
                episode,
                session,
                schedule,
                step_index,
            )
        return RolloutStrategyResult(
            strategy="branching",
            steps_by_row=rows.freeze(expected_steps_per_row=1),
            final_latents=episode.latents.detach(),
            branch_topology=self.config.branch_topology,
            branch_step_index=branch_step,
            shared_prefix_id=shared_prefix_id,
            branch_step_identity=branch_step_identity,
        )

    def _execute_every_policy_timestep(
        self,
        episode: RolloutEpisode,
        session: DynamicsSession,
    ) -> RolloutStrategyResult:
        """Run the TempFlow paper axes without claiming native/GPU parity.

        At each ODE mainline state ``x_t`` the model is evaluated over the B0
        prompt axis.  That exact state/prediction is then expanded onto the
        canonical B0 x K exploration rows before stochastic SDE sampling.
        Every sampled action follows a B0 x K deterministic continuation for
        reward media.  The next policy state remains the B0 ODE mainline, not
        the prior stochastic action.
        """

        import torch

        if episode.request.conditioner is not None:
            raise RolloutContractError(
                "every_policy_timestep currently requires no latent conditioner; "
                "conditioner state cloning is not implemented"
            )
        schedule = session.transition_schedule(device=episode.latents.device)
        groups = self._validate_branch_rows(episode)
        leader_rows, row_to_mainline = self._mainline_batch_indices(
            episode,
            groups,
        )
        conditioning = episode.conditioning
        if not isinstance(conditioning, BatchProjectableModelPayload):
            raise RolloutContractError(
                "every_policy_timestep requires encoded conditioning with the "
                "BatchProjectableModelPayload port"
            )
        mainline_condition_identity = tuple(
            episode.model_condition_identity[index] for index in leader_rows
        )
        mainline_conditioning = project_model_payload_rows(
            conditioning,
            leader_rows,
            label="conditioning",
            identity_attribute="condition_identity",
            expected_identity=mainline_condition_identity,
            require_projection=True,
        )
        mainline_guidance_identity = tuple(
            episode.request.guidance_identity[index] for index in leader_rows
        )
        mainline_guidance = project_model_payload_rows(
            episode.request.guidance,
            leader_rows,
            label="guidance",
            identity_attribute="guidance_identity",
            expected_identity=mainline_guidance_identity,
            require_projection=False,
        )
        replay_rows = tuple(
            int(index) for index in row_to_mainline.to(device="cpu").tolist()
        )
        model_forward_replay = ModelForwardReplayPlan(
            forward_row_indices=leader_rows,
            row_to_forward_position=replay_rows,
            forward_partitions=self._forward_partitions(leader_rows),
        )
        replay_conditioning = project_model_payload_rows(
            mainline_conditioning,
            replay_rows,
            label="conditioning",
            identity_attribute="condition_identity",
            expected_identity=tuple(
                mainline_condition_identity[index] for index in replay_rows
            ),
            require_projection=True,
        )
        if not isinstance(replay_conditioning, BatchProjectableModelPayload):
            raise RolloutContractError(
                "expanded mainline conditioning lost the row-projection port"
            )
        replay_condition_identity = tuple(
            mainline_condition_identity[index] for index in replay_rows
        )
        if (
            getattr(
                replay_conditioning,
                "condition_identity",
                replay_condition_identity,
            )
            != replay_condition_identity
        ):
            raise RolloutContractError(
                "expanded mainline conditioning changed model condition identity"
            )
        # The stochastic policy prediction is evaluated on B0 leaders and
        # expanded onto the B0 x K exploration rows.  Freeze that exact
        # expanded conditioning as the replay snapshot too; retaining the
        # pre-expansion K-row payload would let current-policy recomputation
        # evaluate a different condition than the one that produced old_logp.
        episode.conditioning = replay_conditioning
        episode.model_condition_identity = replay_condition_identity
        mainline_latent_spec = replace(
            episode.request.latent_spec,
            shape=(len(leader_rows), *episode.request.latent_spec.shape[1:]),
        )
        mainline_latents = episode.latents.index_select(
            0,
            torch.tensor(
                leader_rows,
                dtype=torch.int64,
                device=episode.latents.device,
            ),
        ).clone()
        mainline_mask = torch.ones(
            len(leader_rows),
            dtype=torch.bool,
            device=episode.latents.device,
        )
        rows = TrajectoryRowBuilder(episode.request.samples.batch_size)
        terminal_media = []
        last_terminal_latents = None
        item_layout: str | None = None
        for step_index in range(schedule.num_steps - 1):
            transition = self._mainline_transition_input(
                episode,
                schedule,
                step_index,
                latents=mainline_latents,
                latent_spec=mainline_latent_spec,
                conditioning=mainline_conditioning,
                model_condition_identity=mainline_condition_identity,
                guidance=mainline_guidance,
                guidance_identity=mainline_guidance_identity,
                leader_rows=leader_rows,
                mask=mainline_mask,
            )

            # Locked upstream order: the B0 deterministic mainline advances
            # before the B0 x K stochastic draw consumes RNG.
            ode_output = session.deterministic_ode_step(transition)
            mainline_latents = ode_output.next_state.detach()
            episode.latents = mainline_latents.index_select(
                0,
                row_to_mainline,
            ).clone()

            expanded_transition = self._expand_mainline_transition(
                transition,
                row_to_mainline=row_to_mainline,
            )
            transition_port = getattr(episode.request.adapter, "transition", None)
            if callable(transition_port):
                port_result = transition_port(
                    PolicyTransitionRequest(
                        mode="sample",
                        transition_input=expanded_transition,
                        transition_session=session,
                        generator=episode.request.generator,
                    )
                )
                output = port_result.transition_output
            else:
                output = session.sample_transition(
                    expanded_transition,
                    generator=episode.request.generator,
                )
            conditioned_next = self._after_step(
                episode,
                step_index,
                schedule.timesteps[step_index],
                output.sampled_next,
            )
            record = session.make_record(
                expanded_transition,
                output,
                conditioned_next=conditioned_next,
                likelihood_semantics=episode.request.likelihood_semantics,
            )
            rows.append_record(
                record,
                storage_device=(
                    self.rollout_execution_policy.trajectory_storage_device
                ),
            )

            branch_episode = RolloutEpisode(
                request=episode.request,
                conditioning=episode.conditioning,
                model_condition_identity=episode.model_condition_identity,
                latents=conditioned_next.detach(),
                conditioner_state=None,
                condition_payloads=episode.condition_payloads,
                condition_identity=episode.condition_identity,
            )
            for continuation_index in range(step_index + 1, schedule.num_steps):
                self._deterministic_ode_step(
                    branch_episode,
                    session,
                    schedule,
                    continuation_index,
                )
            # The mainline stops at the last trainable state, matching
            # upstream ``timesteps[:-1]``.  Use the last completed branch for
            # the common final media field instead of inventing an extra
            # mainline transition solely for decoding.
            last_terminal_latents = branch_episode.latents.detach()
            decoded = self._decode_transition_terminal(branch_episode)
            observed_layout = resolve_item_media_layout(
                episode.request.samples.task_type,
                decoded,
            )
            if item_layout is None:
                item_layout = observed_layout
            elif item_layout != observed_layout:
                raise RolloutContractError(
                    "transition terminal media layout changed across timesteps"
                )
            terminal_media.append(decoded.tensor)

        assert item_layout is not None
        assert last_terminal_latents is not None
        return RolloutStrategyResult(
            strategy="branching",
            steps_by_row=rows.freeze(
                expected_steps_per_row=schedule.num_steps - 1,
            ),
            final_latents=last_terminal_latents,
            branch_topology=self.config.branch_topology,
            transition_terminal_media=torch.stack(
                terminal_media,
                dim=1,
            ).detach(),
            transition_terminal_media_layout={
                "CHW": "TCHW",
                "FCHW": "TFCHW",
                "FHWC": "TFHWC",
            }[item_layout],
            schedule_snapshot_identity=session.snapshot.snapshot_identity,
            model_forward_replay=model_forward_replay,
        )

    def _forward_partitions(
        self,
        rows: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        configured = self.rollout_execution_policy.forward_microbatch_size
        width = len(rows) if configured is None else min(configured, len(rows))
        return tuple(
            rows[start : start + width] for start in range(0, len(rows), width)
        )

    @staticmethod
    def _mainline_batch_indices(
        episode: RolloutEpisode,
        groups: dict[str, list[int]],
    ) -> tuple[tuple[int, ...], Any]:
        """Return ordered B0 leaders and the canonical B0 x K expansion map."""

        import torch

        leader_rows = tuple(indices[0] for indices in groups.values())
        group_position = {
            group_id: position for position, group_id in enumerate(groups)
        }
        row_to_mainline = torch.tensor(
            [group_position[row.group_id] for row in episode.request.samples.rows],
            dtype=torch.int64,
            device=episode.latents.device,
        )
        return leader_rows, row_to_mainline

    def _mainline_transition_input(
        self,
        episode: RolloutEpisode,
        schedule,
        step_index: int,
        *,
        latents: Any,
        latent_spec,
        conditioning: object,
        model_condition_identity: tuple[str, ...],
        guidance: object | None,
        guidance_identity: tuple[str, ...],
        leader_rows: tuple[int, ...],
        mask: Any,
    ) -> TransitionInput:
        """Evaluate one model-independent B0 mainline transition input."""

        import torch

        batch_size = len(leader_rows)
        timestep = schedule.timesteps[step_index].expand(batch_size).clone()
        next_timestep = schedule.next_timesteps[step_index].expand(batch_size).clone()
        predictions = []
        offset = 0
        for partition in self._forward_partitions(leader_rows):
            stop = offset + len(partition)
            positions = tuple(range(offset, stop))
            if offset == 0 and stop == batch_size:
                chunk_conditioning = conditioning
                chunk_guidance = guidance
            else:
                chunk_condition_identity = tuple(
                    model_condition_identity[position] for position in positions
                )
                chunk_conditioning = project_model_payload_rows(
                    conditioning,
                    positions,
                    label="conditioning",
                    identity_attribute="condition_identity",
                    expected_identity=chunk_condition_identity,
                    require_projection=True,
                )
                chunk_guidance = project_model_payload_rows(
                    guidance,
                    positions,
                    label="guidance",
                    identity_attribute="guidance_identity",
                    expected_identity=tuple(
                        guidance_identity[position] for position in positions
                    ),
                    require_projection=False,
                )
            chunk_condition_identity = tuple(
                model_condition_identity[position] for position in positions
            )
            chunk_guidance_identity = tuple(
                guidance_identity[position] for position in positions
            )
            model_input = ModelInput(
                latents=latents[offset:stop],
                timestep=timestep[offset:stop],
                conditioning=chunk_conditioning,
                guidance=chunk_guidance,
                latent_spec=replace(
                    latent_spec,
                    shape=(len(partition), *latent_spec.shape[1:]),
                ),
                condition_identity=chunk_condition_identity,
                guidance_identity=chunk_guidance_identity,
            )
            chunk = episode.request.adapter.predict(model_input)
            chunk.validate_against(model_input)
            predictions.append(chunk.value)
            offset = stop
        prediction = (
            predictions[0] if len(predictions) == 1 else torch.cat(predictions, dim=0)
        )
        return TransitionInput(
            x_t=latents,
            model_prediction=prediction,
            t=timestep,
            t_next=next_timestep,
            mask=mask,
            transition_index=torch.full(
                (batch_size,),
                step_index,
                dtype=torch.int64,
                device=latents.device,
            ),
            condition_identity=tuple(
                episode.condition_identity[index] for index in leader_rows
            ),
            guidance_identity=guidance_identity,
            storage_dtype_identity=(str(latents.dtype),) * batch_size,
            quantization_identity=tuple(
                episode.request.quantization_identity[index] for index in leader_rows
            ),
        )

    @staticmethod
    def _expand_mainline_transition(
        transition: TransitionInput,
        *,
        row_to_mainline: Any,
    ) -> TransitionInput:
        """Expand one B0 transition to canonical B0 x K policy rows."""

        if row_to_mainline.ndim != 1:
            raise RolloutContractError("row_to_mainline must be a 1-D tensor")
        resolved = tuple(
            int(index) for index in row_to_mainline.to(device="cpu").tolist()
        )

        def tensor_rows(value: Any) -> Any:
            return value.index_select(0, row_to_mainline.to(device=value.device))

        def identity_rows(value: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(value[index] for index in resolved)

        return TransitionInput(
            x_t=tensor_rows(transition.x_t),
            model_prediction=tensor_rows(transition.model_prediction),
            t=tensor_rows(transition.t),
            t_next=tensor_rows(transition.t_next),
            mask=tensor_rows(transition.mask),
            transition_index=tensor_rows(transition.transition_index),
            condition_identity=identity_rows(transition.condition_identity),
            guidance_identity=identity_rows(transition.guidance_identity),
            storage_dtype_identity=identity_rows(transition.storage_dtype_identity),
            quantization_identity=identity_rows(transition.quantization_identity),
        )

    @staticmethod
    def _leader_index(
        episode: RolloutEpisode,
        groups: dict[str, list[int]],
    ) -> Any:
        import torch

        leader_for_row = [0] * episode.request.samples.batch_size
        for indices in groups.values():
            leader = indices[0]
            for row_index in indices:
                leader_for_row[row_index] = leader
        return torch.tensor(
            leader_for_row,
            dtype=torch.int64,
            device=episode.latents.device,
        )

    def _decode_transition_terminal(
        self,
        episode: RolloutEpisode,
    ) -> DecodedMediaBatch:
        decoded = self._decode_media(episode, episode.latents)
        validate_decoded_media(
            decoded,
            expected_batch_size=episode.request.samples.batch_size,
            label="transition terminal media",
        )
        return decoded

    def _group_row_indices(self, samples) -> dict[str, list[int]]:
        """Validate canonical K-repeat geometry before consuming rollout RNG."""

        groups: dict[str, list[int]] = {}
        for row_index, row in enumerate(samples.rows):
            groups.setdefault(row.group_id, []).append(row_index)
        all_occurrences = tuple(row.occurrence_id for row in samples.rows)
        if len(all_occurrences) != len(set(all_occurrences)):
            raise RolloutContractError(
                "branch occurrence_id values must be unique across exploration rows"
            )
        for group_id, indices in groups.items():
            rows = tuple(samples.rows[index] for index in indices)
            if len(rows) != self.config.branch_count:
                raise RolloutContractError(
                    f"branch group {group_id!r} must contain branch_count rows"
                )
            if {row.member_id for row in rows} != set(range(self.config.branch_count)):
                raise RolloutContractError(
                    "branch member_id values must be exactly 0..branch_count-1"
                )
        return groups

    def _validate_branch_rows(
        self,
        episode: RolloutEpisode,
    ) -> dict[str, list[int]]:
        groups = self._group_row_indices(episode.request.samples)
        for indices in groups.values():
            rows = tuple(episode.request.samples.rows[index] for index in indices)
            prompts = {episode.request.samples.prompts[index] for index in indices}
            sources = {
                episode.request.samples.sources[index].source_item_id
                for index in indices
            }
            conditions = {episode.condition_identity[index] for index in indices}
            guidance = {episode.request.guidance_identity[index] for index in indices}
            quantization = {
                episode.request.quantization_identity[index] for index in indices
            }
            phases = {row.phase for row in rows}
            optimizer_steps = {row.optimizer_step for row in rows}
            if any(
                len(values) != 1
                for values in (
                    prompts,
                    sources,
                    conditions,
                    guidance,
                    quantization,
                    phases,
                    optimizer_steps,
                )
            ):
                raise RolloutContractError(
                    "branch rows must share prompt/source/condition/guidance/"
                    "quantization/phase/optimizer-step identity"
                )
            first_metadata = episode.request.samples.metadata[indices[0]]
            if any(
                episode.request.samples.metadata[index] != first_metadata
                for index in indices[1:]
            ):
                raise RolloutContractError("branch rows must share identical metadata")
        return groups


def _tensor_identity(value: Any, *parts: object) -> str:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("identity tensor must be a torch.Tensor")
    owned = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(owned.dtype).encode())
    digest.update(repr(tuple(owned.shape)).encode())
    digest.update(owned.numpy().tobytes())
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()
