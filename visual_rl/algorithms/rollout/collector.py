"""Shared rollout collection over one-step policy and dynamics ports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from visual_rl.algorithms.conditioning.interface import ConditionInitialization
from visual_rl.algorithms.dynamics.interface import (
    TransitionInput,
    TransitionRecord,
    TransitionSchedule,
)
from visual_rl.algorithms.dynamics.session import DynamicsSession, PolicyStepSelection
from visual_rl.algorithms.rollout.builder import (
    RolloutStrategyResult,
    build_trajectory_batch,
    resolve_item_media_layout,
)
from visual_rl.algorithms.rollout.interface import (
    ModelForwardReplayPlan,
    RolloutComponent,
    RolloutContractError,
    RolloutExecution,
    RolloutRequest,
    _model_condition_identities,
    project_model_payload_rows,
)
from visual_rl.core.contracts.runtime import (
    ExecutionPolicyReceipt,
    PolicyTransitionRequest,
    RolloutExecutionPolicy,
)
from visual_rl.data.media import DecodedMediaBatch, DecodedMediaLayout
from visual_rl.data.samples import (
    CameraConditionBatchState,
    CameraConditionPayload,
    ConditionPayload,
    NoCondition,
    NoConditionBatchState,
    StackedSampleBatch,
)
from visual_rl.models.interface import ModelInput, ModelLatentSpec

__all__ = (
    "RolloutCollector",
    "RolloutEpisode",
    "validate_decoded_media",
    "validate_latents",
)


@dataclass(slots=True)
class RolloutEpisode:
    request: RolloutRequest
    conditioning: object
    model_condition_identity: tuple[str, ...]
    latents: Any
    conditioner_state: object | None
    condition_payloads: tuple[ConditionPayload, ...]
    condition_identity: tuple[str, ...]


class RolloutCollector(RolloutComponent, ABC):
    """Template owning trajectory control, never model-specific computation."""

    INTERFACE_VERSION = "1.0"

    def __init__(
        self,
        config: object,
        *,
        execution_policy: ExecutionPolicyReceipt,
        expected_policy_id: str,
    ) -> None:
        if type(execution_policy) is not ExecutionPolicyReceipt:
            raise TypeError("execution_policy must be an ExecutionPolicyReceipt")
        if not isinstance(expected_policy_id, str) or not expected_policy_id:
            raise ValueError("expected_policy_id must be non-empty")
        self.config = config
        self._execution_policy = execution_policy.validated_projection(
            expected_policy_id
        )

    @property
    def execution_policy(self) -> ExecutionPolicyReceipt:
        return self._execution_policy

    @property
    def rollout_execution_policy(self) -> RolloutExecutionPolicy:
        return self._execution_policy.rollout

    @property
    def selection_contract_identity(self) -> str:
        """Checkpoint identity for the rollout's policy-step selection rules.

        Concrete strategies with configurable selection semantics override
        this value.  It intentionally describes selection, not model or
        algorithm aliases.
        """

        return "visual-rl.rollout-selection.default.v1"

    @classmethod
    @abstractmethod
    def describe(cls, config: object) -> object:
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> RolloutCollector:
        if not isinstance(runtime_context, Mapping):
            raise TypeError("runtime_context must be a mapping")
        execution_policy = runtime_context.get("execution_policy")
        expected_policy_id = runtime_context.get("expected_execution_policy_id")
        if type(execution_policy) is not ExecutionPolicyReceipt:
            raise TypeError(
                "runtime_context.execution_policy must be an ExecutionPolicyReceipt"
            )
        if not isinstance(expected_policy_id, str) or not expected_policy_id:
            raise ValueError(
                "runtime_context.expected_execution_policy_id must be non-empty"
            )
        return cls(
            config,
            execution_policy=execution_policy,
            expected_policy_id=expected_policy_id,
        )

    @property
    @abstractmethod
    def num_steps(self) -> int:
        raise NotImplementedError

    def run(self, request: RolloutRequest):
        """Run one strategy and cross the item-to-batch boundary exactly once."""

        return self.run_with_snapshot(request).trajectory

    def run_with_snapshot(self, request: RolloutRequest) -> RolloutExecution:
        """Run while retaining the immutable recomputation/checkpoint boundary."""

        if not isinstance(request, RolloutRequest):
            raise TypeError("request must be a RolloutRequest")
        import torch

        with torch.no_grad():
            episode = self._prepare_episode(request)
            session = request.dynamics_session
            if session is None:
                selection = self._policy_step_selection(episode)
                session = DynamicsSession.create(
                    request.dynamics,
                    num_steps=self.num_steps,
                    device=request.latent_spec.device,
                    selection=selection,
                )
            self._validate_session(episode, session)
            result = self._execute(episode, session)
            replay = result.model_forward_replay
            if replay is None:
                replay = self._independent_forward_replay_plan(
                    episode.request.samples.batch_size
                )
            return RolloutExecution(
                trajectory=self._finalize(episode, result),
                schedule_snapshot=session.snapshot,
                encoded_conditioning=episode.conditioning,
                model_condition_identity=episode.model_condition_identity,
                model_forward_replay=replay,
            )

    def _independent_forward_replay_plan(
        self,
        batch_size: int,
    ) -> ModelForwardReplayPlan:
        return ModelForwardReplayPlan.independent(
            batch_size,
            microbatch_size=(
                self.rollout_execution_policy.forward_microbatch_size
            ),
        )

    @abstractmethod
    def _execute(
        self,
        episode: RolloutEpisode,
        session: DynamicsSession,
    ) -> RolloutStrategyResult:
        raise NotImplementedError

    @abstractmethod
    def _policy_step_selection(
        self,
        episode: RolloutEpisode,
    ) -> PolicyStepSelection:
        raise NotImplementedError

    def _validate_session(
        self,
        episode: RolloutEpisode,
        session: DynamicsSession,
    ) -> None:
        if session.dynamics is not episode.request.dynamics:
            raise RolloutContractError(
                "DynamicsSession must bind the request Dynamics instance"
            )
        if session.snapshot.num_steps != self.num_steps:
            raise RolloutContractError(
                "DynamicsSession schedule length does not match rollout num_steps"
            )
        binding = episode.request.dynamics_replay_binding
        if binding is not None:
            if binding.request.num_steps != self.num_steps:
                raise RolloutContractError(
                    "replay request length does not match rollout num_steps"
                )
            if session.snapshot.scheduler_identity != binding.scheduler_identity:
                raise RolloutContractError(
                    "DynamicsSession scheduler identity does not match replay binding"
                )

    def _prepare_episode(self, request: RolloutRequest) -> RolloutEpisode:
        conditioning = request.encoded_conditioning
        if conditioning is None:
            conditioning = request.adapter.encode(request.samples)
            model_condition_identity = _model_condition_identities(
                conditioning,
                request.samples,
            )
        else:
            model_condition_identity = request.model_condition_identity
        latents = self._prepare_initial_latents(request)
        validate_latents("prepared latents", latents, request.latent_spec)
        state = None
        payloads, identities = _condition_payloads(request.samples)
        if request.conditioner is not None:
            assert request.conditioner_latent_spec is not None
            state = request.conditioner.prepare(
                request.samples.prompts,
                request.conditioner_latent_spec,
                generator=request.generator,
            )
            initialized = request.conditioner.initialize_latents(
                latents,
                state,
                generator=request.generator,
            )
            if not isinstance(initialized, ConditionInitialization):
                raise TypeError(
                    "conditioner initialization must return ConditionInitialization"
                )
            latents = initialized.latents
            state = initialized.state
            validate_latents(
                "conditioned initial latents",
                latents,
                request.latent_spec,
            )
            if initialized.condition_payloads:
                generated_payloads = _validated_conditioner_payloads(
                    initialized.condition_payloads,
                    request.samples.batch_size,
                )
                _validate_input_condition_consistency(
                    payloads,
                    generated_payloads,
                )
                payloads = generated_payloads
                identities = _payload_identities(payloads)
            elif isinstance(
                request.samples.condition_state,
                CameraConditionBatchState,
            ):
                raise RolloutContractError(
                    "a conditioner consuming camera input must emit the exact "
                    "state-derived camera payload"
                )
        return RolloutEpisode(
            request=request,
            conditioning=conditioning,
            model_condition_identity=model_condition_identity,
            latents=latents.detach(),
            conditioner_state=state,
            condition_payloads=payloads,
            condition_identity=identities,
        )

    def _prepare_initial_latents(self, request: RolloutRequest) -> Any:
        """Create the canonical ``[B,...]`` initial latent batch.

        The default is one independent latent per canonical sample row.
        Strategies whose sampling equation introduces a separate exploration
        axis may override this hook, but must return the canonical row layout
        expected by ``request.latent_spec``.  Keeping the hook before
        Conditioner initialization also preserves the existing Conditioner
        contract over canonical rows.
        """

        return request.adapter.prepare_latents(
            request.latent_spec,
            generator=request.generator,
        )

    def _transition_input(
        self,
        episode: RolloutEpisode,
        schedule: TransitionSchedule,
        step_index: int,
        *,
        mask: Any,
    ) -> TransitionInput:
        import torch

        batch_size = episode.request.samples.batch_size
        timestep = schedule.timesteps[step_index].expand(batch_size).clone()
        next_timestep = schedule.next_timesteps[step_index].expand(batch_size).clone()
        prediction = self._partitioned_model_prediction(
            episode,
            timestep=timestep,
        )
        if not isinstance(mask, torch.Tensor):
            raise TypeError("transition mask must be a torch.Tensor")
        return TransitionInput(
            x_t=episode.latents,
            model_prediction=prediction,
            t=timestep,
            t_next=next_timestep,
            mask=mask,
            transition_index=torch.full(
                (batch_size,),
                step_index,
                dtype=torch.int64,
                device=episode.latents.device,
            ),
            condition_identity=episode.condition_identity,
            guidance_identity=episode.request.guidance_identity,
            storage_dtype_identity=(str(episode.latents.dtype),) * batch_size,
            quantization_identity=episode.request.quantization_identity,
        )

    def _partitioned_model_prediction(
        self,
        episode: RolloutEpisode,
        *,
        timestep: Any,
    ) -> Any:
        """Run bounded model forwards while preserving canonical row order."""

        import torch

        batch_size = episode.request.samples.batch_size
        replay = self._independent_forward_replay_plan(batch_size)
        values: list[Any] = []
        canonical_rows = tuple(range(batch_size))
        for rows in replay.forward_partitions or ():
            if rows == canonical_rows:
                latents = episode.latents
                chunk_timestep = timestep
                conditioning = episode.conditioning
                guidance = episode.request.guidance
            else:
                start = rows[0]
                stop = rows[-1] + 1
                if rows != tuple(range(start, stop)):
                    raise RolloutContractError(
                        "independent rollout partitions must be contiguous"
                    )
                latents = episode.latents[start:stop]
                chunk_timestep = timestep[start:stop]
                conditioning = project_model_payload_rows(
                    episode.conditioning,
                    rows,
                    label="conditioning",
                    identity_attribute="condition_identity",
                    expected_identity=tuple(
                        episode.model_condition_identity[row] for row in rows
                    ),
                    require_projection=True,
                )
                guidance = project_model_payload_rows(
                    episode.request.guidance,
                    rows,
                    label="guidance",
                    identity_attribute="guidance_identity",
                    expected_identity=tuple(
                        episode.request.guidance_identity[row] for row in rows
                    ),
                    require_projection=False,
                )
            condition_identity = tuple(
                episode.model_condition_identity[row] for row in rows
            )
            guidance_identity = tuple(
                episode.request.guidance_identity[row] for row in rows
            )
            observed_identity = getattr(
                conditioning,
                "condition_identity",
                condition_identity,
            )
            if observed_identity != condition_identity:
                raise RolloutContractError(
                    "forward partition changed model condition identity"
                )
            model_input = ModelInput(
                latents=latents,
                timestep=chunk_timestep,
                conditioning=conditioning,
                guidance=guidance,
                latent_spec=replace(
                    episode.request.latent_spec,
                    shape=(len(rows), *episode.request.latent_spec.shape[1:]),
                ),
                condition_identity=condition_identity,
                guidance_identity=guidance_identity,
            )
            chunk = episode.request.adapter.predict(model_input)
            chunk.validate_against(model_input)
            values.append(chunk.value)
        prediction = values[0] if len(values) == 1 else torch.cat(values, dim=0)
        if tuple(prediction.shape) != tuple(episode.latents.shape):
            raise RolloutContractError(
                "partitioned model prediction changed canonical latent geometry"
            )
        return prediction

    def _stochastic_step(
        self,
        episode: RolloutEpisode,
        session: DynamicsSession,
        schedule: TransitionSchedule,
        step_index: int,
        *,
        mask: Any,
        inactive_deterministic_path: bool = False,
    ) -> TransitionRecord:
        """The sole policy-step order shared by all rollout strategies."""

        import torch

        transition = self._transition_input(
            episode,
            schedule,
            step_index,
            mask=mask,
        )
        transition_port = getattr(episode.request.adapter, "transition", None)
        if callable(transition_port):
            port_result = transition_port(
                PolicyTransitionRequest(
                    mode="sample",
                    transition_input=transition,
                    transition_session=session,
                    generator=episode.request.generator,
                )
            )
            output = port_result.transition_output
        else:
            # Structural compatibility for narrow unit adapters. The
            # production stage graph always injects DefaultPolicyRuntimePort.
            output = session.sample_transition(
                transition,
                generator=episode.request.generator,
            )
        candidate = output.sampled_next
        if inactive_deterministic_path and bool((~mask).any()):
            deterministic = session.deterministic_ode_step(transition)
            latent_mask = mask.reshape(
                mask.shape[0],
                *([1] * (candidate.ndim - 1)),
            )
            candidate = torch.where(
                latent_mask,
                candidate,
                deterministic.next_state,
            )
        conditioned_next = self._after_step(
            episode,
            step_index,
            schedule.timesteps[step_index],
            candidate,
        )
        record = session.make_record(
            transition,
            output,
            conditioned_next=conditioned_next,
            likelihood_semantics=episode.request.likelihood_semantics,
        )
        episode.latents = conditioned_next.detach()
        return record

    def _deterministic_ode_step(
        self,
        episode: RolloutEpisode,
        session: DynamicsSession,
        schedule: TransitionSchedule,
        step_index: int,
        *,
        transition: TransitionInput | None = None,
    ) -> None:
        """Advance through the Dynamics ODE port, never through the SDE mean."""

        import torch

        if transition is None:
            mask = torch.ones(
                episode.request.samples.batch_size,
                dtype=torch.bool,
                device=episode.latents.device,
            )
            transition = self._transition_input(
                episode,
                schedule,
                step_index,
                mask=mask,
            )
        else:
            if not torch.equal(transition.x_t, episode.latents):
                raise RolloutContractError(
                    "reused ODE transition x_t must match the current mainline state"
                )
            expected_index = torch.full_like(
                transition.transition_index,
                step_index,
            )
            if not torch.equal(transition.transition_index, expected_index):
                raise RolloutContractError(
                    "reused ODE transition index must match the requested step"
                )
        output = session.deterministic_ode_step(transition)
        episode.latents = self._after_step(
            episode,
            step_index,
            schedule.timesteps[step_index],
            output.next_state,
        ).detach()

    def _after_step(
        self,
        episode: RolloutEpisode,
        step_index: int,
        timestep: Any,
        next_latents: Any,
    ) -> Any:
        conditioned = next_latents
        if episode.request.conditioner is not None:
            conditioned = episode.request.conditioner.after_step(
                step_index,
                timestep,
                next_latents,
                episode.conditioner_state,
            )
        validate_latents(
            "conditioned next latents",
            conditioned,
            episode.request.latent_spec,
        )
        return conditioned

    def _finalize(
        self,
        episode: RolloutEpisode,
        result: RolloutStrategyResult,
    ):
        batch_size = episode.request.samples.batch_size
        if (
            type(result.steps_by_row) is not tuple
            or len(result.steps_by_row) != batch_size
            or any(not row for row in result.steps_by_row)
        ):
            raise RolloutContractError(
                "strategy must return at least one compact policy step per batch row"
            )
        validate_latents(
            "final latents",
            result.final_latents,
            episode.request.latent_spec,
        )
        decoded = self._decode_media(episode, result.final_latents)
        media = validate_decoded_media(
            decoded,
            expected_batch_size=batch_size,
            label="decoded media",
        )
        item_layout = resolve_item_media_layout(
            episode.request.samples.task_type,
            decoded,
        )
        return build_trajectory_batch(
            request=episode.request,
            condition_payloads=episode.condition_payloads,
            result=result,
            media=media,
            media_layout=item_layout,
        )

    def _decode_media(
        self,
        episode: RolloutEpisode,
        final_latents: Any,
    ) -> DecodedMediaBatch:
        """Decode bounded row chunks and immediately release GPU video tensors."""

        import torch

        batch_size = episode.request.samples.batch_size
        configured = self.rollout_execution_policy.decode_microbatch_size
        width = batch_size if configured is None else min(configured, batch_size)
        output: Any | None = None
        output_layout: DecodedMediaLayout | None = None
        for start in range(0, batch_size, width):
            stop = min(start + width, batch_size)
            chunk_spec = replace(
                episode.request.latent_spec,
                shape=(stop - start, *episode.request.latent_spec.shape[1:]),
            )
            decoded = episode.request.adapter.decode(
                final_latents[start:stop],
                chunk_spec,
            )
            if not isinstance(decoded, DecodedMediaBatch):
                raise TypeError("adapter.decode() must return DecodedMediaBatch")
            try:
                decoded.assert_integrity()
            except (TypeError, ValueError) as exc:
                raise RolloutContractError(
                    f"decode chunk contract drift: {exc}"
                ) from exc
            if decoded.batch_size != stop - start:
                raise RolloutContractError("decode chunk changed its batch rows")
            if output_layout is None:
                output_layout = decoded.layout
            elif output_layout != decoded.layout:
                raise RolloutContractError("decode chunk layout changed across rows")
            decoded_tensor = decoded.tensor
            if not isinstance(decoded_tensor, torch.Tensor):
                raise TypeError("decoded media payload must be a torch.Tensor")
            chunk = (
                decoded_tensor.detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .contiguous()
            )
            if output is None:
                if start == 0 and stop == batch_size:
                    output = chunk
                    continue
                output = torch.empty(
                    (batch_size, *chunk.shape[1:]),
                    dtype=chunk.dtype,
                    device="cpu",
                )
            elif tuple(output.shape[1:]) != tuple(chunk.shape[1:]):
                raise RolloutContractError("decode chunk geometry changed across rows")
            output[start:stop].copy_(chunk)
        assert output is not None
        assert output_layout is not None
        return DecodedMediaBatch(tensor=output, layout=output_layout)


def validate_decoded_media(
    decoded: DecodedMediaBatch,
    *,
    expected_batch_size: int,
    label: str,
) -> Any:
    """Validate the runtime tensor requirements behind a decoded-media DTO."""

    import torch

    if not isinstance(decoded, DecodedMediaBatch):
        raise TypeError(f"{label} must be a DecodedMediaBatch")
    try:
        decoded.assert_integrity()
    except (TypeError, ValueError) as exc:
        raise RolloutContractError(f"{label} contract drift: {exc}") from exc
    if decoded.batch_size != expected_batch_size:
        raise RolloutContractError(f"{label} must preserve batch rows")
    media = decoded.tensor
    if not isinstance(media, torch.Tensor):
        raise TypeError(f"{label} payload must be a torch.Tensor")
    if not media.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    if media.requires_grad or media.grad_fn is not None:
        raise RolloutContractError(f"{label} must be detached")
    if media.device.type != "cpu":
        raise RolloutContractError(f"{label} must be offloaded to CPU before collation")
    if not bool(torch.isfinite(media).all()):
        raise RolloutContractError(f"{label} must be finite")
    return media


def validate_latents(name: str, value: Any, spec: ModelLatentSpec) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != spec.shape:
        raise RolloutContractError(f"{name} shape does not match latent_spec")
    if value.dtype != spec.dtype or value.device != spec.device:
        raise RolloutContractError(f"{name} dtype/device does not match latent_spec")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if value.requires_grad or value.grad_fn is not None:
        raise RolloutContractError(f"{name} must be detached")
    if not bool(torch.isfinite(value).all()):
        raise RolloutContractError(f"{name} must be finite")


def _condition_payloads(
    samples: StackedSampleBatch,
) -> tuple[tuple[ConditionPayload, ...], tuple[str, ...]]:
    state = samples.condition_state
    if isinstance(state, NoConditionBatchState):
        return (
            tuple(NoCondition() for _ in range(samples.batch_size)),
            ("none",) * samples.batch_size,
        )
    if isinstance(state, CameraConditionBatchState):
        payloads = tuple(
            CameraConditionPayload(
                camera_trajectory=state.camera_trajectory[index].detach(),
                conditioner_config_identity=(state.conditioner_config_identity[index]),
            )
            for index in range(samples.batch_size)
        )
        identities = tuple(payload.condition_identity for payload in payloads)
        if identities != state.row_condition_identities:
            raise RolloutContractError(
                "sample camera condition identities do not match their payloads"
            )
        return payloads, identities
    raise TypeError(f"unsupported condition batch state: {type(state).__name__}")


def _validated_conditioner_payloads(
    payloads: tuple[ConditionPayload, ...],
    batch_size: int,
) -> tuple[ConditionPayload, ...]:
    if type(payloads) is not tuple or len(payloads) != batch_size:
        raise RolloutContractError(
            "conditioner must emit exactly one condition payload per batch row"
        )
    if any(not isinstance(payload, ConditionPayload) for payload in payloads):
        raise TypeError("conditioner emitted a non-ConditionPayload value")
    for payload in payloads:
        payload.validate()
    return payloads


def _payload_identities(
    payloads: tuple[ConditionPayload, ...],
) -> tuple[str, ...]:
    identities: list[str] = []
    for payload in payloads:
        if isinstance(payload, NoCondition):
            identities.append("none")
        elif isinstance(payload, CameraConditionPayload):
            identities.append(payload.condition_identity)
        else:
            raise TypeError(f"unsupported condition payload: {type(payload).__name__}")
    return tuple(identities)


def _validate_input_condition_consistency(
    input_payloads: tuple[ConditionPayload, ...],
    generated_payloads: tuple[ConditionPayload, ...],
) -> None:
    """Reject split-brain camera state instead of silently replacing input."""

    import torch

    for row_index, (input_payload, generated_payload) in enumerate(
        zip(input_payloads, generated_payloads, strict=True)
    ):
        if isinstance(input_payload, NoCondition):
            continue
        if not isinstance(input_payload, CameraConditionPayload) or not isinstance(
            generated_payload,
            CameraConditionPayload,
        ):
            raise RolloutContractError(
                f"conditioner payload kind disagrees with input row {row_index}"
            )
        input_camera = (
            input_payload.camera_trajectory.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )
        generated_camera = (
            generated_payload.camera_trajectory.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )
        if (
            input_payload.conditioner_config_identity
            != generated_payload.conditioner_config_identity
            or input_payload.condition_identity != generated_payload.condition_identity
            or input_camera.shape != generated_camera.shape
            or not torch.equal(input_camera, generated_camera)
        ):
            raise RolloutContractError(
                f"input camera and conditioner state disagree at row {row_index}"
            )
