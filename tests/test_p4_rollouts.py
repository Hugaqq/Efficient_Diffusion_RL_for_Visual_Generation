"""P4 rollout call graph, RNG, identity, and registry contracts."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass, replace

import pytest
import torch

from tests.support.policy_recompute_oracle import compute_full_policy_stats_oracle
from tests.support.runtime_component_loading import load_test_component
from visual_rl.algorithms.catalog import algorithm_domain_catalog_fragments
from visual_rl.algorithms.conditioning.interface import (
    ConditionInitialization,
    LatentConditioner,
    LatentSpec,
)
from visual_rl.algorithms.dynamics.interface import (
    DeterministicTransitionOutput,
    Dynamics,
    TransitionMeanStd,
    TransitionRecord,
)
from visual_rl.algorithms.dynamics.replay import (
    DynamicsReplayBinding,
    DynamicsReplayRequest,
)
from visual_rl.algorithms.dynamics.sd3_flow_sde import (
    SD3FlowSDEDynamics,
    SD3ScheduleReplayState,
)
from visual_rl.algorithms.dynamics.session import (
    DynamicsSession,
    PolicyStepSelection,
)
from visual_rl.algorithms.optimization.recompute import (
    PolicyRecomputer,
    PolicyRecomputeRequest,
)
from visual_rl.algorithms.optimization.slots import UpdateSlotPlan
from visual_rl.algorithms.rollout.branching import BranchingRollout
from visual_rl.algorithms.rollout.config import (
    ROLLOUT_CATALOG_FRAGMENT,
    BranchingRolloutConfig,
    FullTrajectoryRolloutConfig,
    SingleStepRolloutConfig,
)
from visual_rl.algorithms.rollout.full_trajectory import FullTrajectoryRollout
from visual_rl.algorithms.rollout.interface import RolloutContractError, RolloutRequest
from visual_rl.algorithms.rollout.single_step import SingleStepRollout
from visual_rl.algorithms.trainer.interface import StageValue
from visual_rl.algorithms.trainer.stages import RolloutStage
from visual_rl.core.contracts import (
    ComputePrecision,
    ConditionerContract,
    DeclaredContract,
    LatentLayout,
    LikelihoodSemantics,
    MediaKind,
    ModelContract,
    PredictionType,
    TaskKind,
    TimeCoordinate,
    TrainingMode,
)
from visual_rl.composition.recipes import (
    builtin_recipe_definitions,
    get_recipe_definition,
)
from visual_rl.composition.registry import (
    AlgorithmDeclarationResolver,
    DeclarationResolver,
    RegistryError,
    build_catalog,
)
from visual_rl.core.contracts import ExecutionPolicyReceipt
from visual_rl.core.serialization import to_plain_dict
from visual_rl.core.types import StepContext
from visual_rl.data import (
    DecodedMediaBatch,
    GroupPlacementContract,
    GroupPlacementKind,
    MultiSourceSampler,
    PeriodicPhaseSchedule,
    PhaseDefinition,
    SourceSequence,
)
from visual_rl.data.prelude import DataPlanePrelude
from visual_rl.models import (
    BatchRowProjection,
    ModelAdapter,
    ModelInput,
    ModelLatentSpec,
    ModelPrediction,
)
from visual_rl.algorithms.rewards import RewardRuntimeContext
from visual_rl.data.samples import (
    BatchRowContext,
    BranchTopology,
    CameraConditionBatchState,
    CameraConditionPayload,
    ExplicitCollator,
    SourceItemContext,
    T2IItem,
    T2VItem,
)


def test_rollout_descriptors_declare_deterministic_ode_requirements():
    full = FullTrajectoryRollout.describe(
        FullTrajectoryRolloutConfig(num_steps=3)
    ).rollout
    paper = BranchingRollout.describe(
        BranchingRolloutConfig(
            num_steps=4,
            branch_count=2,
            branch_topology=BranchTopology.every_policy_timestep(2),
        )
    ).rollout
    ablation = BranchingRollout.describe(
        BranchingRolloutConfig(
            num_steps=4,
            branch_count=2,
            branch_topology=BranchTopology.single_point_branch_ablation(2),
            branch_step_policy="uniform_intermediate",
        )
    ).rollout
    single = SingleStepRollout.describe(
        SingleStepRolloutConfig(
            selected_timestep_policy="uniform",
            num_steps=4,
        )
    ).rollout
    degenerate_single = SingleStepRollout.describe(
        SingleStepRolloutConfig(
            selected_timestep_policy="uniform",
            num_steps=1,
        )
    ).rollout

    assert full is not None and full.requires_deterministic_ode is False
    assert paper is not None and paper.requires_deterministic_ode is True
    assert ablation is not None and ablation.requires_deterministic_ode is True
    assert single is not None and single.requires_deterministic_ode is True
    assert (
        degenerate_single is not None
        and degenerate_single.requires_deterministic_ode is False
    )


class _FakeAdapter(ModelAdapter):
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.prepared_latent_shapes: list[tuple[int, ...]] = []
        self.predict_shapes: list[tuple[int, ...]] = []
        self.decode_shapes: list[tuple[int, ...]] = []

    @classmethod
    def describe(cls, config):
        del config
        return DeclaredContract(
            component_kind="model",
            component_id="fake-rollout-model",
            model=ModelContract(
                tasks=(TaskKind.T2I, TaskKind.T2V),
                output_media=(MediaKind.IMAGE, MediaKind.VIDEO),
                latent_layouts=(LatentLayout.BCHW, LatentLayout.BCTHW),
                latent_ranks=(4, 5),
                axis_semantics=(
                    ("batch", "channel", "height", "width"),
                    ("batch", "channel", "time", "height", "width"),
                ),
                prediction_types=(PredictionType.FLOW,),
                time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
                training_modes=(TrainingMode.LORA,),
                supported_precisions=(ComputePrecision.FP32,),
                provides_reference_policy=False,
            ),
        )

    @classmethod
    def from_config(cls, config, *, runtime_context):
        del config
        return cls(runtime_context["events"])

    def load_components(self, session):
        return session.freeze()

    def encode(self, batch):
        self.events.append("adapter.encode")
        return _EncodedConditioning(tuple(row.identity for row in batch.rows))

    def prepare_latents(self, latent_spec, *, generator):
        self.events.append("adapter.prepare_latents")
        self.prepared_latent_shapes.append(tuple(latent_spec.shape))
        return torch.randn(
            latent_spec.shape,
            device=latent_spec.device,
            dtype=latent_spec.dtype,
            generator=generator,
        )

    def predict(self, model_input: ModelInput):
        self.events.append("adapter.predict")
        self.predict_shapes.append(tuple(model_input.latents.shape))
        return ModelPrediction(
            value=torch.full_like(model_input.latents, 0.25),
            prediction_type=PredictionType.FLOW,
            condition_identity=model_input.condition_identity,
            guidance_identity=model_input.guidance_identity,
        )

    def decode(self, latents, latent_spec):
        self.events.append("adapter.decode")
        self.decode_shapes.append(tuple(latents.shape))
        assert tuple(latents.shape) == latent_spec.shape
        if latent_spec.layout is LatentLayout.BCTHW:
            return DecodedMediaBatch(
                tensor=latents.permute(0, 2, 1, 3, 4).detach().clone(),
                layout="BFCHW",
            )
        return DecodedMediaBatch(
            tensor=latents.detach().clone(),
            layout="BCHW",
        )


class _DecodeLayoutDriftAdapter(_FakeAdapter):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self._decode_count = 0

    def decode(self, latents, latent_spec):
        decoded = super().decode(latents, latent_spec)
        self._decode_count += 1
        if self._decode_count == 2:
            return DecodedMediaBatch(tensor=decoded.tensor, layout="BFHWC")
        return decoded


@dataclass(frozen=True)
class _EncodedConditioning:
    condition_identity: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return len(self.condition_identity)

    def project_rows(self, projection: BatchRowProjection) -> _EncodedConditioning:
        assert projection.source_batch_size == self.batch_size
        return _EncodedConditioning(projection.project_tuple(self.condition_identity))


@dataclass(frozen=True)
class _RowSensitiveConditioning:
    values: torch.Tensor
    condition_identity: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return len(self.condition_identity)

    def project_rows(
        self,
        projection: BatchRowProjection,
    ) -> _RowSensitiveConditioning:
        assert projection.source_batch_size == self.batch_size
        index = torch.tensor(
            projection.row_indices,
            dtype=torch.int64,
            device=self.values.device,
        )
        return _RowSensitiveConditioning(
            values=self.values.index_select(0, index),
            condition_identity=projection.project_tuple(self.condition_identity),
        )


@dataclass(frozen=True)
class _RowSensitiveGuidance:
    values: torch.Tensor
    guidance_identity: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return len(self.guidance_identity)

    def project_rows(
        self,
        projection: BatchRowProjection,
    ) -> _RowSensitiveGuidance:
        assert projection.source_batch_size == self.batch_size
        index = torch.tensor(
            projection.row_indices,
            dtype=torch.int64,
            device=self.values.device,
        )
        return _RowSensitiveGuidance(
            values=self.values.index_select(0, index),
            guidance_identity=projection.project_tuple(self.guidance_identity),
        )


class _RowSensitiveAdapter(_FakeAdapter):
    """Make both row content and forward batch geometry numerically observable."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.policy_scale = torch.nn.Parameter(torch.tensor(0.2))
        self.reference_scale = torch.tensor(0.1)
        self.reference_predict_shapes: list[tuple[int, ...]] = []

    def encode(self, batch):
        self.events.append("adapter.encode")
        return _RowSensitiveConditioning(
            values=torch.linspace(0.1, 0.7, batch.batch_size),
            condition_identity=tuple(row.identity for row in batch.rows),
        )

    def predict(self, model_input: ModelInput):
        self.events.append("adapter.predict")
        self.predict_shapes.append(tuple(model_input.latents.shape))
        return self._prediction(model_input, self.policy_scale)

    def predict_reference(self, model_input: ModelInput):
        self.events.append("adapter.predict_reference")
        self.reference_predict_shapes.append(tuple(model_input.latents.shape))
        return self._prediction(model_input, self.reference_scale)

    @staticmethod
    def _prediction(model_input: ModelInput, scale: torch.Tensor):
        conditioning = model_input.conditioning
        assert isinstance(conditioning, _RowSensitiveConditioning)
        assert conditioning.condition_identity == model_input.condition_identity
        offset = conditioning.values.reshape(
            conditioning.values.shape[0],
            *([1] * (model_input.latents.ndim - 1)),
        )
        batch_geometry_term = model_input.latents.new_tensor(
            0.013 * model_input.latents.shape[0]
        )
        return ModelPrediction(
            value=model_input.latents * scale + offset + batch_geometry_term,
            prediction_type=PredictionType.FLOW,
            condition_identity=model_input.condition_identity,
            guidance_identity=model_input.guidance_identity,
        )


class _GuidanceAwareAdapter(_RowSensitiveAdapter):
    """Make guidance row projection part of the observable prediction."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.predict_guidance_identities: list[tuple[str, ...]] = []
        self.predict_guidance_values: list[torch.Tensor] = []

    def predict(self, model_input: ModelInput):
        guidance = model_input.guidance
        assert isinstance(guidance, _RowSensitiveGuidance)
        assert guidance.guidance_identity == model_input.guidance_identity
        assert guidance.values.shape == (model_input.latent_spec.batch_size,)
        self.predict_guidance_identities.append(guidance.guidance_identity)
        self.predict_guidance_values.append(guidance.values.detach().clone())
        base = super().predict(model_input)
        offset = guidance.values.reshape(
            guidance.values.shape[0],
            *([1] * (model_input.latents.ndim - 1)),
        )
        return ModelPrediction(
            value=base.value + offset,
            prediction_type=base.prediction_type,
            condition_identity=base.condition_identity,
            guidance_identity=base.guidance_identity,
        )


class _OrderedRowSensitiveAdapter(_RowSensitiveAdapter):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.predict_latents: list[torch.Tensor] = []
        self.predict_condition_identities: list[tuple[str, ...]] = []

    def predict(self, model_input: ModelInput):
        self.predict_latents.append(model_input.latents.detach().clone())
        self.predict_condition_identities.append(model_input.condition_identity)
        return super().predict(model_input)


class _IdentityAwareAdapter(_FakeAdapter):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.model_input_identities: list[tuple[str, ...]] = []

    def encode(self, batch):
        self.events.append("adapter.encode")
        return _EncodedConditioning(tuple(row.identity for row in batch.rows))

    def predict(self, model_input: ModelInput):
        self.model_input_identities.append(model_input.condition_identity)
        return super().predict(model_input)


class _FakeDynamics(Dynamics):
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.mean_std_calls = 0
        self.ode_calls = 0
        self.ode_transition_shapes: list[tuple[int, ...]] = []
        self.ode_model_predictions: list[torch.Tensor] = []
        self.sample_transition_shapes: list[tuple[int, ...]] = []
        self.sample_model_predictions: list[torch.Tensor] = []

    def timesteps(self, *, num_steps: int, device):
        return torch.linspace(1.0, 0.25, num_steps, device=device)

    def terminal_timestep(self, *, device):
        return torch.tensor(-0.125, device=device)

    def add_noise(self, clean, noise, timestep):
        return clean + noise * timestep.reshape(-1, 1, 1, 1)

    def transition_mean_std(self, transition):
        self.mean_std_calls += 1
        self.events.append("dynamics.mean_std")
        shape = (
            transition.batch_size,
            *([1] * (transition.x_t.ndim - 1)),
        )
        dt = (transition.t_next - transition.t).to(transition.x_t.dtype)
        mean = transition.x_t + transition.model_prediction * dt.reshape(shape)
        return TransitionMeanStd(
            mean=mean,
            std=torch.full(
                shape,
                0.2,
                dtype=transition.x_t.dtype,
                device=transition.x_t.device,
            ),
            dt=dt,
        )

    def _deterministic_ode_step(self, transition):
        self.ode_calls += 1
        self.ode_transition_shapes.append(tuple(transition.x_t.shape))
        self.ode_model_predictions.append(transition.model_prediction.detach().clone())
        self.events.append("dynamics.ode_step")
        shape = (
            transition.batch_size,
            *([1] * (transition.x_t.ndim - 1)),
        )
        dt = (transition.t_next - transition.t).to(transition.x_t.dtype)
        return DeterministicTransitionOutput(
            next_state=(
                transition.x_t + dt.reshape(shape) * transition.model_prediction
            ).detach(),
            dt=dt.detach(),
        )


@dataclass(frozen=True)
class _ConditionState:
    size: int


class _FakeConditioner(LatentConditioner):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    @classmethod
    def describe(cls, config):
        del config
        return DeclaredContract(
            component_kind="conditioner",
            component_id="fake-conditioner",
            conditioner=ConditionerContract(
                accepted_tasks=(TaskKind.T2I,),
                accepted_latent_layouts=(LatentLayout.BCHW,),
                payload_type="none",
                has_initialize_hook=True,
                has_after_step_hook=True,
                deterministic_given_state=True,
                replay_state_serializable=True,
                independent_of_policy_parameters=True,
            ),
        )

    @classmethod
    def from_config(cls, config, *, runtime_context):
        del config
        return cls(runtime_context["events"])

    def prepare(self, prompts, latent_spec, *, generator):
        del generator
        self.events.append("conditioner.prepare")
        return _ConditionState(len(prompts))

    def initialize_latents(self, base_latents, state, *, generator):
        del generator
        self.events.append("conditioner.initialize")
        assert state.size == base_latents.shape[0]
        return ConditionInitialization(base_latents.detach(), state)

    def after_step(self, step_index, timestep, next_latents, state):
        del step_index, timestep
        self.events.append("conditioner.after_step")
        assert state.size == next_latents.shape[0]
        return (next_latents + 0.05).detach()


class _CameraPayloadConditioner(_FakeConditioner):
    def __init__(self, events: list[str], *, offset: float) -> None:
        super().__init__(events)
        self.offset = offset

    def initialize_latents(self, base_latents, state, *, generator):
        del generator
        self.events.append("conditioner.initialize")
        trajectory = torch.eye(4, dtype=torch.float32).repeat(3, 1, 1)
        trajectory[:, 0, 3] += self.offset
        payloads = tuple(
            CameraConditionPayload(
                camera_trajectory=trajectory.clone(),
                conditioner_config_identity="camera-config-v1",
            )
            for _ in range(state.size)
        )
        return ConditionInitialization(base_latents.detach(), state, payloads)


def _source(source_id: str, index: int) -> SourceItemContext:
    return SourceItemContext(
        source_item_id=source_id,
        dataset_source_id="main",
        dataset_index=index,
        dataset_revision="dataset-v1",
    )


def _sample_batch(
    *,
    group_sizes: tuple[int, ...],
):
    items = []
    rows = []
    item_index = 0
    for group_index, group_size in enumerate(group_sizes):
        source = _source(f"source-{group_index}", group_index)
        for member_id in range(group_size):
            items.append(T2IItem(prompt=f"prompt-{group_index}", source=source))
            rows.append(
                BatchRowContext(
                    occurrence_id=f"occurrence-{group_index}-{member_id}",
                    group_id=f"group-{group_index}",
                    member_id=member_id,
                    phase="main",
                    optimizer_step=3,
                    source_item_id=source.source_item_id,
                )
            )
            item_index += 1
    return ExplicitCollator().collate_samples(tuple(items), tuple(rows))


def _latent_spec(batch_size: int) -> ModelLatentSpec:
    return ModelLatentSpec(
        shape=(batch_size, 1, 2, 2),
        layout=LatentLayout.BCHW,
        axis_semantics=("batch", "channel", "height", "width"),
        device="cpu",
        dtype=torch.float32,
        spatial_stride=(1, 1),
    )


def _tempflow_upstream_geometry_oracle(
    *,
    prompt_count: int,
    exploration_count: int,
    num_steps: int,
    latent_tail: tuple[int, ...],
    seed: int,
):
    """Analytical CPU oracle for the locked upstream per-step loop order."""

    generator = torch.Generator().manual_seed(seed)
    mainline = torch.randn(
        (prompt_count, *latent_tail),
        generator=generator,
    )
    timesteps = torch.linspace(1.0, 0.25, num_steps)
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([-0.125])))
    x_t = []
    sampled_actions = []
    terminal_latents = []
    for step_index in range(num_steps - 1):
        dt = next_timesteps[step_index] - timesteps[step_index]
        expanded_x_t = mainline.repeat_interleave(exploration_count, dim=0)

        # Upstream computes the B0 ODE result before drawing B0 x K noise.
        mainline = mainline + 0.25 * dt
        mean = expanded_x_t + 0.25 * dt
        sampled = mean + 0.2 * torch.randn(
            expanded_x_t.shape,
            generator=generator,
        )

        terminal = sampled
        for continuation_index in range(step_index + 1, num_steps):
            continuation_dt = (
                next_timesteps[continuation_index] - timesteps[continuation_index]
            )
            terminal = terminal + 0.25 * continuation_dt
        x_t.append(expanded_x_t)
        sampled_actions.append(sampled)
        terminal_latents.append(terminal)
    return (
        torch.stack(x_t, dim=1),
        torch.stack(sampled_actions, dim=1),
        torch.stack(terminal_latents, dim=1),
        generator.get_state(),
    )


def _video_batch(
    batch_size: int = 2,
    *,
    camera_offset: float | None = None,
):
    items = []
    rows = []
    for index in range(batch_size):
        source = _source(f"video-source-{index}", index)
        condition = None
        if camera_offset is not None:
            trajectory = torch.eye(4, dtype=torch.float32).repeat(3, 1, 1)
            trajectory[:, 0, 3] += camera_offset
            condition = CameraConditionPayload(
                camera_trajectory=trajectory,
                conditioner_config_identity="camera-config-v1",
            )
        items.append(
            T2VItem(
                prompt=f"video-{index}",
                source=source,
                **({"condition": condition} if condition is not None else {}),
            )
        )
        rows.append(
            BatchRowContext(
                occurrence_id=f"video-occurrence-{index}",
                group_id=f"video-group-{index}",
                member_id=0,
                phase="main",
                optimizer_step=3,
                source_item_id=source.source_item_id,
            )
        )
    return ExplicitCollator().collate_samples(tuple(items), tuple(rows))


def _video_latent_spec(batch_size: int) -> ModelLatentSpec:
    return ModelLatentSpec(
        shape=(batch_size, 1, 3, 2, 2),
        layout=LatentLayout.BCTHW,
        axis_semantics=("batch", "channel", "time", "height", "width"),
        device="cpu",
        dtype=torch.float32,
        spatial_stride=(1, 1),
        temporal_stride=1,
    )


def _conditioner_spec(batch_size: int) -> LatentSpec:
    return LatentSpec(
        batch_size=batch_size,
        channels=1,
        latent_frames=1,
        latent_height=2,
        latent_width=2,
        output_frames=1,
        output_height=2,
        output_width=2,
        temporal_compression=1,
        spatial_compression=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _video_conditioner_spec(batch_size: int) -> LatentSpec:
    return LatentSpec(
        batch_size=batch_size,
        channels=1,
        latent_frames=3,
        latent_height=2,
        latent_width=2,
        output_frames=3,
        output_height=2,
        output_width=2,
        temporal_compression=1,
        spatial_compression=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _runtime(
    batch,
    *,
    seed: int,
    semantics: LikelihoodSemantics = LikelihoodSemantics.EXACT_ENV_ACTION,
    with_conditioner: bool = False,
    latent_spec: ModelLatentSpec | None = None,
):
    events: list[str] = []
    adapter = _FakeAdapter(events)
    dynamics = _FakeDynamics(events)
    conditioner = _FakeConditioner(events) if with_conditioner else None

    original_sample = dynamics.sample_transition
    original_record = dynamics.make_record

    def logged_sample(transition, *, generator):
        events.append("dynamics.sample_transition")
        dynamics.sample_transition_shapes.append(tuple(transition.x_t.shape))
        dynamics.sample_model_predictions.append(
            transition.model_prediction.detach().clone()
        )
        return original_sample(transition, generator=generator)

    def logged_record(
        transition,
        output,
        *,
        conditioned_next,
        likelihood_semantics,
    ):
        events.append("dynamics.make_record")
        return original_record(
            transition,
            output,
            conditioned_next=conditioned_next,
            likelihood_semantics=likelihood_semantics,
        )

    dynamics.sample_transition = logged_sample
    dynamics.make_record = logged_record
    resolved_latent_spec = latent_spec or _latent_spec(batch.batch_size)
    request = RolloutRequest(
        adapter=adapter,
        dynamics=dynamics,
        samples=batch,
        latent_spec=resolved_latent_spec,
        generator=torch.Generator().manual_seed(seed),
        likelihood_semantics=semantics,
        conditioner=conditioner,
        conditioner_latent_spec=(
            _conditioner_spec(batch.batch_size) if conditioner else None
        ),
    )
    return request, dynamics, events


def _execution_policy(
    *,
    group_size: int,
    forward_microbatch_size: int | None = None,
    decode_microbatch_size: int | None = None,
    trajectory_storage_device: str = "cpu",
) -> ExecutionPolicyReceipt:
    """Build the canonical execution-policy receipt owned outside rollout."""

    return ExecutionPolicyReceipt.from_payload(
        {
            "schema_version": 1,
            "training_mode": "lora",
            "distribution_mode": "single",
            "precision": "fp32",
            "group_size": group_size,
            "rollout": {
                "forward_microbatch_size": forward_microbatch_size,
                "decode_microbatch_size": decode_microbatch_size,
                "trajectory_storage_device": trajectory_storage_device,
            },
            "transform_plan": {
                "schema_version": 1,
                "paradigm": "coupled",
                "transforms": (),
            },
        }
    )


def _full_rollout(
    config: FullTrajectoryRolloutConfig,
    *,
    execution_policy: ExecutionPolicyReceipt,
) -> FullTrajectoryRollout:
    return FullTrajectoryRollout(
        config,
        execution_policy=execution_policy,
        expected_policy_id=execution_policy.policy_id,
    )


def _branching_rollout(
    config: BranchingRolloutConfig,
    *,
    execution_policy: ExecutionPolicyReceipt,
) -> BranchingRollout:
    return BranchingRollout(
        config,
        execution_policy=execution_policy,
        expected_policy_id=execution_policy.policy_id,
    )


def _single_step_rollout(
    config: SingleStepRolloutConfig,
    *,
    execution_policy: ExecutionPolicyReceipt,
) -> SingleStepRollout:
    return SingleStepRollout(
        config,
        execution_policy=execution_policy,
        expected_policy_id=execution_policy.policy_id,
    )


def _canonical_rollout_selection(definition):
    catalog = build_catalog(algorithm_domain_catalog_fragments())
    algorithm = AlgorithmDeclarationResolver().resolve(
        catalog.for_kind("algorithm"),
        definition.algorithm.alias,
        to_plain_dict(definition.algorithm.params),
    )
    slot = algorithm.blueprint.slot("rollout")
    assert slot.component_id is not None
    return slot.component_id, to_plain_dict(slot.params)


def _rollout_registry():
    return build_catalog((ROLLOUT_CATALOG_FRAGMENT,)).for_kind("rollout")


def _load_rollout_declaration(declaration, *, policy: ExecutionPolicyReceipt):
    return load_test_component(
        declaration,
        slot="rollout",
        runtime_context={
            "execution_policy": policy,
            "expected_execution_policy_id": policy.policy_id,
        },
    ).instance


def test_full_rollout_uses_one_shared_policy_step_order_and_stores_all_steps():
    batch = _sample_batch(group_sizes=(2,))
    request, _dynamics, events = _runtime(
        batch,
        seed=13,
        with_conditioner=True,
    )
    policy = _execution_policy(group_size=2)
    result = _full_rollout(
        FullTrajectoryRolloutConfig(num_steps=3),
        execution_policy=policy,
    ).run(request)

    call_graph = [
        event
        for event in events
        if event
        in {
            "adapter.predict",
            "dynamics.sample_transition",
            "conditioner.after_step",
            "dynamics.make_record",
        }
    ]
    assert call_graph == [
        item
        for _ in range(3)
        for item in (
            "adapter.predict",
            "dynamics.sample_transition",
            "conditioner.after_step",
            "dynamics.make_record",
        )
    ]
    assert events.count("adapter.encode") == 1
    assert events.count("adapter.decode") == 1
    assert result.kind == "full_trajectory"
    assert result.transition_count == 3
    assert torch.equal(result.transition_index[0], torch.tensor([0, 1, 2]))
    assert not torch.equal(result.sampled_action, result.conditioned_next)


def test_full_rollout_compacts_records_without_row_index_select(monkeypatch) -> None:
    batch = _sample_batch(group_sizes=(2,))
    request, _dynamics, _events = _runtime(batch, seed=14)

    def forbidden_slice(self, indices):
        del self, indices
        raise AssertionError("full rollout must not materialize per-row records")

    monkeypatch.setattr(TransitionRecord, "slice", forbidden_slice)
    policy = _execution_policy(group_size=2)
    trajectory = _full_rollout(
        FullTrajectoryRolloutConfig(num_steps=3),
        execution_policy=policy,
    ).run(request)

    assert trajectory.transition_count == 3
    assert not hasattr(trajectory, "mean")
    assert not hasattr(trajectory, "std")
    assert not hasattr(trajectory, "dt")


def test_full_rollout_replays_the_exact_forward_microbatch_geometry() -> None:
    batch = _sample_batch(group_sizes=(4,))
    request, dynamics, events = _runtime(batch, seed=15)
    adapter = _RowSensitiveAdapter(events)
    request = replace(request, adapter=adapter)
    policy = _execution_policy(
        group_size=4,
        forward_microbatch_size=2,
        decode_microbatch_size=1,
        trajectory_storage_device="cpu",
    )
    execution = _full_rollout(
        FullTrajectoryRolloutConfig(num_steps=2),
        execution_policy=policy,
    ).run_with_snapshot(request)

    replay = execution.model_forward_replay
    assert replay is not None
    assert replay.forward_partitions == ((0, 1), (2, 3))
    assert replay.partition_identity.startswith("model-forward-partitions.v1:")
    assert adapter.predict_shapes == [(2, 1, 2, 2)] * 4
    assert adapter.decode_shapes == [(1, 1, 2, 2)] * 4
    assert execution.trajectory.media.device.type == "cpu"

    adapter.predict_shapes.clear()
    stats = compute_full_policy_stats_oracle(
        PolicyRecomputeRequest(
            adapter=adapter,
            dynamics=dynamics,
            rollout=execution,
            latent_spec=request.latent_spec,
        )
    )

    assert adapter.predict_shapes == [(2, 1, 2, 2)] * 4
    torch.testing.assert_close(
        stats.current_log_probs,
        execution.trajectory.old_log_probs,
        rtol=0,
        atol=0,
    )


def test_rollout_rng_is_explicit_reproducible_and_seed_sensitive():
    batch = _sample_batch(group_sizes=(2,))

    def run(seed: int):
        request, _dynamics, _events = _runtime(batch, seed=seed)
        policy = _execution_policy(group_size=2)
        return _full_rollout(
            FullTrajectoryRolloutConfig(num_steps=2),
            execution_policy=policy,
        ).run(request)

    first = run(19)
    second = run(19)
    different = run(20)
    torch.testing.assert_close(first.sampled_action, second.sampled_action)
    torch.testing.assert_close(first.media, second.media)
    assert not torch.equal(first.sampled_action, different.sampled_action)


def test_full_rollout_preserves_bcthw_video_latents_and_decoded_media_layout():
    samples = _video_batch()
    request, _dynamics, _events = _runtime(
        samples,
        seed=23,
        latent_spec=_video_latent_spec(samples.batch_size),
    )
    policy = _execution_policy(group_size=1)
    result = _full_rollout(
        FullTrajectoryRolloutConfig(num_steps=2),
        execution_policy=policy,
    ).run(request)

    assert result.x_t.shape == (2, 2, 1, 3, 2, 2)
    assert result.media.shape == (2, 3, 1, 2, 2)
    assert result.media_layout == "BFCHW"


def test_rollout_config_rejects_model_owned_decoded_media_layout() -> None:
    with pytest.raises(ValueError, match="unknown full-trajectory rollout params"):
        FullTrajectoryRolloutConfig.from_mapping(
            {"num_steps": 1, "decoded_media_layout": "bfhwc"},
            context=None,
        )


def test_decode_microbatch_fails_closed_on_chunk_layout_drift() -> None:
    samples = _video_batch(batch_size=2)
    request, _dynamics, events = _runtime(
        samples,
        seed=231,
        latent_spec=_video_latent_spec(samples.batch_size),
    )
    request = replace(request, adapter=_DecodeLayoutDriftAdapter(events))

    policy = _execution_policy(group_size=1, decode_microbatch_size=1)
    with pytest.raises(RolloutContractError, match="layout changed across rows"):
        _full_rollout(
            FullTrajectoryRolloutConfig(num_steps=1),
            execution_policy=policy,
        ).run(request)


def test_conditioner_state_is_the_single_camera_payload_source():
    samples = _video_batch(batch_size=1)
    request, _dynamics, events = _runtime(
        samples,
        seed=24,
        latent_spec=_video_latent_spec(samples.batch_size),
    )
    request = replace(
        request,
        conditioner=_CameraPayloadConditioner(events, offset=0.25),
        conditioner_latent_spec=_video_conditioner_spec(samples.batch_size),
    )

    policy = _execution_policy(group_size=1)
    result = _full_rollout(
        FullTrajectoryRolloutConfig(num_steps=1),
        execution_policy=policy,
    ).run(request)

    assert isinstance(result.condition_state, CameraConditionBatchState)
    row_identity = result.condition_state.row_condition_identities[0]
    assert result.condition_identity == ((row_identity,),)
    assert result.condition_state.camera_trajectory[0, 0, 0, 3] == 0.25


def test_input_camera_must_match_conditioner_generated_state_exactly():
    samples = _video_batch(batch_size=1, camera_offset=0.5)
    request, _dynamics, events = _runtime(
        samples,
        seed=25,
        latent_spec=_video_latent_spec(samples.batch_size),
    )
    request = replace(
        request,
        conditioner=_CameraPayloadConditioner(events, offset=0.75),
        conditioner_latent_spec=_video_conditioner_spec(samples.batch_size),
    )

    policy = _execution_policy(group_size=1)
    with pytest.raises(ValueError, match="input camera.*disagree"):
        _full_rollout(
            FullTrajectoryRolloutConfig(num_steps=1),
            execution_policy=policy,
        ).run(request)
    assert "dynamics.sample_transition" not in events


def test_matching_input_camera_accepts_the_state_derived_payload():
    samples = _video_batch(batch_size=1, camera_offset=0.5)
    request, _dynamics, events = _runtime(
        samples,
        seed=26,
        latent_spec=_video_latent_spec(samples.batch_size),
    )
    request = replace(
        request,
        conditioner=_CameraPayloadConditioner(events, offset=0.5),
        conditioner_latent_spec=_video_conditioner_spec(samples.batch_size),
    )

    policy = _execution_policy(group_size=1)
    result = _full_rollout(
        FullTrajectoryRolloutConfig(num_steps=1),
        execution_policy=policy,
    ).run(request)

    assert isinstance(result.condition_state, CameraConditionBatchState)
    assert result.condition_identity == (
        (result.condition_state.row_condition_identities[0],),
    )


def test_model_prompt_identity_is_separate_from_external_camera_identity():
    samples = _video_batch(batch_size=1, camera_offset=0.5)
    request, _dynamics, events = _runtime(
        samples,
        seed=27,
        latent_spec=_video_latent_spec(samples.batch_size),
    )
    adapter = _IdentityAwareAdapter(events)
    request = replace(
        request,
        adapter=adapter,
        conditioner=_CameraPayloadConditioner(events, offset=0.5),
        conditioner_latent_spec=_video_conditioner_spec(samples.batch_size),
    )

    policy = _execution_policy(group_size=1)
    result = _full_rollout(
        FullTrajectoryRolloutConfig(num_steps=1),
        execution_policy=policy,
    ).run(request)

    assert adapter.model_input_identities == [
        tuple(row.identity for row in samples.rows)
    ]
    assert result.condition_identity == (
        (result.condition_state.row_condition_identities[0],),
    )
    assert adapter.model_input_identities[0][0] != result.condition_identity[0][0]


def test_preencoded_conditioning_skips_encoder_inside_rollout() -> None:
    samples = _sample_batch(group_sizes=(2,))
    request, _dynamics, events = _runtime(samples, seed=28)
    conditioning = _EncodedConditioning(tuple(row.identity for row in samples.rows))
    request = replace(
        request,
        encoded_conditioning=conditioning,
        model_condition_identity=conditioning.condition_identity,
    )

    policy = _execution_policy(group_size=2)
    execution = _full_rollout(
        FullTrajectoryRolloutConfig(num_steps=1),
        execution_policy=policy,
    ).run_with_snapshot(request)

    assert "adapter.encode" not in events
    assert execution.encoded_conditioning is conditioning
    assert execution.model_condition_identity == conditioning.condition_identity


def test_rollout_stage_retains_exact_phase_route_for_reward_context_factory() -> None:
    source = SourceSequence(
        source_id="main",
        revision="dataset-v1",
        items=tuple(
            T2IItem(
                prompt=f"routed-prompt-{index}",
                source=_source(f"routed-source-{index}", index),
            )
            for index in range(2)
        ),
    )
    prelude = DataPlanePrelude(
        phase_schedule=PeriodicPhaseSchedule(
            phases=(
                PhaseDefinition(
                    phase_id="main-phase",
                    start_offset=0,
                    end_offset=1,
                    source_id="main",
                    active_rewards=("quality",),
                ),
            ),
            known_source_ids=frozenset({"main"}),
            known_reward_ids=frozenset({"quality"}),
        ),
        source_sampler=MultiSourceSampler((source,)),
        placement_contract=GroupPlacementContract(
            placement=GroupPlacementKind.LOCAL_COMPLETE,
            global_prompt_batch_size=2,
            group_size=2,
            world_size=1,
            per_rank_microbatch_rows=4,
            gradient_accumulation_steps=1,
        ),
        collator=ExplicitCollator(),
    )
    prelude_value = prelude.build(3)

    def request_factory(samples, identity):
        assert samples is prelude_value.payload.samples
        request, _dynamics, _events = _runtime(
            samples, seed=40 + identity.optimizer_step
        )
        return request

    policy = _execution_policy(group_size=2)
    rollout_value = RolloutStage(
        rollout=_full_rollout(
            FullTrajectoryRolloutConfig(num_steps=1),
            execution_policy=policy,
        ),
        request_factory=request_factory,
    )(StageValue(prelude_value.identity, prelude_value.payload))

    rollout_payload = rollout_value.payload
    assert rollout_payload.data_plane is prelude_value.payload
    assert rollout_payload.phase_binding is prelude_value.payload.phase_binding
    assert rollout_payload.phase_route is prelude_value.payload.route
    assert rollout_payload.active_rewards == ("quality",)
    observed_routes = []

    def reward_context_factory(payload, identity):
        observed_routes.append(payload.phase_route)
        return RewardRuntimeContext(
            StepContext(
                step=identity.optimizer_step,
                seed=40 + identity.optimizer_step,
                rank=0,
                world_size=1,
            )
        )

    context = reward_context_factory(rollout_payload, rollout_value.identity)
    assert context.step_context.step == 3
    assert observed_routes == [prelude_value.payload.route]
    assert observed_routes[0] is prelude_value.payload.route
    prelude.abort_iteration(prelude_value.identity)


def test_exact_and_surrogate_keep_both_states_and_choose_declared_target():
    batch = _sample_batch(group_sizes=(1,))
    exact_request, _dynamics, _events = _runtime(
        batch,
        seed=5,
        semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
        with_conditioner=True,
    )
    surrogate_request, _dynamics, _events = _runtime(
        batch,
        seed=5,
        semantics=LikelihoodSemantics.POST_HOOK_BASE_DENSITY_SURROGATE,
        with_conditioner=True,
    )
    policy = _execution_policy(group_size=1)
    strategy = _full_rollout(
        FullTrajectoryRolloutConfig(num_steps=1),
        execution_policy=policy,
    )
    exact = strategy.run(exact_request)
    surrogate = strategy.run(surrogate_request)

    assert exact.scoring_target is exact.sampled_action
    assert surrogate.scoring_target is surrogate.conditioned_next
    torch.testing.assert_close(exact.sampled_action, surrogate.sampled_action)
    torch.testing.assert_close(
        exact.conditioned_next,
        surrogate.conditioned_next,
    )
    assert not torch.equal(exact.old_log_probs, surrogate.old_log_probs)


def test_branching_has_shared_prefix_identity_and_group_member_rows():
    batch = _sample_batch(group_sizes=(2, 2))
    request, dynamics, events = _runtime(batch, seed=29)
    expected_generator = torch.Generator().manual_seed(29)
    torch.randn(
        (2, *request.latent_spec.shape[1:]),
        generator=expected_generator,
    )
    # Prefix and continuation are deterministic ODE paths. The only second
    # random draw is the selected branch action itself.
    torch.randn(request.latent_spec.shape, generator=expected_generator)
    policy = _execution_policy(group_size=2)
    result = _branching_rollout(
        BranchingRolloutConfig(
            num_steps=4,
            branch_count=2,
            branch_topology=BranchTopology.single_point_branch_ablation(2),
            branch_step_policy="uniform_intermediate",
            branch_step_index=2,
        ),
        execution_policy=policy,
    ).run(request)

    assert result.kind == "branching"
    assert result.transition_count == 1
    assert torch.equal(result.branch_step_index, torch.tensor([2, 2, 2, 2]))
    assert torch.equal(result.transition_index[:, 0], result.branch_step_index)
    assert result.shared_prefix_id[0] == result.shared_prefix_id[1]
    assert result.shared_prefix_id[2] == result.shared_prefix_id[3]
    assert result.shared_prefix_id[0] != result.shared_prefix_id[2]
    assert result.branch_step_identity[0] == result.branch_step_identity[1]
    assert torch.equal(result.x_t[0, 0], result.x_t[1, 0])
    assert torch.equal(result.x_t[2, 0], result.x_t[3, 0])
    assert not torch.equal(result.sampled_action[0, 0], result.sampled_action[1, 0])
    assert dynamics.mean_std_calls == 1
    assert dynamics.ode_calls == 3
    assert events.count("dynamics.sample_transition") == 1
    assert events.count("dynamics.make_record") == 1
    assert torch.equal(request.generator.get_state(), expected_generator.get_state())
    assert {context.batch_row.member_id for context in result.contexts[:2]} == {0, 1}


def test_tempflow_paper_branches_every_mainline_timestep_with_terminal_media():
    batch = _sample_batch(group_sizes=(2, 2))
    request, dynamics, events = _runtime(batch, seed=41)
    (
        expected_x_t,
        expected_sampled_actions,
        expected_terminal_media,
        expected_generator_state,
    ) = _tempflow_upstream_geometry_oracle(
        prompt_count=2,
        exploration_count=2,
        num_steps=4,
        latent_tail=request.latent_spec.shape[1:],
        seed=41,
    )
    topology = BranchTopology.every_policy_timestep(2)

    policy = _execution_policy(group_size=2)
    strategy = _branching_rollout(
        BranchingRolloutConfig(
            num_steps=4,
            branch_count=2,
            branch_topology=topology,
        ),
        execution_policy=policy,
    )
    execution = strategy.run_with_snapshot(request)
    result = execution.trajectory

    assert result.kind == "branching"
    assert result.branch_topology == topology
    assert strategy.selection_contract_identity == (
        strategy.config.selection_contract_identity
    )
    assert strategy.selection_contract_identity != topology.topology_identity
    assert (
        execution.schedule_snapshot.selection_policy
        == strategy.selection_contract_identity
    )
    assert execution.schedule_snapshot.selected_policy_step_indices == (0, 1, 2)
    assert result.transition_count == 3
    assert tuple(result.old_log_probs.shape) == (4, 3)
    assert torch.equal(result.transition_index, result.branch_timestep_index)
    assert result.transition_terminal_media_layout == "BTCHW"
    assert tuple(result.transition_terminal_media.shape[:2]) == (4, 3)
    assert torch.equal(result.exploration_member_index, torch.tensor([0, 1, 0, 1]))
    assert result.branch_step_index is None
    assert result.shared_prefix_id is None
    assert result.branch_step_identity is None
    assert torch.equal(result.x_t[0], result.x_t[1])
    assert not torch.equal(result.sampled_action[0], result.sampled_action[1])
    assert not torch.equal(result.conditioned_next[:, 0], result.x_t[:, 1])
    torch.testing.assert_close(result.x_t, expected_x_t)
    torch.testing.assert_close(result.sampled_action, expected_sampled_actions)
    torch.testing.assert_close(
        result.transition_terminal_media,
        expected_terminal_media,
    )
    torch.testing.assert_close(result.media, result.transition_terminal_media[:, -1])
    assert events.count("dynamics.sample_transition") == 3
    assert events.count("dynamics.make_record") == 3
    # Three stochastic policy transitions resolve SDE mean/std. The nine ODE
    # updates use the explicit deterministic port; each mainline update reuses
    # the prediction already evaluated for that timestep's SDE action.
    assert dynamics.mean_std_calls == 3
    assert dynamics.ode_calls == 9
    assert events.count("adapter.predict") == 9
    b0_shape = (2, *request.latent_spec.shape[1:])
    exploration_shape = request.latent_spec.shape
    expected_forward_shapes = [
        b0_shape,
        exploration_shape,
        exploration_shape,
        exploration_shape,
        b0_shape,
        exploration_shape,
        exploration_shape,
        b0_shape,
        exploration_shape,
    ]
    assert request.adapter.predict_shapes == expected_forward_shapes
    assert dynamics.ode_transition_shapes == expected_forward_shapes
    assert dynamics.sample_transition_shapes == [exploration_shape] * 3
    observed_order = [
        event
        for event in events
        if event
        in {
            "adapter.predict",
            "dynamics.ode_step",
            "dynamics.sample_transition",
            "dynamics.mean_std",
            "dynamics.make_record",
        }
    ]
    assert observed_order == [
        "adapter.predict",
        "dynamics.ode_step",
        "dynamics.sample_transition",
        "dynamics.mean_std",
        "dynamics.make_record",
        "adapter.predict",
        "dynamics.ode_step",
        "adapter.predict",
        "dynamics.ode_step",
        "adapter.predict",
        "dynamics.ode_step",
        "adapter.predict",
        "dynamics.ode_step",
        "dynamics.sample_transition",
        "dynamics.mean_std",
        "dynamics.make_record",
        "adapter.predict",
        "dynamics.ode_step",
        "adapter.predict",
        "dynamics.ode_step",
        "adapter.predict",
        "dynamics.ode_step",
        "dynamics.sample_transition",
        "dynamics.mean_std",
        "dynamics.make_record",
        "adapter.predict",
        "dynamics.ode_step",
    ]
    assert torch.equal(request.generator.get_state(), expected_generator_state)
    strategy_identity = hashlib.sha256(
        (
            f"{topology.topology_identity}\0"
            f"{execution.schedule_snapshot.snapshot_identity}"
        ).encode()
    ).hexdigest()
    row = batch.rows[0]
    context_digest = hashlib.sha256(
        f"branching\0{strategy_identity}\0{row.identity}".encode()
    ).hexdigest()
    assert result.contexts[0].trajectory_id == (f"trajectory-{context_digest[24:48]}")


def test_builtin_tempflow_single_row_forwards_preserve_branch_numerics_and_gradients():
    definition = get_recipe_definition("tempflow_grpo_v1")
    rollout_id, rollout_params = _canonical_rollout_selection(definition)
    declaration = DeclarationResolver().resolve(
        _rollout_registry(),
        rollout_id,
        rollout_params,
    )
    policy = definition.execution_policy.to_receipt()
    strategy = _load_rollout_declaration(declaration, policy=policy)
    assert isinstance(strategy, BranchingRollout)
    assert strategy.rollout_execution_policy.forward_microbatch_size == 1

    branch_count = strategy.config.branch_count
    batch = _sample_batch(group_sizes=(branch_count,))
    request, dynamics, events = _runtime(batch, seed=419)
    adapter = _OrderedRowSensitiveAdapter(events)
    request = replace(request, adapter=adapter)

    execution = strategy.run_with_snapshot(request)
    trajectory = execution.trajectory
    replay = execution.model_forward_replay
    assert replay is not None
    assert replay.forward_row_indices == (0,)
    assert replay.row_to_forward_position == (0,) * branch_count
    assert replay.forward_partitions == ((0,),)

    policy_steps = strategy.config.num_steps - 1
    continuation_steps = strategy.config.num_steps * policy_steps // 2
    assert len(adapter.predict_shapes) == (
        policy_steps + branch_count * continuation_steps
    )
    assert {shape[0] for shape in adapter.predict_shapes} == {1}
    assert dynamics.sample_transition_shapes == [request.latent_spec.shape] * (
        policy_steps
    )

    conditioning = execution.encoded_conditioning
    assert isinstance(conditioning, _RowSensitiveConditioning)
    expected_condition_values = conditioning.values.reshape(
        branch_count,
        *([1] * (trajectory.x_t.ndim - 2)),
    )
    for step, actual in enumerate(dynamics.sample_model_predictions):
        expected = (
            trajectory.x_t[:, step] * adapter.policy_scale.detach()
            + expected_condition_values
            + trajectory.x_t.new_tensor(0.013)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual,
            actual[0:1].expand_as(actual),
            rtol=0,
            atol=0,
        )

    first_branch_inputs = torch.cat(
        adapter.predict_latents[1 : 1 + branch_count],
        dim=0,
    )
    torch.testing.assert_close(
        first_branch_inputs,
        trajectory.conditioned_next[:, 0],
        rtol=0,
        atol=0,
    )
    expected_first_continuation = (
        first_branch_inputs * adapter.policy_scale.detach()
        + expected_condition_values
        + first_branch_inputs.new_tensor(0.013)
    )
    torch.testing.assert_close(
        dynamics.ode_model_predictions[1],
        expected_first_continuation,
        rtol=0,
        atol=0,
    )
    monolithic_first_continuation = (
        first_branch_inputs * adapter.policy_scale.detach()
        + expected_condition_values
        + first_branch_inputs.new_tensor(0.013 * branch_count)
    )
    assert not torch.equal(
        dynamics.ode_model_predictions[1],
        monolithic_first_continuation,
    )

    recompute_request = PolicyRecomputeRequest(
        adapter=adapter,
        dynamics=dynamics,
        rollout=execution,
        latent_spec=request.latent_spec,
    )
    recomputer = PolicyRecomputer()
    adapter.predict_shapes.clear()
    adapter.predict_latents.clear()
    adapter.predict_condition_identities.clear()
    stats = compute_full_policy_stats_oracle(
        recompute_request,
        recomputer=recomputer,
    )
    assert (
        adapter.predict_shapes == [(1, *request.latent_spec.shape[1:])] * policy_steps
    )
    torch.testing.assert_close(
        stats.current_log_probs,
        trajectory.old_log_probs,
        rtol=0,
        atol=0,
    )
    stats.current_log_probs.sum().backward()
    assert adapter.policy_scale.grad is not None
    assert torch.isfinite(adapter.policy_scale.grad)
    assert adapter.policy_scale.grad.abs().item() > 0.0
    monolithic_gradient = adapter.policy_scale.grad.detach().clone()

    adapter.policy_scale.grad = None
    slot_plan = UpdateSlotPlan.from_active_mask(
        trajectory.transition_mask,
        row_microbatch_size=1,
        transition_window_size=1,
    )
    assert slot_plan.active_cells == tuple(
        (row, step) for row in range(branch_count) for step in range(policy_steps)
    )
    for slot in slot_plan.slots:
        slot_stats = recomputer.compute_current_slot(recompute_request, slot)
        slot_stats.current_log_probs.sum().backward()
    assert adapter.policy_scale.grad is not None
    torch.testing.assert_close(
        adapter.policy_scale.grad,
        monolithic_gradient,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_tempflow_slot_recompute_reuses_exact_expanded_leader_conditioning() -> None:
    batch = _sample_batch(group_sizes=(2, 2))
    request, dynamics, events = _runtime(batch, seed=43)
    adapter = _RowSensitiveAdapter(events)
    request = replace(request, adapter=adapter)
    policy = _execution_policy(group_size=2)
    execution = _branching_rollout(
        BranchingRolloutConfig(
            num_steps=3,
            branch_count=2,
            branch_topology=BranchTopology.every_policy_timestep(2),
        ),
        execution_policy=policy,
    ).run_with_snapshot(request)
    trajectory = execution.trajectory
    replay = execution.model_forward_replay
    assert replay is not None
    assert replay.forward_row_indices == (0, 2)
    assert replay.row_to_forward_position == (0, 0, 1, 1)

    conditioning = execution.encoded_conditioning
    assert isinstance(conditioning, _RowSensitiveConditioning)
    torch.testing.assert_close(conditioning.values[0], conditioning.values[1])
    torch.testing.assert_close(conditioning.values[2], conditioning.values[3])
    assert (
        execution.model_condition_identity[0] == (execution.model_condition_identity[1])
    )
    assert (
        execution.model_condition_identity[2] == (execution.model_condition_identity[3])
    )

    recompute_request = PolicyRecomputeRequest(
        adapter=adapter,
        dynamics=dynamics,
        rollout=execution,
        latent_spec=request.latent_spec,
        require_reference_statistics=True,
    )
    recomputer = PolicyRecomputer()
    adapter.predict_shapes.clear()
    adapter.reference_predict_shapes.clear()
    monolithic = compute_full_policy_stats_oracle(
        recompute_request,
        recomputer=recomputer,
    )
    torch.testing.assert_close(
        monolithic.current_log_probs,
        trajectory.old_log_probs,
        rtol=0,
        atol=0,
    )
    assert {shape[0] for shape in adapter.predict_shapes} == {2}
    assert {shape[0] for shape in adapter.reference_predict_shapes} == {2}

    slot_plan = UpdateSlotPlan.from_active_mask(
        trajectory.transition_mask,
        row_microbatch_size=1,
        transition_window_size=1,
    )
    observed = torch.zeros_like(trajectory.old_log_probs)
    adapter.predict_shapes.clear()
    adapter.reference_predict_shapes.clear()
    reference_by_slot = {
        slot.slot_id: recomputer.compute_reference_slot(recompute_request, slot)
        for slot in slot_plan.slots
    }
    for slot in slot_plan.slots:
        stats = recomputer.compute_current_slot(
            recompute_request,
            slot,
            reference_stats=reference_by_slot.pop(slot.slot_id),
        )
        assert stats.current_log_probs.requires_grad
        observed[
            slot.row_indices[0],
            slot.transition_start,
        ] = stats.current_log_probs.detach()[0, 0]
    assert not reference_by_slot

    assert {shape[0] for shape in adapter.predict_shapes} == {2}
    assert {shape[0] for shape in adapter.reference_predict_shapes} == {2}
    torch.testing.assert_close(
        observed,
        monolithic.current_log_probs.detach(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        observed,
        trajectory.old_log_probs,
        rtol=0,
        atol=0,
    )


def test_tempflow_projects_row_batched_guidance_for_mainline_and_recompute() -> None:
    batch = _sample_batch(group_sizes=(2, 2))
    request, dynamics, events = _runtime(batch, seed=443)
    adapter = _GuidanceAwareAdapter(events)
    guidance_identity = ("guidance-a", "guidance-a", "guidance-b", "guidance-b")
    guidance = _RowSensitiveGuidance(
        values=torch.tensor((0.11, 0.11, 0.73, 0.73)),
        guidance_identity=guidance_identity,
    )
    request = replace(
        request,
        adapter=adapter,
        guidance=guidance,
        guidance_identity=guidance_identity,
    )
    policy = _execution_policy(group_size=2, forward_microbatch_size=1)
    execution = _branching_rollout(
        BranchingRolloutConfig(
            num_steps=3,
            branch_count=2,
            branch_topology=BranchTopology.every_policy_timestep(2),
        ),
        execution_policy=policy,
    ).run_with_snapshot(request)
    replay = execution.model_forward_replay
    assert replay is not None
    assert replay.forward_row_indices == (0, 2)
    assert replay.forward_partitions == ((0,), (2,))
    assert replay.row_to_forward_position == (0, 0, 1, 1)
    assert execution.trajectory.guidance_identity == tuple(
        (identity,) * execution.trajectory.transition_count
        for identity in guidance_identity
    )

    # Both B0 mainline chunks and B0*K continuation chunks use the same
    # explicit selection port.  No model forward sees the original four-row
    # guidance payload after its latent batch has been microbatched to one.
    assert adapter.predict_guidance_identities
    assert {len(item) for item in adapter.predict_guidance_identities} == {1}
    expected_value = {"guidance-a": 0.11, "guidance-b": 0.73}
    for identities, values in zip(
        adapter.predict_guidance_identities,
        adapter.predict_guidance_values,
        strict=True,
    ):
        torch.testing.assert_close(
            values,
            torch.tensor((expected_value[identities[0]],)),
        )

    adapter.predict_guidance_identities.clear()
    adapter.predict_guidance_values.clear()
    stats = compute_full_policy_stats_oracle(
        PolicyRecomputeRequest(
            adapter=adapter,
            dynamics=dynamics,
            rollout=execution,
            latent_spec=request.latent_spec,
            guidance=guidance,
        )
    )
    torch.testing.assert_close(
        stats.current_log_probs,
        execution.trajectory.old_log_probs,
        rtol=0,
        atol=0,
    )
    assert adapter.predict_guidance_identities == [
        (identity,)
        for _step in range(execution.trajectory.transition_count)
        for identity in ("guidance-a", "guidance-b")
    ]


def test_row_batched_guidance_identity_is_validated_before_rollout() -> None:
    batch = _sample_batch(group_sizes=(2,))
    request, _dynamics, _events = _runtime(batch, seed=444)
    with pytest.raises(
        RolloutContractError,
        match="guidance payload identity does not match request",
    ):
        replace(
            request,
            guidance=_RowSensitiveGuidance(
                values=torch.tensor((0.2, 0.2)),
                guidance_identity=("payload-a", "payload-a"),
            ),
            guidance_identity=("request-b", "request-b"),
        )


def test_tempflow_draws_b0_initial_latents_for_real_data_plane_k_repeat() -> None:
    prompt_count = 2
    exploration_count = 3
    source = SourceSequence(
        source_id="main",
        revision="dataset-v1",
        items=tuple(
            T2IItem(
                prompt=f"data-plane-prompt-{index}",
                source=_source(f"data-plane-source-{index}", index),
                metadata={"prompt_index": index},
            )
            for index in range(prompt_count)
        ),
    )
    prelude = DataPlanePrelude(
        phase_schedule=PeriodicPhaseSchedule(
            phases=(
                PhaseDefinition(
                    phase_id="main-phase",
                    start_offset=0,
                    end_offset=1,
                    source_id="main",
                    active_rewards=("quality",),
                ),
            ),
            known_source_ids=frozenset({"main"}),
            known_reward_ids=frozenset({"quality"}),
        ),
        source_sampler=MultiSourceSampler((source,)),
        placement_contract=GroupPlacementContract(
            placement=GroupPlacementKind.LOCAL_COMPLETE,
            global_prompt_batch_size=prompt_count,
            group_size=exploration_count,
            world_size=1,
            per_rank_microbatch_rows=prompt_count * exploration_count,
            gradient_accumulation_steps=1,
        ),
        collator=ExplicitCollator(),
    )
    prelude_value = prelude.build(0)
    batch = prelude_value.payload.samples
    assert batch.batch_size == prompt_count * exploration_count
    assert len({row.occurrence_id for row in batch.rows}) == batch.batch_size

    seed = 401
    request, dynamics, _events = _runtime(batch, seed=seed)
    expected_generator = torch.Generator().manual_seed(seed)
    expected_base = torch.randn(
        (prompt_count, *request.latent_spec.shape[1:]),
        generator=expected_generator,
    )
    for _ in range(2):
        torch.randn(request.latent_spec.shape, generator=expected_generator)

    policy = _execution_policy(group_size=exploration_count)
    result = _branching_rollout(
        BranchingRolloutConfig(
            num_steps=3,
            branch_count=exploration_count,
            branch_topology=BranchTopology.every_policy_timestep(exploration_count),
        ),
        execution_policy=policy,
    ).run(request)

    assert request.adapter.prepared_latent_shapes == [
        (prompt_count, *request.latent_spec.shape[1:])
    ]
    assert dynamics.sample_transition_shapes == [request.latent_spec.shape] * 2
    b0_shape = (prompt_count, *request.latent_spec.shape[1:])
    assert request.adapter.predict_shapes == [
        b0_shape,
        request.latent_spec.shape,
        request.latent_spec.shape,
        b0_shape,
        request.latent_spec.shape,
    ]
    assert dynamics.ode_transition_shapes == request.adapter.predict_shapes
    assert torch.equal(request.generator.get_state(), expected_generator.get_state())
    group_order = tuple(dict.fromkeys(row.group_id for row in batch.rows))
    for base_index, group_id in enumerate(group_order):
        row_indices = [
            index for index, row in enumerate(batch.rows) if row.group_id == group_id
        ]
        expected_rows = expected_base[base_index].expand(
            len(row_indices),
            *expected_base.shape[1:],
        )
        torch.testing.assert_close(result.x_t[row_indices, 0], expected_rows)
    assert tuple(context.batch_row_identity for context in result.contexts) == (
        prelude_value.identity.row_identities
    )
    assert tuple(context.batch_row.member_id for context in result.contexts) == (
        prelude_value.identity.member_ids
    )
    prelude.abort_iteration(prelude_value.identity)


def test_branching_rejects_incomplete_group_before_policy_sampling():
    batch = _sample_batch(group_sizes=(1,))
    request, _dynamics, events = _runtime(batch, seed=3)
    policy = _execution_policy(group_size=2)
    strategy = _branching_rollout(
        BranchingRolloutConfig(
            num_steps=3,
            branch_count=2,
            branch_topology=BranchTopology.single_point_branch_ablation(2),
            branch_step_policy="uniform_intermediate",
            branch_step_index=1,
        ),
        execution_policy=policy,
    )
    with pytest.raises(ValueError, match="branch_count rows"):
        strategy.run(request)
    assert "dynamics.sample_transition" not in events


def test_branching_rejects_duplicate_occurrence_and_metadata_drift() -> None:
    original = _sample_batch(group_sizes=(2,))
    duplicate_rows = (
        original.rows[0],
        replace(
            original.rows[1],
            occurrence_id=original.rows[0].occurrence_id,
        ),
    )
    duplicate_batch = replace(original, rows=duplicate_rows)
    duplicate_request, _dynamics, duplicate_events = _runtime(
        duplicate_batch,
        seed=304,
    )
    config = BranchingRolloutConfig(
        num_steps=3,
        branch_count=2,
        branch_topology=BranchTopology.every_policy_timestep(2),
    )
    policy = _execution_policy(group_size=2)
    with pytest.raises(ValueError, match="occurrence_id values must be unique"):
        _branching_rollout(config, execution_policy=policy).run(duplicate_request)
    assert "dynamics.sample_transition" not in duplicate_events

    metadata_batch = replace(
        original,
        metadata=(original.metadata[0], {"semantic_drift": True}),
    )
    metadata_request, _dynamics, metadata_events = _runtime(
        metadata_batch,
        seed=305,
    )
    with pytest.raises(ValueError, match="share identical metadata"):
        _branching_rollout(config, execution_policy=policy).run(metadata_request)
    assert "dynamics.sample_transition" not in metadata_events


def test_single_step_stores_one_action_and_uses_ode_for_both_continuations():
    batch = _sample_batch(group_sizes=(3,))
    request, dynamics, events = _runtime(
        batch,
        seed=31,
        with_conditioner=True,
    )
    policy = _execution_policy(group_size=3)
    result = _single_step_rollout(
        SingleStepRolloutConfig(
            selected_timestep_policy="uniform",
            num_steps=4,
            selected_timestep_index=1,
        ),
        execution_policy=policy,
    ).run(request)

    assert result.kind == "single_step"
    assert result.transition_count == 1
    assert torch.equal(result.selected_timestep_index, torch.tensor([1, 1, 1]))
    assert torch.equal(result.transition_index[:, 0], result.selected_timestep_index)
    assert events.count("dynamics.sample_transition") == 1
    assert events.count("dynamics.make_record") == 1
    assert dynamics.mean_std_calls == 1
    assert dynamics.ode_calls == 3
    assert events.count("conditioner.after_step") == 4
    assert not torch.equal(result.media, result.conditioned_next[:, 0])


def test_single_step_inactive_rows_use_ode_without_a_second_prediction() -> None:
    batch = _sample_batch(group_sizes=(3,))
    request, dynamics, events = _runtime(batch, seed=311)
    config = SingleStepRolloutConfig(
        selected_timestep_policy="uniform",
        num_steps=4,
    )
    selection = PolicyStepSelection.fixed(
        (0, 1, 2),
        num_steps=4,
        generator=torch.Generator().manual_seed(992),
        policy=config.selection_contract_identity,
    )
    session = DynamicsSession.create(
        dynamics,
        num_steps=4,
        device="cpu",
        selection=selection,
    )

    policy = _execution_policy(group_size=3)
    result = _single_step_rollout(config, execution_policy=policy).run(
        replace(request, dynamics_session=session)
    )

    assert torch.equal(result.selected_timestep_index, torch.tensor([0, 1, 2]))
    assert dynamics.mean_std_calls == 3
    assert dynamics.ode_calls == 4
    assert events.count("adapter.predict") == 4
    assert events.count("dynamics.sample_transition") == 3
    assert events.count("dynamics.make_record") == 3


def test_rollout_uses_an_explicit_cursor_free_dynamics_session_selection():
    batch = _sample_batch(group_sizes=(3,))
    request, dynamics, _events = _runtime(batch, seed=32)
    selection_generator = torch.Generator().manual_seed(991)
    config = SingleStepRolloutConfig(
        selected_timestep_policy="uniform",
        num_steps=4,
    )
    selection = PolicyStepSelection.fixed(
        (2, 2, 2),
        num_steps=4,
        generator=selection_generator,
        policy=config.selection_contract_identity,
    )
    session = DynamicsSession.create(
        dynamics,
        num_steps=4,
        device="cpu",
        selection=selection,
    )

    policy = _execution_policy(group_size=3)
    execution = _single_step_rollout(
        config,
        execution_policy=policy,
    ).run_with_snapshot(replace(request, dynamics_session=session))
    result = execution.trajectory

    assert torch.equal(result.selected_timestep_index, torch.tensor([2, 2, 2]))
    assert execution.schedule_snapshot == session.snapshot
    assert session.snapshot.selected_policy_step_indices == (2, 2, 2)
    assert not hasattr(session, "current_step")


def test_single_step_prompt_key_shares_k_repeat_selection_with_first_ten_window():
    batch = _sample_batch(group_sizes=(3, 2))
    request, _dynamics, _events = _runtime(batch, seed=320)
    config = SingleStepRolloutConfig(
        selected_timestep_policy="uniform",
        num_steps=12,
        candidate_timestep_window=(0, 10),
        selection_key="prompt",
        selection_domain="single_process",
    )
    policy = _execution_policy(group_size=3)
    strategy = _single_step_rollout(config, execution_policy=policy)

    execution = strategy.run_with_snapshot(request)
    result = execution.trajectory
    selected = result.selected_timestep_index.tolist()

    assert selected[:3] == [selected[0]] * 3
    assert selected[3:] == [selected[3]] * 2
    assert all(0 <= item < 10 for item in selected)
    assert result.selection_policy_identity == config.selection_contract_identity
    assert result.selection_mapping_identity == (
        execution.schedule_snapshot.randomness_identity
    )
    assert execution.schedule_snapshot.selection_policy == (
        config.selection_contract_identity
    )


def test_single_step_explicit_candidates_never_select_outside_the_set():
    batch = _sample_batch(group_sizes=(4,))
    request, _dynamics, _events = _runtime(batch, seed=321)
    config = SingleStepRolloutConfig(
        selected_timestep_policy="uniform",
        num_steps=7,
        candidate_timestep_indices=(1, 3, 6),
        selection_key="row",
    )

    policy = _execution_policy(group_size=4)
    result = _single_step_rollout(config, execution_policy=policy).run(request)

    assert set(result.selected_timestep_index.tolist()) <= {1, 3, 6}


def test_single_step_selection_contract_identity_covers_mapping_semantics():
    baseline = SingleStepRolloutConfig(
        selected_timestep_policy="uniform",
        num_steps=40,
        candidate_timestep_window=(0, 10),
        selection_key="prompt",
        selection_domain="single_process",
    )
    changed_candidates = replace(
        baseline,
        candidate_timestep_window=(0, 9),
    )
    changed_key = replace(baseline, selection_key="row")
    changed_domain = replace(
        baseline,
        selection_domain="global_rank_broadcast",
    )

    assert (
        len(
            {
                baseline.selection_contract_identity,
                changed_candidates.selection_contract_identity,
                changed_key.selection_contract_identity,
                changed_domain.selection_contract_identity,
            }
        )
        == 4
    )


def test_single_step_distributed_selection_domain_fails_closed():
    batch = _sample_batch(group_sizes=(2,))
    request, _dynamics, events = _runtime(batch, seed=322)
    policy = _execution_policy(group_size=2)
    strategy = _single_step_rollout(
        SingleStepRolloutConfig(
            selected_timestep_policy="uniform",
            num_steps=4,
            candidate_timestep_window=(0, 10),
            selection_key="prompt",
            selection_domain="global_rank_broadcast",
        ),
        execution_policy=policy,
    )

    with pytest.raises(ValueError, match="distributed.*broadcast port"):
        strategy.run(request)
    assert "dynamics.sample_transition" not in events


def test_rollout_request_requires_the_explicit_replay_binding() -> None:
    batch = _sample_batch(group_sizes=(1,))
    request, _dynamics, _events = _runtime(batch, seed=33)
    state = SD3ScheduleReplayState(
        torch.tensor([900.5, 400.25]),
        torch.tensor([1.0, 0.6, 0.1]),
        scheduler_identity="test.scheduler-blueprint.v1",
    )
    replay_request = DynamicsReplayRequest("rollout-request-test", 2)
    binding = DynamicsReplayBinding(
        request=replay_request,
        factory_identity="test.factory.v1",
        replay_state=state,
    )
    dynamics = SD3FlowSDEDynamics(state, replay_binding=binding)

    with pytest.raises(ValueError, match="requires an explicit request binding"):
        replace(request, dynamics=dynamics)

    bound_request = replace(
        request,
        dynamics=dynamics,
        dynamics_replay_binding=binding,
    )
    assert bound_request.dynamics_replay_binding is binding

    wrong_binding = DynamicsReplayBinding(
        request=DynamicsReplayRequest("different-rollout", 2),
        factory_identity="test.factory.v1",
        replay_state=state,
    )
    with pytest.raises(ValueError, match="does not match"):
        replace(
            request,
            dynamics=dynamics,
            dynamics_replay_binding=wrong_binding,
        )


def test_rollout_registry_resolves_three_strict_typed_strategies():
    registry = _rollout_registry()
    assert registry.aliases == (
        "branching",
        "full-trajectory",
        "single-step",
    )
    resolver = DeclarationResolver()
    cases = (
        (
            "full-trajectory",
            {"num_steps": 3},
            FullTrajectoryRollout,
        ),
        (
            "branching",
            {
                "num_steps": 4,
                "branch_count": 2,
                "branch_topology": (
                    BranchTopology.single_point_branch_ablation(2).to_payload()
                ),
                "branch_step_policy": "uniform_intermediate",
            },
            BranchingRollout,
        ),
        (
            "single-step",
            {
                "selected_timestep_policy": "uniform",
            },
            SingleStepRollout,
        ),
    )
    for alias, params, expected_type in cases:
        declaration = resolver.resolve(registry, alias, params)
        policy = _execution_policy(group_size=2)
        instance = _load_rollout_declaration(
            declaration,
            policy=policy,
        )
        assert isinstance(instance, expected_type)
        assert declaration.declared_contract.component_id == alias
        if alias == "branching":
            assert declaration.declared_contract.rollout.physical_transition_count == (
                4,
                4,
            )
            assert (
                declaration.declared_contract.rollout.stored_policy_transition_count
                == (1, 1)
            )
        if alias == "single-step":
            assert declaration.declared_contract.rollout.physical_transition_count == (
                40,
                40,
            )
            assert (
                declaration.declared_contract.rollout.stored_policy_transition_count
                == (1, 1)
            )

    with pytest.raises(RegistryError) as failure:
        resolver.resolve(
            registry,
            "full-trajectory",
            {"num_steps": 3, "model": "forbidden"},
        )
    assert failure.value.code == "provider_failed"
    assert isinstance(failure.value.__cause__, ValueError)
    assert "unknown full-trajectory" in str(failure.value.__cause__)


def test_all_six_builtin_recipes_resolve_the_real_rollout_registry():
    definitions = builtin_recipe_definitions()
    assert len(definitions) == 6
    registry = _rollout_registry()
    resolver = DeclarationResolver()

    for definition in definitions:
        rollout_id, rollout_params = _canonical_rollout_selection(definition)
        declaration = resolver.resolve(
            registry,
            rollout_id,
            rollout_params,
        )
        assert declaration.declared_contract.component_kind == "rollout"
        assert declaration.alias == rollout_id


def test_canonical_rollout_declaration_package_is_import_safe():
    script = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('torch imported during plural rollouts import')
    if name == 'visual_rl.rollouts' or name.startswith('visual_rl.rollouts.'):
        raise RuntimeError('retired rollout package imported')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import visual_rl.algorithms.rollout
assert not any(name == 'visual_rl.rollouts' or name.startswith('visual_rl.rollouts.') for name in sys.modules)
print('import-safe')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "import-safe"
