"""Closed-loop tests for the model-agnostic policy recompute bridge."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields

import numpy as np
import torch

from tests.support.policy_recompute_oracle import compute_full_policy_stats_oracle
from tests.support.policy_update_oracle import LocalTestAccelerator
from visual_rl.algorithms.dynamics.interface import (
    DeterministicTransitionOutput,
    Dynamics,
    TransitionMeanStd,
    TransitionPolicyMetadata,
)
from visual_rl.algorithms.optimization.advantage import GroupZScoreAdvantageProcessor
from visual_rl.algorithms.optimization.credit import GRPOCreditStrategy
from visual_rl.algorithms.optimization.recompute import (
    PolicyRecomputer,
    PolicyRecomputeRequest,
    PolicyStats,
)
from visual_rl.algorithms.rollout.config import (
    FullTrajectoryRolloutConfig,
    SingleStepRolloutConfig,
)
from visual_rl.algorithms.rollout.full_trajectory import FullTrajectoryRollout
from visual_rl.algorithms.rollout.interface import RolloutRequest
from visual_rl.algorithms.rollout.single_step import SingleStepRollout
from visual_rl.algorithms.trainer.grpo import BaseTrainer
from visual_rl.algorithms.trainer.interface import (
    IterationIdentity,
    PrepareRunContext,
    StageValue,
)
from visual_rl.algorithms.trainer.stages import (
    AdvantageStage,
    CreditStage,
    OptimizeStage,
    RewardPipelineStage,
    RolloutStage,
)
from visual_rl.core.contracts import (
    ComputePrecision,
    DeclaredContract,
    LatentLayout,
    MediaKind,
    ModelContract,
    PredictionType,
    TaskKind,
    TimeCoordinate,
    TrainingMode,
)
from visual_rl.core.types import FrozenMapping, StepContext
from visual_rl.core.contracts import (
    ExecutionPolicyReceipt,
    LogicalRewardSpec,
    RewardContract,
    RewardGranularity,
    RewardPlanSpec,
    RewardResourceSpec,
    RewardRouteBinding,
    RewardRouteSpec,
)
from visual_rl.data.media import DecodedMediaBatch
from visual_rl.models import (
    BatchRowProjection,
    ModelAdapter,
    ModelInput,
    ModelLatentSpec,
    ModelPrediction,
)
from visual_rl.algorithms.rewards import (
    PointwiseReward,
    PointwiseRewardOutput,
    RewardRuntimeContext,
    RewardStage,
)
from visual_rl.runtime.reward_resources import RewardPool
from visual_rl.data.samples import (
    BatchRowContext,
    ExplicitCollator,
    SourceItemContext,
    T2IItem,
)
def test_recompute_public_surface_is_slot_only_and_has_one_stats_owner() -> None:
    assert not hasattr(PolicyRecomputer, "compute")
    assert tuple(field.name for field in fields(PolicyStats)) == (
        "grouping",
        "current_log_probs",
        "current_transition_mean",
        "transition_std",
        "reference_transition_mean",
    )
    assert PolicyStats.__module__ == "visual_rl.algorithms.optimization.recompute"
    assert PolicyRecomputer.__module__ == "visual_rl.algorithms.optimization.recompute"


@dataclass(frozen=True)
class _Conditioning:
    condition_identity: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return len(self.condition_identity)

    def project_rows(self, projection: BatchRowProjection) -> _Conditioning:
        return _Conditioning(projection.project_tuple(self.condition_identity))


class _TrainableAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.reference_weight = torch.tensor(-0.125)
        self.encode_calls = 0

    @classmethod
    def describe(cls, config):
        del config
        return DeclaredContract(
            component_kind="model",
            component_id="recompute-fake",
            model=ModelContract(
                tasks=(TaskKind.T2I,),
                output_media=(MediaKind.IMAGE,),
                latent_layouts=(LatentLayout.BCHW,),
                latent_ranks=(4,),
                axis_semantics=(("batch", "channel", "height", "width"),),
                prediction_types=(PredictionType.FLOW,),
                time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
                training_modes=(TrainingMode.LORA,),
                supported_precisions=(ComputePrecision.FP32,),
                provides_reference_policy=True,
            ),
        )

    @classmethod
    def from_config(cls, config, *, runtime_context):
        del config, runtime_context
        return cls()

    def load_components(self, session):
        raise AssertionError("component loading is outside this analytical test")

    def encode(self, batch):
        self.encode_calls += 1
        return _Conditioning(tuple(row.identity for row in batch.rows))

    def prepare_latents(self, latent_spec, *, generator):
        return torch.randn(
            latent_spec.shape,
            dtype=latent_spec.dtype,
            device=latent_spec.device,
            generator=generator,
        )

    def predict(self, model_input: ModelInput):
        return self._prediction(model_input, self.weight)

    def predict_reference(self, model_input: ModelInput):
        return self._prediction(model_input, self.reference_weight)

    @staticmethod
    def _prediction(model_input: ModelInput, weight):
        return ModelPrediction(
            value=model_input.latents * weight,
            prediction_type=PredictionType.FLOW,
            condition_identity=model_input.condition_identity,
            guidance_identity=model_input.guidance_identity,
        )

    def decode(self, latents, latent_spec):
        assert tuple(latents.shape) == latent_spec.shape
        return DecodedMediaBatch(
            tensor=latents.detach().clone(),
            layout="BCHW",
        )


class _Dynamics(Dynamics):
    def __init__(self, metadata: str = "none") -> None:
        self.metadata = metadata

    @property
    def dynamics_config_identity(self) -> str:
        return f"recompute-dynamics:{self.metadata}"

    def timesteps(self, *, num_steps: int, device):
        return torch.linspace(1.0, 0.25, num_steps, device=device)

    def terminal_timestep(self, *, device):
        return torch.tensor(0.0, device=device)

    def add_noise(self, clean, noise, timestep):
        shape = (clean.shape[0], *([1] * (clean.ndim - 1)))
        return clean + noise * timestep.reshape(shape)

    def transition_mean_std(self, transition):
        shape = (transition.batch_size, *([1] * (transition.x_t.ndim - 1)))
        dt = transition.t_next - transition.t
        return TransitionMeanStd(
            mean=(transition.x_t + transition.model_prediction * dt.reshape(shape)),
            std=torch.full(
                shape,
                0.2,
                dtype=transition.x_t.dtype,
                device=transition.x_t.device,
            ),
            dt=dt,
        )

    def _deterministic_ode_step(self, transition):
        shape = (transition.batch_size, *([1] * (transition.x_t.ndim - 1)))
        dt = transition.t_next - transition.t
        return DeterministicTransitionOutput(
            next_state=(
                transition.x_t + transition.model_prediction * dt.reshape(shape)
            ).detach(),
            dt=dt.detach(),
        )

    def policy_metadata(self, transition, stats):
        stats.validate_against(transition)
        if self.metadata == "std":
            return TransitionPolicyMetadata(
                transition_std_dev=stats.std.reshape(
                    transition.batch_size,
                    -1,
                )[:, 0].detach()
            )
        if self.metadata == "coefficient":
            return TransitionPolicyMetadata(
                rectification_coefficient=(
                    transition.x_t.detach()
                    .reshape(transition.batch_size, -1)
                    .abs()
                    .mean(dim=1)
                    + 1.0
                )
            )
        return TransitionPolicyMetadata()


def _batch(batch_size: int = 4):
    items = []
    rows = []
    for index in range(batch_size):
        source = SourceItemContext(
            source_item_id=f"source-{index // 2}",
            dataset_source_id="main",
            dataset_index=index // 2,
            dataset_revision="revision-1",
        )
        items.append(T2IItem(prompt=f"prompt-{index // 2}", source=source))
        rows.append(
            BatchRowContext(
                occurrence_id=f"occurrence-{index // 2}",
                group_id=f"group-{index // 2}",
                member_id=index % 2,
                phase="main",
                optimizer_step=0,
                source_item_id=source.source_item_id,
            )
        )
    return ExplicitCollator().collate_samples(tuple(items), tuple(rows))


def _latent_spec(batch_size: int) -> ModelLatentSpec:
    return ModelLatentSpec(
        shape=(batch_size, 1, 2, 2),
        layout=LatentLayout.BCHW,
        axis_semantics=("batch", "channel", "height", "width"),
        device="cpu",
        dtype=torch.float32,
    )


def _execution_policy(*, group_size: int) -> ExecutionPolicyReceipt:
    return ExecutionPolicyReceipt.from_payload(
        {
            "schema_version": 1,
            "training_mode": "lora",
            "distribution_mode": "single",
            "precision": "fp32",
            "group_size": group_size,
            "rollout": {
                "forward_microbatch_size": 1,
                "decode_microbatch_size": 1,
                "trajectory_storage_device": "cpu",
            },
            "transform_plan": {
                "schema_version": 1,
                "paradigm": "coupled",
                "transforms": (),
            },
        }
    )


def _rollout(adapter, dynamics, *, single_step: bool = False):
    samples = _batch()
    latent_spec = _latent_spec(samples.batch_size)
    execution_policy = _execution_policy(group_size=samples.batch_size)
    request = RolloutRequest(
        adapter=adapter,
        dynamics=dynamics,
        samples=samples,
        latent_spec=latent_spec,
        generator=torch.Generator().manual_seed(1776),
        selection_generator=torch.Generator().manual_seed(91),
        likelihood_semantics="exact_env_action",
    )
    strategy = (
        SingleStepRollout(
            SingleStepRolloutConfig(
                selected_timestep_policy="uniform",
                num_steps=3,
                selected_timestep_index=1,
            ),
            execution_policy=execution_policy,
            expected_policy_id=execution_policy.policy_id,
        )
        if single_step
        else FullTrajectoryRollout(
            FullTrajectoryRolloutConfig(num_steps=3),
            execution_policy=execution_policy,
            expected_policy_id=execution_policy.policy_id,
        )
    )
    return strategy.run_with_snapshot(request), latent_spec


def test_recompute_matches_first_rollout_log_probs_and_preserves_gradient() -> None:
    adapter = _TrainableAdapter()
    dynamics = _Dynamics()
    rollout, latent_spec = _rollout(adapter, dynamics)

    stats = compute_full_policy_stats_oracle(
        PolicyRecomputeRequest(
            adapter=adapter,
            dynamics=dynamics,
            rollout=rollout,
            latent_spec=latent_spec,
        )
    )

    torch.testing.assert_close(
        stats.current_log_probs,
        rollout.trajectory.old_log_probs,
        rtol=0,
        atol=0,
    )
    assert stats.current_log_probs.requires_grad
    stats.current_log_probs.sum().backward()
    assert adapter.weight.grad is not None
    assert bool(torch.isfinite(adapter.weight.grad))


def test_recompute_builds_detached_reference_stats_from_the_same_snapshot() -> None:
    adapter = _TrainableAdapter()
    dynamics = _Dynamics(metadata="std")
    rollout, latent_spec = _rollout(adapter, dynamics)

    stats = compute_full_policy_stats_oracle(
        PolicyRecomputeRequest(
            adapter=adapter,
            dynamics=dynamics,
            rollout=rollout,
            latent_spec=latent_spec,
            require_reference_statistics=True,
        )
    )

    assert rollout.trajectory.transition_std_dev is not None
    assert rollout.trajectory.transition_std_dev.shape == (4, 3)
    assert stats.current_transition_mean is not None
    assert stats.current_transition_mean.requires_grad
    assert stats.reference_transition_mean is not None
    assert not stats.reference_transition_mean.requires_grad
    assert stats.reference_transition_mean.grad_fn is None
    assert stats.transition_std is not None
    assert stats.transition_std.shape == (4, 3, 1, 1, 1)
    assert not torch.equal(
        stats.current_transition_mean.detach(),
        stats.reference_transition_mean,
    )


def test_recompute_does_not_duplicate_rollout_credit_metadata() -> None:
    adapter = _TrainableAdapter()
    dynamics = _Dynamics(metadata="coefficient")
    rollout, latent_spec = _rollout(adapter, dynamics, single_step=True)
    stats = compute_full_policy_stats_oracle(
        PolicyRecomputeRequest(
            adapter=adapter,
            dynamics=dynamics,
            rollout=rollout,
            latent_spec=latent_spec,
        )
    )
    coefficient = rollout.trajectory.rectification_coefficient
    assert coefficient is not None
    assert coefficient.shape == (4, 1)
    assert not hasattr(stats, "rectification_coefficient")
    assert not hasattr(stats, "coefficient_normalization_mean")
    assert not hasattr(stats, "transition_std_dev")


@dataclass
class _RewardResource:
    close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


class _RowReward(PointwiseReward):
    def score(self, *, logical_reward_id, resource, batch):
        assert logical_reward_id == "quality"
        assert isinstance(resource, _RewardResource)
        return PointwiseRewardOutput(
            identity=batch.identity,
            values=np.array([1.0, 3.0, 2.0, 6.0], dtype=np.float64),
            valid_mask=np.ones(batch.batch_size, dtype=np.bool_),
        )


class _Prelude:
    def __init__(self, samples) -> None:
        self.samples = samples
        self.commits = 0
        self.aborts = 0

    def build(self, optimizer_step):
        rows = self.samples.rows
        identity = IterationIdentity(
            optimizer_step=optimizer_step,
            source_id="main",
            phase_id="main",
            row_identities=tuple(row.identity for row in rows),
            group_ids=tuple(row.group_id for row in rows),
            member_ids=tuple(row.member_id for row in rows),
        )
        return StageValue(identity, self.samples)

    def commit_iteration(self, identity):
        assert identity.optimizer_step == 0
        self.commits += 1

    def abort_iteration(self, identity):
        assert identity.optimizer_step == 0
        self.aborts += 1


def test_six_typed_stages_execute_one_complete_grpo_update() -> None:
    samples = _batch()
    adapter = _TrainableAdapter()
    dynamics = _Dynamics()
    latent_spec = _latent_spec(samples.batch_size)
    execution_policy = _execution_policy(group_size=samples.batch_size)
    rollout_component = FullTrajectoryRollout(
        FullTrajectoryRolloutConfig(num_steps=2),
        execution_policy=execution_policy,
        expected_policy_id=execution_policy.policy_id,
    )

    def request_factory(batch, identity):
        assert batch is samples
        return RolloutRequest(
            adapter=adapter,
            dynamics=dynamics,
            samples=batch,
            latent_spec=latent_spec,
            generator=torch.Generator().manual_seed(100 + identity.optimizer_step),
            likelihood_semantics="exact_env_action",
        )

    reward_resource = RewardResourceSpec(
        descriptor=FrozenMapping({"artifact_ref": "quality"}),
        artifact_identity=FrozenMapping({"content_sha256": "a" * 64}),
    )
    plan = RewardPlanSpec(
        resources=(reward_resource,),
        logical_rewards=(
            LogicalRewardSpec(
                logical_reward_id="quality",
                component_declaration_id=(
                    "component-declaration.v1:" + hashlib.sha256(b"quality").hexdigest()
                ),
                resource_identity=reward_resource.resource_identity,
                contract=RewardContract(
                    accepted_media=(MediaKind.IMAGE,),
                    required_payload_type=None,
                    granularity=RewardGranularity.POINTWISE,
                    output_rank=1,
                    frame_aggregation=None,
                ),
            ),
        ),
        routes=(
            RewardRouteSpec(
                source_id="main",
                phase_id="main",
                rewards=(RewardRouteBinding(logical_reward_id="quality", weight=1.0),),
            ),
        ),
    )
    pool = RewardPool(plan, lambda _identity: _RewardResource())
    reward = RewardStage(
        plan=plan,
        pool=pool.view(),
        logical_rewards={"quality": _RowReward()},
    )
    pool.activate()
    optimizer = torch.optim.SGD((adapter.weight,), lr=0.05)
    prelude = _Prelude(samples)
    trainer = BaseTrainer(
        prelude=prelude,
        rollout=RolloutStage(
            rollout=rollout_component,
            request_factory=request_factory,
        ),
        reward=RewardPipelineStage(
            reward,
            runtime_context_factory=lambda _rollout, identity: RewardRuntimeContext(
                StepContext(
                    step=identity.optimizer_step,
                    seed=100 + identity.optimizer_step,
                )
            ),
        ),
        advantage=AdvantageStage(GroupZScoreAdvantageProcessor()),
        credit=CreditStage(
            strategy=GRPOCreditStrategy(clip_range=0.1),
        ),
        optimize=OptimizeStage(
            optimizer=optimizer,
            scaler=None,
            accelerator=LocalTestAccelerator(None),
            prepared_root=(adapter.weight,),
        ),
    )
    trainer.prepare_run(
        PrepareRunContext(
            run_id="fake-run",
            recipe_id="fake-recipe",
            start_optimizer_step=0,
        )
    )
    before = adapter.weight.detach().clone()

    result = trainer.run_iteration(0)

    assert result.stage_order == (
        "prelude",
        "rollout",
        "reward",
        "advantage",
        "credit",
        "optimize",
    )
    assert result.value.payload.update.optimizer_step == 0
    assert result.value.payload.update.logprob_delta_max == 0.0
    assert adapter.encode_calls == 1
    assert not torch.equal(adapter.weight.detach(), before)
    assert prelude.commits == 1
    assert prelude.aborts == 0
    trainer.close()
    pool.close()
