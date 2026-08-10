"""Runnable production lifecycle with narrow injected boundary services."""

from __future__ import annotations

import gc
import hashlib
import weakref
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

import visual_rl.runtime.lifecycle as production_module
from visual_rl.algorithms.catalog import algorithm_domain_catalog_fragments
from visual_rl.algorithms.modules.interface import AlgorithmModule, BoundAlgorithm
from visual_rl.algorithms.dynamics.selection import DynamicsSelectionPolicyState
from visual_rl.algorithms.rollout.request import IterationRolloutRequestFactory
from visual_rl.algorithms.trainer.execution_plan import AlgorithmExecutionPlan
from visual_rl.algorithms.trainer.interface import IterationResult, StageValue
from visual_rl.core.serialization import canonical_json_text, strict_json_load
from visual_rl.artifacts.checkpoint import (
    AtomicCheckpointManager,
    CheckpointProgress,
    derive_reference_policy_state_evidence,
)
from visual_rl.core.contracts import (
    ComputePrecision,
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
from visual_rl.composition.registry import build_catalog
from visual_rl.artifacts.run_manifest import (
    launch_manifest_payload,
    recipe_manifest_payload,
    write_launch_manifest,
    write_recipe_manifest,
)
from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    CatalogFragment,
    ComponentDeclaration,
    ComponentDescriptor,
)
from visual_rl.core.contracts.runtime import AlgorithmStepResult
from visual_rl.core.filesystem_identity import filesystem_file_identity_from_snapshot
from visual_rl.core.types import FrozenMapping
from visual_rl.data import (
    DatasetArtifactBinding,
    GroupPlacementContract,
    GroupPlacementKind,
    MultiSourceSampler,
    PeriodicPhaseSchedule,
    PhaseDefinition,
    PreprocessComponentRole,
    PreprocessDependency,
    PreprocessGeometry,
    PreprocessPortContract,
    SourceLocationBinding,
    SourceSequence,
)
from visual_rl.data.prelude import DataPlanePrelude, PreludeBatchPayload
from visual_rl.errors import ArtifactError, ResumeError
from visual_rl.models import (
    SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA,
    ComponentRole,
    ModelAdapter,
    ModelLatentSpec,
    ModelPreprocessConsumerSpec,
    ModelPreprocessSpec,
    ModelRuntimeNumerics,
    ParameterViewEvidence,
    ParameterViewMode,
    SchedulerArtifactBlueprint,
)
from visual_rl.models.catalog import model_catalog_fragment
from visual_rl.models.numerics.execution import ParameterView
from visual_rl.composition.preflight import (
    ArtifactIdentityRequest,
    ArtifactIdentityResolution,
    RuntimeFacts,
)
from visual_rl.runtime import (
    CheckpointRequest,
    ComponentBindRequest,
    ComponentRuntimeEvidence,
    ControllerState,
    PolicyTensorRuntimeSpec,
    ProductionRuntimeError,
    RuntimeSession,
    SafePointCheckpointReceipt,
    StageAssemblyRequest,
    TrainerStageAssembly,
    TransformExecution,
)
from visual_rl.runtime.checkpoint_binding import _build_full_contract
from visual_rl.runtime.composition import create_run_controller
from visual_rl.runtime.types import RunResult
from visual_rl.data.samples import (
    SourceItemContext,
    StackedSampleBatch,
    T2IItem,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProductionModelConfig:
    artifact_ref: str

    @classmethod
    def from_mapping(cls, values, *, context):
        del context
        if not isinstance(values, dict) or set(values) != {"artifact_ref"}:
            raise ValueError("production model config requires artifact_ref")
        artifact_ref = values["artifact_ref"]
        if not isinstance(artifact_ref, str) or not artifact_ref:
            raise ValueError("artifact_ref must be non-empty")
        return cls(artifact_ref)


class ProductionModelDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "tests.test_runtime_production:ProductionModelConfig"

    @classmethod
    def declare_component(cls, raw_params, *, context):
        del cls, context
        config = ProductionModelConfig.from_mapping(raw_params, context=None)
        return ComponentDeclaration(
            config=config,
            declared_contract=_model_contract(),
        )


def _model_contract() -> DeclaredContract:
    return DeclaredContract(
        component_kind="model",
        component_id="production-test-model",
        model=ModelContract(
            tasks=(TaskKind.T2I,),
            output_media=(MediaKind.IMAGE,),
            latent_layouts=(LatentLayout.BCHW,),
            latent_ranks=(4,),
            axis_semantics=(("batch", "channel", "height", "width"),),
            prediction_types=(PredictionType.FLOW,),
            time_coordinates=(TimeCoordinate.FRACTIONAL_TIMESTEP,),
            training_modes=(TrainingMode.LORA,),
            supported_precisions=(ComputePrecision.BF16,),
            provides_reference_policy=True,
            condition_payload_types=("production_test_conditioning.v1",),
            spatial_stride=(8, 8),
            scheduler_blueprint_schema=(SCHEDULER_ARTIFACT_BLUEPRINT_SCHEMA),
            dynamics_binding_family="sd3.flow-sde.v1",
            schedule_coordinate=TimeCoordinate.FRACTIONAL_TIMESTEP,
            accepted_replay_state_schema_ids=("sd3.schedule-replay.v1",),
        ),
    )


class _SchedulerConfig(dict):
    def __getattr__(self, name):
        return self[name]


class _Scheduler:
    def __init__(self, config=None) -> None:
        self.config = _SchedulerConfig({} if config is None else config)
        self.set_timesteps(num_inference_steps=2, device="cpu")

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, *, num_inference_steps, device) -> None:
        self.timesteps = torch.linspace(
            900.0,
            100.0,
            num_inference_steps,
            device=device,
        )
        self.sigmas = torch.linspace(
            1.0,
            0.1,
            num_inference_steps + 1,
            device=device,
        )


class _Policy(torch.nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float16))

    def forward(self, value):
        return value * self.weight

    def close(self) -> None:
        self.events.append("prepared.close")


class ProductionModelAdapter(ModelAdapter):
    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = "tests.test_runtime_production:ProductionModelConfig"

    def __init__(
        self,
        config: ProductionModelConfig,
        events: list[str],
        precision: ComputePrecision,
    ) -> None:
        self.config = config
        self.events = events
        self.precision = precision
        self._scheduler_blueprint = SchedulerArtifactBlueprint.from_scheduler(
            _Scheduler()
        )

    @classmethod
    def describe(cls, config: object) -> DeclaredContract:
        if not isinstance(config, ProductionModelConfig):
            raise TypeError("config must be ProductionModelConfig")
        return _model_contract()

    @classmethod
    def from_config(cls, config, *, runtime_context):
        if not isinstance(config, ProductionModelConfig):
            raise TypeError("config must be ProductionModelConfig")
        if not isinstance(runtime_context, dict):
            raise TypeError("runtime_context must be a dict")
        events = runtime_context.get("test_events")
        if not isinstance(events, list):
            raise TypeError("test_events must be a list")
        try:
            precision = ComputePrecision(runtime_context.get("precision"))
        except (TypeError, ValueError) as exc:
            raise TypeError("precision must be a supported ComputePrecision") from exc
        assert "launch_spec" not in runtime_context
        assert "reward_artifacts" not in runtime_context
        assert "dataset_artifacts" not in runtime_context
        return cls(config, events, precision)

    @property
    def scheduler_artifact_blueprint(self):
        return self._scheduler_blueprint

    def describe_runtime_numerics(self):
        return ModelRuntimeNumerics(
            rollout_latent_dtype="float32",
            transition_latent_dtype="float32",
        )

    def describe_parameter_view_evidence(
        self,
        parameter_state,
        *,
        distribution_mode,
    ):
        if distribution_mode != "single":
            raise ValueError("production test adapter supports only single process")
        projection = parameter_state.state_projection
        return (
            ParameterViewEvidence(
                parameter_view=ParameterView.CURRENT,
                mode=ParameterViewMode.CURRENT,
                owner_component_names=(projection.standalone_saved_component_names),
                restorable_state_names=projection.standalone_parameter_names,
                source_projection_id=projection.projection_id,
                mutates_parameters_in_place=False,
            ),
            ParameterViewEvidence(
                parameter_view=ParameterView.REFERENCE,
                mode=ParameterViewMode.LORA_DISABLE,
                owner_component_names=(projection.standalone_saved_component_names),
                restorable_state_names=projection.standalone_parameter_names,
                source_projection_id=projection.projection_id,
                mutates_parameters_in_place=False,
            ),
        )

    def describe_preprocess(self):
        return ModelPreprocessSpec(
            implementation_id=("tests.test_runtime_production:ProductionModelAdapter"),
            implementation_revision="production-test-preprocess.v1",
            port=PreprocessPortContract(
                port_id="production-test-preprocess.v1",
                output_payload_type="production_test_conditioning.v1",
                dependencies=(
                    PreprocessDependency(
                        role=PreprocessComponentRole.MODEL,
                        logical_name="model_artifact",
                    ),
                ),
                producer_output_fields=("prompt_embeds",),
                schema_version=2,
            ),
            geometry=PreprocessGeometry(
                height=16,
                width=16,
                aspect_ratio_bucket="16x16",
            ),
            preprocess_config=FrozenMapping({"do_classifier_free_guidance": False}),
        )

    def describe_preprocess_consumption(self):
        return ModelPreprocessConsumerSpec(
            implementation_revision="production-test-consumer.v1",
            payload_type="production_test_conditioning.v1",
            required_modalities=("prompt_text",),
            positive_output_fields=("prompt_embeds",),
        )

    def load_components(self, session):
        session.acquire(
            "policy",
            lambda: _Policy(self.events),
            roles=(ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
        )
        return session.freeze()

    def latent_spec_for_batch(self, batch, *, device, dtype):
        if not isinstance(batch, StackedSampleBatch):
            raise TypeError("batch must be StackedSampleBatch")
        return ModelLatentSpec(
            shape=(batch.batch_size, 1, 2, 2),
            layout=LatentLayout.BCHW,
            axis_semantics=("batch", "channel", "height", "width"),
            device=device,
            dtype=dtype,
            spatial_stride=(8, 8),
            scheduler_patch_size=1,
        )

    def encode(self, batch):
        return batch

    def prepare_latents(self, latent_spec, *, generator):
        return torch.randn(
            latent_spec.shape,
            device=latent_spec.device,
            dtype=latent_spec.dtype,
            generator=generator,
        )

    def predict(self, model_input):
        return model_input

    def predict_reference(self, model_input):
        return model_input

    def decode(self, latents, latent_spec):
        del latent_spec
        return latents


def _catalog():
    return build_catalog(
        (
            model_catalog_fragment(),
            CatalogFragment(
                owner="test_runtime_production",
                kind="model",
                descriptors=(
                    ComponentDescriptor(
                        alias="production-test-model",
                        implementation_class_path=(
                            "tests.test_runtime_production:ProductionModelAdapter"
                        ),
                        declaration_provider_path=(
                            "tests.test_runtime_production:ProductionModelDeclarationProvider"
                        ),
                        optional_dependencies=("torch",),
                    ),
                ),
            ),
            *algorithm_domain_catalog_fragments(),
        )
    )


def _filesystem_identity(
    label: str,
    *,
    node_type: str,
    content_policy: str = "all-files.v1",
) -> FrozenMapping:
    return FrozenMapping(
        {
            "identity_schema": "filesystem-artifact.v1",
            "content_policy": content_policy,
            "node_type": node_type,
            "content_sha256": _digest(label),
            "file_count": 1,
            "byte_count": len(label),
        }
    )


class _ArtifactResolver:
    def __init__(self) -> None:
        self.requests: list[ArtifactIdentityRequest] = []

    def resolve_artifact_identities(self, request):
        if not isinstance(request, ArtifactIdentityRequest):
            raise TypeError("expected ArtifactIdentityRequest")
        self.requests.append(request)
        source_refs = tuple(
            sorted({item.artifact_ref for item in request.resolved.source_plan.sources})
        )
        reward_refs = tuple(
            sorted(item.artifact_ref for item in request.resolved.reward_plan.resources)
        )

        def dataset_identity(artifact_ref: str) -> FrozenMapping:
            location = request.locations.dataset(artifact_ref)
            if location.is_file() and not location.is_symlink():
                return FrozenMapping(
                    filesystem_file_identity_from_snapshot(location.read_bytes())
                )
            return _filesystem_identity(
                f"dataset:{artifact_ref}",
                node_type="file",
            )

        return ArtifactIdentityResolution(
            model_artifact_identity=_filesystem_identity("model", node_type="tree"),
            source_locations=SourceLocationBinding(
                request.resolved.source_plan.plan_id,
                tuple(
                    DatasetArtifactBinding(
                        artifact_ref=artifact_ref,
                        artifact_location=request.locations.dataset(artifact_ref),
                        expected_content_identity=dataset_identity(artifact_ref),
                    )
                    for artifact_ref in source_refs
                ),
            ),
            reward_artifact_identities=tuple(
                (
                    artifact_ref,
                    _filesystem_identity(
                        f"reward:{artifact_ref}",
                        node_type="tree",
                    ),
                )
                for artifact_ref in reward_refs
            ),
            code_artifact_identity=_filesystem_identity(
                "code",
                node_type="tree",
                content_policy="python-code.v1",
            ),
        )


class _Accelerator:
    def __init__(self, *, rebind_buffers_during_prepare: bool = False) -> None:
        self.prepare_calls = 0
        self.rebind_buffers_during_prepare = rebind_buffers_during_prepare

    def prepare(self, *values):
        self.prepare_calls += 1
        if self.rebind_buffers_during_prepare:
            values[0]._apply(lambda tensor: tensor.clone())
        return values

    @contextmanager
    def accumulate(self, root):
        yield root


class _RuntimeSessionFactory:
    def __init__(
        self,
        events: list[str],
        graph_resource: _RewardResource,
        *,
        rebind_buffers_during_prepare: bool = False,
    ) -> None:
        self.events = events
        self.graph_resource = graph_resource
        self.accelerator = _Accelerator(
            rebind_buffers_during_prepare=rebind_buffers_during_prepare
        )

    def create(self, request):
        self.events.append("runtime.create")
        recipe_id = request.environment.materialized.recipe_id
        return RuntimeSession(
            accelerator=self.accelerator,
            runtime_facts=RuntimeFacts(
                distribution_mode="single",
                rank=0,
                local_rank=0,
                world_size=1,
                device="cpu",
                precision="bf16",
                backend=None,
            ),
            peer_recipe_ids=(recipe_id,),
            closer=self._close,
        )

    def _close(self) -> None:
        self.graph_resource.close()
        self.events.append("runtime.close")


class _GraphCloseHook:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("graph.close")


class _RewardResource:
    def __init__(self, events: list[str], *, close_event: str) -> None:
        self.events = events
        self.close_event = close_event
        self.close_calls = 0

    def score(self, batch, context):
        del batch, context
        raise AssertionError("fake reward resource must not execute")

    def close(self) -> None:
        self.close_calls += 1
        self.events.append(self.close_event)


class _RuntimeContexts:
    def __init__(self, events: list[str], reward_resource: _RewardResource) -> None:
        self.events = events
        self.reward_resource = reward_resource

    def context_for(self, request):
        self.events.append(f"context.{request.kind}")
        if request.kind == "model":
            return {"test_events": self.events}
        if request.kind == "reward":
            return {"reward_resource": self.reward_resource}
        if request.kind == "trainer":
            return {"close_hooks": (_GraphCloseHook(self.events),)}
        return {}


class _ComponentBinder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def bind(self, request):
        if not isinstance(request, ComponentBindRequest):
            raise TypeError("expected ComponentBindRequest")
        self.events.append("g3.evidence")
        declared = request.graph.components.binding("model").declared_contract
        runtime_contracts = tuple(
            (
                slot,
                request.graph.components.binding(slot).attest_prepare(
                    runtime_identity=_digest(f"runtime:{slot}"),
                    verified_fields=(
                        (
                            ("model.reference_forward", "verified"),
                            ("runtime.graph", "verified"),
                        )
                        if slot == "model"
                        else (("runtime.graph", "verified"),)
                    ),
                ),
            )
            for slot in sorted(request.graph.components.slots)
        )
        model_contract = dict(runtime_contracts)["model"]
        tensor_runtime = PolicyTensorRuntimeSpec(
            device=request.runtime.session.runtime_facts.device,
            latent_storage_dtype="float32",
            model_compute_precision=request.runtime.session.runtime_facts.precision,
        )
        requirement_payload = {
            "schema_version": 1,
            "kind": "production_test_preprocess_requirement_set",
        }
        requirement_set_id = hashlib.sha256(
            canonical_json_text(requirement_payload).encode("utf-8")
        ).hexdigest()
        preprocess_identity = _digest("preprocess")
        execution_numerics = request.prepared.manager.model_execution_numerics
        assert declared.model is not None
        reference_policy_state = derive_reference_policy_state_evidence(
            algorithm=AlgorithmExecutionPlan.from_spec(
                request.preflight.environment.materialized.resolved.algorithm_spec,
                execution_policy=(
                    request.preflight.environment.materialized.resolved.execution_policy.to_receipt()
                ),
            ),
            model=declared.model,
            model_execution_numerics=execution_numerics,
        )
        return ComponentRuntimeEvidence(
            runtime_bound_contracts=runtime_contracts,
            verified_fields=FrozenMapping(
                {
                    "runtime.graph": "verified",
                    "policy_tensor_runtime": {
                        "spec_id": tensor_runtime.spec_id,
                        "spec": tensor_runtime.to_payload(),
                    },
                    "model_execution_numerics": (execution_numerics.to_payload()),
                    "reference_policy_state": (reference_policy_state.to_payload()),
                    "preprocess": {
                        "identity": preprocess_identity,
                        "requirement_set_id": requirement_set_id,
                        "requirement_set": requirement_payload,
                    },
                }
            ),
            model_runtime_contract=model_contract,
            policy_tensor_runtime_spec=tensor_runtime,
            model_execution_numerics=execution_numerics,
            reference_policy_state_evidence=reference_policy_state,
            preprocess_identity=preprocess_identity,
            preprocess_requirement_set_id=requirement_set_id,
        )


def _prelude() -> DataPlanePrelude:
    revision = "main-revision-v1"
    source = SourceSequence(
        source_id="main",
        revision=revision,
        items=tuple(
            T2IItem(
                prompt=f"prompt-{index}",
                source=SourceItemContext(
                    source_item_id=f"main-{index}",
                    dataset_source_id="main",
                    dataset_index=index,
                    dataset_revision=revision,
                ),
            )
            for index in range(3)
        ),
    )
    schedule = PeriodicPhaseSchedule(
        phases=(
            PhaseDefinition(
                phase_id="main",
                start_offset=0,
                end_offset=1,
                source_id="main",
                active_rewards=("reward_quality",),
            ),
        ),
        known_source_ids=frozenset({"main"}),
        known_reward_ids=frozenset({"reward_quality"}),
    )
    placement = GroupPlacementContract(
        placement=GroupPlacementKind.LOCAL_COMPLETE,
        global_prompt_batch_size=1,
        group_size=2,
        world_size=1,
        per_rank_microbatch_rows=2,
        gradient_accumulation_steps=1,
    )
    from visual_rl.data.samples import ExplicitCollator

    return DataPlanePrelude(
        phase_schedule=schedule,
        source_sampler=MultiSourceSampler((source,)),
        placement_contract=placement,
        collator=ExplicitCollator(),
    )


class _RolloutRequestStage:
    def __init__(self, factory, binding_ids: list[str]) -> None:
        self.factory = factory
        self.binding_ids = binding_ids

    def __call__(self, value):
        payload = value.payload
        if not isinstance(payload, PreludeBatchPayload):
            raise TypeError("expected PreludeBatchPayload")
        request = self.factory(payload.samples, value.identity)
        self.binding_ids.append(request.dynamics_replay_binding.binding_identity)
        return StageValue(value.identity, request)


class _PassThrough:
    def __call__(self, value):
        return StageValue(value.identity, value.payload)


class _RewardPassThrough(_PassThrough):
    def __init__(self, runtime_context_factory) -> None:
        self.runtime_context_factory = runtime_context_factory


class _Optimize:
    def __init__(self, request: StageAssemblyRequest, weights: list[float]) -> None:
        self.optimizer = request.prepared.optimizer
        self.scheduler = request.prepared.lr_scheduler
        self.parameter = request.prepared.manager.parameter_state.parameters()[0]
        self.weights = weights

    def __call__(self, value):
        self.optimizer.zero_grad(set_to_none=True)
        self.parameter.square().sum().backward()
        self.optimizer.step()
        self.scheduler.step()
        weight = float(self.parameter.detach().cpu().item())
        self.weights.append(weight)
        return StageValue(value.identity, {"weight": weight})


class _StageAssembler:
    def __init__(
        self,
        events: list[str],
        reward_resource: _RewardResource,
        *,
        fail_binding: bool = False,
    ) -> None:
        self.events = events
        self.reward_resource = reward_resource
        self.fail_binding = fail_binding
        self.requests: list[StageAssemblyRequest] = []
        self.binding_ids: list[str] = []
        self.weights: list[float] = []
        self.credit_reference_kl_weights: list[float] = []

    def assemble(self, request):
        if not isinstance(request, StageAssemblyRequest):
            raise TypeError("expected StageAssemblyRequest")
        assert request.prepared.manager.runtime_bound is (
            request.evidence.model_runtime_contract
        )
        assert request.graph_binding.bound_contract_id
        self.events.append("stages.assemble")
        self.requests.append(request)
        credit = request.graph.components.component("credit")
        reference_kl_weight = getattr(credit, "reference_kl_weight", None)
        if not isinstance(reference_kl_weight, float):
            raise TypeError("credit must expose a float reference_kl_weight")
        self.credit_reference_kl_weights.append(reference_kl_weight)
        rollout_factory = IterationRolloutRequestFactory(
            adapter=request.prepared.manager.adapter,
            dynamics_factory=request.dynamics_factory,
            num_steps=2,
            likelihood_semantics=LikelihoodSemantics.EXACT_ENV_ACTION,
            base_seed=request.preflight.compiled.training.seed,
            device="cpu",
            dtype=torch.float32,
        )

        def reward_runtime_context_factory(_payload, _identity):
            return object()

        passthrough = _PassThrough()
        assembly = TrainerStageAssembly(
            prelude=_prelude(),
            rollout=_RolloutRequestStage(rollout_factory, self.binding_ids),
            reward=_RewardPassThrough(reward_runtime_context_factory),
            advantage=passthrough,
            credit=passthrough,
            optimize=_Optimize(request, self.weights),
            reward_runtime_context_factory=reward_runtime_context_factory,
            close_resources=(self.reward_resource,),
        )
        if self.fail_binding:
            request.graph.trainer_ports.rollout.bind(_PassThrough())
        return assembly


class _CheckpointSink:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.requests: list[CheckpointRequest] = []

    def checkpoint(self, request):
        if not isinstance(request, CheckpointRequest):
            raise TypeError("expected CheckpointRequest")
        self.events.append("checkpoint")
        self.requests.append(request)
        output_dir = request.bound.preflight.compiled.launch.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            name: output_dir / filename
            for name, filename in {
                "authoritative_checkpoint": "checkpoint.pt",
                "resolved_config_path": "resolved.yaml",
                "manifest_path": "manifest.json",
                "metrics_path": "metrics.jsonl",
                "marker_path": "SUCCESS",
            }.items()
        }
        for path in paths.values():
            path.write_text("production-test\n", encoding="utf-8")
        summary = request.summary
        return RunResult(
            run_id=request.bound.runtime.launch_binding.launch_id,
            output_dir=output_dir,
            committed_steps=summary.committed_steps,
            authoritative_checkpoint=paths["authoritative_checkpoint"],
            resolved_config_path=paths["resolved_config_path"],
            manifest_path=paths["manifest_path"],
            metrics_path=paths["metrics_path"],
            marker_path=paths["marker_path"],
            last_metrics={
                "step": summary.committed_steps - 1,
                "sample_count": summary.last_iteration.value.identity.batch_size,
                "active_transition_count": 1,
                "loss": 0.0,
            },
        )


class _SafePointCheckpointSink(_CheckpointSink):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.safe_point_requests: list[CheckpointRequest] = []

    def checkpoint_safe_point(self, request):
        if not isinstance(request, CheckpointRequest):
            raise TypeError("expected CheckpointRequest")
        self.events.append("safe_checkpoint")
        self.safe_point_requests.append(request)
        checkpoint_path = (
            request.bound.preflight.compiled.launch.output_dir
            / "checkpoints"
            / f"step-{request.summary.committed_steps}"
        )
        checkpoint_path.mkdir(parents=True)
        return SafePointCheckpointReceipt(
            checkpoint_path=checkpoint_path,
            committed_steps=request.summary.committed_steps,
            checkpoint_contract_id="a" * 64,
            progress_id="b" * 64,
            state_tree_id="c" * 64,
        )


class _ContractCapturingCheckpointSink(_CheckpointSink):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.contract = None

    def checkpoint(self, request):
        self.contract = _build_full_contract(request.bound)
        return super().checkpoint(request)


class _TransformExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return TransformExecution(request.plan.plan_id, request.plan.transform_ids)


class _NoEvidenceRestore:
    def __init__(self) -> None:
        self.prepared_calls = 0

    def restore_prepared(self, request):
        del request
        self.prepared_calls += 1
        return 1

    def restore_bound(self, request):
        del request
        raise AssertionError("bound restore must not run without prepared evidence")


def _compatible_checkpoint(contract, root):
    progress = CheckpointProgress(
        global_step=1,
        iteration=1,
        next_optimizer_step=1,
        next_source_id="main",
        next_prompt_batch_id="d" * 64,
        next_phase_id="main",
        active_reward_ids=("quality",),
        source_cursors=(("main", 2),),
        dynamics_selection_policy=DynamicsSelectionPolicyState(base_seed=17),
        gradient_accumulation_position=0,
        ema_state_saved=False,
        reference_state_saved=False,
        execution_transform_plan_id=contract.execution_transform_plan_id,
        rng_state_id="e" * 64,
    )

    def write_state(path):
        (path / "training_state.bin").write_bytes(b"state")

    return (
        AtomicCheckpointManager(root)
        .commit(
            1,
            contract,
            write_state,
            progress=progress,
        )
        .path
    )


def _config(
    tmp_path,
    *,
    transform: str | None = None,
    checkpoint_cadence: int = 2,
    resume_from=None,
):
    output_dir = tmp_path / "run"
    transform_yaml = ""
    if transform is not None:
        preserves = "false" if transform == "unsafe" else "true"
        transform_yaml = f"""
  execution:
    transform_plan:
      schema_version: 1
      paradigm: coupled
      transforms:
        - transform_id: test-transform
          class_path: tests.test_runtime_production:TestTransform
          config: {{}}
          stage: both
          safety: lossless
          supported_training_paradigms: [coupled]
          preserves_parameter_identity: {preserves}
          preserves_state_dict_keys: {preserves}
          deterministic: true
"""
    path = tmp_path / "recipe.yaml"
    resume_value = "null" if resume_from is None else resume_from.as_posix()
    path.write_text(
        f"""schema_version: 2
recipe: flow_grpo_v1
overrides:
  model:
    id: production-test-model
    params:
      artifact_ref: main
  training:
    max_optimizer_steps: 2
    lr_schedule:
      warmup_steps: 0
{transform_yaml}launch:
  output_dir: {output_dir.as_posix()}
  resume_from: {resume_value}
  checkpoint_every_optimizer_steps: {checkpoint_cadence}
  artifacts:
    model: {(tmp_path / "model").as_posix()}
    datasets:
      main: {(tmp_path / "dataset").as_posix()}
    rewards:
      reward_quality: {(tmp_path / "reward").as_posix()}
""",
        encoding="utf-8",
    )
    return path


def _controller(
    tmp_path,
    *,
    transform_executor=None,
    restore_service=None,
    fail_binding: bool = False,
    support_safe_points: bool = False,
    capture_checkpoint_contract: bool = False,
    rebind_buffers_during_prepare: bool = False,
):
    events: list[str] = []
    graph_resource = _RewardResource(events, close_event="graph.resource.close")
    stage_resource = _RewardResource(events, close_event="stage.resource.close")
    artifacts = _ArtifactResolver()
    runtime = _RuntimeSessionFactory(
        events,
        graph_resource,
        rebind_buffers_during_prepare=rebind_buffers_during_prepare,
    )
    stages = _StageAssembler(
        events,
        stage_resource,
        fail_binding=fail_binding,
    )
    if support_safe_points:
        checkpoint = _SafePointCheckpointSink(events)
    elif capture_checkpoint_contract:
        checkpoint = _ContractCapturingCheckpointSink(events)
    else:
        checkpoint = _CheckpointSink(events)
    controller = create_run_controller(
        catalog=_catalog(),
        artifact_resolver=artifacts,
        runtime_factory=runtime,
        runtime_context_provider=_RuntimeContexts(events, graph_resource),
        stage_assembler=stages,
        component_binder=_ComponentBinder(events),
        checkpoint_sink=checkpoint,
        restore_service=restore_service,
        transform_executor=transform_executor,
    )
    return (
        controller,
        events,
        artifacts,
        runtime,
        stages,
        checkpoint,
        stage_resource,
    )


def test_two_update_production_lifecycle_binds_fresh_dynamics_and_run_result(
    tmp_path,
) -> None:
    (
        controller,
        events,
        artifacts,
        runtime,
        stages,
        checkpoint,
        reward_resource,
    ) = _controller(tmp_path)

    result = controller.run(_config(tmp_path))

    assert isinstance(result, RunResult)
    assert result.committed_steps == 2
    assert controller.state is ControllerState.CLOSED
    assert runtime.accelerator.prepare_calls == 1
    assert len(stages.requests) == 1
    assert stages.credit_reference_kl_weights == pytest.approx([0.004])
    assert len(stages.binding_ids) == 2
    assert len(set(stages.binding_ids)) == 2
    assert len(stages.weights) == 2
    assert stages.weights[0] != 1.0
    assert stages.weights[1] != stages.weights[0]
    assert events.index("g3.evidence") < events.index("stages.assemble")
    assert events.index("stages.assemble") < events.index("checkpoint")
    assert events.index("stage.resource.close") < events.index("prepared.close")
    assert events.index("prepared.close") < events.index("graph.close")
    assert events.index("graph.close") < events.index("runtime.close")
    assert events[-1] == "runtime.close"
    assert reward_resource.close_calls == 1
    assert runtime.graph_resource.close_calls == 1
    assert len(checkpoint.requests) == 1
    assert checkpoint.requests[0].summary.update_count == 2
    assert len(artifacts.requests) == 1
    assert artifacts.requests[0].locations.model == tmp_path / "model"
    bound = checkpoint.requests[0].bound
    recipe_path = tmp_path / "run" / "recipe.resolved.json"
    launch_path = tmp_path / "run" / "launch.resolved.json"
    expected_recipe = recipe_manifest_payload(
        bound.preflight.environment.materialized,
        bound.preflight.environment.component_artifact_bindings,
    )
    expected_launch = launch_manifest_payload(
        bound.runtime.launch_binding,
        bound.preflight.compiled.launch,
    )
    assert strict_json_load(recipe_path) == expected_recipe
    assert strict_json_load(launch_path) == expected_launch
    assert recipe_path.read_text(encoding="utf-8") == (
        canonical_json_text(expected_recipe) + "\n"
    )
    assert launch_path.read_text(encoding="utf-8") == (
        canonical_json_text(expected_launch) + "\n"
    )


def test_production_materializes_the_algorithm_owned_blueprint(
    tmp_path,
    monkeypatch,
) -> None:
    controller, events, *_rest = _controller(tmp_path)
    original_materialize = AlgorithmModule.materialize

    def tracked_materialize(
        module,
        policy,
        binding,
        spec,
        materializer,
        *,
        execution_policy,
    ):
        events.append("algorithm.materialize")
        assert binding is policy.binding
        return original_materialize(
            module,
            policy,
            binding,
            spec,
            materializer,
            execution_policy=execution_policy,
        )

    monkeypatch.setattr(
        AlgorithmModule,
        "materialize",
        tracked_materialize,
    )
    assert "bind" not in AlgorithmModule.__dict__

    result = controller.run(_config(tmp_path))

    assert result.committed_steps == 2
    assert events.count("algorithm.materialize") == 1
    assert events.index("algorithm.materialize") < events.index("stages.assemble")


def test_iteration_loop_releases_previous_result_before_next_update(
    tmp_path,
    monkeypatch,
) -> None:
    class _TrajectoryProbe:
        pass

    controller, _events, *_rest = _controller(tmp_path)
    probes: dict[int, weakref.ReferenceType[_TrajectoryProbe]] = {}
    original_run_iteration = BoundAlgorithm.run_iteration

    def tracked_run_iteration(bound_algorithm, optimizer_step):
        if optimizer_step > 0:
            gc.collect()
            assert probes[optimizer_step - 1]() is None
        result = original_run_iteration(bound_algorithm, optimizer_step)
        iteration = result.iteration
        assert isinstance(iteration, IterationResult)
        probe = _TrajectoryProbe()
        probes[optimizer_step] = weakref.ref(probe)
        observed = IterationResult(
            optimizer_step=iteration.optimizer_step,
            value=StageValue(iteration.value.identity, probe),
            stage_order=iteration.stage_order,
        )
        return AlgorithmStepResult(
            optimizer_step=optimizer_step,
            iteration=observed,
            algorithm_binding_id=result.algorithm_binding_id,
        )

    monkeypatch.setattr(
        BoundAlgorithm,
        "run_iteration",
        tracked_run_iteration,
    )

    result = controller.run(_config(tmp_path))

    assert result.committed_steps == 2
    assert probes[0]() is None
    final_probe = probes[1]()
    assert final_probe is not None
    checkpoint = controller._backend.checkpoint_sink
    assert isinstance(checkpoint, _CheckpointSink)
    final_iteration = checkpoint.requests[0].summary.last_iteration
    assert final_iteration.optimizer_step == 1
    assert final_iteration.value.payload is final_probe


def test_production_prepare_accepts_semantically_stable_buffer_rebinding(
    tmp_path,
) -> None:
    controller, _events, _artifacts, runtime, *_rest = _controller(
        tmp_path,
        rebind_buffers_during_prepare=True,
    )

    result = controller.run(_config(tmp_path))

    assert result.committed_steps == 2
    assert runtime.accelerator.prepare_calls == 1


def test_manifests_publish_before_model_construction_in_lifecycle(
    tmp_path,
    monkeypatch,
) -> None:
    controller, events, *_rest = _controller(tmp_path)
    write_recipe = write_recipe_manifest
    write_launch = write_launch_manifest

    def tracked_recipe(path, recipe, artifact_binding_set):
        events.append("recipe.manifest")
        return write_recipe(path, recipe, artifact_binding_set)

    def tracked_launch(path, runtime, launch):
        events.append("launch.manifest")
        return write_launch(path, runtime, launch)

    monkeypatch.setattr(
        production_module,
        "write_recipe_manifest",
        tracked_recipe,
    )
    monkeypatch.setattr(
        production_module,
        "write_launch_manifest",
        tracked_launch,
    )

    controller.run(_config(tmp_path))

    assert events.index("recipe.manifest") < events.index("runtime.create")
    assert events.index("runtime.create") < events.index("launch.manifest")
    assert events.index("launch.manifest") < events.index("context.model")


@pytest.mark.parametrize("mode", ("drift", "symlink"))
def test_recipe_manifest_conflict_fails_before_runtime_or_model(
    tmp_path,
    mode: str,
) -> None:
    target = tmp_path / "run" / "recipe.resolved.json"
    target.parent.mkdir(parents=True)
    backing = tmp_path / "recipe-backing.json"
    if mode == "drift":
        target.write_text("drift\n", encoding="utf-8")
    else:
        backing.write_text("untouched\n", encoding="utf-8")
        target.symlink_to(backing)
    controller, events, _artifacts, _runtime, stages, _checkpoint, resource = (
        _controller(tmp_path)
    )

    with pytest.raises(ArtifactError):
        controller.run(_config(tmp_path))

    assert "runtime.create" not in events
    assert "context.model" not in events
    assert stages.requests == []
    assert resource.close_calls == 0
    if mode == "symlink":
        assert backing.read_text(encoding="utf-8") == "untouched\n"


@pytest.mark.parametrize("mode", ("drift", "symlink"))
def test_launch_manifest_conflict_closes_created_runtime_before_model(
    tmp_path,
    mode: str,
) -> None:
    initial, *_initial_services = _controller(tmp_path)
    initial.run(_config(tmp_path))
    target = tmp_path / "run" / "launch.resolved.json"
    backing = tmp_path / "launch-backing.json"
    if mode == "drift":
        target.write_text("drift\n", encoding="utf-8")
    else:
        target.unlink()
        backing.write_text("untouched\n", encoding="utf-8")
        target.symlink_to(backing)
    controller, events, _artifacts, runtime, stages, _checkpoint, resource = (
        _controller(tmp_path)
    )

    with pytest.raises(ArtifactError):
        controller.run(_config(tmp_path))

    assert events.index("runtime.create") < events.index("runtime.close")
    assert "context.model" not in events
    assert runtime.accelerator.prepare_calls == 0
    assert stages.requests == []
    assert resource.close_calls == 0
    assert runtime.graph_resource.close_calls == 1
    if mode == "symlink":
        assert backing.read_text(encoding="utf-8") == "untouched\n"


def test_nonempty_transform_without_executor_hard_fails_before_g3(tmp_path) -> None:
    controller, _events, _artifacts, _runtime, stages, _checkpoint, resource = (
        _controller(tmp_path)
    )

    with pytest.raises(ProductionRuntimeError, match="requires an executor"):
        controller.run(_config(tmp_path, transform="safe"))

    assert stages.requests == []
    assert resource.close_calls == 0


def test_identity_changing_transform_is_rejected_before_executor_or_g3(
    tmp_path,
) -> None:
    executor = _TransformExecutor()
    controller, _events, _artifacts, _runtime, stages, _checkpoint, resource = (
        _controller(
            tmp_path,
            transform_executor=executor,
        )
    )

    with pytest.raises(
        ProductionRuntimeError,
        match="must preserve parameter identity and state-dict keys",
    ):
        controller.run(_config(tmp_path, transform="unsafe"))

    assert executor.calls == 0
    assert stages.requests == []
    assert resource.close_calls == 0


def test_intermediate_checkpoint_cadence_is_not_silently_ignored(tmp_path) -> None:
    controller, _events, _artifacts, _runtime, stages, _checkpoint, resource = (
        _controller(tmp_path)
    )

    with pytest.raises(ProductionRuntimeError, match="intermediate checkpoint cadence"):
        controller.run(_config(tmp_path, checkpoint_cadence=1))

    assert stages.requests == []
    assert resource.close_calls == 0


def test_iteration_loop_commits_safe_point_at_declared_cadence(tmp_path) -> None:
    controller, events, _artifacts, _runtime, _stages, checkpoint, _resource = (
        _controller(tmp_path, support_safe_points=True)
    )

    result = controller.run(_config(tmp_path, checkpoint_cadence=1))

    assert result.committed_steps == 2
    assert isinstance(checkpoint, _SafePointCheckpointSink)
    assert len(checkpoint.safe_point_requests) == 1
    assert checkpoint.safe_point_requests[0].summary.committed_steps == 1
    assert events.index("safe_checkpoint") < events.index("checkpoint")


def test_invalid_resume_checkpoint_fails_before_restore_or_runtime(tmp_path) -> None:
    resume = tmp_path / "checkpoint"
    resume.mkdir()
    restore = _NoEvidenceRestore()
    controller, events, _artifacts, _runtime, stages, _checkpoint, resource = (
        _controller(tmp_path, restore_service=restore)
    )

    with pytest.raises(ResumeError, match="metadata is incomplete or invalid"):
        controller.run(_config(tmp_path, resume_from=resume))

    assert restore.prepared_calls == 0
    assert "runtime.create" not in events
    assert stages.requests == []
    assert resource.close_calls == 0


def test_complete_resume_still_requires_typed_restore_evidence(tmp_path) -> None:
    first, _events, _artifacts, _runtime, _stages, checkpoint, _resource = _controller(
        tmp_path, capture_checkpoint_contract=True
    )
    first.run(_config(tmp_path))
    assert isinstance(checkpoint, _ContractCapturingCheckpointSink)
    assert checkpoint.contract is not None
    resume = _compatible_checkpoint(
        checkpoint.contract,
        tmp_path / "resume-checkpoints",
    )
    restore = _NoEvidenceRestore()
    controller, events, _artifacts, _runtime, stages, _checkpoint, resource = (
        _controller(tmp_path, restore_service=restore)
    )

    with pytest.raises(TypeError, match="PreparedRestoreResult evidence"):
        controller.run(_config(tmp_path, resume_from=resume))

    assert restore.prepared_calls == 1
    assert events.index("runtime.create") < events.index("context.model")
    assert "runtime.close" in events
    assert stages.requests == []
    assert resource.close_calls == 0


def test_bind_failure_closes_stage_resources_once_before_other_owners(
    tmp_path,
) -> None:
    controller, events, *_rest, resource = _controller(
        tmp_path,
        fail_binding=True,
    )

    with pytest.raises(RuntimeError, match="already bound"):
        controller.run(_config(tmp_path))

    assert resource.close_calls == 1
    assert events.index("stage.resource.close") < events.index("prepared.close")
    assert events.index("prepared.close") < events.index("graph.close")
    assert events.index("graph.close") < events.index("runtime.close")
