"""Concrete v0.8 production lifecycle assembled from existing contracts.

The backend owns lifecycle order and the one model preparation root.  Dataset
iteration, reward execution, runtime probes, restore mechanics, transforms,
and checkpoint persistence remain narrow injected services because those
boundaries require deployment-specific state or side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visual_rl.algorithms.modules.interface import (
    BoundAlgorithm,
)
from visual_rl.algorithms.optimization.execution import (
    OptimizerExecutionSpec,
    build_adamw,
    build_lr_scheduler,
)
from visual_rl.algorithms.trainer.interface import IterationResult, PrepareRunContext
from visual_rl.artifacts.checkpoint.manager import AtomicCheckpointManager
from visual_rl.composition.config.bootstrap import bootstrap_recipe_v2
from visual_rl.composition.config.compiler import default_catalog
from visual_rl.composition.config.integration import (
    DynamicsProjectionRegistry,
    default_dynamics_projection_registry,
)
from visual_rl.composition.config.source import load_source_recipe
from visual_rl.composition.config.specs import LaunchSpec, TrainingSpec
from visual_rl.composition.registry import (
    AlgorithmDeclarationResolver,
    Catalog,
    DeclarationResolver,
    ResolvedComponentDeclaration,
)
from visual_rl.artifacts.run_manifest import (
    assert_launch_manifest_resume_compatible,
    assert_recipe_manifest_resume_compatible,
    write_launch_manifest,
    write_recipe_manifest,
)
from visual_rl.core.contracts import (
    BoundPolicyCapabilities,
    ComputePrecision,
    DistributionMode,
)
from visual_rl.core.contracts.runtime import (
    AlgorithmStepResult,
)
from visual_rl.errors import ResumeError
from visual_rl.models.interface import ModelAdapter
from visual_rl.models.lifecycle.components import ComponentManager
from visual_rl.models.numerics.policy import (
    ParameterDTypePolicy,
)
from visual_rl.composition.preflight import (
    EnvironmentPreflightResult,
    RuntimeBindInput,
    RuntimeBindResult,
    RuntimeFacts,
    RuntimeGraphBindInput,
    bind_runtime,
    bind_runtime_graph,
    run_environment_preflight,
    run_static_preflight,
)
from visual_rl.runtime.reward_resources import RewardResourceState
from visual_rl.runtime.algorithm_binding import (
    bind_per_rollout_dynamics_factory,
    materialize_algorithm_runtime_components,
    validate_algorithm_runtime_contracts,
)
from visual_rl.runtime.algorithm_binding import (
    CanonicalAlgorithmMaterializer,
)
from visual_rl.runtime.component_graph import (
    load_component_graph,
)
from visual_rl.runtime.component_loader import RuntimeComponentLoader
from visual_rl.runtime.controller import RestoreBoundOutcome
from visual_rl.artifacts.terminal import build_launch_runtime_audit
from visual_rl.runtime.model_binding import (
    DefaultPolicyRuntimePort,
    resolve_model_execution_numerics,
)
from visual_rl.runtime.resources import (
    DefaultRuntimeResourceContainer,
)
from visual_rl.runtime.reward_resources import (
    RewardResourceFactory,
    RuntimeResourceAcquisitionError,
)
from visual_rl.runtime.reward_resources import (
    compile_reward_resource_acquisition,
)
from visual_rl.runtime.types import RunResult
from visual_rl.runtime.transforms import execute_runtime_transforms

__all__ = (
    "BoundRestoreRequest",
    "BoundRestoreResult",
    "CheckpointRequest",
    "CheckpointSink",
    "CompiledProductionRun",
    "ComponentBindRequest",
    "ComponentRuntimeBinder",
    "ComponentRuntimeEvidence",
    "PolicyTensorRuntimeSpec",
    "PreparedRestoreRequest",
    "PreparedRestoreResult",
    "ProductionBoundRun",
    "ProductionGraph",
    "ProductionPreflight",
    "ProductionPreparedRun",
    "ProductionRuntime",
    "ProductionRuntimeContextProvider",
    "ProductionRuntimeError",
    "ProductionRuntimeLifecycleBackend",
    "RestoreService",
    "RuntimeContextRequest",
    "RuntimeCreateRequest",
    "RuntimeSession",
    "RuntimeSessionFactory",
    "SafePointCheckpointReceipt",
    "StageAssembler",
    "StageAssemblyRequest",
    "StageCheckpointPortError",
    "StageCheckpointPorts",
    "TrainerStageAssembly",
    "TransformExecution",
    "TransformExecutor",
    "TransformRequest",
)


from visual_rl.runtime.types import (
    PendingRunSummary,
    _TrainerPorts,
    BoundRestoreRequest,
    BoundRestoreResult,
    CheckpointRequest,
    CheckpointSink,
    CompiledProductionRun,
    ComponentBindRequest,
    ComponentRuntimeBinder,
    ComponentRuntimeEvidence,
    PolicyTensorRuntimeSpec,
    PreparedRestoreRequest,
    PreparedRestoreResult,
    ProductionBoundRun,
    ProductionGraph,
    ProductionPreflight,
    ProductionPreparedRun,
    ProductionRuntime,
    ProductionRuntimeContextProvider,
    ProductionRuntimeError,
    RestoreService,
    RuntimeContextRequest,
    RuntimeCreateRequest,
    RuntimeSession,
    RuntimeSessionFactory,
    SafePointCheckpointReceipt,
    StageAssembler,
    StageAssemblyRequest,
    StageCheckpointPortError,
    StageCheckpointPorts,
    TrainerStageAssembly,
    TransformExecution,
    TransformExecutor,
    TransformRequest,
)


_RECIPE_RESOLVED_FILE = "recipe.resolved.json"
_LAUNCH_RESOLVED_FILE = "launch.resolved.json"


class DefaultRuntimeSessionError(RuntimeError):
    """The requested recipe and observed Accelerate runtime are incompatible."""


class DefaultRuntimeContextProvider:
    """Supply no hidden component context beyond backend-owned typed values.

    The production backend already owns the model/dataset/reward artifact
    locations, runtime facts, training semantics, and trainer bind-once ports.
    Keeping this default provider empty is intentional: graph construction must
    not acquire reward models, open workers, or create process-local singletons.
    Deployments may replace this narrow seam for leaf factories, but cannot
    override backend-owned keys.
    """

    def context_for(self, request: RuntimeContextRequest) -> Mapping[str, Any]:
        if not isinstance(request, RuntimeContextRequest):
            raise TypeError("request must be RuntimeContextRequest")
        return {}


@dataclass(frozen=True, slots=True)
class _AcceleratorSettings:
    precision: ComputePrecision
    gradient_accumulation_steps: int
    seed: int
    recipe_id: str

    @classmethod
    def from_request(cls, request: RuntimeCreateRequest) -> _AcceleratorSettings:
        if not isinstance(request, RuntimeCreateRequest):
            raise TypeError("request must be RuntimeCreateRequest")
        if not isinstance(request.environment, EnvironmentPreflightResult):
            raise TypeError("request.environment must be EnvironmentPreflightResult")
        if not isinstance(request.training, TrainingSpec):
            raise TypeError("request.training must be TrainingSpec")
        if not isinstance(request.launch, LaunchSpec):
            raise TypeError("request.launch must be LaunchSpec")

        resolved = request.environment.materialized.resolved
        if request.training != resolved.training:
            raise DefaultRuntimeSessionError(
                "RuntimeCreateRequest training differs from the materialized recipe"
            )
        if request.training.gradient_accumulation_steps != 1:
            raise DefaultRuntimeSessionError(
                "default runtime factory requires gradient_accumulation_steps=1"
            )
        execution = resolved.execution_policy
        distribution_mode = execution.distribution_mode
        if distribution_mode is not DistributionMode.SINGLE:
            raise DefaultRuntimeSessionError(
                "default runtime factory supports only distribution_mode=single"
            )
        precision = execution.precision
        return cls(
            precision=precision,
            gradient_accumulation_steps=request.training.gradient_accumulation_steps,
            seed=request.training.seed,
            recipe_id=request.environment.materialized.recipe_id,
        )

    @property
    def accelerate_mixed_precision(self) -> str:
        if self.precision is ComputePrecision.FP32:
            return "no"
        return self.precision.value


@dataclass(slots=True)
class _AcceleratorCloser:
    accelerator: object
    _closed: bool = field(default=False, init=False, repr=False)

    def validate(self) -> None:
        self._end_training()

    def __call__(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._end_training()()

    def _end_training(self) -> Callable[[], object]:
        end_training = getattr(self.accelerator, "end_training", None)
        if not callable(end_training):
            raise DefaultRuntimeSessionError(
                "Accelerate runtime must provide callable end_training()"
            )
        return end_training


class DefaultRuntimeSessionFactory:
    """Build one single-process Accelerate session and its lazy resource owner."""

    def __init__(
        self,
        *,
        accelerator_factory: Callable[..., object] | None = None,
        set_seed_factory: Callable[..., object] | None = None,
        model_loader: Callable[..., object] | None = None,
        reward_resource_factory: RewardResourceFactory | None = None,
    ) -> None:
        if accelerator_factory is not None and not callable(accelerator_factory):
            raise TypeError("accelerator_factory must be callable or None")
        if set_seed_factory is not None and not callable(set_seed_factory):
            raise TypeError("set_seed_factory must be callable or None")
        if model_loader is not None and not callable(model_loader):
            raise TypeError("model_loader must be callable or None")
        if reward_resource_factory is not None and not isinstance(
            reward_resource_factory,
            RewardResourceFactory,
        ):
            raise TypeError(
                "reward_resource_factory must implement RewardResourceFactory or None"
            )
        self._accelerator_factory = (
            _create_accelerator if accelerator_factory is None else accelerator_factory
        )
        self._set_seed = _set_seed if set_seed_factory is None else set_seed_factory
        self._model_loader = model_loader
        self._reward_resource_factory = reward_resource_factory

    def create(self, request: RuntimeCreateRequest) -> RuntimeSession:
        settings = _AcceleratorSettings.from_request(request)
        accelerator = self._accelerator_factory(
            mixed_precision=settings.accelerate_mixed_precision,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
        )
        closer = _AcceleratorCloser(accelerator)
        resource_container: DefaultRuntimeResourceContainer | None = None
        try:
            closer.validate()
            runtime_facts = _single_process_runtime_facts(
                accelerator,
                expected_precision=settings.precision,
                expected_gradient_accumulation_steps=(
                    settings.gradient_accumulation_steps
                ),
            )
            self._set_seed(settings.seed, device_specific=True)
            reward_resource_factory = self._reward_resource_factory
            if reward_resource_factory is None:
                from visual_rl.runtime.reward_resources import (
                    DefaultRewardResourceFactory,
                )

                reward_resource_factory = DefaultRewardResourceFactory()
            resource_container = DefaultRuntimeResourceContainer(
                reward_resource_factory
            )
            return RuntimeSession(
                accelerator=accelerator,
                runtime_facts=runtime_facts,
                peer_recipe_ids=(settings.recipe_id,),
                model_loader=self._model_loader,
                resource_container=resource_container,
                closer=closer,
            )
        except BaseException as primary:
            _rollback(resource_container, closer, primary)
            raise


def _single_process_runtime_facts(
    accelerator: object,
    *,
    expected_precision: ComputePrecision,
    expected_gradient_accumulation_steps: int,
) -> RuntimeFacts:
    world_size = _integer_fact(accelerator, "num_processes")
    rank = _integer_fact(accelerator, "process_index")
    local_rank = _integer_fact(accelerator, "local_process_index")
    distributed_type = _text_fact(accelerator, "distributed_type").lower()
    if distributed_type != "no" or (world_size, rank, local_rank) != (1, 0, 0):
        raise DefaultRuntimeSessionError(
            "Accelerate initialized a distributed runtime for a single-process recipe: "
            f"distributed_type={distributed_type!r}, world_size={world_size}, "
            f"rank={rank}, local_rank={local_rank}"
        )

    precision = _observed_precision(accelerator)
    if precision is not expected_precision:
        raise DefaultRuntimeSessionError(
            "Accelerate mixed precision differs from the recipe: "
            f"expected={expected_precision.value!r}, observed={precision.value!r}"
        )
    gradient_accumulation_steps = _integer_fact(
        accelerator,
        "gradient_accumulation_steps",
    )
    if gradient_accumulation_steps != expected_gradient_accumulation_steps:
        raise DefaultRuntimeSessionError(
            "Accelerate gradient accumulation differs from TrainingSpec: "
            f"expected={expected_gradient_accumulation_steps}, "
            f"observed={gradient_accumulation_steps}"
        )
    raw_device = _required_fact(accelerator, "device")
    if raw_device is None:
        raise DefaultRuntimeSessionError("Accelerate returned no device")
    device = _materialized_runtime_device(raw_device)
    return RuntimeFacts(
        distribution_mode=DistributionMode.SINGLE.value,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        precision=precision.value,
        backend=None,
    )


def _materialized_runtime_device(raw_device: object) -> str:
    """Resolve an implicit accelerator device to the device tensors report.

    ``torch.device("cuda")`` denotes the current CUDA device but is not equal
    to ``torch.device("cuda:0")`` returned by a tensor allocated there. A
    zero-element allocation asks the active backend for that physical device
    without guessing from ranks or backend-specific global state.
    """

    import torch

    try:
        requested = torch.device(raw_device)
    except (TypeError, RuntimeError):
        raise DefaultRuntimeSessionError(
            "Accelerate returned a device that is not torch-compatible"
        ) from None
    try:
        observed = torch.empty(0, device=requested).device
    except (AssertionError, RuntimeError, TypeError) as exc:
        raise DefaultRuntimeSessionError(
            "Accelerate device could not materialize a runtime tensor"
        ) from exc
    if observed.type != requested.type:
        raise DefaultRuntimeSessionError(
            "Accelerate device type changed while materializing runtime facts"
        )
    return str(observed)


def _observed_precision(accelerator: object) -> ComputePrecision:
    raw = _required_fact(accelerator, "mixed_precision")
    if not isinstance(raw, str):
        raise DefaultRuntimeSessionError("Accelerate mixed_precision must be a string")
    normalized = raw.strip().lower()
    if normalized == "no":
        return ComputePrecision.FP32
    try:
        return ComputePrecision(normalized)
    except ValueError as exc:
        raise DefaultRuntimeSessionError(
            f"Accelerate returned unsupported mixed precision {normalized!r}"
        ) from exc


def _integer_fact(accelerator: object, name: str) -> int:
    value = _required_fact(accelerator, name)
    if type(value) is not int:
        raise DefaultRuntimeSessionError(
            f"Accelerate runtime fact {name!r} must be an integer"
        )
    return value


def _text_fact(accelerator: object, name: str) -> str:
    raw = _required_fact(accelerator, name)
    value = getattr(raw, "value", raw)
    text = str(value).strip()
    if not text:
        raise DefaultRuntimeSessionError(
            f"Accelerate runtime fact {name!r} must be non-empty"
        )
    return text


def _required_fact(accelerator: object, name: str) -> Any:
    sentinel = object()
    value = getattr(accelerator, name, sentinel)
    if value is sentinel:
        raise DefaultRuntimeSessionError(
            f"Accelerate runtime is missing required fact {name!r}"
        )
    return value


def _rollback(
    resource_container: DefaultRuntimeResourceContainer | None,
    closer: _AcceleratorCloser,
    primary: BaseException,
) -> None:
    cleanups = (
        None if resource_container is None else resource_container.close,
        closer,
    )
    for cleanup in cleanups:
        if cleanup is None:
            continue
        try:
            cleanup()
        except BaseException as cleanup_error:  # noqa: BLE001
            if hasattr(primary, "add_note"):
                primary.add_note(
                    "runtime session rollback failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )


def _create_accelerator(**kwargs: object) -> object:
    try:
        from accelerate import Accelerator
    except ImportError as exc:  # pragma: no cover - exercised without train extras
        raise DefaultRuntimeSessionError(
            "default runtime requires the optional 'accelerate' dependency; "
            "install visual-rl[train]"
        ) from exc
    return Accelerator(**kwargs)


def _set_seed(seed: int, *, device_specific: bool) -> None:
    try:
        from accelerate.utils import set_seed
    except ImportError as exc:  # pragma: no cover - inconsistent optional install
        raise DefaultRuntimeSessionError(
            "default runtime requires accelerate.utils.set_seed"
        ) from exc
    set_seed(seed, device_specific=device_specific)


class ProductionRuntimeLifecycleBackend:
    """The first concrete schema-v2 production lifecycle backend."""

    def __init__(
        self,
        *,
        catalog: Catalog | None = None,
        artifact_resolver: object,
        runtime_factory: RuntimeSessionFactory,
        runtime_context_provider: ProductionRuntimeContextProvider,
        stage_assembler: StageAssembler,
        component_binder: ComponentRuntimeBinder,
        checkpoint_sink: CheckpointSink,
        restore_service: RestoreService | None = None,
        transform_executor: TransformExecutor | None = None,
        declaration_resolver: DeclarationResolver | None = None,
        algorithm_declaration_resolver: AlgorithmDeclarationResolver | None = None,
        dynamics_projection_registry: DynamicsProjectionRegistry | None = None,
        runtime_component_loader: RuntimeComponentLoader | None = None,
    ) -> None:
        self.catalog = default_catalog() if catalog is None else catalog
        if not isinstance(self.catalog, Catalog):
            raise TypeError("catalog must be a Catalog or None")
        self.declaration_resolver = (
            DeclarationResolver()
            if declaration_resolver is None
            else declaration_resolver
        )
        if not isinstance(self.declaration_resolver, DeclarationResolver):
            raise TypeError(
                "declaration_resolver must be a DeclarationResolver or None"
            )
        self.algorithm_declaration_resolver = (
            AlgorithmDeclarationResolver()
            if algorithm_declaration_resolver is None
            else algorithm_declaration_resolver
        )
        if not isinstance(
            self.algorithm_declaration_resolver,
            AlgorithmDeclarationResolver,
        ):
            raise TypeError(
                "algorithm_declaration_resolver must be an "
                "AlgorithmDeclarationResolver or None"
            )
        self.dynamics_projection_registry = (
            default_dynamics_projection_registry()
            if dynamics_projection_registry is None
            else dynamics_projection_registry
        )
        if not isinstance(
            self.dynamics_projection_registry,
            DynamicsProjectionRegistry,
        ):
            raise TypeError(
                "dynamics_projection_registry must be a "
                "DynamicsProjectionRegistry or None"
            )
        if runtime_component_loader is not None and not isinstance(
            runtime_component_loader,
            RuntimeComponentLoader,
        ):
            raise TypeError(
                "runtime_component_loader must be RuntimeComponentLoader or None"
            )
        _method(artifact_resolver, "resolve_artifact_identities")
        _method(runtime_factory, "create")
        _method(runtime_context_provider, "context_for")
        _method(stage_assembler, "assemble")
        _method(component_binder, "bind")
        _method(checkpoint_sink, "checkpoint")
        if restore_service is not None:
            _method(restore_service, "restore_prepared")
            _method(restore_service, "restore_bound")
        if transform_executor is not None:
            _method(transform_executor, "execute")
        self.artifact_resolver = artifact_resolver
        self.runtime_factory = runtime_factory
        self.runtime_context_provider = runtime_context_provider
        self.stage_assembler = stage_assembler
        self.component_binder = component_binder
        self.checkpoint_sink = checkpoint_sink
        self.restore_service = restore_service
        self.transform_executor = transform_executor
        self.runtime_component_loader = runtime_component_loader

    def compile(self, config_path: Path) -> CompiledProductionRun:
        source = load_source_recipe(config_path)
        bootstrap = bootstrap_recipe_v2(source)
        launch = bootstrap.require_launch()
        static = run_static_preflight(
            source,
            self.catalog,
            declaration_resolver=self.declaration_resolver,
            algorithm_declaration_resolver=(self.algorithm_declaration_resolver),
            dynamics_projection_registry=self.dynamics_projection_registry,
        )
        if bootstrap.recipe_id != static.resolved.definition_id:
            raise ProductionRuntimeError("bootstrap and compiled recipe ids differ")
        return CompiledProductionRun(
            source=source,
            static=static,
            training=static.resolved.training,
            launch=launch,
        )

    def preflight(self, compiled: CompiledProductionRun) -> ProductionPreflight:
        if not isinstance(compiled, CompiledProductionRun):
            raise TypeError("compiled must be CompiledProductionRun")
        environment = run_environment_preflight(
            compiled.static,
            self.artifact_resolver,  # type: ignore[arg-type]
            artifact_locations=compiled.launch.artifacts,
        )
        result = ProductionPreflight(compiled=compiled, environment=environment)
        _validate_resume_checkpoint_recipe(result)
        _publish_recipe_manifest(result)
        return result

    def create_runtime(self, preflight: ProductionPreflight) -> ProductionRuntime:
        if not isinstance(preflight, ProductionPreflight):
            raise TypeError("preflight must be ProductionPreflight")
        session = self.runtime_factory.create(
            RuntimeCreateRequest(
                environment=preflight.environment,
                training=preflight.compiled.training,
                launch=preflight.compiled.launch,
            )
        )
        if not isinstance(session, RuntimeSession):
            raise TypeError("runtime factory must return RuntimeSession")
        try:
            binding = bind_runtime(
                RuntimeBindInput(
                    environment=preflight.environment,
                    runtime_facts=session.runtime_facts,
                    peer_recipe_ids=session.peer_recipe_ids,
                    launch_audit=build_launch_runtime_audit(preflight.compiled.launch),
                )
            )
            _publish_launch_manifest(preflight, binding)
            return ProductionRuntime(
                preflight=preflight,
                session=session,
                launch_binding=binding,
            )
        except BaseException as primary:
            try:
                session.close()
            except BaseException as cleanup_error:  # noqa: BLE001
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "RuntimeSession cleanup failed after runtime-bind/manifest "
                        f"failure: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise

    def construct_graph(
        self,
        preflight: ProductionPreflight,
        runtime: ProductionRuntime,
    ) -> ProductionGraph:
        ports = _TrainerPorts.create()

        def contexts(
            slot: str,
            declaration: ResolvedComponentDeclaration,
            loaded: Mapping[str, object],
        ) -> Mapping[str, Any]:
            base = self._base_runtime_context(
                slot,
                declaration,
                preflight=preflight,
                runtime=runtime,
                ports=ports,
            )
            supplied = self.runtime_context_provider.context_for(
                RuntimeContextRequest(
                    slot=slot,
                    declaration=declaration,
                    loaded_components=loaded,
                    preflight=preflight,
                    runtime=runtime,
                )
            )
            if not isinstance(supplied, Mapping):
                raise TypeError("runtime context provider must return a mapping")
            overlap = sorted(set(base).intersection(supplied))
            if overlap:
                raise ProductionRuntimeError(
                    "runtime context provider cannot override backend-owned keys: "
                    f"{overlap}"
                )
            return {**base, **supplied}

        graph = load_component_graph(
            preflight.environment,
            runtime.launch_binding,
            runtime_context_factory=contexts,
            loader=self.runtime_component_loader,
        )
        try:
            result = ProductionGraph(components=graph, trainer_ports=ports)
            algorithm = result.algorithm_module
            resolved = preflight.environment.materialized.resolved.algorithm
            if algorithm.config is not resolved.config:
                raise ProductionRuntimeError(
                    "loaded AlgorithmModule does not retain its resolved config"
                )
            if algorithm.blueprint != resolved.blueprint:
                raise ProductionRuntimeError(
                    "loaded AlgorithmModule blueprint differs from the recipe"
                )
            if algorithm.requirements != resolved.requirements:
                raise ProductionRuntimeError(
                    "loaded AlgorithmModule requirements differ from the recipe"
                )
            return result
        except BaseException as primary:
            try:
                graph.close()
            except BaseException as cleanup_error:  # noqa: BLE001
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "internal component graph cleanup failed after algorithm "
                        f"construction: {type(cleanup_error).__name__}: "
                        f"{cleanup_error}"
                    )
            raise

    def prepare(
        self,
        graph: ProductionGraph,
        runtime: ProductionRuntime,
    ) -> ProductionPreparedRun:
        _preacquire_reward_resources(runtime)
        model = graph.components.component("model")
        if not isinstance(model, ModelAdapter):
            raise TypeError("production model component must be ModelAdapter")
        preflight = runtime.preflight
        training = preflight.compiled.training
        offload_device = runtime.session.runtime_facts.extra.get(
            "offload_device",
            "cpu",
        )
        manager = ComponentManager(
            model,
            execution_device=runtime.session.runtime_facts.device,
            offload_device=offload_device,
        )
        try:
            manager.load()
            manager.configure()
            manager.apply_parameter_dtype_policy(
                ParameterDTypePolicy(trainable_parameter_dtype="float32")
            )
            manager.bind_model_execution_numerics(
                resolve_model_execution_numerics(
                    manager,
                    runtime.session.runtime_facts,
                )
            )
            adamw = training.adamw
            lr_schedule = training.lr_schedule
            optimizer_spec = OptimizerExecutionSpec(
                learning_rate=adamw.learning_rate,
                beta1=adamw.beta1,
                beta2=adamw.beta2,
                epsilon=adamw.epsilon,
                weight_decay=adamw.weight_decay,
                amsgrad=adamw.amsgrad,
                schedule_kind=lr_schedule.kind,
                warmup_steps=lr_schedule.warmup_steps,
                min_lr_ratio=lr_schedule.min_lr_ratio,
                max_optimizer_steps=training.max_optimizer_steps,
            )
            optimizer = build_adamw(
                manager.parameter_state.parameters(),
                optimizer_spec,
            )
            scheduler = build_lr_scheduler(optimizer, optimizer_spec)
            handle = manager.prepare(
                accelerator=runtime.session.accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            return ProductionPreparedRun(
                manager=manager,
                handle=handle,
                optimizer=handle.optimizer,
                lr_scheduler=handle.scheduler,
                training=training,
            )
        except BaseException:
            manager.close()
            raise

    def restore_prepared(
        self,
        preflight: ProductionPreflight,
        runtime: ProductionRuntime,
        graph: ProductionGraph,
        prepared: ProductionPreparedRun,
    ) -> PreparedRestoreResult | None:
        resume = preflight.compiled.launch.resume_from
        if resume is None:
            prepared.start_optimizer_step = 0
            _validate_checkpoint_cadence_support(
                preflight,
                prepared,
                self.checkpoint_sink,
            )
            return None
        if self.restore_service is None:
            raise ProductionRuntimeError(
                "launch.resume_from requires an injected restore service"
            )
        result = self.restore_service.restore_prepared(
            PreparedRestoreRequest(
                checkpoint_path=resume,
                preflight=preflight,
                runtime=runtime,
                graph=graph,
                prepared=prepared,
            )
        )
        if not isinstance(result, PreparedRestoreResult):
            raise TypeError(
                "restore service must return PreparedRestoreResult evidence"
            )
        if result.checkpoint_path != resume:
            raise ProductionRuntimeError(
                "prepared restore checkpoint path differs from LaunchSpec"
            )
        start = result.next_optimizer_step
        if (
            type(start) is not int
            or not 0 < start <= prepared.training.max_optimizer_steps
        ):
            raise ProductionRuntimeError(
                "prepared restore returned an invalid next optimizer step"
            )
        prepared.start_optimizer_step = start
        _validate_checkpoint_cadence_support(
            preflight,
            prepared,
            self.checkpoint_sink,
        )
        return result

    def transform(
        self,
        preflight: ProductionPreflight,
        runtime: ProductionRuntime,
        graph: ProductionGraph,
        prepared: ProductionPreparedRun,
    ) -> TransformExecution:
        plan = (
            preflight.environment.materialized.resolved.execution_policy.transform_plan
        )
        return execute_runtime_transforms(
            TransformRequest(
                plan=plan,
                preflight=preflight,
                runtime=runtime,
                graph=graph,
                prepared=prepared,
            ),
            self.transform_executor,
        )

    def bind(
        self,
        preflight: ProductionPreflight,
        runtime: ProductionRuntime,
        graph: ProductionGraph,
        prepared: ProductionPreparedRun,
        transformed: TransformExecution,
    ) -> ProductionBoundRun:
        evidence = self.component_binder.bind(
            ComponentBindRequest(
                preflight=preflight,
                runtime=runtime,
                graph=graph,
                prepared=prepared,
                transforms=transformed,
            )
        )
        if not isinstance(evidence, ComponentRuntimeEvidence):
            raise TypeError("component binder must return ComponentRuntimeEvidence")
        container = runtime.session.resource_container
        if container is None:
            if evidence.bound_reward_resource_ids:
                raise ProductionRuntimeError(
                    "G3 evidence contains physical reward bindings without a "
                    "session resource container"
                )
        else:
            try:
                observed_reward_ids = container.bound_reward_resource_ids
            except RuntimeError as exc:
                raise ProductionRuntimeError(
                    "G3 binder did not acquire the session reward resources"
                ) from exc
            if evidence.bound_reward_resource_ids != observed_reward_ids:
                raise ProductionRuntimeError(
                    "G3 physical reward evidence differs from the session container"
                )
        declared_model = graph.components.binding("model").declared_contract
        if evidence.model_runtime_contract.artifact.declared != declared_model:
            raise ProductionRuntimeError(
                "G3 model runtime contract differs from the resolved descriptor"
            )
        validate_algorithm_runtime_contracts(
            graph.components,
            evidence.runtime_bound_contracts,
        )
        world_size = runtime.session.runtime_facts.world_size
        if world_size > 1 and not evidence.peer_bound_contract_ids:
            raise ProductionRuntimeError(
                "distributed G3 bind requires peer bound contract ids"
            )
        graph_binding = bind_runtime_graph(
            RuntimeGraphBindInput(
                environment=preflight.environment,
                launch=runtime.launch_binding,
                runtime_bound_contracts=evidence.runtime_bound_contracts,
                trainable_topology_id=(
                    prepared.manager.parameter_state.topology.identity
                ),
                prepared_component_names=tuple(sorted(prepared.handle.component_names)),
                execution_transform_plan_id=transformed.plan_id,
                resource_plan_id=prepared.manager.resource_plan.plan_id,
                verified_fields=evidence.verified_fields,
                bound_reward_resource_ids=evidence.bound_reward_resource_ids,
                peer_bound_contract_ids=evidence.peer_bound_contract_ids,
            )
        )
        prepared.manager.bind_runtime(evidence.model_runtime_contract)
        dynamics_factory = bind_per_rollout_dynamics_factory(
            graph.components,
            prepared.manager,
        )
        assembly: TrainerStageAssembly | None = None
        bound_algorithm: BoundAlgorithm | None = None
        try:
            declared_dynamics = graph.components.binding("dynamics").declared_contract
            declared_trainer = graph.components.binding("trainer").declared_contract
            if declared_model.model is None:
                raise ProductionRuntimeError(
                    "resolved model descriptor has no ModelContract"
                )
            if declared_dynamics.dynamics is None:
                raise ProductionRuntimeError(
                    "resolved dynamics descriptor has no DynamicsContract"
                )
            if declared_trainer.trainer is None:
                raise ProductionRuntimeError(
                    "resolved trainer descriptor has no TrainerContract"
                )
            policy_runtime = DefaultPolicyRuntimePort(
                _adapter=prepared.manager.adapter,
                _manager=prepared.manager,
                _prepared_handle=prepared.handle,
                capabilities=BoundPolicyCapabilities.from_contracts(
                    declared_model.model,
                    dynamics=declared_dynamics.dynamics,
                    trainer=declared_trainer.trainer,
                ),
                algorithm_requirements=graph.algorithm_module.requirements,
                runtime_capabilities=evidence.model_runtime_contract,
            )
            resolved = preflight.environment.materialized.resolved
            materializer = CanonicalAlgorithmMaterializer(
                materialize_algorithm_runtime_components(
                    resolved.algorithm_spec,
                    graph.components,
                )
            )
            bound_algorithm = graph.algorithm_module.materialize(
                policy_runtime,
                policy_runtime.binding,
                resolved.algorithm_spec,
                materializer,
                execution_policy=resolved.execution_policy.to_receipt(),
            )
            assembly = self.stage_assembler.assemble(
                StageAssemblyRequest(
                    preflight=preflight,
                    runtime=runtime,
                    graph=graph,
                    prepared=prepared,
                    transforms=transformed,
                    evidence=evidence,
                    graph_binding=graph_binding,
                    dynamics_factory=dynamics_factory,
                    policy_runtime=policy_runtime,
                )
            )
            if not isinstance(assembly, TrainerStageAssembly):
                raise TypeError("stage assembler must return TrainerStageAssembly")
            _bind_trainer_ports(graph.trainer_ports, assembly)
            if bound_algorithm.binding != policy_runtime.binding:
                raise ProductionRuntimeError(
                    "AlgorithmModule returned a different model/algorithm binding"
                )
        except BaseException as primary:
            if bound_algorithm is not None:
                try:
                    bound_algorithm.close()
                except BaseException as cleanup_error:  # noqa: BLE001
                    if hasattr(primary, "add_note"):
                        primary.add_note(
                            "AlgorithmModule rollback close failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
            if assembly is not None:
                try:
                    assembly.close()
                except BaseException as cleanup_error:  # noqa: BLE001
                    if hasattr(primary, "add_note"):
                        primary.add_note(
                            "stage assembly rollback close failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
            raise
        assert assembly is not None
        assert bound_algorithm is not None
        return ProductionBoundRun(
            preflight=preflight,
            runtime=runtime,
            graph=graph,
            prepared=prepared,
            transforms=transformed,
            evidence=evidence,
            graph_binding=graph_binding,
            assembly=assembly,
            algorithm=bound_algorithm,
        )

    def restore_bound(
        self,
        preflight: ProductionPreflight,
        runtime: ProductionRuntime,
        graph: ProductionGraph,
        prepared: ProductionPreparedRun,
        bound: ProductionBoundRun,
        prepared_restore: PreparedRestoreResult | None,
    ) -> RestoreBoundOutcome[RunResult] | None:
        resume = preflight.compiled.launch.resume_from
        if resume is None:
            if prepared_restore is not None:
                raise ProductionRuntimeError(
                    "fresh run received an unexpected prepared restore receipt"
                )
            if prepared.start_optimizer_step != 0:
                raise ProductionRuntimeError(
                    "fresh run start_optimizer_step changed before bound restore"
                )
            return None
        if self.restore_service is None:
            raise ProductionRuntimeError(
                "launch.resume_from requires an injected restore service"
            )
        if not isinstance(prepared_restore, PreparedRestoreResult):
            raise TypeError(
                "resume bound restore requires PreparedRestoreResult evidence"
            )
        if prepared_restore.checkpoint_path != resume:
            raise ProductionRuntimeError(
                "prepared restore checkpoint path differs from LaunchSpec"
            )
        result = self.restore_service.restore_bound(
            BoundRestoreRequest(
                checkpoint_path=resume,
                preflight=preflight,
                runtime=runtime,
                graph=graph,
                prepared=prepared,
                bound=bound,
                prepared_restore=prepared_restore,
            )
        )
        if not isinstance(result, BoundRestoreResult):
            raise TypeError("restore service must return BoundRestoreResult evidence")
        if result.checkpoint_path != resume:
            raise ProductionRuntimeError(
                "bound restore checkpoint path differs from LaunchSpec"
            )
        if result.next_optimizer_step != prepared_restore.next_optimizer_step:
            raise ProductionRuntimeError(
                "prepared and bound restore optimizer steps disagree"
            )
        start = result.next_optimizer_step
        stop = prepared.training.max_optimizer_steps
        if not 0 < start <= stop:
            raise ProductionRuntimeError(
                "bound restore returned an invalid next optimizer step"
            )
        prepared.start_optimizer_step = start
        completed = result.completed_result
        if completed is None:
            if start >= stop:
                raise ProductionRuntimeError(
                    "completed checkpoint must provide its final RunResult"
                )
            return RestoreBoundOutcome()
        if start != stop:
            raise ProductionRuntimeError(
                "non-terminal checkpoint cannot provide a completed RunResult"
            )
        self._validate_run_result(
            bound,
            completed,
            committed_steps=stop,
            completed_restore=True,
        )
        return RestoreBoundOutcome(completed_result=completed)

    def prepare_run(self, bound: ProductionBoundRun) -> None:
        container = bound.runtime.session.resource_container
        if container is not None:
            container.activate()
        facts = bound.runtime.session.runtime_facts
        bound.algorithm.prepare_run(
            PrepareRunContext(
                run_id=bound.runtime.launch_binding.launch_id,
                recipe_id=bound.preflight.environment.materialized.recipe_id,
                start_optimizer_step=bound.prepared.start_optimizer_step,
                runtime_facts=(
                    ("bound_contract_id", bound.graph_binding.bound_contract_id),
                    ("device", facts.device),
                    ("distribution_mode", facts.distribution_mode),
                    ("precision", facts.precision),
                    ("rank", str(facts.rank)),
                    ("world_size", str(facts.world_size)),
                ),
            )
        )

    def run(self, bound: ProductionBoundRun) -> PendingRunSummary:
        start = bound.prepared.start_optimizer_step
        stop = bound.prepared.training.max_optimizer_steps
        if start >= stop:
            raise ProductionRuntimeError("training has no optimizer updates remaining")
        final_iteration: IterationResult[object] | None = None
        cadence = bound.preflight.compiled.launch.checkpoint_every_optimizer_steps
        for optimizer_step in range(start, stop):
            algorithm_result = bound.algorithm.run_iteration(optimizer_step)
            if not isinstance(algorithm_result, AlgorithmStepResult):
                raise TypeError("AlgorithmModule must return AlgorithmStepResult")
            if (
                algorithm_result.algorithm_binding_id
                != bound.algorithm.binding.binding_id
            ):
                raise ProductionRuntimeError(
                    "AlgorithmModule returned the wrong binding identity"
                )
            observed = algorithm_result.iteration
            if not isinstance(observed, IterationResult):
                raise TypeError(
                    "AlgorithmModule iteration must contain IterationResult"
                )
            if observed.optimizer_step != optimizer_step:
                raise ProductionRuntimeError(
                    "trainer returned the wrong optimizer step"
                )
            committed_steps = optimizer_step + 1
            if committed_steps < stop and committed_steps % cadence == 0:
                self._checkpoint_safe_point(
                    bound,
                    PendingRunSummary(
                        recipe_id=(bound.preflight.environment.materialized.recipe_id),
                        launch_id=bound.runtime.launch_binding.launch_id,
                        bound_contract_id=bound.graph_binding.bound_contract_id,
                        start_optimizer_step=start,
                        committed_steps=committed_steps,
                        update_count=committed_steps - start,
                        last_iteration=observed,
                    ),
                    cadence,
                )
            if committed_steps == stop:
                final_iteration = observed
            else:
                # IterationResult may own a complete rollout trajectory.  Drop
                # every backend-local owner before the next run_iteration()
                # call so two trajectories cannot overlap in device memory.
                del observed
                del algorithm_result
        assert final_iteration is not None
        return PendingRunSummary(
            recipe_id=bound.preflight.environment.materialized.recipe_id,
            launch_id=bound.runtime.launch_binding.launch_id,
            bound_contract_id=bound.graph_binding.bound_contract_id,
            start_optimizer_step=start,
            committed_steps=stop,
            update_count=stop - start,
            last_iteration=final_iteration,
        )

    def _checkpoint_safe_point(
        self,
        bound: ProductionBoundRun,
        summary: PendingRunSummary,
        cadence: int,
    ) -> SafePointCheckpointReceipt:
        checkpoint = getattr(self.checkpoint_sink, "checkpoint_safe_point", None)
        if not callable(checkpoint):
            raise ProductionRuntimeError(
                "checkpoint sink cannot honor intermediate checkpoint cadence"
            )
        receipt = checkpoint(
            CheckpointRequest(
                bound=bound,
                summary=summary,
                cadence=cadence,
            )
        )
        if not isinstance(receipt, SafePointCheckpointReceipt):
            raise TypeError(
                "checkpoint_safe_point must return SafePointCheckpointReceipt"
            )
        expected = (
            bound.preflight.compiled.launch.output_dir
            / "checkpoints"
            / f"step-{summary.committed_steps}"
        )
        if (
            receipt.checkpoint_path != expected
            or receipt.committed_steps != summary.committed_steps
        ):
            raise ProductionRuntimeError(
                "safe-point checkpoint receipt differs from the committed step"
            )
        return receipt

    def checkpoint(
        self,
        bound: ProductionBoundRun,
        summary: PendingRunSummary,
    ) -> RunResult:
        result = self.checkpoint_sink.checkpoint(
            CheckpointRequest(
                bound=bound,
                summary=summary,
                cadence=(
                    bound.preflight.compiled.launch.checkpoint_every_optimizer_steps
                ),
            )
        )
        if not isinstance(result, RunResult):
            raise TypeError("checkpoint sink must return runtime.types.RunResult")
        self._validate_run_result(
            bound,
            result,
            committed_steps=summary.committed_steps,
        )
        return result

    @staticmethod
    def _validate_run_result(
        bound: ProductionBoundRun,
        result: RunResult,
        *,
        committed_steps: int,
        completed_restore: bool = False,
    ) -> None:
        if not isinstance(result, RunResult):
            raise TypeError("result must be runtime.types.RunResult")
        if type(completed_restore) is not bool:
            raise TypeError("completed_restore must be a bool")
        launch = bound.preflight.compiled.launch
        if result.run_id != bound.runtime.launch_binding.launch_id:
            raise ProductionRuntimeError(
                "RunResult run_id differs from the bound launch"
            )
        if result.committed_steps != committed_steps:
            raise ProductionRuntimeError(
                "RunResult committed_steps differs from execution/restore"
            )
        if not completed_restore:
            if result.output_dir != launch.output_dir:
                raise ProductionRuntimeError(
                    "RunResult output_dir differs from LaunchSpec"
                )
            return

        resume_from = launch.resume_from
        if resume_from is None:
            raise ProductionRuntimeError(
                "completed restore requires an explicit LaunchSpec resume_from"
            )
        try:
            normalized_resume = resume_from.resolve(strict=True)
        except OSError as exc:
            raise ProductionRuntimeError(
                "completed restore resume_from cannot be resolved"
            ) from exc
        if result.authoritative_checkpoint != normalized_resume:
            raise ProductionRuntimeError(
                "completed restore RunResult checkpoint differs from resume_from"
            )
        if (
            normalized_resume.name != f"step-{committed_steps}"
            or normalized_resume.parent.name != "checkpoints"
        ):
            raise ProductionRuntimeError(
                "completed restore checkpoint is outside the canonical historical run root"
            )
        historical_output_dir = normalized_resume.parent.parent
        if result.output_dir != historical_output_dir:
            raise ProductionRuntimeError(
                "completed restore RunResult output_dir differs from the historical run root"
            )

    def _base_runtime_context(
        self,
        slot: str,
        declaration: ResolvedComponentDeclaration,
        *,
        preflight: ProductionPreflight,
        runtime: ProductionRuntime,
        ports: _TrainerPorts,
    ) -> dict[str, Any]:
        compiled = preflight.compiled
        facts = runtime.session.runtime_facts
        context: dict[str, Any] = {}
        if declaration.kind == "model":
            artifact_ref = getattr(declaration.config, "artifact_ref", None)
            if not isinstance(artifact_ref, str) or not artifact_ref:
                raise ProductionRuntimeError("model artifact_ref must be non-empty")
            context.update(
                {
                    "precision": facts.precision,
                    "model_artifacts": {artifact_ref: compiled.launch.artifacts.model},
                    "model_loader": runtime.session.model_loader,
                }
            )
        elif declaration.kind == "trainer":
            context.update(
                {
                    "prelude": ports.prelude,
                    "rollout": ports.rollout,
                    "reward": ports.reward,
                    "advantage": ports.advantage,
                    "credit": ports.credit,
                    "optimize": ports.optimize,
                }
            )
        elif declaration.kind == "rollout":
            resolved = preflight.environment.materialized.resolved
            context.update(
                {
                    "execution_policy": resolved.execution_policy.to_receipt(),
                    "expected_execution_policy_id": (
                        resolved.algorithm_spec.execution_policy_id
                    ),
                }
            )
        elif declaration.kind == "credit":
            # The algorithm blueprint/spec is the sole owner of beta.  The
            # credit declaration owns clipping/normalization parameters, while
            # its runtime strategy needs this already-validated algorithm
            # scalar to decide whether reference-policy statistics enter the
            # objective.  Supplying it here keeps credit and recompute on the
            # same canonical AlgorithmMaterializationSpec instead of falling
            # back to the component's beta=0 runtime default.
            context["beta"] = (
                preflight.environment.materialized.resolved.algorithm_spec.beta
            )
        if slot != "algorithm" and slot == "model" and declaration.kind != "model":
            raise ProductionRuntimeError("model slot has a non-model declaration")
        return context


def _validate_resume_checkpoint_recipe(preflight: ProductionPreflight) -> None:
    """Validate durable checkpoint metadata before runtime/model construction."""

    resume = preflight.compiled.launch.resume_from
    if resume is None:
        return
    if not resume.parent.is_dir() or resume.parent.is_symlink():
        raise ResumeError(
            "resume checkpoint parent must be a real directory",
            path=str(resume),
        )
    try:
        inspection = AtomicCheckpointManager(resume.parent).inspect_complete(resume)
    except (OSError, TypeError, ValueError) as exc:
        raise ResumeError(
            "resume checkpoint metadata is incomplete or invalid",
            path=str(resume),
        ) from exc
    current_recipe_id = preflight.environment.materialized.recipe_id
    if inspection.contract.recipe_id != current_recipe_id:
        raise ResumeError(
            "resume checkpoint recipe_id differs from the materialized recipe",
            path=str(resume),
        )


def _publish_recipe_manifest(preflight: ProductionPreflight) -> Path:
    launch = preflight.compiled.launch
    destination = launch.output_dir / _RECIPE_RESOLVED_FILE
    if launch.resume_from is not None and (
        destination.exists() or destination.is_symlink()
    ):
        return assert_recipe_manifest_resume_compatible(
            destination,
            preflight.environment.materialized,
            preflight.environment.component_artifact_bindings,
        )
    return write_recipe_manifest(
        destination,
        preflight.environment.materialized,
        preflight.environment.component_artifact_bindings,
    )


def _publish_launch_manifest(
    preflight: ProductionPreflight,
    binding: RuntimeBindResult,
) -> Path:
    launch = preflight.compiled.launch
    destination = launch.output_dir / _LAUNCH_RESOLVED_FILE
    if launch.resume_from is not None and (
        destination.exists() or destination.is_symlink()
    ):
        return assert_launch_manifest_resume_compatible(
            destination,
            binding,
            launch,
        )
    return write_launch_manifest(destination, binding, launch)


def _bind_trainer_ports(
    ports: _TrainerPorts,
    assembly: TrainerStageAssembly,
) -> None:
    ports.prelude.bind(assembly.prelude)
    ports.rollout.bind(assembly.rollout)
    ports.reward.bind(assembly.reward)
    ports.advantage.bind(assembly.advantage)
    ports.credit.bind(assembly.credit)
    ports.optimize.bind(assembly.optimize)


def _validate_checkpoint_cadence_support(
    preflight: ProductionPreflight,
    prepared: ProductionPreparedRun,
    checkpoint_sink: object,
) -> None:
    cadence = preflight.compiled.launch.checkpoint_every_optimizer_steps
    start = prepared.start_optimizer_step
    stop = prepared.training.max_optimizer_steps
    next_boundary = ((start // cadence) + 1) * cadence
    if next_boundary < stop and not callable(
        getattr(checkpoint_sink, "checkpoint_safe_point", None)
    ):
        raise ProductionRuntimeError(
            "checkpoint sink cannot honor intermediate checkpoint cadence"
        )


def _preacquire_reward_resources(runtime: ProductionRuntime) -> None:
    """Attest every reward endpoint before any model weight is loaded.

    The default runtime owns one acquire-once container.  Injected runtimes
    without that container retain their own binder-defined resource lifecycle.
    """

    if not isinstance(runtime, ProductionRuntime):
        raise TypeError("runtime must be ProductionRuntime")
    container = runtime.session.resource_container
    if container is None:
        return
    if not isinstance(container, DefaultRuntimeResourceContainer):
        raise TypeError(
            "runtime resource_container must be DefaultRuntimeResourceContainer"
        )
    preflight = runtime.preflight
    plan, requests = compile_reward_resource_acquisition(
        preflight.environment.materialized,
        preflight.compiled.launch,
        runtime.session.runtime_facts,
    )
    if container.state is RewardResourceState.DECLARED:
        container.acquire(plan, requests)
    elif container.state is RewardResourceState.ACQUIRED:
        try:
            container.assert_acquisition_requests_match(plan, requests)
        except RuntimeResourceAcquisitionError as exc:
            raise ProductionRuntimeError(
                "pre-acquired reward resources differ from production inputs"
            ) from exc
    else:
        raise ProductionRuntimeError(
            "model preparation requires DECLARED or ACQUIRED reward resources; "
            f"found {container.state.value!r}"
        )
    if container.is_active:
        raise ProductionRuntimeError(
            "reward resources must remain inactive before model preparation"
        )


def _method(value: object, name: str) -> None:
    if not callable(getattr(value, name, None)):
        raise TypeError(f"injected service must implement {name}()")
