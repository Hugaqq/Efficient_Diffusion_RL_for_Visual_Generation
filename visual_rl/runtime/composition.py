"""The sole default composition root for schema-v2 production execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from visual_rl.composition.config.integration import DynamicsProjectionRegistry
from visual_rl.composition.preflight import FilesystemArtifactIdentityResolver
from visual_rl.composition.registry import (
    AlgorithmDeclarationResolver,
    Catalog,
    DeclarationResolver,
)
from visual_rl.runtime.algorithm_binding import DefaultStageAssembler
from visual_rl.runtime.checkpoint_binding import (
    CoordinatorCheckpointSink,
    CoordinatorRestoreService,
    CoordinatorRunFinalizer,
)
from visual_rl.runtime.component_graph import DefaultComponentRuntimeBinder
from visual_rl.runtime.component_loader import RuntimeComponentLoader
from visual_rl.runtime.controller import RunController
from visual_rl.runtime.lifecycle import (
    DefaultRuntimeContextProvider,
    DefaultRuntimeSessionFactory,
    ProductionRuntimeLifecycleBackend,
)
from visual_rl.runtime.probes import (
    DefaultModelRuntimeProbe,
    DefaultPreprocessIdentityProvider,
)
from visual_rl.runtime.reward_resources import RewardResourceFactory
from visual_rl.runtime.types import (
    CheckpointSink,
    ComponentRuntimeBinder,
    ProductionRuntimeContextProvider,
    RestoreService,
    RuntimeSessionFactory,
    StageAssembler,
    TransformExecutor,
)

__all__ = ("create_default_run_controller", "create_run_controller")


def create_run_controller(
    *,
    artifact_resolver: object,
    runtime_factory: RuntimeSessionFactory,
    runtime_context_provider: ProductionRuntimeContextProvider,
    stage_assembler: StageAssembler,
    component_binder: ComponentRuntimeBinder,
    checkpoint_sink: CheckpointSink,
    catalog: Catalog | None = None,
    restore_service: RestoreService | None = None,
    transform_executor: TransformExecutor | None = None,
    declaration_resolver: DeclarationResolver | None = None,
    algorithm_declaration_resolver: AlgorithmDeclarationResolver | None = None,
    dynamics_projection_registry: DynamicsProjectionRegistry | None = None,
    runtime_component_loader: RuntimeComponentLoader | None = None,
) -> RunController:
    """Create the sole explicitly service-injected production controller."""

    backend = ProductionRuntimeLifecycleBackend(
        catalog=catalog,
        artifact_resolver=artifact_resolver,
        runtime_factory=runtime_factory,
        runtime_context_provider=runtime_context_provider,
        stage_assembler=stage_assembler,
        component_binder=component_binder,
        checkpoint_sink=checkpoint_sink,
        restore_service=restore_service,
        transform_executor=transform_executor,
        declaration_resolver=declaration_resolver,
        algorithm_declaration_resolver=algorithm_declaration_resolver,
        dynamics_projection_registry=dynamics_projection_registry,
        runtime_component_loader=runtime_component_loader,
    )
    return RunController(backend)


def create_default_run_controller(
    *,
    code_root: Path | None = None,
    model_loader: Callable[..., object] | None = None,
    reward_resource_factory: RewardResourceFactory | None = None,
    dynamics_projection_registry: DynamicsProjectionRegistry | None = None,
) -> RunController:
    """Build one production controller with bounded extension seams.

    Model and reward leaf replacement is useful for the V0 fake-artifact
    vertical slice.  The immutable Dynamics projection registry is the one
    compiler-policy seam needed to add a model binding family without editing
    the core compiler. Callers cannot replace the lifecycle backend, graph,
    binder, stage assembler, checkpoint sink, or restore service through this
    default entry.
    """

    runtime_factory = DefaultRuntimeSessionFactory(
        model_loader=model_loader,
        reward_resource_factory=reward_resource_factory,
    )
    binder = DefaultComponentRuntimeBinder(
        model_probe=DefaultModelRuntimeProbe(),
        preprocess_identity_provider=DefaultPreprocessIdentityProvider(),
    )
    finalizer = CoordinatorRunFinalizer()
    return create_run_controller(
        artifact_resolver=FilesystemArtifactIdentityResolver(code_root),
        runtime_factory=runtime_factory,
        runtime_context_provider=DefaultRuntimeContextProvider(),
        stage_assembler=DefaultStageAssembler(),
        component_binder=binder,
        checkpoint_sink=CoordinatorCheckpointSink(finalizer=finalizer),
        restore_service=CoordinatorRestoreService(finalizer=finalizer),
        dynamics_projection_registry=dynamics_projection_registry,
    )
