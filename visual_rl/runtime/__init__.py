"""Internal v0.8 runtime composition and lifecycle control.

The compatibility exports below are intentionally lazy.  Importing the runtime
package itself is part of the core-wheel contract and must not eagerly import
training-only dependencies such as PyTorch.
"""

from __future__ import annotations

from importlib import import_module

_EXPORT_GROUPS = {
    "algorithm_binding": (
        "AlgorithmRuntimeComponent",
        "AlgorithmRuntimeComponents",
        "AlgorithmRuntimeBindingError",
        "BindOncePrelude",
        "BindOnceStage",
        "CanonicalAlgorithmMaterializationError",
        "CanonicalAlgorithmMaterializer",
        "DefaultStageAssembler",
        "DefaultStageAssemblyError",
        "DynamicsRuntimeBindEvidence",
        "PerRolloutDynamicsFactory",
        "StageBindingError",
        "TrainerAlgorithmExecution",
        "bind_per_rollout_dynamics_factory",
    ),
    "model_binding": (
        "ModelRuntimeProbe",
        "ModelRuntimeProbeRequest",
        "ModelRuntimeProbeResult",
    ),
    "preprocess_binding": (
        "PreprocessIdentityProvider",
        "PreprocessIdentityRequest",
        "PreprocessIdentityResult",
        "PreprocessRequirementCompileError",
        "compile_preprocess_requirement_set",
    ),
    "checkpoint_binding": (
        "CoordinatorCheckpointSink",
        "CoordinatorRestoreService",
        "CoordinatorRunFinalizer",
        "RuntimeCheckpointError",
    ),
    "component_graph": (
        "ComponentRuntimeBindingError",
        "DefaultComponentRuntimeBinder",
        "RuntimeAssemblyError",
        "RuntimeComponentBinding",
        "RuntimeComponentGraph",
        "RuntimeContextFactory",
        "load_component_graph",
    ),
    "controller": (
        "ControllerStage",
        "ControllerState",
        "RestoreBoundOutcome",
        "RuntimeLifecycleBackend",
    ),
    "probes": (
        "DefaultModelRuntimeProbe",
        "DefaultPreprocessIdentityProvider",
    ),
    "lifecycle": (
        "DefaultRuntimeContextProvider",
        "DefaultRuntimeSessionError",
        "DefaultRuntimeSessionFactory",
        "ProductionRuntimeLifecycleBackend",
    ),
    "types": (
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
        "RunResult",
    ),
    "resources": (
        "DefaultRuntimeResourceContainer",
    ),
    "reward_resources": (
        "AcquiredRewardResource",
        "DefaultRewardResourceFactory",
        "DefaultRewardResourceFactoryError",
        "RewardAcquisitionPlanningError",
        "RewardResourceAcquireRequest",
        "RewardResourceBindingFacts",
        "RewardResourceFactory",
        "RuntimeResourceAcquisitionError",
        "bound_reward_resource_id",
        "bound_reward_resource_payload",
        "compile_reward_resource_acquisition",
    ),
}

_EXPORT_MODULE_BY_NAME = {
    name: f"visual_rl.runtime.{module_name}"
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}

__all__ = (
    "AcquiredRewardResource",
    "AlgorithmRuntimeComponent",
    "AlgorithmRuntimeComponents",
    "BindOncePrelude",
    "BindOnceStage",
    "BoundRestoreRequest",
    "BoundRestoreResult",
    "CanonicalAlgorithmMaterializationError",
    "CanonicalAlgorithmMaterializer",
    "CheckpointRequest",
    "CheckpointSink",
    "CompiledProductionRun",
    "ComponentBindRequest",
    "ComponentRuntimeBinder",
    "ComponentRuntimeBindingError",
    "ComponentRuntimeEvidence",
    "ControllerStage",
    "ControllerState",
    "CoordinatorCheckpointSink",
    "CoordinatorRestoreService",
    "CoordinatorRunFinalizer",
    "DefaultComponentRuntimeBinder",
    "DefaultModelRuntimeProbe",
    "DefaultPreprocessIdentityProvider",
    "DefaultRewardResourceFactory",
    "DefaultRewardResourceFactoryError",
    "DefaultRuntimeContextProvider",
    "DefaultRuntimeResourceContainer",
    "DefaultRuntimeSessionError",
    "DefaultRuntimeSessionFactory",
    "DefaultStageAssembler",
    "DefaultStageAssemblyError",
    "DynamicsRuntimeBindEvidence",
    "ModelRuntimeProbe",
    "ModelRuntimeProbeRequest",
    "ModelRuntimeProbeResult",
    "PerRolloutDynamicsFactory",
    "PolicyTensorRuntimeSpec",
    "PreparedRestoreRequest",
    "PreparedRestoreResult",
    "PreprocessIdentityProvider",
    "PreprocessIdentityRequest",
    "PreprocessIdentityResult",
    "PreprocessRequirementCompileError",
    "ProductionBoundRun",
    "ProductionGraph",
    "ProductionPreflight",
    "ProductionPreparedRun",
    "ProductionRuntime",
    "ProductionRuntimeContextProvider",
    "ProductionRuntimeError",
    "ProductionRuntimeLifecycleBackend",
    "RestoreBoundOutcome",
    "RestoreService",
    "RewardResourceAcquireRequest",
    "RewardResourceBindingFacts",
    "RewardResourceFactory",
    "RunResult",
    "RuntimeAssemblyError",
    "RuntimeCheckpointError",
    "RuntimeComponentBinding",
    "RuntimeComponentGraph",
    "RuntimeContextFactory",
    "RuntimeContextRequest",
    "RuntimeCreateRequest",
    "RuntimeLifecycleBackend",
    "RuntimeResourceAcquisitionError",
    "RuntimeSession",
    "RuntimeSessionFactory",
    "SafePointCheckpointReceipt",
    "StageAssembler",
    "StageAssemblyRequest",
    "StageBindingError",
    "StageCheckpointPortError",
    "StageCheckpointPorts",
    "TrainerAlgorithmExecution",
    "TrainerStageAssembly",
    "TransformExecution",
    "TransformExecutor",
    "TransformRequest",
    "bound_reward_resource_id",
    "bound_reward_resource_payload",
    "compile_preprocess_requirement_set",
    "load_component_graph",
)


def __getattr__(name: str) -> object:
    """Resolve a compatibility export only when a caller requests it."""

    module_name = _EXPORT_MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORT_MODULE_BY_NAME))
