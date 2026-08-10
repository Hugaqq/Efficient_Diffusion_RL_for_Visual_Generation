"""Typed data-routing primitives used before rollout expansion."""

from visual_rl.data.group_placement import (
    BoundGroupPlacementLayout,
    CollectiveDomain,
    GroupMemberPlacement,
    GroupPlacementContract,
    GroupPlacementError,
    GroupPlacementKind,
    GroupPlacementLayout,
    PlacedBatchRow,
)
from visual_rl.data.media import DecodedMediaBatch, DecodedMediaLayout
from visual_rl.data.phase_schedule import (
    BatchPhaseBinding,
    ImplicitPhaseRouter,
    PeriodicPhaseSchedule,
    PhaseDefinition,
    PhaseRoute,
    PhaseRouter,
    PhaseScheduleState,
    world_r1_release_phase_schedule,
)
from visual_rl.data.preprocess import (
    PreprocessBarrier,
    PreprocessCacheReader,
    PreprocessCacheWriter,
    PreprocessCompatibilityReceipt,
    PreprocessComponentIdentity,
    PreprocessComponentRole,
    PreprocessConsumerRequirement,
    PreprocessContractError,
    PreprocessDependency,
    PreprocessedItem,
    PreprocessGeometry,
    PreprocessManifest,
    PreprocessPlan,
    PreprocessPortContract,
    PreprocessProducerSpec,
    PreprocessRequirementProvider,
    PreprocessRequirementSet,
    PreprocessTransform,
    PreprocessWriteLease,
)
from visual_rl.data.source_plan import (
    DatasetArtifactBinding,
    DatasetSourceSpec,
    SourceContentBinding,
    SourceLoadError,
    SourceLoadRequest,
    SourceLocationBinding,
    SourcePlanSpec,
)
from visual_rl.data.source_records import (
    SOURCE_DESCRIPTOR_BY_SELECTOR,
    DatasetSourceDescriptor,
)
from visual_rl.data.source_sampler import (
    MultiSourceSampler,
    SamplerPreview,
    SamplerReservation,
    SamplerState,
    SourceSequence,
)
from visual_rl.data.stable_source_loader import load_stable_source_sequences

__all__ = (
    "SOURCE_DESCRIPTOR_BY_SELECTOR",
    "BatchPhaseBinding",
    "BoundGroupPlacementLayout",
    "CollectiveDomain",
    "DataPlaneCheckpointPort",
    "DataPlaneCheckpointView",
    "DataPlanePrelude",
    "DataPlanePreludeState",
    "DatasetArtifactBinding",
    "DatasetSourceDescriptor",
    "DatasetSourceSpec",
    "DecodedMediaBatch",
    "DecodedMediaLayout",
    "GroupMemberPlacement",
    "GroupPlacementContract",
    "GroupPlacementError",
    "GroupPlacementKind",
    "GroupPlacementLayout",
    "ImplicitPhaseRouter",
    "InlinePreprocessPlanFactory",
    "InlinePreprocessPlanRequest",
    "InlinePreprocessPlanResolution",
    "MultiSourceSampler",
    "PeriodicPhaseSchedule",
    "PhaseDefinition",
    "PhaseRoute",
    "PhaseRouter",
    "PhaseScheduleState",
    "PlacedBatchRow",
    "PreludeBatchPayload",
    "PreprocessBarrier",
    "PreprocessCacheReader",
    "PreprocessCacheWriter",
    "PreprocessCompatibilityReceipt",
    "PreprocessComponentIdentity",
    "PreprocessComponentRole",
    "PreprocessConsumerRequirement",
    "PreprocessContractError",
    "PreprocessDependency",
    "PreprocessGeometry",
    "PreprocessManifest",
    "PreprocessPlan",
    "PreprocessPortContract",
    "PreprocessProducerSpec",
    "PreprocessRequirementProvider",
    "PreprocessRequirementSet",
    "PreprocessTransform",
    "PreprocessWriteLease",
    "PreprocessedItem",
    "SamplerPreview",
    "SamplerReservation",
    "SamplerState",
    "SourceContentBinding",
    "SourceLoadError",
    "SourceLoadRequest",
    "SourceLocationBinding",
    "SourcePlanSpec",
    "SourceSequence",
    "load_stable_source_sequences",
    "world_r1_release_phase_schedule",
)


def __getattr__(name: str):
    """Lazily expose optional data owners without broadening import roots."""

    if name in {
        "InlinePreprocessPlanFactory",
        "InlinePreprocessPlanRequest",
        "InlinePreprocessPlanResolution",
    }:
        from visual_rl.data import preprocess_factory

        return getattr(preprocess_factory, name)
    if name in {
        "DataPlaneCheckpointPort",
        "DataPlaneCheckpointView",
        "DataPlanePrelude",
        "DataPlanePreludeState",
        "PreludeBatchPayload",
    }:
        from visual_rl.data import prelude

        return getattr(prelude, name)
    raise AttributeError(name)
