"""Import-safe trainer declaration surface."""

from visual_rl.algorithms.trainer.config import (
    TRAINER_CATALOG_FRAGMENT,
    GRPOTrainerConfig,
    GRPOTrainerDeclarationProvider,
    trainer_catalog_fragment,
)
from visual_rl.algorithms.trainer.execution_plan import (
    AlgorithmExecutionPlan,
    AlgorithmPlanError,
    PolicyMicrobatchCardinality,
    ReplayTarget,
    StageTraceEntry,
    TransitionSelectionKind,
    TransitionSelectionSpec,
    UpdateCardinality,
)
from visual_rl.algorithms.trainer.interface import (
    IterationIdentity,
    IterationResult,
    PrepareRunContext,
    StageValue,
    TrainerComponent,
    TrainerState,
)

__all__ = (
    "TRAINER_CATALOG_FRAGMENT",
    "AlgorithmExecutionPlan",
    "AlgorithmPlanError",
    "GRPOTrainerConfig",
    "GRPOTrainerDeclarationProvider",
    "IterationIdentity",
    "IterationResult",
    "PolicyMicrobatchCardinality",
    "PrepareRunContext",
    "ReplayTarget",
    "StageTraceEntry",
    "StageValue",
    "TrainerComponent",
    "TrainerState",
    "TransitionSelectionKind",
    "TransitionSelectionSpec",
    "UpdateCardinality",
    "trainer_catalog_fragment",
)
