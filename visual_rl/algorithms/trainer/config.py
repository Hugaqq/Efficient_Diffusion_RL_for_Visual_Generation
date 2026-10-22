"""Import-safe declaration for the GRPO-family trainer control flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from visual_rl.core.contracts import (
    DECLARATION_PROVIDER_ABI,
    CatalogFragment,
    ComponentDeclaration,
    ComponentDescriptor,
    DeclaredContract,
    DistributionMode,
    TrainerContract,
    TrainingMode,
)

__all__ = (
    "TRAINER_CATALOG_FRAGMENT",
    "GRPOTrainerConfig",
    "GRPOTrainerDeclarationProvider",
    "trainer_catalog_fragment",
)


@dataclass(frozen=True, slots=True)
class GRPOTrainerConfig:
    """Execution cadence, independent of GRPO objective mathematics."""

    collection_rows_per_commit: int | None = None
    collection_commits: int = 1
    collection_chunk_rows: int = 16

    def __post_init__(self) -> None:
        for name in ("collection_commits", "collection_chunk_rows"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.collection_rows_per_commit is None:
            if self.collection_commits != 1:
                raise ValueError("collection_commits requires collection_rows_per_commit")
        elif (type(self.collection_rows_per_commit) is not int
              or self.collection_rows_per_commit < 1):
            raise ValueError("collection_rows_per_commit must be positive")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        context: object | None,
    ) -> GRPOTrainerConfig:
        del context
        if not isinstance(values, Mapping):
            raise TypeError("grpo trainer params must be a mapping")
        unknown = set(values) - {"collection_rows_per_commit", "collection_commits", "collection_chunk_rows"}
        if unknown:
            raise ValueError(f"unknown grpo trainer params: {sorted(unknown)}")
        return cls(**values)

    def describe_contract(self) -> DeclaredContract:
        return DeclaredContract(
            component_kind="trainer",
            component_id="grpo",
            trainer=TrainerContract(
                accepted_training_modes=(TrainingMode.LORA, TrainingMode.FULL),
                accepted_distribution_modes=(DistributionMode.SINGLE,),
                required_policy_fields=(
                    "active_mask",
                    "algorithm_weight",
                    "base_advantage",
                    "clip_range",
                    "reference_kl_weight",
                ),
                supports_reference_policy=True,
            ),
        )


class GRPOTrainerDeclarationProvider:
    PROVIDER_ABI = DECLARATION_PROVIDER_ABI
    CONFIG_TYPE_PATH = "visual_rl.algorithms.trainer.config:GRPOTrainerConfig"

    @classmethod
    def declare_component(
        cls,
        raw_params: Mapping[str, Any],
        *,
        context: object | None,
    ) -> ComponentDeclaration:
        del cls
        config = GRPOTrainerConfig.from_mapping(raw_params, context=context)
        return ComponentDeclaration(
            config=config, declared_contract=config.describe_contract()
        )


TRAINER_CATALOG_FRAGMENT = CatalogFragment(
    owner="algorithms.trainer",
    kind="trainer",
    descriptors=(
        ComponentDescriptor(
            alias="grpo",
            implementation_class_path=(
                "visual_rl.algorithms.trainer.grpo:RegisteredGRPOTrainer"
            ),
            declaration_provider_path=(
                "visual_rl.algorithms.trainer.config:GRPOTrainerDeclarationProvider"
            ),
        ),
    ),
)


def trainer_catalog_fragment() -> CatalogFragment:
    return TRAINER_CATALOG_FRAGMENT
