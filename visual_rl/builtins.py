"""The single immutable manifest of repository-owned VisualRL components."""

from __future__ import annotations

from visual_rl.core.components import (
    COMPONENT_KINDS,
    ComponentSpec,
)
from visual_rl.errors import ComponentError, UnknownComponentError
from visual_rl.feedback.clients import MockRewardClient
from visual_rl.feedback.image_rewards import (
    PromptColorGuardedRewardClient,
    PromptColorMarginRewardClient,
    PromptColorRewardClient,
)
from visual_rl.feedback.world_r1_rewards import (
    WorldR1Reward3DClient,
    WorldR1RewardGeneralClient,
)
from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter
from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter
from visual_rl.model_adapters.wan import (
    WanFlashAdapter,
    WanWorldR1Adapter,
)
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm
from visual_rl.optimizers.grpo import GRPOAlgorithm
from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm
from visual_rl.rollout.branching import BranchingRollout
from visual_rl.rollout.full_trajectory import FullTrajectoryRollout
from visual_rl.rollout.single_step import SingleStepRollout

__all__ = ["builtin_components", "get_builtin_component"]

_DIFFUSERS_MODEL_DEPENDENCIES = (
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "sentencepiece",
    "google.protobuf",
)

_BUILTIN_COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        kind="model",
        name="tiny_diffusion",
        factory=TinyDiffusionAdapter,
        provides=frozenset(
            {
                "media.image",
                "policy.reference_stats",
                "sampling.full_trajectory",
                "sampling.single_step",
                "sampling.branching",
            }
        ),
        dependencies=("torch",),
    ),
    ComponentSpec(
        kind="model",
        name="sd3_tempflow",
        factory=SD3TempFlowAdapter,
        provides=frozenset(
            {
                "media.image",
                "policy.reference_stats",
                "sampling.full_trajectory",
                "sampling.branching",
            }
        ),
        dependencies=_DIFFUSERS_MODEL_DEPENDENCIES,
    ),
    ComponentSpec(
        kind="model",
        name="wan_flash",
        factory=WanFlashAdapter,
        provides=frozenset({"media.video", "sampling.single_step"}),
        dependencies=_DIFFUSERS_MODEL_DEPENDENCIES,
    ),
    ComponentSpec(
        kind="model",
        name="wan_world_r1",
        factory=WanWorldR1Adapter,
        provides=frozenset(
            {
                "media.video",
                "sampling.full_trajectory",
                "conditioning.camera",
            }
        ),
        dependencies=(*_DIFFUSERS_MODEL_DEPENDENCIES, "einops", "rp"),
    ),
    ComponentSpec(
        kind="rollout",
        name="full_trajectory",
        factory=FullTrajectoryRollout,
        provides=frozenset({"rollout.full_trajectory"}),
        requires=frozenset({"sampling.full_trajectory"}),
        dependencies=("torch",),
    ),
    ComponentSpec(
        kind="rollout",
        name="single_step",
        factory=SingleStepRollout,
        provides=frozenset({"rollout.single_step"}),
        requires=frozenset({"sampling.single_step"}),
        dependencies=("torch",),
    ),
    ComponentSpec(
        kind="rollout",
        name="branching",
        factory=BranchingRollout,
        provides=frozenset({"rollout.branching"}),
        requires=frozenset({"sampling.branching"}),
        dependencies=("torch",),
    ),
    ComponentSpec(
        kind="reward",
        name="mock",
        factory=MockRewardClient,
        dependencies=("numpy",),
    ),
    ComponentSpec(
        kind="reward",
        name="prompt_color",
        factory=PromptColorRewardClient,
        requires=frozenset({"media.image"}),
        dependencies=("numpy",),
    ),
    ComponentSpec(
        kind="reward",
        name="prompt_color_margin",
        factory=PromptColorMarginRewardClient,
        requires=frozenset({"media.image"}),
        dependencies=("numpy",),
    ),
    ComponentSpec(
        kind="reward",
        name="prompt_color_guarded",
        factory=PromptColorGuardedRewardClient,
        requires=frozenset({"media.image"}),
        dependencies=("numpy",),
    ),
    ComponentSpec(
        kind="reward",
        name="reward_general",
        factory=WorldR1RewardGeneralClient,
        requires=frozenset({"media.video"}),
        dependencies=("numpy", "PIL", "requests"),
    ),
    ComponentSpec(
        kind="reward",
        name="reward_3d",
        factory=WorldR1Reward3DClient,
        requires=frozenset({"media.video", "conditioning.camera"}),
        dependencies=("numpy", "PIL", "requests"),
    ),
    ComponentSpec(
        kind="algorithm",
        name="grpo",
        factory=GRPOAlgorithm,
        requires=frozenset({"rollout.full_trajectory"}),
        dependencies=("torch",),
    ),
    ComponentSpec(
        kind="algorithm",
        name="flash_grpo",
        factory=FlashGRPOAlgorithm,
        requires=frozenset({"media.video", "rollout.single_step"}),
        dependencies=("torch",),
    ),
    ComponentSpec(
        kind="algorithm",
        name="tempflow_grpo",
        factory=TempFlowGRPOAlgorithm,
        requires=frozenset({"media.image", "rollout.branching"}),
        dependencies=("torch",),
    ),
)


def builtin_components() -> tuple[ComponentSpec, ...]:
    """Return the canonical fixed-order builtin manifest."""

    return _BUILTIN_COMPONENTS


def get_builtin_component(kind: str, name: str) -> ComponentSpec:
    """Resolve one builtin component by its canonical kind and name."""

    if kind not in COMPONENT_KINDS:
        raise ComponentError(
            f"Unknown component kind {kind!r}; expected one of "
            f"{list(COMPONENT_KINDS)}",
            kind=kind,
            name=name,
        )
    for spec in _BUILTIN_COMPONENTS:
        if spec.kind == kind and spec.name == name:
            return spec
    available = tuple(
        spec.name for spec in _BUILTIN_COMPONENTS if spec.kind == kind
    )
    raise UnknownComponentError(kind, name, available)
