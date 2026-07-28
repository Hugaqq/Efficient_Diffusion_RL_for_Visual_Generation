"""Frozen builtin component manifest and the only name lookup (v0.7 stage 2).

This module is the single declaration source for internal components. It
defines the immutable :class:`ComponentSpec` description, the closed
capability vocabulary with its per-kind owners, and the fixed builtin
manifest tuple consumed by ``builtin_components()`` /
``get_builtin_component()``.

Rules frozen by the master plan (stage 2.2-2.5):

- exactly four kinds: ``model``/``rollout``/``reward``/``algorithm``; no
  ``optimizer``/``feedback_provider``/``runner``/``objective``/``artifact``
  component kind exists;
- ``ComponentSpec`` is a frozen description, not a registry: this module
  provides no add/register/freeze/snapshot/override API and no entry-point
  scanning; ``(kind, name)`` uniqueness is enforced by architecture tests;
- ``factory`` is always the real concrete class, never an import string or
  lazy descriptor; every module imported here stays module-import-level free
  of ``torch``/``diffusers``/``transformers``/``peft``;
- ``dependencies`` values are bare Python import names checkable with
  ``importlib.util.find_spec()``; version constraints live only in
  ``pyproject.toml``;
- there is no per-component version/source-identity field and no
  ``validator``/``REQUIRED_PARAMS``/``DEFAULT_PARAMS`` parallel field;
  component parameters are owned by each factory's ``resolve_params()``.

Transition notes for this incremental phase:

- ``wan_flash`` is not declared yet: no dedicated ``WanFlashAdapter`` class
  exists in the tree. It joins the manifest in the later atomic cutover that
  splits ``model_adapters/wan.py`` into ``WanFlashAdapter`` /
  ``WanWorldR1Adapter``. The ``wan_world_r1`` entry is temporarily backed by
  the existing ``WorldR1WanLegacyAdapter`` class and the factory reference is
  updated in that same cutover.
- Retired aliases (``tempflow_sd3_legacy``, ``world_r1_wan_legacy``, ``wan``,
  ``mock_wan``, ``flash_single_step``, ``remote_pickle``, ``pickscore``,
  ``video_hpsv3``) are deliberately absent and resolve as unknown names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
from visual_rl.model_adapters.wan import WorldR1WanLegacyAdapter
from visual_rl.optimizers.flash_grpo import FlashGRPOAlgorithm
from visual_rl.optimizers.grpo import GRPOAlgorithm
from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm
from visual_rl.rollout.branching import BranchingRollout
from visual_rl.rollout.full_trajectory import FullTrajectoryRollout
from visual_rl.rollout.single_step import SingleStepRollout

__all__ = [
    "CAPABILITY_OWNER",
    "CAPABILITY_VOCABULARY",
    "COMPONENT_KINDS",
    "ComponentKind",
    "ComponentSpec",
    "builtin_components",
    "get_builtin_component",
]

ComponentKind = Literal["model", "rollout", "reward", "algorithm"]

#: The exact four legal component kinds, in canonical order.
COMPONENT_KINDS: tuple[ComponentKind, ...] = ("model", "rollout", "reward", "algorithm")

#: Closed optional-capability vocabulary (stage 2.5). A capability only ever
#: describes semantics that some legal component has and another lacks;
#: base-interface guarantees are never repeated as capabilities.
CAPABILITY_VOCABULARY = frozenset(
    {
        "media.image",
        "media.video",
        "sampling.full_trajectory",
        "sampling.single_step",
        "sampling.branching",
        "rollout.full_trajectory",
        "rollout.single_step",
        "rollout.branching",
        "policy.reference_stats",
        "conditioning.camera",
    }
)

#: Each capability has exactly one legal provider kind. Requirements may be
#: declared by any selected component but are only satisfied by the selected
#: component of the owner kind.
CAPABILITY_OWNER = {
    "media.image": "model",
    "media.video": "model",
    "sampling.full_trajectory": "model",
    "sampling.single_step": "model",
    "sampling.branching": "model",
    "rollout.full_trajectory": "rollout",
    "rollout.single_step": "rollout",
    "rollout.branching": "rollout",
    "policy.reference_stats": "model",
    "conditioning.camera": "model",
}


@dataclass(frozen=True)
class ComponentSpec:
    """Frozen description of one builtin component.

    ``provides``/``requires`` only carry optional cross-component
    capabilities from :data:`CAPABILITY_VOCABULARY`; ``dependencies`` only
    carries import names without version expressions.
    """

    kind: ComponentKind
    name: str
    factory: type

    provides: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    dependencies: tuple[str, ...] = ()


_DIFFUSERS_MODEL_DEPENDENCIES = (
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "sentencepiece",
    "google.protobuf",
)

_BUILTIN_COMPONENTS: tuple[ComponentSpec, ...] = (
    # ------------------------------------------------------------------ model
    ComponentSpec(
        kind="model",
        name="tiny_diffusion",
        factory=TinyDiffusionAdapter,
        provides=frozenset(
            {
                "media.image",
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
                "sampling.full_trajectory",
                "sampling.branching",
            }
        ),
        dependencies=_DIFFUSERS_MODEL_DEPENDENCIES,
    ),
    ComponentSpec(
        kind="model",
        name="wan_world_r1",
        factory=WorldR1WanLegacyAdapter,
        provides=frozenset(
            {
                "media.video",
                "sampling.full_trajectory",
                "conditioning.camera",
            }
        ),
        dependencies=(*_DIFFUSERS_MODEL_DEPENDENCIES, "einops", "rp"),
    ),
    # ---------------------------------------------------------------- rollout
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
    # ----------------------------------------------------------------- reward
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
    # -------------------------------------------------------------- algorithm
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
    """Return the fixed-order, immutable builtin manifest."""

    return _BUILTIN_COMPONENTS


def get_builtin_component(kind: str, name: str) -> ComponentSpec:
    """Look up one builtin component; the only name lookup in the project.

    Unknown kinds raise :class:`ComponentError`; unknown names (including
    every retired alias) raise :class:`UnknownComponentError`, a
    :class:`ComponentError` subclass listing the available same-kind names.
    """

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
