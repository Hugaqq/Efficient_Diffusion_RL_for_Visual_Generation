"""Model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from visual_rl.core.types import FrozenMapping, RolloutBatch


class ModelAdapter(ABC):
    """Base contract for every trainable model adapter.

    v0.7 direction (frozen by the master plan, stage 2.1; wired up by the
    atomic cutover):

    - every concrete adapter declares ``MEDIA_TYPE`` explicitly in its class
      body (``"image"`` or ``"video"``); the current lowercase ``media_type``
      attribute is the legacy spelling being replaced;
    - the only sampling entry is ``sample(RolloutRequest) -> RolloutBatch``
      and the only policy recompute entry is
      ``recompute_policy_stats(batch, *, require_reference=False)``;
      the legacy ``sample(prompts, metadata, rollout_config)`` and
      ``recompute_log_probs()`` are removed by the cutover without aliases;
    - ``sample()``/``recompute_policy_stats()`` save and restore the original
      ``train_module.training`` state on both success and exception; callers
      never switch adapter mode themselves;
    - ``parameters()``/``named_parameters()`` enumerate exactly the
      ``requires_grad=True`` parameters of ``train_module``, in one canonical
      order with unique names and unique object identity.
    """

    name: str
    media_type: str
    #: Final class-body media declaration; concrete adapters set it during
    #: the cutover (annotation only here so existing subclasses keep working).
    MEDIA_TYPE: ClassVar[Literal["image", "video"]]

    @property
    @abstractmethod
    def train_module(self) -> Any:
        """Return the ``nn.Module`` that owns trainable adapter state."""

        raise NotImplementedError

    def parameters(self):
        return [
            parameter
            for parameter in self.train_module.parameters()
            if parameter.requires_grad
        ]

    def named_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.train_module.named_parameters()
            if parameter.requires_grad
        ]

    def train(self, mode: bool = True) -> ModelAdapter:
        self.train_module.train(mode)
        return self

    def eval(self) -> ModelAdapter:
        return self.train(False)

    def state_dict(self) -> dict[str, Any]:
        return self.train_module.state_dict()

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> Any:
        return self.train_module.load_state_dict(state_dict, strict=strict)

    @abstractmethod
    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        raise NotImplementedError

    @abstractmethod
    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        raise NotImplementedError

    def recompute_policy_stats(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool = False,
    ):
        """The only policy recompute entry in the final contract.

        Returns a ``PolicyRecomputeStats`` whose differentiable
        ``new_log_probs`` matches ``batch.old_log_probs.shape``; with
        ``require_reference=False`` no reference forward runs and all three
        reference fields are ``None``. Declared non-abstract so existing
        subclasses keep instantiating until the cutover implements it.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not implement recompute_policy_stats() yet"
        )

    def prepare_for_sampling(self) -> None:
        """Compatibility alias for sampling code that predates ``eval``."""

        self.eval()

    def prepare_for_training(self) -> None:
        """Compatibility alias for training code that predates ``train``."""

        self.train()

    def branch_transition_count(self, rollout_config: dict[str, Any]) -> int:
        """Return the number of valid transition indices for branching."""

        return int(rollout_config.get("num_steps", 1))

    @abstractmethod
    def save_pretrained(self, output_dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Unified component factory protocol (plan stage 2.2). All four component
    # base classes expose this same trio plus ``required_capabilities``;
    # none of them is abstract, so unqualified factories fail early with a
    # clear error instead of breaking existing subclasses.
    # ------------------------------------------------------------------

    @classmethod
    def resolve_params(
        cls,
        raw: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        """Whitelist/default/validate/canonicalize component params.

        The base passthrough only requires a mapping and returns a
        deep-frozen copy; concrete components override it with their exact
        parameter whitelist. It never loads weights, creates GPU objects,
        touches the network or writes files.
        """

        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__}.resolve_params() requires a mapping, "
                f"got {type(raw).__name__}"
            )
        return FrozenMapping(raw)

    @classmethod
    def check_environment(
        cls,
        resolved: Mapping[str, Any],
        context: Any,
    ) -> tuple:
        """Bounded, read-only environment checks; default is no checks."""

        return ()

    @classmethod
    def from_config(
        cls,
        resolved: Mapping[str, Any],
        context: Any,
    ):
        """Construct the runtime component from resolved params.

        Only ``build_runtime_components()`` calls this, with the rank-local
        device/backend/precision from the passed context. No default
        implementation exists.
        """

        raise NotImplementedError(
            f"{cls.__name__} does not implement from_config() yet"
        )

    @classmethod
    def required_capabilities(cls, resolved_params: Mapping[str, Any]) -> frozenset:
        """Conditional capabilities implied by the component's own params.

        The base declares none; components override it only when their own
        resolved parameters require an optional capability (e.g. beta > 0).
        """

        return frozenset()
