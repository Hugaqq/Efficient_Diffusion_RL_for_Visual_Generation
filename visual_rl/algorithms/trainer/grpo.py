"""Canonical six-stage control flow shared by the GRPO algorithm family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from visual_rl.algorithms.trainer.config import GRPOTrainerConfig
from visual_rl.algorithms.trainer.interface import (
    IterationPrelude,
    IterationResult,
    PrepareRunContext,
    StageValue,
    TrainerComponent,
    TrainerState,
    UnaryStage,
)

__all__ = ("BaseTrainer", "GRPOTrainer", "RegisteredGRPOTrainer")


class BaseTrainer:
    """Own stage order, lifecycle, reservation outcome, and step advancement."""

    _STAGE_NAMES = ("rollout", "reward", "advantage", "credit", "optimize")

    def __init__(
        self,
        *,
        prelude: IterationPrelude,
        rollout: UnaryStage,
        reward: UnaryStage,
        advantage: UnaryStage,
        credit: UnaryStage,
        optimize: UnaryStage,
        prepare_hooks: Sequence[object] = (),
        close_hooks: Sequence[object] = (),
    ) -> None:
        if not callable(getattr(prelude, "build", None)):
            raise TypeError("prelude must implement build(optimizer_step)")
        stages = {
            "rollout": rollout,
            "reward": reward,
            "advantage": advantage,
            "credit": credit,
            "optimize": optimize,
        }
        for name, stage in stages.items():
            if not callable(stage):
                raise TypeError(f"{name} stage must be callable")
        if isinstance(prepare_hooks, (str, bytes)) or not isinstance(
            prepare_hooks, Sequence
        ):
            raise TypeError("prepare_hooks must be a sequence")
        if isinstance(close_hooks, (str, bytes)) or not isinstance(
            close_hooks, Sequence
        ):
            raise TypeError("close_hooks must be a sequence")
        self._prelude = prelude
        self._stages = stages
        self._prepare_hooks = _unique_identity(tuple(prepare_hooks))
        self._close_hooks = _unique_identity(tuple(close_hooks))
        self._state = TrainerState.NEW
        self._prepare_context: PrepareRunContext | None = None
        self._next_optimizer_step: int | None = None

    @property
    def state(self) -> TrainerState:
        return self._state

    @property
    def next_optimizer_step(self) -> int | None:
        return self._next_optimizer_step

    def prepare_run(self, context: PrepareRunContext) -> None:
        if self._state is not TrainerState.NEW:
            raise RuntimeError("prepare_run() may be called exactly once")
        if not isinstance(context, PrepareRunContext):
            raise TypeError("context must be a PrepareRunContext")
        for hook in self._prepare_hooks:
            method = getattr(hook, "prepare_run", None)
            if not callable(method):
                raise TypeError("prepare hook must implement prepare_run(context)")
            method(context)
        self._prepare_context = context
        self._next_optimizer_step = context.start_optimizer_step
        self._state = TrainerState.PREPARED

    def run_iteration(self, optimizer_step: int) -> IterationResult[object]:
        if self._state is not TrainerState.PREPARED:
            raise RuntimeError("trainer must be prepared before run_iteration()")
        if optimizer_step != self._next_optimizer_step:
            raise ValueError(
                f"expected optimizer step {self._next_optimizer_step}, got {optimizer_step}"
            )
        value = self._prelude.build(optimizer_step)
        if not isinstance(value, StageValue):
            raise TypeError("prelude must return StageValue")
        if value.identity.optimizer_step != optimizer_step:
            raise ValueError("prelude returned the wrong optimizer step identity")
        identity = value.identity
        next_value: StageValue[object] | None = None
        try:
            for name in self._STAGE_NAMES:
                next_value = self._stages[name](value)
                if not isinstance(next_value, StageValue):
                    raise TypeError(f"{name} stage must return StageValue")
                if next_value.identity is not identity:
                    raise ValueError(
                        f"{name} stage replaced the canonical IterationIdentity object"
                    )
                value = next_value
        except BaseException as primary:
            abort = getattr(self._prelude, "abort_iteration", None)
            if callable(abort):
                try:
                    abort(identity)
                except BaseException as cleanup:  # noqa: BLE001
                    try:
                        primary.add_note(
                            "iteration reservation abort failed: "
                            f"{type(cleanup).__name__}"
                        )
                    except AttributeError:
                        pass
            # A retained traceback must not keep the complete trajectory alive.
            del value, next_value
            raise
        commit = getattr(self._prelude, "commit_iteration", None)
        if callable(commit):
            commit(identity)
        result = IterationResult(
            optimizer_step=optimizer_step,
            value=value,
            stage_order=("prelude", *self._STAGE_NAMES),
        )
        self._next_optimizer_step = optimizer_step + 1
        return result

    def close(self) -> None:
        if self._state is TrainerState.CLOSED:
            return
        errors: list[BaseException] = []
        for hook in reversed(self._close_hooks):
            method = getattr(hook, "close", None)
            if not callable(method):
                errors.append(TypeError("close hook must implement close()"))
                continue
            try:
                method()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        self._state = TrainerState.CLOSED
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                try:
                    primary.add_note(f"additional close error: {type(error).__name__}")
                except AttributeError:
                    break
            raise primary

    def __enter__(self) -> BaseTrainer:  # noqa: PYI034
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class GRPOTrainer(BaseTrainer):
    """One composition point for typed GRPO-family stages."""


class RegisteredGRPOTrainer(GRPOTrainer, TrainerComponent):
    """Runtime-loadable trainer consuming the declaration provider config."""

    INTERFACE_VERSION = "1.0"
    CONFIG_TYPE = "visual_rl.algorithms.trainer.config:GRPOTrainerConfig"

    def __init__(
        self,
        *,
        config: GRPOTrainerConfig,
        prelude: IterationPrelude,
        rollout: UnaryStage,
        reward: UnaryStage,
        advantage: UnaryStage,
        credit: UnaryStage,
        optimize: UnaryStage,
        prepare_hooks: Sequence[object] = (),
        close_hooks: Sequence[object] = (),
    ) -> None:
        if not isinstance(config, GRPOTrainerConfig):
            raise TypeError("config must be GRPOTrainerConfig")
        super().__init__(
            prelude=prelude,
            rollout=rollout,
            reward=reward,
            advantage=advantage,
            credit=credit,
            optimize=optimize,
            prepare_hooks=prepare_hooks,
            close_hooks=close_hooks,
        )
        self._config = config

    @property
    def config(self) -> GRPOTrainerConfig:
        """Return the exact frozen config used by the declaration/runtime ABI."""

        return self._config

    @classmethod
    def describe(cls, config: object) -> object:
        if not isinstance(config, GRPOTrainerConfig):
            raise TypeError("config must be GRPOTrainerConfig")
        return config.describe_contract()

    @classmethod
    def from_config(
        cls,
        config: object,
        *,
        runtime_context: Mapping[str, Any],
    ) -> RegisteredGRPOTrainer:
        if not isinstance(config, GRPOTrainerConfig):
            raise TypeError("config must be GRPOTrainerConfig")
        required = (
            "prelude",
            "rollout",
            "reward",
            "advantage",
            "credit",
            "optimize",
        )
        missing = tuple(name for name in required if name not in runtime_context)
        if missing:
            raise ValueError(
                f"grpo trainer runtime bindings are missing: {list(missing)}"
            )
        return cls(
            config=config,
            prelude=runtime_context["prelude"],
            rollout=runtime_context["rollout"],
            reward=runtime_context["reward"],
            advantage=runtime_context["advantage"],
            credit=runtime_context["credit"],
            optimize=runtime_context["optimize"],
            prepare_hooks=tuple(runtime_context.get("prepare_hooks", ())),
            close_hooks=tuple(runtime_context.get("close_hooks", ())),
        )


def _unique_identity(values: tuple[object, ...]) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[int] = set()
    for value in values:
        identity = id(value)
        if identity not in seen:
            seen.add(identity)
            result.append(value)
    return tuple(result)
