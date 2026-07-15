from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.optimizers.objective import AlgorithmPolicyObjective
from visual_rl.optimizers.base import OptimizerPlugin
from visual_rl.optimizers.update_engine import UpdateEngine


class AlgorithmOptimizerPlugin(OptimizerPlugin):
    def __init__(
        self,
        algorithm,
        advantage_computer,
        optimizer_config=None,
        max_grad_norm=None,
        update_microbatch_size=None,
        precision="fp32",
    ):
        self.algorithm = algorithm
        self.advantage_computer = advantage_computer
        options = dict(optimizer_config or {})
        max_initial_logprob_delta = options.pop(
            "max_initial_logprob_delta",
            None,
        )
        self.require_initial_clipfrac_zero = bool(
            options.pop("require_initial_clipfrac_zero", False)
        )
        self.require_finite_gradients = bool(
            options.pop("require_finite_gradients", True)
        )
        self.require_nonzero_gradients = bool(
            options.pop("require_nonzero_gradients", False)
        )
        self.optimizer_config = options
        self.objective = AlgorithmPolicyObjective(self.algorithm)
        self.update_engine = UpdateEngine(
            self.advantage_computer,
            self.objective,
            max_initial_logprob_delta=max_initial_logprob_delta,
            require_initial_clipfrac_zero=self.require_initial_clipfrac_zero,
            require_finite_gradients=self.require_finite_gradients,
            require_nonzero_gradients=self.require_nonzero_gradients,
            max_grad_norm=max_grad_norm,
            update_microbatch_size=update_microbatch_size,
            precision=precision,
        )
        self.max_initial_logprob_delta = self.update_engine.max_initial_logprob_delta
        self.max_grad_norm = self.update_engine.max_grad_norm
        self.update_microbatch_size = self.update_engine.update_microbatch_size
        self.precision = self.update_engine.precision

    def build_optimizer(self, parameters: Any, train_config: Any) -> Any:
        import torch

        options = {
            "lr": float(train_config.learning_rate),
            "betas": (
                float(train_config.adam_beta1),
                float(train_config.adam_beta2),
            ),
            "weight_decay": float(train_config.adam_weight_decay),
            "eps": float(train_config.adam_epsilon),
        }
        options.update(self.optimizer_config)
        if "betas" in options:
            options["betas"] = tuple(options["betas"])
        return torch.optim.AdamW(parameters, **options)

    @staticmethod
    def _logprob_metrics(batch, new_log_probs) -> dict[str, float]:
        return UpdateEngine._logprob_metrics(batch, new_log_probs)

    def _validate_pre_update(
        self,
        *,
        logprob_metrics: dict[str, float],
        loss_info: dict[str, Any],
    ) -> None:
        from types import SimpleNamespace

        self.update_engine._validate_pre_update(
            logprob_metrics=logprob_metrics,
            objective_output=SimpleNamespace(clipfrac=loss_info["clipfrac"]),
        )

    @staticmethod
    def _gradient_metrics(parameters: list[Any]) -> dict[str, float | int | bool]:
        return UpdateEngine._gradient_metrics(parameters)

    def _clip_gradients(self, parameters: list[Any]) -> dict[str, Any]:
        return self.update_engine._clip_gradients(parameters)

    @staticmethod
    def _advantage_group_ids(batch: RolloutBatch) -> list[Any]:
        if batch.context is not None:
            return list(batch.group_id)
        return [
            item.get("parent_prompt_index", prompt)
            for prompt, item in zip(
                batch.prompts,
                batch.metadata,
                strict=True,
            )
        ]

    def step(
        self,
        adapter: ModelAdapter,
        batch: RolloutBatch,
        rewards: RewardBatch,
        optimizer: Any,
        context: StepContext,
        *,
        recompute_log_probs: Callable[[RolloutBatch], Any] | None = None,
        gradient_sync_context: Callable[[bool], Any] | None = None,
        reduce_tensor_weighted_mean: Callable[[Any, int], Any] | None = None,
        synchronize_failure: Callable[[bool | BaseException | None], bool]
        | None = None,
        before_optimizer_step: Callable[[], Any] | None = None,
        optimizer_step: Callable[..., Any] | None = None,
    ) -> dict[str, float]:
        routing = {
            "recompute_log_probs": recompute_log_probs,
            "gradient_sync_context": gradient_sync_context,
            "before_optimizer_step": before_optimizer_step,
            "optimizer_step": optimizer_step,
        }
        if (
            reduce_tensor_weighted_mean is not None
            or synchronize_failure is not None
        ):
            routing.update(
                reduce_tensor_weighted_mean=reduce_tensor_weighted_mean,
                synchronize_failure=synchronize_failure,
            )
        return self.update_engine.step(
            adapter,
            batch,
            rewards,
            optimizer,
            context,
            **routing,
        )

    def state_dict(self) -> dict[str, Any]:
        advantage_state = self.advantage_computer.state_dict()
        if not isinstance(advantage_state, Mapping):
            raise TypeError("AdvantageComputer.state_dict() must return a mapping")
        state = {"advantage": deepcopy(dict(advantage_state))}
        scaler_state = self.update_engine.scaler_state_dict()
        if scaler_state is not None:
            state["grad_scaler"] = scaler_state
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("AlgorithmOptimizerPlugin state must be a mapping")
        advantage_state = state.get("advantage", {})
        if not isinstance(advantage_state, Mapping):
            raise TypeError(
                "AlgorithmOptimizerPlugin advantage state must be a mapping"
            )
        scaler_state = state.get("grad_scaler")
        if scaler_state is not None and not isinstance(scaler_state, dict):
            raise TypeError("GradScaler state must be a dict or None")
        if scaler_state is not None and self.precision != "fp16":
            raise ValueError("GradScaler state requires precision='fp16'")

        previous_advantage = self.advantage_computer.state_dict()
        if not isinstance(previous_advantage, Mapping):
            raise TypeError("AdvantageComputer.state_dict() must return a mapping")
        previous_advantage = deepcopy(dict(previous_advantage))
        previous_scaler = self.update_engine.scaler_state_dict()
        try:
            self.advantage_computer.load_state_dict(deepcopy(dict(advantage_state)))
            self.update_engine.load_scaler_state_dict(deepcopy(scaler_state))
        except BaseException as exc:
            rollback_errors: list[tuple[str, BaseException]] = []
            try:
                self.advantage_computer.load_state_dict(previous_advantage)
            except BaseException as rollback_error:
                rollback_errors.append(("advantage", rollback_error))
            try:
                self.update_engine.load_scaler_state_dict(previous_scaler)
            except BaseException as rollback_error:
                rollback_errors.append(("gradient scaler", rollback_error))
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                for name, rollback_error in rollback_errors:
                    add_note(
                        f"Failed to restore {name} state after plugin load failure: "
                        f"{type(rollback_error).__name__}: {rollback_error}"
                    )
            raise
