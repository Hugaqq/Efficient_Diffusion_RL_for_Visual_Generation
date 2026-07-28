"""Fixed internal facade around the sole UpdateEngine."""

from __future__ import annotations

from typing import Any

from visual_rl.core.types import RewardBatch, RolloutBatch, StepContext
from visual_rl.optimizers.base import OptimizerPlugin
from visual_rl.optimizers.objective import PolicyObjective
from visual_rl.optimizers.update_engine import UpdateEngine, UpdateResult


class AlgorithmOptimizerPlugin(OptimizerPlugin):
    """Assemble one algorithm, one objective, and one stateless update engine."""

    def __init__(
        self,
        *,
        algorithm: Any,
        advantage_computer: Any,
        update_microbatch_size: int | None,
        precision: str,
        max_grad_norm: float | None,
        max_initial_logprob_delta: float | None,
        require_initial_clipfrac_zero: bool,
        require_finite_gradients: bool,
        require_nonzero_gradients: bool,
    ) -> None:
        self.algorithm = algorithm
        self.advantage_computer = advantage_computer
        self.objective = PolicyObjective()
        self.update_engine = UpdateEngine(
            algorithm=self.algorithm,
            advantage_function=self.advantage_computer,
            objective=self.objective,
            max_initial_logprob_delta=max_initial_logprob_delta,
            require_initial_clipfrac_zero=require_initial_clipfrac_zero,
            require_finite_gradients=require_finite_gradients,
            require_nonzero_gradients=require_nonzero_gradients,
            max_grad_norm=max_grad_norm,
            update_microbatch_size=update_microbatch_size,
            precision=precision,
        )
        self.max_initial_logprob_delta = (
            self.update_engine.max_initial_logprob_delta
        )
        self.max_grad_norm = self.update_engine.max_grad_norm
        self.update_microbatch_size = self.update_engine.update_microbatch_size
        self.precision = self.update_engine.precision

    def build_optimizer(
        self,
        trainable_named_parameters: tuple[tuple[str, Any], ...],
        config: Any,
    ) -> Any:
        import torch

        if type(trainable_named_parameters) is not tuple:
            raise TypeError("trainable_named_parameters must be a tuple")
        names = tuple(name for name, _parameter in trainable_named_parameters)
        parameters = tuple(
            parameter for _name, parameter in trainable_named_parameters
        )
        if not parameters:
            raise ValueError("optimizer requires trainable parameters")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("trainable parameter names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("trainable parameter names must be unique")
        if any(
            not isinstance(parameter, torch.nn.Parameter)
            or not parameter.requires_grad
            for parameter in parameters
        ):
            raise TypeError(
                "trainable parameters must be trainable torch.nn.Parameter"
            )
        identities = tuple(id(parameter) for parameter in parameters)
        if len(identities) != len(set(identities)):
            raise ValueError("trainable parameter identities must be unique")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config.learning_rate),
            betas=(
                float(config.adam_beta1),
                float(config.adam_beta2),
            ),
            weight_decay=float(config.adam_weight_decay),
            eps=float(config.adam_epsilon),
        )
        optimizer_ids = tuple(
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if optimizer_ids != identities:
            raise RuntimeError(
                "AdamW changed trainable parameter identity/order"
            )
        return optimizer

    def step(
        self,
        *,
        batch: RolloutBatch,
        rewards: RewardBatch,
        optimizer: Any,
        scaler: Any | None,
        context: StepContext,
        strategy: Any,
    ) -> UpdateResult:
        return self.update_engine.step(
            batch=batch,
            rewards=rewards,
            optimizer=optimizer,
            scaler=scaler,
            context=context,
            strategy=strategy,
        )

    def close(self) -> None:
        """The final plugin owns no resources or mutable training state."""
