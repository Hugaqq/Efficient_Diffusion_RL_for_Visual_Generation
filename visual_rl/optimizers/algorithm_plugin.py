from typing import Any

from visual_rl.core.types import RolloutBatch, RewardBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.optimizers.base import OptimizerPlugin


class AlgorithmOptimizerPlugin(OptimizerPlugin):
    def __init__(self, algorithm, advantage_computer, optimizer_config=None):
        self.algorithm = algorithm
        self.advantage_computer = advantage_computer
        options = dict(optimizer_config or {})
        self.max_initial_logprob_delta = options.pop(
            "max_initial_logprob_delta",
            None,
        )
        if self.max_initial_logprob_delta is not None:
            self.max_initial_logprob_delta = float(
                self.max_initial_logprob_delta
            )
            if self.max_initial_logprob_delta < 0:
                raise ValueError("max_initial_logprob_delta must be non-negative")
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
        import torch

        new_log_probs = new_log_probs.detach().float()
        old_log_probs = torch.as_tensor(batch.old_log_probs, device=new_log_probs.device).detach().float()
        delta = new_log_probs - old_log_probs
        metrics = {
            "old_logprob_mean": float(old_log_probs.mean().cpu()),
            "new_logprob_mean": float(new_log_probs.mean().cpu()),
            "logprob_delta_mean": float(delta.mean().cpu()),
            "logprob_delta_abs_max": float(delta.abs().max().cpu()),
        }
        if batch.kl is not None:
            kl = torch.as_tensor(batch.kl, device=new_log_probs.device).detach().float()
            metrics["rollout_kl_mean"] = float(kl.mean().cpu())
            metrics["rollout_kl_abs_max"] = float(kl.abs().max().cpu())
        return metrics

    def _validate_pre_update(
        self,
        *,
        logprob_metrics: dict[str, float],
        loss_info: dict[str, Any],
    ) -> None:
        import math

        max_delta = float(logprob_metrics["logprob_delta_abs_max"])
        if not math.isfinite(max_delta):
            raise RuntimeError(
                "Pre-update log-prob parity produced a non-finite delta"
            )
        if (
            self.max_initial_logprob_delta is not None
            and max_delta > self.max_initial_logprob_delta
        ):
            raise RuntimeError(
                "Pre-update log-prob parity gate failed: "
                f"max_abs_delta={max_delta:.6g} exceeds "
                f"{self.max_initial_logprob_delta:.6g}"
            )
        clipfrac = float(loss_info["clipfrac"].detach().cpu())
        if self.require_initial_clipfrac_zero and clipfrac != 0.0:
            raise RuntimeError(
                "Pre-update clipfrac gate failed: "
                f"expected 0, got {clipfrac:.6g}"
            )

    @staticmethod
    def _gradient_metrics(parameters: list[Any]) -> dict[str, float | int | bool]:
        import math

        import torch

        squared_norm = 0.0
        nonzero_count = 0
        tensor_count = 0
        finite = True
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            tensor_count += 1
            detached = gradient.detach().float()
            finite = finite and bool(torch.isfinite(detached).all())
            nonzero_count += int(torch.count_nonzero(detached).item())
            squared_norm += float(
                torch.sum(detached.double() * detached.double()).item()
            )
        return {
            "grad_norm": float(math.sqrt(squared_norm)),
            "grad_nonzero_count": nonzero_count,
            "grad_tensor_count": tensor_count,
            "gradients_finite": finite,
        }

    def step(
        self,
        adapter: ModelAdapter,
        batch: RolloutBatch,
        rewards: RewardBatch,
        optimizer: Any,
        context: dict[str, Any],
    ) -> dict[str, float]:
        del context
        advantage_result = self.advantage_computer.compute(
            batch.prompts,
            rewards.raw,
            rewards.weighted_total,
            group_ids=[
                item.get("parent_prompt_index", prompt)
                for prompt, item in zip(batch.prompts, batch.metadata, strict=True)
            ],
        )
        new_log_probs = adapter.recompute_log_probs(batch)
        loss, loss_info = self.algorithm.compute_loss(
            batch,
            advantage_result.advantages,
            new_log_probs,
        )
        logprob_metrics = self._logprob_metrics(batch, new_log_probs)
        self._validate_pre_update(
            logprob_metrics=logprob_metrics,
            loss_info=loss_info,
        )

        extra_loss_metrics = {}

        for key, value in loss_info.items():
            if key in {"approx_kl", "clipfrac", "policy_loss"}:
                continue
            if hasattr(value, "detach"):
                extra_loss_metrics[key] = float(value.detach().cpu())
            elif isinstance(value, (int, float)):
                extra_loss_metrics[key] = float(value)

        parameters = list(adapter.parameters())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_metrics = self._gradient_metrics(parameters)
        if self.require_finite_gradients and not gradient_metrics[
            "gradients_finite"
        ]:
            raise RuntimeError("Gradient gate failed: non-finite gradient detected")
        if (
            self.require_nonzero_gradients
            and int(gradient_metrics["grad_nonzero_count"]) == 0
        ):
            raise RuntimeError("Gradient gate failed: all gradients are zero")
        optimizer.step()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "reward_mean": float(rewards.weighted_total.mean().cpu()),
            "reward_std": float(rewards.weighted_total.std(unbiased=False).cpu()),
            "approx_kl": float(loss_info["approx_kl"].detach().cpu()),
            "clipfrac": float(loss_info["clipfrac"].detach().cpu()),
            **logprob_metrics,
            **gradient_metrics,
            **extra_loss_metrics,
            **advantage_result.metrics,
        }
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {"advantage": self.advantage_computer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.advantage_computer.load_state_dict(dict(state.get("advantage") or {}))
