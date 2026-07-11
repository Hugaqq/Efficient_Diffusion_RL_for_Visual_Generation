from typing import Any

from visual_rl.core.types import RolloutBatch, RewardBatch
from visual_rl.model_adapters.base import ModelAdapter
from visual_rl.optimizers.base import OptimizerPlugin


class AlgorithmOptimizerPlugin(OptimizerPlugin):
    def __init__(self, algorithm, advantage_computer, optimizer_config=None):
        self.algorithm = algorithm
        self.advantage_computer = advantage_computer
        self.optimizer_config = dict(optimizer_config or {})

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

        extra_loss_metrics = {}

        for key, value in loss_info.items():
            if key in {"approx_kl", "clipfrac", "policy_loss"}:
                continue
            if hasattr(value, "detach"):
                extra_loss_metrics[key] = float(value.detach().cpu())
            elif isinstance(value, (int, float)):
                extra_loss_metrics[key] = float(value)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "reward_mean": float(rewards.weighted_total.mean().cpu()),
            "reward_std": float(rewards.weighted_total.std(unbiased=False).cpu()),
            "approx_kl": float(loss_info["approx_kl"].detach().cpu()),
            "clipfrac": float(loss_info["clipfrac"].detach().cpu()),
            **logprob_metrics,
            **extra_loss_metrics,
            **advantage_result.metrics,
        }
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {"advantage": self.advantage_computer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.advantage_computer.load_state_dict(dict(state.get("advantage") or {}))
