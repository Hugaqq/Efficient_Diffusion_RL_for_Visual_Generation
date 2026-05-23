"""v0.1 trainer: shared loop with mock-reward dry-run support."""

from __future__ import annotations

from typing import Any

from visual_rl.advantages import AdvantageComputer
from visual_rl.algorithms.grpo import GRPOAlgorithm
from visual_rl.configs.schema import VisualRLConfig, section_to_dict
from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.datasets.prompt_dataset import PromptDataset
from visual_rl.rewards.router import RewardRouter
from visual_rl.rollout.cache import RolloutCache
from visual_rl.rollout.full_trajectory import build_rollout_engine
from visual_rl.trainer.base import BaseTrainer
from visual_rl.trainer.checkpoint import save_json
from visual_rl.trainer.logging import JsonlLogger


class VisualRLTrainer(BaseTrainer):
    def __init__(self, config: VisualRLConfig):
        super().__init__(config)

        model_config = section_to_dict(config.model)
        adapter_cls = MODEL_ADAPTERS.get(model_config.get("name", "mock_wan"))
        self.adapter = adapter_cls(model_config)
        self.dataset = PromptDataset.from_config(config.dataset)
        rollout_config = section_to_dict(config.sample)
        rollout_config.update(config.rollout)
        self.rollout = build_rollout_engine(rollout_config)
        self.reward_router = RewardRouter(config.rewards, cache_dir=self.output_dir / "reward_cache")
        self.rollout_cache = RolloutCache(self.output_dir / "rollouts")
        self.algorithm = GRPOAlgorithm.from_config(config.algorithm)
        self.advantage_computer = AdvantageComputer(
            reward_weights=config.rewards.weights,
            per_prompt=config.per_prompt_stat_tracking,
            weight_advantages=config.algorithm.weight_advantages,
            use_global_std=config.sample.global_std,
            max_group_std=config.sample.max_group_std,
            mode=config.algorithm.advantage_mode,
        )
        self.logger = JsonlLogger(self.output_dir / "metrics.jsonl")

    def train(self, max_steps: int | None = None) -> list[dict[str, Any]]:
        max_steps = int(max_steps or self.config.train.max_steps)
        batch_size = int(self.config.sample.batch_size)
        save_every = int(self.config.train.save_every or max_steps)
        optimizer = self.setup_optimizer(self.adapter.parameters())
        all_metrics: list[dict[str, Any]] = []

        for step in range(max_steps):
            epoch_tag = step
            prompts, metadata, _ = self.dataset.batch(step * batch_size, batch_size, epoch_tag=epoch_tag)
            self.rollout.config["epoch_tag"] = epoch_tag
            self.rollout.config["seed"] = self.config.seed + step
            batch = self.rollout.sample(self.adapter, prompts, metadata)
            batch.validate_lightweight()
            rewards = self.reward_router.score(batch.media, batch.prompts, batch.metadata)
            if not rewards.valid_mask.all():
                raise RuntimeError(f"Reward failure at step {step}: {rewards.metadata}")
            self.rollout_cache.save(step, batch, rewards)
            advantage_result = self.advantage_computer.compute(
                batch.prompts,
                rewards.raw,
                rewards.weighted_total,
            )

            new_log_probs = self.adapter.recompute_log_probs(batch)
            loss, loss_info = self.algorithm.compute_loss(batch, advantage_result.advantages, new_log_probs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "reward_mean": float(rewards.weighted_total.mean().cpu()),
                "reward_std": float(rewards.weighted_total.std(unbiased=False).cpu()),
                "approx_kl": float(loss_info["approx_kl"].detach().cpu()),
                "clipfrac": float(loss_info["clipfrac"].detach().cpu()),
                **advantage_result.metrics,
            }
            self.logger.log(metrics)
            all_metrics.append(metrics)
            if (step + 1) % save_every == 0:
                self.adapter.save_pretrained(str(self.output_dir / f"checkpoint_{step + 1:06d}"))
                save_json(self.output_dir / "latest.json", {"step": step + 1})

        return all_metrics
