"""GenRL-style Wan trainer planning layer.

This module is intentionally import-light. It captures the runtime contract for
the upcoming real Wan trainer without importing diffusers, accelerate, or CUDA
components during local smoke tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from visual_rl.advantages import AdvantageComputer
from visual_rl.configs.schema import VisualRLConfig, section_to_dict
from visual_rl.datasets.prompt_dataset import PromptDataset
from visual_rl.rewards.router import RewardRouter
from visual_rl.rollout.cache import RolloutCache
from visual_rl.third_party.legacy import resolve_legacy_repo
from visual_rl.trainer.base import BaseTrainer


@dataclass
class WanRuntimePlan:
    trainer: str
    model_name: str
    model_family: str
    model_path: str
    output_dir: str
    genrl_root: str
    world_r1_root: str
    prompt_count: int
    sample: dict[str, Any]
    train: dict[str, Any]
    algorithm: dict[str, Any]
    reward_weights: dict[str, float]
    train_timesteps: list[int]
    gradient_accumulation_steps: int
    effective_gradient_accumulation_steps: int
    readiness: dict[str, bool]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WanTrainer(BaseTrainer):
    """Runtime shell for the GenRL-style Wan baseline.

    `build_runtime_plan()` is the current local validation path. Real training
    remains deferred until model checkpoints, reward services, and GPU runtime
    are available.
    """

    def __init__(self, config: VisualRLConfig):
        super().__init__(config)
        self.dataset = PromptDataset.from_config(config.dataset)
        self.reward_router = RewardRouter(config.rewards, cache_dir=self.output_dir / "reward_cache")
        self.rollout_cache = RolloutCache(self.output_dir / "rollouts")
        self.advantage_computer = AdvantageComputer(
            reward_weights=config.rewards.weights,
            per_prompt=config.per_prompt_stat_tracking,
            weight_advantages=config.algorithm.weight_advantages,
            use_global_std=config.sample.global_std,
            max_group_std=config.sample.max_group_std,
            mode=config.algorithm.advantage_mode,
        )

    def _reference_roots(self) -> tuple[Path, Path]:
        genrl_root = self.config.legacy.get("genrl_root", "reference_code/GenRL-main")
        world_r1_root = self.config.legacy.get("world_r1_root", "reference_code/World-R1-main")
        return resolve_legacy_repo(genrl_root), resolve_legacy_repo(world_r1_root)

    def build_runtime_plan(self) -> WanRuntimePlan:
        sample = section_to_dict(self.config.sample)
        train = section_to_dict(self.config.train)
        algorithm = section_to_dict(self.config.algorithm)
        model = section_to_dict(self.config.model)
        genrl_root, world_r1_root = self._reference_roots()

        num_train_timesteps = max(1, int(self.config.sample.num_steps * self.config.train.timestep_fraction))
        gas, effective_gas = self.calculate_gradient_accumulation_steps(num_train_timesteps)
        train_timesteps = self.get_train_timesteps(num_train_timesteps)

        readiness = {
            "genrl_root_exists": genrl_root.exists(),
            "world_r1_root_exists": world_r1_root.exists(),
            "model_path_set": bool(self.config.model.model_path),
            "mock_rewards_only": set(self.config.rewards.weights) == {"mock"},
        }
        warnings: list[str] = []
        if not readiness["model_path_set"]:
            warnings.append("model.model_path is empty; this plan is local-only and cannot launch real Wan training.")
        if readiness["mock_rewards_only"]:
            warnings.append("only mock reward is configured; replace rewards before real training.")
        if not readiness["genrl_root_exists"]:
            warnings.append(f"GenRL reference root is missing: {genrl_root}")
        if not readiness["world_r1_root_exists"]:
            warnings.append(f"World-R1 reference root is missing: {world_r1_root}")

        return WanRuntimePlan(
            trainer="wan",
            model_name=str(model.get("name", "")),
            model_family=str(model.get("model_family", "")),
            model_path=str(model.get("model_path", "")),
            output_dir=str(self.output_dir),
            genrl_root=str(genrl_root),
            world_r1_root=str(world_r1_root),
            prompt_count=len(self.dataset),
            sample=sample,
            train=train,
            algorithm=algorithm,
            reward_weights=dict(self.config.rewards.weights),
            train_timesteps=train_timesteps,
            gradient_accumulation_steps=gas,
            effective_gradient_accumulation_steps=effective_gas,
            readiness=readiness,
            warnings=warnings,
        )

    def train(self, *args, **kwargs):
        dry_run = bool(kwargs.pop("dry_run", True))
        if args or kwargs:
            raise TypeError("WanTrainer.train currently accepts only dry_run=True/False")
        if dry_run:
            return self.build_runtime_plan().to_dict()
        raise NotImplementedError(
            "Real Wan training is not wired yet. Next step is binding GenRL sampling/logprob "
            "and FSDP checkpointing behind this runtime contract."
        )
