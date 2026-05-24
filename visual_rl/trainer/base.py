"""GenRL-inspired trainer base for VisualRL v0.2."""

from __future__ import annotations

import datetime as _datetime
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from visual_rl.configs.schema import VisualRLConfig, config_to_dict
from visual_rl.core.seed import seed_everything


class BaseTrainer(ABC):
    """Shared runtime skeleton for concrete trainers.

    This is intentionally lightweight in v0.2: it provides deterministic paths,
    config validation, optimizer construction, checkpoint metadata, and JSONL
    logging without requiring Accelerate/FSDP at import time.
    """

    def __init__(self, config: VisualRLConfig):
        self.config = config
        self._validate_config()
        seed_everything(config.seed)
        self.output_dir = self._setup_output_dir()
        self.global_step = 0

    def _validate_config(self) -> None:
        errors = []
        if not self.config.rewards.weights:
            errors.append("rewards.weights cannot be empty")
        if self.config.sample.sde_window_range and self.config.sample.sde_window_size:
            start, end = self.config.sample.sde_window_range
            if start < 0 or end < start:
                errors.append("sample.sde_window_range must be [start, end] with 0 <= start <= end")
            if end > self.config.sample.num_steps:
                errors.append("sample.sde_window_range cannot exceed sample.num_steps")
            if end - start < self.config.sample.sde_window_size:
                errors.append("sample.sde_window_range span must be >= sample.sde_window_size")
        if self.config.algorithm.weight_advantages and not self.config.per_prompt_stat_tracking:
            # This mode can work without per-prompt stats, but v0.2 keeps it explicit.
            pass
        if errors:
            raise ValueError("; ".join(errors))

    def _setup_output_dir(self) -> Path:
        base = Path(self.config.output_dir or self.config.paths.output_dir)
        deterministic = self.config.trainer.get("deterministic_run_dir", True)
        if deterministic:
            output_dir = base
        else:
            stamp = _datetime.datetime.now(tz=_datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = base / f"{self.config.run_name}_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir = str(output_dir)
        self.config.paths.output_dir = str(output_dir)
        self.config.paths.save_dir = self.config.paths.save_dir or str(output_dir)
        with (output_dir / "config.resolved.json").open("w", encoding="utf-8") as handle:
            json.dump(config_to_dict(self.config), handle, indent=2, sort_keys=True)
        return output_dir

    def calculate_gradient_accumulation_steps(self, num_train_timesteps: int) -> tuple[int, int]:
        base_gas = self.config.train.gradient_accumulation_steps
        if base_gas is None or base_gas <= 0:
            total_chunks = self.config.sample.num_batches_per_epoch
            base_gas = total_chunks // 2 if total_chunks > 1 else 1
        self.config.train.gradient_accumulation_steps = base_gas
        return base_gas, base_gas * max(1, num_train_timesteps)

    def get_train_timesteps(self, num_train_timesteps: int) -> list[int]:
        if self.config.sample.sde_window_size and self.config.sample.sde_window_range:
            start, _end = self.config.sample.sde_window_range
            return list(range(start, start + num_train_timesteps))
        return list(range(num_train_timesteps))

    def setup_optimizer(self, parameters) -> Any:
        import torch

        optimizer_cls = torch.optim.AdamW
        return optimizer_cls(
            parameters,
            lr=float(self.config.train.learning_rate),
            betas=(float(self.config.train.adam_beta1), float(self.config.train.adam_beta2)),
            weight_decay=float(self.config.train.adam_weight_decay),
            eps=float(self.config.train.adam_epsilon),
        )

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError
