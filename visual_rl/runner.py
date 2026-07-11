"""Config-driven VisualRL experiment runner and single training loop."""

from __future__ import annotations

import datetime as _datetime
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

from visual_rl.artifacts import ArtifactManager
from visual_rl.artifacts.checkpoint import (
    build_implementation_identity,
    load_json,
    load_training_state,
    save_json,
    save_training_state,
)
from visual_rl.artifacts.logging import TrainProgressPrinter
from visual_rl.builtins import register_builtin_plugins
from visual_rl.configs.schema import VisualRLConfig, config_to_dict, section_to_dict
from visual_rl.core.registry import MODEL_ADAPTERS
from visual_rl.core.seed import seed_everything
from visual_rl.datasets.prompt_dataset import PromptDataset
from visual_rl.feedback import build_feedback_provider
from visual_rl.optimizers import build_optimizer_plugin
from visual_rl.rollout.cache import RolloutCache
from visual_rl.rollout.full_trajectory import build_rollout_engine


class ExperimentRunner:
    """Build components, execute policy updates, and persist a reproducible run."""

    def __init__(self, config: VisualRLConfig):
        self._validate_process_mode()
        register_builtin_plugins()
        self.config = config
        self._validate_config()
        seed_everything(config.seed)
        self.output_dir = self._setup_output_dir()
        self.global_step = 0

        model_config = section_to_dict(config.model)
        model_config.setdefault("use_lora", config.use_lora)
        model_config.setdefault("lora_path", config.train.lora_path)
        if config.paths.pretrained_model and not model_config.get("model_path"):
            model_config["model_path"] = config.paths.pretrained_model
        adapter_cls = MODEL_ADAPTERS.get(model_config.get("name", "mock_wan"))
        self.adapter = adapter_cls(model_config)
        if (
            config.runner.auto_load_model
            and hasattr(self.adapter, "load")
            and getattr(self.adapter, "pipeline", None) is None
        ):
            self.adapter.load()

        self.dataset = PromptDataset.from_config(config.dataset)
        rollout_config = section_to_dict(config.sample)
        rollout_config.update(config.rollout)
        self.rollout = build_rollout_engine(rollout_config)
        reward_cache_dir = config.rewards.cache_dir or self.output_dir / "reward_cache"
        self.feedback_provider = build_feedback_provider(
            config.rewards,
            cache_dir=reward_cache_dir,
        )
        rollout_cache_dir = None
        if not config.runner.disable_rollout_cache:
            rollout_cache_dir = (
                config.runner.rollout_cache_dir or self.output_dir / "rollouts"
            )
        self.rollout_cache = RolloutCache(rollout_cache_dir)
        self.optimizer_plugin = build_optimizer_plugin(config)
        self.optimizer = self.optimizer_plugin.build_optimizer(
            self.adapter.parameters(),
            config.train,
        )
        self._validate_optimizer_contract(self.optimizer)
        self.checkpoint_identity = build_implementation_identity(
            self.adapter,
            self.optimizer_plugin,
            rollout=self.rollout,
            feedback=self.feedback_provider,
        )
        self.progress = TrainProgressPrinter(
            enabled=config.runner.show_progress,
            interval=config.runner.progress_interval,
            leave=config.runner.progress_leave,
        )
        self.start_step = self._load_resume_if_requested()
        continuing_artifacts = (
            self.start_step > 0
            and (self.output_dir / "sample_manifest.json").exists()
        )
        self.artifacts = ArtifactManager(
            self.output_dir,
            config.run_name,
            config=config_to_dict(config),
            resume=continuing_artifacts,
        )
        if self.start_step > 0:
            self.artifacts.truncate_from_step(self.start_step)
            self.rollout_cache.truncate_from_step(self.start_step)

    def run(self, max_steps: int | None = None) -> list[dict[str, Any]]:
        target_steps = int(
            self.config.train.max_steps if max_steps is None else max_steps
        )
        if target_steps < self.start_step:
            raise ValueError(
                f"max_steps={target_steps} is before resumed step {self.start_step}"
            )
        batch_size = int(self.config.sample.batch_size)
        save_every = int(self.config.train.save_every or max(1, target_steps))
        all_metrics: list[dict[str, Any]] = []

        self.progress.start(target_steps, initial_step=self.start_step)
        try:
            for step in range(self.start_step, target_steps):
                prompts, metadata, epoch_tag = self.dataset.batch(
                    step * batch_size,
                    batch_size,
                    epoch_tag=step,
                )
                self.rollout.config["epoch_tag"] = epoch_tag
                self.rollout.config["seed"] = self.config.seed + step
                batch = self.rollout.sample(self.adapter, prompts, metadata)
                batch.validate_lightweight(
                    strict=self.config.runner.strict_rollout_validation
                )

                rewards = self.feedback_provider.score(batch)
                if not bool(rewards.valid_mask.all()):
                    raise RuntimeError(
                        f"Reward failure at step {step}: {rewards.metadata}"
                    )
                cache_paths = self.rollout_cache.save(step, batch, rewards)
                plugin_metrics = self.optimizer_plugin.step(
                    adapter=self.adapter,
                    batch=batch,
                    rewards=rewards,
                    optimizer=self.optimizer,
                    context={"step": step, "max_steps": target_steps},
                )
                metrics = {"step": step, **plugin_metrics}

                checkpoint_path = None
                checkpoint_metadata = None
                should_checkpoint = (
                    (step + 1) % save_every == 0 or step + 1 == target_steps
                )
                if should_checkpoint:
                    checkpoint_path, checkpoint_metadata = self._save_checkpoint(
                        step + 1
                    )

                self.progress.log_step(metrics, total_steps=target_steps)
                all_metrics.append(metrics)
                self.artifacts.record(
                    step=step,
                    batch=batch,
                    rewards=rewards,
                    metrics=metrics,
                    media_type=self._media_type(),
                    rollout_type=self.config.sample.name,
                    media_paths=cache_paths.get("media_path"),
                    rollout_cache_path=cache_paths.get("rollout_cache_path"),
                    checkpoint_path=checkpoint_path,
                )
                if checkpoint_path is not None and checkpoint_metadata is not None:
                    self._commit_checkpoint(
                        checkpoint_path,
                        checkpoint_metadata,
                    )
                self.global_step = step + 1
        finally:
            self.progress.close()
        return all_metrics

    def _save_checkpoint(self, completed_steps: int) -> tuple[Path, dict[str, Any]]:
        checkpoint_path = self.output_dir / f"checkpoint_{completed_steps:06d}"
        staging_path = self.output_dir / (
            f".{checkpoint_path.name}.tmp-{uuid.uuid4().hex}"
        )
        try:
            self.adapter.save_pretrained(str(staging_path))
            if not staging_path.exists() or not any(
                path.is_file() for path in staging_path.rglob("*")
            ):
                raise RuntimeError(
                    f"Adapter {self.adapter.name!r} did not write checkpoint files"
                )
            metadata = save_training_state(
                staging_path,
                optimizer=self.optimizer,
                plugin=self.optimizer_plugin,
                step=completed_steps,
                config=config_to_dict(self.config),
                implementation=self.checkpoint_identity,
            )
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            staging_path.replace(checkpoint_path)
        except Exception:
            if staging_path.exists():
                shutil.rmtree(staging_path)
            raise
        return checkpoint_path, metadata

    def _commit_checkpoint(
        self,
        checkpoint_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        save_json(
            self.output_dir / "latest.json",
            {
                "step": int(metadata["step"]),
                "checkpoint": checkpoint_path.name,
                "config_fingerprint": metadata["config_fingerprint"],
            },
        )

    def _validate_config(self) -> None:
        errors = []
        if not self.config.rewards.weights:
            errors.append("rewards.weights cannot be empty")
        if self.config.sample.sde_window_range and self.config.sample.sde_window_size:
            start, end = self.config.sample.sde_window_range
            if start < 0 or end < start:
                errors.append(
                    "sample.sde_window_range must be [start, end] with 0 <= start <= end"
                )
            if end > self.config.sample.num_steps:
                errors.append("sample.sde_window_range cannot exceed sample.num_steps")
            if end - start < self.config.sample.sde_window_size:
                errors.append(
                    "sample.sde_window_range span must be >= sample.sde_window_size"
                )
        if errors:
            raise ValueError("; ".join(errors))

    def _setup_output_dir(self) -> Path:
        base = Path(self.config.paths.output_dir)
        if self.config.runner.deterministic_run_dir:
            output_dir = base
        else:
            stamp = _datetime.datetime.now(tz=_datetime.timezone.utc).strftime(
                "%Y%m%d_%H%M%S"
            )
            output_dir = base / f"{self.config.run_name}_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.paths.output_dir = str(output_dir)
        return output_dir

    def _media_type(self) -> str:
        media_type = getattr(self.adapter, "media_type", None)
        if media_type not in {"image", "video"}:
            raise ValueError(
                f"Adapter {self.adapter.name} must declare media_type as 'image' or 'video'."
            )
        return str(media_type)

    def _load_resume_if_requested(self) -> int:
        resume_from = self.config.paths.resume_from
        if not resume_from:
            return 0
        checkpoint_dir, declared_step = self._resolve_resume_checkpoint(resume_from)
        self.adapter.load_checkpoint(str(checkpoint_dir))
        restored_step = load_training_state(
            checkpoint_dir,
            optimizer=self.optimizer,
            plugin=self.optimizer_plugin,
            config=config_to_dict(self.config),
            implementation=self.checkpoint_identity,
        )
        if restored_step != declared_step:
            raise RuntimeError(
                f"Checkpoint step mismatch: path says {declared_step}, state says {restored_step}"
            )
        self.global_step = restored_step
        return restored_step

    @staticmethod
    def _resolve_resume_checkpoint(resume_from: str | Path) -> tuple[Path, int]:
        path = Path(resume_from)
        if path.name == "latest.json":
            latest = load_json(path)
            step = int(latest["step"])
            checkpoint = latest.get("checkpoint", f"checkpoint_{step:06d}")
            return path.parent / checkpoint, step
        if path.is_dir() and (path / "latest.json").exists():
            latest = load_json(path / "latest.json")
            step = int(latest["step"])
            checkpoint = latest.get("checkpoint", f"checkpoint_{step:06d}")
            return path / checkpoint, step
        if path.is_dir() and path.name.startswith("checkpoint_"):
            step = int(path.name.split("_")[-1])
            return path, step
        raise RuntimeError(f"Unsupported resume_from path: {path}")

    @staticmethod
    def _validate_process_mode() -> None:
        raw_world_size = os.environ.get("WORLD_SIZE", "1")
        try:
            world_size = int(raw_world_size)
        except ValueError as exc:
            raise RuntimeError(
                f"WORLD_SIZE must be an integer, got {raw_world_size!r}"
            ) from exc
        if world_size != 1:
            raise RuntimeError(
                "VisualRL v0.6 simplified core supports one process only; "
                f"received WORLD_SIZE={world_size}."
            )

    @staticmethod
    def _validate_optimizer_contract(optimizer: Any) -> None:
        required = ("zero_grad", "step", "state_dict", "load_state_dict")
        missing = [name for name in required if not callable(getattr(optimizer, name, None))]
        if missing:
            raise TypeError(
                "OptimizerPlugin.build_optimizer() returned an object missing "
                f"required methods: {missing}"
            )
