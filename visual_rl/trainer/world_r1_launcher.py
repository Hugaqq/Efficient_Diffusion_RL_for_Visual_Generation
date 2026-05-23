"""Safe launch-plan builder for the legacy World-R1 baseline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class WorldR1LaunchPlan:
    repo_dir: str
    model_path: str
    train_visible_devices: str
    num_processes: int
    output_root: str
    train_num_steps: int = 2
    train_eval_num_steps: int = 2
    train_num_batches_per_epoch: int = 1
    train_batch_size: int = 1
    train_num_image_per_prompt: int = 1
    train_height: int = 256
    train_width: int = 448
    train_frames: int = 17
    wandb_mode: str = "offline"

    def as_env(self) -> dict[str, str]:
        return {
            "MODEL_PATH": self.model_path,
            "TRAIN_VISIBLE_DEVICES": self.train_visible_devices,
            "NUM_PROCESSES": str(self.num_processes),
            "OUTPUT_ROOT": self.output_root,
            "TRAIN_NUM_STEPS": str(self.train_num_steps),
            "TRAIN_EVAL_NUM_STEPS": str(self.train_eval_num_steps),
            "TRAIN_NUM_BATCHES_PER_EPOCH": str(self.train_num_batches_per_epoch),
            "TRAIN_BATCH_SIZE": str(self.train_batch_size),
            "TRAIN_NUM_IMAGE_PER_PROMPT": str(self.train_num_image_per_prompt),
            "TRAIN_HEIGHT": str(self.train_height),
            "TRAIN_WIDTH": str(self.train_width),
            "TRAIN_FRAMES": str(self.train_frames),
            "WANDB_MODE": self.wandb_mode,
        }

    def command(self) -> list[str]:
        return ["bash", "scripts/run_training.sh"]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["env"] = self.as_env()
        payload["command"] = self.command()
        return payload


def build_world_r1_launch_plan(
    model_path: str,
    repo_dir: str | Path = "World-R1-main",
    train_visible_devices: str = "6,7",
    output_root: str = "runs/world_r1_v01",
    smoke: bool = True,
) -> WorldR1LaunchPlan:
    devices = [item for item in train_visible_devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("train_visible_devices must contain at least one GPU index")

    if smoke:
        return WorldR1LaunchPlan(
            repo_dir=str(repo_dir),
            model_path=model_path,
            train_visible_devices=train_visible_devices,
            num_processes=len(devices),
            output_root=output_root,
        )

    return WorldR1LaunchPlan(
        repo_dir=str(repo_dir),
        model_path=model_path,
        train_visible_devices=train_visible_devices,
        num_processes=len(devices),
        output_root=output_root,
        train_num_steps=50,
        train_eval_num_steps=50,
        train_num_batches_per_epoch=24,
        train_height=480,
        train_width=832,
        train_frames=81,
    )

