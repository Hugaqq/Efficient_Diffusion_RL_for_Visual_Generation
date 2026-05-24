"""Dry-run Inferix eval/preview/profiling plan builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from visual_rl.third_party.legacy import resolve_legacy_repo


SELF_FORCING_CONFIG = "example/self_forcing/configs/self_forcing_dmd.yaml"
SELF_FORCING_DEFAULT_CONFIG = "example/self_forcing/configs/default_config.yaml"
PROFILING_CONFIG = "example/profiling/configs/profiling_config.yaml"


@dataclass
class InferixEvalPlan:
    repo_dir: str
    task: str
    checkpoint_path: str
    output_dir: str
    prompt: str
    config_path: str = SELF_FORCING_CONFIG
    default_config_path: str = SELF_FORCING_DEFAULT_CONFIG
    profiling_config: str = PROFILING_CONFIG
    num_output_frames: int = 21
    num_samples: int = 1
    seed: int = 0
    use_ema: bool = False
    enable_profiling: bool = False
    memory_mode: str | None = "balanced"
    vae_chunk_size: int | None = None
    use_memory_manager: bool = False
    no_decode: bool = False

    def command(self) -> list[str]:
        script = "example/profiling/self_forcing_profiling.py" if self.enable_profiling else "example/self_forcing/run_self_forcing.py"
        command = [
            "python",
            script,
            "--config_path",
            self.config_path,
            "--default_config_path",
            self.default_config_path,
            "--checkpoint_path",
            self.checkpoint_path,
            "--prompt",
            self.prompt,
            "--output_folder",
            self.output_dir,
            "--num_output_frames",
            str(self.num_output_frames),
            "--num_samples",
            str(self.num_samples),
            "--seed",
            str(self.seed),
        ]
        if self.use_ema:
            command.append("--use_ema")
        if self.enable_profiling:
            command.extend(
                [
                    "--enable_profiling",
                    "--profiling_config",
                    self.profiling_config,
                    "--profiling_output_dir",
                    str(Path(self.output_dir) / "profiling"),
                ]
            )
        else:
            if self.memory_mode:
                command.extend(["--memory_mode", self.memory_mode])
            if self.vae_chunk_size is not None:
                command.extend(["--vae_chunk_size", str(self.vae_chunk_size)])
            if self.use_memory_manager:
                command.append("--use_memory_manager")
        return command

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = self.command()
        payload["cwd"] = self.repo_dir
        payload["online_rl_ready"] = False
        payload["notes"] = [
            "Dry-run plan only; VisualRL does not execute Inferix in-process yet.",
            "Use for checkpoint preview/profiling/no-decode eval before any online RL wiring.",
            "Logprob/recompute contracts are still required before Inferix can enter training.",
        ]
        return payload


def build_inferix_eval_plan(
    *,
    checkpoint_path: str,
    output_dir: str,
    prompt: str,
    repo_dir: str | Path = "reference_code/Inferix-main",
    task: str = "preview",
    num_output_frames: int = 21,
    num_samples: int = 1,
    seed: int = 0,
    enable_profiling: bool = False,
    no_decode: bool = False,
) -> InferixEvalPlan:
    if task not in {"preview", "profile", "long_video_eval"}:
        raise ValueError(f"Unknown Inferix eval task {task!r}.")
    resolved_repo_dir = resolve_legacy_repo(repo_dir)
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required for Inferix eval planning.")
    if not output_dir:
        raise ValueError("output_dir is required for Inferix eval planning.")
    if not prompt:
        raise ValueError("prompt is required for Inferix eval planning.")
    if num_output_frames <= 0:
        raise ValueError("num_output_frames must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    return InferixEvalPlan(
        repo_dir=str(resolved_repo_dir),
        task=task,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        prompt=prompt,
        num_output_frames=int(num_output_frames),
        num_samples=int(num_samples),
        seed=int(seed),
        enable_profiling=bool(enable_profiling or task == "profile"),
        no_decode=bool(no_decode),
    )


class InferixEvalBackend:
    """Plan-only adapter for Inferix eval tasks.

    Execution stays out of process until checkpoint loading, no-decode paths,
    and logprob/recompute contracts are explicit.
    """

    def generate_preview(self, *, execute: bool = False, **kwargs) -> dict[str, Any]:
        if execute:
            raise NotImplementedError("Inferix execution is not wired into VisualRL yet; use the returned plan.")
        return build_inferix_eval_plan(task="preview", **kwargs).to_dict()

    def profile_checkpoint(self, *, execute: bool = False, **kwargs) -> dict[str, Any]:
        if execute:
            raise NotImplementedError("Inferix profiling execution is not wired into VisualRL yet; use the returned plan.")
        return build_inferix_eval_plan(task="profile", enable_profiling=True, **kwargs).to_dict()

    def run_long_video_eval(self, *, execute: bool = False, **kwargs) -> dict[str, Any]:
        if execute:
            raise NotImplementedError("Inferix long-video execution is not wired into VisualRL yet; use the returned plan.")
        return build_inferix_eval_plan(task="long_video_eval", **kwargs).to_dict()
