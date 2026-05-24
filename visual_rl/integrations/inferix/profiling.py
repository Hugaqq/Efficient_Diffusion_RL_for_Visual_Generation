"""Inferix profiling plan helpers."""

from __future__ import annotations

from pathlib import Path

from visual_rl.eval.inferix_backend import InferixEvalPlan, build_inferix_eval_plan


def build_inferix_profiling_plan(
    *,
    checkpoint_path: str,
    output_dir: str,
    prompt: str,
    repo_dir: str | Path = "reference_code/Inferix-main",
    num_output_frames: int = 21,
    num_samples: int = 1,
    seed: int = 0,
) -> InferixEvalPlan:
    return build_inferix_eval_plan(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        prompt=prompt,
        repo_dir=repo_dir,
        task="profile",
        num_output_frames=num_output_frames,
        num_samples=num_samples,
        seed=seed,
        enable_profiling=True,
    )


__all__ = ["build_inferix_profiling_plan"]
