"""Deterministic loss-descent probes for VisualRL infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TinyLossProbeConfig:
    output_dir: str | Path
    steps: int = 100
    learning_rate: float = 0.1
    batch_size: int = 4
    num_steps: int = 4
    image_size: int = 8
    seed: int = 123
    device: str = "cpu"
    target_bias: tuple[float, float, float] = (0.8, -0.4, -0.4)
    max_final_loss_ratio: float = 0.1
    max_final_bias_error_ratio: float = 0.25
    assert_descent: bool = True


def run_tiny_loss_probe(config: TinyLossProbeConfig) -> dict[str, Any]:
    """Fit a tiny student adapter to a teacher rollout and require loss descent.

    This probe is intentionally supervised and deterministic. Online RL losses
    are often zero or non-monotonic when each rollout is collected from the
    current policy, so this fixed-rollout check isolates the core plumbing:
    adapter parameters, rollout tensors, differentiable logprob recomputation,
    optimizer steps, metric logging, and checkpoint writing.
    """

    import torch

    from visual_rl.algorithms.grpo import GRPOAlgorithm
    from visual_rl.core.seed import seed_everything
    from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter
    from visual_rl.trainer.checkpoint import save_json
    from visual_rl.trainer.logging import JsonlLogger

    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = {
        "name": "tiny_diffusion",
        "extra": {"image_size": config.image_size, "device": config.device},
    }
    teacher = TinyDiffusionAdapter(adapter_config)
    student = TinyDiffusionAdapter(adapter_config)
    target_bias = torch.tensor(config.target_bias, device=student.device, dtype=student.color_bias.dtype)
    with torch.no_grad():
        teacher.color_bias.copy_(target_bias)

    prompts = ["a red square" for _ in range(config.batch_size)]
    metadata = [{"source": "tiny_loss_probe", "target_color": "red"} for _ in range(config.batch_size)]
    batch = teacher.sample(prompts, metadata, {"num_steps": config.num_steps, "seed": config.seed})
    batch.validate_strict()

    optimizer = torch.optim.AdamW(student.parameters(), lr=float(config.learning_rate))
    algorithm = GRPOAlgorithm(clip_range=10.0, adv_clip_max=5.0, beta=0.0)
    advantages = torch.ones_like(batch.old_log_probs, device=student.device)
    logger = JsonlLogger(output_dir / "metrics.jsonl")
    metrics: list[dict[str, Any]] = []

    for step in range(int(config.steps) + 1):
        new_log_probs = student.recompute_log_probs(batch)
        old_log_probs = batch.old_log_probs.to(new_log_probs.device, dtype=new_log_probs.dtype)
        fit_loss = 0.5 * ((new_log_probs - old_log_probs) ** 2).mean()
        transition_nll = (-new_log_probs).mean()
        grpo_loss, loss_info = algorithm.compute_loss(batch, advantages, new_log_probs)
        bias_error = torch.sqrt(((student.color_bias - target_bias) ** 2).mean())

        payload = {
            "step": step,
            "loss": float(fit_loss.detach().cpu()),
            "fit_loss": float(fit_loss.detach().cpu()),
            "transition_nll": float(transition_nll.detach().cpu()),
            "grpo_policy_loss": float(grpo_loss.detach().cpu()),
            "approx_kl": float(loss_info["approx_kl"].detach().cpu()),
            "clipfrac": float(loss_info["clipfrac"].detach().cpu()),
            "bias_error": float(bias_error.detach().cpu()),
            "student_color_bias": [float(item) for item in student.color_bias.detach().cpu().tolist()],
            "target_color_bias": [float(item) for item in target_bias.detach().cpu().tolist()],
        }
        logger.log(payload)
        metrics.append(payload)
        if step == int(config.steps):
            break

        optimizer.zero_grad(set_to_none=True)
        fit_loss.backward()
        optimizer.step()

    initial = metrics[0]
    final = metrics[-1]
    summary = {
        "output_dir": str(output_dir),
        "steps": int(config.steps),
        "batch_size": int(config.batch_size),
        "num_steps": int(config.num_steps),
        "seed": int(config.seed),
        "device": str(student.device),
        "loss_start": initial["loss"],
        "loss_end": final["loss"],
        "loss_ratio": final["loss"] / max(initial["loss"], 1e-12),
        "bias_error_start": initial["bias_error"],
        "bias_error_end": final["bias_error"],
        "bias_error_ratio": final["bias_error"] / max(initial["bias_error"], 1e-12),
        "grpo_policy_loss_start": initial["grpo_policy_loss"],
        "grpo_policy_loss_end": final["grpo_policy_loss"],
        "metrics_path": str(output_dir / "metrics.jsonl"),
        "checkpoint_dir": str(output_dir / "checkpoint_final"),
    }
    student.save_pretrained(summary["checkpoint_dir"])
    save_json(output_dir / "summary.json", summary)

    if config.assert_descent:
        if summary["loss_ratio"] > float(config.max_final_loss_ratio):
            raise RuntimeError(
                "Tiny loss probe did not descend enough: "
                f"loss_start={summary['loss_start']:.6g}, loss_end={summary['loss_end']:.6g}, "
                f"ratio={summary['loss_ratio']:.6g}, max_ratio={config.max_final_loss_ratio:.6g}."
            )
        if summary["bias_error_ratio"] > float(config.max_final_bias_error_ratio):
            raise RuntimeError(
                "Tiny loss probe did not fit target bias enough: "
                f"bias_error_start={summary['bias_error_start']:.6g}, "
                f"bias_error_end={summary['bias_error_end']:.6g}, "
                f"ratio={summary['bias_error_ratio']:.6g}, "
                f"max_ratio={config.max_final_bias_error_ratio:.6g}."
            )

    return summary
