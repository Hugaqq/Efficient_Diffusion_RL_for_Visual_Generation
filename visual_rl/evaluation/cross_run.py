"""Aggregate bounded SD3 runs across independent training seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _load_summary(source: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(source, dict):
        return source, "<in-memory>"
    path = Path(source)
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def _contract(summary: dict[str, Any]) -> tuple[Any, ...]:
    prompt_splits = summary.get("prompt_splits", {})
    heldout = prompt_splits.get("heldout", {})
    preview = summary.get("preview_artifacts", {}).get("after", {})
    return (
        summary.get("target_step"),
        heldout.get("content_sha256"),
        tuple(preview.get("seeds", [])),
        preview.get("sample_count"),
        summary.get("model_path"),
        summary.get("resolution"),
        summary.get("num_steps"),
        summary.get("guidance_scale"),
        summary.get("branch_count"),
        summary.get("sample_batch_size"),
    )


def _validate_group(
    loaded: list[tuple[dict[str, Any], str]],
    *,
    condition: str,
    expected_contract: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    if not loaded:
        raise ValueError(f"at least one {condition} summary is required")
    seeds: set[int] = set()
    contract = expected_contract or _contract(loaded[0][0])
    for summary, source in loaded:
        actual_condition = summary.get("condition", "active")
        if actual_condition != condition:
            raise ValueError(
                f"{source} has condition={actual_condition!r}, expected {condition!r}"
            )
        seed = int(summary["seed"])
        if seed in seeds:
            raise ValueError(f"duplicate {condition} training seed: {seed}")
        seeds.add(seed)
        if _contract(summary) != contract:
            raise ValueError(f"{source} does not match the evaluation contract")
        delta = summary.get("heldout_paired_delta")
        if not delta or not delta.get("eval_seed_cluster_means"):
            raise ValueError(f"{source} is missing paired held-out seed clusters")
    return contract


def _hierarchical_bootstrap(
    cluster_rows: list[np.ndarray],
    *,
    rng: np.random.Generator,
    samples: int,
) -> np.ndarray:
    values = np.empty(samples, dtype=np.float64)
    run_count = len(cluster_rows)
    for index in range(samples):
        selected_runs = rng.integers(0, run_count, size=run_count)
        selected_values = []
        for run_index in selected_runs:
            row = cluster_rows[int(run_index)]
            cluster_indices = rng.integers(0, len(row), size=len(row))
            selected_values.extend(row[cluster_indices])
        values[index] = float(np.mean(selected_values))
    return values


def _group_statistics(
    loaded: list[tuple[dict[str, Any], str]],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
    color: str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    run_rows = []
    run_means = {}
    for summary, _ in loaded:
        delta = summary["heldout_paired_delta"]
        if color is not None:
            delta = delta.get("per_color", {}).get(color)
            if not delta:
                raise ValueError(
                    f"training seed {summary['seed']} is missing color {color!r}"
                )
        row = np.asarray(
            list(delta["eval_seed_cluster_means"].values()),
            dtype=np.float64,
        )
        run_rows.append(row)
        run_means[str(int(summary["seed"]))] = float(row.mean())
    bootstrap = _hierarchical_bootstrap(
        run_rows,
        rng=rng,
        samples=bootstrap_samples,
    )
    means = np.asarray(list(run_means.values()), dtype=np.float64)
    cluster_values = np.concatenate(run_rows)
    return (
        {
            "training_seed_count": len(run_rows),
            "training_seed_means": run_means,
            "mean": float(means.mean()),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "positive_training_seed_fraction": float(np.mean(means > 0.0)),
            "eval_seed_cluster_rms": float(
                np.sqrt(np.mean(cluster_values * cluster_values))
            ),
        },
        bootstrap,
    )


def aggregate_sd3_run_summaries(
    active_summaries: Iterable[str | Path | dict[str, Any]],
    control_summaries: Iterable[str | Path | dict[str, Any]] = (),
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    min_training_seeds: int = 3,
    min_positive_fraction: float = 0.8,
) -> dict[str, Any]:
    """Build an effectiveness gate from independent bounded-run summaries."""

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    active = [_load_summary(source) for source in active_summaries]
    controls = [_load_summary(source) for source in control_summaries]
    contract = _validate_group(active, condition="active")
    if controls:
        _validate_group(
            controls,
            condition="zero_lr_control",
            expected_contract=contract,
        )

    rng = np.random.default_rng(bootstrap_seed)
    active_stats, active_bootstrap = _group_statistics(
        active,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
    )
    colors = sorted(
        active[0][0]["heldout_paired_delta"].get("per_color", {})
    )
    per_color = {}
    for color in colors:
        color_stats, _ = _group_statistics(
            active,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
            color=color,
        )
        per_color[color] = color_stats

    execution_valid = all(
        bool(summary.get("valid")) for summary, _ in [*active, *controls]
    )
    pixel_guardrails_valid = all(
        bool(summary.get("gates", {}).get("pixel_diversity_guardrail"))
        for summary, _ in [*active, *controls]
    )
    control_stats = None
    active_minus_control = None
    control_gate = True
    twice_control_noise_gate = True
    if controls:
        control_stats, control_bootstrap = _group_statistics(
            controls,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )
        comparison = active_bootstrap - control_bootstrap
        active_minus_control = {
            "mean": float(active_stats["mean"] - control_stats["mean"]),
            "ci95_low": float(np.quantile(comparison, 0.025)),
            "ci95_high": float(np.quantile(comparison, 0.975)),
        }
        control_gate = active_minus_control["ci95_low"] > 0.0
        twice_control_noise_gate = active_stats["mean"] > (
            2.0 * control_stats["eval_seed_cluster_rms"]
        )

    seed_count_gate = active_stats["training_seed_count"] >= min_training_seeds
    positive_fraction_gate = (
        active_stats["positive_training_seed_fraction"]
        >= min_positive_fraction
    )
    active_ci_gate = active_stats["ci95_low"] > 0.0
    per_color_gate = bool(per_color) and all(
        stats["mean"] > 0.0 for stats in per_color.values()
    )
    gates = {
        "all_runs_execution_valid": execution_valid,
        "all_runs_pixel_guardrails_valid": pixel_guardrails_valid,
        "minimum_training_seed_count": seed_count_gate,
        "positive_training_seed_fraction": positive_fraction_gate,
        "active_hierarchical_ci95_low_positive": active_ci_gate,
        "every_color_mean_positive": per_color_gate,
        "active_exceeds_zero_lr_control": control_gate,
        "active_exceeds_twice_control_noise_rms": twice_control_noise_gate,
    }
    return {
        "method": "hierarchical_bootstrap_training_seed_then_eval_seed",
        "claim_scope": "pre_registered_heldout_rgb_control",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "contract": {
            "target_step": contract[0],
            "heldout_content_sha256": contract[1],
            "eval_seeds": list(contract[2]),
            "sample_count": contract[3],
            "model_path": contract[4],
            "resolution": contract[5],
            "num_steps": contract[6],
            "guidance_scale": contract[7],
            "branch_count": contract[8],
            "sample_batch_size": contract[9],
        },
        "active": active_stats,
        "zero_lr_control": control_stats,
        "active_minus_zero_lr_control": active_minus_control,
        "per_color": per_color,
        "gates": gates,
        "eligible_for_effectiveness_claim": all(gates.values()),
        "source_summaries": {
            "active": [source for _, source in active],
            "zero_lr_control": [source for _, source in controls],
        },
    }
