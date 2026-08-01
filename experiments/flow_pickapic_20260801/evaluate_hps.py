"""Run and compare the frozen paired HPS evaluation for Flow/Pick-a-Pic.

This is an experiment-only, read-only evaluator. Training still goes through
the sole public ``visual_rl.load(...).run(...)`` path in ``run_with_api.py``.
The evaluator reuses the builtin SD3 adapter and reward client but never
constructs an optimizer or mutates a training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any

import visual_rl as vr

EVAL_SEEDS = (1009, 2027)
EVAL_BATCH_SIZE = 8
BOOTSTRAP_SEED = 729
BOOTSTRAP_REPLICATES = 10_000


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_prompts(path: Path) -> tuple[str, ...]:
    prompts = tuple(path.read_text(encoding="utf-8").splitlines())
    if len(prompts) != 64 or len(set(prompts)) != 64 or any(not p for p in prompts):
        raise ValueError("heldout prompt file must contain 64 unique non-empty rows")
    return prompts


def _checkpoint_identity(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    path = path.expanduser().resolve(strict=True)
    expected = {"adapter.json", "adapter_state.pt"}
    children = {item.name for item in path.iterdir()}
    if children != expected:
        raise ValueError(
            "adapter checkpoint must contain exactly adapter.json and "
            "adapter_state.pt"
        )
    return {
        "path": str(path),
        "adapter_json_sha256": _sha256_file(path / "adapter.json"),
        "adapter_state_sha256": _sha256_file(path / "adapter_state.pt"),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata(
    prompts: tuple[str, ...],
    *,
    start_index: int,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "dataset_epoch": 0,
            "dataset_index": start_index + offset,
            "prompt_id": _sha256_bytes(prompt.encode()),
            "group_id": f"heldout-{start_index + offset:03d}",
            "evaluation": "flow_pickapic_hps_v1",
        }
        for offset, prompt in enumerate(prompts)
    )


def _save_image(tensor: Any, path: Path) -> str:
    import numpy as np
    import torch
    from PIL import Image

    image = (
        tensor.detach()
        .to(device="cpu", dtype=None)
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .numpy()
    )
    payload = Image.fromarray(np.asarray(image), mode="RGB")
    payload.save(path, format="PNG", optimize=False)
    return _sha256_file(path)


def _validation_errors(report: Any) -> list[dict[str, str]]:
    return [
        {"code": item.code, "path": item.path, "message": item.message}
        for item in report.errors
    ]


def run_evaluation(args: argparse.Namespace) -> None:
    import torch

    from visual_rl.core.types import RuntimeBuildContext, StepContext
    from visual_rl.feedback.world_r1_rewards import WorldR1RewardGeneralClient
    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter
    from visual_rl.rollout.full_trajectory import FullTrajectoryRollout

    config_path = args.config.expanduser().resolve(strict=True)
    prompt_path = args.prompts.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    checkpoint = (
        None
        if args.adapter_checkpoint is None
        else args.adapter_checkpoint.expanduser().resolve(strict=True)
    )
    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    images_dir = output_dir / "images"
    images_dir.mkdir()

    experiment = vr.load(config_path)
    report = experiment.validate()
    if not report.ok:
        raise RuntimeError(
            "evaluation preflight failed: "
            + json.dumps(_validation_errors(report), sort_keys=True)
        )
    config = experiment.resolve()
    if config.model.name != "sd3_tempflow":
        raise ValueError("paired HPS evaluation requires model=sd3_tempflow")
    if config.rollout.name != "full_trajectory":
        raise ValueError("paired HPS evaluation requires full_trajectory rollout")
    if [item.name for item in config.reward.components] != ["reward_general"]:
        raise ValueError("paired HPS evaluation requires only reward_general")

    prompts = _read_prompts(prompt_path)
    checkpoint_identity = _checkpoint_identity(checkpoint)
    condition = "base" if checkpoint is None else "trained"
    manifest = {
        "schema_version": 1,
        "protocol": "flow_pickapic_paired_hps_v1",
        "condition": condition,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "prompt_path": str(prompt_path),
        "prompt_sha256": _sha256_file(prompt_path),
        "prompt_count": len(prompts),
        "eval_seeds": list(EVAL_SEEDS),
        "batch_size": EVAL_BATCH_SIZE,
        "num_diffusion_steps": int(config.rollout.params["num_steps"]),
        "precision": config.runtime.precision,
        "adapter_checkpoint": checkpoint_identity,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)

    runtime_context = RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cuda"),
        precision=config.runtime.precision,
    )
    torch.cuda.set_device(runtime_context.device)
    torch.cuda.reset_peak_memory_stats(runtime_context.device)
    torch.manual_seed(0)

    adapter = None
    reward_client = None
    records: list[dict[str, object]] = []
    try:
        adapter = SD3TempFlowAdapter.from_config(
            config.model.params,
            runtime_context,
        )
        if checkpoint is not None:
            adapter.load_checkpoint(checkpoint)
        rollout = FullTrajectoryRollout(
            num_steps=int(config.rollout.params["num_steps"]),
            samples_per_prompt=1,
        )
        reward_client = WorldR1RewardGeneralClient.from_config(
            config.reward.components[0].params,
            runtime_context,
        )
        records_path = output_dir / "scores.jsonl"
        with records_path.open("x", encoding="utf-8") as handle:
            step = 0
            for eval_seed in EVAL_SEEDS:
                for start in range(0, len(prompts), EVAL_BATCH_SIZE):
                    prompt_batch = prompts[start : start + EVAL_BATCH_SIZE]
                    context = StepContext(
                        step=step,
                        seed=eval_seed + start,
                        rank=0,
                        world_size=1,
                    )
                    batch = rollout.sample(
                        adapter=adapter,
                        prompts=prompt_batch,
                        metadata=_metadata(prompt_batch, start_index=start),
                        context=context,
                    )
                    rewards = reward_client.score(batch, context)
                    for offset, reward in enumerate(rewards.values.tolist()):
                        prompt_index = start + offset
                        image_path = images_dir / (
                            f"seed_{eval_seed}_prompt_{prompt_index:03d}.png"
                        )
                        record = {
                            "condition": condition,
                            "eval_seed": eval_seed,
                            "batch_seed": context.seed,
                            "prompt_index": prompt_index,
                            "prompt_sha256": _sha256_bytes(
                                prompt_batch[offset].encode()
                            ),
                            "sample_id": batch.sample_id[offset],
                            "reward": float(reward),
                            "image": str(image_path.relative_to(output_dir)),
                            "image_sha256": _save_image(
                                batch.media[offset],
                                image_path,
                            ),
                        }
                        records.append(record)
                        handle.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(
                        json.dumps(
                            {
                                "condition": condition,
                                "completed": len(records),
                                "target": len(prompts) * len(EVAL_SEEDS),
                                "last_batch_reward_mean": statistics.fmean(
                                    float(item) for item in rewards.values.tolist()
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    step += 1
    finally:
        if reward_client is not None:
            reward_client.close()
        if adapter is not None:
            adapter.close()

    values = [float(record["reward"]) for record in records]
    if len(values) != len(prompts) * len(EVAL_SEEDS):
        raise RuntimeError("evaluation did not produce the complete score matrix")
    summary = {
        "schema_version": 1,
        "protocol": manifest["protocol"],
        "condition": condition,
        "sample_count": len(values),
        "reward_mean": statistics.fmean(values),
        "reward_std": statistics.pstdev(values),
        "reward_min": min(values),
        "reward_max": max(values),
        "gpu_peak_memory_mib": int(
            torch.cuda.max_memory_allocated(runtime_context.device) / (1024**2)
        ),
        "scores_sha256": _sha256_file(output_dir / "scores.jsonl"),
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps({"completed_evaluation": summary}, sort_keys=True), flush=True)


def _load_records(root: Path) -> tuple[dict[str, object], ...]:
    root = root.expanduser().resolve(strict=True)
    rows = tuple(
        json.loads(line)
        for line in (root / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    )
    if len(rows) != 64 * len(EVAL_SEEDS):
        raise ValueError("evaluation must contain exactly 128 score rows")
    keys = tuple((int(row["eval_seed"]), int(row["prompt_index"])) for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("evaluation contains duplicate paired keys")
    return rows


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compare_evaluations(args: argparse.Namespace) -> None:
    base_rows = _load_records(args.base_dir)
    trained_rows = _load_records(args.trained_dir)
    base = {
        (int(row["eval_seed"]), int(row["prompt_index"])): row
        for row in base_rows
    }
    trained = {
        (int(row["eval_seed"]), int(row["prompt_index"])): row
        for row in trained_rows
    }
    if set(base) != set(trained):
        raise ValueError("base and trained evaluations have different paired keys")

    deltas_by_prompt: dict[int, list[float]] = {}
    pairs: list[dict[str, object]] = []
    for key in sorted(base):
        left = base[key]
        right = trained[key]
        for identity in ("prompt_sha256", "sample_id"):
            if left[identity] != right[identity]:
                raise ValueError(f"paired evaluation mismatch for {identity}: {key}")
        delta = float(right["reward"]) - float(left["reward"])
        deltas_by_prompt.setdefault(key[1], []).append(delta)
        pairs.append(
            {
                "eval_seed": key[0],
                "prompt_index": key[1],
                "base_reward": float(left["reward"]),
                "trained_reward": float(right["reward"]),
                "delta": delta,
            }
        )

    prompt_deltas = {
        prompt_index: statistics.fmean(values)
        for prompt_index, values in deltas_by_prompt.items()
    }
    prompt_ids = sorted(prompt_deltas)
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap = sorted(
        statistics.fmean(prompt_deltas[index] for index in rng.choices(prompt_ids, k=64))
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    mean_delta = statistics.fmean(prompt_deltas.values())
    win_rate = statistics.fmean(
        1.0 if value > 0.0 else 0.0 for value in prompt_deltas.values()
    )
    ci_low = _quantile(bootstrap, 0.025)
    ci_high = _quantile(bootstrap, 0.975)
    passed = ci_low > 0.0 and win_rate > 0.5
    result = {
        "schema_version": 1,
        "protocol": "flow_pickapic_paired_hps_v1",
        "pair_count": len(pairs),
        "prompt_count": len(prompt_deltas),
        "eval_seeds": list(EVAL_SEEDS),
        "mean_paired_delta": mean_delta,
        "median_prompt_delta": statistics.median(prompt_deltas.values()),
        "prompt_win_rate": win_rate,
        "cluster_bootstrap_95_ci": [ci_low, ci_high],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "acceptance": {
            "lower_95_ci_gt_zero": ci_low > 0.0,
            "prompt_win_rate_gt_half": win_rate > 0.5,
            "passed": passed,
        },
        "pairs": pairs,
    }
    output = args.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output, result)
    print(json.dumps({"comparison": result["acceptance"]}, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--adapter-checkpoint", type=Path)
    run.set_defaults(handler=run_evaluation)

    compare = commands.add_parser("compare")
    compare.add_argument("--base-dir", type=Path, required=True)
    compare.add_argument("--trained-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(handler=compare_evaluations)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
