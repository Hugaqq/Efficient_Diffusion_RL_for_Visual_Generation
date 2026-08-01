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
EVALUATION_PROTOCOL = "flow_pickapic_paired_hps_v1"
_SHARED_MANIFEST_IDENTITY = (
    "protocol",
    "prompt_sha256",
    "prompt_count",
    "eval_seeds",
    "batch_size",
    "num_diffusion_steps",
    "precision",
    "config_sha256",
)


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
            "adapter checkpoint must contain exactly adapter.json and adapter_state.pt"
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


def _write_json_exclusive(path: Path, payload: object) -> None:
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
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


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
    payload = Image.fromarray(np.asarray(image))
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
    reward_component = config.reward.components[0]
    manifest = {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
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
        "reward_general": {
            "name": reward_component.name,
            "params": {
                "server_revision": reward_component.params["server_revision"],
            },
        },
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)

    runtime_context = RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cuda", 0),
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


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"evaluation {label} must be a JSON object")
    return payload


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"evaluation {field} must be a lowercase SHA-256")
    return value


def _reward_general_revision(manifest: dict[str, object]) -> str | None:
    if "reward_general" not in manifest:
        return None
    reward = manifest["reward_general"]
    if not isinstance(reward, dict):
        raise TypeError("evaluation manifest reward_general must be an object")
    if reward.get("name") != "reward_general":
        raise ValueError(
            "evaluation manifest reward_general.name must be 'reward_general'"
        )
    params = reward.get("params")
    if not isinstance(params, dict):
        raise TypeError("evaluation manifest reward_general.params must be an object")
    revision = params.get("server_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError(
            "evaluation manifest reward_general.params.server_revision must "
            "be non-empty"
        )
    return revision


def _load_manifest(
    root: Path,
    *,
    expected_condition: str,
) -> tuple[Path, dict[str, object]]:
    if expected_condition not in {"base", "trained"}:
        raise ValueError("expected_condition must be base or trained")
    root = root.expanduser().resolve(strict=True)
    manifest = _read_json_object(root / "manifest.json", label="manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("evaluation manifest schema_version must be 1")
    if manifest.get("protocol") != EVALUATION_PROTOCOL:
        raise ValueError(
            f"evaluation manifest protocol must be {EVALUATION_PROTOCOL!r}"
        )
    if manifest.get("condition") != expected_condition:
        raise ValueError(
            f"evaluation manifest condition must be {expected_condition!r}"
        )

    _require_sha256(manifest.get("config_sha256"), field="config_sha256")
    _require_sha256(manifest.get("prompt_sha256"), field="prompt_sha256")
    if manifest.get("prompt_count") != 64:
        raise ValueError("evaluation manifest prompt_count must be 64")
    if manifest.get("eval_seeds") != list(EVAL_SEEDS):
        raise ValueError("evaluation manifest eval_seeds do not match protocol")
    if manifest.get("batch_size") != EVAL_BATCH_SIZE:
        raise ValueError("evaluation manifest batch_size does not match protocol")
    num_diffusion_steps = manifest.get("num_diffusion_steps")
    if type(num_diffusion_steps) is not int or num_diffusion_steps <= 0:
        raise ValueError(
            "evaluation manifest num_diffusion_steps must be a positive integer"
        )
    precision = manifest.get("precision")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("evaluation manifest precision is invalid")

    checkpoint = manifest.get("adapter_checkpoint")
    if expected_condition == "base":
        if checkpoint is not None:
            raise ValueError("base evaluation manifest adapter_checkpoint must be null")
    else:
        if not isinstance(checkpoint, dict):
            raise ValueError(
                "trained evaluation manifest adapter_checkpoint must be non-null"
            )
        _require_sha256(
            checkpoint.get("adapter_json_sha256"),
            field="adapter_checkpoint.adapter_json_sha256",
        )
        _require_sha256(
            checkpoint.get("adapter_state_sha256"),
            field="adapter_checkpoint.adapter_state_sha256",
        )
    _reward_general_revision(manifest)

    summary = _read_json_object(root / "summary.json", label="summary")
    if summary.get("schema_version") != 1:
        raise ValueError("evaluation summary schema_version must be 1")
    if summary.get("protocol") != manifest["protocol"]:
        raise ValueError("evaluation summary protocol does not match manifest")
    if summary.get("condition") != expected_condition:
        raise ValueError("evaluation summary condition does not match manifest")
    expected_sample_count = int(manifest["prompt_count"]) * len(EVAL_SEEDS)
    if summary.get("sample_count") != expected_sample_count:
        raise ValueError("evaluation summary sample_count does not match manifest")
    recorded_scores_sha256 = _require_sha256(
        summary.get("scores_sha256"),
        field="summary.scores_sha256",
    )
    actual_scores_sha256 = _sha256_file(root / "scores.jsonl")
    if recorded_scores_sha256 != actual_scores_sha256:
        raise ValueError("evaluation scores SHA-256 does not match summary")
    return root, manifest


def _validate_manifest_pair(
    base: dict[str, object],
    trained: dict[str, object],
) -> None:
    for field in _SHARED_MANIFEST_IDENTITY:
        if base[field] != trained[field]:
            raise ValueError(
                f"base and trained evaluation manifests differ for {field}"
            )

    base_revision = _reward_general_revision(base)
    trained_revision = _reward_general_revision(trained)
    if (base_revision is None) != (trained_revision is None):
        raise ValueError("base and trained scorer identity availability differs")
    if base_revision != trained_revision:
        raise ValueError("base and trained reward_general server_revision differs")


def _load_records(
    root: Path,
    *,
    expected_condition: str,
) -> tuple[dict[str, object], ...]:
    root = root.expanduser().resolve(strict=True)
    rows = tuple(
        json.loads(line)
        for line in (root / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    )
    if len(rows) != 64 * len(EVAL_SEEDS):
        raise ValueError("evaluation must contain exactly 128 score rows")
    if expected_condition not in {"base", "trained"}:
        raise ValueError("expected_condition must be base or trained")

    keys: list[tuple[int, int]] = []
    prompt_hashes: dict[int, str] = {}
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"evaluation row {row_number} must be a JSON object")
        condition = row.get("condition")
        if condition != expected_condition:
            raise ValueError(
                f"evaluation row {row_number} condition must be {expected_condition!r}"
            )
        eval_seed = row.get("eval_seed")
        prompt_index = row.get("prompt_index")
        if type(eval_seed) is not int or type(prompt_index) is not int:
            raise ValueError(
                f"evaluation row {row_number} paired key must contain integers"
            )
        prompt_sha256 = row.get("prompt_sha256")
        sample_id = row.get("sample_id")
        if not isinstance(prompt_sha256, str) or not prompt_sha256:
            raise ValueError(
                f"evaluation row {row_number} prompt_sha256 must be non-empty"
            )
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"evaluation row {row_number} sample_id must be non-empty")
        reward = row.get("reward")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise ValueError(
                f"evaluation row {row_number} reward must be finite numeric"
            )
        previous_hash = prompt_hashes.setdefault(prompt_index, prompt_sha256)
        if previous_hash != prompt_sha256:
            raise ValueError(
                "evaluation prompt_sha256 differs across seeds for prompt "
                f"{prompt_index}"
            )
        keys.append((eval_seed, prompt_index))

    actual_keys = set(keys)
    if len(actual_keys) != len(keys):
        raise ValueError("evaluation contains duplicate paired keys")
    expected_keys = {
        (eval_seed, prompt_index)
        for eval_seed in EVAL_SEEDS
        for prompt_index in range(64)
    }
    if actual_keys != expected_keys:
        raise ValueError(
            "evaluation paired keys do not match the frozen seed/prompt grid"
        )
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
    base_root, base_manifest = _load_manifest(
        args.base_dir,
        expected_condition="base",
    )
    trained_root, trained_manifest = _load_manifest(
        args.trained_dir,
        expected_condition="trained",
    )
    _validate_manifest_pair(base_manifest, trained_manifest)
    base_rows = _load_records(base_root, expected_condition="base")
    trained_rows = _load_records(
        trained_root,
        expected_condition="trained",
    )
    base = {(int(row["eval_seed"]), int(row["prompt_index"])): row for row in base_rows}
    trained = {
        (int(row["eval_seed"]), int(row["prompt_index"])): row for row in trained_rows
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
        statistics.fmean(
            prompt_deltas[index] for index in rng.choices(prompt_ids, k=64)
        )
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
    _write_json_exclusive(output, result)
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
