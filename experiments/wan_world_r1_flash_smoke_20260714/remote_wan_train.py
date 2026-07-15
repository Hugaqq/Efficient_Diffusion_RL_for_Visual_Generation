"""Run one real-Wan training step and enforce bounded integration gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bundle_sha256(named_tensors) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_checkpoint(checkpoint: Path) -> dict[str, object]:
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    manifest = metadata.get("artifact_manifest") or {}
    actual_files = {
        str(path.relative_to(checkpoint))
        for path in checkpoint.rglob("*")
        if path.is_file() and path.name != "checkpoint.json"
    }
    entries_valid = set(manifest) == actual_files
    for relative, expected in manifest.items():
        path = checkpoint / relative
        entries_valid = entries_valid and path.is_file()
        if path.is_file():
            entries_valid = entries_valid and path.stat().st_size == int(
                expected["size_bytes"]
            )
            entries_valid = entries_valid and file_sha256(path) == expected["sha256"]
    files = {
        relative: (checkpoint / relative).stat().st_size
        for relative in sorted(actual_files)
    }
    full_transformer = any(
        "transformer_state.pt" in name or size > 1_000_000_000
        for name, size in files.items()
    )
    return {
        "valid": bool(entries_valid and not full_transformer),
        "format_version": metadata.get("format_version"),
        "files": files,
        "total_bytes": sum(files.values()),
        "full_transformer_present": full_transformer,
    }


def write_summary(output: Path, payload: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "summary.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expect-update",
        choices=("active", "zero"),
        default="active",
        help="Require either a non-zero LoRA update or an exactly-zero control update.",
    )
    parser.add_argument(
        "--expected-steps",
        type=int,
        default=1,
        help="Require this many completed training steps and the matching final checkpoint.",
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Record call counts and wall time for each runner stage.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        import torch

        from visual_rl import ExperimentRunner, load_config
        from visual_rl.artifacts.status import inspect_run_status

        if args.output.exists():
            raise FileExistsError(f"refusing to reuse output directory: {args.output}")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        config = load_config(args.config)
        config.paths.output_dir = str(args.output)
        config.runner.deterministic_run_dir = True
        config.runner.show_progress = False
        load_started = time.perf_counter()
        runner = ExperimentRunner(config)
        load_seconds = time.perf_counter() - load_started
        stage_profile: dict[str, dict[str, float | int]] = {}

        def profile_method(owner, method_name: str, stage_name: str) -> None:
            original = getattr(owner, method_name)
            stage_profile[stage_name] = {
                "calls": 0,
                "total_seconds": 0.0,
                "max_seconds": 0.0,
            }

            def measured(*method_args, **method_kwargs):
                stage_started = time.perf_counter()
                try:
                    return original(*method_args, **method_kwargs)
                finally:
                    elapsed = time.perf_counter() - stage_started
                    row = stage_profile[stage_name]
                    row["calls"] = int(row["calls"]) + 1
                    row["total_seconds"] = float(row["total_seconds"]) + elapsed
                    row["max_seconds"] = max(float(row["max_seconds"]), elapsed)

            setattr(owner, method_name, measured)

        if args.profile_stages:
            for owner, method_name, stage_name in (
                (runner.rollout, "sample", "rollout"),
                (runner.feedback_provider, "score", "reward"),
                (runner.rollout_cache, "save", "rollout_cache"),
                (runner.optimizer_plugin, "step", "recompute_backward_optimizer"),
                (runner, "_save_checkpoint", "checkpoint_save"),
                (runner.artifacts, "record", "artifact_record"),
                (runner, "_commit_checkpoint", "checkpoint_publish"),
            ):
                profile_method(owner, method_name, stage_name)
        initial = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in runner.adapter.named_parameters()
        }
        initial_hash = tensor_bundle_sha256(runner.adapter.named_parameters())
        train_started = time.perf_counter()
        metrics = runner.run()
        train_seconds = time.perf_counter() - train_started
        final_hash = tensor_bundle_sha256(runner.adapter.named_parameters())
        squared_delta = 0.0
        changed_tensors = 0
        for name, parameter in runner.adapter.named_parameters():
            difference = parameter.detach().float().cpu() - initial[name]
            squared_delta += float(torch.sum(difference.double().square()))
            changed_tensors += int(bool(torch.count_nonzero(difference)))
        delta_l2 = math.sqrt(squared_delta)
        if args.expected_steps < 1:
            raise ValueError("--expected-steps must be at least 1")
        checkpoint = args.output / f"checkpoint_{args.expected_steps:06d}"
        checkpoint_result = verify_checkpoint(checkpoint)
        status = inspect_run_status(args.output / "run_status.json")
        numeric_metrics_finite = all(
            math.isfinite(float(value))
            for metric in metrics
            for value in metric.values()
            if isinstance(value, int | float) and not isinstance(value, bool)
        )
        update_nonzero = bool(
            delta_l2 > 0.0
            and changed_tensors > 0
            and initial_hash != final_hash
        )
        update_exact_zero = bool(
            delta_l2 == 0.0
            and changed_tensors == 0
            and initial_hash == final_hash
        )
        gates = {
            "completed_valid": status.get("observed_state") == "completed"
            and bool(status.get("ready_for_aggregation")),
            "expected_metrics_rows": len(metrics) == args.expected_steps
            and [int(metric.get("step", -1)) for metric in metrics]
            == list(range(args.expected_steps)),
            "configured_steps_match": int(config.train.max_steps)
            == args.expected_steps,
            "metrics_finite": numeric_metrics_finite,
            "gradient_finite": bool(metrics)
            and all(bool(metric.get("gradients_finite")) for metric in metrics),
            "gradient_nonzero": bool(metrics)
            and all(
                int(metric.get("grad_nonzero_count", 0)) > 0
                for metric in metrics
            ),
            "adapter_update_matches_expectation": (
                update_nonzero if args.expect_update == "active" else update_exact_zero
            ),
            "peft_checkpoint_valid": bool(checkpoint_result["valid"]),
        }
        if args.profile_stages:
            expected_calls = {
                "rollout": args.expected_steps,
                "reward": args.expected_steps,
                "rollout_cache": args.expected_steps,
                "recompute_backward_optimizer": args.expected_steps,
                "checkpoint_save": 1,
                "artifact_record": args.expected_steps,
                "checkpoint_publish": 1,
            }
            gates["stage_profile_counts"] = all(
                int(stage_profile[name]["calls"]) == count
                for name, count in expected_calls.items()
            )
            for row in stage_profile.values():
                calls = int(row["calls"])
                row["mean_seconds"] = (
                    float(row["total_seconds"]) / calls if calls else 0.0
                )
        result = {
            "valid": all(gates.values()),
            "gates": gates,
            "config": str(args.config),
            "output": str(args.output),
            "expected_update": args.expect_update,
            "expected_steps": args.expected_steps,
            "stage_profile_enabled": args.profile_stages,
            "stage_profile": stage_profile,
            "timing_seconds": {
                "load": load_seconds,
                "train_and_artifacts": train_seconds,
                "total": time.perf_counter() - started,
            },
            "cuda": {
                "device_name": torch.cuda.get_device_name(0),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "trainable_tensor_count": len(initial),
            "initial_trainable_sha256": initial_hash,
            "final_trainable_sha256": final_hash,
            "changed_trainable_tensor_count": changed_tensors,
            "adapter_delta_l2": delta_l2,
            "adapter_update_nonzero_observed": update_nonzero,
            "adapter_update_exact_zero_observed": update_exact_zero,
            "metrics": metrics,
            "status": status,
            "checkpoint": checkpoint_result,
        }
        write_summary(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    except Exception as exc:  # noqa: BLE001 - preserve real-model failure evidence
        result = {
            "valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
            "traceback": traceback.format_exc(),
        }
        write_summary(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
