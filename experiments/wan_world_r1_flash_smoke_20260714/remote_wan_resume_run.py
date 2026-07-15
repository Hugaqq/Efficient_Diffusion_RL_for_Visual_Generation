"""Run a deterministic real-Wan continuous, split, or resumed segment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback


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


def materialized_lora_modules(named_parameters) -> tuple[set[str], set[str]]:
    """Return modules with paired PEFT A/B tensors and classified tensor names."""

    module_sides: dict[str, set[str]] = {}
    classified_names: set[str] = set()
    for name, _parameter in named_parameters:
        for marker, side in ((".lora_A.", "A"), (".lora_B.", "B")):
            if marker not in name:
                continue
            module_name = name.split(marker, 1)[0]
            module_sides.setdefault(module_name, set()).add(side)
            classified_names.add(name)
            break
    paired_modules = {
        module_name
        for module_name, sides in module_sides.items()
        if sides == {"A", "B"}
    }
    return paired_modules, classified_names


def trainable_topology(adapter) -> dict:
    """Describe configured and actually materialized LoRA target families."""

    named_parameters = list(adapter.named_parameters())
    configured_targets = list(getattr(adapter, "lora_targets", ()))
    paired_modules, classified_names = materialized_lora_modules(named_parameters)
    modules_by_target = {
        target: sorted(
            module_name
            for module_name in paired_modules
            if module_name == target or module_name.endswith(f".{target}")
        )
        for target in configured_targets
    }
    effective_targets = sorted(
        target for target, module_names in modules_by_target.items() if module_names
    )
    unclassified_names = sorted(
        name for name, _parameter in named_parameters if name not in classified_names
    )
    return {
        "configured_lora_target_families": configured_targets,
        "effective_lora_target_families": effective_targets,
        "effective_lora_module_counts": {
            target: len(modules_by_target[target]) for target in effective_targets
        },
        "trainable_tensor_count": len(named_parameters),
        "trainable_parameter_count": int(
            sum(parameter.numel() for _name, parameter in named_parameters)
        ),
        "unclassified_trainable_parameter_names": unclassified_names,
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
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        import torch

        from visual_rl import ExperimentRunner, load_config
        from visual_rl.artifacts.audit import audit_run_artifacts
        from visual_rl.artifacts.status import inspect_run_status

        if args.output.exists():
            raise FileExistsError(f"refusing to reuse output directory: {args.output}")
        config = load_config(args.config)
        config.paths.output_dir = str(args.output)
        config.paths.resume_from = (
            str(args.resume_from) if args.resume_from is not None else ""
        )
        config.train.max_steps = args.max_steps
        config.train.save_every = 1
        config.runner.deterministic_run_dir = True
        config.runner.deterministic_runtime = True
        config.runner.show_progress = False
        runner = ExperimentRunner(config)
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_topology = trainable_topology(runner.adapter)
        initial_hash = tensor_bundle_sha256(runner.adapter.named_parameters())
        metrics = runner.run()
        final_topology = trainable_topology(runner.adapter)
        final_hash = tensor_bundle_sha256(runner.adapter.named_parameters())
        status = inspect_run_status(args.output / "run_status.json")
        audit = audit_run_artifacts(args.output)
        expected_rows = args.max_steps - runner.start_step
        gates = {
            "completed_status": status.get("observed_state") == "completed",
            "authoritative_marker_valid": bool(status.get("marker_valid")),
            "ready_for_aggregation": bool(status.get("ready_for_aggregation")),
            "artifact_audit_valid": bool(audit.get("valid")),
            "expected_metrics_rows": len(metrics) == expected_rows,
            "absolute_target_reached": status.get("completed_steps")
            == args.max_steps
            and status.get("authoritative_completed_steps") == args.max_steps,
            "all_gradients_finite_nonzero": all(
                bool(row.get("gradients_finite"))
                and int(row.get("grad_nonzero_count", 0)) > 0
                for row in metrics
            ),
            "final_checkpoint_present": (
                args.output / f"checkpoint_{args.max_steps:06d}" / "checkpoint.json"
            ).is_file(),
            "configured_lora_targets_materialized": (
                initial_topology["effective_lora_target_families"]
                == sorted(initial_topology["configured_lora_target_families"])
            ),
            "all_trainable_parameters_are_paired_standard_lora": not initial_topology[
                "unclassified_trainable_parameter_names"
            ],
            "trainable_topology_stable": final_topology == initial_topology,
        }
        result = {
            "valid": all(gates.values()),
            "gates": gates,
            "config": str(args.config),
            "output": str(args.output),
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "start_step": runner.start_step,
            "target_step": args.max_steps,
            "initial_trainable_sha256": initial_hash,
            "final_trainable_sha256": final_hash,
            "initial_trainable_topology": initial_topology,
            "final_trainable_topology": final_topology,
            "metrics": metrics,
            "status": status,
            "artifact_audit": audit,
            "cuda": {
                "device_name": torch.cuda.get_device_name(0),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_summary(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    except Exception as exc:  # noqa: BLE001 - preserve real resume failure
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
