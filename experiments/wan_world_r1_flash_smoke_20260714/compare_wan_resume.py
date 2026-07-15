"""Compare deterministic real-Wan continuous two-step and 1+resume runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np
import torch

from visual_rl.artifacts.audit import audit_run_artifacts
from visual_rl.artifacts.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    checkpoint_tree_sha256,
    load_json as load_secure_json,
    read_and_validate_training_state,
    strict_json_loads,
)
from visual_rl.artifacts.manifest import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
)
from visual_rl.artifacts.status import inspect_run_status


EXPECTED_FINAL_STEP = 2
_TIMING_METRIC_NAMES = frozenset(
    {
        "artifact_cycle_steps",
        "reward_latency_p50_s",
        "reward_latency_p95_s",
        "rollout_time_s",
        "reward_time_s",
        "rollout_cache_time_s",
        "update_time_s",
    }
)
_OBSERVATIONAL_METRIC_NAMES = _TIMING_METRIC_NAMES | {
    # Allocator history and the physical GPU affect this measurement even when
    # the training state, gradients, and adapter update are bitwise identical.
    "peak_gpu_memory_bytes",
}


def load_json(path: Path) -> Any:
    return load_secure_json(path)


def read_regular_text(path: Path) -> str:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("secure harness reads require O_NOFOLLOW")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"harness input must be a regular file: {path}")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            return handle.read()
    finally:
        if fd >= 0:
            os.close(fd)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        strict_json_loads(line)
        for line in read_regular_text(path).splitlines()
        if line
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
            left, right
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(
            left, right
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def canonical_records(manifest: SampleManifest) -> list[dict[str, Any]]:
    records = copy.deepcopy(manifest.to_dict()["records"])
    for record in records:
        record.pop("checkpoint_path", None)
    return records


def is_observational_metric(name: str) -> bool:
    return (
        name in _OBSERVATIONAL_METRIC_NAMES
        or name.endswith("_time_s")
        or name.endswith("_seconds")
        or name.endswith("_per_second")
        or name.endswith(("_max_s", "_p50_s", "_p95_s", "_sum_s"))
        or "latency" in name
        or "queue_wait" in name
    )


def canonical_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove runtime/resource observations while retaining train semantics."""

    return [
        {
            key: value
            for key, value in row.items()
            if not is_observational_metric(key)
        }
        for row in copy.deepcopy(rows)
    ]


def canonical_checkpoint_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove the integrity hash of nondeterministic ``torch.save`` bytes.

    Every checkpoint remains individually protected by that hash and its
    commit-marker tree hash. Cross-run equivalence instead compares the fully
    validated deserialized state plus all remaining checkpoint metadata.
    """

    canonical = copy.deepcopy(metadata)
    canonical.pop("training_state_sha256", None)
    return canonical


def hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_run(root: Path, expected_final_step: int) -> dict[str, Any]:
    """Fail closed on v4 checkpoint, v2 manifest, marker, status, and audit."""

    checkpoint = root / f"checkpoint_{expected_final_step:06d}"
    marker_path = root / "commits" / f"commit_{expected_final_step:06d}.json"
    config = load_json(root / "config.resolved.json")
    marker = load_json(marker_path)
    tree_sha256 = checkpoint_tree_sha256(checkpoint, trusted_root=root)
    marker_checkpoint = marker.get("checkpoint")
    marker_tree_valid = (
        marker.get("schema_version") == "1"
        and marker.get("kind") == "artifact_commit"
        and marker.get("completed_steps") == expected_final_step
        and isinstance(marker_checkpoint, dict)
        and marker_checkpoint.get("completed_steps") == expected_final_step
        and marker_checkpoint.get("path") == checkpoint.name
        and marker_checkpoint.get("final_path") == checkpoint.name
        and marker_checkpoint.get("sha256") == tree_sha256
    )
    validated = read_and_validate_training_state(
        checkpoint,
        config=config,
        trusted_root=root,
        use_checkpoint_implementation_identity=True,
    )
    manifest = SampleManifest.load(root / "sample_manifest.json")
    status = inspect_run_status(root / "run_status.json")
    audit = audit_run_artifacts(root)
    authoritative_metrics = []
    for path in sorted((root / "commits").glob("commit_*.json")):
        committed_marker = load_json(path)
        authoritative_metrics.extend(
            dict(step_payload["metric_row"])
            for step_payload in committed_marker["steps"]
        )
    projected_metrics = load_jsonl(root / "metrics.jsonl")
    security_gates = {
        "checkpoint_v4": validated.state.get("format_version")
        == CHECKPOINT_FORMAT_VERSION
        and validated.metadata.get("format_version") == CHECKPOINT_FORMAT_VERSION
        and validated.step == expected_final_step,
        "manifest_v2": manifest.schema_version == SAMPLE_MANIFEST_SCHEMA_VERSION
        and hash_json(manifest.to_dict()) == audit.get("manifest_sha256"),
        "metrics_projection_semantic": canonical_metric_rows(projected_metrics)
        == canonical_metric_rows(authoritative_metrics),
        "marker_tree_valid": marker_tree_valid,
        "status_authoritative": status.get("completed_steps")
        == expected_final_step
        and status.get("authoritative_completed_steps") == expected_final_step
        and bool(status.get("marker_valid"))
        and bool(status.get("ready_for_aggregation")),
        "artifact_audit_valid": bool(audit.get("valid")),
    }
    return {
        "checkpoint": checkpoint,
        "tree_sha256": tree_sha256,
        "validated": validated,
        "manifest": manifest,
        "status": status,
        "audit": audit,
        "authoritative_metrics": authoritative_metrics,
        "security_gates": security_gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = {
        name: load_json(path / "summary.json")
        for name, path in {
            "continuous": args.continuous,
            "split": args.split,
            "resumed": args.resumed,
        }.items()
    }
    validated_runs = {
        "continuous": validate_run(args.continuous, EXPECTED_FINAL_STEP),
        "split": validate_run(args.split, 1),
        "resumed": validate_run(args.resumed, EXPECTED_FINAL_STEP),
    }
    raw_rows = {
        name: value["authoritative_metrics"]
        for name, value in validated_runs.items()
    }
    rows = {
        name: canonical_metric_rows(records) for name, records in raw_rows.items()
    }
    checkpoints = {
        name: validated_runs[name]["checkpoint"]
        for name in ("continuous", "resumed")
    }
    states = {
        name: validated_runs[name]["validated"].state
        for name in ("continuous", "resumed")
    }
    state_keys = sorted(set(states["continuous"]) | set(states["resumed"]))
    state_gates = {
        key: key in states["continuous"]
        and key in states["resumed"]
        and exact(states["continuous"][key], states["resumed"][key])
        for key in state_keys
    }
    adapter_hashes = {
        name: states[name]["adapter_payload_sha256"] for name in checkpoints
    }
    checkpoint_metadata = {
        name: canonical_checkpoint_metadata(
            validated_runs[name]["validated"].metadata
        )
        for name in checkpoints
    }
    checkpoint_tree_byte_exact = (
        validated_runs["continuous"]["tree_sha256"]
        == validated_runs["resumed"]["tree_sha256"]
    )
    continuous_records = canonical_records(validated_runs["continuous"]["manifest"])
    combined_records = canonical_records(validated_runs["split"]["manifest"]) + canonical_records(
        validated_runs["resumed"]["manifest"]
    )
    security_gates = {
        name: all(result["security_gates"].values())
        for name, result in validated_runs.items()
    }
    gates = {
        "all_segments_valid": all(item.get("valid") for item in summaries.values()),
        "all_final_runs_secure": all(security_gates.values()),
        "metrics_rows_2_1_1": [len(rows[name]) for name in rows]
        == [2, 1, 1],
        "resume_loaded_step_one": summaries["resumed"].get("start_step") == 1
        and summaries["resumed"].get("target_step") == 2,
        "step_zero_metrics_exact": rows["continuous"][:1] == rows["split"],
        "step_one_metrics_exact": rows["continuous"][1:] == rows["resumed"],
        "manifest_semantic_exact": continuous_records == combined_records,
        "final_adapter_exact": adapter_hashes["continuous"]
        == adapter_hashes["resumed"],
        "final_training_state_exact": all(state_gates.values()),
        "final_checkpoint_metadata_semantic_exact": exact(
            checkpoint_metadata["continuous"], checkpoint_metadata["resumed"]
        ),
        "split_final_hash_equals_resume_initial": summaries["split"].get(
            "final_trainable_sha256"
        )
        == summaries["resumed"].get("initial_trainable_sha256"),
    }
    result = {
        "schema_version": "1",
        "label": args.label,
        "valid": all(gates.values()),
        "all_exact_gates_passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "observational_metrics_excluded": sorted(
            {
                key
                for records in raw_rows.values()
                for row in records
                for key in row
                if is_observational_metric(key)
            }
        ),
        "byte_diagnostics": {
            "final_checkpoint_tree_byte_exact": checkpoint_tree_byte_exact,
            "note": (
                "torch.save archive bytes are not a semantic equality gate; "
                "each tree is independently hash-validated before the "
                "deserialized state is compared exactly"
            ),
        },
        "run_security": {
            name: {
                "gates": value["security_gates"],
                "checkpoint_tree_sha256": value["tree_sha256"],
                "audit_schema_version": value["audit"].get("schema_version"),
                "manifest_schema_version": value["manifest"].schema_version,
            }
            for name, value in validated_runs.items()
        },
        "training_state_exact": state_gates,
        "adapter_sha256": adapter_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
