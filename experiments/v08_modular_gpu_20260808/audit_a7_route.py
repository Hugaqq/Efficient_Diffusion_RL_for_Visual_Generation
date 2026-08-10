"""Read-only acceptance audit for one completed frozen A7 GPU route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from visual_rl.artifacts.inspection import audit_run

TARGET_STEPS = 20
LOG_FAILURE = re.compile(
    r"out of memory|cuda oom|zero gradient|non[- ]?finite|traceback|runtimeerror",
    re.IGNORECASE,
)
MEMORY_HEADER = (
    "sample_epoch_s",
    "target_pid",
    "target_alive",
    "gpu_index",
    "gpu_uuid",
    "memory_used_mib",
    "memory_total_mib",
    "utilization_gpu_percent",
)


class A7AuditError(RuntimeError):
    """One authoritative final-route invariant failed."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A7AuditError(f"cannot read canonical JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A7AuditError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise A7AuditError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _positive_finite(payload: dict[str, Any], name: str) -> float:
    value = payload.get(name)
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise A7AuditError(f"final metric {name} must be a positive finite float")
    return value


def _reward_identities(resolved: dict[str, Any]) -> dict[str, str]:
    try:
        resources = resolved["materialized_recipe"]["reward_plan"]["plan"][
            "resources"
        ]
    except (KeyError, TypeError) as exc:
        raise A7AuditError("resolved recipe has no materialized reward resources") from exc
    if not isinstance(resources, list) or not resources:
        raise A7AuditError("resolved recipe reward resources must be non-empty")
    identities: dict[str, str] = {}
    for item in resources:
        try:
            artifact_ref = item["descriptor"]["artifact_ref"]
            identity = item["artifact_identity"]
            digest = identity["content_sha256"]
        except (KeyError, TypeError) as exc:
            raise A7AuditError("reward artifact identity payload is malformed") from exc
        if (
            not isinstance(artifact_ref, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", artifact_ref)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or identity.get("identity_schema") != "filesystem-artifact.v1"
        ):
            raise A7AuditError("reward artifact identity is not canonical")
        if artifact_ref in identities:
            raise A7AuditError(f"duplicate reward artifact ref {artifact_ref!r}")
        identities[artifact_ref] = digest
    return dict(sorted(identities.items()))


def _audit_memory(path: Path, *, trainer_pid: int, gpu_index: int) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MEMORY_HEADER:
                raise A7AuditError("GPU memory CSV header drifted")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise A7AuditError(f"cannot read GPU memory CSV {path}: {exc}") from exc
    if len(rows) < 2:
        raise A7AuditError("GPU memory CSV must contain live and terminal samples")

    uuids: set[str] = set()
    totals: set[int] = set()
    live_rows = 0
    peak = 0
    for index, row in enumerate(rows):
        try:
            observed_pid = int(row["target_pid"])
            alive = int(row["target_alive"])
            observed_gpu = int(row["gpu_index"])
            used = int(row["memory_used_mib"])
            total = int(row["memory_total_mib"])
            utilization = int(row["utilization_gpu_percent"])
            int(row["sample_epoch_s"])
        except (TypeError, ValueError) as exc:
            raise A7AuditError(f"GPU memory row {index} is malformed") from exc
        if observed_pid != trainer_pid or observed_gpu != gpu_index:
            raise A7AuditError("GPU memory CSV PID/GPU binding drifted")
        if alive not in {0, 1} or (index < len(rows) - 1 and alive != 1):
            raise A7AuditError("GPU memory target liveness sequence is invalid")
        if index == len(rows) - 1 and alive != 0:
            raise A7AuditError("GPU memory CSV lacks the terminal dead-target row")
        if used < 0 or total <= 0 or used > total or not 0 <= utilization <= 100:
            raise A7AuditError("GPU memory sample is outside driver bounds")
        uuid = row["gpu_uuid"].strip()
        if not uuid.startswith("GPU-"):
            raise A7AuditError("GPU memory sample has an invalid UUID")
        uuids.add(uuid)
        totals.add(total)
        peak = max(peak, used)
        live_rows += alive
    if live_rows < 1 or len(uuids) != 1 or len(totals) != 1:
        raise A7AuditError("GPU memory series is not one stable physical device")
    return {
        "sha256": _sha256(path),
        "sample_count": len(rows),
        "live_sample_count": live_rows,
        "gpu_uuid": next(iter(uuids)),
        "memory_total_mib": next(iter(totals)),
        "peak_memory_used_mib": peak,
    }


def audit_route(
    *,
    evidence_root: Path,
    route: str,
    freeze_record_path: Path,
    reward_identity_path: Path,
) -> dict[str, Any]:
    freeze = _json(freeze_record_path)
    expected_rewards = _json(reward_identity_path)
    launch = _json(evidence_root / "launch-receipts" / f"{route}.json")
    if launch.get("kind") != "visual_rl_a7_launch_receipt":
        raise A7AuditError("launch receipt kind is invalid")
    if launch.get("route") != route:
        raise A7AuditError("launch receipt route differs from requested route")
    expected_freeze_sha = _sha256(freeze_record_path)
    if launch.get("freeze_record_sha256") != expected_freeze_sha:
        raise A7AuditError("launch receipt references a different freeze record")
    if launch.get("code_content_sha256") != freeze["code"]["content_sha256"]:
        raise A7AuditError("launch receipt code identity drifted")
    if launch.get("wheel_sha256") != freeze["wheel"]["sha256"]:
        raise A7AuditError("launch receipt wheel identity drifted")
    config = launch.get("config")
    if not isinstance(config, dict):
        raise A7AuditError("launch receipt config binding is malformed")
    config_name = Path(str(config.get("path"))).name
    if config.get("sha256") != freeze["configs"].get(config_name):
        raise A7AuditError("launch receipt config identity drifted")
    gpu_index = launch.get("physical_gpu_index")
    trainer_pid = launch.get("trainer_pid")
    if type(gpu_index) is not int or gpu_index < 0:
        raise A7AuditError("launch receipt GPU index is invalid")
    if type(trainer_pid) is not int or trainer_pid <= 0:
        raise A7AuditError("launch receipt trainer PID is invalid")

    exitcode_path = evidence_root / "logs" / f"{route}.exitcode"
    try:
        exitcode = int(exitcode_path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError) as exc:
        raise A7AuditError(f"cannot read trainer exit code: {exc}") from exc
    if exitcode != 0:
        raise A7AuditError(f"trainer exited with status {exitcode}")

    run_root = evidence_root / "runs" / route
    report = audit_run(run_root)
    if not report.ok or report.committed_steps != TARGET_STEPS:
        messages = [item.message for item in report.errors]
        raise A7AuditError(
            f"terminal audit failed or committed_steps != {TARGET_STEPS}: {messages}"
        )
    success = _json(run_root / "SUCCESS")
    manifest = _json(run_root / "run_manifest.json")
    progress_document = _json(
        run_root / "checkpoints" / f"step-{TARGET_STEPS}" / "progress.json"
    )
    progress = progress_document.get("progress")
    if not isinstance(progress, dict):
        raise A7AuditError("step-20 progress document has no canonical payload")
    if success.get("committed_steps") != TARGET_STEPS:
        raise A7AuditError("SUCCESS does not prove 20 committed steps")
    if (
        manifest.get("start_optimizer_step") != 0
        or manifest.get("committed_steps") != TARGET_STEPS
        or manifest.get("update_count") != TARGET_STEPS
    ):
        raise A7AuditError("run manifest does not prove 20 fresh optimizer updates")
    if any(
        progress.get(name) != TARGET_STEPS
        for name in ("global_step", "iteration", "next_optimizer_step")
    ):
        raise A7AuditError("step-20 progress counters are inconsistent")

    metrics_path = run_root / "metrics.jsonl"
    try:
        metric_lines = metrics_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise A7AuditError(f"cannot read final metrics: {exc}") from exc
    if len(metric_lines) != 1:
        raise A7AuditError("metrics.jsonl must contain one terminal row")
    metrics = json.loads(metric_lines[0])
    if not isinstance(metrics, dict) or metrics.get("step") != TARGET_STEPS - 1:
        raise A7AuditError("terminal metric row is not optimizer step 19")
    pre_clip = _positive_finite(metrics, "gradient_norm_pre_clip")
    post_clip = _positive_finite(metrics, "gradient_norm_post_clip")

    resolved = _json(run_root / "resolved_recipe.json")
    try:
        code_identity = resolved["materialized_recipe"]["code_artifact_identity"]
    except (KeyError, TypeError) as exc:
        raise A7AuditError("resolved recipe has no code artifact identity") from exc
    if (
        code_identity.get("identity_schema") != "filesystem-artifact.v1"
        or code_identity.get("content_policy") != "python-code.v1"
        or code_identity.get("content_sha256") != freeze["code"]["content_sha256"]
    ):
        raise A7AuditError("resolved recipe code artifact differs from the freeze")
    observed_rewards = _reward_identities(resolved)
    for artifact_ref, digest in observed_rewards.items():
        if artifact_ref in expected_rewards:
            expected_digest = expected_rewards[artifact_ref]["content_sha256"]
            if digest != expected_digest:
                raise A7AuditError(
                    f"reward artifact {artifact_ref!r} differs from the freeze"
                )

    log_path = evidence_root / "logs" / f"{route}.log"
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise A7AuditError(f"cannot read trainer log: {exc}") from exc
    match = LOG_FAILURE.search(log_text)
    if match is not None:
        raise A7AuditError(f"trainer log contains failure signature {match.group(0)!r}")
    memory = _audit_memory(
        evidence_root / "logs" / f"{route}.gpu-memory.csv",
        trainer_pid=trainer_pid,
        gpu_index=gpu_index,
    )
    return {
        "schema_version": 1,
        "kind": "visual_rl_a7_route_acceptance",
        "route": route,
        "accepted": True,
        "target_optimizer_steps": TARGET_STEPS,
        "committed_steps": report.committed_steps,
        "config_sha256": config["sha256"],
        "code_content_sha256": freeze["code"]["content_sha256"],
        "wheel_sha256": freeze["wheel"]["sha256"],
        "freeze_record_sha256": expected_freeze_sha,
        "reward_artifact_content_sha256": observed_rewards,
        "final_gradient_norm_pre_clip": pre_clip,
        "final_gradient_norm_post_clip": post_clip,
        "terminal_checkpoint": str(report.authoritative_checkpoint),
        "checked_checkpoint_count": report.checked_checkpoint_count,
        "stdout_log_sha256": _sha256(log_path),
        "memory": memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("route")
    parser.add_argument("freeze_record", type=Path)
    parser.add_argument("reward_identities", type=Path)
    args = parser.parse_args()
    payload = audit_route(
        evidence_root=args.evidence_root.resolve(),
        route=args.route,
        freeze_record_path=args.freeze_record.resolve(),
        reward_identity_path=args.reward_identities.resolve(),
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
