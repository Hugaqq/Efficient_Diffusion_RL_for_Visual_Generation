"""Cross-route acceptance audit for the complete frozen A7 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROUTE_REWARDS = {
    "flow-grpo-sd3": frozenset({"reward_quality"}),
    "flow-grpo-wan": frozenset({"reward_general"}),
    "tempflow-sd3": frozenset({"reward_quality"}),
    "flash-wan": frozenset({"reward_general"}),
    "world-r1-core-wan": frozenset({"reward_3d", "reward_general"}),
    "world-r1-release-surrogate-wan": frozenset(
        {"reward_3d", "reward_general"}
    ),
}


class A7MatrixAuditError(RuntimeError):
    """The six accepted rows do not form one frozen compatibility matrix."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A7MatrixAuditError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A7MatrixAuditError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise A7MatrixAuditError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def audit_matrix(
    *,
    acceptance_root: Path,
    freeze_record_path: Path,
    reward_identity_path: Path,
) -> dict[str, Any]:
    freeze = _json(freeze_record_path)
    reward_freeze = _json(reward_identity_path)
    freeze_sha = _sha256(freeze_record_path)
    expected_code = freeze["code"]["content_sha256"]
    expected_wheel = freeze["wheel"]["sha256"]
    expected_remote_rewards = {
        name: payload["content_sha256"] for name, payload in reward_freeze.items()
    }

    rows: dict[str, dict[str, Any]] = {}
    for route in sorted(ROUTE_REWARDS):
        path = acceptance_root / f"{route}.json"
        row = _json(path)
        if row.get("kind") != "visual_rl_a7_route_acceptance":
            raise A7MatrixAuditError(f"{route}: acceptance kind is invalid")
        if row.get("route") != route or row.get("accepted") is not True:
            raise A7MatrixAuditError(f"{route}: row is not explicitly accepted")
        if row.get("target_optimizer_steps") != 20 or row.get("committed_steps") != 20:
            raise A7MatrixAuditError(f"{route}: row does not prove 20 commits")
        if row.get("code_content_sha256") != expected_code:
            raise A7MatrixAuditError(f"{route}: code identity differs")
        if row.get("wheel_sha256") != expected_wheel:
            raise A7MatrixAuditError(f"{route}: wheel identity differs")
        if row.get("freeze_record_sha256") != freeze_sha:
            raise A7MatrixAuditError(f"{route}: freeze identity differs")
        rewards = row.get("reward_artifact_content_sha256")
        if not isinstance(rewards, dict) or frozenset(rewards) != ROUTE_REWARDS[route]:
            raise A7MatrixAuditError(f"{route}: reward artifact set differs")
        for artifact_ref, expected_digest in expected_remote_rewards.items():
            if artifact_ref in rewards and rewards[artifact_ref] != expected_digest:
                raise A7MatrixAuditError(
                    f"{route}: {artifact_ref} identity differs from deployment freeze"
                )
        memory = row.get("memory")
        if (
            not isinstance(memory, dict)
            or type(memory.get("sample_count")) is not int
            or memory["sample_count"] < 2
            or type(memory.get("peak_memory_used_mib")) is not int
            or type(memory.get("memory_total_mib")) is not int
            or not 0 < memory["peak_memory_used_mib"] <= memory["memory_total_mib"]
        ):
            raise A7MatrixAuditError(f"{route}: memory evidence is invalid")
        rows[route] = row

    quality_identities = {
        rows[route]["reward_artifact_content_sha256"]["reward_quality"]
        for route in ("flow-grpo-sd3", "tempflow-sd3")
    }
    if len(quality_identities) != 1:
        raise A7MatrixAuditError("SD3 routes used different local reward artifacts")
    config_hashes = {row["config_sha256"] for row in rows.values()}
    if config_hashes != set(freeze["configs"].values()):
        raise A7MatrixAuditError("accepted routes do not exactly cover frozen configs")

    return {
        "schema_version": 1,
        "kind": "visual_rl_a7_matrix_acceptance",
        "accepted": True,
        "route_count": len(rows),
        "routes": {
            route: {
                "committed_steps": row["committed_steps"],
                "config_sha256": row["config_sha256"],
                "reward_artifact_content_sha256": row[
                    "reward_artifact_content_sha256"
                ],
                "peak_memory_used_mib": row["memory"]["peak_memory_used_mib"],
                "memory_total_mib": row["memory"]["memory_total_mib"],
            }
            for route, row in rows.items()
        },
        "code_content_sha256": expected_code,
        "wheel_sha256": expected_wheel,
        "freeze_record_sha256": freeze_sha,
        "reward_quality_content_sha256": next(iter(quality_identities)),
        "reward_general_content_sha256": expected_remote_rewards["reward_general"],
        "reward_3d_content_sha256": expected_remote_rewards["reward_3d"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("acceptance_root", type=Path)
    parser.add_argument("freeze_record", type=Path)
    parser.add_argument("reward_identities", type=Path)
    args = parser.parse_args()
    payload = audit_matrix(
        acceptance_root=args.acceptance_root.resolve(),
        freeze_record_path=args.freeze_record.resolve(),
        reward_identity_path=args.reward_identities.resolve(),
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
