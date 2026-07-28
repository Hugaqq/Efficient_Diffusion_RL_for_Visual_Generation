"""Final read-only W06 evidence gate; not run during source preparation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

from experiments.v0_7.environment_report import probe_git_identity
from experiments.v0_7.interrupt_resume import FAMILY_ORDER, ROLE_SPECS
from experiments.v0_7.mg1_nccl import INTERNAL_NCCL_COMMANDS
from experiments.v0_7.offline_aggregate import (
    ALGORITHMS,
    SEEDS,
    SourceStatus,
    load_rows,
)
from experiments.v0_7.verify_reward_improvement import verify_reward_improvement

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
EVIDENCE = ROOT / "evidence"

_CANDIDATE_FIELDS = {"clean", "commit", "tested"}
_ROLE_FIELDS = {
    "audit_ok",
    "committed_steps",
    "exit_code",
    "output_dir",
    "role",
}
_SOURCE_FIELDS = {
    "algorithm",
    "audit_ok",
    "committed_steps",
    "inspect_ok",
    "seed",
    "source_run",
}
_NATIVE_ITEMS = {
    "checkpoint_resume",
    "current_log_prob",
    "gradient",
    "group_advantage",
    "initial_latent",
    "old_log_prob",
    "parameter_delta",
    "policy_loss",
    "prompt_encoding",
    "reference_kl",
    "rollout_latent",
    "timestep",
    "total_loss",
    "transition_statistics",
}
_RESUME_COMPARISONS = {
    "adapter_tensors",
    "global_step",
    "grad_scaler_state",
    "next_step_inputs",
    "non_timing_metrics",
    "optimizer_state",
    "rng_state",
}
_COMPARISON_FIELDS = {
    "atol",
    "dtype",
    "max_abs_error",
    "max_rel_error",
    "passed",
    "rtol",
    "shape",
    "tensor_name",
}


def verify_evidence(
    evidence_dir: Path = EVIDENCE,
    *,
    repo_root: Path = REPO_ROOT,
    git_probe: Callable[[Path], Mapping[str, object]] = probe_git_identity,
) -> dict[str, Any]:
    """Parse and validate every final W06 evidence class, failing closed."""

    directory = Path(evidence_dir).resolve(strict=True)
    paths = {
        name: _regular_evidence_file(directory, name)
        for name in (
            "flow_native.json",
            "environment.jsonl",
            "mg1_internal.json",
            "q100_reward_rows.jsonl",
            "q100_source_status.json",
            "role_results.json",
        )
    }

    role_candidate, role_outputs = _verify_roles(
        _strict_json_load(paths["role_results.json"])
    )
    native_candidate = _verify_flow_native(
        _strict_json_load(paths["flow_native.json"])
    )
    mg1_candidate = _verify_mg1(
        _strict_json_load(paths["mg1_internal.json"])
    )
    q100_candidate, statuses = _verify_q100_sources(
        _strict_json_load(paths["q100_source_status.json"]),
        rows_path=paths["q100_reward_rows.jsonl"],
    )
    environment_candidate = _verify_environment(
        paths["environment.jsonl"],
    )
    candidates = (
        role_candidate,
        native_candidate,
        mg1_candidate,
        q100_candidate,
        environment_candidate,
    )
    if any(candidate != candidates[0] for candidate in candidates[1:]):
        raise ValueError("evidence candidate commit/clean/tested identity mismatch")
    live = git_probe(Path(repo_root))
    if (
        not isinstance(live, Mapping)
        or set(live) != {"clean", "commit"}
        or live.get("commit") != candidates[0]["commit"]
        or live.get("clean") is not True
    ):
        raise ValueError("evidence candidate does not match current clean Git HEAD")

    rows = load_rows(paths["q100_reward_rows.jsonl"])
    status_by_identity = {
        (status.algorithm, status.seed): status for status in statuses
    }
    for row in rows:
        status = status_by_identity.get((row.algorithm, row.seed))
        if status is None or row.source_run != status.source_run:
            raise ValueError("Q100 reward row/source-status identity mismatch")
    for status in statuses:
        role_name = f"{status.algorithm}_q100_seed{status.seed}"
        if not _source_matches_role(status.source_run, role_outputs[role_name]):
            raise ValueError("Q100 source run disagrees with its role output_dir")
    reward_report = verify_reward_improvement(rows, statuses)
    if set(reward_report) != set(ALGORITHMS):
        raise RuntimeError("Q100 reward report algorithm set is incomplete")
    if not all(item.get("evidence_complete") is True for item in reward_report.values()):
        raise RuntimeError("Q100 evidence is incomplete")
    if not all(item.get("reward_pass") is True for item in reward_report.values()):
        raise RuntimeError("Q100 reward-improvement gate failed")

    return {
        "schema_version": 1,
        "candidate": dict(candidates[0]),
        "role_count": len(role_outputs),
        "semantic_family_count": len(FAMILY_ORDER),
        "flow_native_pass": True,
        "mg1_fixed_test_count": len(INTERNAL_NCCL_COMMANDS),
        "reward": reward_report,
        "overall_pass": True,
    }


def _verify_roles(
    value: object,
) -> tuple[dict[str, object], dict[str, str]]:
    root = _exact_mapping(
        value,
        {"candidate", "families", "roles", "schema_version"},
        label="role results",
    )
    _schema_one(root, label="role results")
    candidate = _candidate(root["candidate"], label="role results")
    families = _exact_mapping(
        root["families"],
        set(FAMILY_ORDER),
        label="role family semantic parity",
    )
    for family, raw in families.items():
        result = _exact_mapping(
            raw,
            {"semantic_parity"},
            label=f"{family} semantic parity",
        )
        if result["semantic_parity"] is not True:
            raise ValueError(f"{family} semantic parity did not pass")
    raw_roles = root["roles"]
    if not isinstance(raw_roles, list):
        raise ValueError("role results roles must be a list")
    expected = {spec.name: spec for spec in ROLE_SPECS}
    seen: set[str] = set()
    outputs: dict[str, str] = {}
    for raw in raw_roles:
        role = _exact_mapping(raw, _ROLE_FIELDS, label="role result")
        name = _nonempty_text(role["role"], label="role result role")
        if name not in expected or name in seen:
            raise ValueError(f"unknown or duplicate role result: {name}")
        seen.add(name)
        spec = expected[name]
        if type(role["exit_code"]) is not int or role["exit_code"] != spec.expected_exit:
            raise ValueError(f"{name} exit code disagrees with RoleSpec")
        if (
            type(role["committed_steps"]) is not int
            or role["committed_steps"] != spec.target_steps
        ):
            raise ValueError(f"{name} committed steps disagree with RoleSpec")
        if type(role["audit_ok"]) is not bool:
            raise TypeError(f"{name} audit_ok must be bool")
        if spec.phase != "interrupted" and role["audit_ok"] is not True:
            raise ValueError(f"{name} must finish audit-ok")
        outputs[name] = _relative_posix(
            role["output_dir"],
            label=f"{name} output_dir",
        )
    if seen != set(expected) or len(raw_roles) != len(expected):
        raise ValueError("role results must contain each of the 30 roles exactly once")
    return candidate, outputs


def _verify_environment(path: Path) -> dict[str, object]:
    expected_fields = {
        "attempt_id",
        "clean",
        "commit",
        "cuda",
        "devices",
        "packages",
        "platform",
        "python",
        "role",
        "tested",
    }
    expected_roles = {spec.name for spec in ROLE_SPECS}
    seen: set[str] = set()
    seen_attempts: set[tuple[str, str]] = set()
    candidate: dict[str, object] | None = None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("environment evidence must not be empty")
    for line in lines:
        value = _strict_json_loads(line)
        item = _exact_mapping(value, expected_fields, label="environment attempt")
        role = _nonempty_text(item["role"], label="environment role")
        if role not in expected_roles:
            raise ValueError(f"unknown environment role: {role}")
        attempt_id = _nonempty_text(
            item["attempt_id"],
            label="environment attempt_id",
        )
        attempt_identity = (role, attempt_id)
        if attempt_identity in seen_attempts:
            raise ValueError("duplicate environment role/attempt identity")
        seen_attempts.add(attempt_identity)
        _nonempty_text(item["python"], label="environment python")
        _nonempty_text(item["platform"], label="environment platform")
        if item["cuda"] is not None and (
            not isinstance(item["cuda"], str) or not item["cuda"]
        ):
            raise ValueError("environment cuda must be null or non-empty")
        if not isinstance(item["devices"], list) or any(
            not isinstance(device, str) or not device
            for device in item["devices"]
        ):
            raise ValueError("environment devices must be a string list")
        packages = item["packages"]
        if (
            not isinstance(packages, Mapping)
            or not packages
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(version, str)
                or not version
                for name, version in packages.items()
            )
        ):
            raise ValueError(
                "environment packages must map non-empty names to versions"
            )
        current = _candidate(
            {
                "clean": item["clean"],
                "commit": item["commit"],
                "tested": item["tested"],
            },
            label="environment attempt",
        )
        if candidate is None:
            candidate = current
        elif current != candidate:
            raise ValueError("environment attempts use different candidates")
        seen.add(role)
    if seen != expected_roles:
        raise ValueError("environment evidence must contain all 30 role attempts")
    assert candidate is not None
    return candidate


def _verify_flow_native(value: object) -> dict[str, object]:
    root = _exact_mapping(
        value,
        {"candidate", "report", "schema_version"},
        label="Flow native envelope",
    )
    _schema_one(root, label="Flow native envelope")
    candidate = _candidate(root["candidate"], label="Flow native envelope")
    report = _exact_mapping(
        root["report"],
        {
            "case",
            "config_path",
            "items",
            "overall_pass",
            "precision",
            "schema_version",
        },
        label="Flow native report",
    )
    _schema_one(report, label="Flow native report")
    expected_identity = {
        "case": "flow_grpo_sd3_case_v1",
        "config_path": "configs/flow_grpo_sd3.yaml",
        "precision": "fp32",
    }
    if any(report[field] != expected for field, expected in expected_identity.items()):
        raise ValueError("Flow native fixed case/config/precision identity drifted")
    if report["overall_pass"] is not True:
        raise ValueError("Flow native overall_pass must be true")
    items = _exact_mapping(report["items"], _NATIVE_ITEMS, label="Flow native items")
    for name, raw_item in items.items():
        item = _exact_mapping(
            raw_item,
            {"comparisons", "passed"},
            label=f"Flow native item {name}",
        )
        if item["passed"] is not True:
            raise ValueError(f"Flow native item {name} did not pass")
        if name == "checkpoint_resume":
            comparisons = _exact_mapping(
                item["comparisons"],
                _RESUME_COMPARISONS,
                label="Flow native checkpoint comparisons",
            )
            if any(value is not True for value in comparisons.values()):
                raise ValueError("Flow native checkpoint comparison failed")
        else:
            comparisons = item["comparisons"]
            if not isinstance(comparisons, list) or not comparisons:
                raise ValueError(f"Flow native item {name} comparisons are empty")
            tensor_names: list[str] = []
            for raw_comparison in comparisons:
                comparison = _exact_mapping(
                    raw_comparison,
                    _COMPARISON_FIELDS,
                    label=f"Flow native item {name} comparison",
                )
                tensor_names.append(_verify_native_comparison(comparison))
            if tensor_names != sorted(tensor_names) or len(tensor_names) != len(
                set(tensor_names)
            ):
                raise ValueError("Flow native tensor comparisons are not unique/sorted")
    return candidate


def _verify_native_comparison(value: Mapping[str, object]) -> str:
    name = _nonempty_text(value["tensor_name"], label="native tensor_name")
    shape = value["shape"]
    if not isinstance(shape, list) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError("native comparison shape must contain non-negative integers")
    _nonempty_text(value["dtype"], label="native comparison dtype")
    for field in ("rtol", "atol", "max_abs_error", "max_rel_error"):
        scalar = value[field]
        if (
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
            or float(scalar) < 0.0
        ):
            raise ValueError(f"native comparison {field} must be finite/non-negative")
    if value["passed"] is not True:
        raise ValueError("native tensor comparison did not pass")
    return name


def _verify_mg1(value: object) -> dict[str, object]:
    root = _exact_mapping(
        value,
        {"candidate", "schema_version", "tests"},
        label="MG1 evidence",
    )
    _schema_one(root, label="MG1 evidence")
    candidate = _candidate(root["candidate"], label="MG1 evidence")
    raw_tests = root["tests"]
    if not isinstance(raw_tests, list):
        raise ValueError("MG1 tests must be a list")
    expected = {command[-1] for command in INTERNAL_NCCL_COMMANDS}
    seen: set[str] = set()
    for raw in raw_tests:
        item = _exact_mapping(raw, {"nodeid", "passed"}, label="MG1 test")
        nodeid = _nonempty_text(item["nodeid"], label="MG1 nodeid")
        if nodeid not in expected or nodeid in seen:
            raise ValueError(f"unknown or duplicate MG1 nodeid: {nodeid}")
        if item["passed"] is not True:
            raise ValueError(f"MG1 test did not pass: {nodeid}")
        seen.add(nodeid)
    if seen != expected or len(raw_tests) != 3:
        raise ValueError("MG1 evidence must contain exactly the fixed three tests")
    return candidate


def _verify_q100_sources(
    value: object,
    *,
    rows_path: Path,
) -> tuple[dict[str, object], tuple[SourceStatus, ...]]:
    root = _exact_mapping(
        value,
        {"candidate", "rows_sha256", "schema_version", "sources"},
        label="Q100 source status",
    )
    _schema_one(root, label="Q100 source status")
    candidate = _candidate(root["candidate"], label="Q100 source status")
    expected_digest = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    if root["rows_sha256"] != expected_digest:
        raise ValueError("Q100 rows digest disagrees with source-status envelope")
    raw_sources = root["sources"]
    if not isinstance(raw_sources, list):
        raise ValueError("Q100 sources must be a list")
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    statuses: list[SourceStatus] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_sources:
        source = _exact_mapping(raw, _SOURCE_FIELDS, label="Q100 source")
        algorithm = source["algorithm"]
        seed = source["seed"]
        identity = (algorithm, seed)
        if identity not in expected or identity in seen:
            raise ValueError(f"unknown or duplicate Q100 source: {identity}")
        if (
            type(source["committed_steps"]) is not int
            or source["committed_steps"] != 100
            or source["inspect_ok"] is not True
            or source["audit_ok"] is not True
        ):
            raise ValueError(f"Q100 source is incomplete or unaudited: {identity}")
        source_run = _nonempty_text(
            source["source_run"],
            label="Q100 source_run",
        )
        seen.add(identity)
        statuses.append(
            SourceStatus(
                algorithm=algorithm,
                seed=seed,
                source_run=source_run,
                committed_steps=100,
                inspect_ok=True,
                audit_ok=True,
            )
        )
    if seen != expected or len(raw_sources) != 12:
        raise ValueError("Q100 source status must contain four algorithms x three seeds")
    return candidate, tuple(
        sorted(statuses, key=lambda item: (item.algorithm, item.seed))
    )


def _candidate(value: object, *, label: str) -> dict[str, object]:
    candidate = _exact_mapping(value, _CANDIDATE_FIELDS, label=f"{label} candidate")
    commit = candidate["commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"{label} candidate commit must be a full lowercase Git SHA")
    if candidate["clean"] is not True or candidate["tested"] is not True:
        raise ValueError(f"{label} candidate must be clean=true and tested=true")
    return dict(candidate)


def _schema_one(value: Mapping[str, object], *, label: str) -> None:
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError(f"{label} schema_version must be integer 1")


def _exact_mapping(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields do not match schema")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _relative_posix(value: object, *, label: str) -> str:
    text = _nonempty_text(value, label=label)
    if "\\" in text:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return text


def _source_matches_role(source_run: str, output_dir: str) -> bool:
    if "\\" in source_run:
        return False
    source = PurePosixPath(source_run)
    if source.as_posix() != source_run or any(
        part in {"", ".", ".."} for part in source.parts
    ):
        return False
    return source_run == output_dir or source_run.endswith(f"/{output_dir}")


def _regular_evidence_file(directory: Path, name: str) -> Path:
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"real experiment evidence is not_run: {name}")
    return path


def _strict_json_load(path: Path) -> Any:
    return _strict_json_loads(path.read_text(encoding="utf-8"))


def _strict_json_loads(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )


if __name__ == "__main__":
    print(
        json.dumps(
            verify_evidence(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
