"""Read-only Q100 artifact scanner and canonical reward-row projection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import visual_rl as vr

from experiments.v0_7.environment_report import (
    atomic_write_bytes,
    evidence_candidate,
    probe_git_identity,
)

ALGORITHMS = (
    "flow_grpo_sd3",
    "tempflow_sd3",
    "flash_wan",
    "world_r1_wan",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = Path(__file__).resolve().parent / "evidence"
SEEDS = (17, 29, 43)
ROW_FIELDS = (
    "algorithm",
    "seed",
    "step",
    "prompt_id",
    "sample_id",
    "weighted_total",
    "source_run",
)


@dataclass(frozen=True)
class RewardRow:
    algorithm: str
    seed: int
    step: int
    prompt_id: str
    sample_id: str
    weighted_total: float
    source_run: str

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm: {self.algorithm}")
        if self.seed not in SEEDS:
            raise ValueError(f"unexpected seed: {self.seed}")
        if type(self.step) is not int or not 0 <= self.step < 100:
            raise ValueError("step must be an integer in 0..99")
        for name in ("prompt_id", "sample_id", "source_run"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            isinstance(self.weighted_total, bool)
            or not isinstance(self.weighted_total, (int, float))
            or not math.isfinite(float(self.weighted_total))
        ):
            raise ValueError("weighted_total must be finite")
        object.__setattr__(self, "weighted_total", float(self.weighted_total))


@dataclass(frozen=True)
class SourceStatus:
    algorithm: str
    seed: int
    source_run: str
    committed_steps: int
    inspect_ok: bool
    audit_ok: bool


def scan_q100_inputs(
    index_path: str | Path,
) -> tuple[tuple[RewardRow, ...], tuple[SourceStatus, ...]]:
    """Scan exactly the 12 indexed runs after public status and audit."""

    path = Path(index_path).resolve(strict=True)
    payload = _strict_json_load(path)
    if not isinstance(payload, dict) or set(payload) != {"runs", "schema_version"}:
        raise ValueError("Q100 index must contain exactly schema_version/runs")
    if payload["schema_version"] != 1 or not isinstance(payload["runs"], list):
        raise ValueError("unsupported Q100 index")
    entries = payload["runs"]
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    actual: set[tuple[str, int]] = set()
    rows: list[RewardRow] = []
    statuses: list[SourceStatus] = []
    for raw in entries:
        if not isinstance(raw, dict) or set(raw) != {
            "algorithm",
            "run_dir",
            "seed",
        }:
            raise ValueError("Q100 run entries have unexpected fields")
        algorithm = raw["algorithm"]
        seed = raw["seed"]
        identity = (algorithm, seed)
        if identity not in expected or identity in actual:
            raise ValueError(f"invalid or duplicate Q100 identity: {identity}")
        actual.add(identity)
        run_dir = _child_path(path.parent.parent, raw["run_dir"])
        status = vr.inspect_run(run_dir)
        audit = vr.audit_run(run_dir)
        source_run = run_dir.as_posix()
        source_status = SourceStatus(
            algorithm=algorithm,
            seed=seed,
            source_run=source_run,
            committed_steps=status.committed_steps,
            inspect_ok=status.ok,
            audit_ok=audit.ok,
        )
        statuses.append(source_status)
        if (
            not status.ok
            or not audit.ok
            or status.committed_steps != 100
            or audit.committed_steps != 100
        ):
            continue
        rows.extend(
            _manifest_rows(
                run_dir / "sample_manifest.json",
                algorithm=algorithm,
                seed=seed,
                source_run=source_run,
            )
        )
    if actual != expected or len(entries) != 12:
        raise ValueError("Q100 index must name exactly four algorithms x three seeds")
    return validate_rows(rows), tuple(
        sorted(statuses, key=lambda item: (item.algorithm, item.seed))
    )


def validate_rows(rows: Iterable[RewardRow | Mapping[str, object]]) -> tuple[RewardRow, ...]:
    """Strictly normalize rows, reject duplicates, and return canonical order."""

    normalized: list[RewardRow] = []
    identities: set[tuple[str, int, str]] = set()
    for raw in rows:
        if isinstance(raw, RewardRow):
            row = raw
        else:
            if set(raw) != set(ROW_FIELDS):
                raise ValueError("reward row fields do not match the canonical schema")
            row = RewardRow(**raw)
        identity = (row.algorithm, row.seed, row.sample_id)
        if identity in identities:
            raise ValueError(f"duplicate reward sample identity: {identity}")
        identities.add(identity)
        normalized.append(row)
    return tuple(
        sorted(
            normalized,
            key=lambda item: (
                item.algorithm,
                item.seed,
                item.step,
                item.prompt_id,
                item.sample_id,
            ),
        )
    )


def load_rows(path: str | Path) -> tuple[RewardRow, ...]:
    """Load strict canonical JSONL without accepting duplicate keys or NaN."""

    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise ValueError(f"blank JSONL line at {line_number}")
        value = _strict_json_loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        rows.append(value)
    return validate_rows(rows)


def load_source_status(path: str | Path) -> tuple[SourceStatus, ...]:
    value = _strict_json_load(Path(path))
    if not isinstance(value, dict) or set(value) != {"schema_version", "sources"}:
        raise ValueError("source status must contain schema_version/sources")
    if value["schema_version"] != 1 or not isinstance(value["sources"], list):
        raise ValueError("unsupported source status")
    statuses: list[SourceStatus] = []
    seen: set[tuple[str, int]] = set()
    for raw in value["sources"]:
        if not isinstance(raw, dict) or set(raw) != {
            "algorithm",
            "audit_ok",
            "committed_steps",
            "inspect_ok",
            "seed",
            "source_run",
        }:
            raise ValueError("source status entry fields are invalid")
        status = SourceStatus(**raw)
        identity = (status.algorithm, status.seed)
        if identity in seen:
            raise ValueError(f"duplicate source status: {identity}")
        seen.add(identity)
        statuses.append(status)
    return tuple(sorted(statuses, key=lambda item: (item.algorithm, item.seed)))


def canonical_rows_bytes(rows: Iterable[RewardRow | Mapping[str, object]]) -> bytes:
    normalized = validate_rows(rows)
    return b"".join(
        _canonical_json(asdict(row)) + b"\n"
        for row in normalized
    )


def canonical_source_status_bytes(statuses: Sequence[SourceStatus]) -> bytes:
    ordered = sorted(statuses, key=lambda item: (item.algorithm, item.seed))
    payload = {
        "schema_version": 1,
        "sources": [asdict(status) for status in ordered],
    }
    return _canonical_json(payload) + b"\n"


def generate_q100_evidence(
    *,
    index_path: Path = EVIDENCE_ROOT / "q100_inputs.json",
    evidence_dir: Path = EVIDENCE_ROOT,
    git_probe: Callable[[Path], Mapping[str, object]] = probe_git_identity,
) -> tuple[Path, Path]:
    """Generate the fixed canonical Q100 rows/status evidence projection."""

    directory = Path(evidence_dir)
    rows_path = directory / "q100_reward_rows.jsonl"
    status_path = directory / "q100_source_status.json"
    status_path.unlink(missing_ok=True)
    rows_path.unlink(missing_ok=True)
    candidate = evidence_candidate(git_probe(REPO_ROOT))
    rows, statuses = scan_q100_inputs(index_path)
    rows_payload = canonical_rows_bytes(rows)
    status_payload = _canonical_json(
        {
            "candidate": candidate,
            "rows_sha256": hashlib.sha256(rows_payload).hexdigest(),
            "schema_version": 1,
            "sources": [asdict(status) for status in statuses],
        }
    ) + b"\n"
    atomic_write_bytes(rows_path, rows_payload)
    atomic_write_bytes(status_path, status_payload)
    return rows_path, status_path


def _manifest_rows(
    path: Path,
    *,
    algorithm: str,
    seed: int,
    source_run: str,
) -> tuple[RewardRow, ...]:
    manifest = _strict_json_load(path)
    if not isinstance(manifest, dict) or set(manifest) != {
        "records",
        "run_id",
        "schema_version",
    }:
        raise ValueError(f"invalid authoritative manifest: {path}")
    if manifest["schema_version"] != "3" or not isinstance(
        manifest["records"],
        list,
    ):
        raise ValueError(f"unsupported authoritative manifest: {path}")
    rows: list[RewardRow] = []
    for record in manifest["records"]:
        if not isinstance(record, dict):
            raise ValueError("manifest record must be an object")
        reward_values = record.get("reward_values")
        if not isinstance(reward_values, dict):
            raise ValueError("manifest reward_values must be an object")
        rows.append(
            RewardRow(
                algorithm=algorithm,
                seed=seed,
                step=record.get("step"),
                prompt_id=record.get("prompt_id"),
                sample_id=record.get("sample_id"),
                weighted_total=reward_values.get("weighted_total"),
                source_run=source_run,
            )
        )
    return tuple(rows)


def _strict_json_load(path: Path) -> Any:
    return _strict_json_loads(path.read_text(encoding="utf-8"))


def _strict_json_loads(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(
        text,
        object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _child_path(base: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("run_dir must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("run_dir must be relative to the Q100 index")
    resolved = (base / candidate).resolve(strict=False)
    root = base.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError("run_dir escapes the Q100 evidence root")
    return resolved


if __name__ == "__main__":
    generate_q100_evidence()
