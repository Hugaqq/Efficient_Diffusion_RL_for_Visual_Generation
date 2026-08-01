"""Score frozen Flow/Pick-a-Pic images with an independent PickScore model.

The evaluator is deliberately outside the VisualRL runtime.  It reads images
already produced by ``evaluate_hps.py`` and never loads a generation model,
constructs an optimizer, or mutates a training run.  Torch and Transformers
are imported lazily only by the real PickScore backend; contract tests can use
a small fake backend without either dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

SOURCE_PROTOCOL = "flow_pickapic_paired_hps_v1"
SCORE_PROTOCOL = "flow_pickapic_pickscore_v1"
COMPARE_PROTOCOL = "flow_pickapic_pickscore_comparison_v1"
EVAL_SEEDS = (1009, 2027)
SOURCE_BATCH_SIZE = 8
PROMPT_COUNT = 64
SAMPLE_COUNT = PROMPT_COUNT * len(EVAL_SEEDS)
BOOTSTRAP_SEED = 729
BOOTSTRAP_REPLICATES = 10_000
REPEAT_PASSES = 2
DEFAULT_MAX_REPEAT_ABS_DELTA = 1e-6
DEFAULT_MINIMUM_CLUSTER_CI95_LOWER = -0.001
DEFAULT_MINIMUM_PROMPT_WIN_RATE = 0.45


class ScorerBackend(Protocol):
    """Minimal backend boundary used by the file-contract evaluator."""

    def score(
        self,
        prompts: Sequence[str],
        image_paths: Sequence[Path],
    ) -> Sequence[float]:
        """Return one finite text-image score for every input pair."""


@dataclass(frozen=True)
class SourceRecord:
    condition: str
    eval_seed: int
    prompt_index: int
    prompt: str
    prompt_sha256: str
    sample_id: str
    hps_reward: float
    image_path: Path
    image_relative_path: str
    image_sha256: str


@dataclass(frozen=True)
class SourceEvaluation:
    root: Path
    condition: str
    prompts_path: Path
    prompt_sha256: str
    records: tuple[SourceRecord, ...]
    generation_identity: dict[str, object]
    generation_identity_sha256: str
    adapter_checkpoint: dict[str, str] | None
    manifest_sha256: str
    scores_sha256: str
    summary_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _json_sha256(payload: object) -> str:
    return _sha256_bytes(_canonical_json(payload))


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _read_prompts(path: Path) -> tuple[str, ...]:
    prompts = tuple(path.read_text(encoding="utf-8").splitlines())
    if (
        len(prompts) != PROMPT_COUNT
        or len(set(prompts)) != PROMPT_COUNT
        or any(not prompt for prompt in prompts)
    ):
        raise ValueError(
            f"prompt file must contain {PROMPT_COUNT} unique non-empty rows"
        )
    return prompts


def _finite_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite numeric")
    return float(value)


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _validate_generation_identity(
    payload: object,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("source generation identity must be a JSON object")
    expected_fields = {
        "protocol",
        "config_sha256",
        "prompt_sha256",
        "prompt_count",
        "eval_seeds",
        "batch_size",
        "num_diffusion_steps",
        "precision",
        "reward_general",
    }
    if set(payload) != expected_fields:
        raise ValueError("source generation identity fields are incomplete")
    if payload.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"source generation protocol must be {SOURCE_PROTOCOL!r}")
    config_sha256 = _require_sha256(
        payload.get("config_sha256"),
        field="source generation config_sha256",
    )
    prompt_sha256 = _require_sha256(
        payload.get("prompt_sha256"),
        field="source generation prompt_sha256",
    )
    if payload.get("prompt_count") != PROMPT_COUNT:
        raise ValueError(f"source generation prompt_count must be {PROMPT_COUNT}")
    if payload.get("eval_seeds") != list(EVAL_SEEDS):
        raise ValueError("source generation eval_seeds do not match the frozen grid")
    if payload.get("batch_size") != SOURCE_BATCH_SIZE:
        raise ValueError(f"source generation batch_size must be {SOURCE_BATCH_SIZE}")
    num_diffusion_steps = payload.get("num_diffusion_steps")
    if type(num_diffusion_steps) is not int or num_diffusion_steps <= 0:
        raise ValueError(
            "source generation num_diffusion_steps must be a positive integer"
        )
    precision = payload.get("precision")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("source generation precision is invalid")
    reward = payload.get("reward_general")
    if not isinstance(reward, dict):
        raise TypeError("source generation reward_general must be a JSON object")
    if set(reward) != {"name", "server_revision"}:
        raise ValueError("source generation reward_general fields are incomplete")
    if reward.get("name") != "reward_general":
        raise ValueError("source generation reward_general name is invalid")
    revision = reward.get("server_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError(
            "source generation reward_general server_revision must be non-empty"
        )
    return {
        "protocol": SOURCE_PROTOCOL,
        "config_sha256": config_sha256,
        "prompt_sha256": prompt_sha256,
        "prompt_count": PROMPT_COUNT,
        "eval_seeds": list(EVAL_SEEDS),
        "batch_size": SOURCE_BATCH_SIZE,
        "num_diffusion_steps": num_diffusion_steps,
        "precision": precision,
        "reward_general": {
            "name": "reward_general",
            "server_revision": revision,
        },
    }


def _generation_identity_from_source_manifest(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    reward = manifest.get("reward_general")
    if not isinstance(reward, dict):
        raise TypeError("source manifest reward_general must be a JSON object")
    params = reward.get("params")
    if not isinstance(params, dict):
        raise TypeError("source manifest reward_general.params must be an object")
    return _validate_generation_identity(
        {
            "protocol": manifest.get("protocol"),
            "config_sha256": manifest.get("config_sha256"),
            "prompt_sha256": manifest.get("prompt_sha256"),
            "prompt_count": manifest.get("prompt_count"),
            "eval_seeds": manifest.get("eval_seeds"),
            "batch_size": manifest.get("batch_size"),
            "num_diffusion_steps": manifest.get("num_diffusion_steps"),
            "precision": manifest.get("precision"),
            "reward_general": {
                "name": reward.get("name"),
                "server_revision": params.get("server_revision"),
            },
        }
    )


def _validate_adapter_checkpoint(
    payload: object,
    *,
    condition: str,
) -> dict[str, str] | None:
    if condition == "base":
        if payload is not None:
            raise ValueError("base source adapter_checkpoint must be null")
        return None
    if condition != "trained":
        raise ValueError("source condition must be 'base' or 'trained'")
    if not isinstance(payload, dict):
        raise TypeError("trained source adapter_checkpoint must be non-null")
    path = payload.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("trained source adapter_checkpoint path must be non-empty")
    return {
        "path": path,
        "adapter_json_sha256": _require_sha256(
            payload.get("adapter_json_sha256"),
            field="adapter_checkpoint.adapter_json_sha256",
        ),
        "adapter_state_sha256": _require_sha256(
            payload.get("adapter_state_sha256"),
            field="adapter_checkpoint.adapter_state_sha256",
        ),
    }


def _resolve_source_image(root: Path, relative: object) -> tuple[Path, str]:
    if not isinstance(relative, str) or not relative:
        raise ValueError("source image must be a non-empty relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError("source image must be a relative path")
    image_path = (root / relative_path).resolve(strict=True)
    try:
        image_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("source image resolves outside the evaluation root") from exc
    if not image_path.is_file():
        raise ValueError("source image must resolve to a regular file")
    return image_path, relative_path.as_posix()


def load_source_evaluation(
    source_dir: Path,
    prompts_path: Path,
) -> SourceEvaluation:
    """Validate and load one immutable HPS evaluator output."""

    root = source_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source evaluation must be a directory")
    prompts_path = prompts_path.expanduser().resolve(strict=True)
    prompts = _read_prompts(prompts_path)
    prompt_file_sha256 = _sha256_file(prompts_path)

    manifest_path = root / "manifest.json"
    scores_path = root / "scores.jsonl"
    summary_path = root / "summary.json"
    manifest = _read_json_object(manifest_path)
    summary = _read_json_object(summary_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("source manifest schema_version must be 1")
    if manifest.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"source manifest protocol must be {SOURCE_PROTOCOL!r}")
    condition = manifest.get("condition")
    if condition not in {"base", "trained"}:
        raise ValueError("source condition must be 'base' or 'trained'")
    generation_identity = _generation_identity_from_source_manifest(manifest)
    adapter_checkpoint = _validate_adapter_checkpoint(
        manifest.get("adapter_checkpoint"),
        condition=condition,
    )
    if generation_identity["prompt_sha256"] != prompt_file_sha256:
        raise ValueError("prompt file SHA256 does not match source manifest")

    scores_sha256 = _sha256_file(scores_path)
    if summary.get("schema_version") != 1:
        raise ValueError("source summary schema_version must be 1")
    if summary.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"source summary protocol must be {SOURCE_PROTOCOL!r}")
    if summary.get("condition") != condition:
        raise ValueError("source summary condition differs from manifest")
    if summary.get("sample_count") != SAMPLE_COUNT:
        raise ValueError(f"source summary sample_count must be {SAMPLE_COUNT}")
    if summary.get("scores_sha256") != scores_sha256:
        raise ValueError("source scores SHA256 does not match summary")

    raw_rows = tuple(
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
    )
    if len(raw_rows) != SAMPLE_COUNT:
        raise ValueError(f"source scores must contain exactly {SAMPLE_COUNT} rows")

    expected_keys = {
        (eval_seed, prompt_index)
        for eval_seed in EVAL_SEEDS
        for prompt_index in range(PROMPT_COUNT)
    }
    seen_keys: set[tuple[int, int]] = set()
    records: list[SourceRecord] = []
    for row_number, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"source score row {row_number} must be a JSON object")
        if row.get("condition") != condition:
            raise ValueError(
                f"source score row {row_number} condition differs from manifest"
            )
        eval_seed = row.get("eval_seed")
        prompt_index = row.get("prompt_index")
        if type(eval_seed) is not int or type(prompt_index) is not int:
            raise ValueError(
                f"source score row {row_number} paired key must contain integers"
            )
        key = (eval_seed, prompt_index)
        if key in seen_keys:
            raise ValueError("source scores contain duplicate paired keys")
        seen_keys.add(key)
        if key not in expected_keys:
            raise ValueError("source scores do not match the frozen seed/prompt grid")

        prompt = prompts[prompt_index]
        prompt_sha256 = _sha256_bytes(prompt.encode())
        if row.get("prompt_sha256") != prompt_sha256:
            raise ValueError(f"source score row {row_number} prompt SHA256 is invalid")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(
                f"source score row {row_number} sample_id must be non-empty"
            )
        hps_reward = _finite_number(
            row.get("reward"),
            field=f"source score row {row_number} reward",
        )
        image_path, image_relative_path = _resolve_source_image(
            root,
            row.get("image"),
        )
        image_sha256 = _sha256_file(image_path)
        if row.get("image_sha256") != image_sha256:
            raise ValueError(f"source score row {row_number} image SHA256 is invalid")
        records.append(
            SourceRecord(
                condition=condition,
                eval_seed=eval_seed,
                prompt_index=prompt_index,
                prompt=prompt,
                prompt_sha256=prompt_sha256,
                sample_id=sample_id,
                hps_reward=hps_reward,
                image_path=image_path,
                image_relative_path=image_relative_path,
                image_sha256=image_sha256,
            )
        )
    if seen_keys != expected_keys:
        raise ValueError("source scores do not match the frozen seed/prompt grid")

    records.sort(key=lambda record: (record.eval_seed, record.prompt_index))
    return SourceEvaluation(
        root=root,
        condition=condition,
        prompts_path=prompts_path,
        prompt_sha256=prompt_file_sha256,
        records=tuple(records),
        generation_identity=generation_identity,
        generation_identity_sha256=_json_sha256(generation_identity),
        adapter_checkpoint=adapter_checkpoint,
        manifest_sha256=_sha256_file(manifest_path),
        scores_sha256=scores_sha256,
        summary_sha256=_sha256_file(summary_path),
    )


def _validate_scorer_identity(identity: Mapping[str, object]) -> dict[str, object]:
    copied = json.loads(_canonical_json(dict(identity)).decode())
    if not isinstance(copied, dict) or not copied:
        raise ValueError("scorer identity must be a non-empty JSON object")
    if copied.get("name") != "pickscore_v1_normalized_prompt_image_cosine":
        raise ValueError("scorer identity has an unsupported name")
    return copied


def _repeat_threshold(value: object) -> float:
    threshold = _finite_number(value, field="max_repeat_abs_delta")
    if threshold < 0.0:
        raise ValueError("max_repeat_abs_delta must be non-negative")
    return threshold


def _freeze_repeat_contract(
    identity: dict[str, object],
    *,
    max_repeat_abs_delta: float,
) -> dict[str, object]:
    inference = identity.setdefault("inference", {})
    if not isinstance(inference, dict):
        raise TypeError("scorer identity inference must be a JSON object")
    existing_passes = inference.get("repeat_passes")
    if existing_passes not in {None, REPEAT_PASSES}:
        raise ValueError(f"scorer repeat_passes must be {REPEAT_PASSES}")
    existing_threshold = inference.get("max_repeat_abs_delta")
    if existing_threshold is not None:
        existing_threshold = _repeat_threshold(existing_threshold)
        if existing_threshold != max_repeat_abs_delta:
            raise ValueError(
                "scorer identity max_repeat_abs_delta differs from score contract"
            )
    inference["repeat_passes"] = REPEAT_PASSES
    inference["max_repeat_abs_delta"] = max_repeat_abs_delta
    return identity


def _write_json_exclusive(path: Path, payload: object) -> None:
    encoded = _canonical_json(payload) + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_scores_exclusive(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def score_source_evaluation(
    *,
    source_dir: Path,
    prompts_path: Path,
    output_dir: Path,
    scorer: ScorerBackend,
    scorer_identity: Mapping[str, object],
    max_repeat_abs_delta: float = DEFAULT_MAX_REPEAT_ABS_DELTA,
) -> dict[str, object]:
    """Score one validated HPS output without touching its images or metadata."""

    output = output_dir.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"PickScore output already exists: {output}")
    source = load_source_evaluation(source_dir, prompts_path)
    threshold = _repeat_threshold(max_repeat_abs_delta)
    identity = _freeze_repeat_contract(
        _validate_scorer_identity(scorer_identity),
        max_repeat_abs_delta=threshold,
    )
    identity_sha256 = _json_sha256(identity)
    prompts = tuple(record.prompt for record in source.records)
    image_paths = tuple(record.image_path for record in source.records)
    pass_values: list[tuple[float, ...]] = []
    for pass_index in range(REPEAT_PASSES):
        raw_values = tuple(scorer.score(prompts, image_paths))
        if len(raw_values) != SAMPLE_COUNT:
            raise ValueError(
                f"scorer pass {pass_index + 1} must return exactly "
                f"{SAMPLE_COUNT} values"
            )
        pass_values.append(
            tuple(
                _finite_number(
                    value,
                    field=f"PickScore pass {pass_index + 1} value {index}",
                )
                for index, value in enumerate(raw_values)
            )
        )
    scores = pass_values[0]
    repeat_deltas = tuple(
        abs(second - first)
        for first, second in zip(pass_values[0], pass_values[1], strict=True)
    )
    repeat_max_abs_delta = max(repeat_deltas)
    if repeat_max_abs_delta > threshold:
        raise ValueError(
            "PickScore repeat_max_abs_delta exceeds max_repeat_abs_delta: "
            f"{repeat_max_abs_delta} > {threshold}"
        )

    rows = tuple(
        {
            "condition": record.condition,
            "eval_seed": record.eval_seed,
            "prompt_index": record.prompt_index,
            "prompt_sha256": record.prompt_sha256,
            "sample_id": record.sample_id,
            "source_image": record.image_relative_path,
            "source_image_sha256": record.image_sha256,
            "source_hps_reward": record.hps_reward,
            "pickscore": score,
            "repeat_abs_delta": repeat_delta,
        }
        for record, score, repeat_delta in zip(
            source.records,
            scores,
            repeat_deltas,
            strict=True,
        )
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol": SCORE_PROTOCOL,
        "condition": source.condition,
        "prompt_count": PROMPT_COUNT,
        "sample_count": SAMPLE_COUNT,
        "eval_seeds": list(EVAL_SEEDS),
        "prompt_sha256": source.prompt_sha256,
        "source": {
            "root": str(source.root),
            "manifest_sha256": source.manifest_sha256,
            "scores_sha256": source.scores_sha256,
            "summary_sha256": source.summary_sha256,
            "generation_identity": source.generation_identity,
            "generation_identity_sha256": source.generation_identity_sha256,
            "adapter_checkpoint": source.adapter_checkpoint,
        },
        "scorer": identity,
        "scorer_identity_sha256": identity_sha256,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        manifest_path = output / "manifest.json"
        scores_path = output / "scores.jsonl"
        _write_json_exclusive(manifest_path, manifest)
        _write_scores_exclusive(scores_path, rows)
        summary: dict[str, object] = {
            "schema_version": 1,
            "protocol": SCORE_PROTOCOL,
            "condition": source.condition,
            "sample_count": SAMPLE_COUNT,
            "pickscore_mean": statistics.fmean(scores),
            "pickscore_std": statistics.pstdev(scores),
            "pickscore_min": min(scores),
            "pickscore_max": max(scores),
            "repeat_passes": REPEAT_PASSES,
            "repeat_max_abs_delta": repeat_max_abs_delta,
            "max_repeat_abs_delta": threshold,
            "manifest_sha256": _sha256_file(manifest_path),
            "scores_sha256": _sha256_file(scores_path),
            "scorer_identity_sha256": identity_sha256,
        }
        _write_json_exclusive(output / "summary.json", summary)
    except BaseException:
        shutil.rmtree(output)
        raise
    return summary


def _load_scored_evaluation(
    root: Path,
    *,
    expected_condition: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    root = root.expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    scores_path = root / "scores.jsonl"
    summary_path = root / "summary.json"
    manifest = _read_json_object(manifest_path)
    summary = _read_json_object(summary_path)
    if manifest.get("schema_version") != 1 or summary.get("schema_version") != 1:
        raise ValueError("scored evaluation schema_version must be 1")
    if manifest.get("protocol") != SCORE_PROTOCOL:
        raise ValueError(f"scored manifest protocol must be {SCORE_PROTOCOL!r}")
    if summary.get("protocol") != SCORE_PROTOCOL:
        raise ValueError(f"scored summary protocol must be {SCORE_PROTOCOL!r}")
    if manifest.get("condition") != expected_condition:
        raise ValueError(f"scored evaluation condition must be {expected_condition!r}")
    if summary.get("condition") != expected_condition:
        raise ValueError("scored summary condition differs from manifest")
    if manifest.get("prompt_count") != PROMPT_COUNT:
        raise ValueError(f"scored prompt_count must be {PROMPT_COUNT}")
    if manifest.get("sample_count") != SAMPLE_COUNT:
        raise ValueError(f"scored sample_count must be {SAMPLE_COUNT}")
    if manifest.get("eval_seeds") != list(EVAL_SEEDS):
        raise ValueError("scored eval_seeds do not match the frozen grid")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise TypeError("scored source identity must be a JSON object")
    source_root = source.get("root")
    if not isinstance(source_root, str) or not source_root:
        raise ValueError("scored source root must be non-empty")
    for field in ("manifest_sha256", "scores_sha256", "summary_sha256"):
        _require_sha256(source.get(field), field=f"scored source {field}")
    generation_identity = _validate_generation_identity(
        source.get("generation_identity")
    )
    generation_identity_sha256 = _json_sha256(generation_identity)
    if source.get("generation_identity_sha256") != generation_identity_sha256:
        raise ValueError("scored source generation identity SHA256 is invalid")
    _validate_adapter_checkpoint(
        source.get("adapter_checkpoint"),
        condition=expected_condition,
    )
    if manifest.get("prompt_sha256") != generation_identity["prompt_sha256"]:
        raise ValueError("scored prompt SHA256 differs from generation identity")
    identity = _validate_scorer_identity(manifest.get("scorer", {}))
    identity_sha256 = _json_sha256(identity)
    if manifest.get("scorer_identity_sha256") != identity_sha256:
        raise ValueError("scorer identity SHA256 is invalid")
    inference = identity.get("inference")
    if not isinstance(inference, dict):
        raise TypeError("scorer identity inference must be a JSON object")
    if inference.get("repeat_passes") != REPEAT_PASSES:
        raise ValueError(f"scorer repeat_passes must be {REPEAT_PASSES}")
    repeat_threshold = _repeat_threshold(inference.get("max_repeat_abs_delta"))

    manifest_sha256 = _sha256_file(manifest_path)
    scores_sha256 = _sha256_file(scores_path)
    if summary.get("sample_count") != SAMPLE_COUNT:
        raise ValueError(f"scored summary sample_count must be {SAMPLE_COUNT}")
    if summary.get("manifest_sha256") != manifest_sha256:
        raise ValueError("scored manifest SHA256 does not match summary")
    if summary.get("scores_sha256") != scores_sha256:
        raise ValueError("scored scores SHA256 does not match summary")
    if summary.get("scorer_identity_sha256") != identity_sha256:
        raise ValueError("scored summary scorer identity SHA256 is invalid")
    if summary.get("repeat_passes") != REPEAT_PASSES:
        raise ValueError(f"scored summary repeat_passes must be {REPEAT_PASSES}")
    summary_threshold = _repeat_threshold(summary.get("max_repeat_abs_delta"))
    if summary_threshold != repeat_threshold:
        raise ValueError("scored summary repeat threshold differs from scorer identity")
    summary_repeat_max = _finite_number(
        summary.get("repeat_max_abs_delta"),
        field="scored summary repeat_max_abs_delta",
    )
    if summary_repeat_max < 0.0 or summary_repeat_max > repeat_threshold:
        raise ValueError("scored summary repeat_max_abs_delta exceeds its contract")

    rows = tuple(
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
    )
    if len(rows) != SAMPLE_COUNT:
        raise ValueError(f"scored evaluation must contain {SAMPLE_COUNT} rows")
    expected_keys = {
        (eval_seed, prompt_index)
        for eval_seed in EVAL_SEEDS
        for prompt_index in range(PROMPT_COUNT)
    }
    seen: set[tuple[int, int]] = set()
    prompt_hashes: dict[int, str] = {}
    repeat_deltas: list[float] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"scored row {row_number} must be a JSON object")
        if row.get("condition") != expected_condition:
            raise ValueError(f"scored row {row_number} condition is invalid")
        eval_seed = row.get("eval_seed")
        prompt_index = row.get("prompt_index")
        if type(eval_seed) is not int or type(prompt_index) is not int:
            raise ValueError(f"scored row {row_number} key must contain integers")
        key = (eval_seed, prompt_index)
        if key in seen:
            raise ValueError("scored evaluation contains duplicate paired keys")
        seen.add(key)
        if key not in expected_keys:
            raise ValueError("scored evaluation does not match the frozen grid")
        prompt_sha256 = row.get("prompt_sha256")
        sample_id = row.get("sample_id")
        image_sha256 = row.get("source_image_sha256")
        if not isinstance(prompt_sha256, str) or not prompt_sha256:
            raise ValueError(f"scored row {row_number} prompt SHA256 is invalid")
        previous_hash = prompt_hashes.setdefault(prompt_index, prompt_sha256)
        if previous_hash != prompt_sha256:
            raise ValueError("prompt SHA256 differs across evaluation seeds")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"scored row {row_number} sample_id is invalid")
        if not isinstance(image_sha256, str) or len(image_sha256) != 64:
            raise ValueError(f"scored row {row_number} image SHA256 is invalid")
        _finite_number(
            row.get("source_hps_reward"),
            field=f"scored row {row_number} source HPS reward",
        )
        _finite_number(
            row.get("pickscore"),
            field=f"scored row {row_number} PickScore",
        )
        repeat_delta = _finite_number(
            row.get("repeat_abs_delta"),
            field=f"scored row {row_number} repeat_abs_delta",
        )
        if repeat_delta < 0.0:
            raise ValueError(
                f"scored row {row_number} repeat_abs_delta must be non-negative"
            )
        repeat_deltas.append(repeat_delta)
    if seen != expected_keys:
        raise ValueError("scored evaluation does not match the frozen grid")
    if max(repeat_deltas) != summary_repeat_max:
        raise ValueError("scored repeat deltas do not match summary")
    return manifest, rows


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compare_scored_evaluations(
    *,
    base_dir: Path,
    trained_dir: Path,
    output: Path,
    minimum_cluster_ci95_lower: float = DEFAULT_MINIMUM_CLUSTER_CI95_LOWER,
    minimum_prompt_win_rate: float = DEFAULT_MINIMUM_PROMPT_WIN_RATE,
) -> dict[str, object]:
    """Apply the staged noninferiority gate to paired PickScore matrices."""

    minimum_ci = _finite_number(
        minimum_cluster_ci95_lower,
        field="minimum_cluster_ci95_lower",
    )
    minimum_win_rate = _finite_number(
        minimum_prompt_win_rate,
        field="minimum_prompt_win_rate",
    )
    if not 0.0 <= minimum_win_rate <= 1.0:
        raise ValueError("minimum_prompt_win_rate must be in [0, 1]")

    base_manifest, base_rows = _load_scored_evaluation(
        base_dir,
        expected_condition="base",
    )
    trained_manifest, trained_rows = _load_scored_evaluation(
        trained_dir,
        expected_condition="trained",
    )
    for field in ("prompt_sha256", "eval_seeds", "scorer_identity_sha256"):
        if base_manifest.get(field) != trained_manifest.get(field):
            raise ValueError(f"base/trained scored manifests differ for {field}")
    base_source = base_manifest["source"]
    trained_source = trained_manifest["source"]
    assert isinstance(base_source, dict) and isinstance(trained_source, dict)
    if (
        base_source["generation_identity_sha256"]
        != trained_source["generation_identity_sha256"]
    ):
        raise ValueError("base/trained scored manifests differ for generation identity")

    base = {(int(row["eval_seed"]), int(row["prompt_index"])): row for row in base_rows}
    trained = {
        (int(row["eval_seed"]), int(row["prompt_index"])): row for row in trained_rows
    }
    if set(base) != set(trained):
        raise ValueError("base/trained PickScore matrices have different keys")

    deltas_by_prompt: dict[int, list[float]] = {}
    pairs: list[dict[str, object]] = []
    for key in sorted(base):
        left = base[key]
        right = trained[key]
        for identity_field in ("prompt_sha256", "sample_id"):
            if left.get(identity_field) != right.get(identity_field):
                raise ValueError(
                    f"paired PickScore mismatch for {identity_field}: {key}"
                )
        delta = float(right["pickscore"]) - float(left["pickscore"])
        deltas_by_prompt.setdefault(key[1], []).append(delta)
        pairs.append(
            {
                "eval_seed": key[0],
                "prompt_index": key[1],
                "base_pickscore": float(left["pickscore"]),
                "trained_pickscore": float(right["pickscore"]),
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
            prompt_deltas[index] for index in rng.choices(prompt_ids, k=PROMPT_COUNT)
        )
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    mean_delta = statistics.fmean(prompt_deltas.values())
    win_rate = statistics.fmean(
        1.0 if value > 0.0 else 0.0 for value in prompt_deltas.values()
    )
    ci_low = _quantile(bootstrap, 0.025)
    ci_high = _quantile(bootstrap, 0.975)
    ci_passed = ci_low >= minimum_ci
    win_rate_passed = win_rate >= minimum_win_rate
    result: dict[str, object] = {
        "schema_version": 1,
        "protocol": COMPARE_PROTOCOL,
        "pair_count": len(pairs),
        "prompt_count": len(prompt_deltas),
        "eval_seeds": list(EVAL_SEEDS),
        "scorer_identity_sha256": base_manifest["scorer_identity_sha256"],
        "mean_paired_delta": mean_delta,
        "median_prompt_delta": statistics.median(prompt_deltas.values()),
        "prompt_win_rate": win_rate,
        "cluster_bootstrap_95_ci": [ci_low, ci_high],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "gate": "stage_noninferiority",
        "thresholds": {
            "minimum_cluster_ci95_lower": minimum_ci,
            "minimum_prompt_win_rate": minimum_win_rate,
            "comparison": "inclusive_greater_than_or_equal",
        },
        "acceptance": {
            "cluster_ci95_lower_gte_minimum": ci_passed,
            "prompt_win_rate_gte_minimum": win_rate_passed,
            "passed": ci_passed and win_rate_passed,
            "interpretation": ("stage_noninferiority_only_not_positive_quality_claim"),
        },
        "pairs": pairs,
    }
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(output, result)
    return result


def _expected_sha256(value: str, *, flag: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{flag} must be a 64-character hexadecimal SHA256")
    return normalized


def _file_within(root: Path, relative: str, *, flag: str) -> Path:
    root = root.expanduser().resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{flag} resolves outside its local model directory") from exc
    if not path.is_file():
        raise ValueError(f"{flag} must resolve to a regular file")
    return path


def _distribution_identity(name: str) -> dict[str, str]:
    distribution = importlib.metadata.distribution(name)
    record = distribution.read_text("RECORD")
    if record is None:
        raise RuntimeError(f"installed distribution {name!r} has no RECORD")
    return {
        "version": distribution.version,
        "record_sha256": _sha256_bytes(record.encode()),
    }


def build_pickscore_identity(args: argparse.Namespace) -> dict[str, object]:
    """Validate local scorer assets and construct their frozen identity."""

    model_root = args.model_path.expanduser().resolve(strict=True)
    processor_root = args.processor_path.expanduser().resolve(strict=True)
    model_weight = _file_within(
        model_root,
        args.model_weight_file,
        flag="--model-weight-file",
    )
    model_config = _file_within(
        model_root,
        args.model_config_file,
        flag="--model-config-file",
    )
    tokenizer = _file_within(
        processor_root,
        args.tokenizer_file,
        flag="--tokenizer-file",
    )
    processor_config = _file_within(
        processor_root,
        args.processor_config_file,
        flag="--processor-config-file",
    )
    actual = {
        "model_weight": _sha256_file(model_weight),
        "model_config": _sha256_file(model_config),
        "tokenizer": _sha256_file(tokenizer),
        "processor_config": _sha256_file(processor_config),
    }
    expected = {
        "model_weight": _expected_sha256(
            args.expected_model_weight_sha256,
            flag="--expected-model-weight-sha256",
        ),
        "model_config": _expected_sha256(
            args.expected_model_config_sha256,
            flag="--expected-model-config-sha256",
        ),
        "tokenizer": _expected_sha256(
            args.expected_tokenizer_sha256,
            flag="--expected-tokenizer-sha256",
        ),
        "processor_config": _expected_sha256(
            args.expected_processor_config_sha256,
            flag="--expected-processor-config-sha256",
        ),
    }
    for name, expected_sha256 in expected.items():
        if actual[name] != expected_sha256:
            raise ValueError(f"{name} SHA256 does not match the expected value")

    return {
        "name": "pickscore_v1_normalized_prompt_image_cosine",
        "direction": "higher_is_better",
        "model": {
            "path": str(model_root),
            "weight_file": args.model_weight_file,
            "weight_sha256": actual["model_weight"],
            "config_file": args.model_config_file,
            "config_sha256": actual["model_config"],
        },
        "processor": {
            "path": str(processor_root),
            "tokenizer_file": args.tokenizer_file,
            "tokenizer_sha256": actual["tokenizer"],
            "config_file": args.processor_config_file,
            "config_sha256": actual["processor_config"],
        },
        "inference": {
            "batch_size": args.batch_size,
            "device": args.device,
            "dtype": args.dtype,
            "local_files_only": True,
            "normalization": "float32_l2_cosine",
            "repeat_passes": REPEAT_PASSES,
            "max_repeat_abs_delta": _repeat_threshold(args.max_repeat_abs_delta),
        },
        "software": {
            "python": platform.python_version(),
            "evaluator_sha256": _sha256_file(Path(__file__).resolve()),
            "torch": _distribution_identity("torch"),
            "transformers": _distribution_identity("transformers"),
            "pillow": _distribution_identity("Pillow"),
        },
    }


class TransformersPickScoreBackend:
    """Local-only PickScore backend with lazy heavy imports."""

    def __init__(
        self,
        *,
        model_path: Path,
        processor_path: Path,
        device: str,
        dtype: str,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype_value = {
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self._torch = torch
        self._device = torch.device(device)
        self._dtype = dtype_value
        self._batch_size = batch_size
        self._processor = AutoProcessor.from_pretrained(
            str(processor_path),
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=dtype_value,
        ).to(self._device)
        self._model.eval()

    def score(
        self,
        prompts: Sequence[str],
        image_paths: Sequence[Path],
    ) -> Sequence[float]:
        if len(prompts) != len(image_paths):
            raise ValueError("prompts and image_paths must have equal length")
        from PIL import Image

        torch = self._torch
        scores: list[float] = []
        for start in range(0, len(prompts), self._batch_size):
            prompt_batch = prompts[start : start + self._batch_size]
            path_batch = image_paths[start : start + self._batch_size]
            images = []
            try:
                for path in path_batch:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                image_inputs = self._processor(
                    images=images,
                    return_tensors="pt",
                )
                text_inputs = self._processor(
                    text=list(prompt_batch),
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    image_features = self._model.get_image_features(
                        pixel_values=image_inputs["pixel_values"].to(
                            device=self._device,
                            dtype=self._dtype,
                        )
                    ).float()
                    text_features = self._model.get_text_features(
                        input_ids=text_inputs["input_ids"].to(self._device),
                        attention_mask=text_inputs["attention_mask"].to(self._device),
                    ).float()
                    image_features = image_features / image_features.norm(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-12)
                    text_features = text_features / text_features.norm(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-12)
                    batch_scores = (image_features * text_features).sum(dim=-1)
                scores.extend(float(value) for value in batch_scores.cpu().tolist())
            finally:
                for image in images:
                    image.close()
        return scores


def _score_command(args: argparse.Namespace) -> None:
    identity = build_pickscore_identity(args)
    scorer = TransformersPickScoreBackend(
        model_path=args.model_path.expanduser().resolve(strict=True),
        processor_path=args.processor_path.expanduser().resolve(strict=True),
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    summary = score_source_evaluation(
        source_dir=args.source_dir,
        prompts_path=args.prompts,
        output_dir=args.output_dir,
        scorer=scorer,
        scorer_identity=identity,
        max_repeat_abs_delta=args.max_repeat_abs_delta,
    )
    print(json.dumps({"completed_pickscore": summary}, sort_keys=True), flush=True)


def _compare_command(args: argparse.Namespace) -> None:
    result = compare_scored_evaluations(
        base_dir=args.base_dir,
        trained_dir=args.trained_dir,
        output=args.output,
        minimum_cluster_ci95_lower=args.minimum_cluster_ci95_lower,
        minimum_prompt_win_rate=args.minimum_prompt_win_rate,
    )
    print(json.dumps({"comparison": result["acceptance"]}, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score")
    score.add_argument("--source-dir", type=Path, required=True)
    score.add_argument("--prompts", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--model-path", type=Path, required=True)
    score.add_argument("--processor-path", type=Path, required=True)
    score.add_argument("--model-weight-file", default="model.safetensors")
    score.add_argument("--model-config-file", default="config.json")
    score.add_argument("--tokenizer-file", default="tokenizer.json")
    score.add_argument("--processor-config-file", default="preprocessor_config.json")
    score.add_argument("--expected-model-weight-sha256", required=True)
    score.add_argument("--expected-model-config-sha256", required=True)
    score.add_argument("--expected-tokenizer-sha256", required=True)
    score.add_argument("--expected-processor-config-sha256", required=True)
    score.add_argument("--device", required=True)
    score.add_argument("--dtype", choices=("float16", "float32"), required=True)
    score.add_argument("--batch-size", type=int, default=16)
    score.add_argument(
        "--max-repeat-abs-delta",
        type=float,
        default=DEFAULT_MAX_REPEAT_ABS_DELTA,
    )
    score.set_defaults(handler=_score_command)

    compare = commands.add_parser("compare")
    compare.add_argument("--base-dir", type=Path, required=True)
    compare.add_argument("--trained-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument(
        "--minimum-cluster-ci95-lower",
        type=float,
        default=DEFAULT_MINIMUM_CLUSTER_CI95_LOWER,
    )
    compare.add_argument(
        "--minimum-prompt-win-rate",
        type=float,
        default=DEFAULT_MINIMUM_PROMPT_WIN_RATE,
    )
    compare.set_defaults(handler=_compare_command)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
