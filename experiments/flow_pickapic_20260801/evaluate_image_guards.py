"""Evaluate deterministic CPU-only image degradation guards.

This experiment-only evaluator consumes the immutable 128-image outputs from
``evaluate_hps.py``.  It never imports VisualRL, generates media, loads a
training checkpoint, or mutates either source evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_PROTOCOL = "flow_pickapic_paired_hps_v1"
GUARD_PROTOCOL = "flow_pickapic_image_guard_v1"
EVAL_SEEDS = (1009, 2027)
PROMPT_COUNT = 64
SAMPLE_COUNT = PROMPT_COUNT * len(EVAL_SEEDS)

MINIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE = 0.8
MAXIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE = 1.5
MAXIMUM_SATURATED_PIXEL_RATE_INCREASE = 0.05
MAXIMUM_BLACK_WHITE_OR_NEAR_CONSTANT_COUNT = 0

EXTREME_SATURATION_THRESHOLD = 0.95
BLACK_CHANNEL_MAXIMUM = 2.0 / 255.0
WHITE_CHANNEL_MINIMUM = 253.0 / 255.0
DEGENERATE_PIXEL_FRACTION = 0.99
NEAR_CONSTANT_CHANNEL_STD_MAXIMUM = 2.0 / 255.0

_SHARED_MANIFEST_IDENTITY = (
    "protocol",
    "config_sha256",
    "prompt_sha256",
    "prompt_count",
    "eval_seeds",
    "batch_size",
    "num_diffusion_steps",
    "precision",
)


@dataclass(frozen=True)
class SourceRecord:
    condition: str
    eval_seed: int
    prompt_index: int
    prompt_sha256: str
    sample_id: str
    image_path: Path
    image_relative_path: str
    image_sha256: str


@dataclass(frozen=True)
class SourceEvaluation:
    condition: str
    manifest: dict[str, Any]
    records: tuple[SourceRecord, ...]
    manifest_path: Path
    scores_path: Path
    summary_path: Path
    manifest_sha256: str
    scores_sha256: str
    summary_sha256: str
    image_grid_sha256: str


@dataclass(frozen=True)
class ImageMetrics:
    width: int
    height: int
    pixel_count: int
    sharpness: float
    extreme_saturated_pixel_count: int
    extreme_saturated_pixel_rate: float
    is_black: bool
    is_white: bool
    is_near_constant: bool

    @property
    def is_degenerate(self) -> bool:
        return self.is_black or self.is_white or self.is_near_constant


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


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"source {label} must be a JSON object")
    return payload


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"source {field} must be a lowercase SHA-256")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"source {field} must be finite numeric")
    return float(value)


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


def _reward_revision(manifest: dict[str, Any]) -> str | None:
    if "reward_general" not in manifest:
        return None
    reward = manifest["reward_general"]
    if not isinstance(reward, dict) or reward.get("name") != "reward_general":
        raise ValueError("source reward_general identity is invalid")
    params = reward.get("params")
    if not isinstance(params, dict):
        raise TypeError("source reward_general.params must be an object")
    revision = params.get("server_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("source reward_general server_revision must be non-empty")
    return revision


def _checkpoint_role_is_valid(manifest: dict[str, Any], condition: str) -> bool:
    checkpoint = manifest.get("adapter_checkpoint")
    if condition == "base":
        return checkpoint is None
    if not isinstance(checkpoint, dict):
        return False
    for field in ("adapter_json_sha256", "adapter_state_sha256"):
        _require_sha256(checkpoint.get(field), field=f"adapter_checkpoint.{field}")
    return True


def _load_source(
    source_dir: Path,
    prompts_path: Path,
    *,
    expected_condition: str,
) -> SourceEvaluation:
    if expected_condition not in {"base", "trained"}:
        raise ValueError("expected condition must be base or trained")
    root = source_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source evaluation must be a directory")
    prompts = _read_prompts(prompts_path)
    prompt_file_sha256 = _sha256_file(prompts_path)

    manifest_path = root / "manifest.json"
    scores_path = root / "scores.jsonl"
    summary_path = root / "summary.json"
    manifest = _read_json_object(manifest_path, label="manifest")
    summary = _read_json_object(summary_path, label="summary")
    if manifest.get("schema_version") != 1:
        raise ValueError("source manifest schema_version must be 1")
    if manifest.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"source manifest protocol must be {SOURCE_PROTOCOL!r}")
    if manifest.get("condition") != expected_condition:
        raise ValueError(f"source manifest condition must be {expected_condition!r}")
    if not _checkpoint_role_is_valid(manifest, expected_condition):
        raise ValueError(
            f"source {expected_condition} adapter_checkpoint role is invalid"
        )
    _require_sha256(manifest.get("config_sha256"), field="config_sha256")
    if manifest.get("prompt_sha256") != prompt_file_sha256:
        raise ValueError("prompt file SHA-256 does not match source manifest")
    if manifest.get("prompt_count") != PROMPT_COUNT:
        raise ValueError(f"source prompt_count must be {PROMPT_COUNT}")
    if manifest.get("eval_seeds") != list(EVAL_SEEDS):
        raise ValueError("source eval_seeds do not match the frozen grid")
    if type(manifest.get("batch_size")) is not int:
        raise ValueError("source batch_size must be an integer")
    if (
        type(manifest.get("num_diffusion_steps")) is not int
        or int(manifest["num_diffusion_steps"]) <= 0
    ):
        raise ValueError("source num_diffusion_steps must be positive")
    if manifest.get("precision") not in {"fp32", "fp16", "bf16"}:
        raise ValueError("source precision is invalid")
    _reward_revision(manifest)

    scores_sha256 = _sha256_file(scores_path)
    if summary.get("schema_version") != 1:
        raise ValueError("source summary schema_version must be 1")
    if summary.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("source summary protocol differs from manifest")
    if summary.get("condition") != expected_condition:
        raise ValueError("source summary condition differs from manifest")
    if summary.get("sample_count") != SAMPLE_COUNT:
        raise ValueError(f"source summary sample_count must be {SAMPLE_COUNT}")
    if summary.get("scores_sha256") != scores_sha256:
        raise ValueError("source scores SHA-256 does not match summary")

    rows = tuple(
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
    )
    if len(rows) != SAMPLE_COUNT:
        raise ValueError(f"source scores must contain exactly {SAMPLE_COUNT} rows")
    expected_keys = {
        (eval_seed, prompt_index)
        for eval_seed in EVAL_SEEDS
        for prompt_index in range(PROMPT_COUNT)
    }
    seen_keys: set[tuple[int, int]] = set()
    seen_images: set[str] = set()
    records: list[SourceRecord] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"source score row {row_number} must be an object")
        if row.get("condition") != expected_condition:
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
        if key not in expected_keys:
            raise ValueError("source scores do not match the frozen seed/prompt grid")
        if key in seen_keys:
            raise ValueError("source scores contain duplicate paired keys")
        seen_keys.add(key)
        prompt_sha256 = _sha256_bytes(prompts[prompt_index].encode())
        if row.get("prompt_sha256") != prompt_sha256:
            raise ValueError(f"source score row {row_number} prompt SHA-256 is invalid")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(
                f"source score row {row_number} sample_id must be non-empty"
            )
        _finite_number(row.get("reward"), field=f"row {row_number} reward")
        image_path, image_relative_path = _resolve_source_image(root, row.get("image"))
        if image_relative_path in seen_images:
            raise ValueError("source scores reuse an image path")
        seen_images.add(image_relative_path)
        image_sha256 = _sha256_file(image_path)
        if row.get("image_sha256") != image_sha256:
            raise ValueError(f"source score row {row_number} image SHA-256 is invalid")
        records.append(
            SourceRecord(
                condition=expected_condition,
                eval_seed=eval_seed,
                prompt_index=prompt_index,
                prompt_sha256=prompt_sha256,
                sample_id=sample_id,
                image_path=image_path,
                image_relative_path=image_relative_path,
                image_sha256=image_sha256,
            )
        )
    if seen_keys != expected_keys:
        raise ValueError("source scores do not match the frozen seed/prompt grid")
    records.sort(key=lambda item: (item.eval_seed, item.prompt_index))
    image_grid = [
        {
            "eval_seed": record.eval_seed,
            "prompt_index": record.prompt_index,
            "image": record.image_relative_path,
            "image_sha256": record.image_sha256,
        }
        for record in records
    ]
    return SourceEvaluation(
        condition=expected_condition,
        manifest=manifest,
        records=tuple(records),
        manifest_path=manifest_path,
        scores_path=scores_path,
        summary_path=summary_path,
        manifest_sha256=_sha256_file(manifest_path),
        scores_sha256=scores_sha256,
        summary_sha256=_sha256_file(summary_path),
        image_grid_sha256=_json_sha256(image_grid),
    )


def _validate_source_pair(
    base: SourceEvaluation,
    trained: SourceEvaluation,
) -> None:
    for field in _SHARED_MANIFEST_IDENTITY:
        if base.manifest.get(field) != trained.manifest.get(field):
            raise ValueError(f"base and trained source manifests differ for {field}")
    base_revision = _reward_revision(base.manifest)
    trained_revision = _reward_revision(trained.manifest)
    if (base_revision is None) != (trained_revision is None):
        raise ValueError("base and trained source scorer identity availability differs")
    if base_revision != trained_revision:
        raise ValueError("base and trained reward_general revision differs")
    for left, right in zip(base.records, trained.records, strict=True):
        left_key = (left.eval_seed, left.prompt_index)
        right_key = (right.eval_seed, right.prompt_index)
        if left_key != right_key:
            raise ValueError("base and trained source paired keys differ")
        for field in ("prompt_sha256", "sample_id"):
            if getattr(left, field) != getattr(right, field):
                raise ValueError(f"paired source mismatch for {field}: {left_key}")


def _image_metrics(record: SourceRecord) -> ImageMetrics:
    import numpy as np
    from PIL import Image

    payload = record.image_path.read_bytes()
    if _sha256_bytes(payload) != record.image_sha256:
        raise ValueError(
            f"source image changed after validation: {record.image_relative_path}"
        )
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "PNG":
            raise ValueError("source image must use the frozen PNG format")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("source image must decode to RGB")
    height, width, _ = rgb.shape
    if height < 3 or width < 3:
        raise ValueError("source image must be at least 3x3 pixels")

    grayscale = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    center = grayscale[1:-1, 1:-1]
    laplacian = (
        4.0 * center
        - grayscale[:-2, 1:-1]
        - grayscale[2:, 1:-1]
        - grayscale[1:-1, :-2]
        - grayscale[1:-1, 2:]
    )
    sharpness = float(np.var(laplacian, dtype=np.float64))

    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    saturation = np.zeros_like(maximum, dtype=np.float64)
    np.divide(
        maximum - minimum,
        maximum,
        out=saturation,
        where=maximum > 0.0,
    )
    extreme = saturation >= EXTREME_SATURATION_THRESHOLD
    pixel_count = height * width
    extreme_count = int(np.count_nonzero(extreme))
    black_fraction = float(np.mean(maximum <= BLACK_CHANNEL_MAXIMUM))
    white_fraction = float(np.mean(minimum >= WHITE_CHANNEL_MINIMUM))
    channel_std = np.std(rgb.reshape((-1, 3)), axis=0, dtype=np.float64)
    metrics = ImageMetrics(
        width=width,
        height=height,
        pixel_count=pixel_count,
        sharpness=sharpness,
        extreme_saturated_pixel_count=extreme_count,
        extreme_saturated_pixel_rate=extreme_count / pixel_count,
        is_black=black_fraction >= DEGENERATE_PIXEL_FRACTION,
        is_white=white_fraction >= DEGENERATE_PIXEL_FRACTION,
        is_near_constant=bool(np.max(channel_std) <= NEAR_CONSTANT_CHANNEL_STD_MAXIMUM),
    )
    for field in (metrics.sharpness, metrics.extreme_saturated_pixel_rate):
        if not math.isfinite(field):
            raise ValueError("computed image metric must be finite")
    return metrics


def _metrics_payload(metrics: ImageMetrics) -> dict[str, object]:
    reasons = []
    if metrics.is_black:
        reasons.append("black")
    if metrics.is_white:
        reasons.append("white")
    if metrics.is_near_constant:
        reasons.append("near_constant")
    return {
        "width": metrics.width,
        "height": metrics.height,
        "pixel_count": metrics.pixel_count,
        "laplacian_sharpness": metrics.sharpness,
        "extreme_saturated_pixel_count": metrics.extreme_saturated_pixel_count,
        "extreme_saturated_pixel_rate": metrics.extreme_saturated_pixel_rate,
        "degenerate_reasons": reasons,
        "is_degenerate": metrics.is_degenerate,
    }


def _source_identity(source: SourceEvaluation) -> dict[str, object]:
    return {
        "condition": source.condition,
        "manifest_sha256": source.manifest_sha256,
        "scores_sha256": source.scores_sha256,
        "summary_sha256": source.summary_sha256,
        "image_grid_sha256": source.image_grid_sha256,
    }


def _assert_source_files_unchanged(source: SourceEvaluation) -> None:
    expected = (
        (source.manifest_path, source.manifest_sha256),
        (source.scores_path, source.scores_sha256),
        (source.summary_path, source.summary_sha256),
    )
    for path, digest in expected:
        if _sha256_file(path) != digest:
            raise ValueError(f"source {path.name} changed during evaluation")


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_image_guards(
    *,
    base_dir: Path,
    trained_dir: Path,
    prompts_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Compare paired base/trained images and write immutable guard evidence."""

    output = output_dir.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"image guard output already exists: {output}")
    parent = output.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"image guard output parent does not exist: {parent}")
    prompts = prompts_path.expanduser().resolve(strict=True)
    base = _load_source(base_dir, prompts, expected_condition="base")
    trained = _load_source(trained_dir, prompts, expected_condition="trained")
    _validate_source_pair(base, trained)

    rows: list[dict[str, object]] = []
    base_metrics: list[ImageMetrics] = []
    trained_metrics: list[ImageMetrics] = []
    for left, right in zip(base.records, trained.records, strict=True):
        left_metrics = _image_metrics(left)
        right_metrics = _image_metrics(right)
        if (left_metrics.width, left_metrics.height) != (
            right_metrics.width,
            right_metrics.height,
        ):
            raise ValueError(
                "paired base/trained image dimensions differ for "
                f"{(left.eval_seed, left.prompt_index)}"
            )
        base_metrics.append(left_metrics)
        trained_metrics.append(right_metrics)
        rows.append(
            {
                "eval_seed": left.eval_seed,
                "prompt_index": left.prompt_index,
                "prompt_sha256": left.prompt_sha256,
                "sample_id": left.sample_id,
                "base": {
                    "image": left.image_relative_path,
                    "image_sha256": left.image_sha256,
                    **_metrics_payload(left_metrics),
                },
                "trained": {
                    "image": right.image_relative_path,
                    "image_sha256": right.image_sha256,
                    **_metrics_payload(right_metrics),
                },
                "delta": {
                    "laplacian_sharpness": (
                        right_metrics.sharpness - left_metrics.sharpness
                    ),
                    "extreme_saturated_pixel_rate": (
                        right_metrics.extreme_saturated_pixel_rate
                        - left_metrics.extreme_saturated_pixel_rate
                    ),
                },
            }
        )

    base_median_sharpness = float(
        statistics.median(item.sharpness for item in base_metrics)
    )
    trained_median_sharpness = float(
        statistics.median(item.sharpness for item in trained_metrics)
    )
    if base_median_sharpness <= 0.0:
        raise ValueError("base median sharpness must be positive")
    sharpness_ratio = trained_median_sharpness / base_median_sharpness
    base_saturated_rate = sum(
        item.extreme_saturated_pixel_count for item in base_metrics
    ) / sum(item.pixel_count for item in base_metrics)
    trained_saturated_rate = sum(
        item.extreme_saturated_pixel_count for item in trained_metrics
    ) / sum(item.pixel_count for item in trained_metrics)
    saturation_increase = trained_saturated_rate - base_saturated_rate
    base_degenerate_count = sum(item.is_degenerate for item in base_metrics)
    trained_degenerate_count = sum(item.is_degenerate for item in trained_metrics)

    sharpness_passed = (
        MINIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE
        <= sharpness_ratio
        <= MAXIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE
    )
    saturation_passed = saturation_increase <= MAXIMUM_SATURATED_PIXEL_RATE_INCREASE
    degeneracy_passed = (
        trained_degenerate_count <= MAXIMUM_BLACK_WHITE_OR_NEAR_CONSTANT_COUNT
    )
    thresholds = {
        "minimum_median_sharpness_ratio_to_base": (
            MINIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE
        ),
        "maximum_median_sharpness_ratio_to_base": (
            MAXIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE
        ),
        "maximum_saturated_pixel_rate_increase": (
            MAXIMUM_SATURATED_PIXEL_RATE_INCREASE
        ),
        "maximum_black_white_or_near_constant_count": (
            MAXIMUM_BLACK_WHITE_OR_NEAR_CONSTANT_COUNT
        ),
    }
    manifest = {
        "schema_version": 1,
        "protocol": GUARD_PROTOCOL,
        "prompt_sha256": _sha256_file(prompts),
        "prompt_count": PROMPT_COUNT,
        "eval_seeds": list(EVAL_SEEDS),
        "sample_count": SAMPLE_COUNT,
        "base_source": _source_identity(base),
        "trained_source": _source_identity(trained),
        "evaluator_sha256": _sha256_file(Path(__file__).resolve()),
        "thresholds": thresholds,
        "metric_definitions": {
            "laplacian_sharpness": (
                "variance of the 4-neighbour discrete Laplacian over "
                "BT.709 grayscale interior pixels"
            ),
            "extreme_saturated_pixel": (
                "HSV saturation >= 0.95, with black pixels assigned zero"
            ),
            "black": "at least 99% of pixels have max RGB <= 2/255",
            "white": "at least 99% of pixels have min RGB >= 253/255",
            "near_constant": (
                "maximum spatial standard deviation across RGB channels <= 2/255"
            ),
        },
    }
    metrics_bytes = b"".join(_canonical_json(row) + b"\n" for row in rows)
    manifest_bytes = _canonical_json(manifest) + b"\n"
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol": GUARD_PROTOCOL,
        "sample_count": SAMPLE_COUNT,
        "prompt_count": PROMPT_COUNT,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "metrics_sha256": _sha256_bytes(metrics_bytes),
        "sharpness": {
            "base_median": base_median_sharpness,
            "trained_median": trained_median_sharpness,
            "trained_to_base_ratio": sharpness_ratio,
            "passed": sharpness_passed,
        },
        "extreme_saturated_pixels": {
            "base_rate": base_saturated_rate,
            "trained_rate": trained_saturated_rate,
            "rate_increase": saturation_increase,
            "passed": saturation_passed,
        },
        "black_white_or_near_constant": {
            "base_count": base_degenerate_count,
            "trained_count": trained_degenerate_count,
            "trained_keys": [
                [record.eval_seed, record.prompt_index]
                for record, metrics in zip(
                    trained.records, trained_metrics, strict=True
                )
                if metrics.is_degenerate
            ],
            "passed": degeneracy_passed,
        },
        "acceptance": {
            "sharpness_passed": sharpness_passed,
            "saturation_passed": saturation_passed,
            "degeneracy_passed": degeneracy_passed,
            "implemented_image_guards_passed": (
                sharpness_passed and saturation_passed and degeneracy_passed
            ),
        },
    }
    summary_bytes = _canonical_json(summary) + b"\n"

    _assert_source_files_unchanged(base)
    _assert_source_files_unchanged(trained)
    if output.exists():
        raise FileExistsError(f"image guard output already exists: {output}")
    output.mkdir()
    try:
        _write_exclusive(output / "manifest.json", manifest_bytes)
        _write_exclusive(output / "metrics.jsonl", metrics_bytes)
        _write_exclusive(output / "summary.json", summary_bytes)
    except BaseException:
        for name in ("summary.json", "metrics.jsonl", "manifest.json"):
            (output / name).unlink(missing_ok=True)
        output.rmdir()
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare deterministic image guards over paired HPS outputs."
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--trained-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = evaluate_image_guards(
        base_dir=args.base_dir,
        trained_dir=args.trained_dir,
        prompts_path=args.prompts,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
