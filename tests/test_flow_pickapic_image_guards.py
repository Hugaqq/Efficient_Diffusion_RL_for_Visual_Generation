"""Contracts for the read-only Flow/Pick-a-Pic image degradation guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.flow_pickapic_20260801.evaluate_image_guards import (
    EVAL_SEEDS,
    MAXIMUM_BLACK_WHITE_OR_NEAR_CONSTANT_COUNT,
    MAXIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE,
    MAXIMUM_SATURATED_PIXEL_RATE_INCREASE,
    MINIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE,
    evaluate_image_guards,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _checkerboard() -> np.ndarray:
    values = (np.indices((8, 8)).sum(axis=0) % 2 * 160 + 48).astype(np.uint8)
    return np.repeat(values[:, :, None], 3, axis=2)


def _write_source(
    root: Path,
    *,
    condition: str,
    prompts: tuple[str, ...],
    image: np.ndarray,
) -> None:
    root.mkdir()
    images = root / "images"
    images.mkdir()
    rows = []
    for eval_seed in EVAL_SEEDS:
        for prompt_index, prompt in enumerate(prompts):
            image_path = images / f"seed_{eval_seed}_prompt_{prompt_index:03d}.png"
            Image.fromarray(image).save(image_path, format="PNG", optimize=False)
            rows.append(
                {
                    "condition": condition,
                    "eval_seed": eval_seed,
                    "prompt_index": prompt_index,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "sample_id": f"sample-{eval_seed}-{prompt_index}",
                    "reward": 0.5,
                    "image": str(image_path.relative_to(root)),
                    "image_sha256": _sha256(image_path),
                }
            )
    scores_path = root / "scores.jsonl"
    scores_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    checkpoint = None
    if condition == "trained":
        checkpoint = {
            "path": "/immutable/trained/checkpoint",
            "adapter_json_sha256": "3" * 64,
            "adapter_state_sha256": "4" * 64,
        }
    manifest = {
        "schema_version": 1,
        "protocol": "flow_pickapic_paired_hps_v1",
        "condition": condition,
        "config_sha256": "1" * 64,
        "prompt_sha256": hashlib.sha256(
            ("\n".join(prompts) + "\n").encode()
        ).hexdigest(),
        "prompt_count": 64,
        "eval_seeds": list(EVAL_SEEDS),
        "batch_size": 8,
        "num_diffusion_steps": 20,
        "precision": "bf16",
        "adapter_checkpoint": checkpoint,
        "reward_general": {
            "name": "reward_general",
            "params": {"server_revision": "world-r1-fixed"},
        },
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(
        root / "summary.json",
        {
            "schema_version": 1,
            "protocol": "flow_pickapic_paired_hps_v1",
            "condition": condition,
            "sample_count": 128,
            "scores_sha256": _sha256(scores_path),
        },
    )


@pytest.fixture
def paired_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    prompts = tuple(f"unique prompt {index}" for index in range(64))
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    image = _checkerboard()
    _write_source(base, condition="base", prompts=prompts, image=image)
    _write_source(trained, condition="trained", prompts=prompts, image=image)
    return base, trained, prompts_path


def test_image_guard_writes_reproducible_bound_paired_evidence(
    paired_sources: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    base, trained, prompts = paired_sources
    output_one = tmp_path / "guard-one"
    output_two = tmp_path / "guard-two"

    summary = evaluate_image_guards(
        base_dir=base,
        trained_dir=trained,
        prompts_path=prompts,
        output_dir=output_one,
    )
    evaluate_image_guards(
        base_dir=base,
        trained_dir=trained,
        prompts_path=prompts,
        output_dir=output_two,
    )

    assert summary["acceptance"]["implemented_image_guards_passed"] is True
    assert summary["sharpness"]["trained_to_base_ratio"] == pytest.approx(1.0)
    assert summary["extreme_saturated_pixels"]["rate_increase"] == 0.0
    assert summary["black_white_or_near_constant"]["trained_count"] == 0
    assert len((output_one / "metrics.jsonl").read_text().splitlines()) == 128
    manifest = json.loads((output_one / "manifest.json").read_text())
    assert manifest["base_source"]["manifest_sha256"] == _sha256(base / "manifest.json")
    assert manifest["base_source"]["scores_sha256"] == _sha256(base / "scores.jsonl")
    assert len(manifest["base_source"]["image_grid_sha256"]) == 64
    assert summary["manifest_sha256"] == _sha256(output_one / "manifest.json")
    assert summary["metrics_sha256"] == _sha256(output_one / "metrics.jsonl")
    for name in ("manifest.json", "metrics.jsonl", "summary.json"):
        assert (output_one / name).read_bytes() == (output_two / name).read_bytes()


def test_image_guard_detects_blur_saturation_and_degenerate_output(
    paired_sources: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    base, trained, prompts = paired_sources
    solid_red = np.zeros((8, 8, 3), dtype=np.uint8)
    solid_red[:, :, 0] = 255
    for image_path in sorted((trained / "images").glob("*.png")):
        Image.fromarray(solid_red).save(image_path, format="PNG", optimize=False)
    rows = [
        json.loads(line) for line in (trained / "scores.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row["image_sha256"] = _sha256(trained / str(row["image"]))
    (trained / "scores.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    source_summary = json.loads((trained / "summary.json").read_text())
    source_summary["scores_sha256"] = _sha256(trained / "scores.jsonl")
    _write_json(trained / "summary.json", source_summary)

    summary = evaluate_image_guards(
        base_dir=base,
        trained_dir=trained,
        prompts_path=prompts,
        output_dir=tmp_path / "failed-guards",
    )

    assert summary["acceptance"] == {
        "sharpness_passed": False,
        "saturation_passed": False,
        "degeneracy_passed": False,
        "implemented_image_guards_passed": False,
    }
    assert summary["black_white_or_near_constant"]["trained_count"] == 128


def test_image_guard_rejects_image_hash_drift_without_creating_output(
    paired_sources: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    base, trained, prompts = paired_sources
    target = next((trained / "images").glob("*.png"))
    target.write_bytes(target.read_bytes() + b"drift")
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="image SHA-256"):
        evaluate_image_guards(
            base_dir=base,
            trained_dir=trained,
            prompts_path=prompts,
            output_dir=output,
        )

    assert not output.exists()


def test_image_guard_rejects_paired_identity_drift(
    paired_sources: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    base, trained, prompts = paired_sources
    scores_path = trained / "scores.jsonl"
    rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    rows[0]["sample_id"] += "-drift"
    scores_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    source_summary = json.loads((trained / "summary.json").read_text())
    source_summary["scores_sha256"] = _sha256(scores_path)
    _write_json(trained / "summary.json", source_summary)

    with pytest.raises(ValueError, match="sample_id"):
        evaluate_image_guards(
            base_dir=base,
            trained_dir=trained,
            prompts_path=prompts,
            output_dir=tmp_path / "guard",
        )


def test_image_guard_output_is_exclusive(
    paired_sources: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    base, trained, prompts = paired_sources
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_image_guards(
            base_dir=base,
            trained_dir=trained,
            prompts_path=prompts,
            output_dir=output,
        )


def test_supported_thresholds_match_frozen_staged_gate() -> None:
    gate_path = (
        Path(__file__).parents[1]
        / "experiments/flow_pickapic_20260801/staged_quality_gate_v1.json"
    )
    image_guard = json.loads(gate_path.read_text())["image_guard"]

    assert image_guard["minimum_median_sharpness_ratio_to_base"] == (
        MINIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE
    )
    assert image_guard["maximum_median_sharpness_ratio_to_base"] == (
        MAXIMUM_MEDIAN_SHARPNESS_RATIO_TO_BASE
    )
    assert image_guard["maximum_saturated_pixel_rate_increase"] == (
        MAXIMUM_SATURATED_PIXEL_RATE_INCREASE
    )
    assert image_guard["maximum_black_white_or_near_constant_count"] == (
        MAXIMUM_BLACK_WHITE_OR_NEAR_CONSTANT_COUNT
    )
