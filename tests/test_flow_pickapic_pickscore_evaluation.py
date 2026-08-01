"""Contracts for the experiment-only Flow/Pick-a-Pic PickScore evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.flow_pickapic_20260801.evaluate_pickscore import (
    EVAL_SEEDS,
    PROMPT_COUNT,
    SAMPLE_COUNT,
    SCORE_PROTOCOL,
    compare_scored_evaluations,
    score_source_evaluation,
)

_SMALL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDAT\x08\xd7c\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00"
    b"\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_hps_source(
    root: Path,
    *,
    condition: str,
    prompts_path: Path,
    invalid_image_hash: bool = False,
    invalid_grid: bool = False,
    config_sha256: str = "1" * 64,
    precision: str = "bf16",
    server_revision: str = "world-r1-fixed",
) -> None:
    root.mkdir()
    images = root / "images"
    images.mkdir()
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    rows = []
    for eval_seed in EVAL_SEEDS:
        for prompt_index, prompt in enumerate(prompts):
            image = images / f"seed_{eval_seed}_prompt_{prompt_index:03d}.png"
            image.write_bytes(_SMALL_PNG)
            row_seed = eval_seed
            if invalid_grid and eval_seed == EVAL_SEEDS[-1] and prompt_index == 63:
                row_seed = 9999
            image_sha256 = _sha256_file(image)
            if invalid_image_hash and eval_seed == EVAL_SEEDS[0] and prompt_index == 0:
                image_sha256 = "0" * 64
            rows.append(
                {
                    "condition": condition,
                    "eval_seed": row_seed,
                    "prompt_index": prompt_index,
                    "prompt_sha256": _sha256_bytes(prompt.encode()),
                    "sample_id": f"sample-{eval_seed}-{prompt_index}",
                    "reward": 0.25,
                    "image": str(image.relative_to(root)),
                    "image_sha256": image_sha256,
                }
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
        "config_sha256": config_sha256,
        "prompt_path": str(prompts_path),
        "prompt_sha256": _sha256_file(prompts_path),
        "prompt_count": PROMPT_COUNT,
        "eval_seeds": list(EVAL_SEEDS),
        "batch_size": 8,
        "num_diffusion_steps": 20,
        "precision": precision,
        "adapter_checkpoint": checkpoint,
        "reward_general": {
            "name": "reward_general",
            "params": {"server_revision": server_revision},
        },
    }
    _write_json(root / "manifest.json", manifest)
    scores = root / "scores.jsonl"
    scores.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "summary.json",
        {
            "schema_version": 1,
            "protocol": "flow_pickapic_paired_hps_v1",
            "condition": condition,
            "sample_count": SAMPLE_COUNT,
            "scores_sha256": _sha256_file(scores),
        },
    )


def _identity(*, revision: str = "a") -> dict[str, object]:
    return {
        "name": "pickscore_v1_normalized_prompt_image_cosine",
        "direction": "higher_is_better",
        "model": {
            "weight_sha256": revision * 64,
            "config_sha256": "b" * 64,
        },
        "processor": {
            "tokenizer_sha256": "c" * 64,
            "config_sha256": "d" * 64,
        },
        "software": {"evaluator_sha256": "e" * 64},
    }


class _FakeScorer:
    def __init__(
        self,
        *,
        trained_offset: float = 0.0,
        repeat_drift: float = 0.0,
    ) -> None:
        self.trained_offset = trained_offset
        self.repeat_drift = repeat_drift
        self.calls: list[tuple[tuple[str, ...], tuple[Path, ...]]] = []

    def score(
        self,
        prompts: tuple[str, ...],
        image_paths: tuple[Path, ...],
    ) -> tuple[float, ...]:
        self.calls.append((prompts, image_paths))
        assert all(path.is_file() for path in image_paths)
        offset = (
            self.trained_offset
            if image_paths and image_paths[0].parents[1].name == "trained"
            else 0.0
        )
        if len(self.calls) % 2 == 0:
            offset += self.repeat_drift
        return tuple(
            0.1 + (index % PROMPT_COUNT) / 10_000 + offset
            for index in range(len(prompts))
        )


@pytest.fixture
def prompts_path(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.txt"
    path.write_text(
        "".join(f"frozen prompt {index}\n" for index in range(PROMPT_COUNT)),
        encoding="utf-8",
    )
    return path


def test_score_reads_validated_hps_output_with_fake_backend(
    tmp_path: Path,
    prompts_path: Path,
) -> None:
    source = tmp_path / "base"
    output = tmp_path / "base_pickscore"
    _write_hps_source(source, condition="base", prompts_path=prompts_path)
    source_scores_sha256 = _sha256_file(source / "scores.jsonl")
    scorer = _FakeScorer()

    summary = score_source_evaluation(
        source_dir=source,
        prompts_path=prompts_path,
        output_dir=output,
        scorer=scorer,
        scorer_identity=_identity(),
    )

    assert summary["protocol"] == SCORE_PROTOCOL
    assert summary["condition"] == "base"
    assert summary["sample_count"] == SAMPLE_COUNT
    assert len(scorer.calls) == 2
    assert len(scorer.calls[0][0]) == SAMPLE_COUNT
    assert scorer.calls[0] == scorer.calls[1]
    assert _sha256_file(source / "scores.jsonl") == source_scores_sha256
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["scores_sha256"] == source_scores_sha256
    assert manifest["scorer"]["model"]["weight_sha256"] == "a" * 64
    assert manifest["scorer"]["inference"] == {
        "repeat_passes": 2,
        "max_repeat_abs_delta": 1e-6,
    }
    persisted_summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted_summary["repeat_passes"] == 2
    assert persisted_summary["repeat_max_abs_delta"] == 0.0
    assert persisted_summary["max_repeat_abs_delta"] == 1e-6
    rows = [
        json.loads(line)
        for line in (output / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == SAMPLE_COUNT
    assert all(row["repeat_abs_delta"] == 0.0 for row in rows)


def test_score_rejects_repeat_drift_without_partial_output(
    tmp_path: Path,
    prompts_path: Path,
) -> None:
    source = tmp_path / "base"
    output = tmp_path / "base_pickscore"
    _write_hps_source(source, condition="base", prompts_path=prompts_path)
    scorer = _FakeScorer(repeat_drift=2e-6)

    with pytest.raises(ValueError, match="repeat_max_abs_delta"):
        score_source_evaluation(
            source_dir=source,
            prompts_path=prompts_path,
            output_dir=output,
            scorer=scorer,
            scorer_identity=_identity(),
            max_repeat_abs_delta=1e-6,
        )

    assert len(scorer.calls) == 2
    assert not output.exists()


def test_score_rejects_invalid_source_image_hash_before_scorer(
    tmp_path: Path,
    prompts_path: Path,
) -> None:
    source = tmp_path / "base"
    _write_hps_source(
        source,
        condition="base",
        prompts_path=prompts_path,
        invalid_image_hash=True,
    )
    scorer = _FakeScorer()

    with pytest.raises(ValueError, match="image SHA256"):
        score_source_evaluation(
            source_dir=source,
            prompts_path=prompts_path,
            output_dir=tmp_path / "output",
            scorer=scorer,
            scorer_identity=_identity(),
        )

    assert scorer.calls == []


def test_score_rejects_noncanonical_source_grid(
    tmp_path: Path,
    prompts_path: Path,
) -> None:
    source = tmp_path / "base"
    _write_hps_source(
        source,
        condition="base",
        prompts_path=prompts_path,
        invalid_grid=True,
    )

    with pytest.raises(ValueError, match="frozen seed/prompt grid"):
        score_source_evaluation(
            source_dir=source,
            prompts_path=prompts_path,
            output_dir=tmp_path / "output",
            scorer=_FakeScorer(),
            scorer_identity=_identity(),
        )


def test_score_rejects_prompt_file_drift(
    tmp_path: Path,
    prompts_path: Path,
) -> None:
    source = tmp_path / "base"
    _write_hps_source(source, condition="base", prompts_path=prompts_path)
    prompts_path.write_text(
        prompts_path.read_text(encoding="utf-8").replace(
            "frozen prompt 0",
            "changed prompt 0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt file SHA256"):
        score_source_evaluation(
            source_dir=source,
            prompts_path=prompts_path,
            output_dir=tmp_path / "output",
            scorer=_FakeScorer(),
            scorer_identity=_identity(),
        )


def test_score_output_is_immutable(
    tmp_path: Path,
    prompts_path: Path,
) -> None:
    source = tmp_path / "base"
    output = tmp_path / "pickscore"
    _write_hps_source(source, condition="base", prompts_path=prompts_path)
    arguments = {
        "source_dir": source,
        "prompts_path": prompts_path,
        "output_dir": output,
        "scorer": _FakeScorer(),
        "scorer_identity": _identity(),
    }
    score_source_evaluation(**arguments)
    manifest_sha256 = _sha256_file(output / "manifest.json")

    with pytest.raises(FileExistsError):
        score_source_evaluation(**arguments)

    assert _sha256_file(output / "manifest.json") == manifest_sha256


def test_compare_uses_prompt_cluster_bootstrap(
    tmp_path: Path,
    prompts_path: Path,
) -> None:
    base_source = tmp_path / "base"
    trained_source = tmp_path / "trained"
    base_scored = tmp_path / "base_scored"
    trained_scored = tmp_path / "trained_scored"
    _write_hps_source(base_source, condition="base", prompts_path=prompts_path)
    _write_hps_source(
        trained_source,
        condition="trained",
        prompts_path=prompts_path,
    )
    scorer = _FakeScorer(trained_offset=0.01)
    score_source_evaluation(
        source_dir=base_source,
        prompts_path=prompts_path,
        output_dir=base_scored,
        scorer=scorer,
        scorer_identity=_identity(),
    )
    score_source_evaluation(
        source_dir=trained_source,
        prompts_path=prompts_path,
        output_dir=trained_scored,
        scorer=scorer,
        scorer_identity=_identity(),
    )

    result = compare_scored_evaluations(
        base_dir=base_scored,
        trained_dir=trained_scored,
        output=tmp_path / "comparison.json",
    )

    assert result["pair_count"] == SAMPLE_COUNT
    assert result["prompt_count"] == PROMPT_COUNT
    assert result["mean_paired_delta"] == pytest.approx(0.01)
    assert result["prompt_win_rate"] == 1.0
    assert result["cluster_bootstrap_95_ci"][0] > 0.0
    assert result["gate"] == "stage_noninferiority"
    assert result["thresholds"] == {
        "minimum_cluster_ci95_lower": -0.001,
        "minimum_prompt_win_rate": 0.45,
        "comparison": "inclusive_greater_than_or_equal",
    }
    assert result["acceptance"]["passed"] is True
    assert result["acceptance"]["interpretation"] == (
        "stage_noninferiority_only_not_positive_quality_claim"
    )

    boundary_result = compare_scored_evaluations(
        base_dir=base_scored,
        trained_dir=trained_scored,
        output=tmp_path / "boundary_comparison.json",
        minimum_cluster_ci95_lower=result["cluster_bootstrap_95_ci"][0],
        minimum_prompt_win_rate=result["prompt_win_rate"],
    )
    assert boundary_result["acceptance"] == {
        "cluster_ci95_lower_gte_minimum": True,
        "prompt_win_rate_gte_minimum": True,
        "passed": True,
        "interpretation": ("stage_noninferiority_only_not_positive_quality_claim"),
    }


def test_compare_rejects_different_scorer_identity(
    tmp_path: Path,
    prompts_path: Path,
) -> None:
    base_source = tmp_path / "base"
    trained_source = tmp_path / "trained"
    base_scored = tmp_path / "base_scored"
    trained_scored = tmp_path / "trained_scored"
    _write_hps_source(base_source, condition="base", prompts_path=prompts_path)
    _write_hps_source(
        trained_source,
        condition="trained",
        prompts_path=prompts_path,
    )
    score_source_evaluation(
        source_dir=base_source,
        prompts_path=prompts_path,
        output_dir=base_scored,
        scorer=_FakeScorer(),
        scorer_identity=_identity(revision="a"),
    )
    score_source_evaluation(
        source_dir=trained_source,
        prompts_path=prompts_path,
        output_dir=trained_scored,
        scorer=_FakeScorer(),
        scorer_identity=_identity(revision="f"),
    )

    with pytest.raises(ValueError, match="scorer_identity_sha256"):
        compare_scored_evaluations(
            base_dir=base_scored,
            trained_dir=trained_scored,
            output=tmp_path / "comparison.json",
        )


@pytest.mark.parametrize(
    "trained_source_kwargs",
    (
        {"config_sha256": "9" * 64},
        {"precision": "fp32"},
        {"server_revision": "world-r1-other"},
    ),
)
def test_compare_rejects_generation_identity_drift(
    tmp_path: Path,
    prompts_path: Path,
    trained_source_kwargs: dict[str, str],
) -> None:
    base_source = tmp_path / "base"
    trained_source = tmp_path / "trained"
    base_scored = tmp_path / "base_scored"
    trained_scored = tmp_path / "trained_scored"
    _write_hps_source(base_source, condition="base", prompts_path=prompts_path)
    _write_hps_source(
        trained_source,
        condition="trained",
        prompts_path=prompts_path,
        **trained_source_kwargs,
    )
    scorer = _FakeScorer()
    score_source_evaluation(
        source_dir=base_source,
        prompts_path=prompts_path,
        output_dir=base_scored,
        scorer=scorer,
        scorer_identity=_identity(),
    )
    score_source_evaluation(
        source_dir=trained_source,
        prompts_path=prompts_path,
        output_dir=trained_scored,
        scorer=scorer,
        scorer_identity=_identity(),
    )

    with pytest.raises(ValueError, match="generation identity"):
        compare_scored_evaluations(
            base_dir=base_scored,
            trained_dir=trained_scored,
            output=tmp_path / "comparison.json",
        )


@pytest.mark.parametrize("condition", ("base", "trained"))
def test_score_rejects_wrong_source_checkpoint_role(
    tmp_path: Path,
    prompts_path: Path,
    condition: str,
) -> None:
    source = tmp_path / condition
    _write_hps_source(source, condition=condition, prompts_path=prompts_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adapter_checkpoint"] = (
        {
            "path": "/wrong/base/checkpoint",
            "adapter_json_sha256": "3" * 64,
            "adapter_state_sha256": "4" * 64,
        }
        if condition == "base"
        else None
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(
        (TypeError, ValueError),
        match=f"{condition} source adapter_checkpoint",
    ):
        score_source_evaluation(
            source_dir=source,
            prompts_path=prompts_path,
            output_dir=tmp_path / "output",
            scorer=_FakeScorer(),
            scorer_identity=_identity(),
        )
