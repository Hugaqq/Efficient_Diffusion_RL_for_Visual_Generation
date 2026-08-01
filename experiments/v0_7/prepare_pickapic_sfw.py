"""Freeze deterministic prompt-only Pick-a-Pic subsets for Flow-GRPO runs.

Version 3 adds a final-test split without changing the frozen v2 training and
validation bytes.  Every selected prompt fits both conditioning token budgets
used by the experiment: SD3's T5 encoder (128 tokens) and the HPS/OpenCLIP
reward encoder (77 tokens).  Final-test selection uses only prompt eligibility
and seeded SHA-256 ordering; it never consumes model or reward outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/yifan123/flow_grpo"
SOURCE_COMMIT = "879042cf5707f8b90daa98d147d7deac2317c5da"
SOURCE_SHA256 = {
    "train": "edb46f78d7d93df0bafb30568074463822312ee339200f627195ccb2ff1d2f7a",
    "test": "d839050acb06aa045e399e76a81e4ade9a55fc743dc0e22e7b3eaeaca5c19107",
}
TRAIN_BIN_COUNTS = {"medium": 30, "long": 40, "very_long": 30}
EVAL_BIN_COUNTS = {"medium": 20, "long": 24, "very_long": 20}
FINAL_TEST_BIN_COUNTS = {"medium": 20, "long": 24, "very_long": 20}
SELECTION_SEED = 729
MAX_T5_TOKENS = 128
MAX_HPS_CLIP_TOKENS = 77
V2_OUTPUT_VERSION = 2
OUTPUT_VERSION = 3
V2_TRAIN_FILENAME = "pickapic_sfw_q100_train_v2.txt"
V2_VALIDATION_FILENAME = "pickapic_sfw_heldout_eval_v2.txt"
V2_PROVENANCE_FILENAME = "pickapic_sfw_provenance_v2.json"
FINAL_TEST_FILENAME = "pickapic_sfw_final_test_v3.txt"
PROVENANCE_FILENAME = "pickapic_sfw_provenance_v3.json"
V2_FROZEN_SHA256 = {
    "q100_train": (
        "bda5208d4f90465063861d52c401fca8b4adcf22f273b892efb5cb848279c3d7"
    ),
    "validation": (
        "26cd082a5677d5de1bfeefa8ff0da2be3e7d21d9ea35091dc7df10aae788f68d"
    ),
    "provenance": (
        "57f37eea53c2420a4355c88e2d3f86205bceb4c68d1553ed2be59dd3e6e0316b"
    ),
}
V3_FROZEN_SHA256 = {
    "final_test": (
        "fe5def8d78ff1233a371cdb029cdc14cd30d4a20baa62714b4594973f6ad58d2"
    ),
    "provenance": (
        "4b6b22afc88a5758373a130ee8fa2b3a6e5604a6b2219905cb9cb81ea4d061db"
    ),
}
FINAL_TEST_ACCESS_POLICY = "sealed_until_final_multi_seed_conclusion"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_source(path: Path, *, expected_sha256: str) -> tuple[str, ...]:
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"source digest mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    prompts = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
    )
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError(f"source contains empty prompts: {path}")
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"source contains duplicate prompts: {path}")
    return prompts


def _length_bin(prompt: str) -> str:
    words = len(prompt.split())
    if words < 6:
        raise ValueError("eligible prompts must contain at least six words")
    if words <= 15:
        return "medium"
    if words <= 30:
        return "long"
    return "very_long"


def _rank(prompt: str, *, split: str) -> str:
    return _sha256_bytes(f"{SELECTION_SEED}:{split}:{prompt}".encode())


def _eligible(
    prompts: tuple[str, ...],
    *,
    t5_tokenizer: Any,
    hps_clip_tokenizer: Any,
    excluded: frozenset[str],
) -> tuple[tuple[str, int, int, str], ...]:
    rows: list[tuple[str, int, int, str]] = []
    for prompt in prompts:
        if prompt in excluded or len(prompt.split()) < 6:
            continue
        t5_token_ids = t5_tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        hps_clip_token_ids = hps_clip_tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        t5_token_count = len(t5_token_ids)
        hps_clip_token_count = len(hps_clip_token_ids)
        if (
            t5_token_count <= MAX_T5_TOKENS
            and hps_clip_token_count <= MAX_HPS_CLIP_TOKENS
        ):
            rows.append(
                (
                    prompt,
                    t5_token_count,
                    hps_clip_token_count,
                    _length_bin(prompt),
                )
            )
    return tuple(rows)


def _select(
    rows: tuple[tuple[str, int, int, str], ...],
    *,
    split: str,
    counts: dict[str, int],
) -> tuple[tuple[str, int, int, str], ...]:
    selected: list[tuple[str, int, int, str]] = []
    for length_bin, count in counts.items():
        candidates = sorted(
            (row for row in rows if row[3] == length_bin),
            key=lambda row: _rank(row[0], split=split),
        )
        if len(candidates) < count:
            raise ValueError(
                f"not enough {length_bin} prompts for {split}: "
                f"need {count}, found {len(candidates)}"
            )
        selected.extend(candidates[:count])
    return tuple(sorted(selected, key=lambda row: _rank(row[0], split=split)))


def _write_frozen_prompts(
    path: Path,
    rows: tuple[tuple[str, int, int, str], ...],
    *,
    expected_sha256: str,
) -> str:
    payload = "".join(
        f"{prompt}\n"
        for prompt, _t5_tokens, _hps_tokens, _bin in rows
    ).encode("utf-8")
    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"frozen prompt digest changed for {path.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise ValueError(
                f"refusing to overwrite non-matching frozen file: {path}"
            )
    else:
        path.write_bytes(payload)
    return actual_sha256


def _write_frozen_json(
    path: Path,
    value: dict[str, object],
    *,
    expected_sha256: str,
) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"frozen provenance digest changed for {path.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(
                f"refusing to overwrite non-matching frozen file: {path}"
            )
    else:
        path.write_bytes(payload)


def _summary(
    rows: tuple[tuple[str, int, int, str], ...],
    *,
    sha256: str,
) -> dict[str, object]:
    bins = _bin_counts(rows)
    t5_token_counts = [tokens for _prompt, tokens, _hps, _bin in rows]
    hps_token_counts = [tokens for _prompt, _t5, tokens, _bin in rows]
    return {
        "count": len(rows),
        "length_bins": bins,
        "max_t5_tokens": max(t5_token_counts),
        "min_t5_tokens": min(t5_token_counts),
        "max_hps_clip_tokens": max(hps_token_counts),
        "min_hps_clip_tokens": min(hps_token_counts),
        "sha256": sha256,
    }


def _bin_counts(
    rows: tuple[tuple[str, int, int, str], ...],
) -> dict[str, int]:
    return {
        name: sum(
            length_bin == name
            for _prompt, _t5_tokens, _hps_tokens, length_bin in rows
        )
        for name in ("medium", "long", "very_long")
    }


def prepare(
    *,
    source_train: Path,
    source_test: Path,
    model_checkpoint: Path,
    output_dir: Path,
) -> dict[str, object]:
    from transformers import AutoTokenizer

    train_source = _read_source(
        source_train,
        expected_sha256=SOURCE_SHA256["train"],
    )
    test_source = _read_source(
        source_test,
        expected_sha256=SOURCE_SHA256["test"],
    )
    t5_tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint / "tokenizer_3",
        local_files_only=True,
    )
    hps_clip_tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint / "tokenizer",
        local_files_only=True,
    )
    t5_tokenizer_fingerprint = _sha256_bytes(
        json.dumps(
            t5_tokenizer.get_vocab(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    hps_clip_tokenizer_fingerprint = _sha256_bytes(
        json.dumps(
            hps_clip_tokenizer.get_vocab(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    train_eligible = _eligible(
        train_source,
        t5_tokenizer=t5_tokenizer,
        hps_clip_tokenizer=hps_clip_tokenizer,
        excluded=frozenset(),
    )
    test_eligible = _eligible(
        test_source,
        t5_tokenizer=t5_tokenizer,
        hps_clip_tokenizer=hps_clip_tokenizer,
        excluded=frozenset(
            prompt
            for prompt, _t5_tokens, _hps_tokens, _bin in train_eligible
        ),
    )
    train_rows = _select(
        train_eligible,
        split="q100_train",
        counts=TRAIN_BIN_COUNTS,
    )
    validation_rows = _select(
        test_eligible,
        split="heldout_eval",
        counts=EVAL_BIN_COUNTS,
    )
    validation_prompts = frozenset(row[0] for row in validation_rows)
    final_test_eligible = tuple(
        row for row in test_eligible if row[0] not in validation_prompts
    )
    final_test_rows = _select(
        final_test_eligible,
        split="final_test",
        counts=FINAL_TEST_BIN_COUNTS,
    )
    selected_prompt_sets = {
        "q100_train": {row[0] for row in train_rows},
        "validation": set(validation_prompts),
        "final_test": {row[0] for row in final_test_rows},
    }
    split_names = tuple(selected_prompt_sets)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            if selected_prompt_sets[left_name].intersection(
                selected_prompt_sets[right_name]
            ):
                raise RuntimeError(
                    f"{left_name} and {right_name} prompt selections overlap"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / V2_TRAIN_FILENAME
    validation_path = output_dir / V2_VALIDATION_FILENAME
    v2_provenance_path = output_dir / V2_PROVENANCE_FILENAME
    final_test_path = output_dir / FINAL_TEST_FILENAME
    train_sha256 = _write_frozen_prompts(
        train_path,
        train_rows,
        expected_sha256=V2_FROZEN_SHA256["q100_train"],
    )
    validation_sha256 = _write_frozen_prompts(
        validation_path,
        validation_rows,
        expected_sha256=V2_FROZEN_SHA256["validation"],
    )
    final_test_sha256 = _write_frozen_prompts(
        final_test_path,
        final_test_rows,
        expected_sha256=V3_FROZEN_SHA256["final_test"],
    )
    source_manifest = {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "train_rows": len(train_source),
        "train_sha256": SOURCE_SHA256["train"],
        "test_rows": len(test_source),
        "test_sha256": SOURCE_SHA256["test"],
    }
    tokenizer_manifest = {
        "seed": SELECTION_SEED,
        "minimum_words": 6,
        "maximum_t5_tokens": MAX_T5_TOKENS,
        "maximum_hps_clip_tokens": MAX_HPS_CLIP_TOKENS,
        "t5_tokenizer_class": type(t5_tokenizer).__name__,
        "t5_tokenizer_vocab_sha256": t5_tokenizer_fingerprint,
        "hps_clip_tokenizer_class": type(hps_clip_tokenizer).__name__,
        "hps_clip_tokenizer_vocab_sha256": hps_clip_tokenizer_fingerprint,
    }
    v2_manifest: dict[str, object] = {
        "schema_version": V2_OUTPUT_VERSION,
        "source": source_manifest,
        "selection": tokenizer_manifest,
        "outputs": {
            "q100_train": _summary(train_rows, sha256=train_sha256),
            "heldout_eval": _summary(
                validation_rows,
                sha256=validation_sha256,
            ),
        },
    }
    _write_frozen_json(
        v2_provenance_path,
        v2_manifest,
        expected_sha256=V2_FROZEN_SHA256["provenance"],
    )

    manifest: dict[str, object] = {
        "schema_version": OUTPUT_VERSION,
        "source": source_manifest,
        "selection": {
            **tokenizer_manifest,
            "final_test_candidate_count": len(final_test_eligible),
            "final_test_candidate_length_bins": _bin_counts(
                final_test_eligible
            ),
            "final_test_candidate_pool": (
                "eligible raw test prompts after excluding train-source "
                "duplicates and the frozen v2 validation prompts"
            ),
            "final_test_rank_domain": "final_test",
            "rank_function": "sha256('<seed>:<split>:<prompt>')",
            "uses_model_outputs": False,
            "uses_reward_outputs": False,
        },
        "splits": {
            "q100_train": {
                **_summary(train_rows, sha256=train_sha256),
                "artifact_version": 2,
                "path": V2_TRAIN_FILENAME,
                "role": "training",
            },
            "validation": {
                **_summary(
                    validation_rows,
                    sha256=validation_sha256,
                ),
                "artifact_version": 2,
                "former_role": "heldout_eval",
                "path": V2_VALIDATION_FILENAME,
                "role": "validation",
                "status": "used_for_c20_tuning",
            },
            "final_test": {
                **_summary(final_test_rows, sha256=final_test_sha256),
                "access_policy": FINAL_TEST_ACCESS_POLICY,
                "allowed_use": (
                    "open once for the final multi-seed conclusion after all "
                    "training and model selection are frozen"
                ),
                "artifact_version": 3,
                "path": FINAL_TEST_FILENAME,
                "role": "final_test",
                "status": "not_used_for_c20_tuning",
            },
        },
        "v2_provenance": {
            "path": V2_PROVENANCE_FILENAME,
            "sha256": V2_FROZEN_SHA256["provenance"],
        },
    }
    manifest_path = output_dir / PROVENANCE_FILENAME
    _write_frozen_json(
        manifest_path,
        manifest,
        expected_sha256=V3_FROZEN_SHA256["provenance"],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-test", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(
        source_train=args.source_train,
        source_test=args.source_test,
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
