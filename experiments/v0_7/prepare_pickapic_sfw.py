"""Freeze deterministic prompt-only Pick-a-Pic subsets for Flow-GRPO runs.

Version 2 keeps every prompt within both conditioning token budgets used by
the experiment: SD3's T5 encoder (128 tokens) and the HPS/OpenCLIP reward
encoder (77 tokens).  The earlier v1 files remain immutable evidence for the
bounded C20 diagnostic and are not overwritten by this script.
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
SELECTION_SEED = 729
MAX_T5_TOKENS = 128
MAX_HPS_CLIP_TOKENS = 77
OUTPUT_VERSION = 2


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


def _write_prompts(
    path: Path,
    rows: tuple[tuple[str, int, int, str], ...],
) -> str:
    payload = "".join(
        f"{prompt}\n"
        for prompt, _t5_tokens, _hps_tokens, _bin in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _summary(
    rows: tuple[tuple[str, int, int, str], ...],
    *,
    sha256: str,
) -> dict[str, object]:
    bins = {
        name: sum(
            length_bin == name
            for _prompt, _t5_tokens, _hps_tokens, length_bin in rows
        )
        for name in ("medium", "long", "very_long")
    }
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
    eval_rows = _select(
        test_eligible,
        split="heldout_eval",
        counts=EVAL_BIN_COUNTS,
    )
    if {row[0] for row in train_rows}.intersection(row[0] for row in eval_rows):
        raise RuntimeError("train and heldout prompt selections overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "pickapic_sfw_q100_train_v2.txt"
    eval_path = output_dir / "pickapic_sfw_heldout_eval_v2.txt"
    train_sha256 = _write_prompts(train_path, train_rows)
    eval_sha256 = _write_prompts(eval_path, eval_rows)
    manifest: dict[str, object] = {
        "schema_version": OUTPUT_VERSION,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "train_rows": len(train_source),
            "train_sha256": SOURCE_SHA256["train"],
            "test_rows": len(test_source),
            "test_sha256": SOURCE_SHA256["test"],
        },
        "selection": {
            "seed": SELECTION_SEED,
            "minimum_words": 6,
            "maximum_t5_tokens": MAX_T5_TOKENS,
            "maximum_hps_clip_tokens": MAX_HPS_CLIP_TOKENS,
            "t5_tokenizer_class": type(t5_tokenizer).__name__,
            "t5_tokenizer_vocab_sha256": t5_tokenizer_fingerprint,
            "hps_clip_tokenizer_class": type(hps_clip_tokenizer).__name__,
            "hps_clip_tokenizer_vocab_sha256": (
                hps_clip_tokenizer_fingerprint
            ),
        },
        "outputs": {
            "q100_train": _summary(train_rows, sha256=train_sha256),
            "heldout_eval": _summary(eval_rows, sha256=eval_sha256),
        },
    }
    manifest_path = output_dir / "pickapic_sfw_provenance_v2.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
