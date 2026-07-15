"""Build an auditable RGB prompt dataset from TempFlow prompt corpora."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from typing import Iterable


COLORS = ("red", "green", "blue")
TOKEN_PATTERN = re.compile(r"[a-z]+")
BLOCKED_TERMS = (
    "nsfw",
    "nude",
    "nudity",
    "naked",
    "porn",
    "pornographic",
    "sexual",
    "erotic",
    "gore",
    "gruesome",
    "decapitated",
    "blood",
    "bloody",
    "bleeding",
    "corpse",
    "murder",
    "kill",
    "killed",
    "slaughter",
    "weapon",
    "weapons",
    "gun",
    "guns",
    "rifle",
    "knife",
    "death",
    "dead",
    "bloodied",
    "bloodstains",
    "killer",
    "undead",
    "zombie",
    "violence",
    "violent",
    "terror",
    "girl",
    "girls",
    "boy",
    "boys",
    "baby",
    "underwear",
    "underware",
    "lingerie",
    "cigarette",
    "smoking",
    "drug",
)
MINOR_PATTERNS = (
    re.compile(r"\b(?:child|children|kid|kids|toddler|preteen|teenage|teenager)\b"),
    re.compile(r"\b(?:schoolgirl|schoolgirls|schoolboy|schoolboys)\b"),
    re.compile(r"\b(?:young|little)\s+(?:girl|boy)\b"),
    re.compile(r"\b(?:[0-9]|1[0-7])\s*(?:yo|y/o|year[ -]old)\b"),
)
MALFORMED_PATTERNS = (
    re.compile(r",\s*,"),
    re.compile(r"(\b\w+\b)(?:\s*,?\s*\1){3,}", re.IGNORECASE),
)
SOURCE_CATALOG = {
    "pickscore": {
        "upstream": "https://huggingface.co/datasets/pickapic-anonymous/pickapic_v1",
        "derivation": "TempFlow-GRPO/dataset/pickscore/prpocess.py extracts unique captions.",
        "license_status": "Verify the pinned upstream dataset card before redistribution.",
    },
    "hpsv2": {
        "upstream": "https://github.com/tgxs002/HPSv2",
        "derivation": "Prompt files distributed with the TempFlow-GRPO snapshot.",
        "license_status": "HPSv2 code is Apache-2.0; prompt-file provenance still requires attribution review.",
    },
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_prompt(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def _target_color(prompt: str) -> str | None:
    tokens = set(TOKEN_PATTERN.findall(prompt.lower()))
    matches = [color for color in COLORS if color in tokens]
    return matches[0] if len(matches) == 1 else None


def _safety_reasons(prompt: str) -> list[str]:
    lower = prompt.lower()
    tokens = set(TOKEN_PATTERN.findall(lower))
    reasons = []
    if any(term in tokens for term in BLOCKED_TERMS):
        reasons.append("blocked_term")
    if any(pattern.search(lower) for pattern in MINOR_PATTERNS):
        reasons.append("minor_reference")
    return reasons


def _quality_reasons(prompt: str) -> list[str]:
    reasons = []
    if any(pattern.search(prompt) for pattern in MALFORMED_PATTERNS):
        reasons.append("malformed_or_repetitive")
    punctuation_count = sum(not char.isalnum() and not char.isspace() for char in prompt)
    if punctuation_count / max(1, len(prompt)) > 0.18:
        reasons.append("excessive_punctuation")
    return reasons


def _is_safe(prompt: str) -> bool:
    return not _safety_reasons(prompt)


def _read_candidates(
    source_files: Iterable[tuple[str, str, Path]],
    *,
    min_words: int,
    max_words: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    stats = Counter()
    for source_name, source_split, path in source_files:
        for source_line, raw_prompt in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(),
            start=1,
        ):
            stats["raw_lines"] += 1
            prompt = _normalize_prompt(raw_prompt)
            word_count = len(TOKEN_PATTERN.findall(prompt.lower()))
            color = _target_color(prompt)
            if not prompt or color is None:
                stats["rejected_color_contract"] += 1
                continue
            if word_count < min_words or word_count > max_words:
                stats["rejected_length"] += 1
                continue
            safety_reasons = _safety_reasons(prompt)
            if safety_reasons:
                stats["rejected_safety"] += 1
                for reason in safety_reasons:
                    stats[f"rejected_safety_{reason}"] += 1
                continue
            quality_reasons = _quality_reasons(prompt)
            if quality_reasons:
                stats["rejected_quality"] += 1
                for reason in quality_reasons:
                    stats[f"rejected_quality_{reason}"] += 1
                continue
            dedupe_key = prompt.casefold()
            if dedupe_key in seen:
                stats["rejected_duplicate"] += 1
                continue
            seen.add(dedupe_key)
            records.append(
                {
                    "prompt": prompt,
                    "prompt_id": _sha256_bytes(prompt.encode("utf-8")),
                    "target_color": color,
                    "source_dataset": source_name,
                    "source_split": source_split,
                    "source_file": f"TempFlow-GRPO/dataset/{source_name}/{source_split}.txt",
                    "source_line": source_line,
                    "word_count": word_count,
                }
            )
            stats["accepted"] += 1
    return records, dict(stats)


def _group_by_color(records: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["target_color"])].append(record)
    return grouped


def _shuffle(records: list[dict[str, object]], seed: str) -> list[dict[str, object]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _select_main_splits(
    records: list[dict[str, object]],
    *,
    train_per_color: int,
    heldout_per_color: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped = _group_by_color(records)
    train: list[dict[str, object]] = []
    heldout: list[dict[str, object]] = []
    required = train_per_color + heldout_per_color
    for color in COLORS:
        candidates = _shuffle(grouped[color], f"{seed}:main:{color}")
        if len(candidates) < required:
            raise ValueError(
                f"Not enough {color} prompts: need {required}, found {len(candidates)}"
            )
        heldout.extend(candidates[:heldout_per_color])
        train.extend(candidates[heldout_per_color:required])
    return _shuffle(train, f"{seed}:train"), _shuffle(heldout, f"{seed}:heldout")


def _select_external(
    records: list[dict[str, object]],
    *,
    per_color: int,
    seed: int,
    excluded_prompt_ids: set[str],
) -> list[dict[str, object]]:
    grouped = _group_by_color(
        record
        for record in records
        if str(record["prompt_id"]) not in excluded_prompt_ids
    )
    selected: list[dict[str, object]] = []
    for color in COLORS:
        candidates = _shuffle(grouped[color], f"{seed}:external:{color}")
        if len(candidates) < per_color:
            raise ValueError(
                f"Not enough external {color} prompts: need {per_color}, found {len(candidates)}"
            )
        selected.extend(candidates[:per_color])
    return _shuffle(selected, f"{seed}:external")


def _select_fast_panel(
    records: list[dict[str, object]],
    *,
    per_color: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped = _group_by_color(records)
    selected: list[dict[str, object]] = []
    for color in COLORS:
        candidates = _shuffle(grouped[color], f"{seed}:fast:{color}")
        if len(candidates) < per_color:
            raise ValueError(
                f"Not enough fast-panel {color} prompts: need {per_color}, found {len(candidates)}"
            )
        selected.extend(candidates[:per_color])
    return _shuffle(selected, f"{seed}:fast")


def _write_split(output_dir: Path, name: str, records: list[dict[str, object]]) -> dict[str, object]:
    txt_path = output_dir / f"{name}.txt"
    jsonl_path = output_dir / f"{name}.jsonl"
    txt_payload = "".join(f"{record['prompt']}\n" for record in records)
    jsonl_payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    txt_path.write_text(txt_payload, encoding="utf-8")
    jsonl_path.write_text(jsonl_payload, encoding="utf-8")
    return {
        "count": len(records),
        "colors": dict(Counter(str(record["target_color"]) for record in records)),
        "sources": dict(Counter(str(record["source_dataset"]) for record in records)),
        "txt": txt_path.name,
        "txt_sha256": _sha256_bytes(txt_payload.encode("utf-8")),
        "jsonl": jsonl_path.name,
        "jsonl_sha256": _sha256_bytes(jsonl_payload.encode("utf-8")),
    }


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    pickscore_dir = Path(args.pickscore_dir).resolve()
    hpsv2_dir = Path(args.hpsv2_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sources = [
        ("pickscore", "train", pickscore_dir / "train.txt"),
        ("hpsv2", "train", hpsv2_dir / "train.txt"),
    ]
    test_sources = [
        ("pickscore", "test", pickscore_dir / "test.txt"),
        ("hpsv2", "test", hpsv2_dir / "test.txt"),
    ]
    for _name, _split, path in [*train_sources, *test_sources]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_pool, train_filter_stats = _read_candidates(
        train_sources,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    test_pool, test_filter_stats = _read_candidates(
        test_sources,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    train, heldout = _select_main_splits(
        train_pool,
        train_per_color=args.train_per_color,
        heldout_per_color=args.heldout_per_color,
        seed=args.seed,
    )
    selected_ids = {
        str(record["prompt_id"])
        for record in [*train, *heldout]
    }
    external = _select_external(
        test_pool,
        per_color=args.external_per_color,
        seed=args.seed,
        excluded_prompt_ids=selected_ids,
    )
    heldout_fast = _select_fast_panel(
        heldout,
        per_color=args.fast_per_color,
        seed=args.seed,
    )
    heldout_pilot = _select_fast_panel(
        heldout,
        per_color=args.pilot_per_color,
        seed=args.seed + 1,
    )

    split_ids = {
        "train": {str(record["prompt_id"]) for record in train},
        "heldout": {str(record["prompt_id"]) for record in heldout},
        "external_test": {str(record["prompt_id"]) for record in external},
    }
    overlap = {
        "train_heldout": len(split_ids["train"] & split_ids["heldout"]),
        "train_external": len(split_ids["train"] & split_ids["external_test"]),
        "heldout_external": len(split_ids["heldout"] & split_ids["external_test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Generated splits overlap: {overlap}")

    manifest: dict[str, object] = {
        "dataset_name": getattr(args, "dataset_name", "tempflow_real_rgb_v2"),
        "purpose": "Research-only online diffusion RL prompt dataset for RGB-control validation.",
        "reuse_scope": "Internal research experiment pending final upstream license review.",
        "seed": args.seed,
        "source_snapshot_revision": args.source_revision,
        "selection": {
            "target_colors": list(COLORS),
            "require_exactly_one_target_color": True,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "blocked_terms": list(BLOCKED_TERMS),
            "minor_patterns": [pattern.pattern for pattern in MINOR_PATTERNS],
            "quality_patterns": [pattern.pattern for pattern in MALFORMED_PATTERNS],
            "dedupe": "NFKC + whitespace normalization + casefold exact match",
        },
        "sources": {
            name: {
                **SOURCE_CATALOG[name],
                "files": {
                    split: {
                        "path": f"TempFlow-GRPO/dataset/{name}/{split}.txt",
                        "sha256": _sha256_file(path),
                    }
                    for source_name, split, path in [*train_sources, *test_sources]
                    if source_name == name
                },
            }
            for name in SOURCE_CATALOG
        },
        "filter_stats": {
            "upstream_train": train_filter_stats,
            "upstream_test": test_filter_stats,
        },
        "overlap": overlap,
        "splits": {
            "train": _write_split(output_dir, "train", train),
            "heldout": _write_split(output_dir, "heldout", heldout),
            "heldout_fast": {
                **_write_split(output_dir, "heldout_fast", heldout_fast),
                "subset_of": "heldout",
            },
            "heldout_pilot": {
                **_write_split(output_dir, "heldout_pilot", heldout_pilot),
                "subset_of": "heldout",
            },
            "external_test": _write_split(output_dir, "external_test", external),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickscore-dir", required=True)
    parser.add_argument("--hpsv2-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--train-per-color", type=int, default=1200)
    parser.add_argument("--heldout-per-color", type=int, default=200)
    parser.add_argument("--external-per-color", type=int, default=20)
    parser.add_argument("--fast-per-color", type=int, default=30)
    parser.add_argument("--pilot-per-color", type=int, default=4)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--dataset-name", default="tempflow_real_rgb_v2")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build_dataset(parse_args()), indent=2, ensure_ascii=False, sort_keys=True))
