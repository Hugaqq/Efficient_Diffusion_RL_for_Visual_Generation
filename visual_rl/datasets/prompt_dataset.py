"""Prompt datasets shared by image and video RL jobs."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
from pathlib import Path
import random
import re
from typing import Any, Iterable
import unicodedata


DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.92
DEFAULT_MAX_NEAR_DUPLICATE_COMPARISONS = 100_000
DEFAULT_MAX_PROMPT_AUDIT_CHARS = 4_096
DEFAULT_MAX_LEAKAGE_REPORT_PAIRS = 1_000


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    metadata: dict[str, Any]
    epoch_tag: int | None = None


def read_prompt_file(
    path: str | Path, *, empty_prompt_policy: str = "error"
) -> list[str]:
    _validate_empty_prompt_policy(empty_prompt_policy)
    prompts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            prompt = line.strip()
            if not prompt:
                if empty_prompt_policy == "error":
                    raise ValueError(
                        f"empty prompt row at line {line_number} in {path}"
                    )
                continue
            prompts.append(prompt)
    return prompts


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def normalize_prompt_text(prompt: str) -> str:
    """Normalize prompt text for split audits without changing training input."""

    normalized = unicodedata.normalize("NFKC", str(prompt)).casefold().strip()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def normalized_prompt_id(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt_text(prompt).encode("utf-8")).hexdigest()


def prompt_split_leakage_report(
    train_prompts: Iterable[str],
    heldout_prompts: Iterable[str],
    *,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    max_comparisons: int = DEFAULT_MAX_NEAR_DUPLICATE_COMPARISONS,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_AUDIT_CHARS,
    max_report_pairs: int = DEFAULT_MAX_LEAKAGE_REPORT_PAIRS,
) -> dict[str, Any]:
    """Audit exact, normalized and near-duplicate leakage within hard bounds.

    Exact and normalized overlap are linear-time set operations. The fuzzy
    audit rejects an oversized Cartesian product instead of silently sampling
    it, so a large split can never be reported as clean without a complete
    bounded comparison.
    """

    if isinstance(near_duplicate_threshold, bool):
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    try:
        near_duplicate_threshold = float(near_duplicate_threshold)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("near_duplicate_threshold must be in (0, 1]") from None
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    for name, value in (
        ("max_comparisons", max_comparisons),
        ("max_prompt_chars", max_prompt_chars),
        ("max_report_pairs", max_report_pairs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    train = [str(prompt).strip() for prompt in train_prompts if str(prompt).strip()]
    heldout = [
        str(prompt).strip() for prompt in heldout_prompts if str(prompt).strip()
    ]
    comparison_count = len(train) * len(heldout)
    if comparison_count > int(max_comparisons):
        raise ValueError(
            "near-duplicate prompt audit exceeds max_comparisons: "
            f"{comparison_count} > {int(max_comparisons)}"
        )

    train_ids = [prompt_id(prompt) for prompt in train]
    heldout_ids = [prompt_id(prompt) for prompt in heldout]
    train_normalized = [normalize_prompt_text(prompt) for prompt in train]
    heldout_normalized = [normalize_prompt_text(prompt) for prompt in heldout]
    for split, prompts in (
        ("train", train_normalized),
        ("heldout", heldout_normalized),
    ):
        oversized = [
            index
            for index, prompt in enumerate(prompts)
            if len(prompt) > int(max_prompt_chars)
        ]
        if oversized:
            raise ValueError(
                f"{split} prompt audit text exceeds max_prompt_chars at indices "
                f"{oversized[:10]}"
            )
    train_normalized_ids = [normalized_prompt_id(prompt) for prompt in train]
    heldout_normalized_ids = [normalized_prompt_id(prompt) for prompt in heldout]

    exact_ids = sorted(set(train_ids).intersection(heldout_ids))
    normalized_ids = set(train_normalized_ids).intersection(heldout_normalized_ids)
    exact_id_set = set(exact_ids)
    normalized_pairs: list[dict[str, Any]] = []
    near_pairs: list[dict[str, Any]] = []
    normalized_count = 0
    near_count = 0
    report_limit = int(max_report_pairs)
    for train_index, (train_norm, train_raw_id, train_norm_id) in enumerate(
        zip(train_normalized, train_ids, train_normalized_ids, strict=True)
    ):
        for heldout_index, (heldout_norm, heldout_raw_id, heldout_norm_id) in enumerate(
            zip(
                heldout_normalized,
                heldout_ids,
                heldout_normalized_ids,
                strict=True,
            )
        ):
            if train_raw_id in exact_id_set and train_raw_id == heldout_raw_id:
                continue
            if train_norm_id in normalized_ids and train_norm_id == heldout_norm_id:
                normalized_count += 1
                if len(normalized_pairs) < report_limit:
                    normalized_pairs.append(
                        {
                            "train_index": train_index,
                            "heldout_index": heldout_index,
                            "train_prompt_id": train_raw_id,
                            "heldout_prompt_id": heldout_raw_id,
                            "normalized_prompt_id": train_norm_id,
                        }
                    )
                continue
            similarity = SequenceMatcher(None, train_norm, heldout_norm).ratio()
            if similarity >= near_duplicate_threshold:
                near_count += 1
                if len(near_pairs) < report_limit:
                    near_pairs.append(
                        {
                            "train_index": train_index,
                            "heldout_index": heldout_index,
                            "train_prompt_id": train_raw_id,
                            "heldout_prompt_id": heldout_raw_id,
                            "similarity": similarity,
                        }
                    )

    report = {
        "near_duplicate_threshold": near_duplicate_threshold,
        "exact_overlap_count": len(exact_ids),
        "exact_overlap_prompt_ids": exact_ids,
        "normalized_overlap_count": normalized_count,
        "normalized_overlap_pairs": normalized_pairs,
        "near_duplicate_count": near_count,
        "near_duplicate_pairs": near_pairs,
        "leakage_detected": bool(exact_ids or normalized_count or near_count),
    }
    if normalized_count > len(normalized_pairs) or near_count > len(near_pairs):
        report["report_truncated"] = True
    return report


def prompt_content_sha256(prompts: Iterable[str]) -> str:
    normalized = _normalize_prompts(prompts)
    payload = "".join(f"{prompt}\n" for prompt in normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_set_snapshot(
    prompts: Iterable[str],
    *,
    split_name: str,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    normalized = _normalize_prompts(prompts)
    ids = [prompt_id(prompt) for prompt in normalized]
    return {
        "split": str(split_name),
        "source_path": None if source_path is None else str(source_path),
        "count": len(normalized),
        "unique_count": len(set(normalized)),
        "content_sha256": prompt_content_sha256(normalized),
        "prompt_ids": ids,
        "prompts": normalized,
    }


def validate_prompt_splits(
    train_prompts: Iterable[str],
    heldout_prompts: Iterable[str],
    *,
    train_path: str | Path | None = None,
    heldout_path: str | Path | None = None,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    max_near_duplicate_comparisons: int = DEFAULT_MAX_NEAR_DUPLICATE_COMPARISONS,
) -> dict[str, Any]:
    train = prompt_set_snapshot(
        train_prompts,
        split_name="train",
        source_path=train_path,
    )
    heldout = prompt_set_snapshot(
        heldout_prompts,
        split_name="heldout",
        source_path=heldout_path,
    )
    if train["count"] != train["unique_count"]:
        raise ValueError("training prompt split contains duplicates")
    if heldout["count"] != heldout["unique_count"]:
        raise ValueError("held-out prompt split contains duplicates")
    leakage = prompt_split_leakage_report(
        train["prompts"],
        heldout["prompts"],
        near_duplicate_threshold=near_duplicate_threshold,
        max_comparisons=max_near_duplicate_comparisons,
    )
    if leakage["leakage_detected"]:
        raise ValueError(
            "training and held-out prompt splits overlap/leak: "
            f"exact={leakage['exact_overlap_count']}, "
            f"normalized={leakage['normalized_overlap_count']}, "
            f"near_duplicate={leakage['near_duplicate_count']} "
            f"at threshold={near_duplicate_threshold}"
        )
    return {
        "train": train,
        "heldout": heldout,
        "overlap_count": 0,
        "overlap_prompt_ids": [],
        "leakage_audit": leakage,
    }


class PromptDataset:
    _SAMPLING_STRATEGIES = {"sequential", "deterministic_shuffle"}

    def __init__(
        self,
        examples: Iterable[PromptExample],
        *,
        sampling_strategy: str = "sequential",
        sampling_seed: int = 0,
    ):
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("PromptDataset requires at least one prompt")
        if sampling_strategy not in self._SAMPLING_STRATEGIES:
            choices = ", ".join(sorted(self._SAMPLING_STRATEGIES))
            raise ValueError(
                f"dataset.sampling_strategy must be one of {choices}, "
                f"got {sampling_strategy!r}"
            )
        self.sampling_strategy = sampling_strategy
        self.sampling_seed = int(sampling_seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PromptDataset":
        if not isinstance(config, dict):
            from dataclasses import asdict

            config = asdict(config)
        prompts = config.get("prompts")
        path = config.get("path")
        repeat = int(config.get("repeat_per_prompt", 1))
        split_name = str(config.get("split_name", "train"))
        empty_prompt_policy = str(config.get("empty_prompt_policy", "error"))
        _validate_empty_prompt_policy(empty_prompt_policy, prefix="dataset.")
        if repeat < 1:
            raise ValueError("dataset.repeat_per_prompt must be positive")
        if prompts and path:
            raise ValueError("dataset config cannot provide both prompts and path")

        if prompts:
            source_prompts = _normalize_prompts(
                prompts,
                empty_prompt_policy=empty_prompt_policy,
                source_label="dataset inline prompts",
            )
        elif path:
            source_prompts = read_prompt_file(
                path,
                empty_prompt_policy=empty_prompt_policy,
            )
        else:
            raise ValueError("dataset config must provide either prompts or path")

        if not source_prompts:
            raise ValueError("dataset prompt source is empty")
        if config.get("require_unique") and len(set(source_prompts)) != len(
            source_prompts
        ):
            raise ValueError("dataset prompt source contains duplicate prompts")

        content_sha256 = prompt_content_sha256(source_prompts)
        expected_sha256 = config.get("content_sha256")
        if expected_sha256 and str(expected_sha256) != content_sha256:
            raise ValueError(
                "dataset prompt content SHA256 mismatch: "
                f"{content_sha256} != {expected_sha256}"
            )

        examples: list[PromptExample] = []
        for prompt in source_prompts:
            metadata = {
                "prompt_id": prompt_id(prompt),
                "split": split_name,
            }
            for _ in range(repeat):
                examples.append(PromptExample(prompt, dict(metadata)))

        dataset = cls(
            examples,
            sampling_strategy=str(
                config.get("sampling_strategy", "sequential")
            ),
            sampling_seed=int(config.get("sampling_seed", 0)),
        )
        dataset.content_sha256 = content_sha256
        dataset.split_name = split_name
        dataset.source_prompts = list(source_prompts)
        dataset.empty_prompt_policy = empty_prompt_policy
        return dataset

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int | tuple[int, int]) -> PromptExample:
        epoch_tag = None
        if isinstance(index, tuple):
            epoch_tag, index = index
            base = self.examples[index]
            return PromptExample(base.prompt, dict(base.metadata), epoch_tag)
        return self.examples[index]

    def batch(
        self, start: int, batch_size: int, epoch_tag: int | None = None
    ) -> tuple[list[str], list[dict[str, Any]], int | None]:
        if start < 0:
            raise ValueError("dataset batch start must be non-negative")
        if batch_size < 1:
            raise ValueError("dataset batch_size must be positive")

        selected: list[PromptExample] = []
        selected_positions: list[tuple[int, int]] = []
        dataset_size = len(self.examples)
        permutations: dict[int, list[int]] = {}
        for logical_position in range(start, start + batch_size):
            dataset_epoch, epoch_offset = divmod(logical_position, dataset_size)
            if self.sampling_strategy == "deterministic_shuffle":
                permutation = permutations.get(dataset_epoch)
                if permutation is None:
                    permutation = list(range(dataset_size))
                    random.Random(self.sampling_seed + dataset_epoch).shuffle(
                        permutation
                    )
                    permutations[dataset_epoch] = permutation
                source_index = permutation[epoch_offset]
            else:
                source_index = epoch_offset
            selected.append(self.examples[source_index])
            selected_positions.append((dataset_epoch, source_index))

        metadata = []
        for item, (dataset_epoch, source_index) in zip(
            selected,
            selected_positions,
            strict=True,
        ):
            item_metadata = dict(item.metadata)
            item_prompt_id = str(
                item_metadata.get("prompt_id") or prompt_id(item.prompt)
            )
            item_metadata["prompt_id"] = item_prompt_id
            item_metadata["group_id"] = _prompt_occurrence_group_id(
                item_prompt_id,
                dataset_epoch=dataset_epoch,
                source_index=source_index,
            )
            if self.sampling_strategy == "deterministic_shuffle":
                item_metadata.update(
                    {
                        "dataset_epoch": dataset_epoch,
                        "dataset_index": source_index,
                        "sampling_strategy": self.sampling_strategy,
                    }
                )
            metadata.append(item_metadata)
        return [item.prompt for item in selected], metadata, epoch_tag

    @staticmethod
    def collate(examples: list[PromptExample]) -> tuple[int | None, list[str], list[dict[str, Any]]]:
        epoch_tags = [item.epoch_tag for item in examples]
        epoch_tag = epoch_tags[0] if all(tag == epoch_tags[0] for tag in epoch_tags) else None
        return epoch_tag, [item.prompt for item in examples], [dict(item.metadata) for item in examples]


def _normalize_prompts(
    prompts: Iterable[str],
    *,
    empty_prompt_policy: str = "skip",
    source_label: str = "prompt entries",
) -> list[str]:
    _validate_empty_prompt_policy(empty_prompt_policy)
    if isinstance(prompts, (str, bytes)):
        raise TypeError("prompt entries must be an iterable of prompt strings")
    normalized = []
    for index, prompt in enumerate(prompts):
        value = str(prompt)
        if "\r" in value or "\n" in value:
            raise ValueError(
                "Inline prompt entries must not contain CR or LF characters"
            )
        value = value.strip()
        if not value:
            if empty_prompt_policy == "error":
                raise ValueError(f"{source_label} contain empty rows at indices [{index}]")
            continue
        normalized.append(value)
    return normalized


def _validate_empty_prompt_policy(policy: str, *, prefix: str = "") -> None:
    if policy not in {"error", "skip"}:
        raise ValueError(f"{prefix}empty_prompt_policy must be 'error' or 'skip'")


def _prompt_occurrence_group_id(
    item_prompt_id: str,
    *,
    dataset_epoch: int,
    source_index: int,
) -> str:
    payload = f"{item_prompt_id}:{dataset_epoch}:{source_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"group-{digest}"
