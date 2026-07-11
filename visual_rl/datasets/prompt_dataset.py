"""Prompt datasets shared by image and video RL jobs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    metadata: dict[str, Any]
    epoch_tag: int | None = None


def read_prompt_file(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def prompt_content_sha256(prompts: Iterable[str]) -> str:
    normalized = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    payload = "".join(f"{prompt}\n" for prompt in normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_set_snapshot(
    prompts: Iterable[str],
    *,
    split_name: str,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    normalized = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
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
    overlap = sorted(set(train["prompt_ids"]).intersection(heldout["prompt_ids"]))
    if overlap:
        raise ValueError(
            f"training and held-out prompt splits overlap by {len(overlap)} prompt(s)"
        )
    return {
        "train": train,
        "heldout": heldout,
        "overlap_count": 0,
        "overlap_prompt_ids": [],
    }


class PromptDataset:
    def __init__(self, examples: Iterable[PromptExample]):
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("PromptDataset requires at least one prompt")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PromptDataset":
        if not isinstance(config, dict):
            from dataclasses import asdict

            config = asdict(config)
        prompts = config.get("prompts")
        path = config.get("path")
        repeat = int(config.get("repeat_per_prompt", 1))
        split_name = str(config.get("split_name", "train"))
        if repeat < 1:
            raise ValueError("dataset.repeat_per_prompt must be positive")
        if prompts and path:
            raise ValueError("dataset config cannot provide both prompts and path")

        if prompts:
            source_prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        elif path:
            source_prompts = read_prompt_file(path)
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

        dataset = cls(examples)
        dataset.content_sha256 = content_sha256
        dataset.split_name = split_name
        dataset.source_prompts = list(source_prompts)
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
        selected = [self.examples[(start + offset) % len(self.examples)] for offset in range(batch_size)]
        return [item.prompt for item in selected], [dict(item.metadata) for item in selected], epoch_tag

    @staticmethod
    def collate(examples: list[PromptExample]) -> tuple[int | None, list[str], list[dict[str, Any]]]:
        epoch_tags = [item.epoch_tag for item in examples]
        epoch_tag = epoch_tags[0] if all(tag == epoch_tags[0] for tag in epoch_tags) else None
        return epoch_tag, [item.prompt for item in examples], [dict(item.metadata) for item in examples]
