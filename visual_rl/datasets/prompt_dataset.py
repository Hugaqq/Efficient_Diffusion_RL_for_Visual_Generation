"""Prompt datasets shared by image and video RL jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    metadata: dict[str, Any]
    epoch_tag: int | None = None


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
        examples: list[PromptExample] = []

        if prompts:
            for prompt in prompts:
                for _ in range(repeat):
                    examples.append(PromptExample(str(prompt), {}))
        elif path:
            file_path = Path(path)
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    prompt = line.strip()
                    if prompt:
                        for _ in range(repeat):
                            examples.append(PromptExample(prompt, {}))
        else:
            raise ValueError("dataset config must provide either prompts or path")

        return cls(examples)

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
