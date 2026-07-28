"""The sole prompt-occurrence dataset used by the training path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import TYPE_CHECKING

from visual_rl.core.types import FrozenMapping

if TYPE_CHECKING:
    from visual_rl.configs.schema import DatasetConfig


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    prompt_id: str
    metadata: FrozenMapping

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("PromptExample.prompt must be non-empty")
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("PromptExample.prompt_id must be non-empty")
        if not isinstance(self.metadata, FrozenMapping):
            object.__setattr__(self, "metadata", FrozenMapping(self.metadata))


class PromptDataset:
    """Map absolute logical positions to deterministic prompt occurrences."""

    _SAMPLING_STRATEGIES = {"sequential", "deterministic_shuffle"}

    def __init__(
        self,
        examples: tuple[PromptExample, ...],
        *,
        sampling_strategy: str,
        sampling_seed: int,
    ) -> None:
        if type(examples) is not tuple or not examples:
            raise ValueError("PromptDataset requires a non-empty example tuple")
        if any(not isinstance(item, PromptExample) for item in examples):
            raise TypeError("examples must contain PromptExample values")
        if sampling_strategy not in self._SAMPLING_STRATEGIES:
            raise ValueError(
                "sampling_strategy must be sequential or deterministic_shuffle"
            )
        if type(sampling_seed) is not int:
            raise TypeError("sampling_seed must be an integer, not bool")
        self.examples = examples
        self.sampling_strategy = sampling_strategy
        self.sampling_seed = sampling_seed
        self._closed = False

    @classmethod
    def from_config(cls, config: "DatasetConfig") -> "PromptDataset":
        from visual_rl.configs.schema import DatasetConfig

        if not isinstance(config, DatasetConfig):
            raise TypeError("PromptDataset.from_config requires DatasetConfig")
        if config.prompts is not None:
            prompts = config.prompts
        elif config.path is not None:
            prompts = _read_prompt_file(
                config.path,
                empty_prompt_policy=config.empty_prompt_policy,
            )
        else:  # guarded by the canonical schema
            raise ValueError("dataset requires exactly one prompt source")
        if not prompts:
            raise ValueError("dataset prompt source is empty")
        if config.require_unique and len(set(prompts)) != len(prompts):
            raise ValueError("dataset prompt source contains duplicate prompts")

        examples = tuple(
            PromptExample(
                prompt=prompt,
                prompt_id=_prompt_id(prompt),
                metadata=FrozenMapping({"split": config.split}),
            )
            for prompt in prompts
            for _ in range(config.repeat_per_prompt)
        )
        return cls(
            examples,
            sampling_strategy=config.sampling_strategy,
            sampling_seed=config.sampling_seed,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PromptExample:
        return self.examples[index]

    def batch(
        self,
        start: int,
        batch_size: int,
    ) -> tuple[tuple[str, ...], tuple[FrozenMapping, ...]]:
        if self._closed:
            raise RuntimeError("PromptDataset is closed")
        if type(start) is not int or start < 0:
            raise ValueError("dataset batch start must be a non-negative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("dataset batch_size must be a positive integer")

        prompts: list[str] = []
        metadata: list[FrozenMapping] = []
        permutations: dict[int, tuple[int, ...]] = {}
        size = len(self.examples)
        for position in range(start, start + batch_size):
            dataset_epoch, epoch_offset = divmod(position, size)
            if self.sampling_strategy == "deterministic_shuffle":
                permutation = permutations.get(dataset_epoch)
                if permutation is None:
                    values = list(range(size))
                    random.Random(
                        self.sampling_seed + dataset_epoch
                    ).shuffle(values)
                    permutation = tuple(values)
                    permutations[dataset_epoch] = permutation
                dataset_index = permutation[epoch_offset]
            else:
                dataset_index = epoch_offset

            example = self.examples[dataset_index]
            row = dict(example.metadata)
            reserved = {
                "dataset_epoch",
                "dataset_index",
                "prompt_id",
                "group_id",
            }
            overlap = reserved.intersection(row)
            if overlap:
                raise ValueError(
                    f"prompt metadata uses reserved keys: {sorted(overlap)}"
                )
            row.update(
                {
                    "dataset_epoch": dataset_epoch,
                    "dataset_index": dataset_index,
                    "prompt_id": example.prompt_id,
                    "group_id": _group_id(
                        example.prompt_id,
                        dataset_epoch=dataset_epoch,
                        dataset_index=dataset_index,
                    ),
                }
            )
            prompts.append(example.prompt)
            metadata.append(FrozenMapping(row))
        return tuple(prompts), tuple(metadata)

    def close(self) -> None:
        self._closed = True


def _read_prompt_file(
    path: Path,
    *,
    empty_prompt_policy: str,
) -> tuple[str, ...]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            prompt = line.strip()
            if not prompt:
                if empty_prompt_policy == "error":
                    raise ValueError(
                        f"empty prompt row at line {line_number} in {path}"
                    )
                continue
            prompts.append(prompt)
    return tuple(prompts)


def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def _group_id(
    prompt_id: str,
    *,
    dataset_epoch: int,
    dataset_index: int,
) -> str:
    value = f"{prompt_id}:{dataset_epoch}:{dataset_index}".encode("utf-8")
    return f"group-{hashlib.sha256(value).hexdigest()[:24]}"
