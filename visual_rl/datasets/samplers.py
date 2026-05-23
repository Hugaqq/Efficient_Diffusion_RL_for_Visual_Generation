"""Sampler utilities."""

from __future__ import annotations

import random


class KRepeatIndexSampler:
    """Prompt-wise repeat sampler for GRPO groups."""

    def __init__(self, dataset_size: int, batch_size: int, k: int, seed: int = 0):
        if batch_size % k != 0:
            raise ValueError("batch_size must be divisible by k")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.k = k
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def next_indices(self) -> list[int]:
        rng = random.Random(self.seed + self.epoch)
        unique_count = self.batch_size // self.k
        choices = [rng.randrange(self.dataset_size) for _ in range(unique_count)]
        indices = [index for index in choices for _ in range(self.k)]
        rng.shuffle(indices)
        return indices


class EpochKRepeatSampler:
    """GenRL-style repeat sampler that attaches epoch tags to indices."""

    def __init__(self, dataset_size: int, batch_size: int, k: int, num_replicas: int = 1, rank: int = 0, seed: int = 0):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = num_replicas * batch_size
        if self.total_samples % k != 0:
            raise ValueError("num_replicas * batch_size must be divisible by k")
        self.unique_per_global_batch = self.total_samples // k
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def next_indices(self) -> list[tuple[int, int]]:
        import torch

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.unique_per_global_batch <= self.dataset_size:
            indices = torch.randperm(self.dataset_size, generator=generator)[: self.unique_per_global_batch].tolist()
        else:
            indices = torch.randint(0, self.dataset_size, (self.unique_per_global_batch,), generator=generator).tolist()
        repeated = [idx for idx in indices for _ in range(self.k)]
        order = torch.randperm(len(repeated), generator=generator).tolist()
        shuffled = [repeated[i] for i in order]
        start = self.rank * self.batch_size
        end = start + self.batch_size
        return [(self.epoch, idx) for idx in shuffled[start:end]]
