"""Small shared runtime contexts used across composition and training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.seed import UINT32_MAX, validate_step_seed_budget  # noqa: F401
from visual_rl.core.serialization import to_plain_dict  # noqa: F401


@dataclass(frozen=True)
class StepContext:
    """Stable identity for one rollout/update step."""

    step: int
    seed: int
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        for name in ("step", "seed", "rank", "world_size"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer, not bool")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not 0 <= self.seed <= UINT32_MAX:
            raise ValueError("seed must fit the canonical uint32 range")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")


@dataclass(frozen=True)
class ValidationCheck:
    """One structured validation/preflight check item."""

    level: Literal["error", "warning"]
    code: str
    path: str
    message: str
    volatile: bool = False


@dataclass(frozen=True)
class ResolutionContext:
    """Context available while resolving configuration paths."""

    config_path: Path
    config_dir: Path


@dataclass(frozen=True)
class ValidationContext:
    """Bounded, read-only environment check context."""

    phase: Literal["validate", "run"]
    config_dir: Path
    distributed_mode: Literal["single", "ddp"]
    world_size: Literal[1, 2]
    backend: str | None
    device: str
    timeout_s: float


@dataclass(frozen=True)
class ValidatedRuntimeEnv:
    """Launch topology parsed once by the CPU-only validator."""

    mode: Literal["single", "ddp"]
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    group_rank: int | None
    group_world_size: int | None
    master_addr: str | None
    master_port: int | None
    visible_gpu_count: int
    raw_launch_env: FrozenMapping

    def __post_init__(self) -> None:
        if not isinstance(self.raw_launch_env, FrozenMapping):
            object.__setattr__(
                self, "raw_launch_env", FrozenMapping(self.raw_launch_env)
            )


@dataclass(frozen=True)
class RuntimeBuildContext:
    """Rank-local values passed to component factories."""

    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device: Any
    precision: Literal["fp32", "fp16", "bf16"]
