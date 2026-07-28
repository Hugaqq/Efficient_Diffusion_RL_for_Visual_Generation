"""Shared data contracts used by rollout, rewards, algorithms, and trainers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import math
import operator
from pathlib import Path
from typing import Any, Literal

#: Canonical seed range shared by Python/NumPy/Torch generators (plan 2.1).
UINT32_MAX = 0xFFFF_FFFF


def validate_step_seed_budget(seed: int, max_steps: int, world_size: int) -> None:
    """Reject seed budgets whose per-step seed formula can overflow uint32.

    The unique per-step seed formula is ``seed + step * world_size + rank``
    with ``step < max_steps`` and ``rank < world_size``; the largest value
    ``seed + (max_steps - 1) * world_size + (world_size - 1)`` must fit the
    canonical uint32 range. Checked before any dataset/rollout work; no
    modulo-based second seed semantics exist.
    """

    for name, value in (
        ("seed", seed),
        ("max_steps", max_steps),
        ("world_size", world_size),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer, not bool")
    if not 0 <= seed <= UINT32_MAX:
        raise ValueError("seed must fit the canonical uint32 range")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    final_seed = seed + (max_steps - 1) * world_size + (world_size - 1)
    if final_seed > UINT32_MAX:
        raise ValueError(
            "seed + (max_steps - 1) * world_size + (world_size - 1) exceeds "
            f"the canonical uint32 range: {final_seed} > {UINT32_MAX}"
        )


@dataclass(frozen=True)
class StepContext:
    """Immutable runtime identity for one rollout/update step.

    ``step``/``seed``/``rank``/``world_size`` are the canonical v0.7 identity
    fields and are validated eagerly: non-bool ints, ``step >= 0``,
    ``0 <= seed <= 0xFFFFFFFF``, ``world_size >= 1`` and
    ``0 <= rank < world_size``. ``epoch_tag``/``policy_version`` are legacy
    fields kept for current call sites; the atomic cutover removes them and
    ``step`` remains the single logical policy version.
    """

    step: int
    seed: int
    epoch_tag: int
    rank: int = 0
    world_size: int = 1
    policy_version: int = 0

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


@dataclass(init=False)
class RolloutBatch:
    """Canonical image/video rollout data with explicit sample identity."""

    prompts: list[str]
    metadata: list[dict[str, Any]]
    media: Any
    latents: Any
    next_latents: Any
    timesteps: Any
    old_log_probs: Any
    kl: Any | None
    sample_id: Any
    prompt_id: Any
    group_id: Any
    branch_id: Any
    transition_mask: Any | None
    media_layout: str | None
    context: StepContext | None
    model_metadata: dict[str, Any]
    model_tensors: dict[str, Any]
    # v0.7 typed contract fields (plan stage 2.1). They are optional during
    # the incremental phase and only meaningful for the matching rollout or
    # algorithm kind; the atomic cutover makes the full set mandatory.
    selected_timestep_index: Any
    flash_coefficient: Any
    branch_step_index: Any
    trajectory_step_index: Any
    transition_std_dev: Any
    camera_trajectory: Any
    recompute_payload: dict[str, Any]
    artifact_metadata: dict[str, Any]

    def __init__(
        self,
        prompts: list[str],
        metadata: list[dict[str, Any]],
        media: Any = None,
        latents: Any = None,
        next_latents: Any = None,
        timesteps: Any = None,
        old_log_probs: Any = None,
        kl: Any | None = None,
        *,
        sample_id: Any = None,
        prompt_id: Any = None,
        group_id: Any = None,
        branch_id: Any = None,
        transition_mask: Any | None = None,
        media_layout: str | None = None,
        context: StepContext | None = None,
        model_metadata: dict[str, Any] | None = None,
        model_tensors: dict[str, Any] | None = None,
        branch_ids: Any = None,
        seed: int | None = None,
        epoch_tag: int | None = None,
        selected_timestep_index: Any = None,
        flash_coefficient: Any = None,
        branch_step_index: Any = None,
        trajectory_step_index: Any = None,
        transition_std_dev: Any = None,
        camera_trajectory: Any = None,
        recompute_payload: dict[str, Any] | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> None:
        if branch_id is not None and branch_ids is not None:
            raise ValueError("Provide branch_id, not both branch_id and branch_ids")
        if context is None and (seed is not None or epoch_tag is not None):
            resolved_epoch = int(epoch_tag or 0)
            context = StepContext(
                step=resolved_epoch,
                seed=int(seed or 0),
                epoch_tag=resolved_epoch,
            )
        elif context is not None:
            if seed is not None and int(seed) != context.seed:
                raise ValueError("seed must match context.seed")
            if epoch_tag is not None and int(epoch_tag) != context.epoch_tag:
                raise ValueError("epoch_tag must match context.epoch_tag")

        self.prompts = list(prompts)
        self.metadata = [dict(item) for item in metadata]
        self.media = media
        self.latents = latents
        self.next_latents = next_latents
        self.timesteps = timesteps
        self.old_log_probs = old_log_probs
        self.kl = kl
        self.prompt_id = prompt_id if prompt_id is not None else self._prompt_ids()
        self._explicit_group_id_rows = _explicit_group_id_rows(
            self.metadata,
            group_id,
        )
        if context is not None:
            _validate_formal_occurrence_groups(
                self.prompt_id,
                self.metadata,
                group_id,
                explicit_rows=self._explicit_group_id_rows,
            )
        self.group_id = group_id if group_id is not None else self._group_ids()
        self.branch_id = (
            branch_id
            if branch_id is not None
            else branch_ids
            if branch_ids is not None
            else self._branch_ids()
        )
        self.sample_id = sample_id if sample_id is not None else self._sample_ids()
        self.transition_mask = (
            transition_mask
            if transition_mask is not None
            else _default_transition_mask(timesteps, old_log_probs, latents)
        )
        self.media_layout = media_layout or _infer_media_layout(media)
        self.context = context
        self.model_metadata = dict(model_metadata or {})
        self.model_tensors = dict(model_tensors or {})
        self.selected_timestep_index = selected_timestep_index
        self.flash_coefficient = flash_coefficient
        self.branch_step_index = branch_step_index
        self.trajectory_step_index = trajectory_step_index
        self.transition_std_dev = transition_std_dev
        self.camera_trajectory = camera_trajectory
        self.recompute_payload = dict(recompute_payload or {})
        self.artifact_metadata = dict(artifact_metadata or {})

    @property
    def branch_ids(self) -> Any:
        """Deprecated read-only alias for branch_id."""

        return self.branch_id

    @property
    def seed(self) -> int | None:
        """Deprecated read-only view of context.seed."""

        return None if self.context is None else self.context.seed

    @property
    def epoch_tag(self) -> int | None:
        """Deprecated read-only view of context.epoch_tag."""

        return None if self.context is None else self.context.epoch_tag

    @property
    def batch_size(self) -> int:
        return len(self.prompts)

    @property
    def shapes(self) -> dict[str, Any]:
        values = {
            name: _shape_tree(getattr(self, name))
            for name in (
                "media",
                "latents",
                "next_latents",
                "timesteps",
                "old_log_probs",
                "kl",
                "transition_mask",
            )
        }
        values["model_tensors"] = _shape_tree(self.model_tensors)
        return {name: shape for name, shape in values.items() if shape is not None}

    def to(self, device: Any, dtype: Any = None) -> RolloutBatch:
        updates = {
            name: _map_tensors(getattr(self, name), "to", device=device, dtype=dtype)
            for name in (
                "media",
                "latents",
                "next_latents",
                "timesteps",
                "old_log_probs",
                "kl",
                "branch_id",
                "transition_mask",
                "model_tensors",
                "selected_timestep_index",
                "flash_coefficient",
                "branch_step_index",
                "trajectory_step_index",
                "transition_std_dev",
                "camera_trajectory",
                "recompute_payload",
            )
        }
        return self._copy_with(**updates)

    def detach(self) -> RolloutBatch:
        updates = {
            name: _map_tensors(getattr(self, name), "detach")
            for name in (
                "media",
                "latents",
                "next_latents",
                "timesteps",
                "old_log_probs",
                "kl",
                "branch_id",
                "transition_mask",
                "model_tensors",
                "selected_timestep_index",
                "flash_coefficient",
                "branch_step_index",
                "trajectory_step_index",
                "transition_std_dev",
                "camera_trajectory",
                "recompute_payload",
            )
        }
        return self._copy_with(**updates)

    def replace(self, **updates: Any) -> RolloutBatch:
        """Return a new batch with selected fields replaced."""

        unknown = set(updates).difference(item.name for item in fields(self))
        if unknown:
            raise TypeError(f"Unknown RolloutBatch fields: {sorted(unknown)}")
        return self._copy_with(**updates)

    def slice(self, indices: Any) -> RolloutBatch:
        """Select a non-empty ordered subset along the sample axis.

        ``trajectory_step_index`` and ``artifact_metadata`` are batch-shared
        values and are carried through unchanged (never sliced, even when a
        coincidental ``T == B`` would make them look batch-indexed).
        """

        resolved = _validate_sample_indices(indices, self.batch_size)
        updates = {
            name: _slice_batch_axis(getattr(self, name), resolved, self.batch_size)
            for name in (
                "prompts",
                "metadata",
                "media",
                "latents",
                "next_latents",
                "timesteps",
                "old_log_probs",
                "kl",
                "sample_id",
                "prompt_id",
                "group_id",
                "branch_id",
                "transition_mask",
                "model_metadata",
                "model_tensors",
                "selected_timestep_index",
                "flash_coefficient",
                "branch_step_index",
                "transition_std_dev",
                "camera_trajectory",
                "recompute_payload",
            )
        }
        selected = self._copy_with(**updates)
        selected._explicit_group_id_rows = tuple(
            self._explicit_group_id_rows[index] for index in resolved
        )
        selected.validate_lightweight()
        return selected

    def select(self, indices: Any) -> RolloutBatch:
        """Alias for :meth:`slice` for index-selection call sites."""

        return self.slice(indices)

    def select_samples(self, indices: Any) -> RolloutBatch:
        """Select samples by explicit ordered indices."""

        return self.slice(indices)

    def validate_lightweight(self, strict: bool = False) -> None:
        if len(self.prompts) != len(self.metadata):
            raise ValueError("prompts and metadata must have the same length")
        for name in ("sample_id", "prompt_id", "group_id", "branch_id"):
            _check_identity(name, getattr(self, name), self.batch_size)
        if len(set(self.sample_id)) != self.batch_size:
            raise ValueError("sample_id values must be unique within a batch")
        if self.context is not None and not isinstance(self.context, StepContext):
            raise ValueError("context must be a StepContext")
        if self.context is not None:
            _validate_formal_occurrence_groups(
                self.prompt_id,
                self.metadata,
                self.group_id,
                explicit_rows=getattr(self, "_explicit_group_id_rows", ()),
            )
        if self.media_layout not in {None, "BCHW", "BFCHW"}:
            raise ValueError("media_layout must be BCHW or BFCHW")
        if self.media_layout is not None:
            expected_ndim = 4 if self.media_layout == "BCHW" else 5
            shape = getattr(self.media, "shape", None)
            if shape is None or len(shape) != expected_ndim:
                raise ValueError(
                    f"media_layout {self.media_layout} requires media with "
                    f"{expected_ndim} dimensions"
                )
            if int(shape[0]) != self.batch_size:
                raise ValueError(
                    f"media batch dimension must be {self.batch_size}, got {shape[0]}"
                )
        if self.transition_mask is not None:
            _check_bool_mask(self.transition_mask)
            self._check_batch_axis(
                "transition_mask", self.transition_mask, self.batch_size, False
            )
        if strict:
            self.validate_strict()

    def validate_strict(self) -> None:
        """Validate batch and transition dimensions before expensive work."""

        batch_size = self.batch_size
        self._check_batch_axis("media", self.media, batch_size, allow_scalar=False)
        for name in (
            "latents",
            "next_latents",
            "timesteps",
            "old_log_probs",
            "kl",
            "transition_mask",
        ):
            self._check_batch_axis(name, getattr(self, name), batch_size, False)

        transition_shapes = {
            name: _shape_tuple(getattr(self, name))
            for name in (
                "latents",
                "next_latents",
                "timesteps",
                "old_log_probs",
                "kl",
                "transition_mask",
            )
            if getattr(self, name) is not None
        }
        expected_prefix: tuple[int, int] | None = None
        for name, shape in transition_shapes.items():
            if shape is None or len(shape) < 2:
                raise ValueError(f"{name} must have [batch, steps] dimensions")
            prefix = (int(shape[0]), int(shape[1]))
            if expected_prefix is None:
                expected_prefix = prefix
            elif prefix != expected_prefix:
                raise ValueError(
                    "transition tensors must share [batch, steps] dimensions: "
                    f"{name} has {prefix}, expected {expected_prefix}"
                )
        for name in ("timesteps", "old_log_probs", "kl", "transition_mask"):
            shape = transition_shapes.get(name)
            if shape is not None and len(shape) != 2:
                raise ValueError(f"{name} must have shape [batch, steps]")

        latent_shape = _shape_tuple(self.latents)
        next_shape = _shape_tuple(self.next_latents)
        if (
            latent_shape is not None
            and next_shape is not None
            and latent_shape != next_shape
        ):
            raise ValueError(
                "latents and next_latents must have the same shape: "
                f"{latent_shape} != {next_shape}"
            )

    def _prompt_ids(self) -> list[str]:
        return [
            str(item.get("prompt_id") or _stable_digest(prompt))
            for prompt, item in zip(self.prompts, self.metadata, strict=False)
        ]

    def _group_ids(self) -> list[str]:
        return [
            str(item.get("group_id") or prompt_id)
            for prompt_id, item in zip(self.prompt_id, self.metadata, strict=False)
        ]

    def _branch_ids(self) -> list[Any]:
        return [
            item.get("branch_id", item.get("sample_index", 0)) for item in self.metadata
        ]

    def _sample_ids(self) -> list[str]:
        return [
            str(
                item.get("sample_id")
                or f"legacy-{_stable_digest(f'{group}:{branch}:{index}')[:24]}"
            )
            for index, (group, branch, item) in enumerate(
                zip(self.group_id, self.branch_id, self.metadata, strict=False)
            )
        ]

    def _copy_with(self, **updates: Any) -> RolloutBatch:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values.update(updates)
        copied = RolloutBatch(**values)
        if "group_id" not in updates and "metadata" not in updates:
            copied._explicit_group_id_rows = tuple(self._explicit_group_id_rows)
        elif "group_id" not in updates:
            copied._explicit_group_id_rows = _explicit_group_id_rows(
                copied.metadata,
                None,
            )
        return copied

    @staticmethod
    def _check_batch_axis(
        name: str, value: Any, batch_size: int, allow_scalar: bool
    ) -> None:
        if value is None:
            return
        shape = getattr(value, "shape", None)
        if shape is None:
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                if len(value) != batch_size:
                    raise ValueError(
                        f"{name} length must match batch size {batch_size}, got {len(value)}"
                    )
            return
        if len(shape) == 0:
            if allow_scalar:
                return
            raise ValueError(f"{name} must have a batch dimension")
        if int(shape[0]) != batch_size:
            raise ValueError(
                f"{name} batch dimension must be {batch_size}, got {shape[0]}"
            )


@dataclass
class RewardBatch:
    """Feedback output before training-time advantage normalization."""

    raw: dict[str, Any]
    weighted: dict[str, Any]
    weighted_total: Any
    valid_mask: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: Any = None

    @property
    def batch_size(self) -> int:
        if self.sample_id is not None:
            return len(self.sample_id)
        shape = _shape_tuple(self.weighted_total)
        if shape:
            return int(shape[0])
        return len(self.weighted_total)

    @property
    def shapes(self) -> dict[str, Any]:
        return {
            "raw": _shape_tree(self.raw),
            "weighted": _shape_tree(self.weighted),
            "weighted_total": _shape_tree(self.weighted_total),
            "valid_mask": _shape_tree(self.valid_mask),
        }

    def to(self, device: Any, dtype: Any = None) -> RewardBatch:
        return RewardBatch(
            raw=_map_tensors(self.raw, "to", device=device, dtype=dtype),
            weighted=_map_tensors(self.weighted, "to", device=device, dtype=dtype),
            weighted_total=_map_tensors(
                self.weighted_total, "to", device=device, dtype=dtype
            ),
            valid_mask=_map_tensors(self.valid_mask, "to", device=device, dtype=dtype),
            metadata=dict(self.metadata),
            sample_id=None if self.sample_id is None else list(self.sample_id),
        )

    def detach(self) -> RewardBatch:
        return RewardBatch(
            raw=_map_tensors(self.raw, "detach"),
            weighted=_map_tensors(self.weighted, "detach"),
            weighted_total=_map_tensors(self.weighted_total, "detach"),
            valid_mask=_map_tensors(self.valid_mask, "detach"),
            metadata=dict(self.metadata),
            sample_id=None if self.sample_id is None else list(self.sample_id),
        )

    def slice(self, indices: Any) -> RewardBatch:
        """Select a non-empty ordered subset along the sample axis."""

        batch_size = self.batch_size
        resolved = _validate_sample_indices(indices, batch_size)
        return RewardBatch(
            raw=_slice_batch_axis(self.raw, resolved, batch_size),
            weighted=_slice_batch_axis(self.weighted, resolved, batch_size),
            weighted_total=_slice_batch_axis(self.weighted_total, resolved, batch_size),
            valid_mask=_slice_batch_axis(self.valid_mask, resolved, batch_size),
            metadata=_slice_batch_axis(self.metadata, resolved, batch_size),
            sample_id=(
                None
                if self.sample_id is None
                else _slice_batch_axis(self.sample_id, resolved, batch_size)
            ),
        )

    def select(self, indices: Any) -> RewardBatch:
        """Alias for :meth:`slice` for index-selection call sites."""

        return self.slice(indices)

    def select_samples(self, indices: Any) -> RewardBatch:
        """Select samples by explicit ordered indices."""

        return self.slice(indices)

    def as_tensors(self) -> RewardBatch:
        """Return the canonical detached CPU tensor representation."""

        if not isinstance(self.raw, Mapping) or not isinstance(self.weighted, Mapping):
            raise TypeError("RewardBatch raw and weighted must be mappings")
        return RewardBatch(
            raw={
                name: _as_cpu_floating_tensor(f"raw.{name}", values)
                for name, values in self.raw.items()
            },
            weighted={
                name: _as_cpu_floating_tensor(f"weighted.{name}", values)
                for name, values in self.weighted.items()
            },
            weighted_total=_as_cpu_floating_tensor(
                "weighted_total", self.weighted_total
            ),
            valid_mask=_as_cpu_bool_tensor("valid_mask", self.valid_mask),
            metadata=dict(self.metadata),
            sample_id=None if self.sample_id is None else list(self.sample_id),
        )

    def canonical(self) -> RewardBatch:
        """Normalize reward fields before validation or training-time use."""

        return self.as_tensors()

    def validate_against(self, batch: RolloutBatch) -> None:
        """Validate reward order and values against one rollout batch."""

        import numpy as np

        batch.validate_lightweight()
        if self.sample_id is None:
            raise ValueError("RewardBatch.sample_id is required for validation")
        _check_identity("sample_id", self.sample_id, batch.batch_size)
        if list(self.sample_id) != list(batch.sample_id):
            raise ValueError("RewardBatch sample_id order must match RolloutBatch")
        if not isinstance(self.raw, Mapping) or not isinstance(self.weighted, Mapping):
            raise ValueError("RewardBatch raw and weighted must be mappings")
        if set(self.raw) != set(self.weighted):
            raise ValueError("RewardBatch raw and weighted keys must match")

        for group_name, values_by_name in (
            ("raw", self.raw),
            ("weighted", self.weighted),
        ):
            for name, values in values_by_name.items():
                _require_vector(f"{group_name}.{name}", values, batch.batch_size)
                _require_finite(f"{group_name}.{name}", values)
        _require_vector("weighted_total", self.weighted_total, batch.batch_size)
        _require_finite("weighted_total", self.weighted_total)
        _require_vector("valid_mask", self.valid_mask, batch.batch_size)
        _check_bool_mask(self.valid_mask)

        total = np.zeros(batch.batch_size, dtype=np.float64)
        for values in self.weighted.values():
            total += _as_numpy(values).astype(np.float64, copy=False)
        if not np.allclose(
            total,
            _as_numpy(self.weighted_total).astype(np.float64, copy=False),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("weighted_total must equal the sum of weighted rewards")


def _map_tensors(value: Any, operation: str, **kwargs: Any) -> Any:
    try:
        import torch
    except ImportError:
        return value

    if isinstance(value, torch.Tensor):
        if operation == "detach":
            return value.detach()
        dtype = kwargs.get("dtype")
        target_dtype = (
            dtype
            if dtype is not None and (value.is_floating_point() or value.is_complex())
            else None
        )
        return value.to(device=kwargs["device"], dtype=target_dtype)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _map_tensors(item, operation, **kwargs))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, operation, **kwargs) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, operation, **kwargs) for item in value]
    return value


def _validate_sample_indices(indices: Any, batch_size: int) -> list[int]:
    if isinstance(indices, (str, bytes)):
        raise TypeError("sample indices must be a non-empty sequence of integers")
    try:
        resolved = list(indices)
    except TypeError as exc:
        raise TypeError(
            "sample indices must be a non-empty sequence of integers"
        ) from exc
    if not resolved:
        raise ValueError("sample indices must not be empty")
    for position, index in enumerate(resolved):
        if isinstance(index, bool):
            raise TypeError(f"sample index at position {position} must be an integer")
        try:
            index = operator.index(index)
        except TypeError as exc:
            raise TypeError(
                f"sample index at position {position} must be an integer"
            ) from exc
        resolved[position] = index
        if index < 0 or index >= batch_size:
            raise IndexError(
                f"sample index {index} is out of bounds for batch size {batch_size}"
            )
    if len(set(resolved)) != len(resolved):
        raise ValueError("sample indices must not contain duplicates")
    return resolved


def _validate_formal_occurrence_groups(
    prompt_ids: Any,
    metadata: list[dict[str, Any]],
    group_ids: Any,
    *,
    explicit_rows: Any = None,
) -> None:
    """Reject ambiguous grouping for repeated formal prompt occurrences."""

    if isinstance(prompt_ids, (str, bytes)) or not hasattr(prompt_ids, "__len__"):
        return
    try:
        prompt_values = list(prompt_ids)
    except TypeError:
        return

    counts: dict[str, int] = {}
    for value in prompt_values:
        if isinstance(value, str):
            counts[value] = counts.get(value, 0) + 1
    repeated_rows = [
        index
        for index, value in enumerate(prompt_values)
        if isinstance(value, str) and counts.get(value, 0) > 1
    ]
    if not repeated_rows:
        return

    if group_ids is None:
        resolved_groups = [item.get("group_id") for item in metadata]
    elif isinstance(group_ids, (str, bytes)) or not hasattr(group_ids, "__len__"):
        resolved_groups = []
    else:
        try:
            resolved_groups = list(group_ids)
        except TypeError:
            resolved_groups = []

    unresolved_rows = [
        index
        for index in repeated_rows
        if index >= len(resolved_groups)
        or (
            explicit_rows is not None
            and (
                index >= len(explicit_rows)
                or not bool(explicit_rows[index])
            )
        )
        or not isinstance(resolved_groups[index], str)
        or not resolved_groups[index].strip()
    ]
    if unresolved_rows:
        raise ValueError(
            "Formal RolloutBatch contains repeated prompt_id values without "
            "unambiguous occurrence group_id values at rows "
            f"{unresolved_rows}; provide an explicit group_id for each prompt "
            "occurrence, reusing one group_id only for an intentional "
            "multi-sample or multi-branch group"
        )


def _explicit_group_id_rows(
    metadata: list[dict[str, Any]],
    group_ids: Any,
) -> tuple[bool, ...]:
    if not isinstance(group_ids, (str, bytes)) and hasattr(group_ids, "__len__"):
        try:
            values = list(group_ids)
        except TypeError:
            values = []
        return tuple(
            index < len(values)
            and isinstance(values[index], str)
            and bool(values[index].strip())
            for index in range(len(metadata))
        )
    return tuple(
        isinstance(item.get("group_id"), str)
        and bool(item["group_id"].strip())
        for item in metadata
    )


def _slice_batch_axis(value: Any, indices: list[int], batch_size: int) -> Any:
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == batch_size:
                index = torch.tensor(indices, device=value.device, dtype=torch.long)
                return value.index_select(0, index)
            return value
    except ImportError:
        pass

    shape = _shape_tuple(value)
    if shape is not None:
        if shape and shape[0] == batch_size:
            try:
                return value[indices]
            except (IndexError, TypeError):
                try:
                    import numpy as np

                    return value[np.asarray(indices, dtype=np.int64)]
                except ImportError:
                    return [value[index] for index in indices]
        return value
    if isinstance(value, Mapping):
        return type(value)(
            (key, _slice_batch_axis(item, indices, batch_size))
            for key, item in value.items()
        )
    if isinstance(value, list):
        if len(value) == batch_size:
            return [value[index] for index in indices]
        return [_slice_batch_axis(item, indices, batch_size) for item in value]
    if isinstance(value, tuple):
        if len(value) == batch_size:
            return tuple(value[index] for index in indices)
        return tuple(_slice_batch_axis(item, indices, batch_size) for item in value)
    return value


def _shape_tree(value: Any) -> Any:
    shape = _shape_tuple(value)
    if shape is not None:
        return shape
    if isinstance(value, Mapping):
        return {key: _shape_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        nested = [_shape_tree(item) for item in value]
        return nested if any(item is not None for item in nested) else None
    return None


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def _infer_media_layout(media: Any) -> str | None:
    shape = _shape_tuple(media)
    if shape is None:
        return None
    if len(shape) == 4:
        return "BCHW"
    if len(shape) == 5:
        return "BFCHW"
    return None


def _default_transition_mask(*values: Any) -> Any | None:
    reference_shape = None
    reference = None
    for value in values:
        shape = _shape_tuple(value)
        if shape is not None and len(shape) >= 2:
            reference_shape = shape[:2]
            reference = value
            break
    if reference_shape is None:
        return None
    try:
        import torch

        if isinstance(reference, torch.Tensor):
            return torch.ones(
                reference_shape,
                dtype=torch.bool,
                device=reference.device,
            )
    except ImportError:
        pass
    try:
        import numpy as np

        return np.ones(reference_shape, dtype=bool)
    except ImportError:
        return [[True] * reference_shape[1] for _ in range(reference_shape[0])]


def _check_identity(name: str, value: Any, batch_size: int) -> None:
    if (
        value is None
        or isinstance(value, (str, bytes))
        or not hasattr(value, "__len__")
    ):
        raise ValueError(f"{name} must be a sequence with length {batch_size}")
    shape = _shape_tuple(value)
    if shape is not None and shape != (batch_size,):
        raise ValueError(f"{name} must have shape ({batch_size},), got {shape}")
    if len(value) != batch_size:
        raise ValueError(f"{name} length must be {batch_size}, got {len(value)}")
    if name != "branch_id":
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{name}[{index}] must be a non-empty string")


def _check_bool_mask(value: Any) -> None:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.dtype != torch.bool:
                raise ValueError("valid/transition masks must have bool dtype")
            return
    except ImportError:
        pass
    try:
        import numpy as np

        array = np.asarray(value)
        if array.dtype != np.bool_:
            raise ValueError("valid/transition masks must have bool dtype")
    except ImportError:
        if not all(isinstance(item, bool) for item in value):
            raise ValueError("valid/transition masks must contain bool values")


def _require_vector(name: str, value: Any, batch_size: int) -> None:
    shape = _shape_tuple(value)
    if shape is not None:
        if shape != (batch_size,):
            raise ValueError(f"{name} must have shape ({batch_size},), got {shape}")
        return
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise ValueError(f"{name} must have shape ({batch_size},)")
    import numpy as np

    if np.asarray(value).shape != (batch_size,):
        raise ValueError(
            f"{name} must have shape ({batch_size},), got {np.asarray(value).shape}"
        )


def _require_finite(name: str, value: Any) -> None:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")
            return
    except ImportError:
        pass

    import numpy as np

    if not np.isfinite(_as_numpy(value)).all():
        raise ValueError(f"{name} must contain only finite values")


def _as_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
            value = value.float()
    except ImportError:
        pass
    if hasattr(value, "numpy"):
        return value.numpy()
    import numpy as np

    return np.asarray(value)


def _as_cpu_floating_tensor(name: str, value: Any) -> Any:
    import torch

    try:
        tensor = (
            value.detach()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"{name} must contain real numeric values")
    return tensor.to(device="cpu", dtype=torch.float32).detach()


def _as_cpu_bool_tensor(name: str, value: Any) -> Any:
    import torch

    try:
        tensor = (
            value.detach()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must contain bool values") from exc
    if tensor.dtype != torch.bool:
        raise ValueError("valid/transition masks must have bool dtype")
    return tensor.to(device="cpu").detach()


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# v0.7 cross-component contract types (master plan stage 2, incremental).
#
# These frozen types are added ahead of the atomic cutover that rewires the
# concrete producers/consumers. They do not change existing behavior; no
# current call site is required to use them yet.
# ---------------------------------------------------------------------------


def _freeze_value(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("FrozenMapping does not accept non-finite floats")
    if isinstance(value, (str, int, float, bool, type(None), Path)):
        return value
    raise TypeError(
        "FrozenMapping values must be JSON-safe scalars/containers or Path, "
        f"got {type(value).__name__}"
    )


class FrozenMapping(Mapping):
    """Internal, pickle-safe read-only mapping shared by every contract.

    Construction recursively defensive-copies: nested mappings become
    ``FrozenMapping`` instances stored as fixed tuple pairs, lists become
    tuples; mutating the caller's original containers afterwards cannot
    affect the constructed object. ``MappingProxyType`` is deliberately not
    used because it cannot be pickled; two-rank ``gather_object()`` payloads
    must round-trip. Use ``to_plain_dict()`` at the YAML/JSON boundary to
    recover plain containers.
    """

    __slots__ = ("_items",)

    def __init__(self, source: Any = ()) -> None:
        raw_items = source.items() if isinstance(source, Mapping) else source
        items = []
        keys = set()
        for key, value in raw_items:
            if not isinstance(key, str):
                raise TypeError("FrozenMapping keys must be strings")
            if key in keys:
                raise ValueError(f"FrozenMapping contains duplicate key {key!r}")
            keys.add(key)
            items.append((key, _freeze_value(value)))
        object.__setattr__(self, "_items", tuple(items))

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("FrozenMapping is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("FrozenMapping is immutable")

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def __hash__(self) -> int:
        return hash(frozenset(self._items))

    def __reduce__(self):
        return type(self), (self._items,)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._items)!r})"


def _reject_non_plain(value: Any) -> None:
    if isinstance(value, (set, frozenset)):
        raise TypeError("to_plain_dict does not accept set/frozenset values")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("to_plain_dict does not accept binary values")
    if callable(value):
        raise TypeError("to_plain_dict does not accept callables")
    try:
        import torch

        if isinstance(value, torch.Tensor):
            raise TypeError("to_plain_dict does not accept torch.Tensor")
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            raise TypeError("to_plain_dict does not accept numpy.ndarray")
    except ImportError:
        pass


def to_plain_dict(value: Any) -> Any:
    """The single strict artifact/config projector (plan stage 2).

    Accepts validated frozen dataclasses, ``FrozenMapping``/``Mapping``,
    tuple/list, ``Path`` and finite JSON scalars, and returns plain
    dict/list/scalar containers. ``Tensor``/``ndarray``/callables/sets/
    arbitrary objects and non-finite floats are rejected before anything is
    written; nothing is silently stringified or listified.
    """

    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(value, "__dataclass_params__", None)
        if parameters is None or not parameters.frozen:
            raise TypeError("to_plain_dict only accepts frozen dataclass instances")
        return {
            item.name: to_plain_dict(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        projected = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("to_plain_dict mapping keys must be strings")
            projected[key] = to_plain_dict(item)
        return projected
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("to_plain_dict does not accept non-finite floats")
        return value
    if isinstance(value, Path):
        return str(value)
    _reject_non_plain(value)
    raise TypeError(f"to_plain_dict does not accept {type(value).__name__}")


@dataclass(frozen=True)
class RolloutRequest:
    """The single rollout request constructed once per step by RolloutEngine.

    ``prompts``/``metadata`` are already expanded to ``B`` rows; identity
    tuples are per-row ``[B]``; ``branch_id`` entries are JSON scalars
    (``str | int | None``). The Adapter echoes every field verbatim and must
    not re-expand, drop or reorder rows.
    """

    prompts: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...]
    sample_id: tuple[str, ...]
    prompt_id: tuple[str, ...]
    group_id: tuple[str, ...]
    branch_id: tuple[Any, ...] | None
    context: StepContext
    kind: Literal["full_trajectory", "single_step", "branching"]
    num_steps: int
    group_size: int
    selected_timestep_index: tuple[int, ...] | None = None
    branch_step_index: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("prompts", "metadata", "sample_id", "prompt_id", "group_id"):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"{name} must be a tuple")
        prompts = self.prompts
        if not prompts or any(not isinstance(item, str) for item in prompts):
            raise ValueError("prompts must be a non-empty tuple of strings")
        batch_size = len(prompts)

        metadata = tuple(
            item if isinstance(item, FrozenMapping) else FrozenMapping(item)
            for item in self.metadata
        )
        if len(metadata) != batch_size:
            raise ValueError("metadata must contain one mapping per prompt")
        object.__setattr__(self, "metadata", metadata)

        for name in ("sample_id", "prompt_id", "group_id"):
            values = getattr(self, name)
            if len(values) != batch_size:
                raise ValueError(f"{name} must contain one value per prompt")
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} entries must be non-empty strings")
            object.__setattr__(self, name, values)
        if len(set(self.sample_id)) != batch_size:
            raise ValueError("sample_id entries must be unique")

        if self.branch_id is not None:
            if type(self.branch_id) is not tuple:
                raise TypeError("branch_id must be a tuple or None")
            branch_ids = self.branch_id
            if len(branch_ids) != batch_size:
                raise ValueError("branch_id must contain one value per prompt")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (str, int, type(None)))
                for item in branch_ids
            ):
                raise TypeError("branch_id entries must be str, int, or None")
            object.__setattr__(self, "branch_id", branch_ids)

        if not isinstance(self.context, StepContext):
            raise TypeError("context must be a StepContext")
        if self.kind not in {"full_trajectory", "single_step", "branching"}:
            raise ValueError(f"unknown rollout kind: {self.kind!r}")
        for name in ("num_steps", "group_size"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer, not bool")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        for name in ("selected_timestep_index", "branch_step_index"):
            values = getattr(self, name)
            if values is None:
                continue
            if type(values) is not tuple:
                raise TypeError(f"{name} must be a tuple or None")
            if len(values) != batch_size:
                raise ValueError(f"{name} must contain one value per prompt")
            if any(type(item) is not int for item in values):
                raise TypeError(f"{name} entries must be integers, not bool")
            if any(not 0 <= item < self.num_steps for item in values):
                raise ValueError(
                    f"{name} entries must satisfy 0 <= index < num_steps"
                )
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class PolicyRecomputeStats:
    """Microbatch-local policy recompute output (never persisted).

    ``new_log_probs`` is differentiable and matches
    ``batch.old_log_probs.shape``; values are only required to be finite at
    ``batch.transition_mask=True`` positions. With
    ``require_reference=False`` the three reference fields must be ``None``
    and no reference forward may have run.
    """

    new_log_probs: Any
    current_transition_mean: Any | None = None
    transition_std: Any | None = None
    reference_transition_mean: Any | None = None

    def validate_against(
        self,
        batch: RolloutBatch,
        *,
        require_reference: bool,
    ) -> None:
        """Enforce the frozen recompute contract before the objective runs."""

        import torch

        old_shape = _shape_tuple(batch.old_log_probs)
        new_shape = _shape_tuple(self.new_log_probs)
        if old_shape is None or new_shape is None or old_shape != new_shape:
            raise ValueError(
                "new_log_probs must have the same shape as old_log_probs: "
                f"{new_shape} != {old_shape}"
            )
        reference_fields = (
            self.current_transition_mean,
            self.transition_std,
            self.reference_transition_mean,
        )
        if not require_reference:
            if any(item is not None for item in reference_fields):
                raise ValueError(
                    "reference statistics require require_reference=True"
                )
        elif any(item is None for item in reference_fields):
            raise ValueError(
                "require_reference=True requires current_transition_mean, "
                "transition_std and reference_transition_mean"
            )

        new_log_probs = self.new_log_probs
        if not isinstance(new_log_probs, torch.Tensor):
            raise TypeError("new_log_probs must be a torch.Tensor")
        if new_log_probs.ndim != 2:
            raise ValueError("new_log_probs must have shape [B, T]")
        if not new_log_probs.is_floating_point():
            raise TypeError("new_log_probs must be a floating-point tensor")
        if not new_log_probs.requires_grad:
            raise ValueError("new_log_probs must require gradients")
        mask = batch.transition_mask
        if mask is not None:
            mask = torch.as_tensor(mask, device=new_log_probs.device)
            if mask.dtype != torch.bool:
                raise TypeError("transition_mask must have bool dtype")
            if tuple(mask.shape) != tuple(new_log_probs.shape):
                raise ValueError(
                    "transition_mask must have the same shape as new_log_probs: "
                    f"{tuple(mask.shape)} != {tuple(new_log_probs.shape)}"
                )
            active = new_log_probs.masked_select(mask)
        else:
            mask = torch.ones_like(new_log_probs, dtype=torch.bool)
            active = new_log_probs
        if not bool(mask.any()):
            raise ValueError("at least one transition must be active")
        if not bool(torch.isfinite(active).all()):
            raise ValueError(
                "new_log_probs must be finite at active transition positions"
            )

        if not require_reference:
            return
        current_mean = self.current_transition_mean
        reference_mean = self.reference_transition_mean
        transition_std = self.transition_std
        for name, value in (
            ("current_transition_mean", current_mean),
            ("reference_transition_mean", reference_mean),
            ("transition_std", transition_std),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor")
            if value.device != new_log_probs.device:
                raise ValueError(f"{name} must be on the new_log_probs device")
        if not current_mean.requires_grad:
            raise ValueError("current_transition_mean must require gradients")
        if reference_mean.requires_grad or reference_mean.grad_fn is not None:
            raise ValueError(
                "reference_transition_mean must be detached without grad_fn"
            )
        if transition_std.requires_grad or transition_std.grad_fn is not None:
            raise ValueError("transition_std must be detached without grad_fn")
        if _shape_tuple(current_mean) != _shape_tuple(reference_mean):
            raise ValueError(
                "current_transition_mean and reference_transition_mean must "
                "have the same shape"
            )
        mean_shape = _shape_tuple(current_mean)
        if mean_shape is None or tuple(mean_shape[:2]) != tuple(new_shape):
            raise ValueError(
                "transition mean shapes must start with the [B, T] log-prob "
                f"prefix, got {mean_shape} for {new_shape}"
            )
        std_shape = _shape_tuple(transition_std)
        broadcast_std = tuple(new_shape) + (1,) * (len(mean_shape) - 2)
        if std_shape not in {tuple(mean_shape), tuple(new_shape), broadcast_std}:
            raise ValueError(
                "transition_std must have the full mean shape, [B, T], or "
                f"[B, T, 1, ..., 1]; got {std_shape}"
            )
        mean_mask = mask.reshape(tuple(new_shape) + (1,) * (len(mean_shape) - 2))
        mean_mask = torch.broadcast_to(mean_mask, mean_shape)
        for name, value in (
            ("current_transition_mean", current_mean),
            ("reference_transition_mean", reference_mean),
        ):
            if not bool(torch.isfinite(value.masked_select(mean_mask)).all()):
                raise ValueError(f"{name} must be finite at active transitions")
        if std_shape == tuple(new_shape):
            std_for_mean = transition_std.reshape(broadcast_std)
        else:
            std_for_mean = transition_std
        std_for_mean = torch.broadcast_to(std_for_mean, mean_shape)
        active_std = std_for_mean.masked_select(mean_mask)
        if not bool(torch.isfinite(active_std).all()):
            raise ValueError("transition_std must be finite at active transitions")
        if not bool((active_std > 0).all()):
            raise ValueError(
                "transition_std must be strictly positive at active transitions"
            )


@dataclass(frozen=True)
class RewardVector:
    """One RewardClient's shard-local frozen result (provider-internal)."""

    sample_id: tuple[str, ...]
    values: Any  # finite reward vector [B]
    shared_metadata: Mapping[str, Any]  # JSON-safe protocol/revision metadata
    sample_metadata: tuple[Mapping[str, Any], ...]  # per-sample evidence [B]

    def __post_init__(self) -> None:
        import torch

        if type(self.sample_id) is not tuple:
            raise TypeError("sample_id must be a tuple")
        if type(self.sample_metadata) is not tuple:
            raise TypeError("sample_metadata must be a tuple")
        sample_id = self.sample_id
        if not sample_id or any(
            not isinstance(item, str) or not item for item in sample_id
        ):
            raise ValueError("sample_id must be a non-empty tuple of strings")
        if len(set(sample_id)) != len(sample_id):
            raise ValueError("sample_id entries must be unique")
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if self.values.ndim != 1 or self.values.shape[0] != len(sample_id):
            raise ValueError("values must have shape [B] matching sample_id")
        if not self.values.is_floating_point():
            raise TypeError("values must be a floating-point tensor")
        if self.values.requires_grad or self.values.grad_fn is not None:
            raise ValueError("values must be detached without grad_fn")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("values must be finite")
        sample_metadata = tuple(
            item if isinstance(item, FrozenMapping) else FrozenMapping(item)
            for item in self.sample_metadata
        )
        if len(sample_metadata) != len(sample_id):
            raise ValueError("sample_metadata must contain one mapping per sample")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "sample_metadata", sample_metadata)
        if not isinstance(self.shared_metadata, FrozenMapping):
            object.__setattr__(
                self, "shared_metadata", FrozenMapping(self.shared_metadata)
            )


@dataclass(frozen=True)
class MetricContribution:
    """One reducible metric: detached finite scalar sum plus a denominator.

    ``denominator=None`` uniquely means cross-rank SUM; a positive integer
    means ``sum(numerator) / sum(denominator)``.
    """

    numerator: Any
    denominator: int | None

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.numerator, torch.Tensor):
            raise TypeError("numerator must be a torch.Tensor")
        if self.numerator.ndim != 0:
            raise ValueError("numerator must be a scalar tensor")
        if self.numerator.requires_grad or self.numerator.grad_fn is not None:
            raise ValueError("numerator must be detached without grad_fn")
        if not bool(torch.isfinite(self.numerator)):
            raise ValueError("numerator must be finite")
        if self.denominator is not None:
            if type(self.denominator) is not int:
                raise TypeError("denominator must be an integer, not bool")
            if self.denominator <= 0:
                raise ValueError("denominator must be a positive int or None (SUM)")


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
    """The only context ``resolve_params()`` may use to normalize paths."""

    config_path: Path  # absolute normalized YAML path
    config_dir: Path  # config_path.parent


@dataclass(frozen=True)
class ValidationContext:
    """Bounded, read-only environment check context (no model/GPU init)."""

    phase: Literal["validate", "run"]
    config_dir: Path
    distributed_mode: Literal["single", "ddp"]
    world_size: Literal[1, 2]  # uniquely derived from distributed_mode
    backend: str | None  # uniquely derived from mode + device
    device: str
    timeout_s: float


@dataclass(frozen=True)
class ValidatedRuntimeEnv:
    """Launch topology parsed once by the CPU-only environment validator."""

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
    """Rank-local build parameters passed to every factory ``from_config()``.

    ``device`` is a ``torch.device`` at runtime (kept ``Any`` here so this
    module stays import-level torch-free); factories must use the passed
    device/backend/precision and never re-derive them from the environment.
    """

    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device: Any
    precision: Literal["fp32", "fp16", "bf16"]
