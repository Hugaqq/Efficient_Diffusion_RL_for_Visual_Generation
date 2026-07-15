"""Shared data contracts used by rollout, rewards, algorithms, and trainers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import hashlib
import operator
from typing import Any


@dataclass(frozen=True)
class StepContext:
    """Immutable runtime identity for one rollout/update step."""

    step: int
    seed: int
    epoch_tag: int
    rank: int = 0
    world_size: int = 1
    policy_version: int = 0


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
        """Select a non-empty ordered subset along the sample axis."""

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
