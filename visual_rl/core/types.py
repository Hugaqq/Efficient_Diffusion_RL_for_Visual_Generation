"""Shared data contracts used by rollout, rewards, algorithms, and trainers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace as dataclass_replace
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
    """The sole immutable runtime identity for one rollout/update step."""

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


def _require_detached_tensor(name: str, value: Any) -> None:
    if bool(getattr(value, "requires_grad", False)) or getattr(
        value, "grad_fn", None
    ) is not None:
        raise ValueError(f"{name} must be detached without grad_fn")


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


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


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


class _FrozenTensorMapping(Mapping):
    """Private immutable mapping for the one flat ``str -> Tensor`` surface."""

    __slots__ = ("_items",)

    def __init__(self, source: Mapping[str, Any] | Any = ()) -> None:
        raw_items = source.items() if isinstance(source, Mapping) else source
        items: list[tuple[str, Any]] = []
        keys: set[str] = set()
        for key, value in raw_items:
            if not isinstance(key, str) or not key:
                raise TypeError("tensor mapping keys must be non-empty strings")
            if key in keys:
                raise ValueError(f"tensor mapping contains duplicate key {key!r}")
            keys.add(key)
            items.append((key, value))
        object.__setattr__(self, "_items", tuple(items))

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("tensor mapping is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("tensor mapping is immutable")

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(self):
        return type(self), (self._items,)

    def __repr__(self) -> str:
        return f"_FrozenTensorMapping({dict(self._items)!r})"


def _reject_non_plain(value: Any) -> None:
    if isinstance(value, (set, frozenset)):
        raise TypeError("to_plain_dict does not accept set/frozenset values")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("to_plain_dict does not accept binary values")
    if callable(value):
        raise TypeError("to_plain_dict does not accept callables")
    root_module = type(value).__module__.partition(".")[0]
    type_name = type(value).__name__
    if root_module == "torch" and type_name == "Tensor":
        raise TypeError("to_plain_dict does not accept torch.Tensor")
    if root_module == "numpy" and type_name == "ndarray":
        raise TypeError("to_plain_dict does not accept numpy.ndarray")


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
        projected = {}
        for item in fields(value):
            plain_name = item.metadata.get("plain_name", item.name)
            if not isinstance(plain_name, str) or not plain_name:
                raise TypeError("dataclass field plain_name must be a non-empty string")
            if plain_name in projected:
                raise ValueError(
                    f"dataclass projection contains duplicate key {plain_name!r}"
                )
            projected[plain_name] = to_plain_dict(getattr(value, item.name))
        return projected
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
            branch_id_values = self.branch_id
            if len(branch_id_values) != batch_size:
                raise ValueError("branch_id must contain one value per prompt")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (str, int, type(None)))
                for item in branch_id_values
            ):
                raise TypeError("branch_id entries must be str, int, or None")
            object.__setattr__(self, "branch_id", branch_id_values)

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
        minimum_steps = {
            "full_trajectory": 2,
            "single_step": 1,
            "branching": 2,
        }[self.kind]
        if self.num_steps < minimum_steps:
            raise ValueError(
                f"{self.kind} requires num_steps >= {minimum_steps}"
            )

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
            upper_bound = (
                self.num_steps - 1
                if name == "branch_step_index" and self.kind == "branching"
                else self.num_steps
            )
            if any(not 0 <= item < upper_bound for item in values):
                raise ValueError(
                    f"{name} entries must satisfy 0 <= index < {upper_bound}"
                )
            object.__setattr__(self, name, values)

        if self.kind == "single_step":
            if self.selected_timestep_index is None:
                raise ValueError(
                    "single_step requires selected_timestep_index for every row"
                )
            if self.branch_step_index is not None:
                raise ValueError("single_step does not accept branch_step_index")
        elif self.kind == "branching":
            if self.branch_step_index is None:
                raise ValueError("branching requires branch_step_index for every row")
            if self.selected_timestep_index is not None:
                raise ValueError(
                    "branching does not accept selected_timestep_index"
                )
        elif (
            self.selected_timestep_index is not None
            or self.branch_step_index is not None
        ):
            raise ValueError(
                "full_trajectory does not accept selected/branch timestep indices"
            )

        group_rows: dict[str, list[int]] = {}
        for row, group_id in enumerate(self.group_id):
            group_rows.setdefault(group_id, []).append(row)
        if any(len(rows) != self.group_size for rows in group_rows.values()):
            raise ValueError("each occurrence group must contain exactly group_size rows")
        for rows in group_rows.values():
            first = rows[0]
            for row in rows[1:]:
                if (
                    self.prompt_id[row] != self.prompt_id[first]
                    or self.prompts[row] != self.prompts[first]
                    or self.metadata[row] != self.metadata[first]
                ):
                    raise ValueError(
                        "rows in one occurrence group must share prompt identity "
                        "and metadata"
                    )


@dataclass(frozen=True)
class RolloutBatch:
    """The sole typed rollout payload crossing training components."""

    prompts: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...]
    media: Any
    latents: Any
    next_latents: Any
    timesteps: Any
    old_log_probs: Any
    transition_mask: Any
    sample_id: tuple[str, ...]
    prompt_id: tuple[str, ...]
    group_id: tuple[str, ...]
    branch_id: tuple[str | int | None, ...] | None
    media_layout: str
    camera_trajectory: Any | None
    context: StepContext
    selected_timestep_index: Any | None
    flash_coefficient: Any | None
    branch_step_index: Any | None
    trajectory_step_index: Any | None
    transition_std_dev: Any | None
    recompute_payload: Mapping[str, Any]
    artifact_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        import torch

        batch_size = len(self.prompts)
        if type(self.prompts) is not tuple or not self.prompts:
            raise TypeError("prompts must be a non-empty tuple")
        if any(not isinstance(item, str) for item in self.prompts):
            raise TypeError("prompts entries must be strings")
        if type(self.metadata) is not tuple or len(self.metadata) != batch_size:
            raise ValueError("metadata must be a tuple containing one row per prompt")
        metadata = tuple(
            item if isinstance(item, FrozenMapping) else FrozenMapping(item)
            for item in self.metadata
        )
        object.__setattr__(self, "metadata", metadata)

        for name in ("sample_id", "prompt_id", "group_id"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != batch_size:
                raise ValueError(f"{name} must be a tuple with shape [B]")
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} entries must be non-empty strings")
        if len(set(self.sample_id)) != batch_size:
            raise ValueError("sample_id entries must be unique")
        if self.branch_id is not None:
            if type(self.branch_id) is not tuple or len(self.branch_id) != batch_size:
                raise ValueError("branch_id must be a tuple with shape [B] or None")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (str, int, type(None)))
                for item in self.branch_id
            ):
                raise TypeError("branch_id entries must be str, int, or None")
        if not isinstance(self.context, StepContext):
            raise TypeError("context must be a StepContext")

        if self.media_layout not in {"BCHW", "BFCHW", "BFHWC"}:
            raise ValueError("media_layout must be BCHW, BFCHW, or BFHWC")
        media_shape = _shape_tuple(self.media)
        expected_media_ndim = 4 if self.media_layout == "BCHW" else 5
        if (
            media_shape is None
            or len(media_shape) != expected_media_ndim
            or media_shape[0] != batch_size
        ):
            raise ValueError(
                f"media_layout {self.media_layout} requires media shape "
                f"[B, ...] with {expected_media_ndim} dimensions"
            )

        tensor_fields = {
            "latents": self.latents,
            "next_latents": self.next_latents,
            "timesteps": self.timesteps,
            "old_log_probs": self.old_log_probs,
            "transition_mask": self.transition_mask,
        }
        for name, value in tensor_fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            _require_detached_tensor(name, value)
        if self.latents.ndim < 3:
            raise ValueError("latents must have shape [B, T, ...X]")
        if tuple(self.next_latents.shape) != tuple(self.latents.shape):
            raise ValueError("next_latents must have the same shape as latents")
        transition_shape = tuple(self.old_log_probs.shape)
        if len(transition_shape) != 2:
            raise ValueError("old_log_probs must have shape [B, T]")
        if transition_shape[0] != batch_size:
            raise ValueError("old_log_probs first dimension must equal B")
        if tuple(self.timesteps.shape) != transition_shape:
            raise ValueError("timesteps must have shape [B, T]")
        if tuple(self.transition_mask.shape) != transition_shape:
            raise ValueError("transition_mask must have shape [B, T]")
        if tuple(self.latents.shape[:2]) != transition_shape:
            raise ValueError("latents must start with the same [B, T] dimensions")
        if not self.old_log_probs.is_floating_point():
            raise TypeError("old_log_probs must be floating point")
        if self.transition_mask.dtype != torch.bool:
            raise TypeError("transition_mask must have bool dtype")
        if not bool(self.transition_mask.any()):
            raise ValueError("at least one transition must be active")
        if not bool(
            torch.isfinite(self.old_log_probs.masked_select(self.transition_mask)).all()
        ):
            raise ValueError("old_log_probs must be finite at active transitions")

        self._validate_optional_tensor(
            "selected_timestep_index",
            self.selected_timestep_index,
            (batch_size,),
            dtype=torch.int64,
        )
        self._validate_optional_tensor(
            "branch_step_index",
            self.branch_step_index,
            (batch_size,),
            dtype=torch.int64,
        )
        self._validate_optional_tensor(
            "trajectory_step_index",
            self.trajectory_step_index,
            (transition_shape[1],),
            dtype=torch.int64,
        )
        self._validate_optional_tensor(
            "flash_coefficient",
            self.flash_coefficient,
            (batch_size, 1),
            positive=True,
        )
        self._validate_optional_tensor(
            "transition_std_dev",
            self.transition_std_dev,
            transition_shape,
            positive=True,
        )

        if self.camera_trajectory is not None:
            camera = self.camera_trajectory
            if not isinstance(camera, torch.Tensor):
                raise TypeError("camera_trajectory must be a torch.Tensor or None")
            _require_detached_tensor("camera_trajectory", camera)
            if camera.dtype != torch.float64:
                raise TypeError("camera_trajectory must use torch.float64")
            if camera.ndim != 4 or tuple(camera.shape[2:]) != (4, 4):
                raise ValueError("camera_trajectory must have shape [B, F, 4, 4]")
            if camera.shape[0] != batch_size:
                raise ValueError("camera_trajectory first dimension must equal B")
            frame_axis = 1
            if camera.shape[1] != media_shape[frame_axis]:
                raise ValueError(
                    "camera_trajectory frame count must match video media frames"
                )
            if self.media_layout == "BCHW":
                raise ValueError("image rollout cannot carry camera_trajectory")
            if not bool(torch.isfinite(camera).all()):
                raise ValueError("camera_trajectory must be finite")

        payload = _FrozenTensorMapping(self.recompute_payload)
        for name, value in payload.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"recompute_payload[{name!r}] must be a torch.Tensor")
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(
                    f"recompute_payload[{name!r}] must have first dimension B"
                )
            _require_detached_tensor(f"recompute_payload[{name!r}]", value)
        object.__setattr__(self, "recompute_payload", payload)
        if not isinstance(self.artifact_metadata, FrozenMapping):
            object.__setattr__(
                self,
                "artifact_metadata",
                FrozenMapping(self.artifact_metadata),
            )

    @property
    def batch_size(self) -> int:
        return len(self.prompts)

    @property
    def transition_count(self) -> int:
        return int(self.old_log_probs.shape[1])

    @staticmethod
    def _validate_optional_tensor(
        name: str,
        value: Any | None,
        shape: tuple[int, ...],
        *,
        dtype: Any | None = None,
        positive: bool = False,
    ) -> None:
        if value is None:
            return
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")
        _require_detached_tensor(name, value)
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if dtype is not None and value.dtype != dtype:
            raise TypeError(f"{name} must use {dtype}")
        if positive:
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")
            if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
                raise ValueError(f"{name} must be finite and strictly positive")

    def validate_against(self, request: RolloutRequest) -> None:
        """Validate Adapter output against the exact request it consumed."""

        import torch

        if self.context is not request.context:
            raise ValueError("RolloutBatch must echo the same StepContext object")
        for name in (
            "prompts",
            "metadata",
            "sample_id",
            "prompt_id",
            "group_id",
            "branch_id",
        ):
            if getattr(self, name) != getattr(request, name):
                raise ValueError(f"RolloutBatch must echo request.{name} unchanged")
        expected_transitions = request.num_steps if request.kind == "full_trajectory" else 1
        if self.transition_count != expected_transitions:
            raise ValueError(
                f"{request.kind} must return T={expected_transitions}, "
                f"got {self.transition_count}"
            )
        if request.kind == "single_step":
            if self.selected_timestep_index is None:
                raise ValueError("single_step requires selected_timestep_index")
            expected = torch.tensor(
                request.selected_timestep_index,
                dtype=torch.int64,
                device=self.selected_timestep_index.device,
            )
            if not torch.equal(self.selected_timestep_index, expected):
                raise ValueError(
                    "selected_timestep_index must echo the request plan"
                )
        elif self.selected_timestep_index is not None:
            raise ValueError(
                "selected_timestep_index is only valid for single_step"
            )
        if request.kind == "branching":
            if self.branch_step_index is None:
                raise ValueError("branching requires branch_step_index")
            expected = torch.tensor(
                request.branch_step_index,
                dtype=torch.int64,
                device=self.branch_step_index.device,
            )
            if not torch.equal(self.branch_step_index, expected):
                raise ValueError("branch_step_index must echo the request plan")
        elif self.branch_step_index is not None:
            raise ValueError("branch_step_index is only valid for branching")

    def replace(self, **updates: Any) -> RolloutBatch:
        unknown = set(updates).difference(item.name for item in fields(self))
        if unknown:
            raise TypeError(f"unknown RolloutBatch fields: {sorted(unknown)}")
        return dataclass_replace(self, **updates)

    def slice(self, indices: Any) -> RolloutBatch:
        resolved = _validate_sample_indices(indices, self.batch_size)
        tensor_index = None

        def slice_value(value: Any) -> Any:
            nonlocal tensor_index
            if value is None:
                return None
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    if tensor_index is None or tensor_index.device != value.device:
                        tensor_index = torch.tensor(
                            resolved,
                            dtype=torch.long,
                            device=value.device,
                        )
                    return value.index_select(0, tensor_index)
            except ImportError:  # pragma: no cover - RolloutBatch requires torch
                pass
            return tuple(value[index] for index in resolved)

        return dataclass_replace(
            self,
            prompts=slice_value(self.prompts),
            metadata=slice_value(self.metadata),
            media=slice_value(self.media),
            latents=slice_value(self.latents),
            next_latents=slice_value(self.next_latents),
            timesteps=slice_value(self.timesteps),
            old_log_probs=slice_value(self.old_log_probs),
            transition_mask=slice_value(self.transition_mask),
            sample_id=slice_value(self.sample_id),
            prompt_id=slice_value(self.prompt_id),
            group_id=slice_value(self.group_id),
            branch_id=slice_value(self.branch_id),
            camera_trajectory=slice_value(self.camera_trajectory),
            selected_timestep_index=slice_value(self.selected_timestep_index),
            flash_coefficient=slice_value(self.flash_coefficient),
            branch_step_index=slice_value(self.branch_step_index),
            transition_std_dev=slice_value(self.transition_std_dev),
            recompute_payload={
                name: slice_value(value)
                for name, value in self.recompute_payload.items()
            },
        )

    def to(self, device: Any, dtype: Any = None) -> RolloutBatch:
        import torch

        def move(value: Any, *, preserve_dtype: bool = False) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            target_dtype = None
            if (
                dtype is not None
                and not preserve_dtype
                and (value.is_floating_point() or value.is_complex())
            ):
                target_dtype = dtype
            return value.to(device=device, dtype=target_dtype)

        return dataclass_replace(
            self,
            media=move(self.media),
            latents=move(self.latents),
            next_latents=move(self.next_latents),
            timesteps=move(self.timesteps),
            old_log_probs=move(self.old_log_probs),
            transition_mask=move(self.transition_mask),
            camera_trajectory=move(self.camera_trajectory, preserve_dtype=True),
            selected_timestep_index=move(self.selected_timestep_index),
            flash_coefficient=move(self.flash_coefficient),
            branch_step_index=move(self.branch_step_index),
            trajectory_step_index=move(self.trajectory_step_index),
            transition_std_dev=move(self.transition_std_dev),
            recompute_payload={
                name: move(value) for name, value in self.recompute_payload.items()
            },
        )

    def detach(self) -> RolloutBatch:
        def detached(value: Any) -> Any:
            return value.detach() if hasattr(value, "detach") else value

        return dataclass_replace(
            self,
            media=detached(self.media),
            latents=detached(self.latents),
            next_latents=detached(self.next_latents),
            timesteps=detached(self.timesteps),
            old_log_probs=detached(self.old_log_probs),
            transition_mask=detached(self.transition_mask),
            camera_trajectory=detached(self.camera_trajectory),
            selected_timestep_index=detached(self.selected_timestep_index),
            flash_coefficient=detached(self.flash_coefficient),
            branch_step_index=detached(self.branch_step_index),
            trajectory_step_index=detached(self.trajectory_step_index),
            transition_std_dev=detached(self.transition_std_dev),
            recompute_payload={
                name: detached(value)
                for name, value in self.recompute_payload.items()
            },
        )


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
class RewardBatch:
    """The only finalized reward payload visible outside the reward stack."""

    sample_id: tuple[str, ...]
    raw: Mapping[str, Any]
    weighted: Mapping[str, Any]
    weighted_total: Any
    valid_mask: Any
    shared_metadata: Mapping[str, Mapping[str, Any]]
    sample_metadata: Mapping[str, tuple[Mapping[str, Any], ...]]

    def __post_init__(self) -> None:
        import torch

        if type(self.sample_id) is not tuple or not self.sample_id:
            raise TypeError("sample_id must be a non-empty tuple")
        if any(not isinstance(item, str) or not item for item in self.sample_id):
            raise ValueError("sample_id entries must be non-empty strings")
        if len(set(self.sample_id)) != len(self.sample_id):
            raise ValueError("sample_id entries must be unique")
        batch_size = len(self.sample_id)
        raw = _FrozenTensorMapping(self.raw)
        weighted = _FrozenTensorMapping(self.weighted)
        if tuple(raw) != tuple(weighted):
            raise ValueError("raw and weighted must have identical ordered keys")
        if not raw:
            raise ValueError("RewardBatch must contain at least one reward component")
        for group_name, values_by_name in (("raw", raw), ("weighted", weighted)):
            for name, value in values_by_name.items():
                _validate_reward_tensor(
                    f"{group_name}.{name}",
                    value,
                    batch_size,
                    dtype=torch.float32,
                )
        _validate_reward_tensor(
            "weighted_total",
            self.weighted_total,
            batch_size,
            dtype=torch.float32,
        )
        if not isinstance(self.valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a torch.Tensor")
        _require_detached_tensor("valid_mask", self.valid_mask)
        if (
            self.valid_mask.device.type != "cpu"
            or self.valid_mask.dtype != torch.bool
            or tuple(self.valid_mask.shape) != (batch_size,)
            or not self.valid_mask.is_contiguous()
        ):
            raise ValueError("valid_mask must be contiguous CPU bool with shape [B]")
        if not bool(self.valid_mask.all()):
            raise ValueError("v0.7 RewardBatch requires every row to be valid")

        total = torch.zeros(batch_size, dtype=torch.float32)
        for value in weighted.values():
            total.add_(value)
        if not torch.allclose(
            total,
            self.weighted_total,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError("weighted_total must equal the sum of weighted rewards")

        shared_metadata = FrozenMapping(self.shared_metadata)
        sample_metadata = FrozenMapping(self.sample_metadata)
        if tuple(shared_metadata) != tuple(raw):
            raise ValueError(
                "shared_metadata must contain every reward component in order"
            )
        if tuple(sample_metadata) != tuple(raw):
            raise ValueError(
                "sample_metadata must contain every reward component in order"
            )
        for name in raw:
            rows = sample_metadata[name]
            if type(rows) is not tuple or len(rows) != batch_size:
                raise ValueError(
                    f"sample_metadata[{name!r}] must contain one row per sample"
                )
            if any(not isinstance(row, FrozenMapping) for row in rows):
                raise TypeError(
                    f"sample_metadata[{name!r}] rows must be mappings"
                )

        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "weighted", weighted)
        object.__setattr__(self, "shared_metadata", shared_metadata)
        object.__setattr__(self, "sample_metadata", sample_metadata)

    @property
    def batch_size(self) -> int:
        return len(self.sample_id)

    def validate_against(self, batch: RolloutBatch) -> None:
        if self.sample_id != batch.sample_id:
            raise ValueError("RewardBatch sample_id order must match RolloutBatch")

    def slice(self, indices: Any) -> RewardBatch:
        import torch

        resolved = _validate_sample_indices(indices, self.batch_size)
        tensor_index = torch.tensor(resolved, dtype=torch.long)
        return RewardBatch(
            sample_id=tuple(self.sample_id[index] for index in resolved),
            raw={
                name: value.index_select(0, tensor_index)
                for name, value in self.raw.items()
            },
            weighted={
                name: value.index_select(0, tensor_index)
                for name, value in self.weighted.items()
            },
            weighted_total=self.weighted_total.index_select(0, tensor_index),
            valid_mask=self.valid_mask.index_select(0, tensor_index),
            shared_metadata=self.shared_metadata,
            sample_metadata={
                name: tuple(rows[index] for index in resolved)
                for name, rows in self.sample_metadata.items()
            },
        )


def _validate_reward_tensor(
    name: str,
    value: Any,
    batch_size: int,
    *,
    dtype: Any,
) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    _require_detached_tensor(name, value)
    if (
        value.device.type != "cpu"
        or value.dtype != dtype
        or tuple(value.shape) != (batch_size,)
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous CPU {dtype} with shape [B]"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


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
