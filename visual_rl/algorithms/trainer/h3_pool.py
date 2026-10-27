"""Single-collection, rank-local CPS rollout into bounded CPU latent storage.

No decoding, reward, credit, optimization, or trainer tail is performed here.
The caller supplies initial noise and keeps the sole model in EVAL at a fixed
behavior version. Version IDs must change on every policy mutation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics, H3CPSRecord

if TYPE_CHECKING:
    import torch


def _id(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


class H3RolloutPool:
    """One attempt, sample-major full-N rollout; only complete pools are readable.

    Public records are immutable CPS records with copying tensor accessors;
    terminal reads are copies. Bounds cover retained tensor payload, not Python
    metadata, caller-owned inputs/read copies, or transient model workspace.
    """

    def __init__(
        self, *, collection_id: str, behavior_id: str,
        samples: tuple[tuple[str, str, int], ...], group_size: int,
        dynamics: H3CPSDynamics, max_samples: int, max_cells: int,
        max_trajectory_bytes: int,
    ) -> None:
        _id("collection_id", collection_id)
        _id("behavior_id", behavior_id)
        if not isinstance(dynamics, H3CPSDynamics):
            raise TypeError("dynamics must be H3CPSDynamics")
        for name, value in (("group_size", group_size), ("max_samples", max_samples),
                            ("max_cells", max_cells), ("max_trajectory_bytes", max_trajectory_bytes)):
            if type(value) is not int or value < (2 if name == "group_size" else 1):
                raise ValueError(f"invalid {name}")
        if type(samples) is not tuple or not samples:
            raise ValueError("samples must be a nonempty ordered tuple")
        if len(samples) > max_samples:
            raise ValueError("collection exceeds max_samples")
        cells = len(samples) * dynamics.schedule.num_steps
        if cells > max_cells:
            raise ValueError("collection exceeds max_cells")
        width = dynamics.layout.packed_width
        # Per cell: FP16 x/action + FP32 diagnostic; per sample: FP16 terminal.
        payload_bytes = cells * (4 * width + 4) + len(samples) * 2 * width
        if payload_bytes > max_trajectory_bytes:
            raise ValueError("collection exceeds max_trajectory_bytes")
        ids, groups = set(), {}
        for item in samples:
            if type(item) is not tuple or len(item) != 3:
                raise ValueError("sample must be (sample_id, group_id, member_index)")
            sid, gid, member = item
            _id("sample_id", sid)
            _id("group_id", gid)
            if sid in ids:
                raise ValueError("duplicate sample_id")
            ids.add(sid)
            if type(member) is not int or not 0 <= member < group_size:
                raise ValueError("invalid member_index")
            members = groups.setdefault(gid, set())
            if member in members:
                raise ValueError("duplicate group member_index")
            members.add(member)
        if any(len(members) != group_size for members in groups.values()):
            raise ValueError("every local group must contain all K members")
        self._collection_id = collection_id
        self._behavior_id = behavior_id
        self._samples = samples
        self._dynamics = dynamics
        self._payload_bytes = payload_bytes
        self._state = "new"
        self._records: tuple[tuple[str, H3CPSRecord], ...] = ()
        self._terminals: dict[str, torch.Tensor] = {}

    @property
    def state(self) -> str:
        return self._state

    @property
    def payload_bytes(self) -> int:
        return self._payload_bytes

    def _identity(self, collection_id: str, behavior_id: str) -> None:
        if collection_id != self._collection_id or behavior_id != self._behavior_id:
            raise ValueError("collection/behavior identity mismatch")

    def collect(
        self, initial_states: Mapping[str, torch.Tensor],
        generators: Mapping[str, tuple[torch.Generator, torch.Generator]], *,
        collection_id: str, behavior_id: str, device: torch.device | str,
        predict: Callable[[str, torch.Tensor, int], torch.Tensor],
        read_behavior_id: Callable[[], str],
        status_hook: Callable[..., None] | None = None,
    ) -> None:
        """Validate all samples before the first forward, then publish atomically.

        Each sample owns two distinct explicit RNG streams, also distinct across
        samples. Failure advances any used streams but never retries/reseeds or
        publishes partial trajectories. Callbacks must not retain device inputs.
        """
        import torch

        if self._state != "new":
            raise RuntimeError("pool collect is allowed only once from new")
        self._state = "collecting"
        try:
            self._identity(collection_id, behavior_id)
            device = torch.device(device)
            if device.type not in ("cpu", "cuda"):
                raise ValueError("target device must be CPU or CUDA")
            if device.type == "cuda" and device.index is None:
                raise ValueError("CUDA device must have an explicit index")
            if not callable(predict) or not callable(read_behavior_id):
                raise TypeError("predict/read_behavior_id must be callable")
            ids = {sid for sid, _, _ in self._samples}
            if not isinstance(initial_states, Mapping) or set(initial_states) != ids:
                raise ValueError("initial_states must cover exactly the sample IDs")
            if not isinstance(generators, Mapping) or set(generators) != ids:
                raise ValueError("generators must cover exactly the sample IDs")
            streams, seen_generators = {}, set()
            for sid, _, _ in self._samples:
                x = initial_states[sid]
                self._dynamics.layout.unpack(x)
                if x.shape[0] != 1 or x.dtype != torch.float16 or x.device.type != "cpu":
                    raise ValueError("initial state must be CPU FP16 B1")
                if x.requires_grad or x.grad_fn is not None or not bool(torch.isfinite(x).all()):
                    raise ValueError("initial state must be detached and finite")
                pair = generators[sid]
                if type(pair) is not tuple or len(pair) != 2:
                    raise ValueError("each sample requires video/audio generator pair")
                for generator in pair:
                    if not isinstance(generator, torch.Generator) or generator.device != device:
                        raise ValueError("explicit generators must match target device")
                    if id(generator) in seen_generators:
                        raise ValueError("video/audio and sample generators must be independent")
                    seen_generators.add(id(generator))
                streams[sid] = pair
            # Snapshot every initial state before callbacks can mutate caller inputs.
            # These occupy the reserved terminal slots and are popped as consumed.
            initial = {sid: initial_states[sid].detach().clone() for sid, _, _ in self._samples}
            pending_records, pending_terminals = [], {}
            with torch.no_grad():
                if read_behavior_id() != self._behavior_id:
                    raise ValueError("behavior version changed before collection")
                for sid, _, _ in self._samples:
                    x = initial.pop(sid).to(device)
                    video_rng, audio_rng = streams[sid]
                    for index in range(self._dynamics.schedule.num_steps):
                        ordinal = len(pending_records)
                        if status_hook is not None:
                            status_hook(f"cell/{ordinal}/forward", local_error=False)
                        # Raw forward failure must tear down the attempt, not
                        # enter a possibly unmatched recovery collective.
                        prediction = predict(sid, x.clone(), index)
                        try:
                            record = self._dynamics.sample(
                                x, prediction, index, video_generator=video_rng, audio_generator=audio_rng,
                            )
                            cpu_record = H3CPSRecord(
                                self._dynamics, index, record.x.cpu(), record.action.cpu(), record.rollout_score.cpu(),
                            )
                        except Exception:
                            if status_hook is not None:
                                status_hook(f"cell/{ordinal}/result", local_error=True)
                            raise
                        if status_hook is not None:
                            status_hook(f"cell/{ordinal}/result", local_error=False)
                        pending_records.append((sid, cpu_record))
                        x = record.action
                        del record, prediction, cpu_record
                    pending_terminals[sid] = x.detach().cpu().clone()
                    del x
                if read_behavior_id() != self._behavior_id:
                    raise ValueError("behavior version changed during collection")
            self._records = tuple(pending_records)
            self._terminals = pending_terminals
            self._state = "ready"
        except BaseException:
            self._state = "failed"
            raise

    def _readable(self, collection_id: str, behavior_id: str) -> None:
        self._identity(collection_id, behavior_id)
        if self._state != "ready":
            raise RuntimeError("pool is not ready")

    def samples(self, *, collection_id: str, behavior_id: str) -> tuple[tuple[str, str, int], ...]:
        self._readable(collection_id, behavior_id)
        return self._samples

    def records(self, *, collection_id: str, behavior_id: str) -> tuple[tuple[str, H3CPSRecord], ...]:
        self._readable(collection_id, behavior_id)
        return self._records

    def terminal(self, sample_id: str, *, collection_id: str, behavior_id: str) -> torch.Tensor:
        self._readable(collection_id, behavior_id)
        _id("sample_id", sample_id)
        return self._terminals[sample_id].clone()

    def release(self) -> None:
        """Idempotently discard pool references; future trainer calls after tail.

        Previously borrowed records remain owned by their readers; callers must
        drop those too. This method makes no optimizer/data-tail commit claim.
        """
        if self._state == "collecting":
            raise RuntimeError("cannot release during collection")
        self._records = ()
        self._terminals.clear()
        self._state = "released"
