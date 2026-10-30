"""Single-node WORLD8 NCCL owner for the existing controller's create_runtime.

No model/optimizer/HTTP, training loop, environment mutation or CPU fallback.
Only create() performs CUDA/PG actions; importing this module is side-effect free.
Device UUID observation requires get_device_properties(...).uuid (full UUID
string, uuid.UUID or PyTorch _CUuuid); missing capability fails closed, never uses device names.
A matching requested UUID set is NOT authorization, exclusivity or free-memory
proof. Host/GPU resource margins, window and model/API admission remain separate.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import os
import re
from threading import Lock
from uuid import UUID

_SESSION_TIMEOUT = timedelta(seconds=180)
_CREATE_LOCK = Lock()
_UUID_TEXT = re.compile(r"(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def _uuid(value) -> str:
    if isinstance(value, UUID):
        value = str(value)
    if not isinstance(value, str) or not _UUID_TEXT.fullmatch(value):
        raise RuntimeError("CUDA device properties must expose a complete UUID; UUID observation capability unavailable")
    canonical = "GPU-" + str(UUID(value.removeprefix("GPU-")))
    if canonical == "GPU-00000000-0000-0000-0000-000000000000":
        raise ValueError("all-zero GPU UUID is not an observed device")
    return canonical


def _env_int(name: str) -> int:
    value = os.environ.get(name)
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError(f"explicit canonical torchrun integer {name} is required")
    return int(value)


def _profile_identity(profile, facts):
    from visual_rl.composition.config.h3_profile import H3Profile, H3LocalFacts
    from visual_rl.core.identity import canonical_identity
    from visual_rl.core.immutable import FrozenMapping
    from visual_rl.data.h3_source import PARSE_VERSION

    if not isinstance(profile, H3Profile) or not isinstance(facts, H3LocalFacts):
        raise TypeError("session requires compiled H3Profile and actual H3LocalFacts")
    if (facts.binding_id != profile.launch.binding_id
            or canonical_identity("h3.semantic.v1", facts.semantic_payload) != facts.semantic_id):
        raise ValueError("profile/local facts binding or semantic identity mismatch")
    model = asdict(profile.model.config)
    for key in ("max_tokens", "max_embedding_bytes", "max_media_bytes", "max_decode_bytes"):
        model.pop(key)
    payload = facts.semantic_payload
    if (payload.get("model") != FrozenMapping(model)
            or payload.get("math") != FrozenMapping(profile.training.payload())
            or payload.get("seed") != profile.launch.seed
            or payload.get("max_optimizer_steps") != profile.launch.max_optimizer_steps
            or payload.get("declared_model_sha256") != profile.expected["model_sha256"]
            or payload.get("cache_provenance") != profile.expected["cache_identity"]
            or payload.get("hps_service_identity") != profile.expected["hps_service_identity"]
            or payload.get("data") != FrozenMapping(dict(train_sha256=facts.train_sha256,
                eval_sha256=facts.eval_sha256, parse=PARSE_VERSION))
            or payload.get("cache_identity") != facts.cache_identity
            or payload.get("model_config_hashes") != facts.model_config_hashes
            or facts.train_sha256 != profile.expected["train_sha256"]
            or facts.eval_sha256 != profile.expected["eval_sha256"]):
        raise ValueError("local facts do not describe this profile")
    return [profile.source_id, profile.model.declaration_id, facts.binding_id, facts.semantic_id]


def _launch_ranks(profile):
    rank, local_rank = _env_int("RANK"), _env_int("LOCAL_RANK")
    if (_env_int("WORLD_SIZE") != 8 or _env_int("LOCAL_WORLD_SIZE") != 8
            or _env_int("GROUP_WORLD_SIZE") != 1 or _env_int("GROUP_RANK") != 0
            or not 0 <= rank < 8 or local_rank != rank):
        raise ValueError("H3 requires single-node torchrun WORLD8, LOCAL_WORLD_SIZE8 and local_rank==rank")
    master = os.environ.get("MASTER_ADDR")
    if (not isinstance(master, str) or not master or master.strip() != master
            or len(master) > 255 or "://" in master or any(c.isspace() for c in master)):
        raise ValueError("explicit torchrun MASTER_ADDR is required")
    if not 1 <= _env_int("MASTER_PORT") <= 65535:
        raise ValueError("invalid torchrun MASTER_PORT")
    expected = profile.launch.gpu_uuids
    if (type(expected) is not tuple or len(expected) != 8
            or any(_uuid(value) != value for value in expected) or len(set(expected)) != 8):
        raise ValueError("profile must request eight distinct canonical full GPU UUIDs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(visible, str) or tuple(visible.split(",")) != expected:
        raise ValueError("CUDA_VISIBLE_DEVICES must explicitly match the eight ordered profile UUIDs")
    return rank, expected


def _observed_uuids(cuda):
    import torch

    if cuda.device_count() != 8:
        raise ValueError("CUDA must expose exactly the eight requested devices")
    # PyTorch 2.8 exposes a pybind _CUuuid rather than str/uuid.UUID.
    # Convert only that concrete runtime type, not arbitrary stringable objects.
    cuda_uuid_type = getattr(torch._C, "_CUuuid", ())
    values = (getattr(cuda.get_device_properties(index), "uuid", None) for index in range(8))
    result = tuple(_uuid(str(value) if isinstance(value, cuda_uuid_type) else value)
                   for value in values)
    if len(set(result)) != 8:
        raise ValueError("actual CUDA UUID mapping contains duplicate devices")
    return result


class H3RuntimeSession:
    """Own only the WORLD this create() call successfully initialized.

    Metadata properties are read-only. close() has no barrier or rescue
    collective; it destroys only the saved group while it is still WORLD.
    It never destroys an externally substituted group or resets CUDA contexts.
    Selected CUDA device is left selected (no environment/RNG/thread changes).
    """
    __slots__ = ("_process_group", "_rank", "_device", "_gpu_uuids", "_closed")

    def __init__(self):
        raise TypeError("use H3RuntimeSession.create(profile, facts)")

    @classmethod
    def create(cls, profile, facts) -> H3RuntimeSession:
        import torch
        import torch.distributed as dist
        from visual_rl.algorithms.optimization.h3_world import _exchange

        if not _CREATE_LOCK.acquire(blocking=False):
            raise RuntimeError("another H3 session creation is in progress")
        owner = None
        try:
            identity = _profile_identity(profile, facts)
            rank, expected = _launch_ranks(profile)
            if not dist.is_available() or not dist.is_nccl_available():
                raise RuntimeError("H3 requires available torch.distributed NCCL")
            if dist.is_initialized():
                raise RuntimeError("refusing an already initialized foreign process group")
            if not torch.cuda.is_available():
                raise RuntimeError("H3 requires CUDA; no CPU production fallback")
            observed = _observed_uuids(torch.cuda)
            if observed != expected:
                raise ValueError("actual logical CUDA UUID mapping differs from requested order")
            device = torch.device("cuda", rank)
            torch.cuda.set_device(device)
            if torch.cuda.current_device() != rank:
                raise ValueError("failed to select the requested logical CUDA device")
            try:
                bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            except (AttributeError, TypeError) as exc:
                raise RuntimeError("H3 requires the public native-BF16 capability query") from exc
            if not bf16:
                raise RuntimeError("selected CUDA device lacks native BF16 support")
            # No retry. Failure inside init before it returns does not confer
            # ownership of any unknown/foreign group left by external code.
            dist.init_process_group(backend="nccl", init_method="env://", timeout=_SESSION_TIMEOUT)
            owner = object.__new__(cls)
            owner._process_group = dist.group.WORLD
            owner._rank, owner._device, owner._gpu_uuids = rank, device, observed
            owner._closed = False
            pg = owner._process_group
            if (pg is None or not dist.is_initialized() or dist.get_backend(pg) != "nccl"
                    or dist.get_world_size(pg) != 8 or dist.get_rank(pg) != rank):
                # A malformed/non-WORLD8 transport cannot join the fixed receipt.
                raise ValueError("initialized group is not the requested NCCL WORLD8/rank")
            local_error = None
            try:
                if torch.cuda.current_device() != rank or _observed_uuids(torch.cuda) != observed:
                    raise ValueError("CUDA selection/UUID mapping changed during initialization")
                payload = [identity, list(observed), rank, 8]
            except Exception as exc:
                local_error, payload = exc, None
            try:
                peers = _exchange("session/admission", rank, pg, payload, local_error is None)
            except BaseException as exc:
                if local_error is not None:
                    raise exc from local_error
                raise
            if local_error is not None:
                raise local_error
            for peer, receipt in enumerate(peers):
                if receipt != [identity, list(observed), peer, 8]:
                    raise ValueError("WORLD session profile/facts/UUID/local-rank metadata mismatch")
            return owner
        except BaseException as error:
            if owner is not None:
                try:
                    owner.close()
                except BaseException as cleanup_error:
                    error.add_note(f"owned WORLD cleanup also failed: {type(cleanup_error).__name__}")
            raise
        finally:
            _CREATE_LOCK.release()

    @property
    def process_group(self):
        if self._closed:
            raise RuntimeError("H3 runtime session is closed")
        return self._process_group

    @property
    def rank(self):
        return self._rank

    @property
    def local_rank(self):
        return self._rank

    @property
    def local_world_size(self):
        return 8

    @property
    def device(self):
        return self._device

    @property
    def gpu_uuids(self):
        return self._gpu_uuids

    def close(self) -> None:
        import torch.distributed as dist

        if self._closed:
            return
        self._closed = True
        pg, self._process_group = self._process_group, None
        if pg is not None and dist.is_initialized() and dist.group.WORLD is pg:
            dist.destroy_process_group(pg)
