"""Fixed WORLD8 local prompt source for H3IterationStages, not a dataset framework.

UTF-8 LF/CRLF records retain every prompt character except the line terminator.
No BOM, lone CR, empty records, shuffle, hidden RNG, or mid-collection recovery.
The adapter is borrowed, not closed here. Only the active reservation retains
conditions/latents/generators; checkpoint state contains JSON primitives only.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
from visual_rl.algorithms.trainer.h3_stages import H3IterationRequest
from visual_rl.algorithms.trainer.interface import IterationIdentity
from visual_rl.models.h3_interface import H3Generators, H3PromptInput
from visual_rl.models.implementations.h3 import H3Adapter

PARSE_VERSION = "h3.utf8-lf-crlf.v1"
SAMPLING_VERSION = "h3.world8-ordered-cycle.v1"
STREAM_VERSION = "h3.sha256-seeds.v1"
_SCHEMA = "h3.prompt-source.v1"
_FILE_LIMIT = 16 * 1024 * 1024
_MAX_STEP = 2**31 - 1


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file(path, expected):
    if type(expected) is not str or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("explicit lowercase SHA256 required")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("prompt path must be an existing local non-symlink file")
    with path.open("rb") as stream:
        raw = stream.read(_FILE_LIMIT + 1)
    if len(raw) > _FILE_LIMIT or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("prompt file byte limit or SHA256 mismatch/drift")
    return raw


def _prompts(raw):
    text = raw.decode("utf-8", errors="strict")
    if text.startswith("\ufeff"):
        raise ValueError("UTF-8 BOM is not a prompt")
    lines = text.split("\n")
    terminated = text.endswith("\n")
    if terminated:
        lines.pop()  # exactly one final terminator, not an extra empty record
    prompts = []
    for index, line in enumerate(lines):
        has_lf = index < len(lines)-1 or terminated
        if has_lf and line.endswith("\r"):
            line = line[:-1]
        if not line or "\r" in line:
            raise ValueError("empty prompt record or unsupported lone CR")
        prompts.append(line)
    return tuple(prompts)


class H3PromptSource:
    """One complete local K group per collection; global occurrence=step*8+rank.

    reserve never advances progress. commit accepts only the exact active
    request. abort/failing reserve poisons the attempt; use a fresh object and
    a complete boundary state to recover, never re-seed missing samples.
    """
    def __init__(self, *, train_path, eval_path, train_sha256: str, eval_sha256: str,
                 adapter: H3Adapter, dynamics: H3CPSDynamics, rank: int, world_size: int,
                 group_size: int, base_seed: int, device, max_samples: int,
                 max_cells: int, max_trajectory_bytes: int, group_layout: str = 'rank_local'):
        if not isinstance(adapter, H3Adapter) or not isinstance(dynamics, H3CPSDynamics):
            raise TypeError("loaded H3Adapter and exact H3CPSDynamics required")
        if type(world_size) is not int or world_size != 8 or type(rank) is not int or not 0 <= rank < 8:
            raise ValueError("fixed rank placement in WORLD8 required")
        if type(group_size) is not int or not 2 <= group_size <= 512:
            raise ValueError("group_size must be in [2,512]")
        if type(base_seed) is not int or not 0 <= base_seed < 2**63:
            raise ValueError("base_seed must be a 63-bit nonnegative integer")
        device = torch.device(device)
        if (device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None)
                or (device.type == "cuda" and device.index != rank)):
            raise ValueError("device must be CPU or explicit rank-local CUDA")
        for value in (max_samples, max_cells, max_trajectory_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("explicit positive sample/cell/trajectory budgets required")
        if adapter.geometry.joint_layout != dynamics.layout or adapter.meta_model.model.config.get("text_dim") != 5120:
            raise ValueError("adapter geometry/text dimension and dynamics contract disagree")
        width, steps = dynamics.layout.packed_width, dynamics.schedule.num_steps
        if steps > 4096:
            raise ValueError("policy steps exceed the WORLD8 admission bound")
        if group_layout not in ('rank_local', 'paired_4x8') or (group_layout == 'paired_4x8' and group_size != 8):
            raise ValueError('invalid prompt group layout')
        self._group_layout = group_layout
        local_samples = 4 if group_layout == 'paired_4x8' else group_size
        self._groups_per_step = 4 if group_layout == 'paired_4x8' else 8
        required_bytes = local_samples * (steps * (4 * width + 4) + 2 * width)
        if local_samples > max_samples or local_samples * steps > max_cells or required_bytes > max_trajectory_bytes:
            raise ValueError("complete collection exceeds sample/cell/trajectory budget")
        self._paths = (Path(train_path), Path(eval_path))
        self._hashes = (train_sha256, eval_sha256)
        train, heldout = (_prompts(_file(p, h)) for p, h in zip(self._paths, self._hashes))
        if len(train) < 8 or not heldout:
            raise ValueError("train requires at least eight records and eval must be nonempty")
        adapter.verify_coverage(train, heldout)
        first = adapter.encode(H3PromptInput(train[0]))
        self._adapter, self._dynamics = adapter, dynamics
        self._train, self._eval = train, heldout
        self._rank, self._device = rank, device
        self._k, self._seed = group_size, base_seed
        self._budgets = (max_samples, max_cells, max_trajectory_bytes)
        self._source_id = "h3-source:" + _digest((PARSE_VERSION, train_sha256, eval_sha256))
        geometry = dict(geometry=asdict(adapter.geometry), cache_identity=first.cache_identity,
                        model_config_sha256=adapter.meta_model.config_sha256,
                        initial_state_version=adapter.initial_state_version)
        self._geometry_id = "h3-geometry:" + _digest(geometry)
        self._binding = dict(parse=PARSE_VERSION, sampling=SAMPLING_VERSION, stream=STREAM_VERSION,
                             source_id=self._source_id, train_sha256=train_sha256, eval_sha256=eval_sha256,
                             train_count=len(train), eval_count=len(heldout), geometry=geometry,
                             world_size=8, placement="global-occurrence-mod-world", sp=1, tp=1,
                             rollout_microbatch=1, replay_microbatch=1, group_size=group_size,
                             num_steps=steps, video_sigmas=dynamics.schedule.video_sigmas.tolist(),
                             audio_sigmas=dynamics.schedule.audio_sigmas.tolist(), eta=dynamics.eta,
                             base_seed=base_seed, device_type=device.type, budgets=list(self._budgets),
                             score_profile="h3_hpsv3_video_only_v1")
        if group_layout == 'paired_4x8':
            self._binding.update(group_layout=group_layout, placement='adjacent-rank-pairs-4x8-v1')
        self._data_identity = "h3-data:" + _digest(self._binding)
        self._next_step = 0
        self._pending = None
        self._conditions = {}
        self._poisoned = False
        self._used = False

    @property
    def source_id(self):
        return self._source_id

    @property
    def data_identity(self):
        return self._data_identity

    @property
    def geometry_id(self):
        return self._geometry_id

    def _behavior(self, step):
        return "h3-behavior:" + _digest((self._data_identity, step))

    def read_behavior_id(self):
        return self._behavior(self._next_step)

    @property
    def next_global_batch_id(self):
        return "h3-batch:" + _digest((self._data_identity, list(range(self._next_step*self._groups_per_step, (self._next_step+1)*self._groups_per_step))))

    def _check_inputs(self):
        for path, expected in zip(self._paths, self._hashes):
            _file(path, expected)
        adapter = self._adapter
        geometry = self._binding["geometry"]
        if (asdict(adapter.geometry) != geometry["geometry"] or adapter.geometry.joint_layout != self._dynamics.layout
                or adapter.meta_model.config_sha256 != geometry["model_config_sha256"]
                or adapter.initial_state_version != geometry["initial_state_version"]):
            raise ValueError("model/geometry/initial-state contract drift")
        adapter.verify_coverage(self._train, self._eval)  # checks index and shard content drift

    def _stream(self, occurrence, member, purpose, modality, device):
        raw = bytes.fromhex(_digest((STREAM_VERSION, self._seed, self._source_id, occurrence, member, purpose, modality)))
        seed = int.from_bytes(raw[:8], "big") & (2**63 - 1)
        return torch.Generator(device=device).manual_seed(seed)

    def reserve(self, step) -> H3IterationRequest:
        if self._poisoned or self._pending is not None:
            raise RuntimeError("source poisoned or reservation already active")
        self._used = True
        try:
            if type(step) is not int or step != self._next_step or not 0 <= step < _MAX_STEP:
                raise ValueError("reserve requires the exact next optimizer step")
            self._check_inputs()
            occurrence = step*self._groups_per_step + (self._rank//2 if self._group_layout == 'paired_4x8' else self._rank)
            members = range((self._rank % 2)*4, (self._rank % 2 + 1)*4) if self._group_layout == 'paired_4x8' else range(self._k)
            prompt = self._train[occurrence % len(self._train)]
            group = "h3-group:" + _digest((self._source_id, occurrence))
            samples = tuple(("h3-sample:" + _digest((self._source_id, occurrence, member)), group, member)
                            for member in members)
            conditions, initial, generators, typed = {}, {}, {}, {}
            for sid, _, member in samples:
                condition = self._adapter.encode(H3PromptInput(prompt))
                if condition.cache_identity != self._binding["geometry"]["cache_identity"] or condition.prompt != prompt:
                    raise ValueError("condition cache/prompt identity drift")
                typed[sid] = condition
                conditions[sid] = condition.embeddings
                spec = self._adapter.latent_spec_for_batch(H3PromptInput(prompt), device="cpu", dtype=torch.float16)
                initial[sid] = self._adapter.prepare_latents(spec, generator=H3Generators(
                    self._stream(occurrence, member, "initial", "video", "cpu"),
                    self._stream(occurrence, member, "initial", "audio", "cpu")))
                generators[sid] = (self._stream(occurrence, member, "rollout", "video", self._device),
                                   self._stream(occurrence, member, "rollout", "audio", self._device))
            identity = IterationIdentity(step, self._source_id, "train", tuple(s[0] for s in samples),
                                         tuple(s[1] for s in samples), tuple(s[2] for s in samples))
            request = H3IterationRequest(
                identity, "h3-collection:" + _digest((self._data_identity, step)), self.read_behavior_id(),
                self._behavior(step+1), samples, {sid: prompt for sid, _, _ in samples}, conditions,
                self._dynamics, self._geometry_id, initial, generators, self._k, *self._budgets,
                group_layout=self._group_layout,
            )
            self._pending, self._conditions = request, typed
            return request
        except BaseException:
            self._poisoned = True
            self._conditions.clear()
            raise

    def conditioning_for(self, sample_id):
        """Actual adapter-produced H3Conditioning, only for the active reservation."""
        if self._pending is None or self._poisoned:
            raise RuntimeError("no live reservation conditioning")
        return self._conditions[sample_id]

    def commit(self, request):
        if self._poisoned or self._pending is None or request is not self._pending:
            raise RuntimeError("commit requires the exact active request")
        try:
            self._check_inputs()
            self._next_step += 1
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._pending = None
            self._conditions.clear()

    def abort(self, request):
        if self._pending is None or request is not self._pending:
            raise RuntimeError("abort requires the exact active request")
        self._pending = None
        self._conditions.clear()
        self._poisoned = True

    def _state(self, step):
        return dict(schema_version=_SCHEMA, binding=self._binding, data_identity=self._data_identity,
                    rank=self._rank, device=str(self._device), next_optimizer_step=step,
                    global_cursor=step*self._groups_per_step, behavior_id=self._behavior(step),
                    next_global_batch_id="h3-batch:" + _digest((self._data_identity, list(range(step*self._groups_per_step, (step+1)*self._groups_per_step)))))

    def state_dict(self):
        if self._pending is not None or self._poisoned:
            raise RuntimeError("data state is only available at a clean collection boundary")
        self._check_inputs()
        return json.loads(_json(self._state(self._next_step)))  # owned JSON-safe snapshot

    def load_state_dict(self, state):
        if self._used or self._pending is not None or self._poisoned:
            raise RuntimeError("data restore requires a fresh unused source")
        self._used = True
        try:
            self._check_inputs()
            if type(state) is not dict or type(state.get("next_optimizer_step")) is not int:
                raise ValueError("invalid data state schema/step")
            step = state["next_optimizer_step"]
            if not 0 <= step <= _MAX_STEP or _json(state) != _json(self._state(step)):
                raise ValueError("data state identity/topology/stream/progress mismatch")
            self._next_step = step
        except BaseException:
            self._poisoned = True
            raise
