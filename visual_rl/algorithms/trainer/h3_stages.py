"""H3 bindings for the existing GRPOTrainer six-stage control and commit tail.

No runner, model loader, PG owner, reward fallback, or checkpoint implementation.
Prediction/decoder/scorer/coordinator/data capabilities are mandatory bindings.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics, H3CPSRecord
from visual_rl.algorithms.optimization.h3_anchor import H3ReplayAnchor
from visual_rl.algorithms.optimization.h3_world import admit_h3_world, gather_h3_credit
from visual_rl.algorithms.optimization.objective import ClippedSurrogateObjective, PolicyLossInputs
from visual_rl.algorithms.trainer.grpo import GRPOTrainer
from visual_rl.algorithms.trainer.h3_pool import H3RolloutPool
from visual_rl.algorithms.trainer.interface import IterationIdentity, StageValue


@dataclass(frozen=True)
class H3IterationRequest:
    """One reserved data iteration; conditions are keyed by actual sample IDs.

    geometry_id identifies the caller's actual geometry/conditioning profile.
    The numerical layout/dual schedule/eta identity is derived here, not supplied
    as an unrelated label. Data commit must install next_behavior_id exactly.
    """
    identity: IterationIdentity
    collection_id: str
    behavior_id: str
    next_behavior_id: str
    samples: tuple[tuple[str, str, int], ...]
    prompts: dict[str, str]
    conditions: dict[str, torch.Tensor]
    dynamics: H3CPSDynamics
    geometry_id: str
    initial_states: dict[str, torch.Tensor]
    generators: dict[str, tuple[torch.Generator, torch.Generator]]
    group_size: int
    max_samples: int
    max_cells: int
    max_trajectory_bytes: int
    score_profile: str = "h3_hpsv3_video_only_v1"


@dataclass
class _Iteration:
    request: H3IterationRequest
    pool: H3RolloutPool
    plan: Any
    coordinator: Any
    rewards: tuple = ()
    advantages: dict | None = None
    credit: PolicyLossInputs | None = None


@dataclass(frozen=True)
class H3UpdateResult:
    collection_id: str
    behavior_id: str
    next_behavior_id: str
    local_cells: int
    scaled_loss: float
    grad_norm: float


def _checked(coordinator, event, work):
    """Only for local work, NEVER wrap forward/backward or another collective."""
    try:
        result = work()
    except Exception:
        coordinator.gate(event, local_error=True)
        raise
    coordinator.gate(event, local_error=False)
    return result


class H3IterationStages:
    """Six bound stages plus reservation tail, orchestrated only by GRPOTrainer.

    data.reserve(step) -> request; data.commit(request) advances data/collection
    and behavior; data.abort(request) abandons a failed reservation, not weights.
    predict(prepared, *, sample_id, x, index, condition, dynamics) uses the sole
    prepared model. decode(terminal) returns a single-sample media lease.
    """
    def __init__(self, *, data, prepared, predict, decode, score_one,
                 coordination_factory, process_group, rank: int, device,
                 read_behavior_id):
        for obj, method in ((data, "reserve"), (data, "commit"), (data, "abort")):
            if not callable(getattr(obj, method, None)):
                raise TypeError(f"data must implement {method}")
        for fn in (predict, decode, score_one, coordination_factory, read_behavior_id):
            if not callable(fn):
                raise TypeError("H3 stage capabilities must be explicitly callable")
        self.data, self.prepared = data, prepared
        self.predict, self.decode, self.score_one = predict, decode, score_one
        self.coordination_factory = coordination_factory
        self.process_group, self.rank, self.device = process_group, rank, torch.device(device)
        self.read_behavior_id = read_behavior_id
        self._active = None
        self._poisoned = False
        self._expected_behavior = None
        self._last_collection = None
        self._trainer_bound = False
        self._objective = ClippedSurrogateObjective()

    def trainer(self) -> GRPOTrainer:
        if self._trainer_bound:
            raise RuntimeError("H3 stages already bound to their trainer")
        self._trainer_bound = True
        return GRPOTrainer(prelude=self, rollout=self.rollout, reward=self.reward,
                           advantage=self.advantage, credit=self.credit, optimize=self.optimize,
                           close_hooks=(self,))

    def _validate(self, request, step):
        if not isinstance(request, H3IterationRequest) or not isinstance(request.identity, IterationIdentity):
            raise TypeError("reserve must return H3IterationRequest with IterationIdentity")
        if request.identity.optimizer_step != step:
            raise ValueError("reserved step differs from trainer step")
        if type(self.rank) is not int or not 0 <= self.rank < 8 or self.rank != dist.get_rank(self.process_group):
            raise ValueError("rank differs from actual process group")
        if self.device.type not in ("cpu", "cuda") or (self.device.type == "cuda" and self.device.index is None):
            raise ValueError("explicit CPU or indexed CUDA device required")
        if (not isinstance(request.next_behavior_id, str) or not request.next_behavior_id
                or request.next_behavior_id == request.behavior_id):
            raise ValueError("next behavior must be a new explicit version")
        if self._expected_behavior is not None and request.behavior_id != self._expected_behavior:
            raise ValueError("non-contiguous behavior version")
        if request.collection_id == self._last_collection or self.read_behavior_id() != request.behavior_id:
            raise ValueError("reused collection or stale behavior")
        if request.score_profile != "h3_hpsv3_video_only_v1":
            raise ValueError("unsupported H3 reward profile")
        pool = H3RolloutPool(collection_id=request.collection_id, behavior_id=request.behavior_id,
                             samples=request.samples, group_size=request.group_size, dynamics=request.dynamics,
                             max_samples=request.max_samples, max_cells=request.max_cells,
                             max_trajectory_bytes=request.max_trajectory_bytes)
        ids = tuple(row[0] for row in request.samples)
        if (request.identity.row_identities != ids
                or request.identity.group_ids != tuple(row[1] for row in request.samples)
                or request.identity.member_ids != tuple(row[2] for row in request.samples)):
            raise ValueError("sample table differs from iteration identity")
        for mapping in (request.prompts, request.conditions, request.initial_states, request.generators):
            if not isinstance(mapping, dict) or set(mapping) != set(ids):
                raise ValueError("sample bindings must cover the exact reserved IDs")
        seen = set()
        for sid in ids:
            if not isinstance(request.prompts[sid], str) or not request.prompts[sid]:
                raise ValueError("explicit prompt required")
            condition = request.conditions[sid]
            if (not isinstance(condition, torch.Tensor) or condition.ndim != 3
                    or condition.shape[0] != 1 or min(condition.shape[1:]) < 1
                    or condition.device.type != "cpu" or not condition.is_floating_point()
                    or condition.requires_grad or not torch.isfinite(condition).all()):
                raise ValueError("conditions must be detached finite CPU [1,L,D] tensors")
            x = request.initial_states[sid]
            request.dynamics.layout.unpack(x)
            if (x.shape[0] != 1 or x.device.type != "cpu" or x.dtype != torch.float16
                    or x.requires_grad or not torch.isfinite(x).all()):
                raise ValueError("initial states must be detached finite CPU FP16 B1")
            pair = request.generators[sid]
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("explicit AV generators required")
            for generator in pair:
                if (not isinstance(generator, torch.Generator) or generator.device != self.device
                        or id(generator) in seen):
                    raise ValueError("independent target-device generators required")
                seen.add(id(generator))
        if not isinstance(self.prepared.optimizer, torch.optim.AdamW):
            raise TypeError("prepared AdamW required")
        if self.prepared.reducer_divisor != 8:
            raise ValueError("H3 requires explicitly prepared reducer divisor 8")
        if self.prepared.scheduler is not None and not callable(getattr(self.prepared.scheduler, "step", None)):
            raise TypeError("invalid prepared scheduler")
        if not callable(getattr(self.prepared.prepared_root, "eval", None)):
            raise TypeError("prepared model root required")
        return pool

    def build(self, optimizer_step):
        if self._poisoned or self._active is not None:
            raise RuntimeError("H3 attempt poisoned or iteration already active")
        self._poisoned = True  # Cleared ONLY by the complete successful tail.
        request, pool, local_error = None, None, False
        admission = dict(collection_id="", behavior_id="", samples=(), group_size=0,
                         num_steps=0, geometry_id="", schedule_id="", score_profile="")
        try:
            try:
                request = self.data.reserve(optimizer_step)
                pool = self._validate(request, optimizer_step)
                d = request.dynamics
                layout = d.layout
                signature = (layout.video_rows, layout.video_channels, layout.audio_rows, layout.audio_channels,
                             d.schedule.video_sigmas.tolist(), d.schedule.audio_sigmas.tolist(), d.eta,
                             H3CPSRecord.score_family, H3CPSRecord.quantization)
                schedule_id = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
                admission = dict(collection_id=request.collection_id, behavior_id=request.behavior_id,
                                 samples=request.samples, group_size=request.group_size,
                                 num_steps=d.schedule.num_steps, geometry_id=request.geometry_id,
                                 schedule_id=schedule_id, score_profile=request.score_profile)
            except Exception:
                local_error = True
            plan = admit_h3_world(process_group=self.process_group, **admission, local_error=local_error)
            if local_error:
                raise RuntimeError("world admission accepted an invalid local request")
            coordinator = self.coordination_factory(plan)
            for name in ("gate", "aux_turns", "clip_grad_norm"):
                if not callable(getattr(coordinator, name, None)):
                    raise TypeError(f"coordinator lacks {name}")
            # Common non-row reservation semantics must also agree before any
            # model call, particularly the version installed by the full tail.
            binding = (optimizer_step, request.identity.source_id, request.identity.phase_id,
                       request.next_behavior_id)
            coordinator.gate("prelude/binding/" + hashlib.sha256(json.dumps(binding).encode()).hexdigest())
            self._active = _Iteration(request, pool, plan, coordinator)
            return StageValue(request.identity, self._active)
        except BaseException:
            if request is not None:
                self.data.abort(request)
            if pool is not None:
                pool.release()
            raise

    def _context(self, value):
        ctx = self._active
        if ctx is None or value.payload is not ctx or value.identity is not ctx.request.identity:
            raise ValueError("stage context/identity was replaced")
        return ctx

    def _predict(self, ctx, sid, x, index):
        return self.predict(self.prepared, sample_id=sid, x=x, index=index,
                            condition=ctx.request.conditions[sid], dynamics=ctx.request.dynamics)

    def _version(self, ctx, event):
        def check():
            version = self.read_behavior_id()
            if version != ctx.request.behavior_id:
                raise ValueError("behavior changed during iteration")
            return version
        return _checked(ctx.coordinator, event, check)

    def rollout(self, value):
        ctx = self._context(value)
        r = ctx.request
        _checked(ctx.coordinator, "rollout/prepare", self.prepared.prepared_root.eval)
        ctx.pool.collect(r.initial_states, r.generators, collection_id=r.collection_id,
                         behavior_id=r.behavior_id, device=self.device,
                         predict=lambda sid, x, i: self._predict(ctx, sid, x, i),
                         read_behavior_id=lambda: self._version(ctx, "rollout/version"),
                         status_hook=lambda event, **kw: ctx.coordinator.gate("rollout/" + event, **kw))
        ctx.coordinator.gate("rollout/complete")
        return StageValue(value.identity, ctx)

    def reward(self, value):
        ctx = self._context(value)
        r = ctx.request
        def work():
            rows = []
            for sid, gid, member in r.samples:
                terminal = ctx.pool.terminal(sid, collection_id=r.collection_id, behavior_id=r.behavior_id)
                with self.decode(terminal) as media:
                    reward = self.score_one(sample_id=sid, prompt=r.prompts[sid], media=media)
                    if type(reward) is not float or not math.isfinite(reward):
                        raise ValueError("reward must be a finite Python float")
                del terminal, media
                rows.append((sid, gid, member, reward, True))
            return tuple(rows)
        ctx.rewards = ctx.coordinator.aux_turns(work)
        return StageValue(value.identity, ctx)

    def advantage(self, value):
        ctx = self._context(value)
        r = ctx.request
        ctx.advantages = gather_h3_credit(ctx.plan, process_group=self.process_group,
                                          collection_id=r.collection_id, behavior_id=r.behavior_id,
                                          rewards=ctx.rewards)
        return StageValue(value.identity, ctx)

    def credit(self, value):
        ctx = self._context(value)
        def make_credit():
            advantages = torch.stack([ctx.advantages[sid] for sid, _, _ in ctx.request.samples])
            advantages = advantages[:, None].expand(-1, ctx.plan.num_steps).clone()
            return PolicyLossInputs(advantages, torch.ones_like(advantages),
                                    torch.ones_like(advantages, dtype=torch.bool), .005, 0.)
        ctx.credit = _checked(ctx.coordinator, "credit/ready", make_credit)
        return StageValue(value.identity, ctx)

    def optimize(self, value):
        ctx = self._context(value)
        r, coord = ctx.request, ctx.coordinator
        identity = dict(collection_id=r.collection_id, behavior_id=r.behavior_id)
        records = ctx.pool.records(**identity)
        anchor = H3ReplayAnchor(**identity, sample_ids=tuple(sid for sid, _, _ in r.samples),
                                dynamics=r.dynamics, max_cells=r.max_cells,
                                max_trajectory_bytes=r.max_trajectory_bytes)
        ordered = iter(records)
        def predict_old(record):
            sid, expected = next(ordered)
            if record.index != expected.index:
                raise ValueError("anchor sample/index order differs from pool")
            return self._predict(ctx, sid, record.x, record.index)
        anchor.capture(records, **identity, device=self.device, predict=predict_old,
                       read_behavior_id=lambda: self._version(ctx, "anchor/version"),
                       status_hook=lambda event, **kw: coord.gate("anchor/" + event, **kw))
        coord.gate("anchor/complete")
        optimizer = self.prepared.optimizer
        _checked(coord, "current/prepare", lambda: optimizer.zero_grad(set_to_none=True))
        scale = ctx.plan.loss_scale(n_slot=1, reducer_divisor=self.prepared.reducer_divisor)
        total = 0.
        for ordinal, (sid, record) in enumerate(records):
            device_record = H3CPSRecord(r.dynamics, record.index, record.x.to(self.device),
                                        record.action.to(self.device), record.rollout_score.to(self.device))
            coord.gate(f"current/cell/{ordinal}/forward")
            prediction = self._predict(ctx, sid, device_record.x, record.index)
            def local_loss():
                current = r.dynamics.evaluate(device_record, prediction).reshape(1, 1)
                row, index = divmod(ordinal, ctx.plan.num_steps)
                advantage = ctx.credit.base_advantage[row:row+1, index:index+1].to(self.device)
                inputs = PolicyLossInputs(advantage, torch.ones_like(advantage),
                                          torch.ones_like(advantage, dtype=torch.bool), .005, 0.)
                old = anchor.score(**identity, sample_id=sid, index=record.index).to(self.device).reshape(1, 1)
                output = self._objective.compute(old_log_probs=old,
                                                policy_stats=SimpleNamespace(current_log_probs=current),
                                                loss_inputs=inputs)
                loss = output.loss * scale
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite scaled loss")
                return loss
            loss = _checked(coord, f"current/cell/{ordinal}/backward", local_loss)
            total += loss.detach().item()
            loss.backward()  # Normal FSDP synchronization, no no_sync/extra division.
            del loss, local_loss, prediction, device_record
        self._version(ctx, "current/complete")
        norm = coord.clip_grad_norm(self.prepared, max_norm=1.)
        def update():
            optimizer.step()
            if self.prepared.scheduler is not None:
                self.prepared.scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        _checked(coord, "optimizer/complete", update)
        return StageValue(value.identity, H3UpdateResult(r.collection_id, r.behavior_id,
                                                       r.next_behavior_id, len(records), total, norm))

    def commit_iteration(self, identity):
        ctx = self._active
        if ctx is None or identity is not ctx.request.identity:
            raise ValueError("tail identity mismatch")
        r = ctx.request
        # BaseTrainer calls this AFTER optimize returns. Partial step/tail errors
        # leave this binding poisoned; never retry an optimizer update in place.
        ctx.coordinator.gate("tail/ready")
        def commit():
            self.data.commit(r)
            if self.read_behavior_id() != r.next_behavior_id:
                raise ValueError("data tail did not advance behavior version")
        _checked(ctx.coordinator, "tail/complete", commit)
        ctx.pool.release()
        self._last_collection, self._expected_behavior = r.collection_id, r.next_behavior_id
        self._active = None
        self._poisoned = False

    def abort_iteration(self, identity):
        self._poisoned = True
        if self._active is not None:
            if identity is not self._active.request.identity:
                raise ValueError("abort identity mismatch")
            self.data.abort(self._active.request)

    def close(self):
        if self._active is not None:
            self._active.pool.release()
            self._active = None
        self._poisoned = True
