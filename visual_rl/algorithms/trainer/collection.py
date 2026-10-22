"""Bounded Flash collection and portable pending-old-policy replay state.

This is a stage/cadence extension of BaseTrainer, not a second training loop.
Only decoded chunk media lives until its real reward returns. The replay pool
contains actual compact transitions and conditioning, never dummy pixels.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

from visual_rl.algorithms.trainer.interface import IterationIdentity, StageValue
from visual_rl.algorithms.trainer.stages import (
    RolloutStage, RolloutStagePayload, RewardedRollout, AdvantagedRollout,
    CreditAssignedRollout,
)
from visual_rl.algorithms.rollout.collector import RolloutEpisode
from visual_rl.algorithms.rollout.interface import RolloutExecution, ModelForwardReplayPlan
from visual_rl.algorithms.dynamics.session import DynamicsSession, ScheduleSnapshot
from visual_rl.algorithms.rewards.types import RewardResult
from visual_rl.core.types import FrozenMapping
from visual_rl.data.samples import NoConditionBatchState


def cpu_tree(value):
    """Move tensor leaves to CPU without copying live model/runtime ports."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if is_dataclass(value):
        return replace(value, **{f.name: cpu_tree(getattr(value, f.name))
                                 for f in fields(value) if f.init})
    if isinstance(value, tuple):
        return tuple(cpu_tree(x) for x in value)
    return value


def snapshot_rows(snapshot, rows):
    return ScheduleSnapshot(
        timesteps=snapshot.timesteps, next_timesteps=snapshot.next_timesteps,
        sigmas=snapshot.sigmas, dt=snapshot.dt,
        dynamics_config_identity=snapshot.dynamics_config_identity,
        scheduler_identity=snapshot.scheduler_identity,
        selection_policy=snapshot.selection_policy,
        selected_policy_step_indices=tuple(snapshot.selected_policy_step_indices[i] for i in rows),
        # This remains the identity of the full collection's selection draw.
        randomness_identity=snapshot.randomness_identity,
        next_selection_state=snapshot.next_selection_state,
    )


def join_conditioning(values):
    import torch
    first = values[0]
    if not is_dataclass(first):
        raise TypeError("streamed collection requires typed dataclass conditioning")
    def join(name):
        items = [getattr(v, name) for v in values]
        if isinstance(items[0], torch.Tensor):
            return torch.cat(items, dim=0)
        if isinstance(items[0], tuple):
            return sum(items, ())
        if all(item is None for item in items):
            return None
        raise TypeError(f"unsupported streamed conditioning field {name}")
    return replace(first, **{f.name: join(f.name) for f in fields(first) if f.init})


def join_trajectories(values):
    import torch
    first = values[0]
    if any(t.kind != "single_step" or not t.reward_media_released
           or not isinstance(t.condition_state, NoConditionBatchState) for t in values):
        raise ValueError("streamed collection is restricted to scored unconditioned Flash replay")
    updates = {}
    for f in fields(first):
        items = [getattr(t, f.name) for t in values]
        if isinstance(items[0], torch.Tensor):
            updates[f.name] = torch.cat(items, dim=0)
        elif isinstance(items[0], tuple):
            updates[f.name] = sum(items, ())
        elif f.name == "condition_state":
            updates[f.name] = NoConditionBatchState(sum(t.batch_size for t in values))
        elif any(item != items[0] for item in items[1:]):
            raise ValueError(f"collection chunk metadata differs: {f.name}")
    return replace(first, **updates)


def join_rewards(values):
    import numpy as np
    first = values[0]
    identity = replace(first.identity, **{
        f.name: sum((getattr(v.identity, f.name) for v in values), ())
        for f in fields(first.identity) if isinstance(getattr(first.identity, f.name), tuple)
    })
    updates = {"identity": identity}
    array_maps = {"component_scores", "weighted_scores", "component_valid_masks", "component_applicable_masks"}
    for f in fields(first):
        if f.name in array_maps:
            updates[f.name] = {k: np.concatenate([getattr(v, f.name)[k] for v in values])
                               for k in getattr(first, f.name)}
        elif f.name in {"weighted_total", "valid_mask"}:
            updates[f.name] = np.concatenate([getattr(v, f.name) for v in values])
        elif f.name == "logical_execution_provenance":
            updates[f.name] = {k: FrozenMapping({"collection_chunks": [
                dict(v.logical_execution_provenance[k]) for v in values
            ]}) for k in first.logical_execution_provenance}
        elif f.name != "identity" and any(getattr(v, f.name) != getattr(first, f.name) for v in values[1:]):
            raise ValueError(f"collection reward metadata differs: {f.name}")
    return replace(first, **updates)


def pack_reward(result):
    return {f.name: (dict(getattr(result, f.name)) if isinstance(getattr(result, f.name), Mapping)
                    else getattr(result, f.name)) for f in fields(result)}


def record_attempt(path, identity):
    """Count physical attempts separately from stable logical row identities."""
    from visual_rl.artifacts.terminal import atomic_write
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = {"identity": identity, "physical_attempts_started": 0}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != identity:
            raise ValueError("physical-attempt journal identity mismatch")
    count = previous["physical_attempts_started"]
    if type(count) is not int or count < 0:
        raise ValueError("invalid physical-attempt counter")
    atomic_write(path, json.dumps({"identity": identity,
        "physical_attempts_started": count + 1}, sort_keys=True).encode())
    return count + 1


def save_chunk(path, payload):
    """Atomic disk chunk: no live ports, graphs, or decoded media."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".chunk-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class StreamedRolloutStage(RolloutStage):
    """Interleave real sampling/decode/reward at a bounded row boundary."""
    def __init__(self, *, reward_pipeline, chunk_rows, collection, cache_dir, recipe_id, **kwargs):
        super().__init__(**kwargs)
        self.reward_pipeline = reward_pipeline
        self.chunk_rows = chunk_rows
        self.collection = collection
        self.cache_dir = Path(cache_dir)
        self.recipe_id = recipe_id

    def __call__(self, value):
        try:
            return self._collect(value)
        except BaseException as error:
            from visual_rl.algorithms.optimization.kernel import _clear_exception_traceback_chain
            _clear_exception_traceback_chain(error)
            del value
            raise error from None

    def _collect(self, value):
        import torch
        from visual_rl.algorithms.trainer.stages import _context
        data = value.payload
        samples = data.samples
        request = self.request_factory(samples, value.identity)
        if request.conditioner is not None or request.guidance is not None:
            raise ValueError("streamed Flash collection requires unconditioned Wan guidance")
        # Draw the complete prompt-keyed timestep mapping ONCE, before chunks.
        virtual = RolloutEpisode(request, None, (), None, None, (), ())
        selection = self.rollout._policy_step_selection(virtual)
        session = DynamicsSession.create(request.dynamics, num_steps=self.rollout.num_steps,
                                         device=request.latent_spec.device, selection=selection)
        behavior_hash = hashlib.sha256()
        for p in self.collection.behavior:
            behavior_hash.update(p.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        collection_key = hashlib.sha256((self.recipe_id + str(value.identity) + behavior_hash.hexdigest()
                                        + session.snapshot.snapshot_identity).encode()).hexdigest()
        cache = self.cache_dir / f"collection-{value.identity.optimizer_step}"
        self.collection.active_cache = cache
        trajectories, rewards, conditions = [], [], []
        for start in range(0, samples.batch_size, self.chunk_rows):
            rows = tuple(range(start, min(start + self.chunk_rows, samples.batch_size)))
            chunk_samples = samples.slice(rows)
            identity = replace(value.identity,
                row_identities=tuple(value.identity.row_identities[i] for i in rows),
                group_ids=tuple(value.identity.group_ids[i] for i in rows),
                member_ids=tuple(value.identity.member_ids[i] for i in rows))
            chunk_path = cache / f"rows-{start}.pt"
            if chunk_path.exists():
                if chunk_path.is_symlink():
                    raise ValueError("collection chunk must be a regular owned file")
                saved = torch.load(chunk_path, map_location="cpu", weights_only=False)
                if saved["collection_key"] != collection_key or saved["row_ids"] != identity.row_identities:
                    raise ValueError("collection disk chunk behavior/row identity mismatch")
                trajectory = saved["trajectory"]
                trajectory.validate()
                if not trajectory.reward_media_released:
                    raise ValueError("collection disk chunk retained unscored media")
                trajectories.append(trajectory)
                rewards.append(RewardResult(**saved["reward"]))
                conditions.append(saved["conditioning"])
                request.generator.set_state(saved["generator_after"])
                del saved, trajectory
                continue
            record_attempt(self.cache_dir.parent / "physical-attempts" / cache.name / f"rows-{start}.json",
                {"collection_key": collection_key, "row_ids": list(identity.row_identities)})
            with _context(self.preprocess_context), torch.no_grad():
                conditioning = cpu_tree(request.adapter.encode(chunk_samples))
            condition_ids = tuple(row.identity for row in chunk_samples.rows)
            chunk_request = replace(request, samples=chunk_samples,
                latent_spec=replace(request.latent_spec, shape=(len(rows), *request.latent_spec.shape[1:])),
                guidance_identity=tuple(request.guidance_identity[i] for i in rows),
                quantization_identity=tuple(request.quantization_identity[i] for i in rows),
                encoded_conditioning=conditioning, model_condition_identity=condition_ids,
                dynamics_session=DynamicsSession.from_snapshot(request.dynamics, snapshot_rows(session.snapshot, rows)))
            with _context(self.rollout_context):
                execution = self.rollout.run_with_snapshot(chunk_request)
            payload = RolloutStagePayload(chunk_samples, chunk_request, execution)
            scored = self.reward_pipeline(StageValue(identity, payload)).payload
            compact = replace(scored.rollout.trajectory, media=None, reward_media_released=True)
            save_chunk(chunk_path, dict(collection_key=collection_key, row_ids=identity.row_identities,
                trajectory=compact, reward=pack_reward(scored.reward_result), conditioning=conditioning,
                generator_after=request.generator.get_state()))
            trajectories.append(compact)
            rewards.append(scored.reward_result)
            conditions.append(conditioning)
            # Do not retain chunk media, VAE outputs or generator-bearing requests.
            del scored, payload, execution, chunk_request, conditioning
        trajectory = join_trajectories(trajectories)
        conditioning = join_conditioning(conditions)
        execution = RolloutExecution(trajectory, session.snapshot, conditioning,
            tuple(row.identity for row in samples.rows),
            ModelForwardReplayPlan.independent(samples.batch_size,
                microbatch_size=self.rollout.rollout_execution_policy.forward_microbatch_size))
        request = replace(request, encoded_conditioning=conditioning,
                          model_condition_identity=execution.model_condition_identity)
        return StageValue(value.identity, RewardedRollout(
            RolloutStagePayload(samples, request, execution, data), join_rewards(rewards)))


class CollectionState:
    """One fixed behavior snapshot, pending credit pool, and CPU LoRA EMA."""
    def __init__(self, *, config, parameters, request_factory, current_context):
        self.config = config
        self.parameters = tuple(parameters)
        self.request_factory = request_factory
        self.current_context = current_context
        self.behavior = None
        self.ema = [p.detach().cpu().clone() for p in self.parameters]
        self.committed_steps = 0
        self.pending = None
        self.commit_cursor = 0
        self.evaluations_completed = []
        self.evaluator = None
        self.active_cache = None
        self.cache_root = None

    def before_iteration(self, step):
        if step != self.committed_steps:
            raise ValueError("collection committed-step cursor differs from trainer")
        if self.evaluator is not None and step in (0, 36, 76, 116) and step not in self.evaluations_completed:
            self.evaluator(step)
            self.evaluations_completed.append(step)

    def close(self):
        self.pending = None
        self.behavior = None
        self.evaluator = None
        self.ema = []
        self.parameters = ()

    def begin(self):
        self.behavior = [p.detach().cpu().clone() for p in self.parameters]

    def behavior_context(self):
        if self.behavior is None:
            raise RuntimeError("missing behavior-policy snapshot")
        return self.parameter_context(self.behavior, execution=True)

    @contextmanager
    def parameter_context(self, values, *, execution=False):
        import torch
        from contextlib import nullcontext
        backup = [p.detach().cpu().clone() for p in self.parameters]
        with torch.no_grad():
            for p, old in zip(self.parameters, values, strict=True):
                p.copy_(old)
        try:
            with (self.current_context() if execution else nullcontext()):
                yield
        finally:
            with torch.no_grad():
                for p, current in zip(self.parameters, backup, strict=True):
                    p.copy_(current)

    def remember(self, value):
        self.pending = value
        self.commit_cursor = 0

    def resume_credit(self, step):
        if self.pending is None:
            return None
        expected = self.pending.identity.optimizer_step + self.commit_cursor
        if step != expected:
            raise ValueError("pending collection commit cursor mismatch")
        return StageValue(replace(self.pending.identity, optimizer_step=step), self.pending.payload)

    def masked(self, value):
        import torch
        credit = value.payload
        rows = self.config.collection_rows_per_commit
        start = self.commit_cursor * rows
        active = credit.loss_inputs.active_mask.clone()
        active[:start] = False
        active[start + rows:] = False
        if credit.advantaged.rewarded.rollout.trajectory.batch_size != rows * self.config.collection_commits:
            raise ValueError("collection cardinality differs from configured commits")
        return StageValue(value.identity, replace(credit,
            loss_inputs=replace(credit.loss_inputs, active_mask=active)))

    def commit(self):
        import torch
        self.committed_steps += 1
        # Explicit commit-based migration of author decay=.9 / interval=8.
        # Author helper uses min((1+s)/(10+s), .9); s is zero-based here.
        if self.committed_steps % 8 == 0:
            decay = min(self.committed_steps / (9 + self.committed_steps), .9)
            with torch.no_grad():
                for shadow, p in zip(self.ema, self.parameters, strict=True):
                    shadow.lerp_(p.detach().cpu(), 1 - decay)
        self.commit_cursor += 1
        if self.commit_cursor == self.config.collection_commits:
            self.pending = None
            self.behavior = None
            self.commit_cursor = 0
            if self.active_cache is not None:
                import shutil
                shutil.rmtree(self.active_cache)
                self.active_cache = None

    def next_batch_identity(self, fallback):
        if self.pending is None:
            return fallback
        start = self.commit_cursor * self.config.collection_rows_per_commit
        rows = self.pending.identity.row_identities[start:start + self.config.collection_rows_per_commit]
        return hashlib.sha256(json.dumps({"pending_behavior_rows": rows,
            "commit_cursor": self.commit_cursor}, sort_keys=True).encode()).hexdigest()

    def state_dict(self):
        pool = None
        if self.pending is not None:
            credit = self.pending.payload
            rollout = credit.advantaged.rewarded.rollout
            result = credit.advantaged.rewarded.reward_result
            reward = pack_reward(result)
            samples = replace(rollout.samples, metadata=tuple(FrozenMapping(m) for m in rollout.samples.metadata))
            pool = dict(identity=self.pending.identity, samples=samples,
                execution=rollout.execution, reward=reward,
                advantage=credit.advantaged.advantage, loss_inputs=credit.loss_inputs)
        return dict(schema_version=1, committed_steps=self.committed_steps,
                    commit_cursor=self.commit_cursor, behavior=self.behavior,
                    ema=self.ema, pool=pool, evaluations_completed=list(self.evaluations_completed))

    def load_state_dict(self, state):
        import torch
        if set(state) != {"schema_version", "committed_steps", "commit_cursor", "behavior", "ema", "pool", "evaluations_completed"} or state["schema_version"] != 1:
            raise ValueError("invalid collection state schema")
        step, cursor = state["committed_steps"], state["commit_cursor"]
        if type(step) is not int or step < 0 or cursor != step % self.config.collection_commits:
            raise ValueError("invalid collection commit cursor")
        for name in ("ema", "behavior"):
            values = state[name]
            if values is None and name == "behavior" and cursor == 0:
                continue
            if not isinstance(values, list) or len(values) != len(self.parameters):
                raise ValueError(f"invalid {name} parameter count")
            for p, saved in zip(self.parameters, values, strict=True):
                if saved.shape != p.shape or saved.dtype != p.dtype or saved.device.type != "cpu" or not bool(torch.isfinite(saved).all()):
                    raise ValueError(f"invalid {name} parameter")
        evaluated = state["evaluations_completed"]
        if (not isinstance(evaluated, list) or evaluated != sorted(set(evaluated))
                or any(type(s) is not int or s not in (0, 36, 76, 116) or s > step for s in evaluated)):
            raise ValueError("invalid evaluation cursor")
        pool = state["pool"]
        pending = None
        if cursor:
            if not isinstance(pool, dict) or pool["identity"].optimizer_step + cursor != step:
                raise ValueError("missing pending old-policy pool")
            request = self.request_factory(pool["samples"], pool["identity"])
            execution = pool["execution"]
            request = replace(request, encoded_conditioning=execution.encoded_conditioning,
                              model_condition_identity=execution.model_condition_identity)
            rollout = RolloutStagePayload(pool["samples"], request, execution)
            rewarded = RewardedRollout(rollout, RewardResult(**pool["reward"]))
            pending = StageValue(pool["identity"], CreditAssignedRollout(
                AdvantagedRollout(rewarded, pool["advantage"]), pool["loss_inputs"]))
        elif pool is not None:
            raise ValueError("unexpected consumed collection pool")
        if pending is not None and self.cache_root is not None:
            candidate = self.cache_root / f"collection-{pending.identity.optimizer_step}"
            self.active_cache = candidate if candidate.exists() else None
        self.pending, self.behavior, self.ema = pending, state["behavior"], state["ema"]
        self.committed_steps, self.commit_cursor = step, cursor
        self.evaluations_completed = list(evaluated)
