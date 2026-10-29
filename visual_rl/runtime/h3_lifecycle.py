"""H3 lifecycle bindings for the existing RunController and GRPOTrainer."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

from visual_rl.composition.config.source import load_source_recipe
from visual_rl.composition.config.h3_profile import compile_h3_profile, preflight_h3_profile
from visual_rl.core.serialization import to_plain_dict
from visual_rl.errors import ConfigError


class RecipeLifecycleBackend:
    """Select one lifecycle at compile time; never introduce another controller."""
    def __init__(self, default, *, code_root=None):
        self.default = self.selected = default
        self.code_root = code_root

    def compile(self, config_path):
        source = load_source_recipe(config_path)
        if source.values.get("recipe") == "h3_t2va_mvp_v1":
            self.selected = H3LifecycleBackend(code_root=self.code_root)
            return self.selected.compile(source)
        return self.default.compile(config_path)

    def __getattr__(self, name):
        return getattr(self.selected, name)


def _phase(session, label, work):
    """Exchange failures of LOCAL work only, never wrap a model collective."""
    from visual_rl.algorithms.optimization.h3_world import _exchange
    result = error = None
    try:
        result = work()
    except Exception as exc:
        error = exc
    try:
        _exchange("lifecycle/" + label, session.rank, session.process_group, None, error is None)
    except BaseException as exc:
        if error is not None:
            raise error from exc
        raise
    if error is not None:
        raise error
    return result


def _execution_device(session):
    return session.device


def _write_json(path, payload):
    from visual_rl.artifacts.terminal import atomic_write, canonical_line
    atomic_write(Path(path), canonical_line(payload))


def _hps(profile, session):
    from visual_rl.algorithms.rewards.h3_hps import H3HPSReward
    from visual_rl.algorithms.rewards.resource_descriptor import RewardResourceDescriptor, RewardRuntimePolicy
    from visual_rl.composition.config.specs import RewardRuntimeBindingSpec
    from visual_rl.composition.preflight.types import RuntimeFacts
    from visual_rl.core.filesystem_identity import filesystem_file_identity_from_snapshot
    from visual_rl.core.identity import canonical_identity
    from visual_rl.core.immutable import FrozenMapping
    from visual_rl.core.protocols.flash_hpsv3 import PROTOCOL, VERSION, MAX_HEALTH_BYTES, VideoContract
    from visual_rl.runtime.reward_resources import RewardResourceAcquireRequest
    launch, config = profile.launch, profile.model.config
    contract = VideoContract(config.frames, config.height, config.width)
    semantic = dict(service_identity=profile.expected["hps_service_identity"], video_contract=contract.payload())
    with launch.hps_manifest.open("rb") as stream:
        raw = stream.read(MAX_HEALTH_BYTES + 1)
    if len(raw) > MAX_HEALTH_BYTES:
        raise ValueError("HPS manifest exceeds size limit")
    descriptor = RewardResourceDescriptor(1,
        "visual_rl.algorithms.rewards.clients.hpsv3:Hpsv3VideoResourceFactory", "h3-hps", PROTOCOL, VERSION,
        FrozenMapping(semantic), RewardRuntimePolicy(("cpu",), ("fp32",), ("remote",)))
    runtime_binding = RewardRuntimeBindingSpec(artifact_ref="h3-hps", execution_domain="remote", device="cpu",
        dtype="fp32", endpoint=launch.hps_origin, timeout_s=launch.hps_timeout_s,
        trusted_hosts=(urlsplit(launch.hps_origin).hostname,), ca_bundle=None,
        max_response_bytes=launch.hps_max_response_bytes)
    request = RewardResourceAcquireRequest(
        reward_resource_spec_id=canonical_identity("reward-resource-spec.v1", semantic), descriptor=descriptor,
        immutable_artifact_identity=FrozenMapping(filesystem_file_identity_from_snapshot(raw)),
        artifact_location=launch.hps_manifest,
        runtime_facts=RuntimeFacts("fsdp", session.rank, session.rank, 8, str(session.device), "bf16", "nccl"),
        runtime_binding=runtime_binding)
    return H3HPSReward(request, video_contract=contract)


@dataclass
class _Graph:
    session: object
    profile: object
    facts: object
    adapter: object
    source: object
    scorer: object
    resources: ExitStack
    run_id: str
    code_root: Path
    code_hash: str

    def close(self):
        self.resources.close()


class H3LifecycleBackend:
    def __init__(self, *, code_root=None):
        self.code_root = Path(code_root or Path(__file__).resolve().parents[1]).resolve()
        self.binding = self.transaction = None
        self.start_step = 0

    def compile(self, source):
        try:
            return compile_h3_profile(source)
        except (ValueError, TypeError, OSError) as exc:
            raise ConfigError(str(exc)) from exc

    def preflight(self, profile):
        from visual_rl.composition.preflight.artifacts import _hash_artifact
        try:
            facts = preflight_h3_profile(profile)
            for artifact in (profile.launch.model, profile.launch.text_cache, self.code_root):
                if profile.launch.output_dir.is_relative_to(artifact):
                    raise ValueError("run output must not be inside model/cache/source artifacts")
            code_hash = _hash_artifact(self.code_root, content_policy="python-code.v1")["content_sha256"]
            return SimpleNamespace(profile=profile, facts=facts, code_hash=code_hash)
        except (ValueError, TypeError, OSError) as exc:
            raise ConfigError(str(exc)) from exc

    def create_runtime(self, preflight):
        from visual_rl.runtime.h3_session import H3RuntimeSession
        return H3RuntimeSession.create(preflight.profile, preflight.facts)

    def construct_graph(self, preflight, runtime):
        from visual_rl.models.implementations.h3 import H3Adapter
        from visual_rl.models.lifecycle.components import ComponentManager
        from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
        from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
        from visual_rl.data.h3_source import H3PromptSource
        from visual_rl.artifacts.checkpoint import AtomicCheckpointManager
        from visual_rl.artifacts.checkpoint.h3_transaction import H3CheckpointTransaction
        profile, facts = preflight.profile, preflight.facts
        launch = profile.launch
        output = launch.output_dir
        owner = dict(schema_version=1, semantic_id=facts.semantic_id, code=preflight.code_hash,
                     gpu_uuids=list(launch.gpu_uuids))

        resources = ExitStack()
        def acquire_output_lock():
            if runtime.rank != 0:
                return
            output.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(output.parent / ('.' + output.name + '.h3.lock'),
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            stream = os.fdopen(fd, 'a+b')
            resources.callback(stream.close)
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def admit_output():
            if runtime.rank != 0:
                return
            if launch.resume_from is None:
                output.mkdir(parents=True, exist_ok=False)
                _write_json(output / "h3-run.json", dict(owner, run_id=uuid4().hex))
            else:
                saved = json.loads((output / "h3-run.json").read_text())
                if {k: saved.get(k) for k in owner} != owner or not isinstance(saved.get("run_id"), str):
                    raise ValueError("output belongs to a different H3 run/code/topology")
                manager = AtomicCheckpointManager(output / "checkpoints")
                latest = manager.latest_complete()
                if (launch.resume_from.parent != output / "checkpoints" or latest is None
                        or latest.path != launch.resume_from):
                    raise ValueError("resume must select this run's latest complete checkpoint")
                if (output / "SUCCESS").exists():
                    raise ValueError("run already succeeded; do not train it again")
        try:
            _phase(runtime, "output-lock", acquire_output_lock)
            _phase(runtime, "output", admit_output)
            run_id = _phase(runtime, "read-run", lambda: json.loads((output / "h3-run.json").read_text())["run_id"])
            def local_graph():
                adapter = H3Adapter.from_config(profile.model.config, runtime_context=dict(
                    artifact_paths={"h3-model": launch.model, "h3-text": launch.text_cache},
                    expected_cache_identity=to_plain_dict(profile.expected["cache_identity"])))
                resources.callback(adapter.close)
                manager = ComponentManager(adapter, execution_device="cpu")
                resources.callback(manager.close)
                manager.load()
                dynamics = H3CPSDynamics(adapter.geometry.joint_layout, H3ScheduleSnapshot.build(profile.training.num_steps))
                source = H3PromptSource(train_path=launch.train, eval_path=launch.eval,
                    train_sha256=facts.train_sha256, eval_sha256=facts.eval_sha256,
                    adapter=adapter, dynamics=dynamics, rank=runtime.rank, world_size=8,
                    group_size=profile.training.group_size, base_seed=launch.seed, device=_execution_device(runtime),
                    max_samples=profile.training.group_size,
                    max_cells=profile.training.group_size * profile.training.num_steps,
                    max_trajectory_bytes=launch.budgets.trajectory_bytes)
                scorer = _hps(profile, runtime)
                resources.callback(scorer.close)
                scorer.acquire()
                return adapter, source, scorer
            adapter, source, scorer = _phase(runtime, "construct", local_graph)
            self.transaction = H3CheckpointTransaction(AtomicCheckpointManager(output / "checkpoints"),
                                                       process_group=runtime.process_group)
            return _Graph(runtime, profile, facts, adapter, source, scorer, resources, run_id,
                          self.code_root, preflight.code_hash)
        except BaseException:
            resources.close()
            raise

    def prepare(self, graph, runtime):
        from visual_rl.models.lifecycle.h3_fsdp import prepare_h3_fsdp
        # prepare owns its internal pre-collective error gates; no rescue collective here.
        return prepare_h3_fsdp(graph.adapter.meta_model, process_group=runtime.process_group,
            device=runtime.device, rank=runtime.rank, local_rank=runtime.local_rank,
            local_world_size=runtime.local_world_size,
            host_weight_limit_bytes=graph.profile.launch.budgets.host_weight_bytes,
            init_seed=graph.profile.launch.seed, constant_scheduler=True)

    def restore_prepared(self, preflight, runtime, graph, prepared):
        from visual_rl.runtime.h3_binding import H3RuntimeBinding
        # Observe the real prepared contract before the prepared restore gate.
        self.binding = _phase(runtime, "prepared-contract", lambda: H3RuntimeBinding(
            profile=graph.profile, facts=graph.facts, adapter=graph.adapter, prepared=prepared,
            source=graph.source, scorer=graph.scorer, process_group=runtime.process_group,
            rank=runtime.rank, device=runtime.device, gpu_uuids=runtime.gpu_uuids,
            code_root=graph.code_root, expected_code_sha256=graph.code_hash))
        try:
            path = graph.profile.launch.resume_from
            if path is not None:
                live = _phase(runtime, "prepared-live", self.binding.validate_live)
                self.transaction.restore_prepared(prepared, path, expected_contract=self.binding.contract,
                    identity=self.binding.state_identity, bindings=live)
            return self.binding  # existing controller ledger owns close even if later binding fails
        except BaseException:
            self.binding.close()
            raise

    def transform(self, preflight, runtime, graph, prepared):
        return None  # AC/FSDP transforms are already owned by prepare, never applied twice.

    def bind(self, preflight, runtime, graph, prepared, transformed):
        from visual_rl.algorithms.trainer.h3_stages import H3IterationStages
        binding = self.binding
        def assemble():
            options = dict(data=graph.source, prepared=prepared,
                predict=binding.predict, decode=binding.decode, score_one=binding.score_one,
                coordination_factory=binding.coordination_factory, process_group=runtime.process_group,
                rank=runtime.rank, device=_execution_device(runtime), read_behavior_id=graph.source.read_behavior_id)
            if graph.profile.training.algorithm == 'cps':
                stages = H3IterationStages(**options)
            else:
                from visual_rl.algorithms.trainer.h3_dual_stages import H3DualIterationStages
                stages = H3DualIterationStages(mode=graph.profile.training.algorithm,
                                               seed=graph.profile.launch.seed, **options)
            trainer = stages.trainer()
            return SimpleNamespace(graph=graph, prepared=prepared, binding=binding, stages=stages,
                trainer=trainer, transaction=self.transaction, close=trainer.close)
        return _phase(runtime, "bind-stages", assemble)

    def restore_bound(self, preflight, runtime, graph, prepared, bound, prepared_restore):
        if graph.profile.launch.resume_from is None:
            return None
        def data_state():
            state = graph.source.state_dict()
            return dict(schema_version=1, rank=runtime.rank, world_size=8,
                content_id=graph.source.data_identity.split(":", 1)[1], source_id=graph.source.source_id,
                next_optimizer_step=state["next_optimizer_step"],
                next_global_batch_id=graph.source.next_global_batch_id.split(":", 1)[1],
                cursor=state["global_cursor"], behavior_id=state["behavior_id"], open_reservations=0, payload=state)
        progress = bound.transaction.restore_bound(
            validate_live=lambda: (bound.binding.contract, bound.binding.state_identity, bound.binding.validate_live()),
            restore_data=graph.source.load_state_dict, read_data=data_state)
        self.start_step = progress.next_optimizer_step
        if not 0 < self.start_step <= graph.profile.launch.max_optimizer_steps:
            raise ValueError("checkpoint position exceeds the configured training target")
        return None

    def prepare_run(self, bound):
        from visual_rl.algorithms.trainer.interface import PrepareRunContext
        bound.trainer.prepare_run(PrepareRunContext(bound.graph.run_id, "h3_t2va_mvp_v1", self.start_step))

    def run(self, bound):
        launch = bound.graph.profile.launch
        committed = None
        metrics = None
        for step in range(self.start_step, launch.max_optimizer_steps):
            result = bound.trainer.run_iteration(step)
            update = result.value.payload
            from visual_rl.algorithms.optimization.h3_world import _exchange
            losses = _exchange('lifecycle/loss', bound.graph.session.rank,
                               bound.graph.session.process_group, float(update.scaled_loss), True)
            metrics = dict(step=step, sample_count=8 * bound.graph.profile.training.group_size,
                active_transition_count=8 * update.local_cells,
                loss=sum(losses) / 8, gradient_norm=float(update.grad_norm))
            # Persist diagnostics before publishing the checkpoint that can depend on them.
            def record_metrics():
                if bound.graph.session.rank == 0:
                    _write_json(launch.output_dir / "h3-last-metrics.json", metrics)
            _phase(bound.graph.session, "metrics", record_metrics)
            if (step + 1) % launch.checkpoint_every_optimizer_steps == 0 or step + 1 == launch.max_optimizer_steps:
                boundary, live = _phase(bound.graph.session, 'checkpoint-boundary', lambda: (
                    bound.binding.checkpoint_boundary(bound.stages, bound.trainer), bound.binding.validate_live()))
                progress, proof, data, _ = boundary
                committed = bound.transaction.checkpoint(bound.prepared, contract=bound.binding.contract,
                    progress=progress, safe_point=proof, identity=bound.binding.state_identity,
                    bindings=live, data_state=data)
        if committed is None:
            committed = bound.transaction.manager.inspect_complete(launch.resume_from).checkpoint
            metrics = json.loads((launch.output_dir / "h3-last-metrics.json").read_text())
        return committed, metrics

    def checkpoint(self, bound, result):
        from visual_rl.runtime.types import RunResult
        from visual_rl.artifacts.terminal import canonical_line, atomic_write
        graph = bound.graph
        output = graph.profile.launch.output_dir
        committed, metrics = result
        count = graph.profile.launch.max_optimizer_steps
        if committed.step != count or bound.trainer.next_optimizer_step != count:
            raise ValueError("H3 final checkpoint does not match the complete training target")
        training = graph.profile.training
        policy_steps = 1 if training.algorithm == "dual_flash" else training.num_steps
        expected_counts = dict(step=count - 1, sample_count=8 * training.group_size,
            active_transition_count=8 * training.group_size * policy_steps)
        if (set(metrics) != {*expected_counts, 'loss', 'gradient_norm'}
                or any(type(metrics[k]) is not int or metrics[k] != v for k, v in expected_counts.items())
                or any(type(metrics[k]) is not float or not math.isfinite(metrics[k])
                       for k in ('loss', 'gradient_norm'))):
            raise ValueError('terminal metrics do not match the complete checkpoint')
        marker = dict(schema_version=1, kind="visual_rl_h3_run_success", run_id=graph.run_id,
            committed_steps=count, checkpoint_relative_path=str(committed.path.relative_to(output)),
            checkpoint_contract_id=committed.checkpoint_contract_id,
            progress_id=committed.progress_id, state_tree_id=committed.state_tree_id)
        def finalize():
            if graph.session.rank != 0:
                return
            bound.transaction.manager.inspect_complete(committed.path, expected_contract=bound.binding.contract)
            _write_json(output / "resolved_recipe.json", to_plain_dict(graph.facts.semantic_payload))
            _write_json(output / "run_manifest.json", dict(marker, gpu_uuids=list(graph.profile.launch.gpu_uuids),
                score="h3_hpsv3_video_only_v1", audio_quality_scored=False))
            atomic_write(output / "metrics.jsonl", canonical_line(dict(schema_version=1, **metrics)))
            _write_json(output / "SUCCESS", marker)
        _phase(graph.session, "finalize", finalize)
        return RunResult(graph.run_id, output, count, committed.path, output / "resolved_recipe.json",
            output / "run_manifest.json", output / "metrics.jsonl", output / "SUCCESS", metrics)
