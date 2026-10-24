"""CPU integration only: real controller/stages/kernel, fake model/reward leaves."""
import json
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch
import yaml

from tests import test_default_composition_e2e as e2e
from tests import test_v2_demo_default_composition as demo
from visual_rl.runtime.composition import create_default_run_controller


def config(root, artifacts, output, *, resume=None, steps=2, rows=192):
    root.mkdir()
    case = next(c for c in demo._DEMO_CASES if c.recipe_id == "flash_grpo_v1")
    path = demo._write_smoke_config(root, case, output_dir=output, artifacts=artifacts,
                                     resume_from=resume, max_optimizer_steps=steps)
    raw = yaml.safe_load(path.read_text())
    raw["overrides"]["training"].update({
        "global_prompt_batch_size": rows * 2 // 8,
        "collection": {"collection_rows_per_commit": rows, "collection_commits": 2,
                       "collection_chunk_rows": 16},
        "policy_recompute": {"row_microbatch_size": 4, "transition_window_size": 1},
    })
    raw["overrides"]["execution"].update({"group_size": 8, "rollout": {
        "forward_microbatch_size": 4, "decode_microbatch_size": 1,
        "trajectory_storage_device": "cpu"}})
    path.write_text(yaml.safe_dump(raw))
    return path


def setup(monkeypatch, tmp_path):
    import visual_rl
    from visual_rl.algorithms.trainer import collection
    checkout = Path(__file__).resolve().parents[1]
    assert Path(visual_rl.__file__).resolve().parent == checkout / "visual_rl"
    assert Path(collection.__file__).resolve() == checkout / "visual_rl/algorithms/trainer/collection.py"
    # These tests compare multiple fresh controllers in ONE process. Diffusers'
    # lazy HTTP/Rich import initializes a style id using Python's global RNG.
    # Prime that one-time import before either controller seeds its run, as in
    # test_default_composition_continuation_matches_continuous_two_step_run.
    # Keep the exact RNG/checkpoint assertions; do not mask restore differences.
    import importlib
    importlib.import_module("diffusers")
    e2e._patch_default_runtime(monkeypatch)
    case = next(c for c in demo._DEMO_CASES if c.recipe_id == "flash_grpo_v1")
    artifacts = demo._write_fake_artifacts(tmp_path / "artifacts", case)
    code = tmp_path / "code"
    code.mkdir()
    (code / "identity.py").write_text("UNIT_TEST = True\n")
    return artifacts, code


def test_real_controller_collects384_then_two192_and_resumes_without_scoring(monkeypatch, tmp_path, caplog):
    from dataclasses import replace
    from scripts.launch_flash140 import SafePointPause, observe_checkpoints, inspect_checkpoint
    caplog.set_level("INFO", logger="visual_rl.flash140")
    from visual_rl.algorithms.optimization.recompute import PolicyRecomputer
    artifacts, code = setup(monkeypatch, tmp_path)
    replay_calls = []
    compute = PolicyRecomputer.compute_current_slot
    def counted(self, request, slot, **kwargs):
        replay_calls.append(slot.row_indices)
        return compute(self, request, slot, **kwargs)
    monkeypatch.setattr(PolicyRecomputer, "compute_current_slot", counted)
    original_score = demo._FakeRewardResource.score
    def shifted_score(self, *, batch):
        result = original_score(self, batch=batch)
        return replace(result, values=result.values + self.score_calls * 10)
    monkeypatch.setattr(demo._FakeRewardResource, "score", shifted_score)
    rewards = demo._FakeRewardResourceFactory()
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                               reward_resource_factory=rewards)
    checkpoints = []
    observe_checkpoints(controller, pause_after=1, on_checkpoint=checkpoints.append)
    output = tmp_path / "out"
    with pytest.raises(SafePointPause) as paused:
        controller.run(config(tmp_path / "first", artifacts, output))
    assert paused.value.record["status"] == "paused_not_success"
    assert paused.value.record["committed_steps"] == 1
    assert checkpoints[0]["metrics"]["active_transition_count"] == 192
    assert rewards.resources[0].score_calls == 24
    assert not (output / "SUCCESS").exists()
    checkpoint = output / "checkpoints/step-1"
    inspected_first = inspect_checkpoint(checkpoint)
    assert inspected_first["pending_rows"] == 384 and inspected_first["commit_cursor"] == 1
    assert inspected_first["trainable_parameter_count"] == inspected_first["ema_parameter_count"] > 0
    inspection, saved = e2e._checkpoint_snapshot(checkpoint)
    state = saved.component_state("data_plane").payload["runtime_state"]
    assert state["committed_steps"] == state["commit_cursor"] == 1
    assert state["schema_version"] == 2
    assert "behavior" not in state and "sampling_fingerprint" not in state
    assert len(replay_calls) == 48
    assert all(len(rows) == 4 for rows in replay_calls)
    assert inspection.progress.ema_state_saved
    assert state["pool"]["execution"].trajectory.media is None
    assert state["pool"]["execution"].trajectory.batch_size == 384
    reward = state["pool"]["reward"]["weighted_total"]
    grouping = state["pool"]["advantage"].grouping
    expected = np.zeros(384)
    for group in dict.fromkeys(grouping.group_ids):
        rows = np.array([i for i, g in enumerate(grouping.group_ids) if g == group])
        expected[rows] = (reward[rows] - reward[rows].mean()) / (reward.std() + 1e-4)
    np.testing.assert_allclose(state["pool"]["advantage"].values.numpy(), expected, rtol=1e-5, atol=1e-6)
    assert saved.component_state("run_checkpoint_summary").payload.last_metrics["active_transition_count"] == 192
    resumed_rewards = demo._FakeRewardResourceFactory()
    resumed = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                            reward_resource_factory=resumed_rewards)
    result = resumed.run(config(tmp_path / "resume", artifacts, output, resume=checkpoint))
    assert result.committed_steps == 2
    assert resumed_rewards.resources[0].score_calls == 0
    assert result.last_metrics["active_transition_count"] == 192
    assert result.last_metrics["logprob_delta_max"] > 0
    assert result.last_metrics["gradient_norm_pre_clip"] > 0
    assert len(replay_calls) == 96  # 48 per commit, no behavior-policy prepass
    inspected_second = inspect_checkpoint(result.authoritative_checkpoint)
    assert inspected_second["pending_rows"] == 0 and inspected_second["commit_cursor"] == 0
    assert inspected_first["trainable_sha256"] != inspected_second["trainable_sha256"]
    assert inspected_first["ema_sha256"] == inspected_second["ema_sha256"]
    _, final = e2e._checkpoint_snapshot(result.authoritative_checkpoint)
    final_state = final.component_state("data_plane").payload["runtime_state"]
    assert final_state["pool"] is None
    assert final_state["committed_steps"] == 2
    final_inspection, _ = e2e._checkpoint_snapshot(result.authoritative_checkpoint)
    assert inspection.progress.source_cursors == final_inspection.progress.source_cursors
    assert dict(final_inspection.progress.source_cursors)["main"] == 48
    continuous = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                               reward_resource_factory=demo._FakeRewardResourceFactory())
    reference = continuous.run(config(tmp_path / "continuous", artifacts, tmp_path / "continuous-out"))
    e2e._assert_checkpoint_state_exact(e2e._checkpoint_snapshot(reference.authoritative_checkpoint),
                                      e2e._checkpoint_snapshot(result.authoritative_checkpoint),
                                      include_run_summary=False)
    assert len(replay_calls) == 192  # another 2 x 48 in continuous reference
    events = [json.loads(r.message)["event"] for r in caplog.records if r.name == "visual_rl.flash140"]
    assert events.count("collection_chunk_durable") == 48
    assert events.count("collection_start") == 2  # none on odd resume
    assert "checkpoint_durable" in events and "optimizer_commit" in events

    # Same real controller/PPO/std/resume regression at SMALL40's 32 ->16+16.
    replay_calls.clear()
    small_out = tmp_path / "small-out"
    small_rewards = demo._FakeRewardResourceFactory()
    small = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                          reward_resource_factory=small_rewards)
    observe_checkpoints(small, pause_after=1)
    with pytest.raises(SafePointPause):
        small.run(config(tmp_path / "small-first", artifacts, small_out, rows=16))
    assert small_rewards.resources[0].score_calls == 2
    assert len(replay_calls) == 4  # four row4 slots, not an extra prepass
    _, small_saved = e2e._checkpoint_snapshot(small_out / "checkpoints/step-1")
    small_state = small_saved.component_state("data_plane").payload["runtime_state"]
    small_pool = small_state["pool"]
    assert small_pool["execution"].trajectory.batch_size == 32
    reward = small_pool["reward"]["weighted_total"]
    expected = np.zeros(32)
    grouping = small_pool["advantage"].grouping
    for group in dict.fromkeys(grouping.group_ids):
        indices = np.array([i for i, g in enumerate(grouping.group_ids) if g == group])
        expected[indices] = (reward[indices] - reward[indices].mean()) / (reward.std() + 1e-4)
    np.testing.assert_allclose(small_pool["advantage"].values.numpy(), expected, rtol=1e-5, atol=1e-6)
    resumed_rewards = demo._FakeRewardResourceFactory()
    small_resume = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                 reward_resource_factory=resumed_rewards)
    final_small = small_resume.run(config(tmp_path / "small-resume", artifacts, small_out, rows=16,
                                           resume=small_out / "checkpoints/step-1"))
    assert resumed_rewards.resources[0].score_calls == 0
    assert final_small.last_metrics["active_transition_count"] == 16
    assert final_small.last_metrics["logprob_delta_max"] > 0
    assert len(replay_calls) == 8
    continuous_small = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                      reward_resource_factory=demo._FakeRewardResourceFactory())
    reference_small = continuous_small.run(config(tmp_path / "small-continuous", artifacts,
                                                   tmp_path / "small-continuous-out", rows=16))
    e2e._assert_checkpoint_state_exact(e2e._checkpoint_snapshot(reference_small.authoritative_checkpoint),
                                      e2e._checkpoint_snapshot(final_small.authoritative_checkpoint),
                                      include_run_summary=False)


def test_collection_disk_chunks_survive_failure_before_first_commit(monkeypatch, tmp_path, caplog):
    from visual_rl.algorithms.trainer.collection import CollectionState
    artifacts, code = setup(monkeypatch, tmp_path)
    fingerprints = []
    begin = CollectionState.begin
    def observed_begin(self):
        begin(self)
        fingerprints.append(self.sampling_fingerprint)
    monkeypatch.setattr(CollectionState, "begin", observed_begin)
    first_rewards = demo._FakeRewardResourceFactory()
    original = demo._FakeRewardResource.score
    def score_or_fail(self, *, batch):
        if self.score_calls == 3:
            raise RuntimeError("during fourth reward chunk")
        return original(self, batch=batch)
    monkeypatch.setattr(demo._FakeRewardResource, "score", score_or_fail)
    output = tmp_path / "out"
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=first_rewards)
    with pytest.raises(RuntimeError, match="fourth reward chunk"):
        controller.run(config(tmp_path / "first", artifacts, output))
    assert len(list((output / "collection-cache/collection-0").glob("*.pt"))) == 3
    assert not (output / "SUCCESS").exists()
    monkeypatch.setattr(demo._FakeRewardResource, "score", original)
    def changed_begin(self):
        with torch.no_grad():
            self.parameters[0].add_(0.5)
        observed_begin(self)
    monkeypatch.setattr(CollectionState, "begin", changed_begin)
    changed_rewards = demo._FakeRewardResourceFactory()
    changed = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=changed_rewards)
    with pytest.raises(ValueError, match="disk chunk behavior/row identity mismatch"):
        changed.run(config(tmp_path / "changed", artifacts, output))
    assert changed_rewards.resources[0].score_calls == 0
    assert fingerprints[0] != fingerprints[1]
    monkeypatch.setattr(CollectionState, "begin", observed_begin)
    rewards = demo._FakeRewardResourceFactory()
    resumed = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=rewards)
    result = resumed.run(config(tmp_path / "retry", artifacts, output))
    assert result.committed_steps == 2
    assert fingerprints[0] == fingerprints[2]
    assert rewards.resources[0].score_calls == 21  # 48 logical videos already durable
    assert not (output / "collection-cache/collection-0").exists()
    attempts = list((output / "physical-attempts/collection-0").glob("*.json"))
    assert len(attempts) == 24
    assert sum(json.loads(p.read_text())["physical_attempts_started"] for p in attempts) == 25

    # The supported resource-factory seam runs after managed rollout exit, with
    # detached media and no autograd at empty_cache. CUDA counters here are
    # explicit fake leaves: ordering/logging evidence, NOT reclaimed GPU bytes.
    from contextlib import contextmanager
    from dataclasses import replace
    from scripts.flash_small40_repair import CacheHandoffRewardFactory, CacheHandoffHpsClient
    from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract
    from visual_rl.models.lifecycle.components import ComponentManager
    caplog.set_level("INFO", logger="visual_rl.flash140")
    active = []
    released = []
    managed = ComponentManager.execution
    @contextmanager
    def tracking(self, policy):
        with managed(self, policy):
            active.append(policy)
            try:
                yield
            finally:
                active.pop()
    monkeypatch.setattr(ComponentManager, "execution", tracking)
    # Fake CUDA is ONLY active inside release_reward_cache; RNG capture must
    # continue to observe the actual CPU-only runtime.
    from scripts import flash_small40_repair as repair
    release = repair.release_reward_cache
    def handoff(phase):
        assert not active
        with monkeypatch.context() as cuda:
            cuda.setattr(torch.cuda, "synchronize", lambda _: None)
            cuda.setattr(torch.cuda, "mem_get_info", lambda _: (10, 100))
            cuda.setattr(torch.cuda, "memory_allocated", lambda _: 20)
            cuda.setattr(torch.cuda, "memory_reserved", lambda _: 60)
            def empty():
                assert not torch.is_grad_enabled()
                released.append(phase)
            cuda.setattr(torch.cuda, "empty_cache", empty)
            release(phase)
    def enabled_handoff(phase):
        with monkeypatch.context() as cuda:
            cuda.setattr(torch.cuda, "is_available", lambda: True)
            handoff(phase)
    monkeypatch.setattr(repair, "release_reward_cache", enabled_handoff)
    class FakeHps(demo._FakeRewardResource, Hpsv3VideoClient):
        fail = False
        def __init__(self):
            demo._FakeRewardResource.__init__(self)
            self.service_identity, self.contract = "1" * 64, VideoContract(5, 2, 2)
        def score(self, *, batch):
            assert released[-1] == "collection_chunk" and not active
            if self.fail:
                raise RuntimeError("ordinary HPS503")
            return original(self, batch=batch)
        def score_video(self, **kwargs):
            assert released[-1] == "evaluation_video"
            return 0.5, {"real_leaf": False}
    clients = []
    facts = demo._FakeRewardResourceFactory()
    def leaf(request):
        acquired = facts(request)
        acquired.resource.close()
        client = FakeHps()
        clients.append(client)
        return replace(acquired, resource=client)
    wrapped_factory = CacheHandoffRewardFactory(leaf)
    before = len(released)
    cache_controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                      reward_resource_factory=wrapped_factory)
    cache_result = cache_controller.run(config(tmp_path / "handoff", artifacts, tmp_path / "handoff-out", rows=16))
    assert len(released) - before == 2  # one release per generated16 chunk, not per optimizer/slot
    assert clients[-1].score_calls == 2 and clients[-1].close_calls == 1
    assert clients[-1].activate_calls == 1
    baseline = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                               reward_resource_factory=demo._FakeRewardResourceFactory())
    baseline_result = baseline.run(config(tmp_path / "handoff-baseline", artifacts, tmp_path / "handoff-reference", rows=16))
    e2e._assert_checkpoint_state_exact(e2e._checkpoint_snapshot(cache_result.authoritative_checkpoint),
                                      e2e._checkpoint_snapshot(baseline_result.authoritative_checkpoint),
                                      include_run_summary=False)
    FakeHps.fail = True
    failing = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                            reward_resource_factory=wrapped_factory)
    with pytest.raises(RuntimeError, match="ordinary HPS503"):
        failing.run(config(tmp_path / "handoff-failed", artifacts, tmp_path / "handoff-failed-out", rows=16))
    assert not (tmp_path / "handoff-failed-out/SUCCESS").exists()
    assert clients[-1].close_calls == 1
    adapted = CacheHandoffHpsClient(FakeHps())
    assert isinstance(adapted, Hpsv3VideoClient)
    assert adapted.score_video(sample_id="fake") == (0.5, {"real_leaf": False})
    adapted.close()
    events = [json.loads(r.message) for r in caplog.records if r.name == "visual_rl.flash140"]
    before_events = [r for r in events if r["event"] == "reward_cuda_handoff_before"]
    after_events = [r for r in events if r["event"] == "reward_cuda_handoff_after"]
    assert len(before_events) == len(after_events) == len(released)
    assert all(e["allocated_bytes"] == 20 and e["reserved_bytes"] == 60 for e in after_events)


def test_step0_gate_media_release_and_resumed_step1_safety(monkeypatch, tmp_path):
    import gc
    from dataclasses import replace
    from visual_rl.algorithms.optimization.recompute import PolicyRecomputer
    from visual_rl.algorithms.rewards.clients.input import resolve_pointwise_reward_input
    artifacts, code = setup(monkeypatch, tmp_path)
    media_refs = []
    score = demo._FakeRewardResource.score
    def observed_score(self, *, batch):
        media_refs.append(weakref.ref(resolve_pointwise_reward_input(batch).media))
        return score(self, batch=batch)
    monkeypatch.setattr(demo._FakeRewardResource, "score", observed_score)
    compute = PolicyRecomputer.compute_current_slot
    mode = "initial-mismatch"
    def injected(self, request, slot, **kwargs):
        if mode == "initial-mismatch":
            # Finite but wrong saved behavior probability; step0 must reject.
            request.rollout.trajectory.old_log_probs.add_(0.05)
        stats = compute(self, request, slot, **kwargs)
        if mode == "nan":
            return replace(stats, current_log_probs=stats.current_log_probs * float("nan"))
        if mode == "later-ratio":
            # Beyond both the initial tolerance and PPO clip; retains a real
            # differentiable forward and unclipped negative-advantage gradients.
            return replace(stats, current_log_probs=stats.current_log_probs + 0.01)
        return stats
    monkeypatch.setattr(PolicyRecomputer, "compute_current_slot", injected)
    output = tmp_path / "bad-initial"
    def controller(rewards=None):
        return create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
            reward_resource_factory=rewards or demo._FakeRewardResourceFactory())
    with pytest.raises(RuntimeError, match="on-policy old/current ratio parity gate"):
        controller().run(config(tmp_path / "first", artifacts, output))
    assert len(media_refs) == 24
    gc.collect()
    assert all(ref() is None for ref in media_refs)
    assert not (output / "SUCCESS").exists()
    assert not (output / "checkpoints/step-1").exists()

    mode = "normal"
    output = tmp_path / "resume-out"
    first = controller()
    sink = first._backend.checkpoint_sink
    checkpoint_safe_point = sink.checkpoint_safe_point
    def stop(request):
        receipt = checkpoint_safe_point(request)
        raise RuntimeError("after durable step1")
    sink.checkpoint_safe_point = stop
    with pytest.raises(RuntimeError, match="after durable step1"):
        first.run(config(tmp_path / "durable", artifacts, output))
    checkpoint = output / "checkpoints/step-1"
    mode = "nan"
    nan_rewards = demo._FakeRewardResourceFactory()
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        controller(nan_rewards).run(config(tmp_path / "nan", artifacts, output, resume=checkpoint))
    assert nan_rewards.resources[0].score_calls == 0
    assert not (output / "checkpoints/step-2").exists()
    assert not (output / "SUCCESS").exists()
    mode = "later-ratio"
    resumed_rewards = demo._FakeRewardResourceFactory()
    result = controller(resumed_rewards).run(config(tmp_path / "valid", artifacts, output, resume=checkpoint))
    assert result.committed_steps == 2
    assert resumed_rewards.resources[0].score_calls == 0
    assert result.last_metrics["logprob_delta_max"] > 1e-4
    assert result.last_metrics["clipfrac"] > 0
    assert np.isfinite(result.last_metrics["gradient_norm_pre_clip"])
    assert result.last_metrics["gradient_norm_pre_clip"] > 0
    gc.collect()
    assert all(ref() is None for ref in media_refs)


def test_cpu_ema_updates_only_each_eighth_commit_and_restores_on_error():
    from visual_rl.algorithms.trainer.collection import CollectionState
    from visual_rl.algorithms.trainer.config import GRPOTrainerConfig
    from visual_rl.runtime.checkpoint_binding import _validate_data_payload_schema
    parameter = torch.nn.Parameter(torch.tensor(0.))
    state = CollectionState(config=GRPOTrainerConfig(192, 2, 16), parameters=(parameter,),
                            request_factory=None)
    original_id = id(parameter)
    with torch.no_grad():
        parameter.fill_(8.)
    for _ in range(7):
        state.commit()
    assert state.ema[0].item() == 0
    state.commit()
    assert state.ema[0].item() == pytest.approx(8 * (1 - 8 / 17))
    with pytest.raises(RuntimeError):
        with state.parameter_context(state.ema):
            assert parameter.item() != 8
            raise RuntimeError("evaluation failed")
    assert parameter.item() == 8
    assert id(state.parameters[0]) == original_id
    state.begin()
    saved = state.state_dict()
    assert saved["schema_version"] == 2
    assert "behavior" not in saved and "sampling_fingerprint" not in saved
    state.load_state_dict(saved)
    assert state.sampling_fingerprint is None
    assert id(state.parameters[0]) == original_id
    assert state.committed_steps == 8 and state.commit_cursor == 0
    assert torch.equal(state.ema[0], saved["ema"][0])
    outer = dict(schema_version=1, kind="data_plane_prelude_state", prelude_id="test",
                 placement_contract_id="test", phase_schedule_state={}, sampler_state={}, runtime_state=saved)
    _validate_data_payload_schema(outer)
    old = dict(saved, schema_version=1, behavior=[parameter.detach().clone()])
    with pytest.raises(ValueError, match="schema"):
        state.load_state_dict(old)
    with pytest.raises(ValueError, match="collection runtime state"):
        _validate_data_payload_schema(dict(outer, runtime_state=old))
    for bad in (dict(saved, commit_cursor=1), dict(saved, committed_steps=9, commit_cursor=1),
                dict(saved, pool={}), dict(saved, commit_cursor=False)):
        with pytest.raises(ValueError, match="cursor|pool"):
            state.load_state_dict(bad)


def test_real_ode_evaluation_stage_runs300x50_and_reuses_durable_receipts(monkeypatch, tmp_path):
    from dataclasses import replace
    from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract
    from visual_rl.algorithms.trainer import evaluation
    from visual_rl.algorithms.rollout.collector import RolloutCollector
    from visual_rl.artifacts.checkpoint import RankRNGSnapshot
    artifacts, code = setup(monkeypatch, tmp_path)
    evaluation_calls = []
    evaluate = evaluation.FlashEvaluationStage.__call__
    def checked_evaluation(self, step):
        before_rng = RankRNGSnapshot.capture_current(0)
        parameters = self.collection.parameters
        before_weights = [p.detach().clone() for p in parameters]
        try:
            return evaluate(self, step)
        finally:
            assert RankRNGSnapshot.capture_current(0) == before_rng
            assert all(p is old for p, old in zip(self.collection.parameters, parameters, strict=True))
            assert all(torch.equal(p, old) for p, old in zip(parameters, before_weights, strict=True))
            evaluation_calls.append(step)
    monkeypatch.setattr(evaluation.FlashEvaluationStage, "__call__", checked_evaluation)
    # Explicit fake leaves only. Production loader enforces the author SHA.
    monkeypatch.setattr(evaluation, "load_evaluation_prompts", lambda path: tuple(f"eval prompt {i}" for i in range(300)))
    class Client(Hpsv3VideoClient):
        fail = True
        def __init__(self):
            self.service_identity = "1" * 64
            self.contract = VideoContract(5, 2, 2)
            self.score_calls = self.eval_calls = self.close_calls = self.activate_calls = 0
        activate = demo._FakeRewardResource.activate
        close = demo._FakeRewardResource.close
        score = demo._FakeRewardResource.score
        def score_video(self, *, sample_id, prompt, frames_base64):
            import random
            random.random()
            np.random.random()
            torch.rand(1)  # exercise global RNG restoration, including failure
            if type(self).fail and self.eval_calls == 3:
                raise RuntimeError("evaluation interruption")
            assert len(frames_base64) == 5
            self.eval_calls += 1
            return 0.5, {"scored_frame_indices": list(range(5)), "selected_frame_indices": [0], "frame_scores": [0.5] * 5}
    clients = []
    base_factory = demo._FakeRewardResourceFactory()
    def factory(request):
        acquired = base_factory(request)
        client = Client()
        clients.append(client)
        return replace(acquired, resource=client)
    steps = []
    original = RolloutCollector.evaluate_deterministic
    def counted(self, request, *, num_steps):
        steps.append(num_steps)
        return original(self, request, num_steps=num_steps)
    monkeypatch.setattr(RolloutCollector, "evaluate_deterministic", counted)
    output = tmp_path / "out"
    def path(name):
        p = config(tmp_path / name, artifacts, output)
        raw = yaml.safe_load(p.read_text())
        raw["overrides"]["training"]["flash_evaluation"] = True
        p.write_text(yaml.safe_dump(raw))
        return p
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=factory)
    with pytest.raises(RuntimeError, match="evaluation interruption"):
        controller.run(path("first"))
    assert len(list((output / "evaluations/step-0").glob("video-*.json"))) == 3
    Client.fail = False
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=factory)
    result = controller.run(path("retry"))
    assert clients[-1].eval_calls == 297
    assert evaluation_calls == [0, 0]
    assert steps == [50] * 301  # one physical failed RPC attempt, 300 logical IDs
    summary = json.loads((output / "evaluations/step-0/summary.json").read_text())
    assert summary["videos_scored"] == 300
    assert summary["physical_video_attempts_started"] == 301
    _, snapshot = e2e._checkpoint_snapshot(result.authoritative_checkpoint)
    assert snapshot.component_state("data_plane").payload["runtime_state"]["evaluations_completed"] == [0]

    # Real SMALL40 dispatch/controller/terminal hook; only heavy model geometry,
    # scorer and MP4 encoder are explicit CPU fake leaves. Production guards,
    # author source hashes, fixed selection, 40 commits and 0/20/40 stay real.
    from visual_rl.models.implementations.wan import WanT2VAdapter
    from visual_rl.algorithms.trainer.flash_presets import SMALL40
    from scripts.launch_flash140 import SMALL_CONFIG
    author = Path(__file__).resolve().parents[3] / "dataset/video"
    assert author.is_dir()  # task's local author data, not downloaded fixtures
    # Restore the actual loader used in production (the earlier300 case used synthetic prompts).
    monkeypatch.undo()
    e2e._patch_default_runtime(monkeypatch)
    monkeypatch.setattr(evaluation.FlashEvaluationStage, "__call__", checked_evaluation)
    tiny_config = lambda cfg: replace(cfg, height=16, width=16, frames=5)
    latent = WanT2VAdapter.latent_spec_for_batch
    validate = WanT2VAdapter._validate_latent_spec
    def tiny_latent(self, *args, **kwargs):
        original_config = self.config
        self.config = tiny_config(original_config)
        try:
            return latent(self, *args, **kwargs)
        finally:
            self.config = original_config
    def tiny_validate(self, *args, **kwargs):
        original_config = self.config
        self.config = tiny_config(original_config)
        try:
            return validate(self, *args, **kwargs)
        finally:
            self.config = original_config
    monkeypatch.setattr(WanT2VAdapter, "latent_spec_for_batch", tiny_latent)
    monkeypatch.setattr(WanT2VAdapter, "_validate_latent_spec", tiny_validate)
    saved_media = []
    def fake_encoder(path, media, contract):
        assert media.tensor.shape[1] == contract.frames == 81  # not21 latent frames
        assert media.tensor.shape[-2:] == (2, 2)  # explicit CPU fake model
        saved_media.append(weakref.ref(media.tensor))
        path.write_bytes(b"CPU fake MP4 encoder output")
    monkeypatch.setattr(evaluation, "write_evaluation_video", fake_encoder)
    evaluation_calls.clear()
    steps.clear()
    seeds = []
    def small_counted(self, request, *, num_steps):
        seeds.append((request.samples.prompts[0], request.generator.initial_seed()))
        steps.append(num_steps)
        return original(self, request, num_steps=num_steps)
    monkeypatch.setattr(RolloutCollector, "evaluate_deterministic", small_counted)
    class SmallClient(Client):
        fail_terminal = True
        def __init__(self):
            super().__init__()
            self.contract = VideoContract(81, 2, 2)
            self.terminal_calls = 0
        def score_video(self, *, sample_id, prompt, frames_base64):
            assert len(frames_base64) == 81
            if "step-40-" in sample_id:
                if type(self).fail_terminal and self.terminal_calls == 3:
                    raise RuntimeError("terminal40 evaluation interruption")
                self.terminal_calls += 1
            self.eval_calls += 1
            # Descriptive nonconstant values permit checking paired differences.
            score = 0.4 + int(sample_id.split("step-")[1].split("-")[0]) / 1000
            return score, {"scored_frame_indices": list(range(81)),
                           "selected_frame_indices": list(range(24)), "frame_scores": [score] * 81}
    small_clients = []
    def small_factory(request):
        acquired = base_factory(request)
        client = SmallClient()
        small_clients.append(client)
        return replace(acquired, resource=client)
    small_out = tmp_path / "small40-out"
    def small_path(name, resume=None):
        p = config(tmp_path / name, artifacts, small_out, steps=40, rows=16, resume=resume)
        raw = yaml.safe_load(p.read_text())
        approved = yaml.safe_load(SMALL_CONFIG.read_text())["overrides"]
        raw["overrides"]["training"] = approved["training"]
        raw["overrides"]["algorithm"]["params"].update(approved["algorithm"]["params"])
        raw["overrides"]["model"]["params"].update(approved["model"]["params"])
        raw["launch"]["checkpoint_every_optimizer_steps"] = 2
        raw["launch"]["artifacts"]["datasets"]["main"] = str(author / "train.txt")
        raw["launch"]["artifacts"]["evaluation_prompts"] = str(author / "test.txt")
        p.write_text(yaml.safe_dump(raw))
        return p
    from scripts.launch_flash140 import SafePointPause, observe_checkpoints
    from scripts.flash_small40_repair import CacheHandoffRewardFactory, validate_small40_resume
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    observe_checkpoints(controller, pause_after=2)  # test-only interruption, production RUN remains40
    with pytest.raises(SafePointPause):
        controller.run(small_path("small40-first"))
    assert small_clients[0].score_calls == 2 and small_clients[0].eval_calls == 16
    step2 = small_out / "checkpoints/step-2"
    restored = validate_small40_resume(small_out, step2)
    assert restored["committed_steps"] == 2 and restored["evaluations_completed"] == [0]
    latest_file = step2.parent / "latest.json"
    latest_bytes = latest_file.read_bytes()
    latest_file.write_text(json.dumps(dict(json.loads(latest_bytes), step=0)))
    with pytest.raises(RuntimeError, match="latest.json"):
        validate_small40_resume(small_out, step2)
    latest_file.write_bytes(latest_bytes)
    newer = step2.parent / "step-4"
    newer.mkdir()
    with pytest.raises(RuntimeError, match="latest checkpoint"):
        validate_small40_resume(small_out, step2)
    newer.rmdir()  # test-only empty collision, never a real checkpoint
    baseline_receipts = {p.name: p.read_bytes() for p in (small_out / "evaluations/step-0").iterdir()}
    # Actual immutable contract decoding rejects a changed code artifact; no
    # owner upgrade can suppress the production prepared/full restore checks.
    changed_code = tmp_path / "changed-code"
    changed_code.mkdir()
    (changed_code / "identity.py").write_text("DIFFERENT = True\n")
    incompatible = create_default_run_controller(code_root=changed_code, model_loader=demo._FakeModelLoader(),
                                                  reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    from visual_rl.errors import ResumeError
    with pytest.raises(ResumeError, match="recipe_id differs"):
        incompatible.run(small_path("small40-wrong-source", step2))
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    with pytest.raises(RuntimeError, match="terminal40 evaluation interruption"):
        controller.run(small_path("small40-resume2", step2))
    assert {p.name: p.read_bytes() for p in (small_out / "evaluations/step-0").iterdir()} == baseline_receipts
    assert not (small_out / "SUCCESS").exists()
    assert not (small_out / "checkpoints/step-40").exists()
    assert (small_out / "checkpoints/step-38").is_dir()
    assert small_clients[0].score_calls + small_clients[-1].score_calls == 40  # no recollection at step2
    assert evaluation_calls == [0, 20, 40]
    assert steps == [20] * 36  #16+16+3 durable+1 interrupted
    assert [seed for _, seed in seeds[:16]] == list(SMALL40.noise_seeds)
    assert seeds[:16] == seeds[16:32]  # identical prompt noise across EMA rounds
    assert len(list((small_out / "evaluations/step-40").glob("video-*.json"))) == 3
    SmallClient.fail_terminal = False
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    small_result = controller.run(small_path("small40-retry", small_out / "checkpoints/step-38"))
    assert small_result.committed_steps == 40 and (small_out / "SUCCESS").is_file()
    assert evaluation_calls == [0, 20, 40, 40]  # no baseline/step20 redo
    assert small_clients[-1].eval_calls == 13  #3 terminal receipts reused
    assert steps == [20] * 49  #48 logical videos +1 interrupted physical RPC
    small_inspection, small_snapshot = e2e._checkpoint_snapshot(small_result.authoritative_checkpoint)
    assert dict(small_inspection.progress.source_cursors)["main"] == 80  #20 x4 prompts, no extra logical collection
    terminal_state = small_snapshot.component_state("data_plane").payload["runtime_state"]
    assert terminal_state["evaluations_completed"] == [0, 20, 40]
    assert terminal_state["committed_steps"] == 40 and terminal_state["commit_cursor"] == 0
    assert terminal_state["pool"] is None
    assert sorted(int(p.name.removeprefix("step-")) for p in (small_out / "checkpoints").iterdir()
                  if p.name.removeprefix("step-").isdigit()) == list(range(2, 41, 2))
    for step in (0, 20, 40):
        root = small_out / f"evaluations/step-{step}"
        assert len(list(root.glob("*.mp4"))) == 16
        summary = json.loads((root / "summary.json").read_text())
        assert summary["sampling_steps"] == 20 and summary["noise_seeds"] == list(SMALL40.noise_seeds)
        assert summary["decoded_frames"] == 81 and summary["fps"] == 15
        assert summary["videos_scored"] == len(summary["per_prompt"]) == 16
        assert summary["physical_video_attempts_started"] == (17 if step == 40 else 16)
        if step:
            assert summary["paired_differences"]["step-0"]["mean"] == pytest.approx(step / 1000)
    import gc
    gc.collect()
    assert all(ref() is None for ref in saved_media)
    # Fresh continuous reference: exact prepared owners/RNG/sampler/EMA/cursors
    # versus the actual step2 continuation (and later terminal receipt retry).
    resumed_out = small_out
    small_out = tmp_path / "small40-continuous"
    reference = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                              reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    reference_result = reference.run(small_path("small40-continuous-config"))
    e2e._assert_checkpoint_state_exact(e2e._checkpoint_snapshot(reference_result.authoritative_checkpoint),
                                      e2e._checkpoint_snapshot(small_result.authoritative_checkpoint),
                                      include_run_summary=False)
    small_out = resumed_out
    # Even already-completed earlier rounds must retain their videos before
    # finalization; a saved evaluations_completed cursor is not enough.
    terminal_summary = json.loads((small_out / "evaluations/step-40/summary.json").read_text())
    evaluation.validate_small40_evaluation_artifacts(small_out, terminal_summary["recipe_id"])
    baseline_video = small_out / "evaluations/step-0/video-000-seed-42000.mp4"
    baseline_bytes = baseline_video.read_bytes()
    baseline_video.unlink()
    with pytest.raises(ValueError, match="video missing/corrupt"):
        evaluation.validate_small40_evaluation_artifacts(small_out, terminal_summary["recipe_id"])
    baseline_video.write_bytes(baseline_bytes)
    # Existing receipt without its associated MP4 must never be silently reused.
    # Reuse the terminal evaluator before close is not possible; exercise the
    # same real journal path by another even-checkpoint restore.
    (small_out / "evaluations/step-40/video-000-seed-42000.mp4").unlink()
    stale = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                          reward_resource_factory=small_factory)
    stale_out = tmp_path / "stale-small40"
    # Clone only evaluation journals to a new owned test output (no SUCCESS).
    import shutil
    shutil.copytree(small_out / "evaluations", stale_out / "evaluations")
    original_out = small_out
    small_out = stale_out
    with pytest.raises(ValueError, match="video missing/corrupt"):
        stale.run(small_path("stale-small40-case", original_out / "checkpoints/step-38"))
    assert not (stale_out / "SUCCESS").exists()


def test_approved140_candidate_compiles_typed_cadence(monkeypatch, tmp_path):
    from scripts import launch_flash140 as launch
    with pytest.raises(SystemExit) as help_result:
        launch.main(["--help"])
    assert help_result.value.code == 0
    from visual_rl.composition.config.compiler import compile_recipe_v2
    from visual_rl.composition.config.source import load_source_recipe
    from visual_rl.composition.config.bootstrap import bootstrap_recipe_v2
    e2e._patch_default_runtime(monkeypatch)
    path = Path(__file__).parents[1] / "experiments/flash_wan_capacity_20261022/approved140-runtime.yaml"
    source = load_source_recipe(path)
    recipe = compile_recipe_v2(source)
    assert recipe.training.collection.collection_rows_per_commit == 192
    assert recipe.training.collection.collection_commits == 2
    assert recipe.training.collection.collection_chunk_rows == 16
    assert recipe.execution_policy.group_size == 8
    assert recipe.execution_policy.rollout.forward_microbatch_size == 4
    assert recipe.execution_policy.rollout.decode_microbatch_size == 1
    assert recipe.training.policy_recompute.row_microbatch_size == 4
    assert recipe.training.policy_recompute.transition_window_size == 1
    assert recipe.training.update_safety.max_initial_logprob_delta == 1e-4
    assert recipe.training.update_safety.require_initial_clipfrac_zero
    assert recipe.training.global_prompt_batch_size * recipe.execution_policy.group_size == 384
    assert recipe.training.max_optimizer_steps == 140
    assert recipe.training.flash_evaluation
    assert bootstrap_recipe_v2(source).require_launch().artifacts.evaluation_prompts.name == "test.txt"
    raw = yaml.safe_load(path.read_text())
    rendered = launch.render_config(raw, tmp_path / "run", tmp_path / "run/checkpoints/step-1")
    assert rendered["overrides"] == raw["overrides"]
    assert rendered["launch"]["artifacts"] == raw["launch"]["artifacts"]
    assert raw["launch"]["resume_from"] is None
    assert rendered["overrides"]["training"]["max_optimizer_steps"] == 140
    launch_path = tmp_path / "rendered.yaml"
    launch_path.write_text(yaml.safe_dump(rendered))
    assert compile_recipe_v2(load_source_recipe(launch_path)).training == recipe.training
    small_path = launch.preset_config("small40")
    assert launch.preset_config("approved140") == path.resolve()
    with pytest.raises(ValueError, match="unknown"):
        launch.preset_config("unapproved")
    small_source = load_source_recipe(small_path)
    small_recipe = compile_recipe_v2(small_source)
    small_raw = yaml.safe_load(small_path.read_text())
    assert small_recipe.training.flash_preset == "small40"
    assert small_recipe.training.max_optimizer_steps == 40
    assert small_recipe.training.global_prompt_batch_size == 4
    assert small_recipe.training.collection.collection_rows_per_commit == 16
    assert small_recipe.training.collection.collection_commits == 2
    assert small_recipe.training.collection.collection_chunk_rows == 16
    assert small_recipe.training.flash_evaluation
    assert small_recipe.training.policy_recompute == recipe.training.policy_recompute
    assert small_recipe.training.update_safety == recipe.training.update_safety
    assert small_recipe.execution_policy == recipe.execution_policy
    assert small_raw["overrides"]["model"] == raw["overrides"]["model"]
    assert small_raw["overrides"]["algorithm"] == raw["overrides"]["algorithm"]
    assert small_raw["overrides"]["rewards"] == raw["overrides"]["rewards"]
    assert small_raw["launch"]["artifacts"] == raw["launch"]["artifacts"]
    assert small_raw["launch"]["checkpoint_every_optimizer_steps"] == 2
    small_render = launch.render_config(small_raw, tmp_path / "small40-run", None)
    assert small_render["overrides"] == small_raw["overrides"]
    launch.check_preset_mode("small40", "run", None, False)
    launch.check_preset_mode("small40", "preflight", None, False)
    for mode in ("preflight", "run", "inspect"):
        launch.check_preset_mode("small40", mode, Path("owned/checkpoints/step-2"), False)
    for mode, resume, retry in (("validate", None, False), ("run", None, True)):
        with pytest.raises(RuntimeError, match="explicit owned latest"):
            launch.check_preset_mode("small40", mode, resume, retry)
    resumed_render = launch.render_config(small_raw, tmp_path / "small40-run", Path("owned/checkpoints/step-2"))
    assert resumed_render["overrides"] == small_raw["overrides"]
    assert resumed_render["launch"]["resume_from"].endswith("owned/checkpoints/step-2")
    import copy
    from visual_rl.errors import ConfigError
    for target in (2, 39, 140):
        bad = copy.deepcopy(small_raw)
        bad["overrides"]["training"]["max_optimizer_steps"] = target
        with pytest.raises(RuntimeError, match="endpoint"):
            launch.render_config(bad, tmp_path / "bad", None)
        p = tmp_path / f"invalid-small-{target}.yaml"
        p.write_text(yaml.safe_dump(bad))
        with pytest.raises(ConfigError, match="SMALL40"):
            compile_recipe_v2(load_source_recipe(p))
    safe = {"gpu_free_gib": 68.03, "host_available_gib": 88, "disk_free_gib": 1024}
    launch.check_budget(safe, admission=True, disk_required_gib=1024)
    for key in safe:
        with pytest.raises(RuntimeError, match="resource safety"):
            launch.check_budget({**safe, key: safe[key] - 0.01}, admission=True, disk_required_gib=1024)
    launch.check_budget(dict(gpu_free_gib=16, host_available_gib=24, disk_free_gib=64))
    with pytest.raises(RuntimeError, match="resource safety"):
        launch.check_budget({**safe, "gpu_free_gib": float("nan")})
    monitor = launch.ResourceMonitor(tmp_path, tmp_path / "resources.json", gpu_reserved_limit=60)
    observed = {**safe, "torch_peak_reserved_gib": 52.03, "process_peak_rss_gib": 10}
    monkeypatch.setattr(launch, "resource_sample", lambda _: observed)
    monitor.sample()  # explicit CPU sample leaf, never described as GPU evidence
    assert json.loads(monitor.report.read_text())["sampled_minima"]["gpu_free_gib"] == 68.03
    monkeypatch.setattr(launch, "resource_sample", lambda _: {**observed, "gpu_free_gib": 15})
    with pytest.raises(RuntimeError, match="resource safety"):
        monitor.sample()
    assert json.loads(monitor.report.read_text())["sampled_minima"]["gpu_free_gib"] == 15
    monkeypatch.setattr(launch, "resource_sample", lambda _: {**observed, "torch_peak_reserved_gib": 61})
    with pytest.raises(RuntimeError, match="peak reservation"):
        monitor.sample()
    # Explicit user-approved GPU policy: no headroom or reserved-peak abort;
    # host/disk and invalid telemetry still fail, and CUDA errors aren't caught.
    launch.check_budget({**safe, "gpu_free_gib": 0}, admission=True, gpu_memory_policy="oom")
    oom_monitor = launch.ResourceMonitor(tmp_path, tmp_path / "oom-resources.json",
                                       gpu_reserved_limit=60, gpu_memory_policy="oom")
    monkeypatch.setattr(launch, "resource_sample", lambda _: {**observed, "gpu_free_gib": 1,
                                                            "torch_peak_reserved_gib": 77})
    oom_monitor.sample()
    assert oom_monitor.peaks["torch_peak_reserved_gib"] == 77
    for key, low in (("host_available_gib", 23), ("disk_free_gib", 63)):
        with pytest.raises(RuntimeError, match="resource safety"):
            launch.check_budget({**safe, key: low}, gpu_memory_policy="oom")
    with pytest.raises(RuntimeError, match="invalid GPU"):
        launch.check_budget({**safe, "gpu_free_gib": float("nan")}, gpu_memory_policy="oom")
    lock = tmp_path / "launch.lock"
    with launch.exclusive_launch(lock, {"pid": 123}):
        with pytest.raises(RuntimeError, match="duplicate"):
            with launch.exclusive_launch(lock, {"pid": 456}):
                pytest.fail("second owner entered")
    with launch.exclusive_launch(lock, {"pid": 789}):
        assert json.loads(lock.read_text())["pid"] == 789
    identity = {"git_sha": "a" * 40, "source_config_sha256": "b" * 64}
    out = launch.prepare_output(tmp_path / "owned", identity=identity)
    with pytest.raises(FileExistsError):
        launch.prepare_output(out, identity=identity)
    launch.prepare_output(out, identity=identity, retry_incomplete=True)
    with pytest.raises(RuntimeError, match="different source/config"):
        launch.prepare_output(out, identity={**identity, "git_sha": "c" * 40}, retry_incomplete=True)
    checkpoint = out / "checkpoints/step-1"
    checkpoint.mkdir(parents=True)
    launch.prepare_output(out, identity=identity, resume=checkpoint)
    with pytest.raises(RuntimeError, match="explicit --resume"):
        launch.prepare_output(out, identity=identity, retry_incomplete=True)
    (out / "checkpoints/step-2").mkdir()
    with pytest.raises(RuntimeError, match="latest"):
        launch.prepare_output(out, identity=identity, resume=checkpoint)
    (out / "checkpoints/step-3.tmp-orphan").mkdir()
    launch.prepare_output(out, identity=identity, resume=out / "checkpoints/step-2")
    assert (out / "checkpoints/step-3.tmp-orphan").exists()
    (out / "SUCCESS").write_text("terminal")
    with pytest.raises(RuntimeError, match="terminal"):
        launch.prepare_output(out, identity=identity, resume=out / "checkpoints/step-2")

    from scripts import flash_small40_repair as repair
    import hashlib
    # Preserve the historical launcher-only repair proof. The new readable
    # runtime is deliberately NOT eligible to continue that old SMALL40 run.
    historical = "f51033af981204f510ad5e4d76ab47e57f26980c"
    assert repair.git_read(launch.ROOT, "diff", repair.REPAIR_BASE, historical, "--name-only", "--", "visual_rl", "experiments", "data") == ""
    real_proof = repair.prove_launcher_upgrade(launch.ROOT, repair.REPAIR_BASE, small_path)
    assert set(real_proof["unchanged_git_objects"]) == {"visual_rl", "experiments", "data"}
    changed_paths = repair.git_read(launch.ROOT, "diff", repair.REPAIR_BASE, "--name-only", "--", "visual_rl", "experiments", "data")
    assert "visual_rl/runtime/model_binding.py" in changed_paths.splitlines()
    def prospective_git(*args):
        if args[0] == "diff":
            return repair.git_read(launch.ROOT, "diff", repair.REPAIR_BASE, "--name-only")
        return repair.git_read(launch.ROOT, *args)
    with pytest.raises(RuntimeError, match="outside the allowed paths"):
        repair.prove_launcher_upgrade(launch.ROOT, historical, small_path, git=prospective_git)
    new_sha = "f" * 40
    def git(*args):
        if args[0] == "merge-base":
            return repair.REPAIR_BASE
        if args[0] == "diff":
            return "scripts/launch_flash140.py\nscripts/flash_small40_repair.py\ndocs/flash_small40_repair.md"
        if args[0] == "hash-object":
            return "config-blob"
        if args[0] == "rev-parse":
            path = args[1].split(":", 1)[1]
            return "config-blob" if path.endswith(".yaml") else "tree-" + path
        raise AssertionError(args)
    proof = repair.prove_launcher_upgrade(launch.ROOT, new_sha, small_path, git=git)
    assert proof["base_sha"] == repair.REPAIR_BASE
    for failure in ("ancestry", "paths", "runtime", "config"):
        def bad_git(*args):
            if failure == "ancestry" and args[0] == "merge-base":
                return "unrelated"
            if failure == "paths" and args[0] == "diff":
                return "visual_rl/algorithms/trainer/collection.py"
            if failure == "runtime" and args == ("rev-parse", f"{new_sha}:visual_rl"):
                return "different-runtime-tree"
            if failure == "config" and args[0] == "hash-object":
                return "changed-config-bytes"
            return git(*args)
        with pytest.raises(RuntimeError, match="ancestry|runtime|content|config"):
            repair.prove_launcher_upgrade(launch.ROOT, new_sha, small_path, git=bad_git)
    monkeypatch.setattr(repair, "git_read", lambda root, *args: git(*args))
    original = tmp_path / "run-002"
    monkeypatch.setattr(repair, "REPAIR_OUTPUT", original)
    original.mkdir()
    step2 = original / "checkpoints/step-2"
    step2.mkdir(parents=True)
    new_identity = dict(git_sha=new_sha, source_config_sha256=hashlib.sha256(small_path.read_bytes()).hexdigest())
    old_identity = dict(new_identity, git_sha=repair.REPAIR_BASE)
    owner_file = original / "flash140-owner.json"
    launch.publish(owner_file, old_identity)
    def authorize(**kwargs):
        return repair.resume_owner(original, kwargs.pop("identity", new_identity), kwargs.pop("resume", step2),
            upgrade_from=kwargs.pop("upgrade_from", repair.REPAIR_BASE), root=launch.ROOT, config=small_path, **kwargs)
    before_owner = owner_file.read_bytes()
    previous, next_owner = authorize()
    assert previous == old_identity and next_owner == {**new_identity, "launcher_upgrade": proof}
    assert owner_file.read_bytes() == before_owner  # preflight/read-only authorization
    for kw in ({"upgrade_from": None}, {"upgrade_from": "a" * 40},
               {"identity": dict(new_identity, source_config_sha256="a" * 64)},
               {"resume": original / "checkpoints/step-0"}):
        with pytest.raises(RuntimeError, match="different source/config"):
            authorize(**kw)
    launch.prepare_output(original, identity=previous, resume=step2)
    launch.publish(owner_file, next_owner)  # same single atomic owner+audit publication as main
    assert authorize(upgrade_from=None) == (next_owner, next_owner)
    launch.prepare_output(original, identity=next_owner, resume=step2)
    launch.publish(owner_file, {**next_owner, "launcher_upgrade": dict(proof, new_sha="forged")})
    with pytest.raises(RuntimeError, match="audit/source"):
        authorize()
    assert list(original.glob("checkpoints/step-*")) == [step2]  # no reset/conversion


def test_approved140_manifest_counts():
    manifest = json.loads((Path(__file__).parents[1] / "experiments/flash_wan_capacity_20261022/workload-140.json").read_text())
    assert manifest["optimizer_commits_target"] == 140
    assert len(manifest["collections"]) == 70
    assert sum(c["training_video_slots"] for c in manifest["collections"]) == 26880
    assert all(c["training_video_slots"] == 2 * c["nominal_rows_per_commit"] == 384 for c in manifest["collections"])
    assert [e["after_optimizer_commits"] for e in manifest["evaluations"]] == [0, 36, 76, 116]
    assert all(e["prompt_slots"] == 300 and e["sampling_steps"] == 50 for e in manifest["evaluations"])
    assert manifest["evaluation_videos_target"] == 1200
    assert not manifest["additional_final_evaluation"]
    from visual_rl.algorithms.trainer.flash_presets import SMALL40, APPROVED140
    from visual_rl.algorithms.trainer.evaluation import load_evaluation_prompts, select_small40_prompts
    assert APPROVED140.optimizer_steps == 140 and APPROVED140.evaluation_steps == (0, 36, 76, 116)
    assert SMALL40.optimizer_steps == 40
    assert SMALL40.optimizer_steps // 2 == 20
    assert SMALL40.optimizer_steps // 2 * SMALL40.collection_rows == 640
    assert SMALL40.collection_rows == 2 * SMALL40.rows_per_commit == 32
    assert SMALL40.evaluation_steps == (0, 20, 40)
    assert SMALL40.prompt_count * len(SMALL40.evaluation_steps) == 48
    assert SMALL40.sampling_steps == 20 and SMALL40.save_videos
    author = Path(__file__).resolve().parents[3] / "dataset/video"
    selected = select_small40_prompts(load_evaluation_prompts(author / "test.txt"), author / "train.txt")
    assert len(set(selected)) == 16
    train = set((author / "train.txt").read_text().splitlines())
    assert not set(selected).intersection(train)
    small = json.loads((Path(__file__).parents[1] / "experiments/flash_wan_capacity_20261022/workload-small40.json").read_text())
    assert small["optimizer_commits_target"] == small["collection_count"] * small["optimizer_commits_per_collection"] == 40
    assert small["training_videos_target"] == small["collection_count"] * small["training_videos_per_collection"] == 640
    assert small["training_videos_per_collection"] == small["prompts_per_collection"] * small["group_size"] == 32
    assert small["evaluation_steps"] == [0, 20, 40] and small["terminal_evaluation_required"]
    assert small["evaluation_videos_target"] == small["evaluation_prompts"] * 3 == 48
    assert small["logical_hps_frame_scores_target"] == (640 + 48) * 81
