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


def config(root, artifacts, output, *, resume=None, steps=2):
    root.mkdir()
    case = next(c for c in demo._DEMO_CASES if c.recipe_id == "flash_grpo_v1")
    path = demo._write_smoke_config(root, case, output_dir=output, artifacts=artifacts,
                                     resume_from=resume, max_optimizer_steps=steps)
    raw = yaml.safe_load(path.read_text())
    raw["overrides"]["training"].update({
        "global_prompt_batch_size": 48,
        "collection": {"collection_rows_per_commit": 192, "collection_commits": 2,
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


def test_collection_disk_chunks_survive_failure_before_first_commit(monkeypatch, tmp_path):
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
