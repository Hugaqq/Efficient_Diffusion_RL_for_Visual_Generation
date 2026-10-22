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
    e2e._patch_default_runtime(monkeypatch)
    case = next(c for c in demo._DEMO_CASES if c.recipe_id == "flash_grpo_v1")
    artifacts = demo._write_fake_artifacts(tmp_path / "artifacts", case)
    code = tmp_path / "code"
    code.mkdir()
    (code / "identity.py").write_text("UNIT_TEST = True\n")
    return artifacts, code


def test_real_controller_collects384_then_two192_and_resumes_without_scoring(monkeypatch, tmp_path):
    from dataclasses import replace
    artifacts, code = setup(monkeypatch, tmp_path)
    original_score = demo._FakeRewardResource.score
    def shifted_score(self, *, batch):
        result = original_score(self, batch=batch)
        return replace(result, values=result.values + self.score_calls * 10)
    monkeypatch.setattr(demo._FakeRewardResource, "score", shifted_score)
    rewards = demo._FakeRewardResourceFactory()
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                               reward_resource_factory=rewards)
    sink = controller._backend.checkpoint_sink
    original = sink.checkpoint_safe_point
    def crash(request):
        receipt = original(request)
        assert rewards.resources[0].score_calls == 24
        raise RuntimeError("after first durable commit")
    sink.checkpoint_safe_point = crash
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="after first durable commit"):
        controller.run(config(tmp_path / "first", artifacts, output))
    checkpoint = output / "checkpoints/step-1"
    inspection, saved = e2e._checkpoint_snapshot(checkpoint)
    state = saved.component_state("data_plane").payload["runtime_state"]
    assert state["committed_steps"] == state["commit_cursor"] == 1
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


def test_collection_disk_chunks_survive_failure_before_first_commit(monkeypatch, tmp_path):
    from dataclasses import replace
    artifacts, code = setup(monkeypatch, tmp_path)
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
    rewards = demo._FakeRewardResourceFactory()
    resumed = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=rewards)
    result = resumed.run(config(tmp_path / "retry", artifacts, output))
    assert result.committed_steps == 2
    assert rewards.resources[0].score_calls == 21  # 48 logical videos already durable
    assert not (output / "collection-cache/collection-0").exists()
    attempts = list((output / "physical-attempts/collection-0").glob("*.json"))
    assert len(attempts) == 24
    assert sum(json.loads(p.read_text())["physical_attempts_started"] for p in attempts) == 25


def test_second_commit_still_enforces_true_behavior_parity_and_releases_media(monkeypatch, tmp_path):
    import gc
    from visual_rl.algorithms.trainer.collection import CollectionState
    from visual_rl.algorithms.rewards.clients.input import resolve_pointwise_reward_input
    artifacts, code = setup(monkeypatch, tmp_path)
    media_refs = []
    score = demo._FakeRewardResource.score
    def observed_score(self, *, batch):
        media_refs.append(weakref.ref(resolve_pointwise_reward_input(batch).media))
        return score(self, batch=batch)
    monkeypatch.setattr(demo._FakeRewardResource, "score", observed_score)
    commit = CollectionState.commit
    def corrupt_after_first(self):
        commit(self)
        if self.committed_steps == 1:
            self.behavior[0].add_(0.5)
    monkeypatch.setattr(CollectionState, "commit", corrupt_after_first)
    output = tmp_path / "out"
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                               reward_resource_factory=demo._FakeRewardResourceFactory())
    with pytest.raises(RuntimeError, match="behavior snapshot.*parity gate") as error:
        controller.run(config(tmp_path / "first", artifacts, output))
    assert len(media_refs) == 24
    gc.collect()
    assert all(ref() is None for ref in media_refs)
    assert not (output / "SUCCESS").exists()
    assert (output / "checkpoints/step-1").exists()
    assert not (output / "checkpoints/step-2").exists()


def test_cpu_ema_updates_only_each_eighth_commit_and_restores_on_error():
    from visual_rl.algorithms.trainer.collection import CollectionState
    from visual_rl.algorithms.trainer.config import GRPOTrainerConfig
    from contextlib import nullcontext
    parameter = torch.nn.Parameter(torch.tensor(0.))
    state = CollectionState(config=GRPOTrainerConfig(192, 2, 16), parameters=(parameter,),
                            request_factory=None, current_context=nullcontext)
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


def test_real_ode_evaluation_stage_runs300x50_and_reuses_durable_receipts(monkeypatch, tmp_path):
    from dataclasses import replace
    from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract
    from visual_rl.algorithms.trainer import evaluation
    from visual_rl.algorithms.rollout.collector import RolloutCollector
    artifacts, code = setup(monkeypatch, tmp_path)
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
    assert steps == [50] * 301  # one physical failed RPC attempt, 300 logical IDs
    summary = json.loads((output / "evaluations/step-0/summary.json").read_text())
    assert summary["videos_scored"] == 300
    assert summary["physical_video_attempts_started"] == 301
    _, snapshot = e2e._checkpoint_snapshot(result.authoritative_checkpoint)
    assert snapshot.component_state("data_plane").payload["runtime_state"]["evaluations_completed"] == [0]


def test_approved140_candidate_compiles_typed_cadence(monkeypatch):
    from visual_rl.composition.config.compiler import compile_recipe_v2
    from visual_rl.composition.config.source import load_source_recipe
    from visual_rl.composition.config.bootstrap import bootstrap_recipe_v2
    e2e._patch_default_runtime(monkeypatch)
    path = Path(__file__).parents[1] / "experiments/flash_wan_capacity_20261022/approved140-runtime.yaml"
    source = load_source_recipe(path)
    recipe = compile_recipe_v2(source)
    assert recipe.training.collection.collection_rows_per_commit == 192
    assert recipe.training.global_prompt_batch_size * recipe.execution_policy.group_size == 384
    assert recipe.training.max_optimizer_steps == 140
    assert recipe.training.flash_evaluation
    assert bootstrap_recipe_v2(source).require_launch().artifacts.evaluation_prompts.name == "test.txt"


def test_approved140_manifest_counts():
    manifest = json.loads((Path(__file__).parents[1] / "experiments/flash_wan_capacity_20261022/workload-140.json").read_text())
    assert manifest["optimizer_commits_target"] == 140
    assert len(manifest["collections"]) == 70
    assert sum(c["training_video_slots"] for c in manifest["collections"]) == 26880
    assert all(c["training_video_slots"] == 2 * c["nominal_rows_per_commit"] == 384 for c in manifest["collections"])
