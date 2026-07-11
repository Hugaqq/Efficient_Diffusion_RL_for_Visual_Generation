from __future__ import annotations


def _flash_config(output_dir):
    from visual_rl.configs.schema import load_config

    config = load_config("visual_rl/configs/presets/flash_tiny_single_step.yaml")
    config.paths.output_dir = str(output_dir)
    config.runner.show_progress = False
    return config


def _optimizer_tensors(optimizer):
    import torch

    values = []
    for state in optimizer.state_dict()["state"].values():
        for key in sorted(state):
            value = state[key]
            if isinstance(value, torch.Tensor):
                values.append((key, value.detach().cpu().clone()))
    return values


def test_resume_matches_continuous_training_parameters_and_adam_state(tmp_path):
    import torch

    from visual_rl.runner import ExperimentRunner

    continuous = ExperimentRunner(_flash_config(tmp_path / "continuous"))
    continuous_metrics = continuous.run(max_steps=2)
    continuous_parameter = continuous.adapter.color_bias.detach().cpu().clone()
    continuous_optimizer = _optimizer_tensors(continuous.optimizer)

    split_dir = tmp_path / "split"
    first = ExperimentRunner(_flash_config(split_dir))
    first.run(max_steps=1)
    state_path = split_dir / "checkpoint_000001" / "training_state.pt"
    assert state_path.exists()
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["implementation"]["adapter"]["class"].endswith(
        ".TinyDiffusionAdapter"
    )
    assert state["implementation"]["trainable_parameters"] == [
        {"name": "color_bias", "shape": [3], "dtype": "torch.float32"}
    ]
    assert state["implementation"]["rollout"]["class"].endswith(
        ".SingleStepRollout"
    )
    assert state["implementation"]["feedback"]["class"].endswith(
        ".RewardRouterFeedbackProvider"
    )
    assert len(state["implementation"]["runtime_tree_sha256"]) == 64

    resume_config = _flash_config(split_dir)
    resume_config.paths.resume_from = str(split_dir / "latest.json")
    resumed = ExperimentRunner(resume_config)
    resumed_metrics = resumed.run(max_steps=2)

    assert [row["step"] for row in resumed_metrics] == [1]
    assert torch.equal(resumed.adapter.color_bias.detach().cpu(), continuous_parameter)
    resumed_optimizer = _optimizer_tensors(resumed.optimizer)
    assert [key for key, _value in resumed_optimizer] == [
        key for key, _value in continuous_optimizer
    ]
    for (_key_a, value_a), (_key_b, value_b) in zip(
        continuous_optimizer,
        resumed_optimizer,
        strict=True,
    ):
        assert torch.equal(value_a, value_b)
    for key in ("loss", "reward_mean", "approx_kl", "clipfrac"):
        assert resumed_metrics[0][key] == continuous_metrics[1][key]
    assert len((split_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_resume_rejects_changed_training_semantics(tmp_path):
    import pytest

    from visual_rl.runner import ExperimentRunner

    run_dir = tmp_path / "run"
    ExperimentRunner(_flash_config(run_dir)).run(max_steps=1)

    changed = _flash_config(run_dir)
    changed.paths.resume_from = str(run_dir / "latest.json")
    changed.algorithm.clip_range = 0.5
    with pytest.raises(RuntimeError, match="does not match checkpoint"):
        ExperimentRunner(changed)


def test_resume_requires_complete_training_state(tmp_path):
    import pytest

    from visual_rl.runner import ExperimentRunner

    run_dir = tmp_path / "run"
    ExperimentRunner(_flash_config(run_dir)).run(max_steps=1)
    (run_dir / "checkpoint_000001" / "training_state.pt").unlink()

    resume = _flash_config(run_dir)
    resume.paths.resume_from = str(run_dir / "latest.json")
    with pytest.raises(RuntimeError, match="missing complete training state"):
        ExperimentRunner(resume)


def test_resume_truncates_artifacts_newer_than_selected_checkpoint(tmp_path):
    import json
    import torch

    from visual_rl.runner import ExperimentRunner

    continuous = ExperimentRunner(_flash_config(tmp_path / "continuous-four"))
    continuous.run(max_steps=4)
    expected_parameter = continuous.adapter.color_bias.detach().cpu().clone()

    run_dir = tmp_path / "interrupted"
    initial_config = _flash_config(run_dir)
    initial_config.train.save_every = 2
    ExperimentRunner(initial_config).run(max_steps=3)
    checkpoint_two = json.loads(
        (run_dir / "checkpoint_000002" / "checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    (run_dir / "latest.json").write_text(
        json.dumps(
            {
                "step": 2,
                "checkpoint": "checkpoint_000002",
                "config_fingerprint": checkpoint_two["config_fingerprint"],
            }
        ),
        encoding="utf-8",
    )

    resume_config = _flash_config(run_dir)
    resume_config.train.save_every = 2
    resume_config.paths.resume_from = str(run_dir / "latest.json")
    resumed = ExperimentRunner(resume_config)
    resumed_metrics = resumed.run(max_steps=4)

    assert [row["step"] for row in resumed_metrics] == [2, 3]
    assert torch.equal(resumed.adapter.color_bias.detach().cpu(), expected_parameter)
    metric_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in metric_rows] == [0, 1, 2, 3]
    manifest = json.loads(
        (run_dir / "sample_manifest.json").read_text(encoding="utf-8")
    )
    assert sorted({record["step"] for record in manifest["records"]}) == [0, 1, 2, 3]


def test_latest_pointer_is_not_committed_when_artifact_write_fails(
    monkeypatch,
    tmp_path,
):
    import pytest

    from visual_rl.runner import ExperimentRunner

    run_dir = tmp_path / "run"
    runner = ExperimentRunner(_flash_config(run_dir))

    def fail_record(**_kwargs):
        raise RuntimeError("artifact failure")

    monkeypatch.setattr(runner.artifacts, "record", fail_record)

    with pytest.raises(RuntimeError, match="artifact failure"):
        runner.run(max_steps=1)

    assert (run_dir / "checkpoint_000001" / "training_state.pt").exists()
    assert not (run_dir / "latest.json").exists()


def test_resume_rejects_checkpoint_metadata_mismatch(tmp_path):
    import json
    import pytest

    from visual_rl.runner import ExperimentRunner

    run_dir = tmp_path / "run"
    ExperimentRunner(_flash_config(run_dir)).run(max_steps=1)
    metadata_path = run_dir / "checkpoint_000001" / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["step"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    resume = _flash_config(run_dir)
    resume.paths.resume_from = str(run_dir / "latest.json")
    with pytest.raises(RuntimeError, match="step does not match"):
        ExperimentRunner(resume)
