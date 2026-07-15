"""CPU-only regression tests for deterministic checkpoint integrity."""

from __future__ import annotations

import hashlib
import json

import pytest

from visual_rl.artifacts.checkpoint import (
    apply_training_state,
    read_and_validate_training_state,
    save_training_state,
)
from visual_rl.configs.schema import VisualRLConfig, config_to_dict
from visual_rl.core.determinism import assert_runtime, runtime_snapshot
from visual_rl.datasets.prompt_dataset import PromptDataset
from visual_rl.model_adapters.mock import MockWanAdapter
from visual_rl.runner import ExperimentRunner


def _config(output_dir, *, resume_from=None):
    config = VisualRLConfig(run_name="c0-integrity")
    config.paths.output_dir = str(output_dir)
    config.paths.resume_from = None if resume_from is None else str(resume_from)
    config.dataset.prompts = ["a test prompt"]
    config.runner.show_progress = False
    return config


def test_adapter_payload_swap_fails_before_adapter_load(tmp_path, monkeypatch):
    runner = ExperimentRunner(_config(tmp_path / "tampered"))
    runner.run(max_steps=1)
    checkpoint = runner.output_dir / "checkpoint_000001"
    metadata = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )

    import torch

    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert metadata["format_version"] == 4
    assert metadata["adapter_payload_sha256"] == state[
        "adapter_payload_sha256"
    ]

    (checkpoint / "mock_adapter.pt").write_bytes(b"replacement payload")
    load_calls = []
    monkeypatch.setattr(
        MockWanAdapter,
        "load_checkpoint",
        lambda self, path: load_calls.append(path),
    )

    with pytest.raises(
        RuntimeError,
        match="Committed checkpoint tree SHA256 mismatch",
    ):
        ExperimentRunner(
            _config(tmp_path / "tampered", resume_from=checkpoint)
        )
    assert load_calls == []


def test_runtime_flag_drift_is_rejected_and_global_state_is_restored():
    import torch

    expected = runtime_snapshot(enabled=True, seed=17)
    original = bool(torch.backends.cudnn.benchmark)
    try:
        torch.backends.cudnn.benchmark = not original
        with pytest.raises(RuntimeError, match="cudnn_benchmark"):
            assert_runtime(expected, context="in the CPU regression test")
    finally:
        torch.backends.cudnn.benchmark = original

    assert runtime_snapshot(enabled=True, seed=17)[
        "cudnn_benchmark"
    ] == expected["cudnn_benchmark"]


def test_multiline_inline_prompts_are_rejected_but_files_remain_line_based(
    tmp_path,
):
    for prompt in ("first\nsecond", "first\rsecond"):
        with pytest.raises(ValueError, match="CR or LF"):
            PromptDataset.from_config({"prompts": [prompt]})

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("first\nsecond\n", encoding="utf-8")
    dataset = PromptDataset.from_config({"path": str(prompt_file)})
    assert dataset.source_prompts == ["first", "second"]
    assert dataset.content_sha256 == hashlib.sha256(
        b"first\nsecond\n"
    ).hexdigest()


def test_legacy_checkpoint_format_v1_requires_explicit_unsafe_opt_in(tmp_path):
    import torch

    run_dir = tmp_path / "legacy"
    runner = ExperimentRunner(_config(run_dir))
    checkpoint = run_dir / "checkpoint_000001"
    runner.adapter.save_pretrained(str(checkpoint))
    save_training_state(
        checkpoint,
        optimizer=runner.optimizer,
        plugin=runner.optimizer_plugin,
        step=1,
        config=config_to_dict(runner.config),
        implementation=runner.checkpoint_identity,
        config_fingerprint_version=1,
    )

    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["format_version"] = 1
    state.pop("adapter_payload_sha256")
    state["rng"]["numpy"] = __import__("numpy").random.get_state()
    torch.save(state, state_path)

    metadata_path = checkpoint / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 1
    metadata.pop("adapter_payload_sha256")
    metadata.pop("training_state_sha256")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="could not be loaded safely"):
        ExperimentRunner(_config(run_dir, resume_from=checkpoint))

    validated = read_and_validate_training_state(
        checkpoint,
        config=config_to_dict(runner.config),
        implementation=runner.checkpoint_identity,
        allow_unsafe_legacy=True,
    )
    assert apply_training_state(
        validated,
        optimizer=runner.optimizer,
        plugin=runner.optimizer_plugin,
    ) == 1
