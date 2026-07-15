"""Unit checks for the remote W5 evidence topology classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import visual_rl


_MODULE_PATH = Path(__file__).with_name("remote_wan_resume_run.py")
_SPEC = importlib.util.spec_from_file_location("remote_wan_resume_run", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
trainable_topology = _MODULE.trainable_topology


class _Adapter:
    lora_targets = ["to_q", "to_out.0"]

    def __init__(self, names: list[str]):
        self._named_parameters = [
            (name, torch.nn.Parameter(torch.zeros(1))) for name in names
        ]

    def named_parameters(self):
        return list(self._named_parameters)


def test_topology_accepts_only_direct_paired_standard_peft_modules():
    topology = trainable_topology(
        _Adapter(
            [
                "transformer.blocks.0.attn.to_q.lora_A.default.weight",
                "transformer.blocks.0.attn.to_q.lora_B.default.weight",
                "transformer.blocks.0.attn.to_out.0.lora_A.default.weight",
                "transformer.blocks.0.attn.to_out.0.lora_B.default.weight",
            ]
        )
    )

    assert topology["effective_lora_target_families"] == ["to_out.0", "to_q"]
    assert topology["effective_lora_module_counts"] == {"to_out.0": 1, "to_q": 1}
    assert topology["unclassified_trainable_parameter_names"] == []


def test_topology_rejects_non_lora_ancestor_and_one_sided_false_positives():
    names = [
        "transformer.blocks.0.attn.to_q.base_layer.weight",
        "transformer.blocks.0.attn.to_q.wrapper.lora_A.default.weight",
        "transformer.blocks.0.attn.to_q.wrapper.lora_B.default.weight",
        "transformer.blocks.0.attn.to_out.0.lora_A.default.weight",
    ]
    topology = trainable_topology(_Adapter(names))

    assert topology["effective_lora_target_families"] == []
    assert topology["effective_lora_module_counts"] == {}
    assert topology["unclassified_trainable_parameter_names"] == [names[0]]


def test_runner_configures_determinism_before_harness_touches_cuda(
    monkeypatch, tmp_path
):
    events: list[tuple[str, object]] = []

    class _Runner:
        def __init__(self, runner_config):
            events.append(
                ("runner", runner_config.runner.deterministic_runtime)
            )

    class _CudaProxy:
        def __init__(self, real_cuda):
            self._real_cuda = real_cuda

        def __getattr__(self, name):
            attribute = getattr(self._real_cuda, name)
            if not callable(attribute):
                return attribute

            def fail_on_cuda_call(*_args, **_kwargs):
                events.append(("cuda", name))
                raise RuntimeError("stop on first harness CUDA call")

            return fail_on_cuda_call

    config = SimpleNamespace(
        paths=SimpleNamespace(output_dir="", resume_from=""),
        train=SimpleNamespace(max_steps=0, save_every=0),
        runner=SimpleNamespace(
            deterministic_run_dir=False,
            deterministic_runtime=False,
            show_progress=True,
        ),
    )
    monkeypatch.setattr(visual_rl, "ExperimentRunner", _Runner)
    monkeypatch.setattr(visual_rl, "load_config", lambda _path: config)
    monkeypatch.setattr(torch, "cuda", _CudaProxy(torch.cuda))
    monkeypatch.setattr(
        "sys.argv",
        [
            "remote_wan_resume_run.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--output",
            str(tmp_path / "output"),
            "--max-steps",
            "1",
        ],
    )

    assert _MODULE.main() == 1
    assert events[0] == ("runner", True)
    assert len(events) == 2
    assert events[1][0] == "cuda"
