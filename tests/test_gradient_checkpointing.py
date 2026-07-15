"""CPU-only contracts for fail-closed Diffusers gradient checkpointing."""

from __future__ import annotations

from contextlib import nullcontext
import json
import sys
import types

import pytest
import torch

import visual_rl as vr
from visual_rl.artifacts.checkpoint import config_fingerprint
from visual_rl.configs.schema import VisualRLConfig, config_to_dict
from visual_rl.model_adapters.diffusers_common import (
    GradientCheckpointingState,
    configure_gradient_checkpointing,
    validate_gradient_checkpointing_checkpoint_metadata,
)
from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter
from visual_rl.preflight import StaticPreflightError, static_preflight


class _CheckpointingTransformer(torch.nn.Module):
    def __init__(self, *, changes_state: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self._gradient_checkpointing = False
        self.changes_state = changes_state
        self.enable_calls = 0
        self.disable_calls = 0

    @property
    def is_gradient_checkpointing(self):
        return self._gradient_checkpointing

    def enable_gradient_checkpointing(self):
        self.enable_calls += 1
        if self.changes_state:
            self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.disable_calls += 1
        if self.changes_state:
            self._gradient_checkpointing = False


class _PeftCheckpointingTransformer(torch.nn.Module):
    def __init__(self, base_model, *, effective_override: bool | None = None):
        super().__init__()
        self.base_model = base_model
        self.lora_weight = torch.nn.Parameter(torch.ones(()))
        self.effective_override = effective_override

    @property
    def is_gradient_checkpointing(self):
        if self.effective_override is not None:
            return self.effective_override
        return self.base_model.is_gradient_checkpointing


@pytest.mark.parametrize("requested", [True, False])
def test_common_helper_calls_selected_api_and_verifies_effective_state(requested):
    transformer = _CheckpointingTransformer()
    transformer._gradient_checkpointing = not requested

    state = configure_gradient_checkpointing(
        transformer,
        requested,
        context="fake transformer",
    )

    assert state.requested is requested
    assert state.effective is requested
    assert transformer.enable_calls == int(requested)
    assert transformer.disable_calls == int(not requested)


def test_common_helper_rejects_missing_api_and_false_effective_claim():
    with pytest.raises(RuntimeError, match="does not support enable"):
        configure_gradient_checkpointing(
            object(),
            True,
            context="missing transformer",
        )

    liar = _CheckpointingTransformer(changes_state=False)
    with pytest.raises(RuntimeError, match="did not take effect"):
        configure_gradient_checkpointing(
            liar,
            True,
            context="lying transformer",
        )


def _config_for_model(name: str) -> VisualRLConfig:
    config = VisualRLConfig(run_name="gradient-checkpointing-preflight")
    config.paths.output_dir = "/tmp/gradient-checkpointing-preflight"
    config.model.name = name
    return config


@pytest.mark.parametrize(
    "model_name",
    ["sd3_tempflow", "tempflow_sd3_legacy", "world_r1_wan_legacy"],
)
@pytest.mark.parametrize("requested", [True, False])
def test_static_preflight_rejects_unsupported_builtin_without_blocking_extensions(
    model_name,
    requested,
):
    supported = _config_for_model(model_name)
    supported.model.extra["gradient_checkpointing"] = requested
    static_preflight(supported)

    unsupported = _config_for_model("mock_wan")
    unsupported.model.extra["gradient_checkpointing"] = requested
    with pytest.raises(StaticPreflightError, match="does not support"):
        static_preflight(unsupported)

    unsupported.model.extra["gradient_checkpointing"] = 1
    with pytest.raises(StaticPreflightError, match="must be a bool"):
        static_preflight(unsupported)

    external = _config_for_model("research_transformer")
    external.model.extra = {
        "target": "research.adapters:ResearchAdapter",
        "version": "v1",
        "dependencies": [],
        "gradient_checkpointing": True,
    }
    static_preflight(external)


@pytest.mark.parametrize(
    ("requested", "effective"),
    [(None, None), (True, True), (False, False)],
)
def test_checkpoint_metadata_accepts_complete_matching_provenance(
    requested,
    effective,
):
    validate_gradient_checkpointing_checkpoint_metadata(
        {
            "gradient_checkpointing_requested": requested,
            "gradient_checkpointing_effective": effective,
        },
        GradientCheckpointingState(requested=requested, effective=effective),
        context="test checkpoint",
    )


def test_checkpoint_metadata_absence_preserves_legacy_compatibility():
    validate_gradient_checkpointing_checkpoint_metadata(
        {"format_version": 1},
        GradientCheckpointingState(requested=True, effective=True),
        context="legacy checkpoint",
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"gradient_checkpointing_requested": True},
        {"gradient_checkpointing_effective": True},
    ],
)
def test_checkpoint_metadata_rejects_partial_provenance(metadata):
    with pytest.raises(RuntimeError, match="must contain both"):
        validate_gradient_checkpointing_checkpoint_metadata(
            metadata,
            GradientCheckpointingState(requested=True, effective=True),
            context="partial checkpoint",
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "gradient_checkpointing_requested": 1,
            "gradient_checkpointing_effective": True,
        },
        {
            "gradient_checkpointing_requested": True,
            "gradient_checkpointing_effective": "true",
        },
    ],
)
def test_checkpoint_metadata_rejects_invalid_field_types(metadata):
    with pytest.raises(RuntimeError, match="must be bool or null"):
        validate_gradient_checkpointing_checkpoint_metadata(
            metadata,
            GradientCheckpointingState(requested=True, effective=True),
            context="invalid checkpoint",
        )


@pytest.mark.parametrize("requested", [True, False])
def test_checkpoint_metadata_rejects_explicit_request_with_null_effective_state(
    requested,
):
    with pytest.raises(RuntimeError, match="request has no effective state"):
        validate_gradient_checkpointing_checkpoint_metadata(
            {
                "gradient_checkpointing_requested": requested,
                "gradient_checkpointing_effective": None,
            },
            GradientCheckpointingState(
                requested=requested,
                effective=requested,
            ),
            context="unverified checkpoint",
        )


def test_checkpoint_metadata_rejects_runtime_mismatch():
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        validate_gradient_checkpointing_checkpoint_metadata(
            {
                "gradient_checkpointing_requested": False,
                "gradient_checkpointing_effective": False,
            },
            GradientCheckpointingState(requested=True, effective=True),
            context="mismatched checkpoint",
        )


def test_high_level_descriptors_expose_opt_in_without_default_config_drift():
    wan_default = vr.models.Wan("/models/wan").to_config()["model"]["extra"]
    sd3_default = vr.models.SD3("/models/sd3").to_config()["model"]["extra"]
    assert "gradient_checkpointing" not in wan_default
    assert "gradient_checkpointing" not in sd3_default
    assert (
        vr.models.Wan("/models/wan", gradient_checkpointing=True)
        .to_config()["model"]["extra"]["gradient_checkpointing"]
        is True
    )
    assert (
        vr.models.SD3("/models/sd3", gradient_checkpointing=False)
        .to_config()["model"]["extra"]["gradient_checkpointing"]
        is False
    )
    with pytest.raises(TypeError, match="bool or None"):
        vr.models.Wan("/models/wan", gradient_checkpointing=1)


def test_explicit_gradient_checkpointing_remains_inside_v2_resume_boundary():
    config = _config_for_model("world_r1_wan_legacy")
    absent = config_fingerprint(config_to_dict(config))
    config.model.extra["gradient_checkpointing"] = False
    disabled = config_fingerprint(config_to_dict(config))
    config.model.extra["gradient_checkpointing"] = True
    enabled = config_fingerprint(config_to_dict(config))

    assert len({absent, disabled, enabled}) == 3


def _install_fake_sd3_runtime(monkeypatch, transformer):
    class _Pipeline:
        def __init__(self):
            self.transformer = transformer
            self.vae = torch.nn.Linear(1, 1)
            self.text_encoder = torch.nn.Linear(1, 1)
            self.text_encoder_2 = torch.nn.Linear(1, 1)
            self.text_encoder_3 = torch.nn.Linear(1, 1)

        def to(self, _device):
            return self

    class _StableDiffusion3Pipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return _Pipeline()

    diffusers = types.ModuleType("diffusers")
    diffusers.StableDiffusion3Pipeline = _StableDiffusion3Pipeline
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    modules = {
        "flow_grpo": types.ModuleType("flow_grpo"),
        "flow_grpo.diffusers_patch": types.ModuleType("flow_grpo.diffusers_patch"),
    }
    function_modules = {
        "flow_grpo.diffusers_patch.sd3_pipeline_with_logprob": (
            "pipeline_with_logprob"
        ),
        "flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_perstep": (
            "pipeline_with_logprob"
        ),
        "flow_grpo.diffusers_patch.sd3_sde_with_logprob": "sde_step_with_logprob",
        "flow_grpo.diffusers_patch.train_dreambooth_lora_sd3": "encode_prompt",
    }
    for module_name, function_name in function_modules.items():
        module = types.ModuleType(module_name)
        setattr(module, function_name, lambda *_args, **_kwargs: None)
        modules[module_name] = module
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(
        "visual_rl.model_adapters.sd3.legacy_repo_path",
        lambda _root: nullcontext(),
    )


def test_sd3_load_applies_request_and_checkpoint_records_effective_state(
    tmp_path,
    monkeypatch,
):
    transformer = _CheckpointingTransformer()
    _install_fake_sd3_runtime(monkeypatch, transformer)

    adapter = SD3TempFlowAdapter(
        {
            "model_path": "/offline/fake-sd3",
            "device": "cpu",
            "dtype": "float32",
            "use_lora": False,
            "extra": {
                "defer_load": True,
                "gradient_checkpointing": True,
            },
        }
    )
    adapter.load()

    assert transformer.enable_calls == 1
    assert adapter._gradient_checkpointing_metadata() == {
        "gradient_checkpointing_requested": True,
        "gradient_checkpointing_effective": True,
    }

    adapter.save_pretrained(tmp_path)
    metadata = json.loads(
        (tmp_path / "adapter_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["gradient_checkpointing_requested"] is True
    assert metadata["gradient_checkpointing_effective"] is True

    metadata["gradient_checkpointing_requested"] = False
    metadata["gradient_checkpointing_effective"] = False
    metadata_path = tmp_path / "adapter_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        adapter.load_checkpoint(tmp_path)

    metadata.pop("gradient_checkpointing_requested")
    metadata.pop("gradient_checkpointing_effective")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    adapter.load_checkpoint(tmp_path)


def test_sd3_rejects_gradient_checkpointing_drift_after_peft_attach(monkeypatch):
    transformer = _CheckpointingTransformer()
    wrapped = _PeftCheckpointingTransformer(
        transformer,
        effective_override=False,
    )
    _install_fake_sd3_runtime(monkeypatch, transformer)
    monkeypatch.setattr(
        "visual_rl.model_adapters.sd3.apply_peft_lora",
        lambda module, **_kwargs: wrapped if module is transformer else None,
    )
    adapter = SD3TempFlowAdapter(
        {
            "model_path": "/offline/fake-sd3",
            "device": "cpu",
            "dtype": "float32",
            "use_lora": True,
            "extra": {
                "defer_load": True,
                "gradient_checkpointing": True,
            },
        }
    )

    with pytest.raises(
        RuntimeError,
        match="SD3 active transformer after PEFT attach.*state drifted",
    ):
        adapter.load()

    assert transformer.enable_calls == 1
    assert adapter.pipeline.transformer is wrapped
