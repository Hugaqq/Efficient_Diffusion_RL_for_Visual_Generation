"""CPU-only contracts for fail-closed Diffusers gradient checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from visual_rl.core.types import (
    ResolutionContext,
    RuntimeBuildContext,
    ValidationContext,
)
from visual_rl.errors import ConfigError
from visual_rl.model_adapters.diffusers_common import (
    configure_gradient_checkpointing,
    verify_gradient_checkpointing,
)
from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter


class _CheckpointingTransformer(torch.nn.Module):
    def __init__(self, *, changes_state: bool = True) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self._gradient_checkpointing = False
        self.changes_state = changes_state
        self.enable_calls = 0
        self.disable_calls = 0

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing

    def enable_gradient_checkpointing(self) -> None:
        self.enable_calls += 1
        if self.changes_state:
            self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.disable_calls += 1
        if self.changes_state:
            self._gradient_checkpointing = False


def _raw_sd3_params(*, gradient_checkpointing: object) -> dict[str, object]:
    return {
        "checkpoint": "checkpoint",
        "reference_repo": "reference",
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_target_modules": ["to_q", "to_v"],
        "gradient_checkpointing": gradient_checkpointing,
        "guidance_scale": 4.5,
        "resolution": 64,
        "max_sequence_length": 32,
    }


def _resolution_context(tmp_path: Path) -> ResolutionContext:
    config_path = (tmp_path / "config.yaml").resolve()
    return ResolutionContext(
        config_path=config_path,
        config_dir=config_path.parent,
    )


def _runtime_context() -> RuntimeBuildContext:
    return RuntimeBuildContext(
        rank=0,
        local_rank=0,
        world_size=1,
        backend=None,
        device=torch.device("cpu"),
        precision="fp32",
    )


def _validation_context(tmp_path: Path) -> ValidationContext:
    return ValidationContext(
        phase="validate",
        config_dir=tmp_path.resolve(),
        distributed_mode="single",
        world_size=1,
        backend=None,
        device="cpu",
        timeout_s=30.0,
    )


@pytest.mark.parametrize("requested", [True, False])
def test_common_helper_calls_selected_api_and_verifies_effective_state(
    requested: bool,
) -> None:
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
    assert (
        verify_gradient_checkpointing(
            transformer,
            state,
            context="fake transformer",
        )
        == state
    )


def test_common_helper_fails_closed_on_missing_or_ineffective_api() -> None:
    with pytest.raises(RuntimeError, match="does not support enable"):
        configure_gradient_checkpointing(
            object(),
            True,
            context="missing transformer",
        )

    transformer = _CheckpointingTransformer(changes_state=False)
    with pytest.raises(RuntimeError, match="did not take effect"):
        configure_gradient_checkpointing(
            transformer,
            True,
            context="ineffective transformer",
        )


@pytest.mark.parametrize("requested", [True, False])
def test_sd3_component_resolves_checks_and_constructs_one_canonical_bool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: bool,
) -> None:
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "reference").mkdir()
    resolved = SD3TempFlowAdapter.resolve_params(
        _raw_sd3_params(gradient_checkpointing=requested),
        _resolution_context(tmp_path),
    )

    assert resolved["gradient_checkpointing"] is requested
    assert (
        SD3TempFlowAdapter.check_environment(
            resolved,
            _validation_context(tmp_path),
        )
        == ()
    )

    monkeypatch.setattr(
        SD3TempFlowAdapter,
        "_load_base_pipeline",
        lambda self: None,
    )
    adapter = SD3TempFlowAdapter.from_config(resolved, _runtime_context())
    assert adapter.gradient_checkpointing is requested
    assert adapter.checkpoint == (tmp_path / "checkpoint").resolve()
    assert adapter.reference_repo == (tmp_path / "reference").resolve()


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_sd3_component_rejects_non_bool_gradient_checkpointing(
    tmp_path: Path,
    invalid: object,
) -> None:
    with pytest.raises(
        ConfigError,
        match=r"gradient_checkpointing must be bool",
    ):
        SD3TempFlowAdapter.resolve_params(
            _raw_sd3_params(gradient_checkpointing=invalid),
            _resolution_context(tmp_path),
        )


def test_adapter_checkpoint_metadata_remains_mechanical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "reference").mkdir()
    resolved = SD3TempFlowAdapter.resolve_params(
        _raw_sd3_params(gradient_checkpointing=True),
        _resolution_context(tmp_path),
    )
    monkeypatch.setattr(
        SD3TempFlowAdapter,
        "_load_base_pipeline",
        lambda self: None,
    )
    adapter = SD3TempFlowAdapter.from_config(resolved, _runtime_context())
    adapter.transformer = _CheckpointingTransformer()

    checkpoint_dir = tmp_path / "adapter"
    adapter.save_checkpoint(checkpoint_dir)
    metadata = json.loads(
        (checkpoint_dir / "adapter.json").read_text(encoding="utf-8")
    )

    assert set(metadata) == {
        "format_version",
        "state_file",
        "state_sha256",
        "parameters",
    }
    assert metadata["format_version"] == 1
    assert metadata["parameters"] == [
        {
            "name": "weight",
            "shape": [],
            "dtype": "torch.float32",
        }
    ]
    assert all("gradient_checkpointing" not in key for key in metadata)

    with torch.no_grad():
        adapter.transformer.weight.fill_(7.0)
    adapter.load_checkpoint(checkpoint_dir)
    assert adapter.transformer.weight.item() == pytest.approx(1.0)
