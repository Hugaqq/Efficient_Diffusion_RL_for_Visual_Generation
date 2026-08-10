"""CPU-only contracts for fail-closed Diffusers gradient checkpointing."""

from __future__ import annotations

import pytest
import torch

from visual_rl.models.implementations.common_diffusers import (
    configure_gradient_checkpointing,
    verify_gradient_checkpointing,
)
from visual_rl.models.implementations.sd3 import SD3Config


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


@pytest.mark.parametrize("requested", [True, False])
def test_common_helper_fails_closed_on_missing_or_ineffective_api(
    requested: bool,
) -> None:
    method = "enable" if requested else "disable"
    with pytest.raises(RuntimeError, match=f"does not support {method}"):
        configure_gradient_checkpointing(
            object(),
            requested,
            context="missing transformer",
        )

    transformer = _CheckpointingTransformer(changes_state=False)
    transformer._gradient_checkpointing = not requested
    with pytest.raises(RuntimeError, match="did not take effect"):
        configure_gradient_checkpointing(
            transformer,
            requested,
            context="ineffective transformer",
        )


@pytest.mark.parametrize("requested", [True, False])
def test_common_verifier_rejects_state_drift_after_configuration(
    requested: bool,
) -> None:
    transformer = _CheckpointingTransformer()
    state = configure_gradient_checkpointing(
        transformer,
        requested,
        context="wrapped transformer",
    )
    transformer._gradient_checkpointing = not requested

    with pytest.raises(RuntimeError, match="state drifted"):
        verify_gradient_checkpointing(
            transformer,
            state,
            context="wrapped transformer",
        )


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_common_helper_rejects_non_bool_requests(invalid: object) -> None:
    with pytest.raises(TypeError, match="gradient_checkpointing must be bool"):
        configure_gradient_checkpointing(
            _CheckpointingTransformer(),
            invalid,
            context="fake transformer",
        )


@pytest.mark.parametrize("requested", [True, False])
def test_sd3_config_preserves_one_canonical_bool(requested: bool) -> None:
    config = SD3Config.from_mapping(
        {
            "artifact_ref": "main",
            "gradient_checkpointing": requested,
        },
        context=None,
    )

    assert config.gradient_checkpointing is requested


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_sd3_config_rejects_non_bool_gradient_checkpointing(invalid: object) -> None:
    with pytest.raises(
        TypeError,
        match=r"gradient_checkpointing must be bool",
    ):
        SD3Config.from_mapping(
            {
                "artifact_ref": "main",
                "gradient_checkpointing": invalid,
            },
            context=None,
        )
