"""Canonical SD3 precision and current/reference-view contracts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

from visual_rl.core.contracts import ComputePrecision, LatentLayout
from visual_rl.models import ModelInput, ModelLatentSpec, ModelPortError
from visual_rl.models.implementations.sd3 import (
    SD3Adapter,
    SD3Conditioning,
    SD3Config,
    _SD3PromptEncoder,
)
from visual_rl.models.numerics.execution import ParameterView


class _PromptPipeline:
    def encode_prompt(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        batch_size = len(kwargs["prompt"])
        return (
            torch.zeros((batch_size, 2, 4), dtype=torch.float32),
            torch.ones((batch_size, 2, 4), dtype=torch.float32),
            torch.zeros((batch_size, 4), dtype=torch.float32),
            torch.ones((batch_size, 4), dtype=torch.float32),
        )


class _ReferenceTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_delta = torch.nn.Parameter(torch.tensor(0.25))
        self.register_buffer("base", torch.tensor(0.5))
        self.reference_depth = 0
        self.fail_reference = False
        self.hidden_states: list[torch.Tensor] = []

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        self.hidden_states.append(hidden_states.detach().clone())
        if self.fail_reference and self.reference_depth:
            raise RuntimeError("reference forward failed")
        value = self.base
        if not self.reference_depth:
            value = value + self.lora_delta
        return (torch.zeros_like(hidden_states) + value.to(hidden_states.dtype),)

    @contextmanager
    def disable_adapter(self):
        self.reference_depth += 1
        try:
            yield
        finally:
            self.reference_depth -= 1


def _canonical_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    SD3Adapter,
    _ReferenceTransformer,
    ModelInput,
    list[ParameterView],
]:
    adapter = SD3Adapter(
        SD3Config(
            artifact_ref="main",
            guidance_scale=1.0,
            gradient_checkpointing=False,
            resolution=16,
        ),
        artifact_path=tmp_path,
        precision=ComputePrecision.BF16,
        model_loader=None,
    )
    transformer = _ReferenceTransformer()
    adapter._reference_context = transformer.disable_adapter
    views: list[ParameterView] = []

    def forward_prepared(
        component_name: str,
        *args: object,
        parameter_view: object | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        assert component_name == "transformer"
        assert args == ()
        view = ParameterView.CURRENT if parameter_view is None else ParameterView(
            parameter_view
        )
        views.append(view)
        return transformer(**kwargs)

    monkeypatch.setattr(adapter, "_forward_prepared", forward_prepared)
    identity = ("batch-row-0",)
    conditioning = SD3Conditioning(
        prompt_embeds=torch.zeros((1, 2, 4), dtype=torch.bfloat16),
        pooled_prompt_embeds=torch.zeros((1, 4), dtype=torch.bfloat16),
        negative_prompt_embeds=torch.zeros((1, 2, 4), dtype=torch.bfloat16),
        negative_pooled_prompt_embeds=torch.zeros(
            (1, 4),
            dtype=torch.bfloat16,
        ),
        condition_identity=identity,
    )
    latent_spec = ModelLatentSpec(
        shape=(1, 1, 2, 2),
        layout=LatentLayout.BCHW,
        axis_semantics=("batch", "channel", "height", "width"),
        device="cpu",
        dtype=torch.float32,
        spatial_stride=(8, 8),
    )
    model_input = ModelInput(
        latents=torch.full(latent_spec.shape, 1.0001, dtype=torch.float32),
        timestep=torch.tensor([900.5], dtype=torch.float32),
        conditioning=conditioning,
        guidance=None,
        latent_spec=latent_spec,
        condition_identity=identity,
        guidance_identity=("cfg:1.0",),
    )
    return adapter, transformer, model_input, views


def test_sd3_prompt_encoder_casts_all_fields_to_requested_precision() -> None:
    encoder = _SD3PromptEncoder(_PromptPipeline(), torch.bfloat16)

    encoded = encoder.encode(("red", "blue"), 128, 4.5)

    assert len(encoded) == 4
    assert all(value.dtype is torch.bfloat16 for value in encoded)
    assert all(value.device.type == "cpu" for value in encoded)


def test_sd3_current_and_reference_views_cast_only_model_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, transformer, model_input, views = _canonical_case(tmp_path, monkeypatch)
    assert not torch.equal(
        model_input.latents.to(torch.bfloat16).float(),
        model_input.latents,
    )

    current = adapter.predict(model_input)
    reference = adapter.predict_reference(model_input)

    assert views == [ParameterView.CURRENT, ParameterView.REFERENCE]
    assert [value.dtype for value in transformer.hidden_states] == [
        torch.bfloat16,
        torch.bfloat16,
    ]
    assert current.value.dtype is torch.float32
    assert reference.value.dtype is torch.float32
    torch.testing.assert_close(current.value, torch.full_like(current.value, 0.75))
    torch.testing.assert_close(
        reference.value,
        torch.full_like(reference.value, 0.5),
    )
    assert transformer.reference_depth == 0

    current.value.sum().backward()
    assert transformer.lora_delta.grad is not None
    assert bool(torch.isfinite(transformer.lora_delta.grad))


def test_sd3_reference_context_restores_after_forward_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, transformer, model_input, _views = _canonical_case(
        tmp_path,
        monkeypatch,
    )
    transformer.fail_reference = True

    with pytest.raises(RuntimeError, match="reference forward failed"):
        adapter.predict_reference(model_input)

    assert transformer.reference_depth == 0
    transformer.fail_reference = False
    current = adapter.predict(model_input)
    torch.testing.assert_close(current.value, torch.full_like(current.value, 0.75))


def test_sd3_reference_view_requires_loaded_disable_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _transformer, model_input, _views = _canonical_case(
        tmp_path,
        monkeypatch,
    )
    adapter._reference_context = None

    with pytest.raises(ModelPortError, match="reference_context has not been loaded"):
        adapter.predict_reference(model_input)
