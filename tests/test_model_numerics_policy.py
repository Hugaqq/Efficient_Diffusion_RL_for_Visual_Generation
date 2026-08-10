"""Parameter dtype, forward autocast, and parameter-view policy tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from visual_rl.models.lifecycle.components import (
    ComponentBinding,
    ComponentRole,
    ModelComponents,
    OwnershipState,
)
from visual_rl.models.numerics.execution import ParameterView
from visual_rl.models.numerics.policy import (
    FloatingBufferPolicy,
    ForwardAutocastPolicy,
    FrozenParameterPolicy,
    ModelExecutionNumericsEvidence,
    ModelNumericsPolicyError,
    ParameterDTypePolicy,
    ParameterDTypePolicyOwner,
    ParameterViewEvidence,
    ParameterViewMode,
    require_parameter_view_evidence,
)
from visual_rl.models.state.parameters import ParameterStateManager


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[1.0, -1.0]], dtype=torch.float16)
        )
        self.frozen = torch.nn.Parameter(
            torch.tensor([2.0], dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.register_buffer(
            "running_scale",
            torch.tensor([0.5], dtype=torch.float16),
        )
        self.register_buffer("token_count", torch.tensor([7], dtype=torch.int64))


def _components() -> tuple[ModelComponents, _Policy]:
    policy = _Policy()
    components = ModelComponents(
        (
            ComponentBinding(
                "policy",
                policy,
                (ComponentRole.INFERENCE, ComponentRole.TRAINABLE),
            ),
        )
    )
    return components, policy


def test_trainable_fp32_preserves_loaded_frozen_parameters_and_buffers() -> None:
    components, module = _components()
    state = [OwnershipState.CONFIGURED]
    owner = ParameterDTypePolicyOwner(
        components,
        ownership_state=lambda: state[0],
        parameter_state=ParameterStateManager(components),
    )
    original_ids = (id(module.weight), id(module.frozen), id(module.running_scale))

    rebound = owner.apply(ParameterDTypePolicy("float32"))

    assert module.weight.dtype is torch.float32
    assert module.frozen.dtype is torch.bfloat16
    assert module.running_scale.dtype is torch.float16
    assert module.token_count.dtype is torch.int64
    assert original_ids == (
        id(module.weight),
        id(module.frozen),
        id(module.running_scale),
    )
    assert rebound is owner.parameter_state
    assert rebound.topology.entries[0].dtype == "torch.float32"
    assert owner.validate_applied() is rebound
    assert len(owner.policy_id) == 64
    assert len(owner.application_id) == 64


def test_explicit_frozen_and_buffer_casts_are_independent_and_integer_safe() -> None:
    components, module = _components()
    owner = ParameterDTypePolicyOwner(
        components,
        ownership_state=lambda: OwnershipState.LOADED,
    )
    policy = ParameterDTypePolicy(
        trainable_parameter_dtype="float32",
        frozen_parameter_policy=FrozenParameterPolicy.EXPLICIT_DTYPE,
        frozen_parameter_dtype="float16",
        floating_buffer_policy=FloatingBufferPolicy.EXPLICIT_DTYPE,
        floating_buffer_dtype="bfloat16",
    )

    owner.apply(policy)

    assert module.weight.dtype is torch.float32
    assert module.frozen.dtype is torch.float16
    assert module.running_scale.dtype is torch.bfloat16
    assert module.token_count.dtype is torch.int64
    assert (
        policy.policy_id
        == ParameterDTypePolicy(
            trainable_parameter_dtype="float32",
            frozen_parameter_policy="explicit_dtype",
            frozen_parameter_dtype="float16",
            floating_buffer_policy="explicit_dtype",
            floating_buffer_dtype="bfloat16",
        ).policy_id
    )
    with pytest.raises(FrozenInstanceError):
        policy.trainable_parameter_dtype = "float16"  # type: ignore[misc]


def test_dtype_owner_rejects_repeat_post_prepare_and_live_drift() -> None:
    components, module = _components()
    owner = ParameterDTypePolicyOwner(
        components,
        ownership_state=lambda: OwnershipState.CONFIGURED,
    )
    owner.apply(ParameterDTypePolicy("float32"))
    with pytest.raises(ModelNumericsPolicyError, match="exactly one"):
        owner.apply(ParameterDTypePolicy("float32"))

    prepared_owner = ParameterDTypePolicyOwner(
        components,
        ownership_state=lambda: OwnershipState.CONFIGURED,
    )
    prepared_owner._ownership_state = lambda: OwnershipState.PREPARED
    with pytest.raises(ModelNumericsPolicyError, match="before prepare"):
        prepared_owner.apply(ParameterDTypePolicy("float32"))

    drift_components, drift_module = _components()
    drift_owner = ParameterDTypePolicyOwner(
        drift_components,
        ownership_state=lambda: OwnershipState.CONFIGURED,
    )
    drift_module.frozen.data = drift_module.frozen.data.float()
    with pytest.raises(ModelNumericsPolicyError, match="dtype drifted"):
        drift_owner.apply(ParameterDTypePolicy("float32"))

    module.running_scale.data = module.running_scale.data.float()
    with pytest.raises(ModelNumericsPolicyError, match="dtype drifted"):
        owner.validate_applied()


def test_prepared_dtype_validation_allows_only_buffer_object_rebinding() -> None:
    components, module = _components()
    state = [OwnershipState.CONFIGURED]
    owner = ParameterDTypePolicyOwner(
        components,
        ownership_state=lambda: state[0],
    )
    rebound = owner.apply(ParameterDTypePolicy("float32"))
    parameter_ids = tuple(id(parameter) for parameter in module.parameters())
    buffer_ids = tuple(id(buffer) for buffer in module.buffers())

    # Device placement uses Module._apply: Parameters retain optimizer identity,
    # while registered buffers are rebound to newly allocated tensor objects.
    module._apply(lambda tensor: tensor.clone())
    assert tuple(id(parameter) for parameter in module.parameters()) == parameter_ids
    assert tuple(id(buffer) for buffer in module.buffers()) != buffer_ids

    with pytest.raises(ModelNumericsPolicyError, match="dtype drifted"):
        owner.validate_applied()
    state[0] = OwnershipState.PREPARED
    assert owner.validate_applied() is rebound

    module._apply(lambda tensor: tensor.clone())
    with pytest.raises(ModelNumericsPolicyError, match="dtype drifted"):
        owner.validate_applied()


def test_first_prepared_validation_still_rejects_parameter_rebinding() -> None:
    components, module = _components()
    state = [OwnershipState.CONFIGURED]
    owner = ParameterDTypePolicyOwner(
        components,
        ownership_state=lambda: state[0],
    )
    owner.apply(ParameterDTypePolicy("float32"))

    module.weight = torch.nn.Parameter(module.weight.detach().clone())
    state[0] = OwnershipState.PREPARED
    with pytest.raises(ModelNumericsPolicyError, match="dtype drifted"):
        owner.validate_applied()


def test_cpu_forward_autocast_is_scoped_and_restores_after_failure() -> None:
    policy = ForwardAutocastPolicy(
        stage="train",
        parameter_view=ParameterView.CURRENT,
        device_type="cpu",
        compute_dtype="bfloat16",
        enabled=True,
    )
    left = torch.randn(8, 8)
    right = torch.randn(8, 8)
    assert not torch.is_autocast_enabled("cpu")

    result = policy.run_forward(torch.matmul, left, right)

    assert result.dtype is torch.bfloat16
    assert not torch.is_autocast_enabled("cpu")
    with pytest.raises(
        RuntimeError,
        match="forward failed",
    ), policy.forward_context():
        assert torch.is_autocast_enabled("cpu")
        raise RuntimeError("forward failed")
    assert not torch.is_autocast_enabled("cpu")
    assert (
        policy.policy_id
        == ForwardAutocastPolicy(
            stage="train",
            parameter_view="current",  # type: ignore[arg-type]
            device_type="cpu",
            compute_dtype="bfloat16",
            enabled=True,
        ).policy_id
    )

    disabled = ForwardAutocastPolicy(
        stage="rollout",
        parameter_view=ParameterView.CURRENT,
        device_type="cpu",
        compute_dtype="float32",
        enabled=False,
    )
    assert disabled.run_forward(torch.matmul, left, right).dtype is torch.float32
    assert not torch.is_autocast_enabled("cpu")


def test_autocast_support_matrix_fails_closed() -> None:
    with pytest.raises(ModelNumericsPolicyError, match="only supports"):
        ForwardAutocastPolicy(
            stage="train",
            parameter_view=ParameterView.CURRENT,
            device_type="cpu",
            compute_dtype="float16",
            enabled=True,
        )
    with pytest.raises(ModelNumericsPolicyError, match="device_type"):
        ForwardAutocastPolicy(
            stage="train",
            parameter_view=ParameterView.CURRENT,
            device_type="mps",
            compute_dtype="float16",
            enabled=True,
        )
    if not torch.cuda.is_available():
        cuda_policy = ForwardAutocastPolicy(
            stage="train",
            parameter_view=ParameterView.REFERENCE,
            device_type="cuda",
            compute_dtype="float16",
            enabled=True,
        )
        with pytest.raises(
            ModelNumericsPolicyError,
            match="CUDA is unavailable",
        ), cuda_policy.forward_context():
            pass


def test_parameter_view_evidence_identity_roundtrip_and_tamper_rejection() -> None:
    projection_id = "a" * 64
    current = ParameterViewEvidence(
        parameter_view=ParameterView.CURRENT,
        mode=ParameterViewMode.CURRENT,
        owner_component_names=("policy",),
        restorable_state_names=("policy.weight",),
        source_projection_id=projection_id,
        mutates_parameters_in_place=False,
    )
    reference = ParameterViewEvidence(
        parameter_view=ParameterView.REFERENCE,
        mode=ParameterViewMode.LORA_DISABLE,
        owner_component_names=("policy",),
        restorable_state_names=("policy.lora_A", "policy.lora_B"),
        source_projection_id=projection_id,
        mutates_parameters_in_place=False,
    )
    ema = ParameterViewEvidence(
        parameter_view=ParameterView.EMA,
        mode=ParameterViewMode.IN_PLACE_SWAP,
        owner_component_names=("policy",),
        restorable_state_names=("policy.ema_weight",),
        source_projection_id=projection_id,
        mutates_parameters_in_place=True,
    )

    assert len({current.evidence_id, reference.evidence_id, ema.evidence_id}) == 3
    assert ParameterViewEvidence.from_payload(reference.to_payload()) == reference
    assert (
        require_parameter_view_evidence(
            reference,
            parameter_view=ParameterView.REFERENCE,
            source_projection_id=projection_id,
            owner_component_names=("policy",),
        )
        is reference
    )
    with pytest.raises(TypeError, match="bool"):
        require_parameter_view_evidence(
            True,
            parameter_view=ParameterView.REFERENCE,
            source_projection_id=projection_id,
        )

    tampered = reference.to_payload()
    tampered["restorable_state_names"] = ["policy.other"]
    with pytest.raises(ModelNumericsPolicyError, match="identity mismatch"):
        ParameterViewEvidence.from_payload(tampered)
    with pytest.raises(ModelNumericsPolicyError, match="disagrees"):
        ParameterViewEvidence(
            parameter_view=ParameterView.REFERENCE,
            mode=ParameterViewMode.IN_PLACE_SWAP,
            owner_component_names=("policy",),
            restorable_state_names=("policy.weight",),
            source_projection_id=projection_id,
            mutates_parameters_in_place=False,
        )


def test_execution_numerics_evidence_roundtrip_and_tamper_rejection() -> None:
    projection_id = "b" * 64
    current = ParameterViewEvidence(
        parameter_view=ParameterView.CURRENT,
        mode=ParameterViewMode.CURRENT,
        owner_component_names=("policy",),
        restorable_state_names=("policy.weight",),
        source_projection_id=projection_id,
        mutates_parameters_in_place=False,
    )
    evidence = ModelExecutionNumericsEvidence(
        parameter_dtype_policy=ParameterDTypePolicy("float32"),
        forward_autocast_policies=(
            ForwardAutocastPolicy(
                stage="train",
                parameter_view=ParameterView.CURRENT,
                device_type="cpu",
                compute_dtype="bfloat16",
                enabled=True,
            ),
        ),
        parameter_view_evidence=(current,),
    )

    assert ModelExecutionNumericsEvidence.from_payload(evidence.to_payload()) == (
        evidence
    )
    assert evidence.autocast_policy(
        "train",  # type: ignore[arg-type]
        ParameterView.CURRENT,
    ).enabled
    assert evidence.view_evidence(ParameterView.CURRENT) is current

    tampered = evidence.to_payload()
    tampered["execution_numerics_id"] = "0" * 64
    with pytest.raises(ModelNumericsPolicyError, match="identity mismatch"):
        ModelExecutionNumericsEvidence.from_payload(tampered)
