"""Auditable decisions for evidence-gated scale-out stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCALING_DECISION_SCHEMA_VERSION = "1"


def validate_conditional_scaling(config: Any) -> None:
    """Reject unavailable scale-out modes before runtime side effects."""

    values = _scaling_values(config)
    if values["split_roles"]:
        raise ValueError(
            "runner.conditional_scaling.split_roles (C13) is not enabled: it "
            "requires measured idle-time evidence after the DDP baseline"
        )
    if values["fsdp2"]:
        raise ValueError(
            "runner.conditional_scaling.fsdp2 (C14) is not enabled: it requires "
            "measured Wan memory pressure after the DDP baseline"
        )


def build_scaling_trigger_decision(config: Any) -> dict[str, Any]:
    """Return a deterministic artifact explaining why C13/C14 remain disabled."""

    values = _scaling_values(config)
    validate_conditional_scaling(values)
    return {
        "schema_version": SCALING_DECISION_SCHEMA_VERSION,
        "policy": "evidence_gated",
        "stages": {
            "C13": {
                "name": "rollout_reward_train_role_split",
                "requested": values["split_roles"],
                "triggered": False,
                "enabled": False,
                "decision": "not_triggered",
                "runtime_validation": "not_run",
                "required_evidence": [
                    "DDP baseline profiling",
                    "measured rollout or reward idle-time bottleneck",
                    "policy_version and max_staleness validation",
                ],
                "observed_evidence": "not_provided",
                "reason": (
                    "No profiling evidence demonstrates that native DDP cannot "
                    "address the measured bottleneck."
                ),
            },
            "C14": {
                "name": "fsdp2_distributed_checkpoint",
                "requested": values["fsdp2"],
                "triggered": False,
                "enabled": False,
                "decision": "not_triggered",
                "runtime_validation": "not_run",
                "required_evidence": [
                    "Wan peak-memory profile",
                    "DDP model-replication memory bottleneck",
                    "adapter parameter-reference audit",
                ],
                "observed_evidence": "not_provided",
                "reason": (
                    "No measured single-device capacity failure or DDP replication "
                    "bottleneck justifies FSDP2."
                ),
            },
        },
    }


def _scaling_values(config: Any) -> dict[str, bool]:
    if isinstance(config, Mapping):
        source = config
    else:
        source = {
            "split_roles": getattr(config, "split_roles", None),
            "fsdp2": getattr(config, "fsdp2", None),
        }
    unknown = sorted(set(source).difference({"split_roles", "fsdp2"}))
    if unknown:
        raise ValueError(f"Unknown conditional scaling fields: {unknown}")
    values: dict[str, bool] = {}
    for name in ("split_roles", "fsdp2"):
        value = source.get(name, False)
        if not isinstance(value, bool):
            raise TypeError(f"conditional scaling {name} must be a bool")
        values[name] = value
    return values


__all__ = [
    "SCALING_DECISION_SCHEMA_VERSION",
    "build_scaling_trigger_decision",
    "validate_conditional_scaling",
]
