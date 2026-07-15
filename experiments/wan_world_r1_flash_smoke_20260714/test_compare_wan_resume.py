"""Contract checks for W5 semantic resume comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("compare_wan_resume.py")
_SPEC = importlib.util.spec_from_file_location("compare_wan_resume", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
canonical_checkpoint_metadata = _MODULE.canonical_checkpoint_metadata
canonical_metric_rows = _MODULE.canonical_metric_rows


def test_canonical_metrics_exclude_observations_but_keep_training_values():
    row = {
        "step": 1,
        "loss": 0.25,
        "reward_mean": 0.5,
        "grad_norm": 0.75,
        "step_time_s": 1.0,
        "samples_per_second": 2.0,
        "peak_gpu_memory_bytes": 3,
    }

    assert canonical_metric_rows([row]) == [
        {"step": 1, "loss": 0.25, "reward_mean": 0.5, "grad_norm": 0.75}
    ]


def test_checkpoint_metadata_ignores_only_serialization_byte_hash():
    left = {
        "format_version": 4,
        "step": 2,
        "adapter_payload_sha256": "adapter",
        "training_state_sha256": "left-bytes",
    }
    right = {**left, "training_state_sha256": "right-bytes"}

    assert canonical_checkpoint_metadata(left) == canonical_checkpoint_metadata(right)
    assert canonical_checkpoint_metadata(left)["adapter_payload_sha256"] == "adapter"
