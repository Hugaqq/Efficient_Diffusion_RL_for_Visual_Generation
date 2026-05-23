"""Algorithm factory."""

from __future__ import annotations

from typing import Any

from visual_rl.core.registry import ALGORITHMS


def build_algorithm(config: Any):
    if not isinstance(config, dict):
        from dataclasses import asdict

        config = asdict(config)
    import visual_rl.algorithms.grpo  # noqa: F401
    import visual_rl.algorithms.flash_grpo  # noqa: F401
    import visual_rl.algorithms.tempflow_grpo  # noqa: F401

    name = config.get("name", "grpo")
    cls = ALGORITHMS.get(name)
    if hasattr(cls, "from_config"):
        return cls.from_config(config)
    return cls(**config)
