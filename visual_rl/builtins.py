"""Explicit registration of VisualRL's built-in plugins."""

from __future__ import annotations


def register_builtin_plugins() -> None:
    """Import built-ins so direct Python API use never depends on the CLI."""

    import visual_rl.optimizers.flash_grpo  # noqa: F401
    import visual_rl.optimizers.grpo  # noqa: F401
    import visual_rl.optimizers.tempflow_grpo  # noqa: F401
    import visual_rl.model_adapters.mock  # noqa: F401
    import visual_rl.model_adapters.sd3  # noqa: F401
    import visual_rl.model_adapters.tiny_diffusion  # noqa: F401
    import visual_rl.model_adapters.wan  # noqa: F401
    import visual_rl.feedback.clients  # noqa: F401
    import visual_rl.feedback.image_rewards  # noqa: F401
    import visual_rl.feedback.world_r1_rewards  # noqa: F401
    import visual_rl.rollout.branching  # noqa: F401
    import visual_rl.rollout.full_trajectory  # noqa: F401
    import visual_rl.rollout.single_step  # noqa: F401
