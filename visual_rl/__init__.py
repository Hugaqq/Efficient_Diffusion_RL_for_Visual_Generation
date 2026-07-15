"""VisualRL integration package with lazy public exports."""

from importlib import import_module
from typing import Any

__version__ = "0.6.0"

_PUBLIC_EXPORTS = {
    "ArtifactManager": ("visual_rl.artifacts", "ArtifactManager"),
    "CallbackContext": ("visual_rl.callbacks", "CallbackContext"),
    "CallbackError": ("visual_rl.callbacks", "CallbackError"),
    "RunCallback": ("visual_rl.callbacks", "RunCallback"),
    "Evaluator": ("visual_rl.evaluation", "Evaluator"),
    "EvaluationContext": ("visual_rl.evaluation", "EvaluationContext"),
    "EvaluationResult": ("visual_rl.evaluation", "EvaluationResult"),
    "ExperimentRunner": ("visual_rl.runner", "ExperimentRunner"),
    "FeedbackProvider": ("visual_rl.feedback", "FeedbackProvider"),
    "ManifestBuilder": ("visual_rl.artifacts", "ManifestBuilder"),
    "ModelAdapter": ("visual_rl.model_adapters.base", "ModelAdapter"),
    "OptimizerPlugin": ("visual_rl.optimizers", "OptimizerPlugin"),
    "RewardBatch": ("visual_rl.core.types", "RewardBatch"),
    "RolloutBatch": ("visual_rl.core.types", "RolloutBatch"),
    "RolloutEngine": ("visual_rl.rollout.base", "RolloutEngine"),
    "SampleManifest": ("visual_rl.artifacts", "SampleManifest"),
    "SampleRecord": ("visual_rl.artifacts", "SampleRecord"),
    "VisualRLConfig": ("visual_rl.configs.schema", "VisualRLConfig"),
    "Experiment": ("visual_rl.experiment", "Experiment"),
    "RunResult": ("visual_rl.experiment", "RunResult"),
    "RewardExecution": ("visual_rl.experiment", "RewardExecution"),
    "Train": ("visual_rl.experiment", "Train"),
    "advantages": ("visual_rl.experiment", "advantages"),
    "load_config": ("visual_rl.configs.schema", "load_config"),
    "validate_config": ("visual_rl.configs.schema", "validate_config"),
    "models": ("visual_rl.experiment", "models"),
    "objectives": ("visual_rl.experiment", "objectives"),
    "register_algorithm": ("visual_rl.plugins", "register_algorithm"),
    "register_feedback_provider": (
        "visual_rl.plugins",
        "register_feedback_provider",
    ),
    "register_model_adapter": ("visual_rl.plugins", "register_model_adapter"),
    "register_optimizer_plugin": (
        "visual_rl.plugins",
        "register_optimizer_plugin",
    ),
    "register_reward_client": ("visual_rl.plugins", "register_reward_client"),
    "register_rollout_engine": ("visual_rl.plugins", "register_rollout_engine"),
    "rewards": ("visual_rl.experiment", "rewards"),
    "rollouts": ("visual_rl.experiment", "rollouts"),
}

__all__ = [
    "ArtifactManager",
    "CallbackContext",
    "CallbackError",
    "RunCallback",
    "Evaluator",
    "EvaluationContext",
    "EvaluationResult",
    "Experiment",
    "ExperimentRunner",
    "FeedbackProvider",
    "ManifestBuilder",
    "ModelAdapter",
    "OptimizerPlugin",
    "RewardBatch",
    "RewardExecution",
    "RunResult",
    "RolloutEngine",
    "RolloutBatch",
    "SampleManifest",
    "SampleRecord",
    "VisualRLConfig",
    "Train",
    "advantages",
    "load_config",
    "validate_config",
    "models",
    "objectives",
    "register_algorithm",
    "register_feedback_provider",
    "register_model_adapter",
    "register_optimizer_plugin",
    "register_reward_client",
    "register_rollout_engine",
    "rewards",
    "rollouts",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
