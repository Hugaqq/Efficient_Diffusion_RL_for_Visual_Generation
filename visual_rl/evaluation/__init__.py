"""Held-out and cross-run evaluation utilities."""

from visual_rl.evaluation.cross_run import aggregate_sd3_run_summaries
from visual_rl.evaluation.base import EvaluationContext, EvaluationResult, Evaluator

__all__ = [
    "EvaluationContext",
    "EvaluationResult",
    "Evaluator",
    "aggregate_sd3_run_summaries",
]
