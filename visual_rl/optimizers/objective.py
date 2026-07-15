"""Policy-objective adapters for the unified optimizer update path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ObjectiveOutput:
    """Loss and observable metrics emitted by one policy objective."""

    loss: Any
    policy_loss: Any
    approx_kl: Any
    clipfrac: Any
    metrics: dict[str, Any]


class PolicyObjective:
    """Map a rollout batch, advantages, and new log-probs to an objective."""

    def prepare_batch(self, batch: Any, advantages: Any) -> Any:
        """Prepare full-batch constants before update microbatch slicing."""

        del advantages
        return batch

    def requires_global_batch_reduction(self) -> bool:
        """Whether distributed preparation needs one global weighted mean."""

        return False

    def global_batch_reduction(
        self,
        batch: Any,
        advantages: Any,
    ) -> tuple[Any, int]:
        """Return the local mean and count for full-batch normalization."""

        del batch, advantages
        raise RuntimeError("This objective does not use a global batch reduction")

    def apply_global_batch_reduction(
        self,
        batch: Any,
        advantages: Any,
        global_mean: Any,
    ) -> Any:
        """Apply a globally reduced mean before microbatch slicing."""

        del advantages, global_mean
        return batch

    def reduction_weight(self, batch: Any, advantages: Any) -> int:
        """Return the number of elements reduced by this objective."""

        import torch

        del advantages
        values = torch.as_tensor(batch.old_log_probs)
        if values.ndim == 0 or values.shape[0] != batch.batch_size:
            raise ValueError("old_log_probs must have a batch dimension")
        if batch.transition_mask is None:
            return values.numel()
        mask = torch.as_tensor(batch.transition_mask, dtype=torch.bool)
        if tuple(mask.shape) != tuple(values.shape):
            raise ValueError(
                "transition_mask must have the same shape as old_log_probs"
            )
        return int(mask.sum().item())

    def metric_reduction_weight(
        self,
        batch: Any,
        advantages: Any,
        metric_name: str,
    ) -> int:
        """Return the reduction weight for one objective metric."""

        del metric_name
        return self.reduction_weight(batch, advantages)

    def __call__(
        self,
        batch: Any,
        advantages: Any,
        new_log_probs: Any,
    ) -> ObjectiveOutput:
        raise NotImplementedError


class AlgorithmPolicyObjective(PolicyObjective):
    """Adapt an existing algorithm loss kernel without changing its math."""

    _REQUIRED_METRICS = ("policy_loss", "approx_kl", "clipfrac")

    def __init__(self, algorithm: Any):
        if not callable(getattr(algorithm, "compute_loss", None)):
            raise TypeError("algorithm must define compute_loss(batch, advantages, new_log_probs)")
        self.algorithm = algorithm

    def prepare_batch(self, batch: Any, advantages: Any) -> Any:
        prepare = getattr(self.algorithm, "prepare_batch", None)
        if prepare is None:
            return super().prepare_batch(batch, advantages)
        return prepare(batch, advantages)

    def requires_global_batch_reduction(self) -> bool:
        required = getattr(
            self.algorithm,
            "requires_global_batch_reduction",
            None,
        )
        if required is None:
            return False
        value = required()
        if not isinstance(value, bool):
            raise TypeError(
                "algorithm.requires_global_batch_reduction must return bool"
            )
        return value

    def global_batch_reduction(
        self,
        batch: Any,
        advantages: Any,
    ) -> tuple[Any, int]:
        reduction = getattr(self.algorithm, "global_batch_reduction", None)
        if reduction is None:
            return super().global_batch_reduction(batch, advantages)
        return reduction(batch, advantages)

    def apply_global_batch_reduction(
        self,
        batch: Any,
        advantages: Any,
        global_mean: Any,
    ) -> Any:
        apply_reduction = getattr(
            self.algorithm,
            "apply_global_batch_reduction",
            None,
        )
        if apply_reduction is None:
            return super().apply_global_batch_reduction(
                batch,
                advantages,
                global_mean,
            )
        return apply_reduction(batch, advantages, global_mean)

    def reduction_weight(self, batch: Any, advantages: Any) -> int:
        reduction_weight = getattr(self.algorithm, "reduction_weight", None)
        if reduction_weight is None:
            return super().reduction_weight(batch, advantages)
        return reduction_weight(batch, advantages)

    def metric_reduction_weight(
        self,
        batch: Any,
        advantages: Any,
        metric_name: str,
    ) -> int:
        metric_reduction_weight = getattr(
            self.algorithm,
            "metric_reduction_weight",
            None,
        )
        if metric_reduction_weight is None:
            return self.reduction_weight(batch, advantages)
        return metric_reduction_weight(batch, advantages, metric_name)

    def __call__(
        self,
        batch: Any,
        advantages: Any,
        new_log_probs: Any,
    ) -> ObjectiveOutput:
        result = self.algorithm.compute_loss(batch, advantages, new_log_probs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("algorithm.compute_loss must return (loss, metrics)")
        loss, metrics = result
        if not isinstance(metrics, dict):
            raise TypeError("algorithm.compute_loss metrics must be a dict")
        missing = [name for name in self._REQUIRED_METRICS if name not in metrics]
        if missing:
            raise ValueError(
                "algorithm.compute_loss metrics missing required fields: "
                f"{', '.join(missing)}"
            )
        return ObjectiveOutput(
            loss=loss,
            policy_loss=metrics["policy_loss"],
            approx_kl=metrics["approx_kl"],
            clipfrac=metrics["clipfrac"],
            metrics=dict(metrics),
        )
