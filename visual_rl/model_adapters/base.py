"""Model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from visual_rl.core.types import RolloutBatch


class ModelAdapter(ABC):
    name: str
    media_type: str

    @property
    @abstractmethod
    def train_module(self) -> Any:
        """Return the ``nn.Module`` that owns trainable adapter state."""

        raise NotImplementedError

    def parameters(self):
        return [
            parameter
            for parameter in self.train_module.parameters()
            if parameter.requires_grad
        ]

    def named_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.train_module.named_parameters()
            if parameter.requires_grad
        ]

    def train(self, mode: bool = True) -> ModelAdapter:
        self.train_module.train(mode)
        return self

    def eval(self) -> ModelAdapter:
        return self.train(False)

    def state_dict(self) -> dict[str, Any]:
        return self.train_module.state_dict()

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> Any:
        return self.train_module.load_state_dict(state_dict, strict=strict)

    @abstractmethod
    def sample(self, prompts: list[str], metadata: list[dict[str, Any]], rollout_config: dict[str, Any]) -> RolloutBatch:
        raise NotImplementedError

    @abstractmethod
    def recompute_log_probs(self, batch: RolloutBatch) -> Any:
        raise NotImplementedError

    def prepare_for_sampling(self) -> None:
        """Compatibility alias for sampling code that predates ``eval``."""

        self.eval()

    def prepare_for_training(self) -> None:
        """Compatibility alias for training code that predates ``train``."""

        self.train()

    def branch_transition_count(self, rollout_config: dict[str, Any]) -> int:
        """Return the number of valid transition indices for branching."""

        return int(rollout_config.get("num_steps", 1))

    @abstractmethod
    def save_pretrained(self, output_dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        raise NotImplementedError
