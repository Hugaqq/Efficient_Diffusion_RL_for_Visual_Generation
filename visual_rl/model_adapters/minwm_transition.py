"""Transition kernels used by the minimal MinWM Wan RL adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class X0RenoiseTransition:
    """One sampled non-terminal x0 re-noising transition."""

    next_latent: Any
    log_prob: Any


@dataclass(frozen=True)
class X0RenoiseKernel:
    """Gaussian transition induced by MinWM's x0 re-noising sampler.

    Given the wrapper's flow prediction, the clean prediction and the
    non-terminal transition distribution are

    ``x0 = x_t - sigma_t * flow``

    ``x_next ~ Normal((1 - sigma_next) * x0, sigma_next**2 I)``.

    Log probabilities are always evaluated in fp32 and averaged over every
    latent dimension outside the batch axis.  The sampled/observed next latent
    is detached, while the distribution mean retains the model gradient.
    """

    eps: float = 1e-8

    def __post_init__(self) -> None:
        if isinstance(self.eps, bool):
            raise TypeError("eps must be a finite positive float")
        try:
            value = float(self.eps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("eps must be a finite positive float") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("eps must be finite and positive")
        object.__setattr__(self, "eps", value)

    def x0_from_flow(
        self,
        x_t: Any,
        flow_pred: Any,
        sigma_t: Any,
    ) -> Any:
        """Convert a MinWM flow prediction to its clean-latent prediction."""

        x_t_fp32 = self._floating_tensor("x_t", x_t)
        flow_fp32 = self._floating_tensor(
            "flow_pred",
            flow_pred,
            device=x_t_fp32.device,
        )
        if tuple(x_t_fp32.shape) != tuple(flow_fp32.shape):
            raise ValueError(
                "x_t and flow_pred must have identical shapes; "
                f"got {tuple(x_t_fp32.shape)} and {tuple(flow_fp32.shape)}"
            )
        sigma_t_fp32 = self._sigma(
            "sigma_t",
            sigma_t,
            x_t_fp32,
            require_positive=False,
        )
        return x_t_fp32 - sigma_t_fp32 * flow_fp32

    def from_observation(
        self,
        x0_pred: Any,
        sigma_next: Any,
        observed_next: Any,
    ) -> X0RenoiseTransition:
        """Score a next latent sampled by the native MinWM scheduler path.

        Production MinWM sampling draws noise in native ``BFCHW`` layout and
        model dtype, then applies ``FlowMatchScheduler.add_noise`` before the
        adapter sees the resulting ``BCFHW`` observation.  Re-drawing noise in
        this generic fp32 kernel would change both RNG element mapping and
        low-precision rounding.  This method therefore treats the native next
        latent as the observation while retaining an fp32, differentiable
        distribution mean for policy-gradient replay.
        """

        import torch

        x0_fp32 = self._floating_tensor("x0_pred", x0_pred)
        sigma_next_fp32 = self._sigma(
            "sigma_next",
            sigma_next,
            x0_fp32,
            require_positive=True,
        )
        std = torch.broadcast_to(sigma_next_fp32, x0_fp32.shape)
        mean = (1.0 - std) * x0_fp32
        observed = self._floating_tensor(
            "observed_next",
            observed_next,
            device=mean.device,
        ).detach()
        if tuple(observed.shape) != tuple(mean.shape):
            raise ValueError(
                "observed_next must have the same shape as x0_pred; "
                f"got {tuple(observed.shape)} and {tuple(mean.shape)}"
            )
        log_prob = self._log_prob_from_distribution(observed, mean, std)
        return X0RenoiseTransition(
            # Preserve the native observation dtype for the next model call.
            next_latent=observed_next.detach(),
            log_prob=log_prob,
        )

    def _log_prob_from_distribution(
        self,
        observed: Any,
        mean: Any,
        std: Any,
    ) -> Any:
        import torch

        elementwise = (
            -((observed.detach() - mean).square()) / (2.0 * (std.square() + self.eps))
            - torch.log(std + self.eps)
            - 0.5 * math.log(2.0 * math.pi)
        )
        if elementwise.ndim == 0:
            raise ValueError("transition tensors must include a batch dimension")
        if elementwise.ndim == 1:
            return elementwise
        return elementwise.flatten(start_dim=1).mean(dim=1)

    @staticmethod
    def _floating_tensor(name: str, value: Any, *, device: Any | None = None) -> Any:
        import torch

        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device)
        elif device is not None:
            value = value.to(device=device)
        if not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        return value.to(dtype=torch.float32)

    @staticmethod
    def _sigma(
        name: str,
        value: Any,
        reference: Any,
        *,
        require_positive: bool,
    ) -> Any:
        import torch

        sigma = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=reference.device,
        )
        if sigma.numel() == 0 or not bool(torch.isfinite(sigma).all().item()):
            raise ValueError(f"{name} must be finite")
        if require_positive and not bool((sigma > 0).all().item()):
            raise ValueError(f"{name} must be greater than zero")

        if (
            sigma.ndim > 0
            and sigma.ndim < reference.ndim
            and sigma.shape[0] == reference.shape[0]
            and all(int(size) == 1 for size in sigma.shape[1:])
        ):
            sigma = sigma.reshape(
                tuple(sigma.shape) + (1,) * (reference.ndim - sigma.ndim)
            )
        try:
            broadcast_shape = torch.broadcast_shapes(sigma.shape, reference.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(sigma.shape)} is not broadcastable to "
                f"latent shape {tuple(reference.shape)}"
            ) from exc
        if tuple(broadcast_shape) != tuple(reference.shape):
            raise ValueError(
                f"{name} shape {tuple(sigma.shape)} would expand the latent "
                f"shape {tuple(reference.shape)}"
            )
        return sigma


__all__ = ["X0RenoiseKernel", "X0RenoiseTransition"]
