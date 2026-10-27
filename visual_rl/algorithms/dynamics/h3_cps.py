"""Joint H3 CPS proxy scores, NOT Gaussian likelihoods or exact KL.

Independently implements the selected math from UniRL
8cfbac17765004afb68fef08a4e011a4c8daf555 (Apache-2.0):
sde/kernels.py CPSSDEStrategy and minimax_h3/diffusion.py dual streams.
No eta-zero ODE path, sign conversion, conditioning, or old-policy anchor here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, ClassVar

from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.models.h3_joint import H3JointLayout

if TYPE_CHECKING:
    import torch


def _finite(name: str, value: torch.Tensor) -> None:
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _state(name: str, value: torch.Tensor, layout: H3JointLayout) -> None:
    import torch

    layout.unpack(value)  # Authoritative joint shape validation.
    if value.dtype != torch.float16:
        raise ValueError(f"{name} must use FP16 storage")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} must be detached")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError(f"{name} must be on CPU or CUDA")
    _finite(name, value)


@dataclass(frozen=True)
class H3CPSStats:
    """Ephemeral FP32 diagnostics; mean retains prediction's gradient."""

    mean: torch.Tensor
    video_q: torch.Tensor
    audio_q: torch.Tensor
    video_dt: torch.Tensor
    audio_dt: torch.Tensor


@dataclass(frozen=True)
class H3CPSDynamics:
    """Immutable numerical contract; prediction is already noise-ward.

    States/actions are FP16. Predictions may be FP16/BF16/FP32/FP64;
    all transition and score arithmetic is FP32. No global RNG is used.
    """

    layout: H3JointLayout
    schedule: H3ScheduleSnapshot
    eta: float = 0.7

    def __post_init__(self) -> None:
        if not isinstance(self.layout, H3JointLayout) or not isinstance(self.schedule, H3ScheduleSnapshot):
            raise TypeError("CPS requires H3 joint layout and exact schedule")
        if type(self.eta) not in (int, float) or not math.isfinite(self.eta) or not 0 < self.eta <= 1:
            raise ValueError("CPS eta must be finite and 0 < eta <= 1")

    def stats(self, x: torch.Tensor, prediction: torch.Tensor, index: int) -> H3CPSStats:
        """Single mathematical implementation used by sample and evaluate."""
        import torch

        sigmas = self.schedule.transition(index)
        _state("x", x, self.layout)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("prediction must be a tensor")
        if prediction.shape != x.shape or prediction.device != x.device:
            raise ValueError("prediction shape/device must match x exactly")
        if prediction.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("prediction must use a supported floating dtype")
        prediction = prediction.float()
        _finite("prediction in FP32", prediction)
        states = self.layout.unpack(x.float())
        predictions = self.layout.unpack(prediction)
        means, qs, dts = [], [], []
        for state, pred, s, u in zip(states, predictions, sigmas[::2], sigmas[1::2]):
            s, u = s.to(x.device), u.to(x.device)
            q = u * math.sin(self.eta * math.pi / 2)
            if index != self.schedule.num_steps - 1 and q.item() == 0:
                raise ValueError("nonterminal CPS q underflowed to zero")
            radicand = u * u - q * q
            if not bool(torch.isfinite(radicand)) or radicand.item() < 0:
                raise ValueError("invalid CPS noise coefficient")
            x0 = state - s * pred
            e_hat = state + (1 - s) * pred
            mean = (1 - u) * x0 + torch.sqrt(radicand) * e_hat
            _finite("CPS mean", mean)
            means.append(mean)
            qs.append(q)
            dts.append(u - s)
        return H3CPSStats(self.layout.pack(*means), *qs, *dts)

    def _score(self, action: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        # A joint element mean is exactly (Dv*score_v + Da*score_a)/(Dv+Da).
        # There is deliberately no q divisor or log(q), including at q=0.
        score = -(action.detach().float() - mean).square().mean(dim=1)
        _finite("CPS proxy score", score)
        return score

    def sample(
        self, x: torch.Tensor, prediction: torch.Tensor, index: int, *,
        video_generator: torch.Generator, audio_generator: torch.Generator,
    ) -> H3CPSRecord:
        """Draw video then audio FP32 noise, even at terminal q=0.

        Returns owned detached storage and a rollout proxy-score diagnostic,
        NOT a frozen replay old-policy anchor.
        """
        import torch

        with torch.no_grad():
            stats = self.stats(x, prediction, index)
            for generator in (video_generator, audio_generator):
                if not isinstance(generator, torch.Generator):
                    raise TypeError("explicit video/audio torch.Generator required")
                if generator.device != x.device:
                    raise ValueError("generator device must match x")
            video_mean, audio_mean = self.layout.unpack(stats.mean)
            video_noise = torch.randn(video_mean.shape, dtype=torch.float32, device=x.device, generator=video_generator)
            audio_noise = torch.randn(audio_mean.shape, dtype=torch.float32, device=x.device, generator=audio_generator)
            action = self.layout.pack(
                video_mean + stats.video_q * video_noise,
                audio_mean + stats.audio_q * audio_noise,
            ).to(torch.float16)
            _finite("FP16 action (quantization overflow)", action)
            score = self._score(action, stats.mean)
            return H3CPSRecord(self, index, x, action, score)

    def evaluate(self, record: H3CPSRecord, prediction: torch.Tensor) -> torch.Tensor:
        """Fixed-action FP32 score [B]; only current prediction carries gradients."""
        if not isinstance(record, H3CPSRecord):
            raise TypeError("evaluate requires H3CPSRecord")
        if record.dynamics != self:
            raise ValueError("record layout/schedule/eta contract mismatch")
        stats = self.stats(record._x, prediction, record.index)
        return self._score(record._action, stats.mean)


@dataclass(frozen=True, init=False, eq=False)
class H3CPSRecord:
    """Owned fixed action/state plus immutable numerical contract, no graph.

    Public tensor properties return copies. Private buffers are internal only.
    The dynamics value contains only layout/schedule/eta, no model or RNG.
    """

    dynamics: H3CPSDynamics
    index: int
    _x: torch.Tensor = field(repr=False)
    _action: torch.Tensor = field(repr=False)
    _rollout_score: torch.Tensor = field(repr=False)
    score_family: ClassVar[str] = "cps_proxy_negative_mse"
    quantization: ClassVar[str] = "fp16"

    def __init__(
        self, dynamics: H3CPSDynamics, index: int, x: torch.Tensor,
        action: torch.Tensor, rollout_score: torch.Tensor,
    ) -> None:
        import torch

        if not isinstance(dynamics, H3CPSDynamics):
            raise TypeError("record requires H3CPSDynamics")
        dynamics.schedule.transition(index)
        _state("x", x, dynamics.layout)
        _state("action", action, dynamics.layout)
        if action.shape != x.shape or action.device != x.device:
            raise ValueError("action shape/device must match x")
        if not isinstance(rollout_score, torch.Tensor):
            raise TypeError("rollout_score must be a tensor")
        if rollout_score.shape != (x.shape[0],) or rollout_score.dtype != torch.float32 or rollout_score.device != x.device:
            raise ValueError("rollout_score must be FP32 [B] on x device")
        _finite("rollout_score", rollout_score)
        object.__setattr__(self, "dynamics", dynamics)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "_x", x.detach().clone())
        object.__setattr__(self, "_action", action.detach().clone())
        object.__setattr__(self, "_rollout_score", rollout_score.detach().clone())

    @property
    def x(self) -> torch.Tensor:
        return self._x.clone()

    @property
    def action(self) -> torch.Tensor:
        return self._action.clone()

    @property
    def rollout_score(self) -> torch.Tensor:
        return self._rollout_score.clone()
