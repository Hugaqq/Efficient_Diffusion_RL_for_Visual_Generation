# Formula adaptation: Copyright 2026 Jayce-Ping (Flow-Factory).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""H3 target-only Flow-SDE and fixed-action density (FP32, batch one).

Small formula adaptation from Flow-Factory 3dd5ca4d, scheduler/minimax_h3.py
(Copyright 2026 Jayce-Ping, Apache-2.0). H3 velocity is DATA-ward, hence -v
in the sigma-time drift. No ODE/zero-noise density, CFG, prefix or padding.
The aggregate is a scalar-count weighted MEAN surrogate, not joint log density.
"""
from dataclasses import dataclass
import math

import torch

NAMES = ("video", "audio")


@dataclass(frozen=True)
class JointDensity:
    means: dict
    stds: dict
    component: dict
    counts: dict
    aggregate: torch.Tensor

    def detached(self):
        return JointDensity(*({k: v.detach().cpu().clone() for k, v in values.items()}
                              for values in (self.means, self.stds, self.component)),
                            dict(self.counts), self.aggregate.detach().cpu().clone())


def weighted_mean(component, counts):
    if set(component) != set(NAMES) or set(counts) != set(NAMES):
        raise ValueError("both video and audio densities/counts required")
    if any(type(counts[n]) is not int or counts[n] <= 0 for n in NAMES):
        raise ValueError("both modalities require positive generated scalar counts")
    if any(component[n].shape != (1,) or not bool(torch.isfinite(component[n]).all()) for n in NAMES):
        raise ValueError("density must be finite batch-one means")
    return sum(component[n] * counts[n] for n in NAMES) / sum(counts.values())


def flow_mean_std(x, velocity, sigma, following, sigma_max, noise):
    """Exact FF Flow-SDE order/sign. Endpoints excluded from TRAINABLE steps.

    Eligible training indices are interior: 0 < next_sigma < sigma < 1.
    sigma_max is explicitly retained for the FF singularity convention; the
    first sigma=1 and final next_sigma=0 are deliberately ODE-only here.
    """
    if (x.dtype != torch.float32 or velocity.dtype != torch.float32 or x.ndim != 3
            or x.shape[0] != 1 or x.shape != velocity.shape or x.device != velocity.device
            or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(velocity).all())):
        raise ValueError("H3 transition requires finite matching FP32 [1,rows,channels]")
    if (not math.isfinite(noise) or noise <= 0 or not 0 < float(following) < float(sigma) < 1
            or not 0 < float(sigma_max) < 1):
        raise ValueError("no density for zero-noise/deterministic or endpoint transitions")
    s, nxt, maximum = (torch.as_tensor(v, dtype=torch.float32, device=x.device)
                       for v in (sigma, following, sigma_max))
    dt = nxt - s
    diffusion = torch.sqrt(s / (1 - torch.where(s == 1.0, maximum, s))) * noise
    mean = x * (1 + diffusion**2 / (2 * s) * dt) + (-velocity) * (
        1 + diffusion**2 * (1 - s) / (2 * s)) * dt
    std = diffusion * torch.sqrt(-dt)
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std)) or float(std) <= 0:
        raise ValueError("invalid H3 Gaussian parameters")
    return mean, std


def joint_transition(state, velocity, sigmas, following, sigma_max, masks, *, noise,
                     generator=None, action=None):
    """Sample OR evaluate one paired transition; replay never calls RNG.

    Masks are exact scalar-shape bool tensors. False dimensions have no density
    contribution and are anchored to x_current during sampling. T2VA uses all
    true masks and validates its packed layout separately. Given actions are
    always detached in the density; gradients flow only through predicted mean.
    """
    for mapping in (state, velocity, sigmas, following, sigma_max, masks):
        if set(mapping) != set(NAMES):
            raise ValueError("joint transition requires exactly video/audio")
    if action is None and generator is None:
        raise ValueError("sampling requires an explicit CPU generator")
    if action is not None and set(action) != set(NAMES):
        raise ValueError("fixed action must contain both modalities")
    means, stds, values, counts, result = {}, {}, {}, {}, {}
    for name in NAMES:
        x, mask = state[name], masks[name]
        if mask.dtype != torch.bool or mask.shape != x.shape or mask.device != x.device:
            raise ValueError("scoring masks must match the generated scalar geometry")
        counts[name] = int(mask.sum())
        if counts[name] <= 0:
            raise ValueError("cannot omit a modality from joint RL")
        mean, std = flow_mean_std(x, velocity[name], sigmas[name], following[name], sigma_max[name], noise)
        if action is None:
            eps = torch.randn(x.shape, generator=generator, dtype=torch.float32, device="cpu").to(x.device)
            target = torch.where(mask, mean + std * eps, x).detach()
        else:
            target = action[name].detach()
            if (target.shape != x.shape or target.dtype != torch.float32 or target.device != x.device
                    or not bool(torch.isfinite(target).all()) or not torch.equal(target[~mask], x[~mask])):
                raise ValueError("fixed action geometry/anchors/nonfinite mismatch")
        log_values = -(target - mean).square() / (2 * std.square()) - torch.log(std) - 0.5 * math.log(2 * math.pi)
        values[name] = log_values[mask].mean().reshape(1)
        means[name], stds[name], result[name] = mean, std, target
    return result, JointDensity(means, stds, values, counts, weighted_mean(values, counts))


def validate_target_layout(state, prompt, layout):
    """T2VA only: all AV rows generated, text separately conditioned, no padding."""
    keys = {"video_indices", "audio_indices", "text_indices", "position_ids", "token_tags"}
    if set(layout) != keys or prompt.ndim != 3 or prompt.shape[0] != 1:
        raise ValueError("invalid target-only H3 packed conditioning")
    total = prompt.shape[1] + sum(state[n].shape[1] for n in NAMES)
    indices = []
    for name, rows, tag in (("video", state["video"].shape[1], 0),
                             ("audio", state["audio"].shape[1], 2), ("text", prompt.shape[1], 1)):
        idx = layout[name + "_indices"]
        if idx.dtype != torch.long or idx.ndim != 1 or idx.numel() != rows:
            raise ValueError("prefix/padding or incorrect component row count unsupported")
        indices.append(idx)
        if not bool((layout["token_tags"][idx] == tag).all()):
            raise ValueError("modality tags disagree with packed indices")
    if (layout["position_ids"].shape != (total, 3) or layout["token_tags"].shape != (total,)
            or not torch.equal(torch.cat(indices).sort().values, torch.arange(total, device=indices[0].device))):
        raise ValueError("packed rows must partition text/video/audio exactly")
    return {name: torch.ones_like(state[name], dtype=torch.bool) for name in NAMES}
