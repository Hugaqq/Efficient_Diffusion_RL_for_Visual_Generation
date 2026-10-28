"""Experimental H3 dual-stream Flow-SDE numerical contract.

Separate from the established H3 CPS baseline. No claim of official Flash
mathematical equivalence, trainer integration, convergence or GPU validation.
Both streams use their own shifted schedule; equal-index transitions have a
joint element-weighted Gaussian log probability. Deterministic transitions
have no density and must never be admitted as policy samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.models.h3_joint import H3JointLayout


@dataclass(frozen=True)
class H3DualFlowSDE:
    layout: H3JointLayout
    schedule: H3ScheduleSnapshot

    def __post_init__(self):
        if not isinstance(self.layout, H3JointLayout) or not isinstance(self.schedule, H3ScheduleSnapshot):
            raise TypeError('dual Flow-SDE requires H3 layout and schedule')
        if self.schedule.num_steps < 2:
            raise ValueError('dual Flow-SDE requires at least two schedule transitions')

    @staticmethod
    def _stream(x, prediction, sigma, next_sigma, sigma_max):
        s = sigma.to(device=x.device, dtype=torch.float32)
        u = next_sigma.to(device=x.device, dtype=torch.float32)
        maximum = sigma_max.to(device=x.device, dtype=torch.float32)
        dt = u - s
        if not bool(torch.isfinite(dt)) or dt.item() >= 0 or s.item() <= 0 or maximum.item() <= 0:
            raise ValueError('stochastic transition requires positive sigma and decreasing grid')
        # Per-stream Flash reference Flow-SDE, with sigma_min=0.
        diffusion = maximum * s
        std = diffusion * torch.sqrt(-dt)
        velocity_factor = (1 + diffusion.square() * (1-s)/(2*s)) * dt
        mean = x.float() * (1 + diffusion.square()/(2*s)*dt) + prediction.float()*velocity_factor
        coefficient = 1 / (torch.sqrt(-dt)/diffusion + diffusion*torch.sqrt(-dt)*(1-s)/(2*s))
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std)) or std.item() <= 0 or not bool(torch.isfinite(coefficient)) or coefficient.item() <= 0:
            raise ValueError('invalid dual Flow-SDE statistics')
        return mean, std, coefficient

    def _parts(self, x, prediction, index):
        if not isinstance(x, torch.Tensor) or not isinstance(prediction, torch.Tensor):
            raise TypeError('state and prediction must be tensors')
        if x.shape != prediction.shape or x.device != prediction.device or not x.is_floating_point() or not prediction.is_floating_point():
            raise ValueError('state and prediction must have matching shapes/device and floating dtype')
        if x.requires_grad or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(prediction).all()):
            raise ValueError('state must be detached and inputs finite')
        states = self.layout.unpack(x)
        predictions = self.layout.unpack(prediction)
        vs, vu, aus, auu = self.schedule.transition(index)
        grids = (self.schedule.video_sigmas, self.schedule.audio_sigmas)
        return tuple(self._stream(a, b, s, u, grid[1]) for a,b,s,u,grid in
                     zip(states, predictions, (vs,aus), (vu,auu), grids))

    def deterministic_step(self, x, prediction, index):
        if x.requires_grad or x.shape != prediction.shape or x.device != prediction.device:
            raise ValueError('deterministic inputs incompatible')
        states, predictions = self.layout.unpack(x.float()), self.layout.unpack(prediction.float())
        vs, vu, aus, auu = self.schedule.transition(index)
        with torch.no_grad():
            result = self.layout.pack(*(state + (u-s).to(x.device)*pred for state,pred,s,u in
                zip(states,predictions,(vs,aus),(vu,auu))))
            if not bool(torch.isfinite(result).all()):
                raise ValueError('nonfinite deterministic transition')
            return result.to(x.dtype).detach()

    def sample(self, x, prediction, index, *, video_generator, audio_generator):
        if any(not isinstance(g, torch.Generator) or g.device != x.device for g in (video_generator,audio_generator)) or video_generator is audio_generator:
            raise ValueError('independent per-stream device generators required')
        with torch.no_grad():
            stats = self._parts(x, prediction, index)
            out = []
            for (mean,std,_), generator in zip(stats,(video_generator,audio_generator)):
                out.append(mean + std*torch.randn(mean.shape, generator=generator, device=x.device, dtype=torch.float32))
            action = self.layout.pack(*out).to(x.dtype)
            if not bool(torch.isfinite(action).all()):
                raise ValueError('quantized action is nonfinite')
            return action.detach()

    def evaluate(self, x, action, prediction, index):
        if action.shape != x.shape or action.device != x.device or action.requires_grad or not bool(torch.isfinite(action).all()):
            raise ValueError('fixed action must match state and be finite detached')
        stats = self._parts(x, prediction, index)
        components = []
        for a,(mean,std,_) in zip(self.layout.unpack(action),stats):
            element = -0.5*((a.detach().float()-mean)/std).square() - torch.log(std) - 0.5*math.log(2*math.pi)
            components.append(element.reshape(a.shape[0],-1).mean(dim=1))
        sizes = (self.layout.video_rows*self.layout.video_channels, self.layout.audio_rows*self.layout.audio_channels)
        joint = sum(score * (size/sum(sizes)) for score,size in zip(components,sizes))
        if not bool(torch.isfinite(joint).all()):
            raise ValueError('nonfinite joint density')
        return joint, tuple(components), tuple(s[2].detach() for s in stats)

    def rectified_score(self, joint, components, coefficients, world_means):
        """Keep joint score/ratio value unchanged while scaling each stream gradient.

        world_means must be externally reduced over the same selected rows and
        must remain detached. This is a proposed dual-stream extension, not a
        scalar average of video/audio correction coefficients.
        """
        if len(components)!=2 or len(coefficients)!=2 or len(world_means)!=2:
            raise ValueError('exactly two stream values required')
        sizes = (self.layout.video_rows*self.layout.video_channels, self.layout.audio_rows*self.layout.audio_channels)
        corrected = joint
        for component, coefficient, mean, size in zip(components,coefficients,world_means,sizes):
            if not isinstance(mean, torch.Tensor) or mean.requires_grad or mean.numel()!=1 or not bool(torch.isfinite(mean).all()) or mean.item() <= 0:
                raise ValueError('global stream mean must be positive, finite and detached')
            factor = coefficient.to(component.device) / mean.to(component.device)
            corrected = corrected + (size/sum(sizes))*(factor-1)*(component-component.detach())
        return corrected
