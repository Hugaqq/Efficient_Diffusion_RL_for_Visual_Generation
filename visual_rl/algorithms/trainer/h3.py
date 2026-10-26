"""Small, explicit H3 joint Flow-GRPO diagnostic algorithm.

No second service scheduler or framework: H3-specific paired records bridge the
concrete model root to VisualRL's existing canonical PPO objective. Fresh only;
Wan's controller, collection, optimizer transactions and restore are untouched.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image

from visual_rl.models.h3 import H3Request
from visual_rl.algorithms.dynamics.h3_joint import NAMES, joint_transition, validate_target_layout
from visual_rl.algorithms.rollout.h3 import sigma_schedule, joint_euler_step, export_result
from visual_rl.algorithms.optimization.objective import PolicyLossInputs, clipped_surrogate


@dataclass(frozen=True)
class H3Diagnostic:
    prompt: str = "A cinematic wide shot of snowy mountain peaks under a clear blue sky."
    width: int = 96
    height: int = 64
    frames: int = 124
    points: int = 40
    group_size: int = 2
    seed: int = 42000
    stochastic_indices: tuple = (20,)
    noise: float = 0.7
    rank: int = 4
    alpha: int = 8
    learning_rate: float = 1e-4
    clip_range: float = 0.001
    max_grad_norm: float = 1.0
    parity_atol: float = 1e-4

    def __post_init__(self):
        # Deliberately ONE predeclared experiment, not a silent capacity fallback.
        if (self.width, self.height, self.frames, self.points, self.group_size) != (96, 64, 124, 40, 2):
            raise ValueError("only the declared 96x64x124/40-point/group2 DIAGNOSTIC is supported")
        if (type(self.stochastic_indices) is not tuple or self.stochastic_indices != (20,)
                or any(type(i) is not int for i in self.stochastic_indices) or self.noise != 0.7):
            raise ValueError("diagnostic trains interior stochastic transition20 at noise0.7 only")
        if (type(self.rank) is not int or type(self.alpha) is not int
                or (self.rank, self.alpha, self.learning_rate, self.clip_range, self.max_grad_norm, self.parity_atol) != (
                    4, 8, 1e-4, 0.001, 1.0, 1e-4)):
            raise ValueError("diagnostic optimization settings must match the declared workload")
        H3Request("diagnostic", self.prompt, self.seed, self.width, self.height, self.frames, self.points)

    @classmethod
    def read(cls, path):
        values = json.loads(Path(path).read_text())
        values["stochastic_indices"] = tuple(values["stochastic_indices"])
        return cls(**values)

    def request(self, step, row):
        return H3Request(f"update-{step+1}-sample-{row}", self.prompt,
                         self.seed + step * self.group_size + row,
                         self.width, self.height, self.frames, self.points)


def cpu_values(values):
    return {name: value.detach().cpu().clone() for name, value in values.items()}


@dataclass(frozen=True)
class H3Transition:
    sample_id: str
    group_id: str
    prompt_text: str
    seed: int
    index: int
    state: dict
    action: dict
    prompt: torch.Tensor
    layout: dict
    masks: dict
    sigmas: dict
    following: dict
    sigma_max: dict
    noise: float
    old: object
    generator_after: torch.Tensor
    forward_partition: tuple = (0,)  # one entire B1 packed joint sequence


def collect_sample(adapter, request, config, group_id, event):
    """One forward partition=one complete B1 joint packed sequence, always."""
    with torch.no_grad():
        prompt = adapter.encoded_prompt
        state, layout, geometry = adapter.prepare_latents(request, prompt)
        masks = validate_target_layout(state, prompt, layout)
        grids = {"video": sigma_schedule(config.points, 12.0), "audio": sigma_schedule(config.points, 3.0)}
        maxima = {n: grids[n][1] for n in NAMES}
        generator = torch.Generator(device="cpu").manual_seed(request.seed + 1000000)
        records = []
        with adapter.policy(gradients=False):
            for index in range(config.points - 1):
                sigmas = {n: grids[n][index] for n in NAMES}
                following = {n: grids[n][index+1] for n in NAMES}
                velocity = adapter.predict(state, prompt, layout, 1-sigmas["video"], 1-sigmas["audio"])
                if index in config.stochastic_indices:
                    action, density = joint_transition(state, velocity, sigmas, following, maxima, masks,
                                                       noise=config.noise, generator=generator)
                    records.append(H3Transition(request.sample_id, group_id, request.prompt, request.seed, index,
                        cpu_values(state), cpu_values(action), prompt.detach().cpu().clone(), cpu_values(layout),
                        cpu_values(masks), cpu_values(sigmas), cpu_values(following), cpu_values(maxima),
                        config.noise, density.detached(), generator.get_state().clone()))
                    state = action
                else:
                    state = joint_euler_step(state, velocity, sigmas, following)
                event("joint_rollout_step", sample_id=request.sample_id, index=index,
                      stochastic=index in config.stochastic_indices,
                      video_sigma=float(sigmas["video"]), audio_sigma=float(sigmas["audio"]))
                del velocity
        state = cpu_values(state)
    return state, geometry, tuple(records)


def replay(adapter, record):
    """No RNG and no new action. Must be inside the policy grad/no-grad scope."""
    device = adapter.device
    state, action, masks, layout = (dict((n, x.to(device)) for n, x in values.items())
                                    for values in (record.state, record.action, record.masks, record.layout))
    prompt = record.prompt.to(device)
    validate_target_layout(state, prompt, layout)
    velocity = adapter.predict(state, prompt, layout, 1-record.sigmas["video"], 1-record.sigmas["audio"])
    _, density = joint_transition(state, velocity, record.sigmas, record.following, record.sigma_max,
                                   masks, noise=record.noise, action=action)
    return density


def aligned_advantages(records, sample_ids, rewards):
    if (len(records) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids)
            or len(rewards) != len(sample_ids) or len(sample_ids) < 2):
        raise ValueError("requires aligned, distinct group samples")
    if any(not row for row in records):
        raise ValueError("each sample requires recorded stochastic transitions")
    if len({r[0].group_id for r in records}) != 1 or len({r[0].prompt_text for r in records}) != 1:
        raise ValueError("reward normalization requires one same-prompt group")
    if len({r[0].seed for r in records}) != len(records):
        raise ValueError("group samples require distinct explicit seeds")
    if any(not row or any(r.sample_id != sid or r.group_id != row[0].group_id for r in row)
           for row, sid in zip(records, sample_ids)):
        raise ValueError("reward/trajectory sample identity mismatch")
    values = torch.tensor(rewards, dtype=torch.float32)
    if not bool(torch.isfinite(values).all()) or float(values.std(unbiased=False)) <= 1e-8:
        raise ValueError("real rewards degenerate/nonfinite; no fabricated advantages or update")
    return (values-values.mean()) / (values.std(unbiased=False)+1e-8)


def hps_preprocess(video):
    """Reward definition: select81 uniform rounded source frames, PIL BICUBIC
    resize each entire RGB frame to832x480 (aspect ratio intentionally warped).
    Original AV remains untouched. No claim of scoring all124 H3 frames.
    """
    if (video.dtype != torch.uint8 or video.device.type != "cpu" or video.ndim != 4
            or video.shape[-1] != 3 or video.shape[0] < 81):
        raise ValueError("reward adapter requires original CPU uint8 FHWC >=81 frames")
    indices = torch.linspace(0, video.shape[0]-1, 81, dtype=torch.float64).round().to(torch.long)
    frames = [torch.from_numpy(np.array(Image.fromarray(video[i].numpy()).resize(
                (832, 480), resample=Image.Resampling.BICUBIC), copy=True)) for i in indices]
    return torch.stack(frames), {"source_shape": list(video.shape), "scorer_shape": [81, 480, 832, 3],
        "source_frame_indices": indices.tolist(), "resize": "PIL_RGB_BICUBIC_full_frame_warp",
        "audio_reward": False, "scorer_top_k": 24}


class H3VideoReward:
    def __init__(self, origin, identity):
        from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient
        from visual_rl.core.protocols.flash_hpsv3 import VideoContract
        self.client = Hpsv3VideoClient(origin=origin, service_identity=identity,
            contract=VideoContract(81, 480, 832), timeout_s=300, max_response_bytes=1024*1024)
        try:
            self.client.healthcheck()
        except BaseException:
            self.client.close()
            raise

    def __call__(self, result):
        from visual_rl.algorithms.rewards.clients.hpsv3 import jpeg_frames
        frames, metadata = hps_preprocess(result.video)
        score, receipt = self.client.score_video(sample_id=result.request.sample_id, prompt=result.request.prompt,
                            frames_base64=jpeg_frames(frames, "BFHWC", self.client.contract))
        return score, {**metadata, **receipt}

    def close(self):
        self.client.close()


def trainable_digest(pairs):
    # ONLY small optimized LoRA parameters, not a weight-integrity inventory.
    digest = hashlib.sha256()
    for name, parameter in pairs:
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimize_group(adapter, records, sample_ids, rewards, config, event):
    advantages = aligned_advantages(records, sample_ids, rewards)
    total = sum(len(row) for row in records)
    adapter.assert_owner()
    parameters = [p for _, p in adapter.trainables]
    before = trainable_digest(adapter.trainables)
    before_values = {n: p.detach().cpu().clone() for n, p in adapter.trainables}
    adapter.optimizer.zero_grad(set_to_none=True)
    metrics = []
    try:
        with adapter.policy(gradients=True):
            for row, advantage in zip(records, advantages):
                for record in row:
                    event("replay_forward", sample_id=record.sample_id, transition_index=record.index)
                    current = replay(adapter, record)
                    deltas = {}
                    for name in (*NAMES, "aggregate"):
                        value = current.aggregate if name == "aggregate" else current.component[name]
                        old = record.old.aggregate if name == "aggregate" else record.old.component[name]
                        delta = float((value.detach().cpu()-old).abs().max())
                        deltas[name] = delta
                        if delta > config.parity_atol:
                            raise ValueError(f"same-policy fixed-action {name} density mismatch: {delta}")
                    if current.counts != record.old.counts:
                        raise ValueError("replay generated scalar counts changed")
                    # Existing VisualRL objective, B=T=1 per replay graph. Accumulate
                    # EXACT total transition mean, not a mean of unequal minibatches.
                    new = current.aggregate.reshape(1, 1)
                    old = record.old.aggregate.to(adapter.device).reshape(1, 1)
                    adv = advantage.to(adapter.device).reshape(1, 1)
                    inputs = PolicyLossInputs(adv, torch.ones_like(adv), torch.ones_like(adv, dtype=torch.bool), config.clip_range)
                    output = clipped_surrogate(old_log_probs=old, new_log_probs=new, inputs=inputs)
                    event("backward", sample_id=record.sample_id, transition_index=record.index)
                    (output.policy_loss / total).backward()
                    metrics.append({"sample_id": record.sample_id, "transition_index": record.index,
                        "old": float(old), "current": float(new.detach()), "delta": deltas,
                        "component_old": {n: float(record.old.component[n]) for n in NAMES},
                        "component_current": {n: float(current.component[n].detach()) for n in NAMES},
                        "scalar_counts": current.counts, "loss": float(output.policy_loss.detach()),
                        "ratio": float(torch.exp(new.detach()-old)), "clipfrac": float(output.clipfrac.detach())})
                    event("replay_backward_complete", **metrics[-1])
                    del current, new, old, inputs, output
            gradients = [(n, p.grad) for n, p in adapter.trainables]
            if any(g is None or not bool(torch.isfinite(g).all()) for _, g in gradients):
                raise ValueError("missing/nonfinite selected LoRA gradient")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True))
            if not math_is_positive(grad_norm):
                raise ValueError("zero or nonfinite LoRA gradient; no optimizer commit")
            active = [n for n, g in gradients if bool((g != 0).any())]
            post_norm = float(torch.stack([g.detach().float().norm().square() for _, g in gradients]).sum().sqrt())
            if not math_is_positive(post_norm):
                raise ValueError("invalid clipped LoRA gradient")
            # Native BF16 model, FP32 optimized storage: ordinary AdamW, no scaler,
            # one step only after all B1 graphs finished backward and safety checks.
            event("optimizer_start", gradient_norm=grad_norm)
            adapter.optimizer.step()
            adapter.lr_scheduler.step()
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise ValueError("nonfinite LoRA after optimizer; process cannot retry")
            after = trainable_digest(adapter.trainables)
            changes = {n: p.detach().cpu()-before_values[n] for n, p in adapter.trainables}
            delta_l2 = sum(float(x.double().square().sum()) for x in changes.values()) ** 0.5
            delta_max = max(float(x.abs().max()) for x in changes.values())
            if before == after or not math_is_positive(delta_l2):
                raise ValueError("optimizer did not change actual LoRA Parameters")
        adapter.assert_owner()
        return {"gradient_norm_pre_clip": grad_norm, "gradient_norm_post_clip": post_norm,
            "lora_delta_l2": delta_l2, "lora_delta_max": delta_max, "gradient_nonzero_names": active,
            "trainable_before": before, "trainable_after": after, "transitions": metrics,
            "rewards": list(rewards), "advantages": advantages.tolist(), "optimizer_steps": 1}
    finally:
        adapter.optimizer.zero_grad(set_to_none=True)


def math_is_positive(value):
    import math
    return math.isfinite(value) and value > 0


def atomic_checkpoint(adapter, output, step, config, audit, provenance):
    """Durable post-update snapshot, no pending group; resume NOT supported.
    Sufficient trainable/optimizer/LR/RNG evidence is saved, not an untested
    claim of exact restart. No base-weight save/scan or old-run conversion.
    """
    folder = Path(output) / f"step-{step}"
    temporary = folder.with_name(folder.name + ".tmp")
    if folder.exists() or temporary.exists():
        raise FileExistsError("checkpoint already exists; preserve failed attempts")
    temporary.mkdir()
    payload = {"schema_version": 1, "step": step, "diagnostic": True, "resume_supported": False,
        "config": asdict(config), "provenance": provenance, "audit": audit,
        "lora": {n: p.detach().cpu().clone() for n, p in adapter.trainables},
        "optimizer": adapter.optimizer.state_dict(), "lr": adapter.lr_scheduler.state_dict(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    with (temporary / "training.pt").open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, folder)
    descriptor = os.open(folder.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return folder


class H3GRPODiagnostic:
    """H3 algorithm owns loop/density/reward; model owns calculation/lifetime."""
    def __init__(self, adapter, reward, config, output, provenance, publish, safety):
        self.adapter, self.reward, self.config = adapter, reward, config
        self.output, self.provenance = Path(output), provenance
        self.publish, self.safety = publish, safety
        self.phase = "initialize"
        self.committed_steps = 0

    def event(self, phase, **fields):
        self.phase = phase
        self.safety()
        record = {"event": phase, "committed_steps": self.committed_steps, **fields}
        logging.info(json.dumps(record, allow_nan=False))
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def run(self, updates):
        if updates not in (1, 5):
            raise ValueError("only one or five fresh diagnostic updates authorized")
        a, c = self.adapter, self.config
        self.event("encode")
        a.encoded_prompt = a.encode(c.request(0, 0))
        self.event("prepare_policy")
        a.prepare_policy(rank=c.rank, alpha=c.alpha, learning_rate=c.learning_rate)
        for step in range(updates):
            folder = self.output / f"group-{step+1}"
            folder.mkdir()
            records, finals, ids = [], [], []
            for row in range(c.group_size):
                request = c.request(step, row)
                self.event("rollout", sample_id=request.sample_id)
                state, geometry, transitions = collect_sample(a, request, c, f"group-{step+1}", self.event)
                records.append(transitions)
                finals.append((request, state, geometry))
                ids.append(request.sample_id)
            a.park_policy()  # all no-grad collection graphs gone; same Parameters retained
            torch.save(records, folder / "trajectory.pt")
            scores, reward_receipts = [], []
            for request, state, geometry in finals:
                self.event("decode", sample_id=request.sample_id)
                result = a.decode(request, state, geometry)
                self.event("export", sample_id=request.sample_id)
                media_directory = folder / request.sample_id
                media_directory.mkdir()
                media = export_result(result, media_directory)
                self.event("reward", sample_id=request.sample_id)
                score, receipt = self.reward(result)
                scores.append(score)
                reward_receipts.append({"sample_id": request.sample_id, "reward": score,
                                         "preprocessing_and_score": receipt, "media": media})
                del result
            del finals
            self.publish(folder / "rewards.json", reward_receipts)
            self.event("replay")
            audit = optimize_group(a, records, ids, scores, c, self.event)
            # From here any failure is fatal, not an automatic retry. Only advance
            # the advertised durable counter after the atomic snapshot is present.
            self.event("checkpoint")
            checkpoint = atomic_checkpoint(a, self.output, step+1, c, audit, self.provenance)
            self.committed_steps = step+1
            self.publish(folder / "update.json", {**audit, "checkpoint": str(checkpoint),
                                                   "committed_steps": self.committed_steps})
            self.event("optimizer_commit", trainable_after=audit["trainable_after"])
            del records
        return {"committed_steps": self.committed_steps, "diagnostic": True, "normal_capacity_verified": False,
                "audio_quality_reward": False, "resume_supported": False}
