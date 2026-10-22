"""Approved Flash evaluation stage: 300 prompts, 50 ODE steps, existing HPSv3.

No optimizer/LoRA owner is created. Per-video receipts make recovery logically
idempotent; failed physical generation/RPC attempts are not extra logical videos.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from visual_rl.algorithms.trainer.interface import IterationIdentity
from visual_rl.algorithms.trainer.collection import cpu_tree, record_attempt
from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient, jpeg_frames
from visual_rl.artifacts.terminal import atomic_write
from visual_rl.artifacts.checkpoint import RankRNGSnapshot
from visual_rl.core.protocols.flash_hpsv3 import canonical_json
from visual_rl.data.samples import T2VItem, SourceItemContext, BatchRowContext, ExplicitCollator
from visual_rl.runtime.flash140_logging import progress

EVALUATION_PROMPT_SHA256 = "96536b01ab2d091158118c4fd9f548950abce4fa5fa9a7f9ac2ebaae47776da5"


def load_evaluation_prompts(path):
    if path is None:
        raise ValueError("Flash evaluation requires launch.artifacts.evaluation_prompts")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EVALUATION_PROMPT_SHA256:
        raise ValueError("evaluation prompt artifact SHA256 differs from approved 300-prompt data")
    prompts = tuple(raw.decode("utf-8").splitlines())
    if len(prompts) != 300 or any(not p.strip() for p in prompts):
        raise ValueError("evaluation requires exactly 300 nonempty prompts")
    return prompts


class FlashEvaluationStage:
    def __init__(self, *, prompts, collection, request_factory, rollout,
                 preprocess_context, rollout_context, resource, output_dir, recipe_id):
        if len(prompts) != 300:
            raise ValueError("Flash evaluation requires 300 prompts")
        self.prompts = prompts
        self.collection = collection
        self.request_factory = replace(request_factory, num_steps=50)
        self.rollout = rollout
        self.preprocess_context = preprocess_context
        self.rollout_context = rollout_context
        self.resource = resource
        self.output_dir = output_dir
        self.recipe_id = recipe_id

    def __call__(self, committed_step):
        import torch
        # Dedicated per-video generators already isolate rollout RNG; preserve
        # all global RNG streams as well (text encoder/service implementation).
        rng = RankRNGSnapshot.capture_current(0)
        try:
            with self.collection.parameter_context(self.collection.ema):
                self._evaluate(committed_step)
        finally:
            rng.restore_current()

    def _evaluate(self, step):
        import torch
        resource = self.resource()
        if not isinstance(resource, Hpsv3VideoClient):
            raise TypeError("approved Flash evaluation requires the bound real HPSv3 client")
        root = self.output_dir / "evaluations" / f"step-{step}"
        root.mkdir(parents=True, exist_ok=True)
        weight_hash = hashlib.sha256()
        for shadow in self.collection.ema:
            weight_hash.update(shadow.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        identity = {"schema_version": 1, "recipe_id": self.recipe_id,
                    "committed_step": step, "prompt_sha256": EVALUATION_PROMPT_SHA256,
                    "ema_sha256": weight_hash.hexdigest(), "sampling_steps": 50,
                    "prompt_count": 300, "service_identity": resource.service_identity}
        manifest = root / "identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("evaluation journal belongs to different EMA/data/service")
        else:
            atomic_write(manifest, canonical_json(identity))
        progress("evaluation_start", committed_steps=step, prompts=300, sampling_steps=50,
                 warning="Baseline evaluation precedes training; 300 real videos can take many hours")
        scores = []
        for index, prompt in enumerate(self.prompts):
            sample_id = f"flash-evaluation-step-{step}-prompt-{index}"
            receipt = root / f"video-{index:03d}.json"
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if (record["sample_id"] != sample_id or record["prompt"] != prompt
                        or not isinstance(record["score"], (int, float)) or not math.isfinite(record["score"])
                        or record["scored_frame_indices"] != list(range(resource.contract.frames))
                        or len(record["frame_scores"]) != resource.contract.frames
                        or len(record["selected_frame_indices"]) != resource.contract.top_k):
                    raise ValueError("evaluation receipt identity/frame contract mismatch")
                scores.append(record["score"])
                progress("evaluation_receipt_reused", committed_steps=step, completed_prompts=index + 1, total_prompts=300)
                continue
            record_attempt(root / f"attempt-{index:03d}.json", {"sample_id": sample_id, **identity})
            progress("evaluation_video_start", committed_steps=step, prompt_index=index, total_prompts=300)
            source = SourceItemContext(source_item_id=f"evaluation-{index}", dataset_source_id="evaluation",
                                       dataset_index=index, dataset_revision=EVALUATION_PROMPT_SHA256)
            item = T2VItem(prompt=prompt, source=source)
            row = BatchRowContext(occurrence_id=sample_id, group_id=sample_id, member_id=0,
                                  phase="evaluation", optimizer_step=step, source_item_id=source.source_item_id)
            samples = ExplicitCollator().collate_samples((item,), (row,))
            iteration = IterationIdentity(step, "evaluation", "evaluation", (row.identity,), (sample_id,), (0,))
            request = self.request_factory(samples, iteration)
            with self.preprocess_context(), torch.no_grad():
                conditioning = cpu_tree(request.adapter.encode(samples))
            request = replace(request, encoded_conditioning=conditioning,
                              model_condition_identity=(row.identity,))
            media = None
            try:
                with self.rollout_context():
                    media = self.rollout.evaluate_deterministic(request, num_steps=50)
                frames = jpeg_frames(media.tensor[0], media.layout, resource.contract)
                score, metadata = resource.score_video(sample_id=sample_id, prompt=prompt, frames_base64=frames)
                record = {"sample_id": sample_id, "prompt": prompt, "score": score, **metadata}
                atomic_write(receipt, canonical_json(record))
                scores.append(score)
                progress("evaluation_video_durable", committed_steps=step, completed_prompts=index + 1,
                         total_prompts=300, reward=score)
            finally:
                del media, request, conditioning
        atomic_write(root / "summary.json", canonical_json({**identity,
            "videos_scored": len(scores), "reward_mean": sum(scores) / len(scores),
            "physical_video_attempts_started": sum(json.loads(path.read_text())["physical_attempts_started"]
                for path in root.glob("attempt-*.json")),
            "full_gpu_validation": False}))
        progress("evaluation_complete", committed_steps=step, videos_scored=len(scores),
                 reward_mean=sum(scores) / len(scores))
