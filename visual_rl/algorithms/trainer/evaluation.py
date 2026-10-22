"""Approved EMA ODE evaluation with durable per-video receipts.

SMALL40 uses fixed prompt noise, saves every MP4 before its score receipt, and
runs at terminal40 before SUCCESS. Original140 remains 300x50 without MP4s.
"""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from visual_rl.algorithms.trainer.interface import IterationIdentity
from visual_rl.algorithms.trainer.collection import cpu_tree, record_attempt
from visual_rl.algorithms.trainer.flash_presets import APPROVED140, SMALL40
from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient, jpeg_frames
from visual_rl.artifacts.terminal import atomic_write
from visual_rl.artifacts.checkpoint import RankRNGSnapshot
from visual_rl.core.protocols.flash_hpsv3 import canonical_json
from visual_rl.data.samples import T2VItem, SourceItemContext, BatchRowContext, ExplicitCollator
from visual_rl.runtime.flash140_logging import progress

EVALUATION_PROMPT_SHA256 = "96536b01ab2d091158118c4fd9f548950abce4fa5fa9a7f9ac2ebaae47776da5"
SELECTION_DIR = Path(__file__).resolve().parents[3] / "data/flash_small40"


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


def select_small40_prompts(full_prompts, train_path):
    """Validate full author data first; select committed indices, never rewards."""
    manifest = json.loads((SELECTION_DIR / "evaluation-selection.json").read_text())
    raw = Path(train_path).read_bytes()
    if (manifest["source_sha256"] != EVALUATION_PROMPT_SHA256
            or hashlib.sha256(raw).hexdigest() != manifest["train_sha256"]
            or len(raw.decode().splitlines()) != 19700 or len(full_prompts) != 300):
        raise ValueError("SMALL40 author source hash/count mismatch")
    entries = manifest["entries"]
    prompts = tuple(e["prompt"] for e in entries)
    if (len(entries) != 16 or len(set(prompts)) != 16
            or tuple(e["seed"] for e in entries) != SMALL40.noise_seeds
            or len({e["index"] for e in entries}) != 16
            or any(type(e["index"]) is not int or not 0 <= e["index"] < 300
                   or full_prompts[e["index"]] != e["prompt"] for e in entries)
            or tuple((SELECTION_DIR / "evaluation-16.txt").read_text().splitlines()) != prompts):
        raise ValueError("SMALL40 committed selection/text/seed mismatch")
    categories = ("people/action", "animals", "nature", "complex scenes")
    if any(sum(e["category"] == c for e in entries) != 4 for c in categories):
        raise ValueError("SMALL40 requires four prompts per frozen category")
    train = {p.strip() for p in raw.decode().splitlines()}
    if any(p.strip() in train for p in prompts):
        raise ValueError("SMALL40 evaluation overlaps training")
    return prompts


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_evaluation_video(path, media, contract):
    """Atomic 15fps H264 MP4, exact decoded geometry (81, NOT 21 latent frames).

    Stream one frame at a time; no second full float/uint8 video copy. CPU unit
    tests mock this encoder leaf only, not the evaluation/receipt lifecycle.
    """
    import torch
    import imageio.v2 as imageio
    video = media.tensor[0]
    shape = ((contract.frames, 3, contract.height, contract.width) if media.layout == "BFCHW"
             else (contract.frames, contract.height, contract.width, 3))
    if (media.layout not in ("BFCHW", "BFHWC") or tuple(video.shape) != shape
            or video.device.type != "cpu" or video.requires_grad):
        raise ValueError("evaluation MP4 decoded CPU frame contract mismatch")
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".mp4", dir=path.parent)
    os.close(descriptor)
    try:
        with imageio.get_writer(temporary, fps=15, codec="libx264", macro_block_size=1,
                                pixelformat="yuv420p") as writer:
            for frame in video:
                if frame.is_floating_point():
                    if not bool(torch.isfinite(frame).all()):
                        raise ValueError("evaluation MP4 frame must be finite")
                    frame = (frame * 255).round().clamp(0, 255).to(torch.uint8)
                elif frame.dtype != torch.uint8:
                    raise ValueError("evaluation MP4 requires float/uint8 frames")
                if media.layout == "BFCHW":
                    frame = frame.permute(1, 2, 0)
                writer.append_data(frame.contiguous().numpy())
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if path.stat().st_size == 0:
        raise ValueError("empty evaluation MP4")


class FlashEvaluationStage:
    def __init__(self, *, prompts, collection, request_factory, rollout,
                 preprocess_context, rollout_context, resource, output_dir, recipe_id,
                 preset=APPROVED140):
        if len(prompts) != preset.prompt_count:
            raise ValueError(f"Flash {preset.name} evaluation requires {preset.prompt_count} prompts")
        self.preset = preset
        self.prompts = prompts
        self.collection = collection
        self.request_factory = replace(request_factory, num_steps=preset.sampling_steps)
        self.rollout = rollout
        self.preprocess_context = preprocess_context
        self.rollout_context = rollout_context
        self.resource = resource
        self.output_dir = output_dir
        self.recipe_id = recipe_id

    def __call__(self, committed_step):
        rng = RankRNGSnapshot.capture_current(0)
        try:
            with self.collection.parameter_context(self.collection.ema):
                self._evaluate(committed_step)
        finally:
            rng.restore_current()

    def _evaluate(self, step):
        import torch
        preset = self.preset
        resource = self.resource()
        if not isinstance(resource, Hpsv3VideoClient):
            raise TypeError("approved Flash evaluation requires the bound real HPSv3 client")
        root = self.output_dir / "evaluations" / f"step-{step}"
        root.mkdir(parents=True, exist_ok=True)
        weight_hash = hashlib.sha256()
        for shadow in self.collection.ema:
            weight_hash.update(shadow.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        prompt_hash = (hashlib.sha256(canonical_json(list(self.prompts))).hexdigest()
                       if preset.save_videos else EVALUATION_PROMPT_SHA256)
        identity = {"schema_version": 2 if preset.save_videos else 1, "recipe_id": self.recipe_id,
                    "committed_step": step, "prompt_sha256": prompt_hash,
                    "ema_sha256": weight_hash.hexdigest(), "sampling_steps": preset.sampling_steps,
                    "prompt_count": preset.prompt_count, "service_identity": resource.service_identity}
        if preset.save_videos:
            identity.update(preset=preset.name, noise_seeds=list(preset.noise_seeds), fps=15,
                            decoded_frames=resource.contract.frames)
        manifest = root / "identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("evaluation journal belongs to different EMA/data/service")
        else:
            atomic_write(manifest, canonical_json(identity))
        progress("evaluation_start", committed_steps=step, prompts=preset.prompt_count,
                 sampling_steps=preset.sampling_steps, preset=preset.name,
                 warning="Baseline evaluation precedes training" if step == 0 else "EMA evaluation")
        records = []
        for index, prompt in enumerate(self.prompts):
            sample_id = f"flash-evaluation-step-{step}-prompt-{index}"
            seed = preset.noise_seeds[index] if preset.noise_seeds else None
            receipt = root / f"video-{index:03d}.json"
            video = root / f"video-{index:03d}-seed-{seed}.mp4"
            if receipt.exists():
                record = json.loads(receipt.read_text())
                self._validate_record(record, sample_id, prompt, resource.contract)
                if preset.save_videos:
                    if (record.get("seed") != seed or record.get("video_path") != video.name
                            or record.get("fps") != 15 or not video.is_file() or video.stat().st_size == 0
                            or record.get("video_sha256") != file_sha256(video)):
                        raise ValueError("evaluation receipt video missing/corrupt or seed mismatch")
                records.append(record)
                progress("evaluation_receipt_reused", committed_steps=step, completed_prompts=index + 1,
                         total_prompts=preset.prompt_count)
                continue
            record_attempt(root / f"attempt-{index:03d}.json", {"sample_id": sample_id, **identity})
            progress("evaluation_video_start", committed_steps=step, prompt_index=index,
                     total_prompts=preset.prompt_count, seed=seed)
            source = SourceItemContext(source_item_id=f"evaluation-{index}", dataset_source_id="evaluation",
                                       dataset_index=index, dataset_revision=prompt_hash)
            item = T2VItem(prompt=prompt, source=source)
            row = BatchRowContext(occurrence_id=sample_id, group_id=sample_id, member_id=0,
                                  phase="evaluation", optimizer_step=step, source_item_id=source.source_item_id)
            samples = ExplicitCollator().collate_samples((item,), (row,))
            iteration = IterationIdentity(step, "evaluation", "evaluation", (row.identity,), (sample_id,), (0,))
            request = self.request_factory(samples, iteration)
            if seed is not None:
                # Explicit per-prompt streams do NOT depend on step/sample ID.
                request = replace(request,
                    generator=torch.Generator(device=request.latent_spec.device).manual_seed(seed),
                    selection_generator=torch.Generator(device=request.latent_spec.device).manual_seed(seed))
            with self.preprocess_context(), torch.no_grad():
                conditioning = cpu_tree(request.adapter.encode(samples))
            request = replace(request, encoded_conditioning=conditioning,
                              model_condition_identity=(row.identity,))
            media = frames = None
            try:
                with self.rollout_context():
                    media = self.rollout.evaluate_deterministic(request, num_steps=preset.sampling_steps)
                if preset.save_videos:
                    write_evaluation_video(video, media, resource.contract)
                frames = jpeg_frames(media.tensor[0], media.layout, resource.contract)
                score, metadata = resource.score_video(sample_id=sample_id, prompt=prompt, frames_base64=frames)
                record = {"sample_id": sample_id, "prompt": prompt, "score": score, **metadata}
                self._validate_record(record, sample_id, prompt, resource.contract)
                if preset.save_videos:
                    record.update(seed=seed, video_path=video.name, video_sha256=file_sha256(video), fps=15)
                atomic_write(receipt, canonical_json(record))
                records.append(record)
                progress("evaluation_video_durable", committed_steps=step, completed_prompts=index + 1,
                         total_prompts=preset.prompt_count, reward=score)
            finally:
                del media, frames, request, conditioning
        scores = [r["score"] for r in records]
        summary = {**identity, "videos_scored": len(scores), "reward_mean": sum(scores) / len(scores),
            "physical_video_attempts_started": sum(json.loads(path.read_text())["physical_attempts_started"]
                for path in root.glob("attempt-*.json")), "full_gpu_validation": False}
        if preset.save_videos:
            summary["per_prompt"] = records
            summary["paired_differences"] = {}
            for earlier in preset.evaluation_steps:
                if earlier >= step:
                    break
                previous = json.loads((self.output_dir / "evaluations" / f"step-{earlier}" / "summary.json").read_text())
                if any(previous.get(k) != identity[k] for k in
                       ("recipe_id", "prompt_sha256", "noise_seeds", "sampling_steps", "service_identity")):
                    raise ValueError("paired evaluation identity mismatch")
                old = previous["per_prompt"]
                if len(old) != len(records) or any(a["prompt"] != b["prompt"] or a["seed"] != b["seed"]
                                                   for a, b in zip(records, old, strict=True)):
                    raise ValueError("paired evaluation prompt/seed mismatch")
                delta = [a["score"] - b["score"] for a, b in zip(records, old, strict=True)]
                if not all(math.isfinite(d) for d in delta):
                    raise ValueError("paired evaluation differences must be finite")
                summary["paired_differences"][f"step-{earlier}"] = {"per_prompt": delta, "mean": sum(delta) / len(delta)}
            summary["interpretation"] = "Descriptive paired HPS differences only; no quality claim."
        atomic_write(root / "summary.json", canonical_json(summary))
        progress("evaluation_complete", committed_steps=step, videos_scored=len(scores),
                 reward_mean=summary["reward_mean"])

    @staticmethod
    def _validate_record(record, sample_id, prompt, contract):
        if (record["sample_id"] != sample_id or record["prompt"] != prompt
                or type(record["score"]) not in (int, float) or not math.isfinite(record["score"])
                or record["scored_frame_indices"] != list(range(contract.frames))
                or len(record["frame_scores"]) != contract.frames
                or any(type(s) not in (int, float) or not math.isfinite(s) for s in record["frame_scores"])
                or len(record["selected_frame_indices"]) != contract.top_k):
            raise ValueError("evaluation receipt identity/frame contract mismatch")


def validate_small40_evaluation_artifacts(output_dir, recipe_id):
    """Before SUCCESS (including terminal-checkpoint recovery), require all48
    finite receipts AND their hashed MP4s. A saved cursor alone is insufficient.
    """
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract
    prompts = tuple((SELECTION_DIR / "evaluation-16.txt").read_text().splitlines())
    prompt_hash = hashlib.sha256(canonical_json(list(prompts))).hexdigest()
    contract = VideoContract(81, 480, 832)
    for step in SMALL40.evaluation_steps:
        root = Path(output_dir) / "evaluations" / f"step-{step}"
        summary = json.loads((root / "summary.json").read_text())
        if (summary["recipe_id"] != recipe_id or summary["committed_step"] != step
                or summary["preset"] != "small40" or summary["prompt_sha256"] != prompt_hash
                or summary["sampling_steps"] != 20 or summary["prompt_count"] != 16
                or summary["noise_seeds"] != list(SMALL40.noise_seeds)
                or summary["videos_scored"] != 16 or summary["decoded_frames"] != 81
                or summary["fps"] != 15 or len(summary["per_prompt"]) != 16):
            raise ValueError("terminal SMALL40 evaluation summary mismatch")
        scores = []
        for index, (prompt, seed) in enumerate(zip(prompts, SMALL40.noise_seeds, strict=True)):
            record = json.loads((root / f"video-{index:03d}.json").read_text())
            FlashEvaluationStage._validate_record(record,
                f"flash-evaluation-step-{step}-prompt-{index}", prompt, contract)
            video = root / f"video-{index:03d}-seed-{seed}.mp4"
            if (record != summary["per_prompt"][index] or record.get("seed") != seed
                    or record.get("fps") != 15 or record.get("video_path") != video.name
                    or not video.is_file() or video.stat().st_size == 0
                    or record.get("video_sha256") != file_sha256(video)):
                raise ValueError("terminal evaluation receipt video missing/corrupt or seed mismatch")
            scores.append(record["score"])
        if not math.isclose(summary["reward_mean"], sum(scores) / 16, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("terminal evaluation mean mismatch")
