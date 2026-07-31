#!/usr/bin/env python3
"""Multi-GPU backend for the 3D reward (fail-closed strict manager).

world_r1_fail_closed_v1: every decode/scorer/worker failure raises from
``compute_batch_scores`` instead of being swallowed into a finite 0.0 default
score.  All Queue/Process objects come from an explicit ``spawn`` context and
every row yields exactly one tagged terminal envelope.
"""

import datetime
import hashlib
import json
import math
import multiprocessing
import os
import queue as queue_module
import threading
import time
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from services.world_r1_strict.native.reward_3d_backend import (
    DEFAULT_RECONSTRUCTION_MODEL,
    Reward3DBackend,
)
from services.world_r1_strict.process_supervision import supervised_worker_entry

STRICT_MANAGER_PROTOCOL = "world_r1_fail_closed_v1"
STRICT_REWARD_KIND = "reward_3d"
STRICT_MANAGER_TIMEOUT_S = 1800.0
STRICT_CLEANUP_TIMEOUT_S = 10.0
STRICT_FATAL_EXIT_CODE = 70
_RESULT_POLL_S = 0.05
MAX_RECONSTRUCTION_FRAMES = 16
MAX_QWEN_SCORE_FRAMES = 8
LPIPS_ALEXNET_CHECKPOINT_ENV = "WORLD_R1_LPIPS_ALEXNET_CHECKPOINT"
LPIPS_ALEXNET_CHECKPOINT_SIZE = 244_408_911
LPIPS_ALEXNET_CHECKPOINT_SHA256 = (
    "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
)


class StrictManagerError(RuntimeError):
    """Base class for fail-closed strict manager failures."""


class StrictManagerNotReadyError(StrictManagerError):
    """compute_batch_scores was called on a non-ready manager."""


class StrictManagerInitError(StrictManagerError):
    """Worker initialization failed, timed out or lost a worker."""


class StrictRewardDecodeError(StrictManagerError):
    """One request row could not be decoded into RGB frames."""


class StrictRewardComputeError(StrictManagerError):
    """A worker reported ROW_ERROR or an invalid result envelope."""


class StrictRewardTimeoutError(StrictManagerError):
    """The single strict deadline was exhausted."""


class StrictWorkerDeathError(StrictManagerError):
    """A spawn worker process died while a batch was in flight."""


def _uniform_sample_indices(length, maximum):
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("maximum must be a positive integer")
    if type(length) is not int or length <= 0:
        raise ValueError("length must be a positive integer")
    if length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [0]
    return [
        round(index * (length - 1) / (maximum - 1))
        for index in range(maximum)
    ]


def _uniform_sample(values, maximum):
    indices = _uniform_sample_indices(len(values), maximum)
    return [values[index] for index in indices]


def _resolve_lpips_alexnet_checkpoint():
    raw_path = os.environ.get(LPIPS_ALEXNET_CHECKPOINT_ENV)
    if not raw_path:
        raise StrictManagerInitError(
            f"{LPIPS_ALEXNET_CHECKPOINT_ENV} is required when LPIPS is enabled"
        )
    path = os.path.abspath(raw_path)
    if os.path.islink(path) or not os.path.isfile(path):
        raise StrictManagerInitError(
            f"{LPIPS_ALEXNET_CHECKPOINT_ENV} must be an existing non-symlink file"
        )
    if os.path.getsize(path) != LPIPS_ALEXNET_CHECKPOINT_SIZE:
        raise StrictManagerInitError("LPIPS AlexNet checkpoint size mismatch")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != LPIPS_ALEXNET_CHECKPOINT_SHA256:
        raise StrictManagerInitError("LPIPS AlexNet checkpoint digest mismatch")
    return path


def _load_offline_lpips_model(*, checkpoint_path, device, torch_module):
    try:
        import lpips as lpips_lib
    except Exception as exc:
        raise RuntimeError(f"Failed to import lpips: {exc}") from exc

    # pnet_rand=True is intentional: it prevents torchvision from downloading
    # ImageNet weights.  We then load the exact, pre-verified AlexNet trunk
    # explicitly while retaining LPIPS' packaged v0.1 calibration layers.
    model = lpips_lib.LPIPS(
        net="alex",
        pnet_rand=True,
        pretrained=True,
        verbose=False,
    )
    source = torch_module.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(source, dict):
        raise TypeError("LPIPS AlexNet checkpoint is not a state dictionary")

    trunk_state = {}
    for key, value in source.items():
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "features":
            continue
        feature_index = int(parts[1])
        if feature_index < 2:
            slice_name = "slice1"
        elif feature_index < 5:
            slice_name = "slice2"
        elif feature_index < 8:
            slice_name = "slice3"
        elif feature_index < 10:
            slice_name = "slice4"
        elif feature_index < 12:
            slice_name = "slice5"
        else:
            continue
        trunk_state[f"{slice_name}.{feature_index}.{parts[2]}"] = value

    model.net.load_state_dict(trunk_state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def _require_gsplat_cuda_extension():
    try:
        from gsplat.cuda._backend import _C
    except Exception as exc:
        raise StrictManagerInitError(
            "failed to import the gsplat CUDA extension"
        ) from exc
    if _C is None or not hasattr(_C, "CameraModelType"):
        raise StrictManagerInitError(
            "gsplat CUDA extension is unavailable; compile it before startup"
        )


def _terminate_and_reap(workers, *, timeout_s=STRICT_CLEANUP_TIMEOUT_S):
    """Bounded two-phase reaping: terminate(5s) then kill(5s), else exit(70)."""

    live = [worker for worker in workers if worker.is_alive()]
    for worker in live:
        worker.terminate()
    deadline = time.monotonic() + timeout_s / 2
    for worker in live:
        worker.join(timeout=max(deadline - time.monotonic(), 0.0))
    survivors = [worker for worker in live if worker.is_alive()]
    if survivors:
        for worker in survivors:
            worker.kill()
        deadline = time.monotonic() + timeout_s / 2
        for worker in survivors:
            worker.join(timeout=max(deadline - time.monotonic(), 0.0))
    stubborn = [worker for worker in survivors if worker.is_alive()]
    if stubborn:
        # Cleanup itself failed: the only WSGI worker must die so the
        # deployment supervisor rebuilds the whole process.
        os._exit(STRICT_FATAL_EXIT_CODE)


def _close_queue(task_queue):
    try:
        task_queue.cancel_join_thread()
    except Exception:
        pass
    try:
        task_queue.close()
    except Exception:
        pass


def reward_3d_worker_process(
    gpu_id,
    model_name,
    scorer_type,
    lpips_alexnet_checkpoint,
    task_queue,
    result_queue,
    init_queue,
):
    """
    Worker process that runs the 3D reward stack on a specific GPU.

    Args:
        gpu_id: CUDA device ID
        model_name: Reconstruction model name
        scorer_type: Fixed local scorer type (`qwen`)
        lpips_alexnet_checkpoint: Verified local AlexNet state dictionary, or None
        task_queue: Queue to receive tasks (batch_idx, video_frames, prompt, save_dir)
        result_queue: Queue to send results including per-video artifact paths
    """
    try:
        # The spawn child inherits the parent's CUDA_VISIBLE_DEVICES unchanged;
        # gpu_id is a PyTorch logical device index resolved by CUDA/PyTorch.
        import torch as torch_local
        torch_local.cuda.set_device(gpu_id)
        device = torch_local.device(f"cuda:{gpu_id}")
        _require_gsplat_cuda_extension()

        print(f"[Process {os.getpid()}] Initializing 3D reward backend on physical GPU {gpu_id}")
        reward_3d_backend = Reward3DBackend(device=device, model_name=model_name)

        # Initialize Qwen3-VL scorer
        # IMPORTANT: In multiprocessing with CUDA_VISIBLE_DEVICES, we need to use explicit device
        # instead of device_map="auto" to avoid device mismatch issues
        print(f"[Process {os.getpid()}] Initializing Qwen3-VL scorer on GPU {gpu_id}")

        # Create a custom QwenVLScorer that doesn't use device_map="auto"
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        class LocalQwenVLScorer:
            """Local QwenVL scorer without device_map for multiprocessing"""
            def __init__(self, device, dtype):
                self.device = device
                self.dtype = dtype
                model_path = os.environ.get("WORLD_R1_QWEN_MODEL")
                if not model_path:
                    raise RuntimeError("WORLD_R1_QWEN_MODEL is required")
                model_path = os.path.abspath(model_path)
                if os.path.islink(model_path) or not os.path.isdir(model_path):
                    raise RuntimeError(
                        "WORLD_R1_QWEN_MODEL must be an existing non-symlink directory"
                    )

                # Load model on specific device (NOT device_map="auto")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=True,
                ).to(device)
                self.model.requires_grad_(False)
                self.model.eval()

                if hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = False

                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    local_files_only=True,
                )

                # Temperature for logit-based scoring
                self.temperature = 1.0

                # Video evaluation prompt (from qwenvl3.py) - for GS video
                # Modified to only output a single digit for logit-based scoring
                self.video_task_template = '''You are given a text prompt: "{prompt}"
According the generated video sequence and the prompt, evaluate the video quality.

1. Analyze the video content and progression across all frames.
2. Identify key visual elements and instructions from the prompt.
3. Evaluate how well this video follows the prompt:
   - Does the camera movement align with the prompt? **If the camera is static, the score should be 0.**
   - Are all required elements present across the video?
   - Are object counts, colors, and positions accurate?
   - Is there logical temporal consistency between frames?

Provide a score from 0 to 9:
- 9: Perfect alignment with prompt and high quality
- 7-8: Very good alignment with minor issues
- 5-6: Good alignment but noticeable problems
- 3-4: Poor alignment with major issues
- 1-2: No alignment or very low quality
- 0: The camera is static, no movement.

Output only a single digit (0-9):'''

                # Image evaluation prompt (from pointvlm_v3.py) - for meta view (GS render)
                # Modified to only output a single digit for logit-based scoring
                self.image_task_template = '''You are a professional 3D vision expert. I used a text prompt to generate a video and reconstructed a corresponding 3D Pointmap from the video.
Original Prompt:
{text_prompt}

Your task is to judge the quality of the original video by analyzing the provided image of its resulting 3D pointmap. A good video (smooth, orbiting camera) creates a good pointmap. A bad video (static, jittery, or zooming) creates a bad pointmap.
Please provide a score from 0 to 9 based on these criteria:

- 9: Excellent - A dense, clean, and complete 3D model. Perfect 360° orbital motion, high stability.
- 7-8: Good - A clear object with strong 3D structure. May have minor holes or noise. Good, smooth camera arc with strong parallax.
- 4-6: Mediocre - Object is recognizable, but the map is sparse, noisy, or "flat" (lacks 3D depth). Poor parallax (e.g., just a zoom or pan instead of an orbit), or the video was jittery, blurry, or had object/lighting inconsistencies.
- 2-3: Poor - A chaotic jumble of points or a simple 2D projection. Static camera (no motion) or completely unstable.
- 0-1: Very Poor - Empty or just random noise. Unusable.

Output only a single digit (0-9):'''

            def __call__(self, prompts, videos):
                """Score videos with prompts"""
                import base64
                from io import BytesIO

                import torchvision.transforms.functional as F_vision

                def pil_image_to_base64(image):
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    return f"data:image;base64,{encoded_image_text}"

                all_rewards = []

                for prompt, video in zip(prompts, videos):
                    try:
                        # Convert tensor to PIL frames
                        if isinstance(video, torch_local.Tensor):
                            # IMPORTANT: Move to CPU first
                            video = video.cpu()
                            if video.dtype.is_floating_point:
                                video = (video.clamp(0, 1) * 255.0).to(torch_local.uint8)
                            pil_frames = [F_vision.to_pil_image(video[i]) for i in range(video.shape[0])]
                        else:
                            pil_frames = video

                        pil_frames = _uniform_sample(
                            pil_frames,
                            MAX_QWEN_SCORE_FRAMES,
                        )

                        # Choose template and message type based on input
                        if len(pil_frames) == 1:
                            # Single image (meta_view): use image prompt from pointvlm_v3.py
                            task = self.image_task_template.format(text_prompt=prompt)
                            image_base64 = pil_image_to_base64(pil_frames[0])
                            message = {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image_base64},
                                    {"type": "text", "text": task},
                                ],
                            }
                        else:
                            # Multiple frames (gs_video): use video prompt from qwenvl3.py
                            task = self.video_task_template.format(prompt=prompt)
                            video_base64 = [pil_image_to_base64(frame) for frame in pil_frames]
                            message = {
                                "role": "user",
                                "content": [
                                    {"type": "video", "video": video_base64},
                                    {"type": "text", "text": task},
                                ],
                            }

                        # Process
                        text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info([[message]])
                        batch_data = self.processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )

                        # Move inputs to device
                        input_ids = batch_data['input_ids'].to(self.device)
                        attention_mask = batch_data['attention_mask'].to(self.device)
                        pixel_values = batch_data.get('pixel_values')
                        if pixel_values is not None:
                            pixel_values = pixel_values.to(self.device)
                        pixel_values_videos = batch_data.get('pixel_values_videos')
                        if pixel_values_videos is not None:
                            pixel_values_videos = pixel_values_videos.to(self.device)
                        image_grid_thw = batch_data.get('image_grid_thw')
                        if image_grid_thw is not None:
                            image_grid_thw = image_grid_thw.to(self.device)
                        video_grid_thw = batch_data.get('video_grid_thw')
                        if video_grid_thw is not None:
                            video_grid_thw = video_grid_thw.to(self.device)

                        # Forward pass to get logits (not generation)
                        cache_position = torch_local.arange(0, input_ids.shape[1], device=self.device)

                        with torch_local.no_grad():
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=None,
                                past_key_values=None,
                                inputs_embeds=None,
                                labels=None,
                                use_cache=False,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=True,
                                pixel_values=pixel_values,
                                pixel_values_videos=pixel_values_videos,
                                image_grid_thw=image_grid_thw,
                                video_grid_thw=video_grid_thw,
                                rope_deltas=None,
                                cache_position=cache_position,
                                second_per_grid_ts=None
                            )

                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                        # Get logits at the last valid token position
                        last_valid_indices = (attention_mask != 0).cumsum(dim=1).argmax(dim=1)
                        last_token_logits = logits[torch_local.arange(logits.size(0)), last_valid_indices, :]

                        # Extract logits for tokens 15-24 (corresponding to digits 0-9)
                        # Token IDs: 15='0', 16='1', ..., 24='9'
                        digit_logits = last_token_logits[:, 15:25]  # (batch_size, 10)

                        # Compute score as weighted average using softmax
                        scores_range = torch_local.arange(0, 10, device=self.device).float()  # [0, 1, 2, ..., 9]
                        probs = (digit_logits / self.temperature).softmax(dim=-1)
                        raw_score = (probs * scores_range).sum(dim=-1).item()

                        # Normalize to 0-1 range (0-9 scale -> 0-1)
                        score = raw_score / 9.0

                        print(f"[Process {os.getpid()}] Logit-based score: {raw_score:.2f}/9 = {score:.4f}")
                        print(f"[Process {os.getpid()}] Digit probabilities: {probs[0]}")

                        all_rewards.append(score)

                        # Clear memory
                        del batch_data, input_ids, attention_mask, outputs, logits
                        if pixel_values is not None:
                            del pixel_values
                        if pixel_values_videos is not None:
                            del pixel_values_videos
                        torch_local.cuda.empty_cache()

                    except Exception as e:
                        print(f"[Process {os.getpid()}] Error scoring video: {e}")
                        import traceback
                        traceback.print_exc()
                        torch_local.cuda.empty_cache()
                        raise

                return all_rewards

        if scorer_type != "qwen":
            raise ValueError("the bundled 3D reward supports only scorer_type='qwen'")
        gs_scorer = LocalQwenVLScorer(
            device=device,
            dtype=torch_local.bfloat16,
        )
        meta_scorer = gs_scorer
        print(
            f"[Process {os.getpid()}] 3D reward backend + "
            f"Qwen3-VL ready on GPU {gpu_id}"
        )

        lpips_model = None

        def compute_lpips_gs_score(gs_video_tensor, input_video_tensor):
            nonlocal lpips_model
            if lpips_model is None:
                if not lpips_alexnet_checkpoint:
                    raise RuntimeError(
                        "LPIPS was requested without a verified AlexNet checkpoint"
                    )
                lpips_model = _load_offline_lpips_model(
                    checkpoint_path=lpips_alexnet_checkpoint,
                    device=device,
                    torch_module=torch_local,
                )

            if gs_video_tensor.dim() != 4 or input_video_tensor.dim() != 4:
                raise RuntimeError(
                    f"Invalid LPIPS tensor shapes: gs={gs_video_tensor.shape}, input={input_video_tensor.shape}"
                )

            num_frames = min(gs_video_tensor.shape[0], input_video_tensor.shape[0])
            if num_frames <= 0:
                raise RuntimeError("Empty video for LPIPS scoring")

            gs_clip = gs_video_tensor[:num_frames].float()
            input_clip = input_video_tensor[:num_frames].float()

            if input_clip.shape[-2:] != gs_clip.shape[-2:]:
                input_clip = torch_local.nn.functional.interpolate(
                    input_clip,
                    size=gs_clip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            gs_norm = gs_clip * 2.0 - 1.0
            input_norm = input_clip * 2.0 - 1.0

            with torch_local.no_grad():
                lpips_values = lpips_model(gs_norm, input_norm)

            lpips_mean = lpips_values.mean().item()
            gs_score = float(np.clip(1.0 - lpips_mean, 0.0, 1.0))
            print(f"[Process {os.getpid()}] LPIPS mean: {lpips_mean:.4f}, GS score: {gs_score:.4f}")
            return gs_score

        # Signal that initialization is complete
        init_queue.put(("READY", gpu_id))

        # Process tasks from queue
        while True:
            task = task_queue.get()

            # Check for shutdown signal
            if task is None:
                print(f"[Process {os.getpid()}] Received shutdown signal")
                break

            camera_trajectory = None
            if len(task) == 4:
                batch_idx, video_frames, prompt, save_dir = task
                use_lpips = False
            elif len(task) == 5:
                batch_idx, video_frames, prompt, save_dir, use_lpips = task
            elif len(task) == 6:
                batch_idx, video_frames, prompt, save_dir, use_lpips, camera_trajectory = task
            else:
                raise ValueError(f"Unexpected task format: {task}")

            try:
                print(f"[Process {os.getpid()}] Processing batch {batch_idx} on GPU {gpu_id}")
                reconstruction_indices = _uniform_sample_indices(
                    len(video_frames),
                    MAX_RECONSTRUCTION_FRAMES,
                )
                if camera_trajectory is not None:
                    if len(camera_trajectory) != len(video_frames):
                        raise ValueError(
                            "camera trajectory must align with input video frames"
                        )
                    camera_trajectory = [
                        camera_trajectory[index]
                        for index in reconstruction_indices
                    ]
                video_frames = [
                    video_frames[index] for index in reconstruction_indices
                ]

                # Convert PIL images to tensor
                frames_tensors = []
                for img in video_frames:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Convert to tensor (C, H, W) in range [0, 1]
                    frame_tensor = torch_local.from_numpy(np.array(img)).float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
                    frames_tensors.append(frame_tensor)

                # Stack to (T, C, H, W)
                video_tensor = torch_local.stack(frames_tensors, dim=0).to(device)

                gs_video, meta_view, camera_motion_score, trajectory_comparison_image = reward_3d_backend(
                    video_tensor,
                    camera_trajectory=camera_trajectory,
                )

                # Debug artifact writes are best-effort only: they never feed
                # back into gs/meta/camera/final scores, so failures warn and
                # continue instead of poisoning the row.
                gs_video_path = ""
                meta_view_path = ""
                trajectory_comparison_path = ""
                if save_dir:
                    try:
                        import cv2 as cv2_local

                        # Save GS video
                        gs_video_path = os.path.join(save_dir, f"batch_{batch_idx}_gs_video.mp4")
                        gs_frames_np = (gs_video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                        height, width = gs_frames_np.shape[1:3]
                        fourcc = cv2_local.VideoWriter_fourcc(*'mp4v')
                        gs_writer = cv2_local.VideoWriter(gs_video_path, fourcc, 24, (width, height))
                        for frame in gs_frames_np:
                            # Convert RGB to BGR for OpenCV
                            frame_bgr = cv2_local.cvtColor(frame, cv2_local.COLOR_RGB2BGR)
                            gs_writer.write(frame_bgr)
                        gs_writer.release()

                        # Save meta view (single image, not video)
                        meta_view_path = os.path.join(save_dir, f"batch_{batch_idx}_meta_view.png")
                        meta_view_np = (meta_view.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        meta_view_bgr = cv2_local.cvtColor(meta_view_np, cv2_local.COLOR_RGB2BGR)
                        cv2_local.imwrite(meta_view_path, meta_view_bgr)

                        if trajectory_comparison_image is not None:
                            trajectory_comparison_path = os.path.join(save_dir, f"batch_{batch_idx}_trajectory_comparison.png")
                            trajectory_comparison_bgr = cv2_local.cvtColor(trajectory_comparison_image, cv2_local.COLOR_RGB2BGR)
                            cv2_local.imwrite(trajectory_comparison_path, trajectory_comparison_bgr)
                    except Exception as debug_exc:
                        print(f"[Process {os.getpid()}] Warning: debug artifact write failed: {debug_exc}")

                # Debug: Check tensor properties
                print(f"[Process {os.getpid()}] gs_video shape: {gs_video.shape}, dtype: {gs_video.dtype}, range: [{gs_video.min():.3f}, {gs_video.max():.3f}]")
                print(f"[Process {os.getpid()}] meta_view shape: {meta_view.shape}, dtype: {meta_view.dtype}, range: [{meta_view.min():.3f}, {meta_view.max():.3f}]")

                # Score gs_video with gs_scorer or LPIPS
                print(f"[Process {os.getpid()}] Scoring GS video for batch {batch_idx} (lpips={use_lpips})")
                if use_lpips:
                    gs_score = compute_lpips_gs_score(gs_video, video_tensor)
                else:
                    gs_scores = gs_scorer([prompt], [gs_video])
                    if not gs_scores:
                        raise RuntimeError("Qwen scorer returned an empty result for the GS video")
                    gs_score = gs_scores[0]

                # Score meta_view with meta_scorer (single image - Qwen or OpenAI)
                # Need to add batch dimension for image: (3, H, W) -> (1, 3, H, W)
                print(f"[Process {os.getpid()}] Scoring meta view for batch {batch_idx}")
                meta_view_batch = meta_view.unsqueeze(0)  # (1, 3, H, W)
                meta_scores = meta_scorer([prompt], [meta_view_batch])
                if not meta_scores:
                    raise RuntimeError("Qwen scorer returned an empty result for the meta view")
                meta_score = meta_scores[0]

                # Paper setting: R_3D = S_meta + S_recon + S_traj, each bounded in [0, 1].
                gs_score = float(np.clip(gs_score, 0.0, 1.0))
                meta_score = float(np.clip(meta_score, 0.0, 1.0))
                camera_motion_score = float(np.clip(camera_motion_score, 0.0, 1.0))
                final_score = gs_score + meta_score + camera_motion_score

                print(f"[Process {os.getpid()}] Batch {batch_idx} completed:")
                print(f"  GS score: {gs_score:.3f}, Meta score: {meta_score:.3f}, Camera motion: {camera_motion_score:.3f}, Final: {final_score:.3f}")

                result_queue.put((
                    "ROW_OK",
                    batch_idx,
                    (
                        gs_score,
                        meta_score,
                        camera_motion_score,
                        final_score,
                        gs_video_path,
                        meta_view_path,
                        trajectory_comparison_path,
                    ),
                ))

            except Exception as e:
                print(f"[Process {os.getpid()}] Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

                # Try to recover CUDA state
                try:
                    torch_local.cuda.empty_cache()
                    torch_local.cuda.synchronize()
                except:
                    pass

                result_queue.put(("ROW_ERROR", batch_idx, "ROW_COMPUTE_FAILED"))

    except Exception as e:
        print(f"[Process {os.getpid()}] Fatal error in worker process: {e}")
        import traceback
        traceback.print_exc()
        init_queue.put(("INIT_ERROR", gpu_id, "WORKER_INIT_FAILED"))


class MultiGPUReward3DManager:
    """Fail-closed manager for multi-GPU 3D reward evaluation."""

    STRICT_MANAGER_PROTOCOL = STRICT_MANAGER_PROTOCOL
    STRICT_REWARD_KIND = STRICT_REWARD_KIND

    def __init__(self, model_name=DEFAULT_RECONSTRUCTION_MODEL, scorer_type="qwen", use_lpips=False):
        if scorer_type != "qwen":
            raise ValueError("the bundled 3D reward supports only scorer_type='qwen'")
        self.model_name = model_name
        self.scorer_type = "qwen"
        self.use_lpips = use_lpips
        self._ready = False
        self._closed = False
        self._request_lock = threading.Lock()
        self._mp_context = multiprocessing.get_context("spawn")
        self._workers = []
        self._task_queues = []
        self._result_queue = None
        # Controlled-dependency seam used only by the deployment
        # fault-injection gate; production keeps the patch constant.
        self._score_timeout_s = STRICT_MANAGER_TIMEOUT_S
        self.num_gpus = 0
        self.current_gpu_index = 0  # For round-robin assignment
        self.call_counter = 0  # Track number of calls for logging
        self.last_results = None
        self._lpips_alexnet_checkpoint = None

    def is_ready(self) -> bool:
        return self._ready and not self._closed

    def _start_worker(self, logical_index, task_queue, result_queue, init_queue):
        process = self._mp_context.Process(
            target=supervised_worker_entry,
            args=(
                os.getpid(),
                __name__,
                "reward_3d_worker_process",
                (
                    logical_index,
                    self.model_name,
                    self.scorer_type,
                    self._lpips_alexnet_checkpoint,
                    task_queue,
                    result_queue,
                    init_queue,
                ),
            ),
            daemon=True,
        )
        process.start()
        return process

    def initialize(self):
        if self._closed:
            raise StrictManagerInitError("manager is closed")
        if self._ready or self._workers:
            raise StrictManagerInitError("initialize() must be called exactly once")
        if not torch.cuda.is_available():
            raise StrictManagerInitError(
                "CUDA is required for the strict 3D reward manager"
            )
        if self.model_name is None:
            self.model_name = os.environ.get("WORLD_R1_DA3_MODEL")
        if not self.model_name:
            raise StrictManagerInitError("WORLD_R1_DA3_MODEL is required")
        if self.use_lpips:
            self._lpips_alexnet_checkpoint = _resolve_lpips_alexnet_checkpoint()

        deadline = time.monotonic() + self._score_timeout_s
        self.num_gpus = torch.cuda.device_count()
        print(f"Initializing the strict 3D reward backend on {self.num_gpus} GPUs")
        self._result_queue = self._mp_context.Queue()
        init_queue = self._mp_context.Queue()
        try:
            for gpu_id in range(self.num_gpus):
                task_queue = self._mp_context.Queue()
                self._task_queues.append(task_queue)
                self._workers.append(
                    self._start_worker(gpu_id, task_queue, self._result_queue, init_queue)
                )
                print(f"Started spawn worker for logical GPU {gpu_id}")

            ready_count = 0
            while ready_count < self.num_gpus:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StrictManagerInitError("worker initialization timed out")
                for worker in self._workers:
                    if not worker.is_alive():
                        raise StrictManagerInitError(
                            "a worker process exited during initialization"
                        )
                try:
                    envelope = init_queue.get(timeout=min(remaining, _RESULT_POLL_S))
                except queue_module.Empty:
                    continue
                tag, gpu_id = envelope[0], envelope[1]
                if tag == "READY":
                    print(f"Worker for logical GPU {gpu_id} is ready")
                    ready_count += 1
                elif tag == "INIT_ERROR":
                    raise StrictManagerInitError(
                        f"worker for logical GPU {gpu_id} failed to initialize: "
                        f"{envelope[2]}"
                    )
                else:
                    raise StrictManagerInitError(
                        f"unexpected initialization envelope {tag!r}"
                    )
        except BaseException:
            self._poison_and_shutdown()
            _close_queue(init_queue)
            raise
        _close_queue(init_queue)
        self._ready = True
        print(f"Strict 3D reward manager initialized with {self.num_gpus} spawn worker processes")

    def compute_batch_scores(self, batch_videos, batch_prompts, camera_trajectories=None, use_lpips=None):
        """
        Compute scores for a batch with load balancing across GPUs.

        Any decode/scorer/worker/timeout failure poisons the manager and
        raises; no finite 0.0 default score is ever returned.
        """
        # The single deadline is defined BEFORE attempting lock acquisition;
        # lock wait, decode, enqueue, scoring and collection all consume the
        # same remaining budget.
        deadline = time.monotonic() + self._score_timeout_s
        acquired = self._request_lock.acquire(
            timeout=max(deadline - time.monotonic(), 0.0)
        )
        if not acquired:
            raise StrictRewardTimeoutError(
                "timed out waiting for the strict manager request lock"
            )
        try:
            if not self.is_ready():
                raise StrictManagerNotReadyError(
                    "strict 3D reward manager is not ready"
                )
            try:
                return self._compute_locked(
                    batch_videos,
                    batch_prompts,
                    camera_trajectories=camera_trajectories,
                    use_lpips=use_lpips,
                    deadline=deadline,
                )
            except BaseException:
                self._poison_and_shutdown()
                raise
        finally:
            self._request_lock.release()

    def _compute_locked(self, batch_videos, batch_prompts, *, camera_trajectories, use_lpips, deadline):
        if use_lpips is None:
            use_lpips = self.use_lpips
        if use_lpips and self._lpips_alexnet_checkpoint is None:
            raise StrictRewardComputeError(
                "LPIPS must be enabled when the manager is initialized"
            )
        batch_size = len(batch_videos)
        if len(batch_prompts) != batch_size:
            raise StrictRewardComputeError(
                "batch_videos and batch_prompts must have the same length"
            )

        if camera_trajectories is None:
            camera_trajectories = [None] * batch_size
        elif len(camera_trajectories) != batch_size:
            raise StrictRewardComputeError(
                "camera_trajectories must align with batch_videos one-to-one"
            )

        save_debug_artifacts = (
            os.environ.get("WORLD_R1_SAVE_DEBUG_ARTIFACTS") == "1"
        )
        if save_debug_artifacts:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = f"logs/reward_3d/call_{self.call_counter}_{timestamp}"
            os.makedirs(batch_dir, exist_ok=True)
        else:
            batch_dir = ""
        self.call_counter += 1

        # Decode every row in the parent before enqueueing anything; a bad
        # frame JPEG is a structured decode failure, never a default score.
        tasks = []
        for batch_idx, (video_frames, prompt, camera_trajectory) in enumerate(
            zip(batch_videos, batch_prompts, camera_trajectories)
        ):
            if not video_frames:
                raise StrictRewardDecodeError(f"request row {batch_idx} has no frames")
            try:
                frames = [Image.open(BytesIO(frame_bytes)).convert('RGB') for frame_bytes in video_frames]
                for frame in frames:
                    frame.load()
            except Exception as exc:
                raise StrictRewardDecodeError(
                    f"request row {batch_idx} contains an undecodable frame"
                ) from exc
            tasks.append((batch_idx, frames, prompt, batch_dir, use_lpips, camera_trajectory))

        # Distribute tasks to workers using round-robin
        for batch_idx, frames, prompt, task_batch_dir, task_use_lpips, camera_trajectory in tasks:
            gpu_idx = self.current_gpu_index % self.num_gpus
            self.task_queues_put(gpu_idx, (batch_idx, frames, prompt, task_batch_dir, task_use_lpips, camera_trajectory))
            self.current_gpu_index += 1

        # Collect exactly one tagged terminal envelope per received row.
        results = {
            'final_scores': [None] * batch_size,
            'gs_scores': [None] * batch_size,
            'meta_scores': [None] * batch_size,
            'camera_motion_scores': [None] * batch_size,
        }
        per_video_results = []
        received = 0
        while received < len(tasks):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StrictRewardTimeoutError(
                    "timed out collecting strict worker results"
                )
            for worker in self._workers:
                if not worker.is_alive():
                    raise StrictWorkerDeathError(
                        "a strict reward worker process died mid-batch"
                    )
            try:
                envelope = self._result_queue.get(timeout=min(remaining, _RESULT_POLL_S))
            except queue_module.Empty:
                continue
            tag, batch_idx, payload = envelope
            if tag == "ROW_ERROR":
                raise StrictRewardComputeError(
                    f"request row {batch_idx} failed in the worker: {payload}"
                )
            if tag != "ROW_OK":
                raise StrictRewardComputeError(
                    f"unexpected worker result envelope {tag!r}"
                )
            (gs_score, meta_score, camera_motion_score, final_score,
             gs_video_path, meta_view_path, trajectory_comparison_path) = payload
            for name, value in (
                ("gs_score", gs_score),
                ("meta_score", meta_score),
                ("camera_motion_score", camera_motion_score),
                ("final_score", final_score),
            ):
                if not math.isfinite(float(value)):
                    raise StrictRewardComputeError(
                        f"request row {batch_idx} returned a non-finite {name}"
                    )
            results['gs_scores'][batch_idx] = gs_score
            results['meta_scores'][batch_idx] = meta_score
            results['camera_motion_scores'][batch_idx] = camera_motion_score
            results['final_scores'][batch_idx] = final_score
            received += 1

            per_video_results.append({
                "video_id": batch_idx,
                "prompt": batch_prompts[batch_idx],
                "gs_score": float(gs_score),
                "meta_score": float(meta_score),
                "camera_motion_score": float(camera_motion_score),
                "final_score": float(final_score),
                "lpips": bool(use_lpips),
                "gs_video_path": gs_video_path,
                "meta_view_path": meta_view_path,
                "trajectory_comparison_path": trajectory_comparison_path,
            })

        # Score logging is a debug side effect and must never fail the batch.
        if batch_dir:
            try:
                results_path = os.path.join(batch_dir, "reward_3d_scores.json")
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(per_video_results, f, ensure_ascii=False, indent=2)
            except Exception as log_exc:
                print(f"Warning: failed to write 3D reward score log: {log_exc}")

        self.last_results = {
            "batch_dir": batch_dir,
            "final_scores": list(results["final_scores"]),
            "gs_scores": list(results["gs_scores"]),
            "meta_scores": list(results["meta_scores"]),
            "camera_motion_scores": list(results["camera_motion_scores"]),
            "per_video_results": per_video_results,
        }

        return results['final_scores']

    def task_queues_put(self, gpu_idx, task):
        self._task_queues[gpu_idx].put(task)

    def _poison_and_shutdown(self):
        """Idempotent fail-closed cleanup shared by every failure path."""

        if self._closed:
            return
        self._closed = True
        self._ready = False
        workers = list(self._workers)
        self._workers = []
        task_queues = list(self._task_queues)
        self._task_queues = []
        result_queue = self._result_queue
        self._result_queue = None
        for task_queue in task_queues:
            _close_queue(task_queue)
        if result_queue is not None:
            _close_queue(result_queue)
        _terminate_and_reap(workers)

    def shutdown(self):
        """Shutdown all worker processes (same idempotent primitive)."""
        self._poison_and_shutdown()
