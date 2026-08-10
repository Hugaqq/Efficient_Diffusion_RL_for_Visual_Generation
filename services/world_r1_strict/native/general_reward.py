"""Multi-GPU backend for the general reward (fail-closed strict manager).

world_r1_fail_closed_v1: every decode/scorer/worker failure raises from
``compute_batch_scores`` instead of being swallowed into a finite 0.5 default
score.  Workers are explicit ``spawn`` processes so any poisoned manager can
be terminated with zero residuals.
"""

import hashlib
import importlib.util
import math
import multiprocessing
import os
import queue as queue_module
import shutil
import threading
import time
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from services.world_r1_strict.process_supervision import supervised_worker_entry

STRICT_MANAGER_PROTOCOL = "world_r1_fail_closed_v1"
STRICT_REWARD_KIND = "reward_general"
STRICT_MANAGER_TIMEOUT_S = 1800.0
STRICT_CLEANUP_TIMEOUT_S = 10.0
STRICT_FATAL_EXIT_CODE = 70
_RESULT_POLL_S = 0.05
_HPS_BPE_NAME = "bpe_simple_vocab_16e6.txt.gz"
_HPS_BPE_SIZE = 1_356_917
_HPS_BPE_SHA256 = (
    "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
)


class StrictManagerError(RuntimeError):
    """Base class for fail-closed strict manager failures."""


class StrictManagerNotReadyError(StrictManagerError):
    """compute_batch_scores was called on a non-ready manager."""


class StrictManagerInitError(StrictManagerError):
    """Worker initialization failed, timed out or lost a worker."""


class StrictRewardDecodeError(StrictManagerError):
    """One request row could not be decoded into an RGB image."""


class StrictRewardComputeError(StrictManagerError):
    """A worker reported ROW_ERROR or an invalid result envelope."""


class StrictRewardTimeoutError(StrictManagerError):
    """The single strict deadline was exhausted."""


class StrictWorkerDeathError(StrictManagerError):
    """A spawn worker process died while a batch was in flight."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hps_bpe(path: Path, *, source: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{source} HPS tokenizer resource is not a regular file")
    if path.stat().st_size != _HPS_BPE_SIZE:
        raise RuntimeError(f"{source} HPS tokenizer resource has the wrong size")
    if _file_sha256(path) != _HPS_BPE_SHA256:
        raise RuntimeError(f"{source} HPS tokenizer resource has the wrong digest")


def _install_bundled_hps_bpe() -> None:
    """Repair the PyPI hpsv2 wheel's omitted tokenizer file without downloads."""

    bundled = Path(__file__).with_name("assets") / _HPS_BPE_NAME
    _validate_hps_bpe(bundled, source="bundled")

    spec = importlib.util.find_spec("hpsv2")
    if spec is None or spec.origin is None:
        raise RuntimeError("hpsv2 must be installed in the reward environment")
    package_root = Path(spec.origin).resolve(strict=True).parent
    target = package_root / "src" / "open_clip" / _HPS_BPE_NAME
    if target.exists() or target.is_symlink():
        _validate_hps_bpe(target, source="installed")
        return

    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(bundled, temporary)
        _validate_hps_bpe(temporary, source="staged")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    _validate_hps_bpe(target, source="installed")


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


class GeneralRewardInstance:
    """Single general-reward worker pinned to one logical GPU index."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self._checkpoint: Path | None = None
        self._device = None
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def load_model(self):
        checkpoint_raw = os.environ.get("WORLD_R1_HPS_CHECKPOINT")
        if not checkpoint_raw:
            raise RuntimeError("WORLD_R1_HPS_CHECKPOINT is required")
        checkpoint = Path(checkpoint_raw)
        if not checkpoint.is_absolute() or checkpoint.is_symlink():
            raise RuntimeError(
                "WORLD_R1_HPS_CHECKPOINT must be an absolute non-symlink path"
            )
        checkpoint = checkpoint.resolve(strict=True)
        if not checkpoint.is_file():
            raise RuntimeError("WORLD_R1_HPS_CHECKPOINT must name a regular file")

        _install_bundled_hps_bpe()
        from hpsv2.src.open_clip import (
            create_model_and_transforms,
            get_tokenizer,
        )

        device = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model, _preprocess_train, preprocess = create_model_and_transforms(
            "ViT-H-14",
            pretrained=None,
            precision="fp32",
            device="cpu",
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        checkpoint_payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(checkpoint_payload, dict)
            or not isinstance(checkpoint_payload.get("state_dict"), dict)
        ):
            raise RuntimeError("HPS checkpoint must contain a state_dict mapping")
        model.load_state_dict(checkpoint_payload["state_dict"], strict=True)
        del checkpoint_payload
        model.to(device)
        model.eval()

        self._checkpoint = checkpoint
        self._device = device
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = get_tokenizer("ViT-H-14")
        print(
            f"General reward worker {self.gpu_id} initialized from {checkpoint}"
        )

    @torch.no_grad()
    def compute_score(self, images, prompts):
        # Any HPS scorer exception propagates: the strict worker converts it
        # into a ROW_ERROR envelope instead of a 0.5 default score.
        if (
            self._checkpoint is None
            or self._device is None
            or self._model is None
            or self._preprocess is None
            or self._tokenizer is None
        ):
            raise RuntimeError("general reward worker is not initialized")
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        image_batch = torch.stack(
            [self._preprocess(image) for image in images],
            dim=0,
        ).to(device=self._device, non_blocking=True)
        text_batch = self._tokenizer(list(prompts)).to(
            device=self._device,
            non_blocking=True,
        )
        # The resource contract declares fp32. CUDA autocast would silently
        # execute HPS in fp16 and can quantize nearby completion scores into an
        # exact tie, producing an all-zero GRPO advantage. Preserve one real
        # fp32 forward and its full score resolution.
        outputs = self._model(image_batch, text_batch)
        image_features = outputs["image_features"]
        text_features = outputs["text_features"]
        scores = torch.diagonal(image_features @ text_features.T)
        result = [float(value) for value in scores.float().cpu().tolist()]
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        return result


def _general_worker_main(
    logical_index,
    task_queue,
    result_queue,
    init_queue,
    *,
    instance_factory=None,
):
    """Spawn worker: one tagged terminal envelope per received row."""

    try:
        # The spawn child inherits the parent's CUDA_VISIBLE_DEVICES unchanged;
        # logical_index is a PyTorch logical device index.
        if torch.cuda.is_available():
            torch.cuda.set_device(logical_index)
        factory = GeneralRewardInstance if instance_factory is None else instance_factory
        instance = factory(logical_index)
        instance.load_model()
    except Exception:
        init_queue.put(("INIT_ERROR", logical_index, "WORKER_INIT_FAILED"))
        return

    init_queue.put(("READY", logical_index))
    while True:
        task = task_queue.get()
        if task is None:
            return
        row_index, image, prompt = task
        try:
            score = float(instance.compute_score([image], [prompt])[0])
        except Exception:
            result_queue.put(("ROW_ERROR", row_index, "SCORER_EXCEPTION"))
            continue
        result_queue.put(("ROW_OK", row_index, score))


class MultiGPUGeneralRewardManager:
    """Fail-closed multi-GPU manager for general reward evaluation."""

    STRICT_MANAGER_PROTOCOL = STRICT_MANAGER_PROTOCOL
    STRICT_REWARD_KIND = STRICT_REWARD_KIND

    def __init__(self):
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

    def is_ready(self) -> bool:
        return self._ready and not self._closed

    def _start_worker(
        self,
        logical_index,
        task_queue,
        result_queue,
        init_queue,
        *,
        instance_factory=None,
    ):
        if instance_factory is None:
            process = self._mp_context.Process(
                target=supervised_worker_entry,
                args=(
                    os.getpid(),
                    __name__,
                    "_general_worker_main",
                    (logical_index, task_queue, result_queue, init_queue),
                ),
                daemon=True,
            )
        else:
            process = self._mp_context.Process(
                target=_general_worker_main,
                args=(logical_index, task_queue, result_queue, init_queue),
                kwargs={"instance_factory": instance_factory},
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
                "CUDA is required for the strict general reward manager"
            )
        deadline = time.monotonic() + self._score_timeout_s
        self.num_gpus = torch.cuda.device_count()
        self._result_queue = self._mp_context.Queue()
        init_queue = self._mp_context.Queue()
        try:
            for gpu_id in range(self.num_gpus):
                task_queue = self._mp_context.Queue()
                self._task_queues.append(task_queue)
                self._workers.append(
                    self._start_worker(
                        gpu_id, task_queue, self._result_queue, init_queue
                    )
                )
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
        print(
            f"Strict general reward manager initialized with "
            f"{self.num_gpus} spawn worker processes"
        )

    def compute_batch_scores(self, batch_images, batch_prompts):
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
                    "strict general reward manager is not ready"
                )
            try:
                return self._compute_locked(batch_images, batch_prompts, deadline)
            except BaseException:
                self._poison_and_shutdown()
                raise
        finally:
            self._request_lock.release()

    def _compute_locked(self, batch_images, batch_prompts, deadline):
        if len(batch_images) != len(batch_prompts):
            raise StrictRewardComputeError(
                "batch_images and batch_prompts must have the same length"
            )
        decoded = []
        for row_index, (image_bytes, prompt) in enumerate(
            zip(batch_images, batch_prompts)
        ):
            try:
                image = Image.open(BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.load()
            except Exception as exc:
                raise StrictRewardDecodeError(
                    f"request row {row_index} is not a decodable image"
                ) from exc
            decoded.append((row_index, image, prompt))

        for row_index, image, prompt in decoded:
            worker_slot = row_index % self.num_gpus
            self._task_queues[worker_slot].put((row_index, image, prompt))

        results = [None] * len(decoded)
        received = 0
        while received < len(decoded):
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
                envelope = self._result_queue.get(
                    timeout=min(remaining, _RESULT_POLL_S)
                )
            except queue_module.Empty:
                continue
            tag, row_index, payload = envelope
            if tag == "ROW_ERROR":
                raise StrictRewardComputeError(
                    f"request row {row_index} failed in the worker: {payload}"
                )
            if tag != "ROW_OK":
                raise StrictRewardComputeError(
                    f"unexpected worker result envelope {tag!r}"
                )
            value = float(payload)
            if not math.isfinite(value):
                raise StrictRewardComputeError(
                    f"request row {row_index} returned a non-finite score"
                )
            results[row_index] = value
            received += 1
        return results

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
        self._poison_and_shutdown()
