"""VisualRL command line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_PRESET_DIR = Path(__file__).resolve().parents[1] / "visual_rl" / "configs" / "presets"
_SCRIPT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _register_builtin_plugins() -> None:
    from visual_rl.builtins import register_builtin_plugins

    register_builtin_plugins()


def smoke_imports(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    import visual_rl
    from visual_rl.core.registry import ALGORITHMS, MODEL_ADAPTERS, REWARD_CLIENTS

    payload = {
        "visual_rl_version": visual_rl.__version__,
        "algorithms": ALGORITHMS.keys(),
        "model_adapters": MODEL_ADAPTERS.keys(),
        "reward_clients": REWARD_CLIENTS.keys(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def smoke_mock(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from visual_rl.runner import ExperimentRunner

    default_config = _PRESET_DIR / "world_r1_wan_v02_mock.yaml"
    config = load_config(args.config or default_config)
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    trainer = ExperimentRunner(config)
    metrics = trainer.run(max_steps=args.steps)
    print(json.dumps({"output_dir": config.paths.output_dir, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


def tempflow_smoke(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from visual_rl.runner import ExperimentRunner

    default_config = _PRESET_DIR / "tempflow_tiny_branching.yaml"
    config = load_config(args.config or default_config)
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    trainer = ExperimentRunner(config)
    metrics = trainer.run(max_steps=args.steps)
    print(json.dumps({"output_dir": config.paths.output_dir, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


def flash_smoke(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from visual_rl.runner import ExperimentRunner

    default_config = _PRESET_DIR / "flash_tiny_single_step.yaml"
    config = load_config(args.config or default_config)
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    trainer = ExperimentRunner(config)
    metrics = trainer.run(max_steps=args.steps)
    print(json.dumps({"output_dir": config.paths.output_dir, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


def adapter_probe(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config, section_to_dict
    from visual_rl.core.registry import MODEL_ADAPTERS

    if args.config:
        config = load_config(args.config)
        model_config = section_to_dict(config.model)
        model_config.setdefault("use_lora", config.use_lora)
        model_config.setdefault("lora_path", config.train.lora_path)
    else:
        model_config = {"name": args.adapter, "model_path": args.model_path or "", "model_family": "image", "extra": {}}
    model_config.setdefault("extra", {})
    if args.device:
        model_config["extra"]["device"] = args.device
    if not args.load:
        model_config["extra"]["defer_load"] = True

    adapter_cls = MODEL_ADAPTERS.get(model_config["name"])
    adapter = adapter_cls(model_config)
    payload = {"adapter": adapter.name, "loaded": bool(args.load), "model_path": model_config.get("model_path", "")}
    if args.load:
        params = list(adapter.parameters())
        payload["trainable_parameter_tensors"] = len(params)
        payload["trainable_parameters"] = int(sum(parameter.numel() for parameter in params))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _reward_client_summary(clients: dict[str, Any]) -> dict[str, str | None]:
    summary = {}
    for alias, client_config in clients.items():
        if isinstance(client_config, dict):
            summary[alias] = client_config.get("name")
        else:
            summary[alias] = getattr(client_config, "name", None)
    return summary


def _validate_config_registry_names(config) -> None:
    from visual_rl.core.registry import ALGORITHMS, MODEL_ADAPTERS, REWARD_CLIENTS

    MODEL_ADAPTERS.get(config.model.name)
    ALGORITHMS.get(config.algorithm.name)
    client_summary = _reward_client_summary(config.rewards.clients)
    if not config.rewards.weights:
        raise ValueError("rewards.weights cannot be empty.")
    missing_clients = sorted(set(config.rewards.weights) - set(client_summary))
    if missing_clients and client_summary:
        missing = ", ".join(repr(name) for name in missing_clients)
        raise ValueError(f"Missing reward client configuration for reward weight(s): {missing}.")
    for alias, client_name in client_summary.items():
        if client_name is None:
            raise ValueError(f"Reward client {alias!r} is missing a name.")
        REWARD_CLIENTS.get(client_name)
    if not client_summary:
        for reward_name in config.rewards.weights:
            REWARD_CLIENTS.get(reward_name)


def _config_validation_payload(path: str, config) -> dict[str, Any]:
    runner = {
        "strict_rollout_validation": config.runner.strict_rollout_validation,
    }
    return {
        "path": path,
        "run_name": config.run_name,
        "model": {
            "name": config.model.name,
            "model_family": config.model.model_family,
        },
        "model_family": config.model.model_family,
        "sample": {
            "name": config.sample.name,
        },
        "algorithm": {
            "name": config.algorithm.name,
        },
        "output_dir": config.paths.output_dir,
        "rewards": {
            "names": sorted(config.rewards.weights),
            "clients": _reward_client_summary(config.rewards.clients),
        },
        "runner": runner,
    }


def validate_config(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config

    configs = []
    errors = []
    for path in args.configs:
        try:
            config = load_config(path)
            _validate_config_registry_names(config)
        except Exception as exc:  # noqa: BLE001 - CLI reports structured validation errors
            errors.append({"path": str(path), "message": str(exc)})
            continue
        configs.append(_config_validation_payload(str(path), config))

    payload = {
        "valid": not errors,
        "configs": configs,
    }
    if errors:
        payload["errors"] = errors
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _rollout_probe_error_payload(path: str, errors: list[str]) -> dict[str, Any]:
    return {
        "valid": False,
        "config_path": path,
        "errors": errors,
    }


def _rollout_probe_payload(args: argparse.Namespace) -> dict[str, Any]:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config, section_to_dict
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.datasets.prompt_dataset import PromptDataset
    from visual_rl.rollout.full_trajectory import build_rollout_engine

    config = load_config(args.config)
    _validate_config_registry_names(config)

    model_config = section_to_dict(config.model)
    model_config.setdefault("use_lora", config.use_lora)
    model_config.setdefault("lora_path", config.train.lora_path)
    if config.paths.pretrained_model and not model_config.get("model_path"):
        model_config["model_path"] = config.paths.pretrained_model
    adapter_cls = MODEL_ADAPTERS.get(model_config.get("name", "mock_wan"))
    adapter = adapter_cls(model_config)

    rollout_config = section_to_dict(config.sample)
    rollout_config.update(config.rollout)
    if args.num_steps is not None:
        rollout_config["num_steps"] = args.num_steps
    seed = int(config.seed if args.seed is None else args.seed)
    epoch_tag = 0
    rollout_config["epoch_tag"] = epoch_tag
    rollout_config["seed"] = seed
    rollout = build_rollout_engine(rollout_config)

    dataset = PromptDataset.from_config(config.dataset)
    batch_size = int(config.sample.batch_size if args.batch_size is None else args.batch_size)
    if batch_size <= 0:
        raise ValueError(f"rollout-probe requires --batch-size > 0, got {batch_size}.")
    if int(rollout_config.get("num_steps", 1)) <= 0:
        raise ValueError(f"rollout-probe requires --num-steps > 0, got {rollout_config.get('num_steps')}.")
    prompts, metadata, _ = dataset.batch(0, batch_size, epoch_tag=epoch_tag)
    batch = rollout.sample(adapter, prompts, metadata)
    if args.strict:
        batch.validate_strict()
    else:
        batch.validate_lightweight(strict=False)

    return {
        "valid": True,
        "config_path": str(args.config),
        "run_name": config.run_name,
        "adapter": getattr(adapter, "name", model_config.get("name")),
        "adapter_key": model_config.get("name"),
        "sample": {"name": config.sample.name},
        "rollout": {
            "name": rollout_config.get("name", "full_trajectory"),
            "epoch_tag": epoch_tag,
        },
        "input_prompt_count": len(prompts),
        "prompt_count": len(batch.prompts),
        "media_shape": _shape_list(batch.media),
        "latents_shape": _shape_list(batch.latents),
        "next_latents_shape": _shape_list(batch.next_latents),
        "timesteps_shape": _shape_list(batch.timesteps),
        "old_log_probs_shape": _shape_list(batch.old_log_probs),
        "kl_shape": _shape_list(batch.kl),
        "branch_ids_shape": _shape_list(batch.branch_ids),
        "shapes": {
            "media": _shape_list(batch.media),
            "latents": _shape_list(batch.latents),
            "next_latents": _shape_list(batch.next_latents),
            "timesteps": _shape_list(batch.timesteps),
            "old_log_probs": _shape_list(batch.old_log_probs),
            "kl": _shape_list(batch.kl),
            "branch_ids": _shape_list(batch.branch_ids),
        },
        "model_metadata": dict(batch.model_metadata),
        "seed": seed,
        "strict": bool(args.strict),
    }


def rollout_probe(args: argparse.Namespace) -> int:
    try:
        payload = _rollout_probe_payload(args)
    except Exception as exc:  # noqa: BLE001 - CLI reports structured probe errors
        payload = _rollout_probe_error_payload(str(args.config), [str(exc)])
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _reward_probe_error_payload(path: str, errors: list[str]) -> dict[str, Any]:
    return {
        "valid": False,
        "config_path": path,
        "errors": errors,
    }


_MISSING = object()


def _section_get(section, key: str, default: Any = _MISSING) -> Any:
    if section is None:
        return default
    if isinstance(section, Mapping):
        return section.get(key, default)
    return getattr(section, key, default)


def _positive_int(value: Any, source: str) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid reward-probe media size {source}={value!r}; expected a positive integer.") from exc
    if parsed <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"Invalid reward-probe media size {source}={value!r}; expected a positive integer.")
    return parsed


def _height_width_from_section(section, source: str) -> tuple[int, int] | None:
    height = _section_get(section, "height")
    width = _section_get(section, "width")
    if height is _MISSING and width is _MISSING:
        return None
    if height is _MISSING or width is _MISSING:
        raise ValueError(f"Invalid reward-probe media size in {source}; height and width must both be set.")
    return _positive_int(height, f"{source}.height"), _positive_int(width, f"{source}.width")


def _square_from_section(section, source: str, key: str) -> tuple[int, int] | None:
    value = _section_get(section, key)
    if value is _MISSING:
        return None
    size = _positive_int(value, f"{source}.{key}")
    return size, size


def _infer_probe_image_size(config) -> tuple[int, int]:
    model = _section_get(config, "model", None)
    extra = _section_get(model, "extra", {}) or {}

    for source, section in (("model.extra", extra), ("model", model)):
        height_width = _height_width_from_section(section, source)
        if height_width is not None:
            return height_width
        for key in ("resolution", "image_size"):
            square = _square_from_section(section, source, key)
            if square is not None:
                return square

    media_shape = _section_get(model, "media_shape", []) or []
    if len(media_shape) >= 2:
        return (
            _positive_int(media_shape[-2], "model.media_shape[-2]"),
            _positive_int(media_shape[-1], "model.media_shape[-1]"),
        )
    return 16, 16


def _target_color_for_prompt(prompt: str, metadata: dict[str, Any], fallback_index: int) -> str:
    color = metadata.get("target_color")
    if color is not None:
        return str(color)
    lower_prompt = prompt.lower()
    for candidate in ("red", "green", "blue"):
        if candidate in lower_prompt:
            return candidate
    return ("red", "green", "blue")[fallback_index % 3]


def _synthetic_reward_probe_media(
    prompts: list[str], metadata: list[dict[str, Any]], config, seed: int
):
    import torch

    height, width = _infer_probe_image_size(config)
    generator = torch.Generator().manual_seed(int(seed))
    media = torch.rand((len(prompts), 3, height, width), generator=generator, dtype=torch.float32) * 0.05
    color_to_index = {"red": 0, "green": 1, "blue": 2}
    for index, prompt in enumerate(prompts):
        color = _target_color_for_prompt(prompt, metadata[index], fallback_index=index)
        channel = color_to_index.get(color, 0)
        media[index, channel].fill_(0.95)
        metadata[index].setdefault("target_color", color)
    return media


def _tensor_json_summary(value) -> dict[str, Any]:
    shape = _shape_list(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return {"shape": shape, "values": value.detach().cpu().tolist()}
    except Exception:  # noqa: BLE001 - fall back to numpy/list conversion below
        pass
    try:
        import numpy as np

        array = np.asarray(value)
        return {"shape": [int(item) for item in array.shape], "values": array.tolist()}
    except Exception:  # noqa: BLE001 - last-resort JSON-safe representation
        return {"shape": shape, "values": value}


def _reward_probe_payload(args: argparse.Namespace) -> dict[str, Any]:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from visual_rl.datasets.prompt_dataset import PromptDataset
    from visual_rl.feedback.router import RewardRouter

    config = load_config(args.config)
    _validate_config_registry_names(config)

    dataset = PromptDataset.from_config(config.dataset)
    batch_size = int(config.sample.batch_size if args.batch_size is None else args.batch_size)
    if batch_size <= 0:
        raise ValueError(f"reward-probe requires --batch-size > 0, got {batch_size}.")
    seed = int(config.seed if args.seed is None else args.seed)
    prompts, metadata, _ = dataset.batch(0, batch_size, epoch_tag=0)
    media = _synthetic_reward_probe_media(prompts, metadata, config, seed)

    router = RewardRouter(config.rewards, cache_dir=None)
    rewards = router.score(media, prompts, metadata)
    valid_mask = rewards.valid_mask.detach().cpu().tolist()
    payload = {
        "valid": bool(all(valid_mask)),
        "config_path": str(args.config),
        "run_name": config.run_name,
        "prompt_count": len(prompts),
        "prompts": prompts,
        "media_shape": _shape_list(media),
        "media_height": int(media.shape[-2]),
        "media_width": int(media.shape[-1]),
        "reward_names": sorted(config.rewards.weights),
        "raw": {name: _tensor_json_summary(value) for name, value in rewards.raw.items()},
        "weighted": {name: _tensor_json_summary(value) for name, value in rewards.weighted.items()},
        "weighted_total": _tensor_json_summary(rewards.weighted_total),
        "valid_mask": valid_mask,
        "metadata": rewards.metadata,
        "fail_policy": config.rewards.fail_policy,
        "seed": seed,
    }
    if not payload["valid"]:
        payload["errors"] = [f"Reward router returned invalid mask: {valid_mask}."]
    return payload


def reward_probe(args: argparse.Namespace) -> int:
    try:
        payload = _reward_probe_payload(args)
    except Exception as exc:  # noqa: BLE001 - CLI reports structured probe errors
        payload = _reward_probe_error_payload(str(args.config), [str(exc)])
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


def _shape_list(value) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(item) for item in shape]
    if isinstance(value, (list, tuple)):
        return [len(value)]
    return []


def _tensor_finite(value) -> bool:
    import torch

    detached = value.detach() if hasattr(value, "detach") else torch.as_tensor(value)
    return bool(torch.isfinite(detached).all().item())


def _tensor_sha256(value) -> str:
    import torch

    tensor = torch.as_tensor(value).detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _trainable_parameter_sha256(adapter) -> str:
    named_parameters = getattr(adapter, "named_parameters", None)
    if callable(named_parameters):
        items = list(named_parameters())
    else:
        items = [
            (f"parameter_{index}", parameter)
            for index, parameter in enumerate(adapter.parameters())
        ]
    digest = hashlib.sha256()
    for name, parameter in sorted(items, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(parameter).encode("ascii"))
    return digest.hexdigest()


_TEMPFLOW_IMAGE_SMOKE_SPECS = {
    "sd3_tempflow": {
        "label": "SD3",
        "model_family": "sd3",
        "metadata_source": "sd3_numeric_smoke",
    },
}

_IMAGE_PREVIEW_ADAPTERS = {
    **_TEMPFLOW_IMAGE_SMOKE_SPECS,
    "tiny_diffusion": {
        "label": "TinyDiffusion",
        "model_family": "image",
        "metadata_source": "image_preview",
    },
}

_SD3_BOUNDED_DEFAULT_MAX_STEPS = 5
_SD3_BOUNDED_LONG_MAX_STEPS = 100
_SD3_BOUNDED_LARGE_MAX_STEPS = 2000


def _tempflow_image_model_config(args: argparse.Namespace, adapter_key: str) -> dict[str, Any]:
    spec = _TEMPFLOW_IMAGE_SMOKE_SPECS[adapter_key]
    extra = {"resolution": args.resolution}
    for key in ("dtype", "lora_rank", "lora_alpha", "max_sequence_length"):
        value = getattr(args, key, None)
        if value is not None:
            extra[key] = value
    if args.device:
        extra["device"] = args.device
    if args.repo_root:
        extra["repo_root"] = args.repo_root
    return {
        "name": adapter_key,
        "model_family": spec["model_family"],
        "model_path": args.model_path,
        "use_lora": not getattr(args, "disable_lora", False),
        "extra": extra,
    }


def _image_preview_model_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.adapter in _TEMPFLOW_IMAGE_SMOKE_SPECS:
        return _tempflow_image_model_config(args, args.adapter)
    if args.adapter == "tiny_diffusion":
        extra = {"resolution": args.resolution, "image_size": args.resolution}
        if args.device:
            extra["device"] = args.device
        return {
            "name": "tiny_diffusion",
            "model_family": "image",
            "model_path": args.model_path,
            "use_lora": False,
            "extra": extra,
        }
    raise KeyError(f"Unsupported image-preview adapter {args.adapter!r}.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return {
                "kind": "torch.Tensor",
                "shape": _shape_list(value),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    except Exception:  # noqa: BLE001 - metadata serialization should degrade gracefully
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return {
                "kind": "numpy.ndarray",
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
        if isinstance(value, np.generic):
            return value.item()
    except Exception:  # noqa: BLE001 - metadata serialization should degrade gracefully
        pass
    return str(value)


def _media_to_numpy(value: Any):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:  # noqa: BLE001 - fall through to numpy conversion
        pass
    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            import numpy as np

            return np.asarray(value.convert("RGB"))
    except Exception:  # noqa: BLE001 - pillow is optional
        pass

    import numpy as np

    return np.asarray(value)


def _normalize_rgb_to_uint8(image):
    import numpy as np

    original_dtype = image.dtype
    if original_dtype == np.uint8:
        return image
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if image.size == 0:
        raise ValueError("Cannot save empty media as PNG.")
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0.0 or maximum > 1.0:
        if -1.00001 <= minimum and maximum <= 1.00001:
            image = (image + 1.0) / 2.0
        elif 0.0 <= minimum and maximum <= 255.0:
            image = image / 255.0
        elif maximum > minimum:
            image = (image - minimum) / (maximum - minimum)
        else:
            image = np.zeros_like(image, dtype=np.float32)
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _media_item_to_uint8_rgb(media: Any, index: int = 0):
    import numpy as np

    if isinstance(media, list | tuple):
        if not media:
            raise ValueError("Cannot save empty media sequence as PNG.")
        media = media[index]

    image = _media_to_numpy(media)
    channel_first = False
    if image.ndim == 4:
        if index >= image.shape[0]:
            raise ValueError(f"Media batch index {index} is out of range for shape {tuple(image.shape)}.")
        if image.shape[1] in (1, 3, 4):
            image = image[index]
            channel_first = True
        elif image.shape[-1] in (1, 3, 4):
            image = image[index]
        else:
            raise ValueError(f"Unsupported 4D media shape for PNG export: {tuple(image.shape)}.")
    if image.ndim == 3:
        if channel_first or (image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4)):
            image = np.moveaxis(image, 0, -1)
    elif image.ndim == 2:
        image = image[..., None]
    else:
        raise ValueError(f"Unsupported media shape for PNG export: {tuple(image.shape)}.")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"PNG export requires RGB-like media, got shape {tuple(image.shape)}.")
    return _normalize_rgb_to_uint8(image)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    import struct
    import zlib

    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def _write_png_rgb(path: Path, image) -> None:
    import struct
    import zlib

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"PNG writer expects HWC RGB uint8, got shape {tuple(image.shape)}.")
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"PNG writer expects 3 channels, got {channels}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _save_preview_pngs(media: Any, output_dir: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        image = _media_item_to_uint8_rgb(media, index=index)
        path = output_dir / f"preview_{index:03d}.png"
        _write_png_rgb(path, image)
        paths.append(path)
    return paths


def _image_preview_payload(args: argparse.Namespace) -> dict[str, Any]:
    _register_builtin_plugins()
    from visual_rl.core.registry import MODEL_ADAPTERS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = _image_preview_model_config(args)
    adapter = MODEL_ADAPTERS.get(args.adapter)(model_config)
    rollout_config = {
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "output_type": "pt",
    }
    batch = adapter.sample(
        [args.prompt],
        [{"source": "image_preview", "adapter_key": args.adapter}],
        rollout_config,
    )
    batch.validate_strict()

    png_paths = _save_preview_pngs(batch.media, output_dir, count=len(batch.prompts))
    metadata_path = output_dir / "metadata.json"
    payload = {
        "valid": True,
        "adapter": getattr(adapter, "name", args.adapter),
        "adapter_key": args.adapter,
        "model_family": model_config.get("model_family"),
        "model_path": args.model_path,
        "repo_root": args.repo_root,
        "prompt": args.prompt,
        "resolution": args.resolution,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "device": str(getattr(adapter, "device", args.device)),
        "output_dir": str(output_dir),
        "png_path": str(png_paths[0]) if png_paths else None,
        "png_paths": [str(path) for path in png_paths],
        "metadata_path": str(metadata_path),
        "prompt_count": len(batch.prompts),
        "media_shape": _shape_list(batch.media),
        "latents_shape": _shape_list(batch.latents),
        "next_latents_shape": _shape_list(batch.next_latents),
        "timesteps_shape": _shape_list(batch.timesteps),
        "old_log_probs_shape": _shape_list(batch.old_log_probs),
        "kl_shape": _shape_list(batch.kl),
        "branch_ids_shape": _shape_list(batch.branch_ids),
        "model_metadata": _json_safe(batch.model_metadata),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def image_preview(args: argparse.Namespace) -> int:
    payload = _image_preview_payload(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _image_panel_preview_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Render a fixed prompt panel at several diffusion-step settings."""

    import time

    _register_builtin_plugins()
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.datasets.prompt_dataset import (
        prompt_content_sha256,
        prompt_id,
        read_prompt_file,
    )

    prompts = read_prompt_file(args.prompts_file)
    max_prompts = getattr(args, "max_prompts", None)
    if max_prompts is not None:
        if int(max_prompts) < 1:
            raise ValueError("image-panel-preview --max-prompts must be positive")
        prompts = prompts[: int(max_prompts)]
    if not prompts:
        raise ValueError("image-panel-preview prompt file is empty")
    num_steps_list = [int(value) for value in args.num_steps_list]
    seeds = [int(value) for value in args.seeds]
    if not num_steps_list or any(value < 1 for value in num_steps_list):
        raise ValueError("image-panel-preview diffusion steps must be positive")
    if len(set(num_steps_list)) != len(num_steps_list):
        raise ValueError("image-panel-preview diffusion steps must be unique")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("image-panel-preview seeds must be non-empty and unique")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = _image_preview_model_config(args)
    model_config["use_lora"] = False
    adapter = MODEL_ADAPTERS.get(args.adapter)(model_config)
    records = []
    for num_steps in num_steps_list:
        for prompt_index, prompt in enumerate(prompts):
            for seed in seeds:
                step_dir = output_dir / f"steps_{num_steps:02d}" / f"seed_{seed}"
                png_path = step_dir / f"preview_{prompt_index:03d}.png"
                rollout_config = {
                    "num_steps": num_steps,
                    "guidance_scale": float(args.guidance_scale),
                    "seed": seed,
                    "output_type": "pt",
                }
                metadata = [
                    {
                        "source": "image_panel_preview",
                        "adapter_key": args.adapter,
                        "prompt_id": prompt_id(prompt),
                        "prompt_index": prompt_index,
                        "eval_seed": seed,
                        "num_steps": num_steps,
                    }
                ]
                peak_memory_mb = None
                if str(getattr(adapter, "device", args.device)).startswith("cuda"):
                    import torch

                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                started = time.perf_counter()
                batch = adapter.sample([prompt], metadata, rollout_config)
                batch.validate_strict()
                if str(getattr(adapter, "device", args.device)).startswith("cuda"):
                    import torch

                    torch.cuda.synchronize()
                    peak_memory_mb = float(
                        torch.cuda.max_memory_allocated() / (1024 * 1024)
                    )
                elapsed_seconds = time.perf_counter() - started
                _write_png_rgb(
                    png_path,
                    _media_item_to_uint8_rgb(batch.media, index=0),
                )
                records.append(
                    {
                        "prompt": prompt,
                        "prompt_id": prompt_id(prompt),
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "num_steps": num_steps,
                        "png_path": str(png_path),
                        "elapsed_seconds": elapsed_seconds,
                        "peak_memory_allocated_mb": peak_memory_mb,
                        "image_guardrail": _image_guardrail_summary(batch.media),
                        "media_shape": _shape_list(batch.media),
                        "model_metadata": _json_safe(batch.model_metadata),
                    }
                )

    by_steps = {}
    for num_steps in num_steps_list:
        step_records = [
            record for record in records if record["num_steps"] == num_steps
        ]
        memory_values = [
            float(record["peak_memory_allocated_mb"])
            for record in step_records
            if record["peak_memory_allocated_mb"] is not None
        ]
        by_steps[str(num_steps)] = {
            "sample_count": len(step_records),
            "elapsed_seconds_mean": sum(
                float(record["elapsed_seconds"]) for record in step_records
            )
            / len(step_records),
            "elapsed_seconds_total": sum(
                float(record["elapsed_seconds"]) for record in step_records
            ),
            "peak_memory_allocated_mb_max": (
                max(memory_values) if memory_values else None
            ),
        }
    payload = {
        "valid": True,
        "adapter": getattr(adapter, "name", args.adapter),
        "adapter_key": args.adapter,
        "model_path": args.model_path,
        "repo_root": args.repo_root,
        "prompts_file": args.prompts_file,
        "prompt_count": len(prompts),
        "prompt_content_sha256": prompt_content_sha256(prompts),
        "num_steps_list": num_steps_list,
        "seeds": seeds,
        "resolution": int(args.resolution),
        "guidance_scale": float(args.guidance_scale),
        "device": str(getattr(adapter, "device", args.device)),
        "dtype": str(getattr(adapter, "dtype", args.dtype)),
        "use_lora": False,
        "output_dir": str(output_dir),
        "record_count": len(records),
        "by_steps": by_steps,
        "records": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def image_panel_preview(args: argparse.Namespace) -> int:
    payload = _image_panel_preview_payload(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate_sd3_bounded_trainer_args(args: argparse.Namespace) -> None:
    if args.adapter != "sd3_tempflow":
        raise ValueError(f"sd3-bounded-trainer-smoke only supports --adapter sd3_tempflow, got {args.adapter!r}.")
    for key in ("resolution", "num_steps", "steps", "lora_rank", "lora_alpha"):
        value = int(getattr(args, key))
        if value <= 0:
            raise ValueError(f"sd3-bounded-trainer-smoke requires --{key.replace('_', '-')} > 0, got {value}.")
    steps = int(args.steps)
    allow_long_run = bool(getattr(args, "allow_long_run", False))
    allow_large_run = bool(getattr(args, "allow_large_run", False))
    if steps > _SD3_BOUNDED_LARGE_MAX_STEPS:
        raise ValueError(
            "sd3-bounded-trainer-smoke is capped at "
            f"--steps <= {_SD3_BOUNDED_LARGE_MAX_STEPS}, got {args.steps}."
        )
    if steps > _SD3_BOUNDED_LONG_MAX_STEPS and not allow_large_run:
        raise ValueError(
            "sd3-bounded-trainer-smoke large runs require "
            f"--allow-large-run above {_SD3_BOUNDED_LONG_MAX_STEPS} step(s)."
        )
    if (
        steps > _SD3_BOUNDED_DEFAULT_MAX_STEPS
        and not allow_long_run
        and not allow_large_run
    ):
        raise ValueError(
            "sd3-bounded-trainer-smoke defaults to short smoke runs. "
            f"Pass --allow-long-run to run more than {_SD3_BOUNDED_DEFAULT_MAX_STEPS} step(s)."
        )
    if args.resume_from and args.disable_lora:
        raise ValueError("sd3-bounded-trainer-smoke cannot combine --resume-from with --disable-lora.")
    logprob_atol = float(getattr(args, "logprob_atol", 1e-5))
    if logprob_atol < 0:
        raise ValueError(
            "sd3-bounded-trainer-smoke requires --logprob-atol >= 0, "
            f"got {logprob_atol}."
        )
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise ValueError(f"sd3-bounded-trainer-smoke --resume-from does not exist: {resume_path}")
        if not resume_path.is_dir():
            raise ValueError(f"sd3-bounded-trainer-smoke --resume-from must be a checkpoint directory: {resume_path}")
        resume_step = _checkpoint_step_from_path(args.resume_from)
        if int(args.steps) <= resume_step:
            raise ValueError(
                "sd3-bounded-trainer-smoke --steps is an absolute target and "
                f"must exceed resumed step {resume_step}, got {args.steps}"
            )
    baseline_eval = getattr(args, "baseline_eval", None)
    if baseline_eval and not Path(baseline_eval).is_file():
        raise ValueError(
            "sd3-bounded-trainer-smoke --baseline-eval does not exist: "
            f"{baseline_eval}"
        )

    train_prompt_file = getattr(args, "train_prompts_file", None)
    heldout_prompt_file = getattr(args, "heldout_prompts_file", None)
    if bool(train_prompt_file) != bool(heldout_prompt_file):
        raise ValueError(
            "sd3-bounded-trainer-smoke requires both --train-prompts-file and "
            "--heldout-prompts-file"
        )
    eval_seeds = list(getattr(args, "eval_seeds", []) or [])
    if not eval_seeds:
        raise ValueError("sd3-bounded-trainer-smoke requires at least one eval seed")
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("sd3-bounded-trainer-smoke eval seeds must be unique")
    eval_max_prompts = getattr(args, "eval_max_prompts", None)
    if eval_max_prompts is not None and int(eval_max_prompts) < 1:
        raise ValueError(
            "sd3-bounded-trainer-smoke --eval-max-prompts must be positive"
        )
    branch_count = int(getattr(args, "branch_count", 2))
    if branch_count < 2:
        raise ValueError(
            "sd3-bounded-trainer-smoke --branch-count must be at least 2"
        )
    sample_batch_size = int(getattr(args, "sample_batch_size", 1))
    if sample_batch_size < 1:
        raise ValueError(
            "sd3-bounded-trainer-smoke --sample-batch-size must be positive"
        )
    condition = str(getattr(args, "condition", "active"))
    if condition not in {"active", "zero_lr_control"}:
        raise ValueError(
            "sd3-bounded-trainer-smoke --condition must be active or "
            "zero_lr_control"
        )
    execution_mode = str(
        getattr(args, "tempflow_execution_mode", "policy-identity")
    )
    if execution_mode not in {"reference-compatible", "policy-identity"}:
        raise ValueError(
            "sd3-bounded-trainer-smoke --tempflow-execution-mode must be "
            "reference-compatible or policy-identity"
        )
    if bool(getattr(args, "allow_initial_clipping", False)) and (
        execution_mode != "reference-compatible"
    ):
        raise ValueError(
            "sd3-bounded-trainer-smoke --allow-initial-clipping is only valid "
            "with --tempflow-execution-mode reference-compatible"
        )


def _sd3_bounded_trainer_config(args: argparse.Namespace):
    from visual_rl.configs.schema import load_config
    from visual_rl.datasets.prompt_dataset import (
        prompt_content_sha256,
        read_prompt_file,
        validate_prompt_splits,
    )

    default_config = _PRESET_DIR / "sd3_tempflow_adapter.yaml"
    config = load_config(default_config)
    output_dir = str(Path(args.output_dir))
    config.run_name = "sd3_bounded_trainer_smoke"
    config.seed = int(args.seed)
    config.paths.output_dir = output_dir
    config.paths.pretrained_model = args.model_path
    config.paths.resume_from = args.resume_from

    config.model.name = args.adapter
    config.model.model_family = _TEMPFLOW_IMAGE_SMOKE_SPECS[args.adapter]["model_family"]
    config.model.model_path = args.model_path
    execution_mode = str(
        getattr(args, "tempflow_execution_mode", "policy-identity")
    )
    reference_mode = execution_mode == "reference-compatible"
    if bool(getattr(args, "allow_initial_clipping", False)) and not reference_mode:
        raise ValueError(
            "allow_initial_clipping is only valid in reference-compatible mode"
        )
    config.model.extra = {
        **dict(config.model.extra),
        "repo_root": args.repo_root,
        "resolution": int(args.resolution),
        "dtype": args.dtype,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "max_sequence_length": int(args.max_sequence_length),
        "tempflow_reference_mode": reference_mode,
    }
    if args.device:
        config.model.extra["device"] = args.device
    config.use_lora = not bool(args.disable_lora)
    config.train.lora_path = None

    if args.train_prompts_file:
        train_prompts = read_prompt_file(args.train_prompts_file)
        heldout_prompts = read_prompt_file(args.heldout_prompts_file)
        validate_prompt_splits(
            train_prompts,
            heldout_prompts,
            train_path=args.train_prompts_file,
            heldout_path=args.heldout_prompts_file,
        )
        config.dataset.path = args.train_prompts_file
        config.dataset.prompts = []
        config.dataset.split_name = "train"
        config.dataset.content_sha256 = prompt_content_sha256(train_prompts)
        config.dataset.require_unique = True
        config.dataset.sampling_strategy = "deterministic_shuffle"
        config.dataset.sampling_seed = int(args.seed)
        config.evaluation.path = args.heldout_prompts_file
        config.evaluation.content_sha256 = prompt_content_sha256(heldout_prompts)
        config.evaluation.split_name = "heldout"
        config.evaluation.seeds = list(args.eval_seeds)
        config.evaluation.max_prompts = args.eval_max_prompts
    else:
        config.dataset.path = None
        config.dataset.prompts = [args.prompt]
        config.dataset.split_name = "train"
        config.dataset.content_sha256 = prompt_content_sha256([args.prompt])
        config.dataset.require_unique = True
        config.dataset.sampling_strategy = "sequential"
        config.dataset.sampling_seed = int(args.seed)
        config.evaluation.path = None
        config.evaluation.content_sha256 = None
        config.evaluation.seeds = list(args.eval_seeds)
        config.evaluation.max_prompts = 1
    config.dataset.repeat_per_prompt = 1
    config.sample.name = "branching"
    config.sample.batch_size = int(getattr(args, "sample_batch_size", 1))
    config.sample.num_steps = int(args.num_steps)
    config.sample.guidance_scale = float(args.guidance_scale)
    config.rollout = {
        **dict(config.rollout),
        "output_type": "pt",
        "branch_count": int(getattr(args, "branch_count", 2)),
        "exploration_k": int(getattr(args, "branch_count", 2)),
    }

    config.algorithm.name = "tempflow_grpo"
    config.algorithm.objective_version = (
        "reference_v1" if reference_mode else "policy_identity_v1"
    )
    allow_initial_clipping = bool(
        getattr(args, "allow_initial_clipping", False)
    )
    config.optimizer.params = {
        **dict(config.optimizer.params),
        "max_initial_logprob_delta": (
            None
            if allow_initial_clipping
            else float(getattr(args, "logprob_atol", 1e-5))
        ),
        "require_initial_clipfrac_zero": not allow_initial_clipping,
        "require_finite_gradients": True,
        "require_nonzero_gradients": True,
    }
    reward_name = getattr(args, "reward_name", None) or (
        "prompt_color_margin" if args.train_prompts_file else "prompt_color"
    )
    config.rewards.weights = {reward_name: 1.0}
    config.rewards.clients = {
        reward_name: {
            "name": reward_name,
            "version": "v1",
        }
    }
    config.rewards.fail_policy = "raise"

    config.train.max_steps = int(args.steps)
    config.train.save_every = int(args.steps)
    if getattr(args, "condition", "active") == "zero_lr_control":
        config.train.learning_rate = 0.0
    config.runner.strict_rollout_validation = True
    config.runner.disable_rollout_cache = bool(args.disable_rollout_cache)
    config.runner.deterministic_run_dir = True
    config.runner.deterministic_runtime = bool(
        getattr(args, "deterministic_runtime", False)
    )
    return config


def _comma_separated_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integer seeds"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def _parameter_snapshots(parameters: list[Any]) -> list[Any]:
    return [parameter.detach().float().cpu().clone() for parameter in parameters]


def _parameter_delta_summary(parameters: list[Any], snapshots: list[Any]) -> dict[str, float | int]:
    import math

    import torch

    max_abs = 0.0
    squared_sum = 0.0
    nonzero_count = 0
    for parameter, before in zip(parameters, snapshots, strict=True):
        delta = parameter.detach().float().cpu() - before
        if delta.numel() == 0:
            continue
        max_abs = max(max_abs, float(delta.abs().max().item()))
        squared_sum += float(torch.sum(delta * delta).item())
        nonzero_count += int(torch.count_nonzero(delta).item())
    return {
        "parameter_delta_abs_max": max_abs,
        "parameter_delta_l2": float(math.sqrt(squared_sum)),
        "parameter_delta_nonzero_count": nonzero_count,
    }


def _parameter_behavior_gates(
    delta_summary: dict[str, float | int],
    condition: str,
) -> tuple[bool, bool, bool]:
    updated = bool(
        int(delta_summary["parameter_delta_nonzero_count"]) > 0
        and float(delta_summary["parameter_delta_abs_max"]) > 0.0
    )
    unchanged = bool(
        int(delta_summary["parameter_delta_nonzero_count"]) == 0
        and float(delta_summary["parameter_delta_abs_max"]) == 0.0
        and float(delta_summary["parameter_delta_l2"]) == 0.0
    )
    expected_behavior = unchanged if condition == "zero_lr_control" else updated
    return updated, unchanged, expected_behavior


def _checkpoint_summary(paths: list[Path]) -> list[dict[str, Any]]:
    summary = []
    for path in paths:
        files = sorted(item for item in path.rglob("*") if item.is_file())
        summary.append(
            {
                "path": str(path),
                "file_count": len(files),
                "files": [str(item.relative_to(path)) for item in files[:20]],
            }
        )
    return summary


def _resume_checkpoint_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    summary = _checkpoint_summary([Path(path)])
    if not summary:
        return None
    item = summary[0]
    if item["file_count"] <= 0:
        raise RuntimeError(f"Resume checkpoint directory is empty: {path}")
    return item


def _checkpoint_step_from_path(path: str | None) -> int:
    if not path:
        return 0
    checkpoint_path = Path(path)
    if checkpoint_path.name == "latest.json" and checkpoint_path.is_file():
        return int(json.loads(checkpoint_path.read_text(encoding="utf-8"))["step"])
    if checkpoint_path.is_dir() and (checkpoint_path / "latest.json").is_file():
        return int(
            json.loads(
                (checkpoint_path / "latest.json").read_text(encoding="utf-8")
            )["step"]
        )
    if checkpoint_path.is_dir() and (checkpoint_path / "checkpoint.json").is_file():
        return int(
            json.loads(
                (checkpoint_path / "checkpoint.json").read_text(encoding="utf-8")
            )["step"]
        )
    match = re.search(r"checkpoint[_-](\d+)", checkpoint_path.name)
    return int(match.group(1)) if match else 0


_BOUNDED_TRAINER_REQUIRED_METRIC_KEYS = (
    "reward_mean",
    "reward_std",
    "approx_kl",
    "clipfrac",
    "old_logprob_mean",
    "new_logprob_mean",
    "logprob_delta_abs_max",
    "rollout_kl_mean",
    "tempflow_active_timestep_frac",
    "tempflow_noise_weight_mean",
    "grad_norm",
    "grad_nonzero_count",
    "gradients_finite",
)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in bounded trainer metrics at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Bounded trainer metrics line {line_number} is not a JSON object.")
            rows.append(row)
    return rows


def _validate_bounded_trainer_metrics(rows: list[dict[str, Any]], *, expected_steps: int) -> None:
    if len(rows) < expected_steps:
        raise RuntimeError(
            f"Bounded trainer smoke wrote {len(rows)} metrics row(s), expected at least {expected_steps}."
        )
    final = rows[-1]
    missing = [key for key in _BOUNDED_TRAINER_REQUIRED_METRIC_KEYS if key not in final]
    if missing:
        raise RuntimeError(f"Bounded trainer final metrics row is missing required key(s): {', '.join(missing)}.")
    import math

    non_finite = []
    for key in _BOUNDED_TRAINER_REQUIRED_METRIC_KEYS:
        value = final[key]
        if key == "gradients_finite":
            if value is not True:
                non_finite.append(key)
            continue
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
            non_finite.append(key)
    if non_finite:
        raise RuntimeError(f"Bounded trainer final metrics row has non-finite/non-numeric key(s): {', '.join(non_finite)}.")


def _validate_bounded_trainer_artifacts(
    metrics_path: Path,
    latest_path: Path,
    checkpoints: list[dict[str, Any]],
    *,
    expected_steps: int,
) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        raise RuntimeError(f"Bounded trainer smoke did not write metrics file: {metrics_path}")
    metrics_rows = _load_jsonl_rows(metrics_path)
    _validate_bounded_trainer_metrics(metrics_rows, expected_steps=expected_steps)
    if not latest_path.exists():
        raise RuntimeError(f"Bounded trainer smoke did not write latest checkpoint pointer: {latest_path}")
    if not checkpoints:
        raise RuntimeError("Bounded trainer smoke did not create any checkpoint_* directory.")
    if not any(item["file_count"] > 0 for item in checkpoints):
        raise RuntimeError("Bounded trainer smoke checkpoint directories are empty.")
    return metrics_rows


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _final_trainer_metric_extract(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    final = metrics[-1]
    tokens = ("reward", "logprob", "log_prob", "kl", "clip", "tempflow")
    return {key: value for key, value in final.items() if any(token in key.lower() for token in tokens)}


def _tensor_mean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        import torch

        tensor = torch.as_tensor(value).detach().float()
        if tensor.numel() == 0:
            return None
        return float(tensor.mean().cpu())
    except Exception:  # noqa: BLE001 - preview summaries should degrade to shape-only metadata
        return None


def _bounded_prompt_split_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    from visual_rl.datasets.prompt_dataset import (
        prompt_set_snapshot,
        read_prompt_file,
        validate_prompt_splits,
    )

    if not args.train_prompts_file:
        return {
            "mode": "single_prompt_smoke",
            "train": prompt_set_snapshot(
                [args.prompt],
                split_name="train",
            ),
            "heldout": None,
            "overlap_count": None,
        }
    snapshot = validate_prompt_splits(
        read_prompt_file(args.train_prompts_file),
        read_prompt_file(args.heldout_prompts_file),
        train_path=args.train_prompts_file,
        heldout_path=args.heldout_prompts_file,
    )
    snapshot["mode"] = "train_heldout"
    return snapshot


def _image_guardrail_summary(media: Any) -> dict[str, float]:
    import numpy as np

    image = _media_item_to_uint8_rgb(media, index=0).astype(np.float32) / 255.0
    spatial_std = float(image.reshape(-1, 3).std(axis=0).mean())
    dynamic_range = float(np.quantile(image, 0.95) - np.quantile(image, 0.05))
    saturation = float((image.max(axis=2) - image.min(axis=2)).mean())
    luminance_mean = float(image.mean())
    return {
        "spatial_std": spatial_std,
        "dynamic_range_90": dynamic_range,
        "saturation_mean": saturation,
        "luminance_mean": luminance_mean,
    }


def _aggregate_guardrail(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted(records[0]["image_guardrail"])
    return {
        key: float(
            sum(float(record["image_guardrail"][key]) for record in records)
            / len(records)
        )
        for key in keys
    }


def _bounded_heldout_summary(
    trainer: Any,
    args: argparse.Namespace,
    phase: str,
    output_dir: Path,
    *,
    milestone_step: int,
) -> dict[str, Any]:
    import numpy as np

    from visual_rl.datasets.prompt_dataset import prompt_id, read_prompt_file

    prompts = read_prompt_file(args.heldout_prompts_file)
    if args.eval_max_prompts is not None:
        prompts = prompts[: int(args.eval_max_prompts)]
    if not prompts:
        raise ValueError("held-out evaluation prompt set is empty")
    seeds = [int(seed) for seed in args.eval_seeds]
    preview_dir = output_dir / "previews" / phase
    preview_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    png_paths: list[Path] = []
    first_model_metadata: dict[str, Any] | None = None

    for prompt_index, prompt in enumerate(prompts):
        for seed in seeds:
            rollout_config = {
                "num_steps": int(args.num_steps),
                "guidance_scale": float(args.guidance_scale),
                "seed": seed,
                "output_type": "pt",
                "epoch_tag": milestone_step,
            }
            metadata = [
                {
                    "source": "sd3_bounded_heldout",
                    "phase": phase,
                    "split": "heldout",
                    "prompt_id": prompt_id(prompt),
                    "adapter_key": args.adapter,
                    "eval_seed": seed,
                }
            ]
            batch = trainer.adapter.sample([prompt], metadata, rollout_config)
            batch.validate_strict()
            rewards = trainer.feedback_provider.score(batch)
            if not bool(rewards.valid_mask.all()):
                raise RuntimeError(
                    f"Bounded trainer {phase} held-out reward failure: "
                    f"{rewards.metadata}"
                )
            record_index = len(records)
            png_path = preview_dir / f"preview_{record_index:03d}.png"
            _write_png_rgb(
                png_path,
                _media_item_to_uint8_rgb(batch.media, index=0),
            )
            png_paths.append(png_path)
            reward_value = float(rewards.weighted_total.detach().float().cpu()[0])
            reward_metadata = _json_safe(rewards.metadata)
            target_values = []
            if isinstance(reward_metadata, dict):
                for reward_name in (
                    "prompt_color_guarded",
                    "prompt_color_margin",
                    "prompt_color",
                ):
                    target_values = reward_metadata.get(reward_name, {}).get(
                        "targets",
                        [],
                    )
                    if target_values:
                        break
            records.append(
                {
                    "prompt": prompt,
                    "prompt_id": prompt_id(prompt),
                    "prompt_index": prompt_index,
                    "seed": seed,
                    "target_color": target_values[0] if target_values else None,
                    "reward": reward_value,
                    "png_path": str(png_path),
                    "old_logprob_mean": _tensor_mean_float(batch.old_log_probs),
                    "rollout_kl_mean": _tensor_mean_float(batch.kl),
                    "image_guardrail": _image_guardrail_summary(batch.media),
                }
            )
            if first_model_metadata is None:
                first_model_metadata = _json_safe(batch.model_metadata)

    scores = np.asarray([record["reward"] for record in records], dtype=np.float64)
    per_color = {}
    for color in sorted(
        {record["target_color"] for record in records if record["target_color"]}
    ):
        color_scores = [
            record["reward"]
            for record in records
            if record["target_color"] == color
        ]
        per_color[color] = {
            "count": len(color_scores),
            "reward_mean": float(np.mean(color_scores)),
            "reward_std": float(np.std(color_scores)),
        }

    metadata_path = preview_dir / "metadata.json"
    payload = {
        "phase": phase,
        "milestone_step": int(milestone_step),
        "split": "heldout",
        "preview_dir": str(preview_dir),
        "png_path": str(png_paths[0]),
        "png_paths": [str(path) for path in png_paths],
        "metadata_path": str(metadata_path),
        "prompt": prompts[0],
        "prompt_count": len(prompts),
        "sample_count": len(records),
        "seeds": seeds,
        "num_steps": int(args.num_steps),
        "guidance_scale": float(args.guidance_scale),
        "reward_mean": float(scores.mean()),
        "reward_std": float(scores.std()),
        "reward_min": float(scores.min()),
        "reward_max": float(scores.max()),
        "per_color": per_color,
        "image_guardrail": _aggregate_guardrail(records),
        "model_metadata": first_model_metadata or {},
        "records": records,
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def _paired_heldout_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    before_by_key = {
        (record["prompt_id"], int(record["seed"])): record
        for record in before.get("records", [])
    }
    after_by_key = {
        (record["prompt_id"], int(record["seed"])): record
        for record in after.get("records", [])
    }
    if set(before_by_key) != set(after_by_key) or not before_by_key:
        raise RuntimeError("held-out before/after panels are not paired")
    ordered_keys = sorted(before_by_key)
    delta_records = [
        {
            "prompt_id": key[0],
            "seed": key[1],
            "target_color": before_by_key[key].get("target_color"),
            "delta": float(after_by_key[key]["reward"])
            - float(before_by_key[key]["reward"]),
        }
        for key in ordered_keys
    ]
    deltas = np.asarray(
        [record["delta"] for record in delta_records],
        dtype=np.float64,
    )
    seed_values = sorted({record["seed"] for record in delta_records})
    seed_cluster_means = np.asarray(
        [
            np.mean(
                [
                    record["delta"]
                    for record in delta_records
                    if record["seed"] == seed
                ]
            )
            for seed in seed_values
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(0)
    bootstrap_indices = rng.integers(
        0,
        len(seed_cluster_means),
        size=(2000, len(seed_cluster_means)),
    )
    bootstrap_means = seed_cluster_means[bootstrap_indices].mean(axis=1)
    per_color = {}
    for color in sorted(
        {
            record["target_color"]
            for record in delta_records
            if record["target_color"]
        }
    ):
        color_records = [
            record
            for record in delta_records
            if record["target_color"] == color
        ]
        color_seed_values = sorted({record["seed"] for record in color_records})
        color_seed_means = np.asarray(
            [
                np.mean(
                    [
                        record["delta"]
                        for record in color_records
                        if record["seed"] == seed
                    ]
                )
                for seed in color_seed_values
            ],
            dtype=np.float64,
        )
        color_bootstrap_indices = rng.integers(
            0,
            len(color_seed_means),
            size=(2000, len(color_seed_means)),
        )
        color_bootstrap_means = color_seed_means[
            color_bootstrap_indices
        ].mean(axis=1)
        per_color[color] = {
            "paired_count": len(color_records),
            "eval_seed_cluster_count": len(color_seed_means),
            "eval_seed_cluster_means": {
                str(seed): float(value)
                for seed, value in zip(
                    color_seed_values,
                    color_seed_means,
                    strict=True,
                )
            },
            "reward_delta_mean": float(
                np.mean([record["delta"] for record in color_records])
            ),
            "reward_delta_ci95_low": float(
                np.quantile(color_bootstrap_means, 0.025)
            ),
            "reward_delta_ci95_high": float(
                np.quantile(color_bootstrap_means, 0.975)
            ),
        }
    guardrail_ratios = {}
    for key, before_value in before.get("image_guardrail", {}).items():
        after_value = float(after.get("image_guardrail", {}).get(key, 0.0))
        guardrail_ratios[key] = (
            None
            if float(before_value) == 0.0
            else after_value / float(before_value)
        )
    return {
        "paired_count": len(deltas),
        "ci_method": "bootstrap_over_eval_seed_cluster_means",
        "eval_seed_cluster_count": len(seed_cluster_means),
        "eval_seed_cluster_means": {
            str(seed): float(value)
            for seed, value in zip(
                seed_values,
                seed_cluster_means,
                strict=True,
            )
        },
        "reward_delta_mean": float(deltas.mean()),
        "reward_delta_std": float(deltas.std()),
        "reward_delta_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "reward_delta_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "reward_improved_fraction": float(np.mean(deltas > 0.0)),
        "per_color": per_color,
        "guardrail_after_over_before": guardrail_ratios,
    }


def _load_baseline_evaluation(
    args: argparse.Namespace,
    before: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_path = getattr(args, "baseline_eval", None)
    if baseline_path:
        source = Path(baseline_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        source = Path(before["metadata_path"])
        payload = before
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    snapshot = {
        "source_path": str(source),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "milestone_step": int(payload.get("milestone_step", 0)),
        "sample_count": int(payload.get("sample_count", 0)),
    }
    (output_dir / "baseline_evaluation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload, snapshot


def _training_trend_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    rewards = np.asarray(
        [float(row["reward_mean"]) for row in metrics],
        dtype=np.float64,
    )
    if rewards.size == 0:
        return {"step_count": 0}
    window = max(1, min(5, rewards.size // 2))
    slope = (
        0.0
        if rewards.size == 1
        else float(np.polyfit(np.arange(rewards.size), rewards, 1)[0])
    )
    return {
        "step_count": int(rewards.size),
        "reward_linear_slope_per_step": slope,
        "reward_first_window_mean": float(rewards[:window].mean()),
        "reward_last_window_mean": float(rewards[-window:].mean()),
        "reward_last_minus_first_window": float(
            rewards[-window:].mean() - rewards[:window].mean()
        ),
        "clipfrac_max": float(max(float(row["clipfrac"]) for row in metrics)),
        "logprob_delta_abs_max": float(
            max(float(row["logprob_delta_abs_max"]) for row in metrics)
        ),
        "grad_norm_min": float(min(float(row["grad_norm"]) for row in metrics)),
        "grad_norm_max": float(max(float(row["grad_norm"]) for row in metrics)),
        "all_gradients_finite": bool(
            all(bool(row["gradients_finite"]) for row in metrics)
        ),
    }


def _bounded_preview_summary(trainer: Any, args: argparse.Namespace, phase: str, output_dir: Path) -> dict[str, Any]:
    preview_dir = output_dir / "previews" / phase
    preview_dir.mkdir(parents=True, exist_ok=True)
    rollout_config = {
        "num_steps": int(args.num_steps),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
        "output_type": "pt",
        "epoch_tag": 0 if phase == "before" else int(args.steps),
    }
    metadata = [{"source": "sd3_bounded_trainer_smoke", "phase": phase, "adapter_key": args.adapter}]
    batch = trainer.adapter.sample([args.prompt], metadata, rollout_config)
    batch.validate_strict()
    rewards = trainer.feedback_provider.score(batch)
    if not rewards.valid_mask.all():
        raise RuntimeError(f"Bounded trainer {phase} preview reward failure: {rewards.metadata}")
    png_paths = _save_preview_pngs(batch.media, preview_dir, count=len(batch.prompts))
    metadata_path = preview_dir / "metadata.json"
    payload = {
        "phase": phase,
        "preview_dir": str(preview_dir),
        "png_path": str(png_paths[0]) if png_paths else None,
        "png_paths": [str(path) for path in png_paths],
        "metadata_path": str(metadata_path),
        "prompt": args.prompt,
        "prompt_count": len(batch.prompts),
        "seed": int(args.seed),
        "num_steps": int(args.num_steps),
        "guidance_scale": float(args.guidance_scale),
        "media_shape": _shape_list(batch.media),
        "latents_shape": _shape_list(batch.latents),
        "next_latents_shape": _shape_list(batch.next_latents),
        "timesteps_shape": _shape_list(batch.timesteps),
        "old_log_probs_shape": _shape_list(batch.old_log_probs),
        "kl_shape": _shape_list(batch.kl),
        "old_logprob_mean": _tensor_mean_float(batch.old_log_probs),
        "rollout_kl_mean": _tensor_mean_float(batch.kl),
        "reward_mean": _tensor_mean_float(rewards.weighted_total),
        "reward_names": sorted(rewards.raw),
        "reward_raw": {name: _tensor_json_summary(value) for name, value in rewards.raw.items()},
        "reward_weighted": {name: _tensor_json_summary(value) for name, value in rewards.weighted.items()},
        "reward_metadata": _json_safe(rewards.metadata),
        "model_metadata": _json_safe(batch.model_metadata),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _sd3_bounded_trainer_smoke_payload(args: argparse.Namespace) -> dict[str, Any]:
    _validate_sd3_bounded_trainer_args(args)
    _register_builtin_plugins()
    from visual_rl.artifacts.checkpoint import save_json
    from visual_rl.runner import ExperimentRunner

    config = _sd3_bounded_trainer_config(args)
    prompt_splits = _bounded_prompt_split_snapshot(args)
    _validate_config_registry_names(config)
    resume_base_step = _checkpoint_step_from_path(args.resume_from)
    steps_executed = int(args.steps) - resume_base_step
    trainer = ExperimentRunner(config)
    output_dir = Path(config.paths.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    latest_path = output_dir / "latest.json"
    summary_path = output_dir / "summary.json"

    parameters = list(trainer.adapter.parameters())
    snapshots = _parameter_snapshots(parameters)
    resume_summary = _resume_checkpoint_summary(args.resume_from)
    save_json(output_dir / "prompt_splits.json", prompt_splits)
    if args.heldout_prompts_file:
        before_preview = _bounded_heldout_summary(
            trainer,
            args,
            "before",
            output_dir,
            milestone_step=resume_base_step,
        )
    else:
        before_preview = _bounded_preview_summary(
            trainer,
            args,
            "before",
            output_dir,
        )
    metrics = trainer.run(max_steps=args.steps)
    if args.heldout_prompts_file:
        after_preview = _bounded_heldout_summary(
            trainer,
            args,
            "after",
            output_dir,
            milestone_step=int(args.steps),
        )
        baseline_preview, baseline_snapshot = _load_baseline_evaluation(
            args,
            before_preview,
            output_dir,
        )
        heldout_segment_delta = _paired_heldout_delta(
            before_preview,
            after_preview,
        )
        heldout_delta = _paired_heldout_delta(
            baseline_preview,
            after_preview,
        )
    else:
        after_preview = _bounded_preview_summary(
            trainer,
            args,
            "after",
            output_dir,
        )
        baseline_preview = before_preview
        baseline_snapshot = None
        heldout_segment_delta = None
        heldout_delta = None
    delta_summary = _parameter_delta_summary(parameters, snapshots)
    checkpoint_dirs = sorted(path for path in output_dir.glob("checkpoint_*") if path.is_dir())
    checkpoints = _checkpoint_summary(checkpoint_dirs)
    metrics_rows = _validate_bounded_trainer_artifacts(
        metrics_path,
        latest_path,
        checkpoints,
        expected_steps=steps_executed,
    )
    final_metrics = metrics[-1] if metrics else {}
    final_metrics_artifact = metrics_rows[-1] if metrics_rows else {}
    condition = str(getattr(args, "condition", "active"))
    (
        parameter_update_valid,
        parameter_unchanged_valid,
        parameter_behavior_valid,
    ) = _parameter_behavior_gates(
        delta_summary,
        condition,
    )
    training_trend = _training_trend_summary(metrics)
    allow_initial_clipping = bool(
        getattr(args, "allow_initial_clipping", False)
    )
    initial_policy_gate = bool(
        allow_initial_clipping
        or (
            float(training_trend.get("clipfrac_max", float("inf"))) == 0.0
            and float(
                training_trend.get("logprob_delta_abs_max", float("inf"))
            )
            <= float(args.logprob_atol)
        )
    )
    numerical_gate = bool(
        training_trend.get("all_gradients_finite", False)
        and float(training_trend.get("grad_norm_min", 0.0)) > 0.0
        and initial_policy_gate
    )
    heldout_contract_valid = bool(
        not args.heldout_prompts_file
        or (
            prompt_splits.get("overlap_count") == 0
            and heldout_delta is not None
            and int(heldout_delta["paired_count"])
            == int(before_preview["sample_count"])
        )
    )
    execution_valid = bool(
        parameter_behavior_valid and numerical_gate and heldout_contract_valid
    )
    guardrail_ratios = (
        {} if heldout_delta is None else heldout_delta["guardrail_after_over_before"]
    )
    pixel_guardrail_limits = {
        "spatial_std": (0.8, 1.25),
        "dynamic_range_90": (0.8, 1.25),
        "saturation_mean": (0.0, 1.15),
        "luminance_mean": (0.8, 1.15),
    }
    pixel_guardrail_passed = bool(
        not guardrail_ratios
        or all(
            ratio is not None
            and lower <= float(ratio) <= upper
            for key, (lower, upper) in pixel_guardrail_limits.items()
            for ratio in [guardrail_ratios.get(key)]
        )
    )
    independent_training_seed_count = 1
    if heldout_delta is None:
        reward_trend_gate = True
        reward_trend_rule = "not_applicable_single_prompt_smoke"
    elif int(args.steps) <= 5:
        reward_trend_gate = bool(
            float(heldout_delta["reward_delta_mean"]) >= -0.02
        )
        reward_trend_rule = "step5_mean_delta_at_least_minus_0.02"
    elif int(args.steps) < 50:
        reward_trend_gate = bool(
            float(heldout_delta["reward_delta_mean"]) > 0.0
        )
        reward_trend_rule = "step20_mean_delta_positive"
    else:
        reward_trend_gate = bool(
            float(heldout_delta["reward_delta_ci95_low"]) > 0.0
        )
        reward_trend_rule = "step50_eval_seed_cluster_ci95_low_positive"
    payload = {
        "valid": execution_valid,
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "latest_path": str(latest_path),
        "checkpoint_dirs": [str(path) for path in checkpoint_dirs],
        "checkpoint_summary": checkpoints,
        "latest": _load_json_if_present(latest_path),
        "resume_from": args.resume_from,
        "resume_checkpoint_summary": resume_summary,
        "source_checkpoint_summary": resume_summary,
        "resume_loaded": bool(args.resume_from),
        "resume_base_step": resume_base_step,
        "resume_steps": steps_executed if args.resume_from else 0,
        "steps_executed": steps_executed,
        "target_step": int(args.steps),
        "effective_total_step": int(args.steps),
        "steps": int(args.steps),
        "metrics_line_count": len(metrics_rows),
        "required_metric_keys": list(_BOUNDED_TRAINER_REQUIRED_METRIC_KEYS),
        "adapter": getattr(trainer.adapter, "name", args.adapter),
        "adapter_key": args.adapter,
        "model_path": args.model_path,
        "repo_root": args.repo_root,
        "prompt": args.prompt,
        "train_prompts_file": args.train_prompts_file,
        "heldout_prompts_file": args.heldout_prompts_file,
        "prompt_splits_path": str(output_dir / "prompt_splits.json"),
        "prompt_splits": prompt_splits,
        "resolution": int(args.resolution),
        "num_steps": int(args.num_steps),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
        "condition": condition,
        "learning_rate": float(config.train.learning_rate),
        "branch_count": int(getattr(args, "branch_count", 2)),
        "sample_batch_size": int(getattr(args, "sample_batch_size", 1)),
        "dataset_sampling": {
            "strategy": config.dataset.sampling_strategy,
            "seed": int(config.dataset.sampling_seed),
        },
        "independent_training_seed_count": independent_training_seed_count,
        "device": str(getattr(trainer.adapter, "device", args.device)),
        "dtype": str(getattr(trainer.adapter, "dtype", args.dtype)),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "max_sequence_length": int(args.max_sequence_length),
        "use_lora": not bool(args.disable_lora),
        "logprob_atol": float(getattr(args, "logprob_atol", 1e-5)),
        "allow_initial_clipping": allow_initial_clipping,
        "tempflow_execution_mode": str(
            getattr(args, "tempflow_execution_mode", "policy-identity")
        ),
        "tempflow_reference_mode": bool(
            getattr(trainer.adapter, "tempflow_reference_mode", False)
        ),
        "strict_rollout_validation": config.runner.strict_rollout_validation,
        "rollout_cache_disabled": config.runner.disable_rollout_cache,
        "rollout_cache_path": None if config.runner.disable_rollout_cache else str(output_dir / "rollouts"),
        "metrics": _json_safe(metrics),
        "final_metrics": _json_safe(final_metrics),
        "final_metric_extract": _json_safe(_final_trainer_metric_extract(metrics)),
        "final_metrics_artifact": _json_safe(final_metrics_artifact),
        "final_metric_artifact_extract": _json_safe(_final_trainer_metric_extract(metrics_rows)),
        "preview_artifacts": {
            "before": before_preview,
            "after": after_preview,
        },
        "heldout_paired_delta": heldout_delta,
        "heldout_segment_delta": heldout_segment_delta,
        "baseline_evaluation": baseline_snapshot,
        "training_trend": training_trend,
        "gates": {
            "parameter_update": parameter_update_valid,
            "parameter_unchanged": parameter_unchanged_valid,
            "parameter_behavior": parameter_behavior_valid,
            "parameter_behavior_rule": (
                "parameters_must_remain_bitwise_unchanged"
                if condition == "zero_lr_control"
                else "at_least_one_trainable_parameter_must_change"
            ),
            "numerical": numerical_gate,
            "initial_policy": initial_policy_gate,
            "initial_policy_rule": (
                "reference_compatible_initial_clipping_explicitly_allowed"
                if allow_initial_clipping
                else "initial_clipfrac_zero_and_logprob_within_atol"
            ),
            "heldout_contract": heldout_contract_valid,
            "pixel_diversity_guardrail": pixel_guardrail_passed,
            "pixel_guardrail_limits": {
                key: {"min_ratio": lower, "max_ratio": upper}
                for key, (lower, upper) in pixel_guardrail_limits.items()
            },
            "reward_trend": reward_trend_gate,
            "reward_trend_rule": reward_trend_rule,
            "eligible_for_next_milestone": bool(
                condition == "active"
                and execution_valid
                and pixel_guardrail_passed
                and reward_trend_gate
            ),
            "eligible_for_100_steps": bool(
                condition == "active"
                and execution_valid
                and pixel_guardrail_passed
                and int(args.steps) >= 50
                and heldout_delta is not None
                and float(heldout_delta["reward_delta_ci95_low"]) > 0.0
                and reward_trend_gate
                and independent_training_seed_count >= 3
            ),
            "eligible_for_100_steps_requires": [
                "50-step paired held-out CI95 lower bound > 0",
                "pixel diversity guardrail passes",
                "at least 3 independent training seeds",
            ],
        },
        "trainable_parameter_tensors": len(parameters),
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        **delta_summary,
    }
    save_json(summary_path, payload)
    return payload


def sd3_bounded_trainer_smoke(args: argparse.Namespace) -> int:
    payload = _sd3_bounded_trainer_smoke_payload(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["valid"]:
        raise RuntimeError(
            "SD3 bounded trainer failed execution gates: "
            + json.dumps(payload.get("gates", {}), sort_keys=True)
        )
    return 0


def _sd3_image_numeric_smoke_payload(args: argparse.Namespace) -> dict:
    _register_builtin_plugins()
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.core.seed import seed_everything

    adapter_key = "sd3_tempflow"
    seed_everything(int(args.seed))
    spec = _TEMPFLOW_IMAGE_SMOKE_SPECS[adapter_key]
    model_config = _tempflow_image_model_config(args, adapter_key)
    adapter = MODEL_ADAPTERS.get(adapter_key)(model_config)
    rollout_config = {
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }
    batch = adapter.sample([args.prompt], [{"source": spec["metadata_source"], "adapter_key": adapter_key}], rollout_config)
    batch.validate_strict()
    recomputed = adapter.recompute_log_probs(batch)

    import torch

    old_log_probs = batch.old_log_probs.detach()
    recomputed_log_probs = recomputed.detach()
    if recomputed_log_probs.shape != old_log_probs.shape:
        raise ValueError(
            f"{spec['label']} recomputed logprobs shape diverged from sampled logprobs: "
            f"{tuple(recomputed_log_probs.shape)} != {tuple(old_log_probs.shape)}"
        )
    max_abs_logprob_delta = float((recomputed_log_probs - old_log_probs).abs().max().item())
    params = list(adapter.parameters())
    trainable_parameters = int(sum(parameter.numel() for parameter in params))
    model_metadata = dict(batch.model_metadata)
    reference_repo = model_metadata.get("reference_repo", args.repo_root)
    payload = {
        "adapter": adapter.name,
        "adapter_key": adapter_key,
        "model_family": spec["model_family"],
        "model_path": args.model_path,
        "repo_root": args.repo_root,
        "reference_repo": reference_repo,
        "prompt": args.prompt,
        "resolution": args.resolution,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "max_sequence_length": getattr(args, "max_sequence_length", None),
        "media_shape": _shape_list(batch.media),
        "latents_shape": _shape_list(batch.latents),
        "next_latents_shape": _shape_list(batch.next_latents),
        "timesteps_shape": _shape_list(batch.timesteps),
        "old_log_probs_shape": _shape_list(old_log_probs),
        "recomputed_log_probs_shape": _shape_list(recomputed_log_probs),
        "media_finite": _tensor_finite(batch.media),
        "old_log_probs_finite": _tensor_finite(old_log_probs),
        "recomputed_log_probs_finite": _tensor_finite(recomputed_log_probs),
        "max_abs_logprob_delta": max_abs_logprob_delta,
        "trainable_parameter_tensors": len(params),
        "trainable_parameters": trainable_parameters,
        "device": str(getattr(adapter, "device", args.device)),
        "dtype": str(getattr(adapter, "dtype", args.dtype)),
        "model_metadata": model_metadata,
        "shapes": {
            "media": _shape_list(batch.media),
            "latents": _shape_list(batch.latents),
            "next_latents": _shape_list(batch.next_latents),
            "timesteps": _shape_list(batch.timesteps),
            "old_log_probs": _shape_list(old_log_probs),
            "recomputed_log_probs": _shape_list(recomputed_log_probs),
            "kl": _shape_list(batch.kl),
            "branch_ids": _shape_list(batch.branch_ids),
        },
    }
    if not payload["media_finite"] or not payload["old_log_probs_finite"] or not payload["recomputed_log_probs_finite"]:
        raise ValueError(f"{spec['label']} numeric smoke produced non-finite tensors.")
    if not torch.allclose(recomputed_log_probs, old_log_probs, atol=args.logprob_atol, rtol=0.0):
        raise ValueError(
            f"{spec['label']} recomputed logprobs diverged from sampled logprobs: "
            f"max_abs_delta={max_abs_logprob_delta:.6g}, atol={args.logprob_atol:.6g}"
        )
    return payload


def _sd3_numeric_smoke_payload(args: argparse.Namespace) -> dict:
    return _sd3_image_numeric_smoke_payload(args)


def sd3_numeric_smoke(args: argparse.Namespace) -> int:
    payload = _sd3_numeric_smoke_payload(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _branch_step_index_arg(value: str) -> int | str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("branch step index must be a non-negative integer or 'auto'") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("branch step index must be a non-negative integer or 'auto'")
    return parsed


def _branch_ids_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    except AttributeError:
        return [int(item) for item in value]


def _branching_transition_contract(batch: Any) -> dict[str, Any]:
    metadata_keys = sorted({str(key) for item in batch.metadata for key in item})
    return {
        "model_metadata": _json_safe(batch.model_metadata),
        "sample_metadata": _json_safe(batch.metadata),
        "metadata_keys": metadata_keys,
        "branch_ids": _branch_ids_list(batch.branch_ids),
        "parent_prompt_indices": [item.get("parent_prompt_index") for item in batch.metadata],
        "branch_step_indices": [item.get("branch_step_index") for item in batch.metadata],
        "branch_timestep_values": [item.get("branch_timestep_value") for item in batch.metadata],
    }


def _sd3_branching_numeric_smoke_payload(args: argparse.Namespace) -> dict[str, Any]:
    _register_builtin_plugins()
    from visual_rl.core.registry import MODEL_ADAPTERS
    from visual_rl.core.seed import seed_everything
    from visual_rl.rollout.full_trajectory import build_rollout_engine

    import torch

    if int(args.branch_count) < 2:
        raise ValueError(f"sd3-branching-numeric-smoke requires --branch-count >= 2, got {args.branch_count}.")
    if float(args.logprob_atol) < 0:
        raise ValueError(f"sd3-branching-numeric-smoke requires --logprob-atol >= 0, got {args.logprob_atol}.")
    if float(args.clip_range) <= 0:
        raise ValueError(f"sd3-branching-numeric-smoke requires --clip-range > 0, got {args.clip_range}.")

    adapter_key = "sd3_tempflow"
    seed_everything(int(args.seed))
    spec = _TEMPFLOW_IMAGE_SMOKE_SPECS[adapter_key]
    model_config = _tempflow_image_model_config(args, adapter_key)
    adapter = MODEL_ADAPTERS.get(adapter_key)(model_config)
    base_rollout_config = {
        "name": "branching",
        "num_steps": int(args.num_steps),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
        "output_type": "pt",
        "branch_count": int(args.branch_count),
        "exploration_k": int(args.branch_count),
        "include_main": False,
        "branch_timestep_strategy": "cycle",
        "epoch_tag": 0,
    }
    transition_counter = getattr(adapter, "branch_transition_count", None)
    transition_count = int(
        transition_counter(base_rollout_config)
        if callable(transition_counter)
        else base_rollout_config["num_steps"]
    )
    if transition_count < 1:
        raise ValueError("sd3-branching-numeric-smoke requires at least one branch transition.")

    requested_step = args.branch_step_index
    if requested_step == "auto":
        transition_indices = list(range(transition_count))
    else:
        requested_step = int(requested_step)
        if requested_step >= transition_count:
            raise ValueError(
                "sd3-branching-numeric-smoke --branch-step-index is out of range: "
                f"got {requested_step}, valid range is [0, {transition_count - 1}]."
            )
        transition_indices = [requested_step]

    per_transition = []
    old_log_probs_by_transition = []
    recomputed_log_probs_by_transition = []
    policy_sha256_before = _trainable_parameter_sha256(adapter)
    for transition_index in transition_indices:
        rollout_config = {
            **base_rollout_config,
            "branch_timesteps": [transition_index],
        }
        rollout = build_rollout_engine(rollout_config)
        batch = rollout.sample(
            adapter,
            [args.prompt],
            [
                {
                    "source": "sd3_branching_numeric_smoke",
                    "adapter_key": adapter_key,
                }
            ],
        )
        batch.validate_strict()
        old_log_probs = batch.old_log_probs.detach()
        recomputed_log_probs = adapter.recompute_log_probs(batch).detach()
        if recomputed_log_probs.shape != old_log_probs.shape:
            raise ValueError(
                f"{spec['label']} branching transition {transition_index} recomputed logprobs shape diverged "
                f"from sampled logprobs: {tuple(recomputed_log_probs.shape)} != {tuple(old_log_probs.shape)}"
            )

        logprob_delta = recomputed_log_probs - old_log_probs
        ratio = torch.exp(logprob_delta)
        finite = {
            "media": _tensor_finite(batch.media),
            "latents": _tensor_finite(batch.latents),
            "next_latents": _tensor_finite(batch.next_latents),
            "timesteps": _tensor_finite(batch.timesteps),
            "old_log_probs": _tensor_finite(old_log_probs),
            "recomputed_log_probs": _tensor_finite(recomputed_log_probs),
            "logprob_delta": _tensor_finite(logprob_delta),
            "kl": batch.kl is None or _tensor_finite(batch.kl),
            "branch_ids": batch.branch_ids is None or _tensor_finite(batch.branch_ids),
        }
        max_abs_delta = float(logprob_delta.abs().max().item())
        clipfrac = float(((ratio - 1.0).abs() > float(args.clip_range)).float().mean().item())
        within_atol = bool(
            all(finite.values())
            and torch.allclose(
                recomputed_log_probs,
                old_log_probs,
                atol=float(args.logprob_atol),
                rtol=0.0,
            )
        )
        per_transition.append(
            {
                "transition_index": transition_index,
                "branch_step_index": transition_index,
                "within_atol": within_atol,
                "max_abs_logprob_delta": max_abs_delta,
                "clipfrac": clipfrac,
                "finite": finite,
                "shapes": {
                    "media": _shape_list(batch.media),
                    "latents": _shape_list(batch.latents),
                    "next_latents": _shape_list(batch.next_latents),
                    "timesteps": _shape_list(batch.timesteps),
                    "old_log_probs": _shape_list(old_log_probs),
                    "recomputed_log_probs": _shape_list(recomputed_log_probs),
                    "logprob_delta": _shape_list(logprob_delta),
                    "kl": _shape_list(batch.kl),
                    "branch_ids": _shape_list(batch.branch_ids),
                },
                "contract_metadata": _branching_transition_contract(batch),
                "initial_latent_sha256": _tensor_sha256(
                    batch.model_tensors.get("initial_latents", batch.latents[:, :1])
                ),
                "scheduler_timesteps_sha256": _tensor_sha256(
                    batch.model_tensors.get("scheduler_timesteps", batch.timesteps)
                ),
                "scheduler_sigmas_sha256": (
                    _tensor_sha256(batch.model_tensors["scheduler_sigmas"])
                    if "scheduler_sigmas" in batch.model_tensors
                    else None
                ),
            }
        )
        old_log_probs_by_transition.append(old_log_probs)
        recomputed_log_probs_by_transition.append(recomputed_log_probs)

    stacked_old_log_probs = torch.stack(old_log_probs_by_transition, dim=0)
    stacked_recomputed_log_probs = torch.stack(recomputed_log_probs_by_transition, dim=0)
    stacked_delta = stacked_recomputed_log_probs - stacked_old_log_probs
    ratio = torch.exp(stacked_delta)
    overall_old_finite = _tensor_finite(stacked_old_log_probs)
    overall_recomputed_finite = _tensor_finite(stacked_recomputed_log_probs)
    overall_delta_finite = _tensor_finite(stacked_delta)
    overall_max_abs_delta = float(stacked_delta.abs().max().item())
    overall_clipfrac = float(((ratio - 1.0).abs() > float(args.clip_range)).float().mean().item())
    failed_transition_indices = [
        item["transition_index"] for item in per_transition if not item["within_atol"]
    ]
    policy_sha256_after = _trainable_parameter_sha256(adapter)
    policy_unchanged = policy_sha256_before == policy_sha256_after
    replay_fingerprint_fields = (
        "initial_latent_sha256",
        "scheduler_timesteps_sha256",
        "scheduler_sigmas_sha256",
    )
    replay_fingerprint_unique_counts = {
        field: len({item[field] for item in per_transition})
        for field in replay_fingerprint_fields
    }
    transformer_training_states = {
        item["contract_metadata"]["model_metadata"].get(
            "transformer_training"
        )
        for item in per_transition
    }
    replay_fingerprints_consistent = bool(
        all(count == 1 for count in replay_fingerprint_unique_counts.values())
        and len(transformer_training_states) == 1
    )
    params = list(adapter.parameters())
    model_metadata = [item["contract_metadata"]["model_metadata"] for item in per_transition]
    reference_repo = model_metadata[0].get("reference_repo", args.repo_root)
    rollout_tensors_finite = all(
        all(item["finite"].values()) for item in per_transition
    )
    payload = {
        "valid": (
            not failed_transition_indices
            and policy_unchanged
            and replay_fingerprints_consistent
        ),
        "adapter": adapter.name,
        "adapter_key": adapter_key,
        "model_family": spec["model_family"],
        "model_path": args.model_path,
        "repo_root": args.repo_root,
        "reference_repo": reference_repo,
        "prompt": args.prompt,
        "resolution": int(args.resolution),
        "num_steps": int(args.num_steps),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
        "max_sequence_length": getattr(args, "max_sequence_length", None),
        "branch_count": int(args.branch_count),
        "branch_step_index": args.branch_step_index,
        "sampling_mode": (
            "same_seed_replay_per_transition"
            if len(transition_indices) > 1
            else "single_transition_rollout"
        ),
        "transition_count": transition_count,
        "evaluated_transition_count": len(transition_indices),
        "evaluated_transition_indices": transition_indices,
        "failed_transition_indices": failed_transition_indices,
        "logprob_atol": float(args.logprob_atol),
        "clip_range": float(args.clip_range),
        "max_abs_logprob_delta": overall_max_abs_delta,
        "overall_max_abs_logprob_delta": overall_max_abs_delta,
        "clipfrac": overall_clipfrac,
        "overall_clipfrac": overall_clipfrac,
        "old_log_probs_finite": overall_old_finite,
        "recomputed_log_probs_finite": overall_recomputed_finite,
        "logprob_delta_finite": overall_delta_finite,
        "media_finite": all(item["finite"]["media"] for item in per_transition),
        "rollout_tensors_finite": rollout_tensors_finite,
        "shapes": {
            "old_log_probs": _shape_list(stacked_old_log_probs),
            "recomputed_log_probs": _shape_list(stacked_recomputed_log_probs),
            "logprob_delta": _shape_list(stacked_delta),
        },
        "per_transition": per_transition,
        "trainable_parameter_tensors": len(params),
        "trainable_parameters": int(sum(parameter.numel() for parameter in params)),
        "trainable_parameter_sha256_before": policy_sha256_before,
        "trainable_parameter_sha256_after": policy_sha256_after,
        "trainable_parameters_unchanged": policy_unchanged,
        "replay_fingerprints_consistent": replay_fingerprints_consistent,
        "replay_fingerprint_unique_counts": replay_fingerprint_unique_counts,
        "transformer_training_states": sorted(
            str(value) for value in transformer_training_states
        ),
        "device": str(getattr(adapter, "device", args.device)),
        "dtype": str(getattr(adapter, "dtype", args.dtype)),
        "contract_metadata": {
            "rollout": "branching",
            "branching_mode": "shared_prefix",
            "branch_count": int(args.branch_count),
            "include_main": False,
            "requested_branch_step_index": args.branch_step_index,
            "transition_count": transition_count,
            "evaluated_transition_indices": transition_indices,
            "strict_rollout_validation": True,
            "logprob_atol": float(args.logprob_atol),
            "clip_range": float(args.clip_range),
            "model_metadata_by_transition": model_metadata,
            "sampling_mode": (
                "same_seed_replay_per_transition"
                if len(transition_indices) > 1
                else "single_transition_rollout"
            ),
        },
    }
    return payload


def sd3_branching_numeric_smoke(args: argparse.Namespace) -> int:
    payload = _sd3_branching_numeric_smoke_payload(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["valid"]:
        if not payload["trainable_parameters_unchanged"]:
            raise ValueError(
                "SD3 branching parity smoke changed trainable parameters"
            )
        if not payload["replay_fingerprints_consistent"]:
            raise ValueError(
                "SD3 branching same-seed replay fingerprints diverged"
            )
        failed = ", ".join(str(item) for item in payload["failed_transition_indices"])
        raise ValueError(
            "SD3 branching recomputed logprobs diverged from sampled logprobs for transition(s) "
            f"{failed}: max_abs_delta={payload['max_abs_logprob_delta']:.6g}, "
            f"atol={payload['logprob_atol']:.6g}"
        )
    return 0


def tiny_loss_probe(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from scripts.loss_probe import TinyLossProbeConfig, run_tiny_loss_probe

    config = TinyLossProbeConfig(
        output_dir=args.output_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        image_size=args.image_size,
        seed=args.seed,
        device=args.device,
        target_bias=tuple(float(item) for item in args.target_bias),
        max_final_loss_ratio=args.max_final_loss_ratio,
        max_final_bias_error_ratio=args.max_final_bias_error_ratio,
        assert_descent=not args.no_assert_descent,
    )
    summary = run_tiny_loss_probe(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def checkpoint_inventory(args: argparse.Namespace) -> int:
    from scripts.checkpoint_inventory import build_checkpoint_inventory

    payload = build_checkpoint_inventory(
        args.roots,
        required_adapters=args.require_adapter,
        max_depth=args.max_depth,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


def world_r1_reward_server_probe(args: argparse.Namespace) -> int:
    from scripts.world_r1_reward_probe import (
        WorldR1RewardServerProbeConfig,
        run_world_r1_reward_server_probe,
    )
    from visual_rl.feedback.clients import redact_error_text, redact_url

    config = WorldR1RewardServerProbeConfig(
        reward=args.reward,
        url=args.url,
        timeout=args.timeout,
        retries=args.retries,
        batch_size=args.batch_size,
        frames=args.frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
        prompt=args.prompt,
        protocol_mode=args.protocol_mode,
        wire_format=args.wire_format,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        trusted_hosts=tuple(args.trusted_host),
        max_response_bytes=args.max_response_bytes,
        execute_http=args.http,
    )
    try:
        payload = run_world_r1_reward_server_probe(config)
    except Exception as exc:  # noqa: BLE001 - CLI emits structured probe errors
        payload = {
            "valid": False,
            "reward": args.reward,
            "url": redact_url(args.url),
            "errors": [redact_error_text(exc)],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def wan_checkpoint_probe(args: argparse.Namespace) -> int:
    from scripts.wan_checkpoint_probe import WanCheckpointProbeConfig, run_wan_checkpoint_probe

    config = WanCheckpointProbeConfig(
        model_path=args.model_path,
        repo_root=args.repo_root,
        torch_dtype=args.torch_dtype,
        device=args.device,
        local_files_only=not args.allow_download,
        low_cpu_mem_usage=not args.no_low_cpu_mem_usage,
        manifest_only=args.manifest_only,
    )
    try:
        payload = run_wan_checkpoint_probe(config)
    except Exception as exc:  # noqa: BLE001 - CLI emits structured probe errors
        payload = {
            "valid": False,
            "model_path": args.model_path,
            "repo_root": args.repo_root,
            "errors": [str(exc)],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


def wan_plan(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from scripts.wan_runtime_plan import WanRuntimePlanner

    default_config = _SCRIPT_CONFIG_DIR / "wan_runtime_plan.yaml"
    config = load_config(args.config or default_config)
    if args.model_path is not None:
        config.model.model_path = args.model_path
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    planner = WanRuntimePlanner(config)
    print(json.dumps(planner.build_runtime_plan().to_dict(), indent=2, sort_keys=True))
    return 0


def world_r1_plan(args: argparse.Namespace) -> int:
    from scripts.world_r1_launcher import build_world_r1_launch_plan

    plan = build_world_r1_launch_plan(
        model_path=args.model_path,
        repo_dir=args.repo_dir,
        train_visible_devices=args.gpus,
        output_root=args.output_root,
        smoke=not args.full,
    )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def remote_sd3_cli_smoke(args: argparse.Namespace) -> int:
    from scripts.remote_smoke import RemoteSd3CliSmokeConfig, dumps_payload, run_remote_sd3_cli_smoke

    stage_name = args.stage_name or RemoteSd3CliSmokeConfig().stage_name
    config = RemoteSd3CliSmokeConfig(
        server=args.server,
        remote_root=args.remote_root,
        gpu=args.gpu,
        model_path=args.model_path,
        repo_root=args.repo_root,
        conda_env=args.conda_env,
        conda_bin=args.conda_bin,
        resolution=args.resolution,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        dtype=args.dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_sequence_length=args.max_sequence_length,
        branch_count=args.branch_count,
        sample_batch_size=args.sample_batch_size,
        condition=args.condition,
        tempflow_execution_mode=args.tempflow_execution_mode,
        allow_initial_clipping=args.allow_initial_clipping,
        logprob_atol=args.logprob_atol,
        bounded_steps=args.bounded_steps,
        resume_steps=args.resume_steps,
        idle_memory_mb=args.idle_memory_mb,
        idle_util_pct=args.idle_util_pct,
        stage_name=stage_name,
        prompt=args.prompt,
        train_prompts_file=args.train_prompts_file,
        heldout_prompts_file=args.heldout_prompts_file,
        baseline_eval=args.baseline_eval,
        eval_seeds=list(args.eval_seeds),
        eval_max_prompts=args.eval_max_prompts,
        run_bounded_trainer=args.run_bounded_trainer,
        run_resume_validation=args.run_resume_validation,
        allow_long_run=args.allow_long_run,
        allow_large_run=args.allow_large_run,
        dry_run=args.dry_run,
    )
    payload = run_remote_sd3_cli_smoke(config)
    print(dumps_payload(payload))
    return 0 if payload.get("dry_run", False) or payload.get("ok", False) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="visual-rl-tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    imports_parser = subparsers.add_parser("smoke-imports")
    imports_parser.set_defaults(func=smoke_imports)

    mock_parser = subparsers.add_parser("smoke-mock")
    mock_parser.add_argument("--config", default=None)
    mock_parser.add_argument("--output-dir", default=None)
    mock_parser.add_argument("--steps", type=int, default=2)
    mock_parser.set_defaults(func=smoke_mock)

    tempflow_parser = subparsers.add_parser("tempflow-smoke")
    tempflow_parser.add_argument("--config", default=None)
    tempflow_parser.add_argument("--output-dir", default=None)
    tempflow_parser.add_argument("--steps", type=int, default=2)
    tempflow_parser.set_defaults(func=tempflow_smoke)

    flash_parser = subparsers.add_parser("flash-smoke")
    flash_parser.add_argument("--config", default=None)
    flash_parser.add_argument("--output-dir", default=None)
    flash_parser.add_argument("--steps", type=int, default=2)
    flash_parser.set_defaults(func=flash_smoke)

    preview_parser = subparsers.add_parser("image-preview")
    preview_parser.add_argument(
        "--adapter",
        choices=sorted(_IMAGE_PREVIEW_ADAPTERS),
        required=True,
    )
    preview_parser.add_argument("--model-path", required=True)
    preview_parser.add_argument("--repo-root", default=None)
    preview_parser.add_argument("--prompt", default="a red square")
    preview_parser.add_argument("--resolution", type=int, default=256)
    preview_parser.add_argument("--num-steps", type=int, default=3)
    preview_parser.add_argument("--guidance-scale", type=float, default=4.5)
    preview_parser.add_argument("--seed", type=int, default=23)
    preview_parser.add_argument("--device", default=None)
    preview_parser.add_argument("--output-dir", required=True)
    preview_parser.set_defaults(func=image_preview)

    panel_preview_parser = subparsers.add_parser("image-panel-preview")
    panel_preview_parser.add_argument(
        "--adapter",
        choices=sorted(_IMAGE_PREVIEW_ADAPTERS),
        default="sd3_tempflow",
    )
    panel_preview_parser.add_argument("--model-path", required=True)
    panel_preview_parser.add_argument("--repo-root", default=None)
    panel_preview_parser.add_argument("--prompts-file", required=True)
    panel_preview_parser.add_argument("--max-prompts", type=int, default=None)
    panel_preview_parser.add_argument(
        "--num-steps-list",
        type=_comma_separated_ints,
        default=[3, 12, 20, 28],
    )
    panel_preview_parser.add_argument(
        "--seeds",
        type=_comma_separated_ints,
        default=[1701],
    )
    panel_preview_parser.add_argument("--resolution", type=int, default=256)
    panel_preview_parser.add_argument("--guidance-scale", type=float, default=4.5)
    panel_preview_parser.add_argument("--device", default=None)
    panel_preview_parser.add_argument("--dtype", default="bfloat16")
    panel_preview_parser.add_argument("--max-sequence-length", type=int, default=128)
    panel_preview_parser.add_argument("--output-dir", required=True)
    panel_preview_parser.set_defaults(func=image_panel_preview)

    sd3_trainer_parser = subparsers.add_parser("sd3-bounded-trainer-smoke")
    sd3_trainer_parser.add_argument("--adapter", choices=["sd3_tempflow"], default="sd3_tempflow")
    sd3_trainer_parser.add_argument("--model-path", required=True)
    sd3_trainer_parser.add_argument("--repo-root", required=True)
    sd3_trainer_parser.add_argument("--prompt", default="a small red cube on a white table")
    sd3_trainer_parser.add_argument("--train-prompts-file", default=None)
    sd3_trainer_parser.add_argument("--heldout-prompts-file", default=None)
    sd3_trainer_parser.add_argument(
        "--reward-name",
        choices=[
            "prompt_color",
            "prompt_color_margin",
            "prompt_color_guarded",
        ],
        default=None,
    )
    sd3_trainer_parser.add_argument("--baseline-eval", default=None)
    sd3_trainer_parser.add_argument(
        "--eval-seeds",
        type=_comma_separated_ints,
        default=[1701, 1702, 1703],
    )
    sd3_trainer_parser.add_argument("--eval-max-prompts", type=int, default=None)
    sd3_trainer_parser.add_argument("--sample-batch-size", type=int, default=1)
    sd3_trainer_parser.add_argument("--branch-count", type=int, default=2)
    sd3_trainer_parser.add_argument(
        "--condition",
        choices=["active", "zero_lr_control"],
        default="active",
    )
    sd3_trainer_parser.add_argument("--resolution", type=int, default=256)
    sd3_trainer_parser.add_argument("--num-steps", type=int, default=3)
    sd3_trainer_parser.add_argument("--guidance-scale", type=float, default=4.5)
    sd3_trainer_parser.add_argument("--seed", type=int, default=101)
    sd3_trainer_parser.add_argument("--device", default=None)
    sd3_trainer_parser.add_argument("--dtype", default="bfloat16")
    sd3_trainer_parser.add_argument("--lora-rank", type=int, default=8)
    sd3_trainer_parser.add_argument("--lora-alpha", type=int, default=16)
    sd3_trainer_parser.add_argument("--max-sequence-length", type=int, default=128)
    sd3_trainer_parser.add_argument("--logprob-atol", type=float, default=1e-5)
    sd3_trainer_parser.add_argument(
        "--tempflow-execution-mode",
        choices=["reference-compatible", "policy-identity"],
        default="policy-identity",
        help=(
            "Use strict shared-prefix policy identity by default, or explicitly "
            "select upstream-compatible six-branch recompute for parity audits."
        ),
    )
    sd3_trainer_parser.add_argument(
        "--allow-initial-clipping",
        action="store_true",
        help=(
            "Allow the known reference-compatible initial drift only for an "
            "explicit parity audit; policy-identity mode always rejects it."
        ),
    )
    sd3_trainer_parser.add_argument("--steps", type=int, default=1)
    sd3_trainer_parser.add_argument("--output-dir", required=True)
    sd3_trainer_parser.add_argument("--resume-from", default=None)
    sd3_trainer_parser.add_argument("--allow-long-run", action="store_true")
    sd3_trainer_parser.add_argument("--allow-large-run", action="store_true")
    sd3_trainer_parser.add_argument(
        "--deterministic-runtime",
        action="store_true",
        help=(
            "Enable the validated exact-reproducibility runtime. "
            "PYTHONHASHSEED must equal --seed before process start."
        ),
    )
    sd3_trainer_parser.add_argument("--disable-lora", action="store_true")
    sd3_trainer_parser.add_argument("--disable-rollout-cache", dest="disable_rollout_cache", action="store_true", default=True)
    sd3_trainer_parser.add_argument("--enable-rollout-cache", dest="disable_rollout_cache", action="store_false")
    sd3_trainer_parser.set_defaults(func=sd3_bounded_trainer_smoke)

    adapter_parser = subparsers.add_parser("adapter-probe")
    adapter_parser.add_argument("--config", default=None)
    adapter_parser.add_argument("--adapter", default="sd3_tempflow")
    adapter_parser.add_argument("--model-path", default=None)
    adapter_parser.add_argument("--device", default=None)
    adapter_parser.add_argument("--load", action="store_true")
    adapter_parser.set_defaults(func=adapter_probe)

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("configs", nargs="+")
    validate_parser.set_defaults(func=validate_config)

    rollout_probe_parser = subparsers.add_parser("rollout-probe")
    rollout_probe_parser.add_argument("config")
    rollout_probe_parser.add_argument("--batch-size", type=int, default=None)
    rollout_probe_parser.add_argument("--num-steps", type=int, default=None)
    rollout_probe_parser.add_argument("--seed", type=int, default=None)
    rollout_probe_parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    rollout_probe_parser.add_argument("--no-strict", dest="strict", action="store_false")
    rollout_probe_parser.set_defaults(func=rollout_probe)

    reward_probe_parser = subparsers.add_parser("reward-probe")
    reward_probe_parser.add_argument("config")
    reward_probe_parser.add_argument("--batch-size", type=int, default=None)
    reward_probe_parser.add_argument("--seed", type=int, default=None)
    reward_probe_parser.set_defaults(func=reward_probe)

    def add_sd3_numeric_model_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--model-path", required=True)
        command_parser.add_argument("--repo-root", default=None)
        command_parser.add_argument("--prompt", default="a red square")
        command_parser.add_argument("--resolution", type=int, default=256)
        command_parser.add_argument("--num-steps", type=int, default=3)
        command_parser.add_argument("--guidance-scale", type=float, default=4.5)
        command_parser.add_argument("--seed", type=int, default=23)
        command_parser.add_argument("--device", default=None)
        command_parser.add_argument("--dtype", default="bfloat16")
        command_parser.add_argument("--lora-rank", type=int, default=32)
        command_parser.add_argument("--lora-alpha", type=int, default=64)
        command_parser.add_argument("--max-sequence-length", type=int, default=128)
        command_parser.add_argument("--logprob-atol", type=float, default=1e-5)
        command_parser.add_argument("--disable-lora", action="store_true")

    sd3_smoke_parser = subparsers.add_parser("sd3-numeric-smoke")
    add_sd3_numeric_model_options(sd3_smoke_parser)
    sd3_smoke_parser.set_defaults(func=sd3_numeric_smoke)

    sd3_branching_smoke_parser = subparsers.add_parser("sd3-branching-numeric-smoke")
    add_sd3_numeric_model_options(sd3_branching_smoke_parser)
    sd3_branching_smoke_parser.add_argument("--branch-count", type=int, default=2)
    sd3_branching_smoke_parser.add_argument(
        "--branch-step-index",
        type=_branch_step_index_arg,
        default="auto",
        help="Non-negative transition index, or 'auto' to validate every branch transition.",
    )
    sd3_branching_smoke_parser.add_argument("--clip-range", type=float, default=0.01)
    sd3_branching_smoke_parser.set_defaults(func=sd3_branching_numeric_smoke)

    tiny_loss_parser = subparsers.add_parser("tiny-loss-probe")
    tiny_loss_parser.add_argument("--output-dir", default="runs/tiny_loss_probe")
    tiny_loss_parser.add_argument("--steps", type=int, default=100)
    tiny_loss_parser.add_argument("--learning-rate", type=float, default=0.1)
    tiny_loss_parser.add_argument("--batch-size", type=int, default=4)
    tiny_loss_parser.add_argument("--num-steps", type=int, default=4)
    tiny_loss_parser.add_argument("--image-size", type=int, default=8)
    tiny_loss_parser.add_argument("--seed", type=int, default=123)
    tiny_loss_parser.add_argument("--device", default="cpu")
    tiny_loss_parser.add_argument("--target-bias", type=float, nargs=3, default=(0.8, -0.4, -0.4))
    tiny_loss_parser.add_argument("--max-final-loss-ratio", type=float, default=0.1)
    tiny_loss_parser.add_argument("--max-final-bias-error-ratio", type=float, default=0.25)
    tiny_loss_parser.add_argument("--no-assert-descent", action="store_true")
    tiny_loss_parser.set_defaults(func=tiny_loss_probe)

    checkpoint_parser = subparsers.add_parser("checkpoint-inventory")
    checkpoint_parser.add_argument("roots", nargs="+")
    checkpoint_parser.add_argument("--require-adapter", action="append", default=[])
    checkpoint_parser.add_argument("--max-depth", type=int, default=5)
    checkpoint_parser.set_defaults(func=checkpoint_inventory)

    world_r1_reward_parser = subparsers.add_parser("world-r1-reward-server-probe")
    world_r1_reward_parser.add_argument("--reward", choices=["reward_general", "reward_3d"], required=True)
    world_r1_reward_parser.add_argument("--url", required=True)
    world_r1_reward_parser.add_argument(
        "--protocol-mode",
        choices=["reference_v1", "strict_v2"],
        default="reference_v1",
    )
    world_r1_reward_parser.add_argument(
        "--wire-format",
        choices=["json_v1", "legacy_pickle"],
        default="json_v1",
    )
    world_r1_reward_parser.add_argument("--allow-unsafe-pickle", action="store_true")
    world_r1_reward_parser.add_argument("--trusted-host", action="append", default=[])
    world_r1_reward_parser.add_argument(
        "--max-response-bytes", type=int, default=16 * 1024 * 1024
    )
    world_r1_reward_parser.add_argument(
        "--http", action="store_true", help="Explicitly send the prepared HTTP request."
    )
    world_r1_reward_parser.add_argument("--timeout", type=float, default=5.0)
    world_r1_reward_parser.add_argument("--retries", type=int, default=0)
    world_r1_reward_parser.add_argument("--batch-size", type=int, default=1)
    world_r1_reward_parser.add_argument("--frames", type=int, default=2)
    world_r1_reward_parser.add_argument("--height", type=int, default=4)
    world_r1_reward_parser.add_argument("--width", type=int, default=4)
    world_r1_reward_parser.add_argument("--seed", type=int, default=123)
    world_r1_reward_parser.add_argument("--prompt", default="a red cube")
    world_r1_reward_parser.set_defaults(func=world_r1_reward_server_probe)

    wan_checkpoint_parser = subparsers.add_parser("wan-checkpoint-probe")
    wan_checkpoint_parser.add_argument("--model-path", required=True)
    wan_checkpoint_parser.add_argument("--repo-root", default="reference_code/World-R1-main")
    wan_checkpoint_parser.add_argument("--torch-dtype", default="auto")
    wan_checkpoint_parser.add_argument("--device", default="")
    wan_checkpoint_parser.add_argument("--manifest-only", action="store_true")
    wan_checkpoint_parser.add_argument("--allow-download", action="store_true")
    wan_checkpoint_parser.add_argument("--no-low-cpu-mem-usage", action="store_true")
    wan_checkpoint_parser.set_defaults(func=wan_checkpoint_probe)

    plan_parser = subparsers.add_parser("world-r1-plan")
    plan_parser.add_argument("--model-path", required=True)
    plan_parser.add_argument("--repo-dir", default="reference_code/World-R1-main")
    plan_parser.add_argument("--gpus", default="6,7")
    plan_parser.add_argument("--output-root", default="runs/world_r1_reference")
    plan_parser.add_argument("--full", action="store_true")
    plan_parser.set_defaults(func=world_r1_plan)

    wan_parser = subparsers.add_parser("wan-plan")
    wan_parser.add_argument("--config", default=None)
    wan_parser.add_argument("--model-path", default=None)
    wan_parser.add_argument("--output-dir", default=None)
    wan_parser.set_defaults(func=wan_plan)

    from scripts.remote_smoke import (
        DEFAULT_CONDA_BIN,
        DEFAULT_CONDA_ENV,
        DEFAULT_LEGACY_REPO_ROOT,
        DEFAULT_REMOTE_ROOT,
        DEFAULT_SERVER,
    )

    remote_sd3_parser = subparsers.add_parser("remote-sd3-cli-smoke")
    remote_sd3_parser.add_argument("--server", default=DEFAULT_SERVER)
    remote_sd3_parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
    )
    remote_sd3_parser.add_argument("--gpu", type=int, default=2)
    remote_sd3_parser.add_argument("--model-path", default="")
    remote_sd3_parser.add_argument("--repo-root", default=DEFAULT_LEGACY_REPO_ROOT)
    remote_sd3_parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    remote_sd3_parser.add_argument("--conda-bin", default=DEFAULT_CONDA_BIN)
    remote_sd3_parser.add_argument("--resolution", type=int, default=256)
    remote_sd3_parser.add_argument("--num-steps", type=int, default=3)
    remote_sd3_parser.add_argument("--guidance-scale", type=float, default=4.5)
    remote_sd3_parser.add_argument("--seed", type=int, default=23)
    remote_sd3_parser.add_argument("--dtype", default="bfloat16")
    remote_sd3_parser.add_argument("--lora-rank", type=int, default=32)
    remote_sd3_parser.add_argument("--lora-alpha", type=int, default=64)
    remote_sd3_parser.add_argument("--max-sequence-length", type=int, default=128)
    remote_sd3_parser.add_argument("--branch-count", type=int, default=2)
    remote_sd3_parser.add_argument("--sample-batch-size", type=int, default=1)
    remote_sd3_parser.add_argument(
        "--condition",
        choices=["active", "zero_lr_control"],
        default="active",
    )
    remote_sd3_parser.add_argument("--logprob-atol", type=float, default=1e-5)
    remote_sd3_parser.add_argument(
        "--tempflow-execution-mode",
        choices=["reference-compatible", "policy-identity"],
        default="policy-identity",
    )
    remote_sd3_parser.add_argument(
        "--allow-initial-clipping",
        action="store_true",
    )
    remote_sd3_parser.add_argument("--bounded-steps", type=int, default=1)
    remote_sd3_parser.add_argument("--resume-steps", type=int, default=1)
    remote_sd3_parser.add_argument("--idle-memory-mb", type=int, default=1024)
    remote_sd3_parser.add_argument("--idle-util-pct", type=int, default=5)
    remote_sd3_parser.add_argument("--stage-name", default=None)
    remote_sd3_parser.add_argument("--prompt", default="a red square")
    remote_sd3_parser.add_argument("--train-prompts-file", default=None)
    remote_sd3_parser.add_argument("--heldout-prompts-file", default=None)
    remote_sd3_parser.add_argument("--baseline-eval", default=None)
    remote_sd3_parser.add_argument(
        "--eval-seeds",
        type=_comma_separated_ints,
        default=[1701, 1702, 1703],
    )
    remote_sd3_parser.add_argument("--eval-max-prompts", type=int, default=None)
    remote_sd3_parser.add_argument("--run-bounded-trainer", dest="run_bounded_trainer", action="store_true", default=True)
    remote_sd3_parser.add_argument("--skip-bounded-trainer", dest="run_bounded_trainer", action="store_false")
    remote_sd3_parser.add_argument("--run-resume-validation", dest="run_resume_validation", action="store_true", default=True)
    remote_sd3_parser.add_argument("--skip-resume-validation", dest="run_resume_validation", action="store_false")
    remote_sd3_parser.add_argument("--allow-long-run", action="store_true")
    remote_sd3_parser.add_argument("--allow-large-run", action="store_true")
    remote_sd3_parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    remote_sd3_parser.add_argument("--execute", dest="dry_run", action="store_false")
    remote_sd3_parser.set_defaults(func=remote_sd3_cli_smoke)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
