"""Local-only H3 meta/LoRA construction and owner-only full CPU initialization.

Uses public Diffusers/Accelerate/PEFT/safetensors APIs, never vendor imports.
H3 topology/precision reference: UniRL 8cfbac17765004afb68fef08a4e011a4c8daf555
(Apache-2.0), minimax_h3/bundle.py, vendor/transformer_minimax_h3.py and
minimax_h3_t2va_trainside.yaml. No upstream implementation copied.
This is NOT a sharded loader: the initialization owner holds the full CPU state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch

_FP32_ROOTS = frozenset(("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope"))
_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
_H3_MODULE = "diffusers.models.transformers.transformer_minimax_h3"
_H3_CLASS = "MiniMaxH3Transformer3DModel"


def _resolve_h3_class():
    try:
        return getattr(importlib.import_module(_H3_MODULE), _H3_CLASS)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"H3 requires optional {_H3_MODULE}.{_H3_CLASS}; no reference/fake fallback") from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict:
    value = json.loads(data, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _local_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("artifact path escape or invalid filename")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"artifact file missing or path escape: {name}")
    return path


@dataclass(frozen=True)
class H3TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class H3MetaModel:
    model: torch.nn.Module
    transformer_dir: Path
    config_sha256: str
    block_names: tuple[str, ...]
    target_names: tuple[str, ...]
    base_to_model: Mapping[str, str]
    state_specs: Mapping[str, H3TensorSpec]
    lora_names: tuple[str, ...]
    # Nonpersistent values are NOT in state_dict; prepare must restore these
    # after any to_empty call, which otherwise destroys their constructor values.
    nonpersistent_buffers: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class H3CPUState:
    state: dict[str, torch.Tensor]
    nonpersistent_buffers: dict[str, torch.Tensor]
    estimated_weight_bytes: int
    file_sha256: Mapping[str, str]


def build_h3_meta(local_model_dir: str | Path, *, rank: int = 32, alpha: int = 64) -> H3MetaModel:
    """Construct params on meta; leave constructor buffers real, no weight reads.

    FQNs keep the H3 root: targets gain `.base_layer.{weight,bias}` and
    `.lora_{A,B}.default.weight`. No PeftModel/base_model.model prefix.
    The returned model has no optimizer or frozen parameter-identity manager.
    """
    root = Path(local_model_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("H3 requires an existing local model directory, not a Hub ID")
    directory = (root / "transformer").resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise ValueError("local transformer directory missing or path escape")
    config_path = _local_file(directory, "config.json")
    config_bytes = config_path.read_bytes()
    config = _json(config_bytes)
    if config.get("_class_name") != _H3_CLASS:
        raise ValueError("config must declare MiniMaxH3Transformer3DModel")
    for name, expected in (("in_channels", 24), ("audio_in_channels", 32)):
        if type(config.get(name)) is not int or config[name] != expected:
            raise ValueError(f"invalid H3 config {name}")
    if config.get("patch_size") != [1, 2, 2]:
        raise ValueError("invalid H3 patch_size")
    for name in ("text_dim", "hidden_size", "num_layers"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"invalid H3 config {name}")
    for name, value in (("rank", rank), ("alpha", alpha)):
        if type(value) is not int or value < 1:
            raise ValueError(f"LoRA {name} must be a positive integer")
    cls = _resolve_h3_class()
    if set(getattr(cls, "_keep_in_fp32_modules", ())) != _FP32_ROOTS:
        raise ValueError("H3 FP32 island declaration mismatch")
    try:
        from accelerate import init_empty_weights
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise ImportError("H3 meta construction requires Accelerate and PEFT") from exc
    loaded_config = cls.load_config(str(directory), local_files_only=True)
    if dict(loaded_config) != config:
        raise ValueError("local config changed or dependency altered config")
    # fork_rng protects the caller even if a dependency's constructor uses RNG.
    with torch.random.fork_rng(devices=[]), init_empty_weights(include_buffers=False):
        model = cls.from_config(loaded_config)
    if not isinstance(model, torch.nn.Module) or not all(p.is_meta for p in model.parameters()):
        raise ValueError("H3 construction must leave every parameter on meta")
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != config["num_layers"]:
        raise ValueError("invalid H3 main block structure")
    for name, in_features, out_features in (
        ("proj_in", 96, config["hidden_size"]), ("audio_proj_in", 32, config["hidden_size"]),
        ("context_embedder", config["text_dim"], config["hidden_size"]),
        ("proj_out", config["hidden_size"], 96), ("audio_proj_out", config["hidden_size"], 32),
    ):
        layer = getattr(model, name, None)
        if not isinstance(layer, torch.nn.Linear) or (layer.in_features, layer.out_features) != (in_features, out_features):
            raise ValueError(f"invalid H3 projection shape: {name}")
    names = tuple(f"transformer_blocks.{i}" for i in range(len(blocks)))
    targets = tuple(f"{name}.{suffix}" for name in names for suffix in _TARGETS)
    try:
        layers = [model.get_submodule(name) for name in targets]
    except AttributeError as exc:
        raise ValueError("each H3 main block must contain all six LoRA targets") from exc
    if not targets or any(not isinstance(layer, torch.nn.Linear) for layer in layers) or len({id(x) for x in layers}) != len(layers):
        raise ValueError("LoRA targets must be distinct main-block Linear layers")
    params = tuple(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in params}) != len(params):
        raise ValueError("aliased base parameters are unsupported")
    base_shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    for name, parameter in params:
        parameter.requires_grad_(False)
        parameter.data = parameter.data.to(torch.float32 if name.split(".")[0] in _FP32_ROOTS else torch.bfloat16)
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
    for name, buffer in model.named_buffers():
        if buffer.device.type != "cpu" or not buffer.is_floating_point() or not bool(torch.isfinite(buffer).all()):
            raise ValueError(f"H3 constructor buffer must be real finite CPU floating point: {name}")
        if name.split(".")[0] in _FP32_ROOTS and buffer.dtype != torch.float32:
            raise ValueError(f"H3 FP32 buffer was constructed at wrong precision: {name}")
    if "rope.inv_freq" not in dict(model.named_buffers()):
        raise ValueError("missing real rope.inv_freq buffer")
    lora_config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., bias="none",
                             target_modules=list(targets), init_lora_weights=True)
    with torch.random.fork_rng(devices=[]), init_empty_weights(include_buffers=False):
        injected = inject_adapter_in_model(lora_config, model, adapter_name="default", low_cpu_mem_usage=True)
    if injected is not model:
        raise ValueError("PEFT must inject in-place without a new root")
    lora = tuple(f"{target}.lora_{side}.default.weight" for target in targets for side in ("A", "B"))
    actual_trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if actual_trainable != set(lora):
        raise ValueError("PEFT trainable scope differs from exact main-block LoRA targets")
    for name, parameter in model.named_parameters():
        if not parameter.is_meta:
            raise ValueError("PEFT must leave all parameters on meta")
        if name in actual_trainable:
            parameter.data = parameter.data.to(torch.float32)
    if any(getattr(m, "gradient_checkpointing", False) for m in model.modules()):
        raise ValueError("internal gradient checkpointing must be disabled")
    mapping = {}
    for name in base_shapes:
        parent, _, leaf = name.rpartition(".")
        mapping[name] = f"{parent}.base_layer.{leaf}" if parent in targets else name
    state = model.state_dict()
    if len(set(mapping.values())) != len(mapping) or set(state) != set(mapping.values()) | set(lora):
        raise ValueError("base/LoRA state mapping is not a complete bijection")
    if any(tuple(state[target].shape) != base_shapes[source] for source, target in mapping.items()):
        raise ValueError("PEFT changed a base tensor shape")
    specs = {name: H3TensorSpec(tuple(t.shape), t.dtype) for name, t in state.items()}
    for name, parameter in model.named_parameters():
        expected_dtype = torch.float32 if name in lora or name.split(".")[0] in _FP32_ROOTS else torch.bfloat16
        if parameter.dtype != expected_dtype:
            raise ValueError(f"post-PEFT precision mismatch: {name}")
    buffers = {name: t.detach().clone() for name, t in model.named_buffers() if name not in state}
    if set(buffers) != {"rope.inv_freq"}:
        raise ValueError("unsupported nonpersistent H3 buffers; review reconstruction explicitly")
    return H3MetaModel(model, directory, hashlib.sha256(config_bytes).hexdigest(), names, targets,
                       MappingProxyType(mapping), MappingProxyType(specs), lora, MappingProxyType(buffers))


def _weight_headers(directory: Path) -> tuple[dict[str, Path], dict[str, dict]]:
    """Validate JSON and real safetensors metadata without calling get_tensor."""
    from safetensors import safe_open

    single = directory / "diffusion_pytorch_model.safetensors"
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    if single.exists() == index.exists():
        raise ValueError("require exactly one safetensors single file or shard index; no bin fallback")
    declared = None
    if index.exists():
        data = _json(_local_file(directory, index.name).read_bytes())
        declared = data.get("weight_map")
        if not isinstance(declared, dict) or not declared:
            raise ValueError("shard index requires nonempty weight_map")
        paths = {name: _local_file(directory, file) for name, file in declared.items()}
        if any(p.suffix != ".safetensors" for p in paths.values()):
            raise ValueError("only safetensors shards are supported")
        files = sorted(set(paths.values()))
    else:
        files = [_local_file(directory, single.name)]
    locations, headers = {}, {}
    for path in files:
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 2 <= size <= min(100_000_000, path.stat().st_size - 8):
                raise ValueError("invalid safetensors header size")
            header = _json(stream.read(size))
        header.pop("__metadata__", None)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(header):
                raise ValueError("safetensors header keys disagree")
            for name, info in header.items():
                if name in locations:
                    raise ValueError(f"duplicate tensor across shards: {name}")
                if declared is not None and (name not in paths or paths[name] != path):
                    raise ValueError(f"undeclared or misplaced shard tensor: {name}")
                view = handle.get_slice(name)
                if info["shape"] != view.get_shape() or info["dtype"] != view.get_dtype():
                    raise ValueError("safetensors metadata mismatch")
                locations[name], headers[name] = path, info
    if declared is not None and set(locations) != set(declared):
        raise ValueError("shard index missing actual tensor keys")
    return locations, headers


def load_h3_cpu_state(
    meta: H3MetaModel, *, is_init_owner: bool, host_weight_limit_bytes: int, init_seed: int,
) -> H3CPUState:
    """Owner only; complete CPU base + FP32 Kaiming-A/zero-B initialization.

    Budget conservatively includes twice source file bytes plus twice complete
    destination bytes and constructor-buffer copies. It is NOT a host RSS bound.
    Header validation and admission happen before any tensor payload read.
    """
    if is_init_owner is not True:
        raise PermissionError("only the designated initialization owner may read full CPU weights")
    if type(host_weight_limit_bytes) is not int or host_weight_limit_bytes < 1:
        raise ValueError("host_weight_limit_bytes must be an explicit positive integer")
    if type(init_seed) is not int or not 0 <= init_seed < 2**63:
        raise ValueError("init_seed must be an integer in [0,2**63)")
    if hashlib.sha256(_local_file(meta.transformer_dir, "config.json").read_bytes()).hexdigest() != meta.config_sha256:
        raise ValueError("local config changed since meta construction")
    locations, headers = _weight_headers(meta.transformer_dir)
    if set(locations) != set(meta.base_to_model):
        raise ValueError("checkpoint keys must exactly cover all base state FQNs")
    for name, header in headers.items():
        spec = meta.state_specs[meta.base_to_model[name]]
        if tuple(header["shape"]) != spec.shape:
            raise ValueError(f"checkpoint shape mismatch: {name}")
        if header["dtype"] not in ("F32", "BF16", "F16"):
            raise ValueError(f"unsupported checkpoint dtype: {name}: {header['dtype']}")
    files = sorted(set(locations.values()))
    budget = (2 * sum(p.stat().st_size for p in files)
              + 2 * sum(s.nbytes for s in meta.state_specs.values())
              + 2 * sum(t.numel() * t.element_size() for t in meta.nonpersistent_buffers.values()))
    if budget > host_weight_limit_bytes:
        raise ValueError(f"host weight budget exceeded: estimated {budget} > {host_weight_limit_bytes}")
    from safetensors import safe_open

    state, identities = {}, {"config.json": meta.config_sha256}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        identities[str(path.relative_to(meta.transformer_dir))] = digest.hexdigest()
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                target = meta.base_to_model[name]
                view = handle.get_slice(name)
                if view.get_shape() != headers[name]["shape"] or view.get_dtype() != headers[name]["dtype"]:
                    raise ValueError(f"checkpoint metadata changed after admission: {name}")
                tensor = handle.get_tensor(name).to(dtype=meta.state_specs[target].dtype, copy=True)
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"nonfinite checkpoint tensor after conversion: {name}")
                state[target] = tensor
    generator = torch.Generator(device="cpu").manual_seed(init_seed)
    for name in meta.lora_names:
        tensor = torch.empty(meta.state_specs[name].shape, dtype=torch.float32, device="cpu")
        if ".lora_A." in name:
            torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), generator=generator)
        else:
            tensor.zero_()
        state[name] = tensor
    if set(state) != set(meta.state_specs):
        raise ValueError("incomplete CPU initialization state")
    return H3CPUState(state, {name: t.clone() for name, t in meta.nonpersistent_buffers.items()},
                      budget, MappingProxyType(identities))
