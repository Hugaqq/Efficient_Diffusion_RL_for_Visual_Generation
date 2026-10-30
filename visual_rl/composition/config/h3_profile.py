"""Closed schema2 H3 profile; no controller, model loading, HTTP or filesystem writes.

Root: schema_version:2, recipe:h3_t2va_mvp_v1, overrides, launch.
Overrides (optional groups): model.params (height,width,frames,fps,guidance_scale,
  lora_rank,lora_alpha); training (group_size,num_steps). All other math is fixed.
Launch (all keys required): output_dir,resume_from,artifacts,expected,execution,
  seed,max_optimizer_steps,checkpoint_every_optimizer_steps,resume_policy,budgets,hps.
Artifacts: model,text_cache,train,eval,hps_manifest (local paths, YAML-relative).
Expected: model_sha256,train_sha256,eval_sha256,cache_identity,hps_service_identity.
Execution: world_size:8,sp:1,tp:1,gpu_uuids (eight ordered distinct GPU UUIDs).
Budgets: host_weight_bytes,trajectory_bytes,media_bytes,decode_bytes,condition_bytes,
  max_tokens,embedding_bytes,prompt_file_bytes (positive integers, explicit).
  Prompt reads use min(prompt_file_bytes, H3PromptSource's hard file limit).
Hps: origin,timeout_s,max_response_bytes. Resume policy must be
  complete_collection_same_topology; null resume_from means a fresh run.

compile_h3_profile validates declarations/paths only. preflight_h3_profile reads
local content and produces a path-independent semantic ID, not a G3 receipt.
Model weights are only header-inspected, NOT fully hashed or loaded: their
expected identity remains DECLARED until runtime artifact materialization.
GPU UUID syntax is not proof of authorization/availability. No live health,
CUDA/API capability, checkpoint completeness or resource RSS claim is made.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path
import re
from collections.abc import Mapping

from visual_rl.composition.config.source import SourceRecipe, load_source_recipe
from visual_rl.composition.registry import DeclarationResolver, Registry, ResolvedComponentDeclaration
from visual_rl.core.identity import canonical_identity
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict
from visual_rl.models.catalog import MODEL_CATALOG_FRAGMENT, positive_int, strict_values
from visual_rl.models.h3_interface import H3_INITIAL_STATE_VERSION

_RECIPE = "h3_t2va_mvp_v1"
_RESUME = "complete_collection_same_topology"
_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def _exact(value, keys, label):
    keys = frozenset(keys)
    return strict_values(value, allowed=keys, required=keys, label=label)


def _hash(value):
    from visual_rl.core.protocols.flash_hpsv3 import sha256_id
    value = sha256_id(value)
    if value == "0" * 64:
        raise ValueError("placeholder all-zero SHA256 is not an expected identity")
    return value


def _positive_float(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class H3Budgets:
    host_weight_bytes: int
    trajectory_bytes: int
    media_bytes: int
    decode_bytes: int
    condition_bytes: int
    max_tokens: int
    embedding_bytes: int
    prompt_file_bytes: int

    def __post_init__(self):
        for item in fields(self):
            positive_int(item.name, getattr(self, item.name))


@dataclass(frozen=True)
class H3Training:
    group_size: int = 8
    num_steps: int = 10
    algorithm: str = "cps"

    def __post_init__(self):
        if type(self.group_size) is not int or not 2 <= self.group_size <= 64:
            raise ValueError("group_size must be an integer in [2,64]")
        if type(self.num_steps) is not int or not 1 <= self.num_steps <= 100:
            raise ValueError("num_steps must be an integer in [1,100]")
        if self.algorithm not in ("cps", "dual_flow", "dual_flash"):
            raise ValueError("algorithm must be cps, dual_flow or dual_flash")
        if self.algorithm != "cps" and self.num_steps < 2:
            raise ValueError("dual Flow-SDE requires at least two transitions")

    def payload(self):
        from visual_rl.data import h3_source

        result = dict(group_size=self.group_size, num_steps=self.num_steps,
            groups_per_rank=1, world_size=8, sp=1, tp=1, microbatch=1, chunk_size=1,
            optimizer_updates_per_collection=1, score="cps_proxy_negative_mse", av_weighting="generated_element_mean",
            action_dtype="fp16", score_dtype="fp32", schedule_dtype="fp32", video_shift=12., audio_shift=3., eta=.7,
            model_precision="mixed_bf16_fp32_islands", lora_master_dtype="fp32", aux_dtype="fp32",
            decode_backend="rank_local_cuda_fp32_cpu_media_v1",
            fp32_roots=("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope"),
            lora_targets="main_transformer_blocks_six_linear_only.v1",
            policy_steps="all_including_terminal_q0", old_policy="frozen_replay", beta=0., ema=False, reference=False,
            optimizer="adamw", lr=1e-4, betas=(.9, .999), eps=1e-12, weight_decay=0., clip_range=.005,
            max_grad_norm=1., lr_schedule="constant", initial_state_version=H3_INITIAL_STATE_VERSION,
            sampling=h3_source.SAMPLING_VERSION,
            random_streams=h3_source.STREAM_VERSION,
            reward="h3_hpsv3_video_only_v1", reward_weight=1., audio_quality_scored=False)
        if self.algorithm != "cps":
            result.update(algorithm=self.algorithm, score="dual_gaussian_element_mean_v1",
                          eta=None, policy_steps=("all_stochastic" if self.algorithm == "dual_flow" else "one_group_selected"),
                          selection="h3-dual-prompt-group-sha256-rejection-v1",
                          deterministic="dual_euler_nonpolicy_v1",
                          rectification=("per_stream_world_mean_gradient_v1" if self.algorithm == "dual_flash" else "none"))
        return result


@dataclass(frozen=True)
class H3Launch:
    output_dir: Path
    resume_from: Path | None
    model: Path
    text_cache: Path
    train: Path
    eval: Path
    hps_manifest: Path
    gpu_uuids: tuple[str, ...]
    seed: int
    max_optimizer_steps: int
    checkpoint_every_optimizer_steps: int
    budgets: H3Budgets
    hps_origin: str
    hps_timeout_s: float
    hps_max_response_bytes: int

    @property
    def binding_id(self):
        payload = asdict(self)
        for key in ("output_dir", "resume_from", "model", "text_cache", "train", "eval", "hps_manifest"):
            payload[key] = None if payload[key] is None else str(payload[key])
        return canonical_identity("h3.launch.v1", payload)


@dataclass(frozen=True)
class H3Profile:
    model: ResolvedComponentDeclaration
    training: H3Training
    launch: H3Launch
    expected: FrozenMapping
    source_id: str


@dataclass(frozen=True)
class H3LocalFacts:
    semantic_id: str
    semantic_payload: FrozenMapping
    binding_id: str
    train_sha256: str
    eval_sha256: str
    cache_identity: str
    model_config_hashes: FrozenMapping
    hps_manifest_sha256: str
    estimated_host_weight_bytes: int
    minimum_trajectory_bytes: int
    minimum_condition_bytes: int
    # This is deliberately not a model-weight/G3/GPU attestation.
    model_weights_verified: bool = False


def compile_h3_profile(source: str | Path | SourceRecipe) -> H3Profile:
    from visual_rl.algorithms.rewards.clients.hpsv3 import loopback_origin
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract

    source = source if isinstance(source, SourceRecipe) else load_source_recipe(source)
    raw = _exact(to_plain_dict(source.values), ("schema_version", "recipe", "overrides", "launch"), "H3 root")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 2 or raw["recipe"] != _RECIPE:
        raise ValueError("requires schema_version integer2 and recipe h3_t2va_mvp_v1")
    overrides = strict_values(raw["overrides"], allowed=frozenset(("model", "training")), required=frozenset(), label="H3 overrides")
    model_group = _exact(overrides.get("model", {"params": {}}), ("params",), "H3 model")
    model_params = strict_values(model_group["params"], allowed=frozenset(("height", "width", "frames", "fps", "guidance_scale", "lora_rank", "lora_alpha")),
                                 required=frozenset(), label="H3 model params")
    train_params = strict_values(overrides.get("training", {}), allowed=frozenset(("group_size", "num_steps", "algorithm")), required=frozenset(), label="H3 training")
    training = H3Training(**train_params)
    launch = _exact(raw["launch"], ("output_dir", "resume_from", "artifacts", "expected", "execution", "seed",
        "max_optimizer_steps", "checkpoint_every_optimizer_steps", "resume_policy", "budgets", "hps"), "H3 launch")
    budgets = H3Budgets(**_exact(launch["budgets"], (f.name for f in fields(H3Budgets)), "H3 budgets"))
    params = dict(artifact_ref="h3-model", text_cache_ref="h3-text", max_tokens=budgets.max_tokens,
                  max_embedding_bytes=budgets.embedding_bytes, max_media_bytes=budgets.media_bytes, max_decode_bytes=budgets.decode_bytes)
    model = DeclarationResolver().resolve(Registry("model", MODEL_CATALOG_FRAGMENT.descriptors), "h3-t2va", params | model_params)
    config = model.config
    VideoContract(config.frames, config.height, config.width)  # unchanged protocol's frame/shape limits
    if config.frames != 124:
        raise ValueError("this H3 HPS profile requires exactly124 frames")
    execution = _exact(launch["execution"], ("world_size", "sp", "tp", "gpu_uuids"), "H3 execution")
    for key, expected in (("world_size", 8), ("sp", 1), ("tp", 1)):
        if type(execution[key]) is not int or execution[key] != expected:
            raise ValueError("H3 requires fixed WORLD8/SP1/TP1; no elastic topology")
    uuids = execution["gpu_uuids"]
    if (not isinstance(uuids, list) or len(uuids) != 8 or any(not isinstance(u, str) or not _UUID.fullmatch(u)
            or u == "GPU-00000000-0000-0000-0000-000000000000" for u in uuids) or len(set(uuids)) != 8):
        raise ValueError("eight distinct explicit authorized GPU UUIDs are required; no index/default placeholders")
    if launch["resume_policy"] != _RESUME:
        raise ValueError("only complete_collection_same_topology resume is supported")
    if type(launch["seed"]) is not int or not 0 <= launch["seed"] < 2**63:
        raise ValueError("seed must be integer in [0,2**63)")
    steps = positive_int("max_optimizer_steps", launch["max_optimizer_steps"])
    cadence = positive_int("checkpoint_every_optimizer_steps", launch["checkpoint_every_optimizer_steps"])
    if cadence > steps:
        raise ValueError("checkpoint cadence must not exceed max_optimizer_steps")
    def path(value):
        if not isinstance(value, str) or not value or value.strip() != value or "://" in value:
            raise ValueError("explicit local paths are required; no URLs or pending placeholders")
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else source.path.parent / candidate).resolve()
    artifacts = _exact(launch["artifacts"], ("model", "text_cache", "train", "eval", "hps_manifest"), "H3 artifacts")
    paths = {key: path(value) for key, value in artifacts.items()}
    expected = _exact(launch["expected"], ("model_sha256", "train_sha256", "eval_sha256", "cache_identity", "hps_service_identity"), "H3 expected")
    for key in ("model_sha256", "train_sha256", "eval_sha256", "hps_service_identity"):
        expected[key] = _hash(expected[key])
    cache_fields = ("model_sha256", "encoder_sha256", "tokenizer_sha256", "processor_sha256", "template_sha256", "truncation_sha256")
    cache = _exact(expected["cache_identity"], (*cache_fields, "target_layer", "normalized", "add_special_tokens", "feature_dim", "dtype"), "cache identity")
    for key in cache_fields:
        _hash(cache[key])
    if cache["model_sha256"] != expected["model_sha256"]:
        raise ValueError("cache/model declared identity mismatch")
    # Full reader validation is repeated against actual files during preflight.
    if (type(cache["target_layer"]) is not int or cache["target_layer"] != 50 or cache["normalized"] is not False
            or cache["add_special_tokens"] is not False or type(cache["feature_dim"]) is not int or cache["feature_dim"] != 5120
            or cache["dtype"] not in ("float32", "float16", "bfloat16")):
        raise ValueError("invalid layer50/raw/5120 cache identity")
    hps = _exact(launch["hps"], ("origin", "timeout_s", "max_response_bytes"), "H3 HPS")
    bound = H3Launch(output_dir=path(launch["output_dir"]), resume_from=None if launch["resume_from"] is None else path(launch["resume_from"]),
        **paths, gpu_uuids=tuple(uuids), seed=launch["seed"], max_optimizer_steps=steps, checkpoint_every_optimizer_steps=cadence,
        budgets=budgets, hps_origin=loopback_origin(hps["origin"]), hps_timeout_s=_positive_float(hps["timeout_s"], "HPS timeout"),
        hps_max_response_bytes=positive_int("HPS response bound", hps["max_response_bytes"]))
    if bound.output_dir in paths.values() or bound.output_dir == bound.resume_from:
        raise ValueError("output_dir cannot be an input artifact or resume checkpoint")
    return H3Profile(model, training, bound, FrozenMapping(expected), source.config_source_id)


def _read(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing local regular file or symlink: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"local file exceeds byte budget: {path.name}")
    return raw


def _prompts(path, expected, limit, minimum):
    from visual_rl.data import h3_source

    raw = _read(path, min(limit, h3_source._FILE_LIMIT))
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise ValueError("prompt file SHA256 mismatch")
    prompts = h3_source._prompts(raw)  # One authority for BOM/CR/empty/exact text rules.
    if len(prompts) < minimum:
        raise ValueError("prompt files require enough rows")
    return prompts, observed


def preflight_h3_profile(profile: H3Profile) -> H3LocalFacts:
    """Read-only local content check; no model construction, acquire or checkpoint restore."""
    from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
    from visual_rl.data import h3_source
    from visual_rl.core.protocols.flash_hpsv3 import MAX_HEALTH_BYTES, VideoContract, decode_json, validate_health
    from visual_rl.models.implementations.h3_decode import H3CPUDecoder
    from visual_rl.models.implementations.h3_loading import _weight_headers, _TARGETS, _FP32_ROOTS, _state_read_budget
    from visual_rl.models.implementations.h3_text_cache import H3TextCache

    if not isinstance(profile, H3Profile):
        raise TypeError("preflight requires H3Profile")
    launch, expected, config = profile.launch, profile.expected, profile.model.config
    budget, training = launch.budgets, profile.training
    train, train_hash = _prompts(launch.train, expected["train_sha256"], budget.prompt_file_bytes, 8)
    evaluation, eval_hash = _prompts(launch.eval, expected["eval_sha256"], budget.prompt_file_bytes, 1)
    cache = H3TextCache(launch.text_cache, expected_identity=expected["cache_identity"],
                       max_tokens=budget.max_tokens, max_embedding_bytes=budget.embedding_bytes)
    cache.verify_coverage(train, evaluation)
    largest = 0
    for prompt in dict.fromkeys((*train, *evaluation)):
        embedding = cache.get(prompt)  # one bounded tensor at a time, including finite validation
        largest = max(largest, embedding.numel() * embedding.element_size())
        del embedding
    condition_bytes = training.group_size * largest
    width = config.geometry.joint_layout.packed_width
    trajectory_bytes = training.group_size * training.num_steps * (4 * width + 4) + training.group_size * 2 * width
    media, workspace = H3CPUDecoder.estimate_bytes(config.geometry)
    if (condition_bytes > budget.condition_bytes or trajectory_bytes > budget.trajectory_bytes
            or media > budget.media_bytes or workspace > budget.decode_bytes):
        raise ValueError("H3 condition/trajectory/media/decode budget cannot fit fixed minimum workload")
    decoder = H3CPUDecoder(launch.model, max_media_bytes=budget.media_bytes, max_decode_bytes=budget.decode_bytes)
    decoder._configs()  # existing strict config/stats/path checks; never load either VAE
    configs, header_sets = {}, {}
    for component in ("transformer", "vae", "audio_vae"):
        directory = launch.model / component
        if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob("*")):
            raise ValueError("model component symlink is unsupported")
        raw = _read(directory / "config.json", 1024 * 1024)
        configs[component] = hashlib.sha256(raw).hexdigest()
        locations, headers = _weight_headers(directory.resolve())  # actual safetensors headers, no get_tensor
        if not headers or any(header["dtype"] not in ("F16", "BF16", "F32") for header in headers.values()):
            raise ValueError("model requires nonempty floating safetensors weights")
        header_sets[component] = headers
        if component == "transformer":
            model_config = decode_json(raw, 1024 * 1024)
    if (model_config.get("_class_name") != "MiniMaxH3Transformer3DModel"
            or any(type(model_config.get(key)) is not int or model_config[key] != value
                   for key, value in (("in_channels", 24), ("audio_in_channels", 32), ("text_dim", 5120)))
            or model_config.get("patch_size") != [1, 2, 2]
            or any(type(value) is not int for value in model_config["patch_size"])):
        raise ValueError("H3 transformer config family mismatch")
    layers = positive_int("num_layers", model_config.get("num_layers"))
    headers = header_sets["transformer"]
    lora_numel = 0
    for block in range(layers):
        for target in _TARGETS:
            header = headers.get(f"transformer_blocks.{block}.{target}.weight")
            if header is None or len(header["shape"]) != 2:
                raise ValueError("transformer header missing main-block LoRA target")
            lora_numel += config.lora_rank * sum(header["shape"])
    # Same mixed-precision destination and direct-IO working-set formula as
    # the actual loader. Exact model/FQN validation still follows on meta;
    # preflight neither allocates state nor claims observed host availability.
    destination_bytes = 4 * lora_numel + sum(
        math.prod(h["shape"]) * (4 if name.split('.')[0] in _FP32_ROOTS else 2)
        for name, h in headers.items())
    rope_bytes = 4 * positive_int('rope_freq_dim', model_config.get('rope_freq_dim', 16))
    host_bytes = _state_read_budget(destination_bytes, headers, rope_bytes)
    if host_bytes > budget.host_weight_bytes:
        raise ValueError("host weight budget cannot fit transformer initialization estimate")
    raw_hps = _read(launch.hps_manifest, MAX_HEALTH_BYTES)
    contract = VideoContract(config.frames, config.height, config.width)
    validate_health(decode_json(raw_hps, MAX_HEALTH_BYTES), expected["hps_service_identity"], contract)
    if launch.resume_from is not None and (not launch.resume_from.is_dir() or launch.resume_from.is_symlink()):
        raise ValueError("resume_from must name an existing local checkpoint directory")
    schedule = H3ScheduleSnapshot.build(training.num_steps)
    model_settings = asdict(config)
    for key in ("max_tokens", "max_embedding_bytes", "max_media_bytes", "max_decode_bytes"):
        model_settings.pop(key)
    payload = FrozenMapping(dict(recipe=_RECIPE, model=model_settings, math=training.payload(),
        seed=launch.seed, max_optimizer_steps=launch.max_optimizer_steps,
        data=dict(train_sha256=train_hash, eval_sha256=eval_hash, parse=h3_source.PARSE_VERSION),
        declared_model_sha256=expected["model_sha256"], model_config_hashes=configs,
        weight_header_ids={component: canonical_identity("h3.weight-headers.v1",
            {name: {"shape": header["shape"], "dtype": header["dtype"]} for name, header in headers.items()})
            for component, headers in header_sets.items()},
        cache_identity=cache.identity_sha256, cache_provenance=to_plain_dict(expected["cache_identity"]),
        hps_service_identity=expected["hps_service_identity"], video_contract=contract.payload(),
        video_sigmas=schedule.video_sigmas.tolist(), audio_sigmas=schedule.audio_sigmas.tolist(),
        resume_policy=_RESUME))
    return H3LocalFacts(canonical_identity("h3.semantic.v1", payload), payload, launch.binding_id,
        train_hash, eval_hash, cache.identity_sha256, FrozenMapping(configs), hashlib.sha256(raw_hps).hexdigest(),
        host_bytes, trajectory_bytes, condition_bytes)
