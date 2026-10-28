"""Build real H3 layer-50 text features from local Qwen3-VL weights.

Uses the native HF path already documented in the H3 single-card branch:
Qwen2TokenizerFast, no specials/template/normalization, hidden_states[50].
No downloads, remote model Python, fake features or implicit truncation.
One prompt per forward, CPU weight offload, one explicitly selected GPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(root, names):
    records = {}
    for name in sorted(set(names)):
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("input manifest requires local basenames")
        path = root / name
        records[name] = {"bytes": path.stat().st_size, "sha256": digest_file(path)}
        print(json.dumps({"event": "input_hashed", "component": root.name, "file": name}), flush=True)
    return hashlib.sha256(canonical(records)).hexdigest(), records


def build(args):
    import torch
    import transformers
    from accelerate import cpu_offload
    from safetensors.torch import save_file
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration
    from visual_rl.composition.preflight.artifacts import _hash_artifact
    from visual_rl.data.h3_source import _prompts
    from visual_rl.models.implementations.h3_text_cache import H3TextCache

    if args.max_tokens < 1 or args.max_prompts < 1:
        raise ValueError("positive explicit token/prompt bounds required")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("set CUDA_VISIBLE_DEVICES to exactly one authorized GPU UUID")
    raw, model = args.raw_root.resolve(strict=True), args.model.resolve(strict=True)
    output = args.output.absolute()
    staging = output.with_name(output.name + ".partial")
    for source in (raw, model, Path(__file__).resolve().parents[1]):
        if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
            raise ValueError("cache output must be separate from weights/source")
    if output.exists() or staging.exists():
        raise FileExistsError("preserve existing cache/partial; choose a new output")
    groups, prompt_hashes = [], {}
    for label, path in (("train", args.train), ("eval", args.eval)):
        if path.is_symlink() or path.stat().st_size > 16 * 1024**2:
            raise ValueError("prompt input must be a bounded regular file")
        data = path.read_bytes()
        groups.append(_prompts(data))
        prompt_hashes[label] = hashlib.sha256(data).hexdigest()
    prompts = list(dict.fromkeys((*groups[0], *groups[1])))
    if len(groups[0]) < 8 or not groups[1] or set(groups[0]) & set(groups[1]):
        raise ValueError("need >=8 train rows and disjoint nonempty evaluation prompts")
    if len(prompts) > args.max_prompts:
        raise ValueError("prompt count exceeds explicit preparation bound")

    enc = raw / "text_encoder"
    index = json.loads((enc / "model.safetensors.index.json").read_text())
    weights = sorted(set(index["weight_map"].values()))
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in weights):
        raise ValueError("invalid native encoder shard names")
    source_bytes = sum((enc / name).stat().st_size for name in weights)
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if available < source_bytes + 24 * 1024**3:
        raise MemoryError("insufficient host RAM for encoder plus 24GiB reserve")
    device = torch.device("cuda:0")
    if torch.cuda.mem_get_info(device)[0] < 16 * 1024**3:
        raise MemoryError("need 16GiB free for bounded encoder offload")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    print(json.dumps({"event": "hash_model_begin", "path": str(model)}), flush=True)
    model_identity = _hash_artifact(model, content_policy="all-files.v1")["content_sha256"]
    encoder_hash, encoder_files = hash_files(enc, ["config.json", "model.safetensors.index.json", *weights])
    tok_root = raw / "tokenizer"
    tokenizer_hash, tokenizer_files = hash_files(tok_root, [p.name for p in tok_root.iterdir() if p.is_file()])
    tokenizer = Qwen2TokenizerFast.from_pretrained(tok_root, local_files_only=True)
    tokenized = [tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"] for prompt in prompts]
    if any(not ids or len(ids) > args.max_tokens for ids in tokenized):
        raise ValueError("empty/overlength prompt; implicit truncation is forbidden")
    processor = {"kind": "text-only-no-multimodal-processor", "transformers": transformers.__version__,
                 "encoder_python_sha256": digest_file(inspect.getfile(Qwen3VLForConditionalGeneration)),
                 "tokenizer_python_sha256": digest_file(inspect.getfile(Qwen2TokenizerFast)),
                 "producer_sha256": digest_file(__file__)}
    identity = dict(model_sha256=model_identity, encoder_sha256=encoder_hash, tokenizer_sha256=tokenizer_hash,
                    processor_sha256=hashlib.sha256(canonical(processor)).hexdigest(),
                    template_sha256=hashlib.sha256(canonical({"template": "identity", "chat_template": False})).hexdigest(),
                    truncation_sha256=hashlib.sha256(canonical({"policy": "reject", "max_tokens": args.max_tokens})).hexdigest(),
                    target_layer=50, normalized=False, add_special_tokens=False, feature_dim=5120, dtype="float32")
    metadata = {"h3_text_identity_sha256": hashlib.sha256(canonical(identity)).hexdigest()}
    entries, shards = {}, {}
    print(json.dumps({"event": "encoder_load_begin", "weight_bytes": source_bytes, "prompts": len(prompts)}), flush=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(enc, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="sdpa")
    try:
        if encoder.config.text_config.num_hidden_layers <= 50 or encoder.config.text_config.hidden_size != 5120:
            raise ValueError("requires raw intermediate layer50, width5120")
        encoder.eval().requires_grad_(False)
        encoder.lm_head = torch.nn.Identity()
        cpu_offload(encoder.model, execution_device=device, offload_buffers=True)
        with torch.inference_mode():
            for ordinal, (prompt, ids) in enumerate(zip(prompts, tokenized)):
                inputs = torch.tensor([ids], device=device, dtype=torch.long)
                result = encoder.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    mm_token_type_ids=torch.zeros_like(inputs), use_cache=False, output_hidden_states=True)
                feature = result.hidden_states[50][0].detach().to(device="cpu", dtype=torch.float32).contiguous()
                del result, inputs
                if feature.shape != (len(ids), 5120) or not bool(torch.isfinite(feature).all()):
                    raise ValueError("invalid real encoder feature")
                key = hashlib.sha256(prompt.encode()).hexdigest()
                name = f"prompt-{ordinal:06d}.safetensors"
                save_file({key: feature}, staging / name, metadata=metadata)
                shards[name] = digest_file(staging / name)
                entries[key] = {"shard": name, "tensor": key, "tokens": len(ids)}
                del feature
                print(json.dumps({"event": "prompt_encoded", "ordinal": ordinal, "prompt_sha256": key,
                                  "tokens": len(ids)}), flush=True)
    finally:
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    payload = dict(schema_version="visual_rl.h3-text-cache.v1", identity=identity, shards=shards, entries=entries)
    (staging / "index.json").write_bytes(canonical(payload) + b"\n")
    cache = H3TextCache(staging, expected_identity=identity, max_tokens=args.max_tokens,
                        max_embedding_bytes=args.max_tokens * 5120 * 4)
    cache.verify_coverage(*groups)
    provenance = dict(identity=identity, processor=processor, encoder_files=encoder_files,
                      tokenizer_files=tokenizer_files, prompt_file_sha256=prompt_hashes,
                      unique_prompts=len(prompts), max_tokens=args.max_tokens,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      validation="real encoder outputs, strict cache reader and train/eval coverage; not training")
    (staging / "provenance.json").write_bytes(canonical(provenance) + b"\n")
    if output.exists():
        raise FileExistsError("output appeared during preparation")
    staging.rename(output)
    print(json.dumps({"event": "cache_ready", "output": str(output), **provenance}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root", "model", "train", "eval", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--max-prompts", type=int, required=True)
    build(p.parse_args())


if __name__ == "__main__":
    main()
