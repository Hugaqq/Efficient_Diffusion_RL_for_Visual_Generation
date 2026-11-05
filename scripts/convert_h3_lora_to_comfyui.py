"""Export an H3 training LoRA and convert it to the ComfyUI MiniMax-H3 LoRA format.

CPU only. Live export reads step-N DCP LoRA tensors with no process group.
--weights ema validates the complete checkpoint tree, then reconstructs the
rank-local CPU FP32 EMA shards; it never substitutes live weights if EMA is absent.

Training side (framecode, PEFT, Diffusers MiniMaxH3Transformer3DModel):
  transformer_blocks.{i}.{attn.to_q, attn.to_k, attn.to_v, attn.to_out.0,
                          ff.net.0.proj, ff.net.2}
  lora_A [r, in], lora_B [out, r], FP32 master, scaling = alpha / r
  (r=32, alpha=64 -> 2), delta W = scaling * B @ A.
ComfyUI side (comfy/ldm/minimax/model.py, native key names):
  diffusion_model.blocks.{i}.attn.qkv_proj   forward splits rows contiguously [q; k; v]
  diffusion_model.blocks.{i}.attn.out_proj
  diffusion_model.blocks.{i}.mlp.fc1         swiglu reads rows as [gate; up]
  diffusion_model.blocks.{i}.mlp.fc2
  delta W = (alpha / lora_A.shape[0]) * lora_B @ lora_A
Mapping (exact, only concatenation/zero padding/row permutation):
  qkv_proj: A = cat(A_q, A_k, A_v) [3r, in]; B = blockdiag(B_q, B_k, B_v) [3*inner, 3r];
            alpha = scaling * 3r, so B@A*alpha/(3r) = cat(dW_q, dW_k, dW_v).
  out_proj, fc2: unchanged, alpha = scaling * r.
  fc1:      Diffusers SwiGLU rows are [value; gate]; native/ComfyUI rows are
            [gate; up], so B rows are swapped halves; A unchanged.
  token_refiner blocks: not trained (no LoRA) -> omitted (= zero delta).

Subcommands:
  export      step-N checkpoint (dir containing h3-state/dcp) -> training-key safetensors
  convert     training-key safetensors (or --checkpoint directly) -> ComfyUI LoRA safetensors
  --self-test random-tensor CPU check that both representations give the same delta W
              (and the same layer outputs for q/k/v split and SwiGLU).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
_TRAIN_KEY = re.compile(r"^(?:model\.)?transformer_blocks\.(\d+)\.(.+)\.lora_([AB])(?:\.default)?\.weight$")
_COMFY_PREFIX = "diffusion_model."


def _torch():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    return torch


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(tensor):
    torch = _torch()
    t = tensor.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()


# ---------------------------------------------------------------- training LoRA

def normalize_training(state):
    """Return {block: {target: {"A": t, "B": t}}}; reject anything unexpected."""
    blocks = {}
    for name, tensor in state.items():
        match = _TRAIN_KEY.match(name)
        if not match or match.group(2) not in TARGETS:
            raise ValueError(f"unexpected training LoRA key: {name}")
        slot = blocks.setdefault(int(match.group(1)), {}).setdefault(match.group(2), {})
        if match.group(3) in slot:
            raise ValueError(f"duplicate LoRA tensor: {name}")
        slot[match.group(3)] = tensor
    if not blocks or sorted(blocks) != list(range(len(blocks))):
        raise ValueError("LoRA blocks must be a contiguous 0..N-1 range")
    rank = None
    for index, targets in blocks.items():
        if set(targets) != set(TARGETS):
            raise ValueError(f"block {index} does not have exactly the six LoRA targets")
        for target, pair in targets.items():
            if set(pair) != {"A", "B"}:
                raise ValueError(f"block {index} {target} lacks lora_A or lora_B")
            a, b = pair["A"], pair["B"]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError(f"block {index} {target} has incompatible A/B shapes")
            rank = a.shape[0] if rank is None else rank
            if a.shape[0] != rank:
                raise ValueError("mixed LoRA ranks are unsupported")
        q, k, v = (targets[t] for t in ("attn.to_q", "attn.to_k", "attn.to_v"))
        if not (q["A"].shape == k["A"].shape == v["A"].shape and q["B"].shape == k["B"].shape == v["B"].shape):
            raise ValueError(f"block {index} q/k/v LoRA shapes differ")
        if targets["ff.net.0.proj"]["B"].shape[0] % 2:
            raise ValueError("ff.net.0.proj output must be even (fused SwiGLU)")
    return blocks, rank


def training_delta(pair, scaling):
    return (pair["B"] @ pair["A"]) * scaling


# ---------------------------------------------------------------- conversion

def convert(blocks, rank, alpha, dtype=None):
    """Training blocks -> ComfyUI state dict (diffusion_model.* keys)."""
    torch = _torch()
    scaling = alpha / rank
    out = {}

    def put(module, a, b, module_alpha):
        prefix = f"{_COMFY_PREFIX}{module}"
        out[f"{prefix}.lora_A.weight"] = a.contiguous() if dtype is None else a.to(dtype).contiguous()
        out[f"{prefix}.lora_B.weight"] = b.contiguous() if dtype is None else b.to(dtype).contiguous()
        out[f"{prefix}.alpha"] = torch.tensor(float(module_alpha), dtype=torch.float32)

    for index in sorted(blocks):
        t = blocks[index]
        q, k, v = (t[name] for name in ("attn.to_q", "attn.to_k", "attn.to_v"))
        a = torch.cat([q["A"], k["A"], v["A"]], dim=0)
        b = torch.block_diag(q["B"], k["B"], v["B"])
        put(f"blocks.{index}.attn.qkv_proj", a, b, scaling * a.shape[0])
        put(f"blocks.{index}.attn.out_proj", t["attn.to_out.0"]["A"], t["attn.to_out.0"]["B"], scaling * rank)
        fc1_b = t["ff.net.0.proj"]["B"]
        value, gate = fc1_b.chunk(2, dim=0)
        put(f"blocks.{index}.mlp.fc1", t["ff.net.0.proj"]["A"], torch.cat([gate, value], dim=0), scaling * rank)
        put(f"blocks.{index}.mlp.fc2", t["ff.net.2"]["A"], t["ff.net.2"]["B"], scaling * rank)
    return out


def comfy_delta(state, module, dtype=None):
    """delta W as ComfyUI's LoRAAdapter computes it: lora_B @ lora_A * alpha / lora_A.shape[0]
    (ComfyUI uses an FP32 intermediate; dtype=float64 is only for the functional self-check)."""
    torch = _torch()
    dtype = torch.float32 if dtype is None else dtype
    prefix = f"{_COMFY_PREFIX}{module}"
    a = state[f"{prefix}.lora_A.weight"].to(dtype)
    b = state[f"{prefix}.lora_B.weight"].to(dtype)
    alpha = float(state[f"{prefix}.alpha"].item()) / a.shape[0]
    return (b @ a) * alpha


def expected_native_delta(blocks, index, module, scaling):
    """Training delta W placed in the ComfyUI/native row layout."""
    torch = _torch()
    t = blocks[index]
    if module == "attn.qkv_proj":
        return torch.cat([training_delta(t[n], scaling) for n in ("attn.to_q", "attn.to_k", "attn.to_v")], dim=0)
    if module == "attn.out_proj":
        return training_delta(t["attn.to_out.0"], scaling)
    if module == "mlp.fc1":
        value, gate = training_delta(t["ff.net.0.proj"], scaling).chunk(2, dim=0)
        return torch.cat([gate, value], dim=0)
    if module == "mlp.fc2":
        return training_delta(t["ff.net.2"], scaling)
    raise ValueError(module)


MODULES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def verify(blocks, rank, alpha, comfy_state, indices):
    torch = _torch()
    scaling = alpha / rank
    worst = 0.0
    rows = []
    for index in indices:
        for module in MODULES:
            want = expected_native_delta(blocks, index, module, scaling).double()
            got = comfy_delta(comfy_state, f"blocks.{index}.{module}").double()
            if want.shape != got.shape:
                raise AssertionError(f"block {index} {module}: shape {tuple(got.shape)} != {tuple(want.shape)}")
            denom = want.abs().max().item()
            err = (got - want).abs().max().item()
            rel = err / denom if denom else err
            worst = max(worst, rel)
            rows.append(dict(block=index, module=module, max_abs_err=err, max_abs_delta=denom, rel=rel))
    return worst, rows


# ---------------------------------------------------------------- IO

def load_dcp_lora(checkpoint, *, weights='live'):
    """Read the requested LoRA view on CPU; never fall back between live and EMA."""
    torch = _torch()
    if weights == 'ema':
        from visual_rl.artifacts.checkpoint.h3_ema import export_ema_lora
        return export_ema_lora(checkpoint)
    if weights != 'live':
        raise ValueError('weights must be live or ema')
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    path = checkpoint
    for candidate in (os.path.join(checkpoint, "h3-state", "dcp"), os.path.join(checkpoint, "dcp")):
        if os.path.isdir(candidate):
            path = candidate
            break
    reader = FileSystemReader(path)
    metadata = reader.read_metadata()
    model = {}
    for key, item in metadata.state_dict_metadata.items():
        if not key.startswith("model."):
            continue
        if not _TRAIN_KEY.match(key):
            raise ValueError(f"unexpected model entry in DCP (not a LoRA tensor): {key}")
        model[key[len("model."):]] = torch.empty(tuple(item.size), dtype=item.properties.dtype)
    if not model:
        raise ValueError("DCP has no model.* LoRA entries")
    state = {"model": model}
    dcp.load(state, storage_reader=reader, no_dist=True)
    return state["model"], path


def _checkpoint_identity(checkpoint):
    info = {}
    for name in ("h3_transaction.json", "h3-state/manifest.json", "h3-state/h3_state.json"):
        path = os.path.join(checkpoint, name)
        if os.path.isfile(path):
            info[name] = _sha_file(path)
    meta = os.path.join(load_dcp_path(checkpoint), ".metadata")
    if os.path.isfile(meta):
        info["dcp/.metadata"] = _sha_file(meta)
    return info


def load_dcp_path(checkpoint):
    candidate = os.path.join(checkpoint, "h3-state", "dcp")
    return candidate if os.path.isdir(candidate) else checkpoint


def save(state, path, metadata):
    from safetensors.torch import save_file
    tmp = path + ".tmp"
    save_file(state, tmp, metadata={k: (v if isinstance(v, str) else json.dumps(v, sort_keys=True))
                                    for k, v in metadata.items()})
    os.replace(tmp, path)
    return _sha_file(path)


def load_training_file(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as handle:
        meta = handle.metadata() or {}
        state = {k: handle.get_tensor(k) for k in handle.keys()}
    return state, meta


# ---------------------------------------------------------------- self test

def self_test():
    torch = _torch()
    gen = torch.Generator().manual_seed(1234)
    heads, head_dim, hidden, ffn, rank, alpha, nblocks = 3, 4, 10, 7, 3, 5, 2
    inner = heads * head_dim
    shapes = {"attn.to_q": (inner, hidden), "attn.to_k": (inner, hidden), "attn.to_v": (inner, hidden),
              "attn.to_out.0": (hidden, inner), "ff.net.0.proj": (2 * ffn, hidden), "ff.net.2": (hidden, ffn)}
    train = {}
    base = {}
    for i in range(nblocks):
        for target, (out_f, in_f) in shapes.items():
            train[f"model.transformer_blocks.{i}.{target}.lora_A.default.weight"] = torch.randn(rank, in_f, generator=gen)
            train[f"model.transformer_blocks.{i}.{target}.lora_B.default.weight"] = torch.randn(out_f, rank, generator=gen)
            base[(i, target)] = torch.randn(out_f, in_f, generator=gen)
    blocks, r = normalize_training(train)
    assert r == rank
    comfy = convert(blocks, rank, alpha)
    expected_keys = {f"{_COMFY_PREFIX}blocks.{i}.{m}.{s}" for i in range(nblocks) for m in MODULES
                     for s in ("lora_A.weight", "lora_B.weight", "alpha")}
    assert set(comfy) == expected_keys, "unexpected ComfyUI key set"
    worst, _ = verify(blocks, rank, alpha, comfy, range(nblocks))
    assert worst <= 1e-6, f"delta W mismatch, relative error {worst}"
    # Functional check: training forward (separate PEFT layers, Diffusers SwiGLU) vs ComfyUI
    # forward (fused qkv split contiguously, merged weights, swiglu on [gate; up]).
    scaling = alpha / rank
    x = torch.randn(5, hidden, generator=gen, dtype=torch.float64)
    for i in range(nblocks):
        t = blocks[i]
        def peft(target, inp):
            p = t[target]
            return inp @ base[(i, target)].double().T + (inp @ p["A"].double().T) @ p["B"].double().T * scaling
        q, k, v = (peft(n, x) for n in ("attn.to_q", "attn.to_k", "attn.to_v"))
        w_qkv = torch.cat([base[(i, n)] for n in ("attn.to_q", "attn.to_k", "attn.to_v")]).double()
        w_qkv = w_qkv + comfy_delta(comfy, f"blocks.{i}.attn.qkv_proj", torch.float64)
        cq, ck, cv = (x @ w_qkv.T).split(inner, dim=-1)
        for a, b in ((q, cq), (k, ck), (v, cv)):
            assert torch.allclose(a, b, rtol=1e-6, atol=1e-9), "qkv forward mismatch"
        h, g = peft("ff.net.0.proj", x).chunk(2, dim=-1)
        train_ff = h * torch.nn.functional.silu(g)
        value, gate = base[(i, "ff.net.0.proj")].chunk(2, dim=0)
        w_fc1 = torch.cat([gate, value]).double() + comfy_delta(comfy, f"blocks.{i}.mlp.fc1", torch.float64)
        cg, cu = (x @ w_fc1.T).chunk(2, dim=-1)
        comfy_ff = torch.nn.functional.silu(cg) * cu
        assert torch.allclose(train_ff, comfy_ff, rtol=1e-6, atol=1e-9), "SwiGLU forward mismatch"
    # bf16 storage option keeps the mapping (values rounded once, nothing else).
    comfy_bf16 = convert(blocks, rank, alpha, dtype=torch.bfloat16)
    assert all(comfy_bf16[k].dtype == torch.bfloat16 for k in comfy_bf16 if not k.endswith(".alpha"))
    # Real-geometry spot check: H3 dims, one block, scaling 64/32.
    real = {}
    rshapes = {"attn.to_q": (7168, 5376), "attn.to_k": (7168, 5376), "attn.to_v": (7168, 5376),
               "attn.to_out.0": (5376, 7168), "ff.net.0.proj": (28672, 5376), "ff.net.2": (5376, 14336)}
    for target, (out_f, in_f) in rshapes.items():
        real[f"transformer_blocks.0.{target}.lora_A.default.weight"] = torch.randn(32, in_f, generator=gen) * 0.01
        real[f"transformer_blocks.0.{target}.lora_B.default.weight"] = torch.randn(out_f, 32, generator=gen) * 1e-4
    rblocks, rr = normalize_training(real)
    rcomfy = convert(rblocks, rr, 64)
    rworst, _ = verify(rblocks, rr, 64, rcomfy, [0])
    assert rworst <= 1e-6, f"real-geometry delta W mismatch {rworst}"
    shapes_out = {k: tuple(v.shape) for k, v in rcomfy.items() if "qkv" in k}
    assert shapes_out[f"{_COMFY_PREFIX}blocks.0.attn.qkv_proj.lora_B.weight"] == (21504, 96)
    assert float(rcomfy[f"{_COMFY_PREFIX}blocks.0.attn.qkv_proj.alpha"]) == 192.0
    print(json.dumps(dict(self_test="pass", toy_rel_err=worst, h3_geometry_rel_err=rworst,
                          qkv_lora_B_shape=[21504, 96], qkv_alpha=192.0, other_alpha=64.0)))


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="mode")
    e = sub.add_parser("export", help="step-N DCP -> training-key LoRA safetensors (FP32)")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--lora-alpha", type=float, default=64.0)
    e.add_argument('--weights', choices=('live', 'ema'), default='live')
    c = sub.add_parser("convert", help="training LoRA -> ComfyUI LoRA safetensors")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="training-key LoRA safetensors (from export)")
    src.add_argument("--checkpoint", help="step-N checkpoint directory (reads DCP directly)")
    c.add_argument("--output", required=True)
    c.add_argument("--lora-alpha", type=float, help="training alpha; default: export metadata, else 64")
    c.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    c.add_argument('--weights', choices=('live', 'ema'), default=None,
                   help='checkpoint: default live; input file: infer its declared view')
    c.add_argument("--verify-blocks", default="first,last", help="'all', 'none' or comma list / first,last")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.mode is None:
        parser.error("choose --self-test, export or convert")
    torch = _torch()
    started = time.time()
    if args.mode == "export":
        state, dcp_path = load_dcp_lora(args.checkpoint, weights=args.weights)
        blocks, rank = normalize_training(state)
        meta = dict(format="h3-training-lora.v1", source_checkpoint=os.path.abspath(args.checkpoint),
                    source_dcp=os.path.abspath(dcp_path), identity=_checkpoint_identity(args.checkpoint),
                    lora_rank=str(rank), lora_alpha=str(args.lora_alpha), blocks=str(len(blocks)),
                    dtype=str(next(iter(state.values())).dtype))
        if args.weights == 'ema':
            meta['weights'] = 'ema'
        sha = save({k: v.contiguous() for k, v in sorted(state.items())}, args.output, meta)
        print(json.dumps(dict(output=args.output, sha256=sha, tensors=len(state), rank=rank, blocks=len(blocks),
                              seconds=round(time.time() - started, 1), identity=meta["identity"])))
        return 0
    if args.input:
        state, meta = load_training_file(args.input)
        weights = meta.get('weights', 'live')
        if weights not in ('live', 'ema') or (args.weights is not None and args.weights != weights):
            raise ValueError('requested weights differ from the input LoRA view')
        source = dict(source_file=os.path.abspath(args.input), source_sha256=_sha_file(args.input),
                      source_meta=meta)
        alpha = args.lora_alpha if args.lora_alpha is not None else float(meta.get("lora_alpha", 64))
    else:
        weights = args.weights or 'live'
        state, dcp_path = load_dcp_lora(args.checkpoint, weights=weights)
        source = dict(source_checkpoint=os.path.abspath(args.checkpoint), identity=_checkpoint_identity(args.checkpoint))
        alpha = args.lora_alpha if args.lora_alpha is not None else 64.0
    blocks, rank = normalize_training(state)
    dtype = torch.bfloat16 if args.dtype == "bf16" else None
    comfy = convert(blocks, rank, alpha, dtype=dtype)
    if args.verify_blocks == "none":
        indices = []
    elif args.verify_blocks == "all":
        indices = sorted(blocks)
    else:
        indices = sorted({0 if p == "first" else max(blocks) if p == "last" else int(p)
                          for p in args.verify_blocks.split(",") if p})
    worst, rows = verify(blocks, rank, alpha, comfy, indices) if indices else (None, [])
    if indices and args.dtype == "fp32" and worst > 1e-6:
        raise SystemExit(f"verification failed: relative delta-W error {worst}")
    meta = dict(format="comfyui-minimax-h3-lora", converter="framecode scripts/convert_h3_lora_to_comfyui.py",
                training_rank=str(rank), training_alpha=str(alpha), scaling=str(alpha / rank),
                qkv="A=cat(A_q,A_k,A_v); B=blockdiag(B_q,B_k,B_v); alpha=scaling*3r",
                fc1="B rows [value;gate]->[gate;value]", dtype=args.dtype, source=source)
    if weights == 'ema':
        meta['weights'] = 'ema'
    sha = save(comfy, args.output, meta)
    print(json.dumps(dict(output=args.output, sha256=sha, tensors=len(comfy), rank=rank, alpha=alpha,
                          verified_blocks=indices, worst_rel_err=worst, dtype=args.dtype,
                          seconds=round(time.time() - started, 1))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
