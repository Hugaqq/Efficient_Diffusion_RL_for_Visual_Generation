"""Per-tensor SHA256 of H3 transformer weights in the ComfyUI/native key space.

Standard library only (no torch, no CUDA): reads safetensors headers and raw
tensor bytes with pread, so the same file runs on the training host and on the
A100 evaluation host.

  comfy     hash every tensor of a ComfyUI single-file checkpoint (raw bytes).
  artifact  hash the Diffusers H3 transformer directory used for training,
            after mapping each tensor to its ComfyUI/native name and layout:
              transformer_blocks.i.attn.to_{q,k,v}  -> blocks.i.attn.qkv_proj
                  ("contiguous": [q; k; v] rows, what ComfyUI's forward splits;
                   "interleaved": per-head [q_h; k_h; v_h], the raw release layout)
              transformer_blocks.i.attn.to_out.0    -> blocks.i.attn.out_proj
              transformer_blocks.i.ff.net.0.proj    -> blocks.i.mlp.fc1 (halves swapped:
                  Diffusers [value; gate] -> native [gate; value])
              transformer_blocks.i.ff.net.2         -> blocks.i.mlp.fc2
              (same for token_refiner.refiner_blocks.j -> token_refiner.blocks.j)
            Byte reordering only; no dtype conversion, so equal hashes mean
            bit-identical tensors.
  compare   compare two hash JSON files: equal / different / one-sided.

Example (training host):
  nice -n 19 ionice -c3 python scripts/h3_comfy_tensor_hashes.py artifact \
      --dir .../artifact/transformer --qkv-layout both --out artifact-hashes.json
Example (A100):
  nice -n 19 python3 scripts/h3_comfy_tensor_hashes.py comfy \
      --file .../minimax_h3_fl2va_pruned_bf16.safetensors --out comfy-hashes.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import struct
import sys
import time

_CHUNK = 64 * 1024 * 1024
_ITEM = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1, "I8": 1, "BOOL": 1,
         "F8_E4M3": 1, "F8_E5M2": 1}
_PREFIX = "diffusion_model."


def read_header(path):
    with open(path, "rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
    metadata = header.pop("__metadata__", None)
    base = 8 + length
    for name, info in header.items():
        begin, end = info["data_offsets"]
        info["file"], info["begin"], info["end"] = path, base + begin, base + end
        expected = _ITEM.get(info["dtype"])
        count = 1
        for size in info["shape"]:
            count *= size
        if expected is not None and expected * count != end - begin:
            raise ValueError(f"{path}: {name} byte length does not match dtype/shape")
    return header, metadata


class _Hasher:
    def __init__(self):
        self.sha = hashlib.sha256()
        self.nbytes = 0
        self.fds = {}

    def add(self, info, start=0, stop=None):
        """Hash bytes [start, stop) of one tensor (offsets relative to the tensor)."""
        size = info["end"] - info["begin"]
        stop = size if stop is None else stop
        if not 0 <= start <= stop <= size:
            raise ValueError("byte range outside tensor")
        fd = self.fds.get(info["file"])
        if fd is None:
            fd = self.fds[info["file"]] = os.open(info["file"], os.O_RDONLY)
        position = info["begin"] + start
        end = info["begin"] + stop
        while position < end:
            block = os.pread(fd, min(_CHUNK, end - position), position)
            if not block:
                raise IOError("short read")
            self.sha.update(block)
            self.nbytes += len(block)
            position += len(block)

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


def _row_bytes(info):
    rows = info["shape"][0]
    return (info["end"] - info["begin"]) // rows


def _entry(sha, dtype, shape, nbytes):
    return {"sha256": sha, "dtype": dtype, "shape": list(shape), "bytes": nbytes}


def hash_comfy(path, names=None):
    header, metadata = read_header(path)
    result = {}
    for name in sorted(header):
        short = name[len(_PREFIX):] if name.startswith(_PREFIX) else name
        if names and short not in names:
            continue
        info = header[name]
        hasher = _Hasher()
        try:
            hasher.add(info)
        finally:
            hasher.close()
        result[short] = _entry(hasher.sha.hexdigest(), info["dtype"], info["shape"], hasher.nbytes)
    return result, metadata


_BLOCK = re.compile(r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")
_RENAMES = (
    ("proj_in.", "video_patch_proj."), ("audio_proj_in.", "audio_patch_proj."),
    ("context_embedder.", "condition_proj."), ("time_embedder.linear_1.", "time_embedder.proj_in."),
    ("time_embedder.linear_2.", "time_embedder.proj_out."), ("norm_out.norm.", "final_layer.norm."),
    ("norm_out.linear.", "final_layer.adaln_proj.linear."), ("proj_out.", "final_layer.video_out."),
    ("audio_proj_out.", "final_layer.audio_out."),
)


def native_name(name):
    """Diffusers H3 tensor name -> native/ComfyUI name (qkv handled by the caller)."""
    match = _BLOCK.match(name)
    if match:
        root = "blocks" if match.group(1) == "transformer_blocks" else "token_refiner.blocks"
        rest = match.group(3)
        rest = (rest.replace("attn.norm_q.", "attn.q_norm.").replace("attn.norm_k.", "attn.k_norm.")
                .replace("attn.to_out.0.", "attn.out_proj.").replace("ff.net.0.proj.", "mlp.fc1.")
                .replace("ff.net.2.", "mlp.fc2."))
        return f"{root}.{match.group(2)}.{rest}"
    for old, new in _RENAMES:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def hash_artifact(directory, layouts, heads, head_dim, names=None):
    header = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        part, _ = read_header(path)
        if set(part) & set(header):
            raise ValueError("duplicate tensor across shards")
        header.update(part)
    if not header:
        raise ValueError("no safetensors files in artifact directory")
    result = {}
    consumed = set()
    for name in sorted(header):
        if name in consumed:
            continue
        match = _BLOCK.match(name)
        if match and match.group(3) in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"):
            prefix = name.rsplit(".attn.", 1)[0] + ".attn."
            parts = [header[prefix + f"to_{x}.weight"] for x in "qkv"]
            consumed.update(prefix + f"to_{x}.weight" for x in "qkv")
            target = native_name(prefix + "qkv_proj.weight")
            if names and target not in names:
                continue
            rows = parts[0]["shape"][0]
            if rows != heads * head_dim or any(p["shape"] != parts[0]["shape"] or p["dtype"] != parts[0]["dtype"] for p in parts):
                raise ValueError(f"unexpected q/k/v shapes for {prefix}")
            shape = [3 * rows] + parts[0]["shape"][1:]
            row = _row_bytes(parts[0])
            for layout in layouts:
                hasher = _Hasher()
                try:
                    if layout == "contiguous":
                        for part in parts:
                            hasher.add(part)
                    else:
                        for head in range(heads):
                            for part in parts:
                                hasher.add(part, head * head_dim * row, (head + 1) * head_dim * row)
                finally:
                    hasher.close()
                key = target if layout == layouts[0] else f"{target}#{layout}"
                result[key] = _entry(hasher.sha.hexdigest(), parts[0]["dtype"], shape, hasher.nbytes)
            continue
        consumed.add(name)
        info = header[name]
        target = native_name(name)
        if names and target not in names:
            continue
        hasher = _Hasher()
        try:
            if target.endswith(".mlp.fc1.weight"):
                half = (info["end"] - info["begin"]) // 2
                hasher.add(info, half, None)  # gate half first (native [gate; value])
                hasher.add(info, 0, half)
            else:
                hasher.add(info)
        finally:
            hasher.close()
        result[target] = _entry(hasher.sha.hexdigest(), info["dtype"], info["shape"], hasher.nbytes)
    return result


def compare(left, right):
    a = {k: v for k, v in left["tensors"].items() if "#" not in k}
    b = {k: v for k, v in right["tensors"].items() if "#" not in k}
    alt = {k.split("#")[0]: v for side in (left, right) for k, v in side["tensors"].items() if "#" in k}
    report = {"equal": [], "different": [], "only_left": sorted(set(a) - set(b)), "only_right": sorted(set(b) - set(a))}
    for name in sorted(set(a) & set(b)):
        if a[name]["sha256"] == b[name]["sha256"]:
            report["equal"].append(name)
        else:
            entry = {"name": name, "left": {k: a[name][k] for k in ("dtype", "shape")},
                     "right": {k: b[name][k] for k in ("dtype", "shape")}}
            if name in alt and alt[name]["sha256"] in (a[name]["sha256"], b[name]["sha256"]):
                entry["matches_alternate_layout"] = True
            report["different"].append(entry)
    return report


def _pattern(names):
    counts = {}
    for name in names:
        key = re.sub(r"\.\d+\.", ".N.", name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("comfy")
    c.add_argument("--file", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--names", nargs="*")
    a = sub.add_parser("artifact")
    a.add_argument("--dir", required=True, help="Diffusers transformer directory")
    a.add_argument("--out", required=True)
    a.add_argument("--qkv-layout", choices=("contiguous", "interleaved", "both"), default="both")
    a.add_argument("--heads", type=int, default=56)
    a.add_argument("--head-dim", type=int, default=128)
    a.add_argument("--names", nargs="*")
    m = sub.add_parser("compare")
    m.add_argument("left")
    m.add_argument("right")
    m.add_argument("--out")
    args = parser.parse_args(argv)
    if args.mode == "compare":
        with open(args.left) as fl, open(args.right) as fr:
            report = compare(json.load(fl), json.load(fr))
        summary = {"equal": len(report["equal"]), "different": len(report["different"]),
                   "only_left": _pattern(report["only_left"]), "only_right": _pattern(report["only_right"]),
                   "different_patterns": _pattern(d["name"] for d in report["different"])}
        print(json.dumps(summary, indent=1))
        if args.out:
            with open(args.out, "w") as stream:
                json.dump(dict(summary=summary, **report), stream, indent=1)
        return 0
    started = time.time()
    names = set(args.names) if args.names else None
    if args.mode == "comfy":
        tensors, metadata = hash_comfy(args.file, names)
        source = dict(kind="comfy", file=os.path.abspath(args.file), bytes=os.path.getsize(args.file), metadata=metadata)
    else:
        layouts = ("contiguous", "interleaved") if args.qkv_layout == "both" else (args.qkv_layout,)
        tensors = hash_artifact(args.dir, layouts, args.heads, args.head_dim, names)
        source = dict(kind="artifact", dir=os.path.abspath(args.dir), qkv_layouts=list(layouts),
                      note="qkv_proj key = first layout; '#interleaved' = alternate layout")
    out = dict(source=source, seconds=round(time.time() - started, 1),
               total_bytes=sum(v["bytes"] for k, v in tensors.items() if "#" not in k), tensors=tensors)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as stream:
        json.dump(out, stream, indent=1, sort_keys=True)
    os.replace(tmp, args.out)
    print(json.dumps(dict(out=args.out, tensors=len(tensors), seconds=out["seconds"], total_bytes=out["total_bytes"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
