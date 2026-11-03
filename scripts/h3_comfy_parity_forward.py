"""Inference-only framecode H3 forwards for the ComfyUI parity check (no training, no optimizer).

prepare (CPU): fixed initial noise (FP16-representable, ComfyUI layout) and the training text-cache
               embedding of one prompt, so both stacks consume byte-identical inputs.
run (GPU):     the training transformer (Diffusers H3 + PEFT LoRA built by framecode's own
               build_h3_meta; base weights from the training artifact; LoRA cast to bf16 exactly as
               FSDP2 param_dtype=bf16 does; FP32 roots stay FP32), split over two cards with
               accelerate hooks (per-layer arithmetic is the same as the FSDP forward, only the
               parameter placement differs). For each LoRA variant (zero, x1, xK) it saves the
               noise-ward prediction at sigma=1 and the terminal latent of the training ODE
               (H3DualFlowSDE.deterministic_step, FP16 state) on the shared 10-step grid.
Outputs use ComfyUI layout: video [1,24,T,H/16,W/16], audio [1,32,2,A].
"""
from __future__ import annotations

import argparse
import json
import os
import time


def _cfg(path):
    with open(path) as f:
        return json.load(f)


def to_rows(video, audio):
    """ComfyUI latents -> framecode joint rows (inverse of h3_decode / comfy patchify)."""
    _, c, t, hh, ww = video.shape
    v = video.reshape(1, c, t, hh // 2, 2, ww // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(1, t * (hh // 2) * (ww // 2), c * 4)
    a = audio[0].permute(1, 2, 0).reshape(1, audio.shape[2] * audio.shape[3], audio.shape[1])
    return v, a


def from_rows(v, a, shape_v, shape_a):
    _, c, t, hh, ww = shape_v
    video = v.float().reshape(1, t, hh // 2, ww // 2, c, 2, 2).permute(0, 4, 1, 2, 5, 3, 6).reshape(shape_v)
    audio = a.float().reshape(shape_a[2], shape_a[3], shape_a[1]).permute(2, 0, 1).unsqueeze(0)
    return video.contiguous(), audio.contiguous()


def prepare(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from safetensors.torch import save_file
    from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry
    from visual_rl.models.implementations.h3_text_cache import H3TextCache
    geo = H3T2VAGeometry(args.height, args.width, 124)
    c, t, h, w = geo.latent_shape
    g = torch.manual_seed(args.seed)
    video = torch.randn((1, c, t, h, w), dtype=torch.float32, generator=g).half().float()
    audio = torch.randn((1, 32, 2, geo.audio_latents), dtype=torch.float32, generator=g).half().float()
    v, a = to_rows(video, audio)
    assert v.shape == (1, geo.joint_layout.video_rows, 96) and a.shape == (1, geo.joint_layout.audio_rows, 32)
    rv, ra = from_rows(v, a, video.shape, audio.shape)
    assert torch.equal(rv, video) and torch.equal(ra, audio), "layout bridge round trip failed"
    os.makedirs(args.out, exist_ok=True)
    save_file({"t0": video, "t1": audio}, os.path.join(args.out, f"noise_{args.width}x{args.height}_seed{args.seed}.safetensors"))
    cfg = _cfg(args.config)
    cache = H3TextCache(cfg["launch"]["artifacts"]["text_cache"], expected_identity=cfg["launch"]["expected"]["cache_identity"],
                        max_tokens=cfg["budgets"]["max_tokens"], max_embedding_bytes=cfg["budgets"]["embedding_bytes"])
    prompts = [json.loads(line) for line in open(args.prompts)]
    p = next(x for x in prompts if x["eval_id"] == args.eval_id)
    emb = cache.get(p["prompt"]).float().contiguous()
    save_file({"cond": emb}, os.path.join(args.out, f"cache_{args.eval_id}.safetensors"),
              metadata={"prompt_sha256": p["prompt_sha256"], "cache_identity": cache.identity_sha256})
    print(json.dumps(dict(video=list(video.shape), audio=list(audio.shape), cond=list(emb.shape), cache=cache.identity_sha256)))


def run(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from accelerate import dispatch_model
    from visual_rl.models.implementations.h3_loading import build_h3_meta
    from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry
    from visual_rl.models.implementations.h3_forward import h3_t2va_forward
    from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
    from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
    t_start = time.time()
    meta = build_h3_meta(args.artifact, rank=32, alpha=64)
    model = meta.model
    tdir = os.path.join(args.artifact, "transformer")
    index = _cfg(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    state = {}
    for shard in sorted(set(index.values())):
        with safe_open(os.path.join(tdir, shard), "pt", device="cpu") as f:
            for name in f.keys():
                target = meta.base_to_model[name]
                state[target] = f.get_tensor(name).to(meta.state_specs[target].dtype)
    lora_fp32 = load_file(args.lora)
    lora = {}
    for name in meta.lora_names:
        key = name if name in lora_fp32 else name.replace("model.", "", 1)
        lora[name] = lora_fp32[key]
    for name, t in lora.items():
        state[name] = torch.zeros_like(t)
    assert set(state) == set(meta.state_specs), "state does not cover the model"
    model.load_state_dict(state, assign=True, strict=True)
    del state
    for name, buf in meta.nonpersistent_buffers.items():
        mod, _, leaf = name.rpartition(".")
        model.get_submodule(mod).register_buffer(leaf, buf.clone(), persistent=False)
    params = dict(model.named_parameters())
    for name in meta.lora_names:  # FSDP2 MixedPrecisionPolicy(param_dtype=bf16) forward dtype
        params[name].data = params[name].data.to(torch.bfloat16)
    nblocks = len(meta.block_names)
    device_map = {n: 0 for n, _ in model.named_children() if n != "transformer_blocks"}
    device_map.update({f"transformer_blocks.{i}": (0 if i < args.split else 1) for i in range(nblocks)})
    model.eval()
    model = dispatch_model(model, device_map=device_map, main_device=0)
    params = dict(model.named_parameters())  # placement may replace Parameter objects
    assert all(params[n].dtype == torch.bfloat16 and params[n].device.type == "cuda" for n in meta.lora_names)
    t_loaded = time.time()
    noise = load_file(args.noise)
    video0, audio0 = noise["t0"], noise["t1"]
    geo = H3T2VAGeometry(args.height, args.width, 124)
    layout = geo.joint_layout
    v_rows, a_rows = to_rows(video0, audio0)
    x0 = layout.pack(v_rows.half(), a_rows.half()).to("cuda:0")
    emb = load_file(args.cond)["cond"].to("cuda:0")
    sched = H3ScheduleSnapshot.build(args.steps)
    dyn = H3DualFlowSDE(layout, sched)
    fwd = lambda **kw: model(**kw)

    def predict(x, i):
        vs, _, as_, _ = sched.transition(i)
        return h3_t2va_forward(fwd, x=x, geometry=geo, text_embeddings=emb, video_sigma=vs, audio_sigma=as_)

    out, timing = {}, {}
    variants = [("base", 0.0)] + [(f"lora{k:g}", k) for k in args.scales]
    for tag, k in variants:
        for name in meta.lora_names:
            src = lora[name]
            val = src * k if ".lora_B." in name else src
            params[name].data.copy_(val.to(torch.bfloat16))
        t0 = time.time()
        with torch.no_grad():
            pred = predict(x0, 0)
            pv, pa = layout.unpack(pred.float())
            out[f"{tag}.v1.t0"], out[f"{tag}.v1.t1"] = from_rows(pv.cpu(), pa.cpu(), video0.shape, audio0.shape)
            x = x0
            for i in range(sched.num_steps):
                p = pred if i == 0 else predict(x, i)
                x = dyn.deterministic_step(x, p, i)
            fv, fa = layout.unpack(x.float())
            out[f"{tag}.final.t0"], out[f"{tag}.final.t1"] = from_rows(fv.cpu(), fa.cpu(), video0.shape, audio0.shape)
        torch.cuda.synchronize()
        timing[tag] = round(time.time() - t0, 1)
        print(json.dumps(dict(variant=tag, scale=k, seconds=timing[tag])), flush=True)
    save_file({k: v.contiguous() for k, v in out.items()}, args.output,
              metadata={"steps": str(args.steps), "scales": json.dumps(args.scales), "lora": args.lora})
    print(json.dumps(dict(output=args.output, load_s=round(t_loaded - t_start, 1), timing=timing,
                          peak_mib=[torch.cuda.max_memory_allocated(d) // 2**20 for d in (0, 1)])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--eval-id", default="S01")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--artifact", required=True)
    r.add_argument("--lora", required=True, help="training-key LoRA safetensors (export)")
    r.add_argument("--noise", required=True)
    r.add_argument("--cond", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--scales", type=float, nargs="+", default=[1.0, 128.0])
    r.add_argument("--steps", type=int, default=10)
    r.add_argument("--split", type=int, default=25)
    r.add_argument("--width", type=int, default=384)
    r.add_argument("--height", type=int, default=256)
    args = ap.parse_args()
    prepare(args) if args.mode == "prepare" else run(args)


if __name__ == "__main__":
    main()
