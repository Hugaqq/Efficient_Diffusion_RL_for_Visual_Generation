# Real-Wan World-R1 and Flash-GRPO bounded smoke

This experiment uses the pinned `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
checkpoint and the supplied World-R1 reference patch. It is intentionally
bounded to one LoRA update on one RTX 5090 32 GB GPU per run.

The World-R1 run validates the real Wan rollout, local video reward, GRPO
advantage, log-prob recomputation, backward pass, parameter update, and PEFT
checkpoint path. It does not deploy the paper's `reward_3d` and
`reward_general` servers, so it is an integration smoke rather than a
World-R1 effectiveness result.

The original W4 Flash-GRPO run validated the same real Wan model with
VisualRL's selected single-timestep loss and rectification by narrowing a
World-R1 trajectory. W7 subsequently added and validated the native Flash
sampler: real reference and infra media, selected transition, log-prob, loss,
and gradients are bitwise identical after fixing upstream RNG plumbing. The
native path reduces retained transition-state storage, but no speed claim is
made until the separate profiler/overhead experiments run.

Hard gates for both runs are: pinned model and source identities match, the
summary is valid, exactly one metrics row exists, all rewards/log-probs/losses
and gradients are finite, at least one gradient and LoRA parameter update are
nonzero, and the checkpoint contains PEFT adapter weights without a full Wan
transformer state.

`verify_wan_snapshot.py` checks all 19 required files against the laptop-side
size/SHA256 manifest and pinned revision after transfer. W1 is executed by
`remote_wan_load_sample.py`; W2 reloads its PEFT checkpoint in a fresh process
with `remote_wan_roundtrip.py`. W3/W4 use `remote_wan_train.py`, which records
load/train timing, peak CUDA memory, initial/final trainable hashes, parameter
delta, finite/nonzero gradient gates, run status, and checkpoint integrity.
These four files are frozen historical harnesses. In particular,
`remote_wan_train.py` touches CUDA before constructing `ExperimentRunner` and
must not be reused with the current deterministic runtime; post-merge W5 uses
`remote_wan_resume_run.py`, whose startup-order regression is covered by
`test_remote_wan_resume_run.py`.
`W6_W7_RESULTS.md` records the real HPS/3D reward work and native Flash parity,
including all failed attempts and the exact staging-only reference patches.
