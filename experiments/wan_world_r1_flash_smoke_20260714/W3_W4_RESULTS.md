# W3/W4: real Wan training closure

Both bounded one-step training paths passed on physical GPU2 with the pinned
real Wan snapshot and the hash-matched World-R1 reference patch.

W3 expanded one prompt into two real videos, scored the resulting video tensor
with the local color reward, normalized a nonzero two-sample GRPO advantage,
recomputed both trajectory timesteps, and completed backward/optimizer/update.
All 480 LoRA gradient tensors were finite, 5,894,217 gradient elements were
nonzero, and the trainable parameter delta was L2 `0.188665`. Peak CUDA
allocated/reserved memory was 13.60/13.92 GiB. The v2 checkpoint was 71.37MB,
passed its file hash manifest, and contained PEFT plus training state but no
full transformer.

W4 used the same real model but the Flash-style rollout/optimizer path. It
selected timestep 0, applied rectification weight 1.0, and trained only the
selected log-prob. All 480 gradient tensors were finite, 5,898,240 gradient
elements were nonzero, and the trainable delta was L2 `0.171065`. Its peak
memory and 71.37MB PEFT-only v2 checkpoint met the same gates.

The World-R1 run took 9.95s to load and 2.50s for training plus artifacts; the
Flash-style run took 8.30s to load and 1.83s for training plus artifacts. These
are one-run observations, not performance claims. Current Flash obtains the
full two-step trajectory and narrows it afterward, so W4 proves interface and
update closure only; native selected-step parity and efficiency remain W7.

The local color reward is also not the paper's `reward_3d` or
`reward_general`. Reference reward parity and failure handling remain W6.
Finally, one update proves trainability, not model-quality improvement; W8
active/control remains locked behind W5-W7.
