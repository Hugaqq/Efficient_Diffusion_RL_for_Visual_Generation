# W6/W7: reference rewards and native Flash sampler

Status on 2026-07-14: W6, W7, and W7b have passed. W8 bounded real-reward
active/control is now unlocked for preflight.

## W6 reward_general

The first live HPSv2 attempt is retained as a failure. The reference server
caught an internal model-loading error and returned the literal score `0.5`
with HTTP 200 for both samples. Direct scoring exposed that HPSv2 has an
undeclared LAION CLIP-H dependency, and the PyPI package also omitted its BPE
vocabulary. This demonstrates why HTTP success alone is not a valid reward
gate.

The missing vocabulary and the exact CLIP-H checkpoint were then pinned and
verified. Attempt 3 compared three paths on GPU2: direct HPSv2, the World-R1
reference client/server, and VisualRL's production reward client. All three
returned `[0.260009765625, 0.1943359375]`; both parity differences were exactly
zero, the red-on-black sample ranked above the blue mismatch, and a malformed
pickle returned HTTP 500 with a traceback. The HPS checkpoint was
`1,972,490,005` bytes with SHA256
`c57a38fb4a2f7e7c15bf00da2ea377cdf165448b4dd1052a484c215a998c9837`.
Peak CUDA allocated/reserved memory was 7.33/7.68 GiB.

The production client was corrected before attempt 3: it now emits the JPEG
byte-list protocol expected by World-R1, validates response type/count/finiteness,
and fails closed on malformed responses. The earlier implementation sent raw
tensors and therefore had not actually satisfied the real service contract.

Attempt 5 also closed a failure hidden by the reference implementation itself.
The unmodified server caught an internal HPS/image error and returned the
literal score `0.5` with HTTP 200. A staging-only patch now lets this error reach
the HTTP boundary as status 500. With the patched staging server, direct HPS,
reference HTTP, and the production infra client again returned exactly
`[0.260009765625, 0.1943359375]`; both parity differences were zero. A valid
pickle containing invalid image bytes returned HTTP 500 with a traceback, as
did a malformed pickle. No `0.5` fallback was observed. Peak CUDA
allocated/reserved memory was 7.36/7.72 GiB. Attempt 4 is retained separately:
it failed before model startup because the launcher and probe both created the
same output directory, so it provides no model or reward evidence.

## W6 reward_3d

All 16 registered HPS/CLIP-H/DA3/Qwen files were re-read from the server and
passed fixed revision, size, and SHA256 verification. They total
20,227,282,934 bytes.

Attempt 1 loaded DA3-GIANT and Qwen3-VL on GPU2 but failed the reward gate.
World-R1's worker caught a Gaussian-renderer error, returned score `0.0`, empty
artifact paths, and HTTP 200. The production client rejected that response.
The root cause was environmental: `gsplat` had no compiled CUDA backend because
the host provided CUDA runtime libraries but no `nvcc` compiler. Peak GPU memory
was 20,315 MiB, so this was not a 32 GB capacity failure.

A user-owned CUDA 12.8.61 compiler prefix was created without root and gsplat
1.5.3 was compiled for RTX 5090 `sm_120`. The resulting 19,249,168-byte extension
has SHA256
`424a5f7072d451797359e5f0f9d9012265d021f1c98eb035ead74687610e612e`.
Attempt 2 then passed every registered gate. The reference score was
`0.6791287263234457`, the infra score was `0.6791287064552307`, and their
absolute difference was `1.9868214962137642e-08`. Reconstruction, meta-view,
and camera-motion components were finite and summed exactly; the generated MP4
and PNG artifacts were nonempty; cache replay was exact; malformed input
returned HTTP 500 with a traceback; and the probe stopped only its own process
group. Peak GPU memory was 20,665 MiB.

## W7 native Flash sampler

Attempt 1 used the real pinned Wan model, 20 denoising steps, selected index 3
(scheduler timestep 944), two samples, five frames at 64 px, and deterministic
CUDA runtime. The state before the selected transition matched exactly, and
the recomputed log-prob, loss, and all 480 gradient tensors matched exactly.
The selected next latent, old log-prob, and video did not match.

The first divergence was an upstream RNG-plumbing defect: the native sampler
used the passed generator for the initial latent but failed to pass it to the
selected SDE step, which silently consumed global CUDA RNG. A staging-only
patch binds that SDE draw to the same generator. It does not change the SDE
formula.

After that correction, attempt 2 passed every registered gate. Reference and
infra media, before/after latent, timestep, old log-prob, KL, recomputed
log-prob, policy loss, and all 480 gradient tensors were bitwise identical.
Gradient SHA256 was
`b0c9cfff3bab88a46568fe065aaea794b1f81e7cb565a2fd2cab340caa7ea6ae`,
maximum gradient difference was zero, and the trainable parameter hash was
unchanged. Peak CUDA allocated/reserved memory was 13.62/13.91 GiB.

The native contract retains two latent states for one stochastic transition;
the equivalent 20-step full trajectory retains 21 states. This is a 10.5x
reduction in retained transition-state bytes for this shape. It is not a
throughput claim: the native sampler still executes all 20 denoising forwards.

The compatibility staging copy also fixes four release/API issues without
changing the algorithm formula: scalarizing a homogeneous selected-index list,
matching Diffusers 0.33 `check_inputs`, treating optional Wan config fields
safely, and using a no-op context when the transformer has no cache API.

## W7b heterogeneous selected-index batching

Attempt 1 completed the two independent scalar reference samples but the probe
incorrectly assumed that negative prompt embeddings were tensors when
`train_cfg=false`; it stopped before the infra grouped path and is retained as
a harness failure. The production merge contract and the probe were corrected
to accept an all-`None` optional tensor while rejecting partially missing data.

Attempt 2 used two different prompts with selected indices `[2, 1]`, forcing the
adapter to execute groups in a different order and then restore the original
sample order. Against independent scalar reference calls, media, prompt
embeddings, before/after latents, timesteps, old/new log-probs, and KL were all
bitwise identical. Policy loss was exactly `-0.031630873680114746` on both
paths; all 480 gradient tensors were bitwise identical with SHA256
`f366eaf1f0c45b0887f566c95d9d8b3038eba66f6a875afb8b0ef6d481bd3e73`;
and parameters were unchanged. Peak CUDA allocated/reserved memory was
13.61/13.86 GiB. W7b therefore passes, but it remains a correctness result, not
a throughput improvement claim.
