# Copyright 2026 The MiniMax and HuggingFace Teams. Apache-2.0.
# Extracted from Diffusers aaf57d03a3edfbd9de457d2872417a92f149a91d; see docs/licenses/h3-diffusers-Apache-2.0.txt
from __future__ import annotations
import numpy as np
import torch
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1,4,4,4,4)
_ROPE_SPATIAL_SCALE = 32

def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """
    Pack video latents into transformer rows.

    Args:
        latents (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`):
            The latents to pack.
        patch_size (`tuple[int, int, int]`): The `(t, h, w)` patch.

    Returns:
        `torch.Tensor` of shape `(batch_size * num_patches, channels * prod(patch_size))`: The packed rows, ordered
        frame-major then row-major.
    """
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f'Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.')
    latents = latents.reshape(batch_size, channels, num_frames // patch_t, patch_t, height // patch_h, patch_h, width // patch_w, patch_w)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()

def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """
    One aspect-normalized spatial rotary axis: `dim // patch` coordinates centred on the unit interval, scaled up by
    32. The right endpoint is excluded, so a square canvas spans `[0, 32)`.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)

def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """The rotary time of every latent frame, starting at `origin`. Spacing is non-uniform: `5/3 * (1, 4, 4, 4, 4)`."""
    spans = torch.tensor([_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)] for index in range(num_latent_frames)], dtype=torch.float64)
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The `(h, w)` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing='ij')
    return (torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid)

def build_packed_sequence(text_token_tags: torch.Tensor, num_latent_frames: int, latent_height: int, latent_width: int, num_audio_latents: int, patch_size: tuple[int, int, int], audio_channels: int, audio_tag: int, video_tag: int, keyframe_anchors: tuple[str, ...]=()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
        Build the `[text | keyframe conditions | target audio | target video]` layout used by the `t2va` and `fl2va`
        tasks.

        Args:
            text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
                The modality tag of every text row. Text is tagged `1`, except for the rows of a keyframe's vision
                block, which MiniMax-H3 tags `0` (video).
            num_latent_frames (`int`): Number of target latent frames.
            latent_height (`int`): Target latent height.
            latent_width (`int`): Target latent width.
            num_audio_latents (`int`): Number of target audio latents per channel.
            patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
            audio_channels (`int`): Channels the soundtrack is packed channel-major over, i.e.
                `components.audio_channels`.
            audio_tag (`int`), video_tag (`int`):
                The modality tags an audio and a video row carry, i.e. `components.audio_tag` and
                `components.video_tag`.
            keyframe_anchors (`tuple[str, ...]`):
                One entry per keyframe conditioning block, in packed order: `"first"` anchors the block at the first
                latent frame, `"last"` at the last one.

        Returns:
            `tuple`: `position_ids`, `token_tags`, `video_indices`, `audio_indices`, `text_indices`, and the number of
            leading video and audio rows that are conditioning rather than generated.
        """
    _, patch_h, patch_w = patch_size
    rows_per_frame = latent_height // patch_h * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows
    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    for index, anchor in enumerate(keyframe_anchors):
        if anchor == 'first':
            anchor_time = float(num_text_tokens)
        elif anchor == 'last':
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                spans[offset::len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
            anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat([torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64), torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64)])
    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)
    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag
    return (position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, 0)

def build_row_timesteps(video_indices: torch.Tensor, audio_indices: torch.Tensor, num_condition_video_rows: int, num_condition_audio_rows: int, num_text_tokens: int, video_timestep: float, audio_timestep: float, condition_video_timestep: float, condition_audio_timestep: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
        Assign a timestep to every row of the packed sequence and reduce it to the transformer's `(timestep,
        timestep_indices)` pair.

        One forward serves rows at different noise levels: the generated video and audio rows step down their own
        schedules while the conditioning rows stay pinned at their noise-augmentation level. Text rows never reach an
        output head and inherit the video timestep.

        Args:
            video_indices (`torch.Tensor`): Sequence positions of the video rows, conditioning rows first.
            audio_indices (`torch.Tensor`): Sequence positions of the audio rows, reference rows first.
            num_condition_video_rows (`int`): How many leading video rows are conditioning rows.
            num_condition_audio_rows (`int`): How many leading audio rows are reference rows.
            num_text_tokens (`int`): Number of text rows, which never reach an output head.
            video_timestep (`float`): Timestep of the generated video rows.
            audio_timestep (`float`): Timestep of the generated audio rows.
            condition_video_timestep (`float`): Timestep of the video conditioning rows.
            condition_audio_timestep (`float`): Timestep of the audio reference rows.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the distinct timesteps, sorted, and the index of every row into them.
        """
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
    row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
    row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
    row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)

class MiniMaxH3Scheduler:

    def __init__(self, shift: float=12.0):
        if shift <= 0:
            raise ValueError(f'`shift` must be positive, got {shift}.')
        self.num_inference_steps: int | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._shift = float(shift)
        self._step_index: int | None = None
        self._begin_index: int | None = None

    def set_timesteps(self, num_inference_steps: int | None=None, device: str | torch.device | None=None, sigmas: list[float] | torch.Tensor | None=None) -> None:
        """
        Build the sigma / timestep schedule.

        The grid is `linspace(1, 0, num_inference_steps)` pushed through the exponential shift, with consecutive
        duplicates collapsed. The terminal `0` is already part of that grid — the shift maps `0` to exactly `0` — so
        the schedule holds `num_inference_steps` sigmas and drives `num_inference_steps - 1` model evaluations, exposed
        as `self.timesteps = 1 - sigmas[:-1]`.

        Args:
            num_inference_steps (`int`, *optional*):
                Number of sigma grid points, terminal `0` included. Ignored when `sigmas` is given.
            device (`str` or `torch.device`, *optional*):
                Device the schedule tensors are moved to. The grid itself is always built on CPU in float32 so the
                schedule does not depend on the accelerator.
            sigmas (`list[float]` or `torch.Tensor`, *optional*):
                A fully-formed sigma schedule, used verbatim (no shifting, no deduplication). It must be strictly
                decreasing and terminate at `0.0`.
        """
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(f'`set_timesteps` requires either an explicit `sigmas` schedule or `num_inference_steps` >= 2, got {num_inference_steps}.')
            base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
            sigmas = self._shift * base / (1 + (self._shift - 1) * base)
            sigmas = torch.unique_consecutive(sigmas)
        else:
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
            if sigmas.numel() < 2 or not bool((sigmas[1:] < sigmas[:-1]).all()) or sigmas[-1].item() != 0.0:
                raise ValueError('`sigmas` must hold at least two strictly decreasing values ending at 0.0.')
        self.sigmas = sigmas.to(device=device)
        self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        """
        Map a timestep value to its index in the schedule.

        Args:
            timestep (`float` or `torch.Tensor`):
                A value taken from `self.timesteps`. The schedule is strictly increasing in `t`, so the match is
                unique.

        Returns:
            `int`: The index of `timestep`.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) == 0:
            raise ValueError('Passed `timestep` is not in `self.timesteps`. Make sure to use values from `scheduler.timesteps`.')
        return indices[0].item()

    def step(self, model_output: torch.FloatTensor, timestep: float | torch.FloatTensor, sample: torch.FloatTensor, return_dict: bool=True) -> MiniMaxH3SchedulerOutput | tuple:
        """
        Take one Euler (`eta = 0`) step.

        The model output is a data-ward velocity, so the denoised estimate is `x0 = x_t + (1 - t) * v` — note the `+`,
        the opposite of the usual flow-match convention. The update is then the blend `x_next = r*x_t + (1 - r)*x0`
        with `r = sigma_next / sigma`, evaluated in float32 for half-precision samples.

        Args:
            model_output (`torch.FloatTensor`):
                The transformer's velocity prediction at `timestep`.
            timestep (`float` or `torch.FloatTensor`):
                The current timestep, one of `self.timesteps` (so `timestep == 1 - sigma`).
            sample (`torch.FloatTensor`):
                The current sample `x_t`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`MiniMaxH3SchedulerOutput`] instead of a plain tuple.

        Returns:
            [`MiniMaxH3SchedulerOutput`] or `tuple`: the sample for the next step.
        """
        if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor) and (not timestep.is_floating_point())):
            raise ValueError('Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to `MiniMaxH3Scheduler.step()` is not supported. Make sure to pass one of the `scheduler.timesteps` values.')
        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep) if self._begin_index is None else self._begin_index
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        denoised = sample + sigma_from_timestep * model_output
        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
        prev_sample = prev_sample.to(dtype=sample.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)

MINIMAX_H3_TRANSFORMER_DROPPED_KEYS = ('rope.inv_freq',)

def reorder_interleaved_qkv(weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int) -> torch.Tensor:
    """Reorder a *raw-checkpoint* per-head-interleaved fused QKV weight into `[q_all; k_all; v_all]`.

    The original checkpoint shards store rows as `[head0: q(head_dim) k(head_dim) v(head_dim), head1: q, k, v, ...]`.
    The reference implementation applies exactly this reorder at load time (`_reorder_grouped_qkv_to_qkv` with one head
    per query group), so `[q_all; k_all; v_all]` is the reference's in-memory / state-dict layout. There is no
    transpose.
    """
    expected_rows = num_attention_heads * 3 * attention_head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(f'fused qkv weight has {weight.shape[0]} rows, expected {expected_rows} = {num_attention_heads} heads * 3 * {attention_head_dim}.')
    grouped = weight.reshape(num_attention_heads, 3 * attention_head_dim, *weight.shape[1:])
    query, key, value = grouped.split(attention_head_dim, dim=1)
    return torch.cat([tensor.reshape(num_attention_heads * attention_head_dim, *weight.shape[1:]) for tensor in (query, key, value)], dim=0)

def split_fused_qkv(weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int) -> tuple[torch.Tensor, ...]:
    """Split a fused `[q_all; k_all; v_all]` QKV weight into separate `to_q` / `to_k` / `to_v` weights.

    The input is the *reference model* layout — what `MiniMaxH3DiTModel.state_dict()` holds after the reference's
    load-time reorder — i.e. the three logical projection matrices stacked contiguously, NOT the raw checkpoint's
    per-head interleave (see `reorder_interleaved_qkv`, which the shard streamer applies first).
    """
    inner_dim = num_attention_heads * attention_head_dim
    if weight.shape[0] != 3 * inner_dim:
        raise ValueError(f'fused qkv weight has {weight.shape[0]} rows, expected {3 * inner_dim} = 3 * {num_attention_heads} heads * {attention_head_dim}.')
    query, key, value = weight.split(inner_dim, dim=0)
    return tuple((tensor.contiguous() for tensor in (query, key, value)))

def convert_transformer_key(source_key: str, tensor: torch.Tensor, config: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    """Convert one original key/tensor pair into the diffusers key/tensor pair(s) it maps to."""
    if source_key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        return []
    target_key = source_key
    if target_key.startswith('token_refiner.blocks.'):
        target_key = target_key.replace('token_refiner.blocks.', 'token_refiner.refiner_blocks.', 1)
    elif target_key.startswith('blocks.'):
        target_key = target_key.replace('blocks.', 'transformer_blocks.', 1)
    target_key = target_key.replace('time_embedder.proj_in.', 'time_embedder.linear_1.')
    target_key = target_key.replace('time_embedder.proj_out.', 'time_embedder.linear_2.')
    target_key = target_key.replace('video_patch_proj.', 'proj_in.')
    target_key = target_key.replace('audio_patch_proj.', 'audio_proj_in.')
    target_key = target_key.replace('condition_proj.', 'context_embedder.')
    target_key = target_key.replace('final_layer.norm.', 'norm_out.norm.')
    target_key = target_key.replace('final_layer.adaln_proj.linear.', 'norm_out.linear.')
    target_key = target_key.replace('final_layer.video_out.', 'proj_out.')
    target_key = target_key.replace('final_layer.audio_out.', 'audio_proj_out.')
    target_key = target_key.replace('.attn.q_norm.', '.attn.norm_q.')
    target_key = target_key.replace('.attn.k_norm.', '.attn.norm_k.')
    target_key = target_key.replace('.attn.out_proj.', '.attn.to_out.0.')
    if target_key.endswith('.attn.qkv_proj.weight'):
        query, key, value = split_fused_qkv(tensor, config['num_attention_heads'], config['attention_head_dim'])
        prefix = target_key.removesuffix('qkv_proj.weight')
        return [(f'{prefix}to_q.weight', query), (f'{prefix}to_k.weight', key), (f'{prefix}to_v.weight', value)]
    if target_key.endswith('.mlp.fc1.weight'):
        gate, value = tensor.chunk(2, dim=0)
        target_key = target_key.replace('.mlp.fc1.weight', '.ff.net.0.proj.weight')
        return [(target_key, torch.cat([value, gate], dim=0).contiguous())]
    target_key = target_key.replace('.mlp.fc2.', '.ff.net.2.')
    return [(target_key, tensor)]

MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS = ('decoder.mask_token',)

def convert_video_vae_key(source_key: str, tensor: torch.Tensor, config: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    """Convert one original video-VAE key/tensor pair into the diffusers key/tensor pair(s) it maps to.

    `quant_conv` / `post_quant_conv`, the encoder's `conv_in` / `norm_out` / `conv_out` and the ViT decoder's
    `register_tokens` / `norm_out` / `proj_out` / `norm{1,2}` / `scale{1,2}` are pure pass-throughs. What moves:

    * the encoder's CNN levels are renamed from the original CompVis spelling onto the diffusers autoencoder idiom:
      `down.{i}.block.{j}` -> `down_blocks.{i}.resnets.{j}`, `nin_shortcut` -> `conv_shortcut`, and
      `down.{i}.downsample` -> `down_blocks.{i}.downsamplers.0`,
    * the ViT decoder's `x_embedder` becomes `proj_in`, the counterpart of the `proj_out` it already ships,
    * the fused per-head-interleaved `attn.to_qkv` is split into `attn.to_q` / `to_k` / `to_v`,
    * `attn.to_out` becomes `attn.to_out.0` (diffusers wraps the output projection in an `nn.ModuleList`),
    * the gated FFN's `w1` / `w2` become `ff.net.0.proj` / `ff.net.2`, and the two halves of `w1` are swapped because
      diffusers' `SwiGLU` reads `[up; gate]` where the reference stores `[gate; up]`.
    """
    if source_key in MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS:
        return []
    if '.attn.to_qkv.' in source_key:
        reordered = reorder_interleaved_qkv(tensor, config['decoder_num_attention_heads'], config['decoder_attention_head_dim'])
        query, key, value = split_fused_qkv(reordered, config['decoder_num_attention_heads'], config['decoder_attention_head_dim'])
        prefix, suffix = source_key.split('.attn.to_qkv.')
        return [(f'{prefix}.attn.to_q.{suffix}', query), (f'{prefix}.attn.to_k.{suffix}', key), (f'{prefix}.attn.to_v.{suffix}', value)]
    target_key = _rename_video_vae_key(source_key)
    if '.ff.w1.' in source_key:
        gate, up = tensor.chunk(2, dim=0)
        return [(target_key, torch.cat([up, gate], dim=0).contiguous())]
    return [(target_key, tensor)]

def _rename_video_vae_key(source_key: str) -> str:
    """Rename one original video-VAE key onto its diffusers module path (no tensor transform)."""
    target_key = source_key
    if target_key.startswith('encoder.down.'):
        level, rest = target_key.removeprefix('encoder.down.').split('.', 1)
        rest = rest.replace('block.', 'resnets.', 1).replace('nin_shortcut.', 'conv_shortcut.', 1)
        rest = rest.replace('downsample.', 'downsamplers.0.', 1)
        target_key = f'encoder.down_blocks.{level}.{rest}'
    target_key = target_key.replace('decoder.x_embedder.', 'decoder.proj_in.')
    target_key = target_key.replace('.attn.to_out.', '.attn.to_out.0.')
    target_key = target_key.replace('.ff.w1.', '.ff.net.0.proj.')
    target_key = target_key.replace('.ff.w2.', '.ff.net.2.')
    return target_key
