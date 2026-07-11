from __future__ import annotations

import pytest


def test_sd3_adapter_maps_reference_perstep_output_to_shared_branch_batch(tmp_path):
    from types import SimpleNamespace

    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter
    from visual_rl.optimizers.tempflow_grpo import TempFlowGRPOAlgorithm
    from visual_rl.rollout.full_trajectory import build_rollout_engine

    repo_root = tmp_path / "TempFlow-GRPO-main"
    repo_root.mkdir()
    adapter = SD3TempFlowAdapter(
        {
            "model_path": str(tmp_path / "model"),
            "repo_root": str(repo_root),
            "extra": {"defer_load": True, "device": "cpu", "dtype": "float32"},
        }
    )
    adapter.pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(
            timesteps=torch.tensor([900.5, 600.25, 300.0])
        )
    )
    adapter.transformer = torch.nn.Linear(1, 1)
    adapter._pipeline_with_logprob_perstep = object()
    adapter._encode_text = lambda prompts: (
        torch.arange(len(prompts) * 2, dtype=torch.float32).reshape(len(prompts), 2),
        torch.arange(len(prompts), dtype=torch.float32).reshape(len(prompts), 1),
    )

    parent_count = 2
    exploration_k = 3
    rows = parent_count * exploration_k
    branch_media = [
        torch.full((rows, 3, 4, 4), float(step)) for step in range(2)
    ]
    main_latents = [
        torch.full((parent_count, 2, 2), float(step)) for step in range(3)
    ]
    sde_latents = [
        torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 2, 2) + step * 100
        for step in range(2)
    ]
    log_probs = [
        torch.arange(rows, dtype=torch.float32) + step * 10
        for step in range(2)
    ]
    kls = [torch.tensor([0.1, 0.2]) + step for step in range(2)]
    adapter._call_pipeline_with_logprob_perstep = lambda **_kwargs: (
        branch_media,
        main_latents,
        sde_latents,
        log_probs,
        kls,
    )

    rollout = build_rollout_engine(
        {
            "name": "branching",
            "num_steps": 3,
            "branch_count": 2,
            "exploration_k": 2,
            "include_main": False,
            "branch_timesteps": "auto",
            "epoch_tag": 1,
            "seed": 5,
        }
    )
    batch = rollout.sample(adapter, ["red", "blue"], [{}, {}])

    assert batch.prompts == ["red", "red", "blue", "blue"]
    assert batch.branch_ids.tolist() == [0, 1, 0, 1]
    assert batch.timesteps.dtype == torch.float32
    assert batch.timesteps.flatten().tolist() == pytest.approx([600.25] * 4)
    assert batch.old_log_probs.flatten().tolist() == [10.0, 11.0, 13.0, 14.0]
    assert batch.kl.flatten().tolist() == pytest.approx([1.1, 1.1, 1.2, 1.2])
    assert batch.model_metadata["trajectory_step_indices"] == [1]
    assert [item["branch_step_index"] for item in batch.metadata] == [1] * 4
    assert [item["branch_timestep_value"] for item in batch.metadata] == [
        600.25
    ] * 4
    assert batch.model_metadata["branch_step_candidates"] == [0, 1]
    assert batch.model_metadata["transition_count"] == 2

    advantages = TempFlowGRPOAlgorithm(
        credit_assignment="branch_timestep",
        noise_weighting={"enabled": False},
    )._expand_advantages(batch, torch.ones(4), torch.zeros(4, 1))
    assert advantages.tolist() == [[1.0], [1.0], [1.0], [1.0]]
    algorithm = TempFlowGRPOAlgorithm(
        credit_assignment="branch_timestep",
        noise_weighting={"enabled": True},
    )
    weights = algorithm._noise_weights(batch, torch.zeros(4, 1))
    expected_weight = (0.5**0.5) / ((1.0 + 0.5**0.5) / 2.0)
    assert weights.flatten().tolist() == pytest.approx([expected_weight] * 4)


def test_sd3_perstep_wrapper_binds_and_restores_sde_generator():
    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    seen = []

    def upstream_sde_step(*, determistic=False, generator=None):
        seen.append((determistic, generator))
        return generator

    namespace = {"sde_step_with_logprob": upstream_sde_step}
    exec(  # noqa: S102 - isolated function globals model the upstream module
        "def pipeline():\n"
        "    return sde_step_with_logprob(determistic=False)\n",
        namespace,
    )
    pipeline = torch.no_grad()(namespace["pipeline"])
    adapter = SD3TempFlowAdapter.__new__(SD3TempFlowAdapter)
    adapter._pipeline_with_logprob_perstep = pipeline
    generator = torch.Generator().manual_seed(17)

    with adapter._perstep_sde_generator(generator):
        assert pipeline() is generator

    assert seen == [(False, generator)]
    assert namespace["sde_step_with_logprob"] is upstream_sde_step


def test_sd3_full_wrapper_binds_and_restores_sde_generator():
    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    seen = []

    def upstream_sde_step(*, determistic=False, generator=None):
        seen.append((determistic, generator))
        return generator

    namespace = {"sde_step_with_logprob": upstream_sde_step}
    exec(  # noqa: S102 - isolated function globals model the upstream module
        "def pipeline():\n"
        "    return sde_step_with_logprob(determistic=False)\n",
        namespace,
    )
    pipeline = torch.no_grad()(namespace["pipeline"])
    adapter = SD3TempFlowAdapter.__new__(SD3TempFlowAdapter)
    adapter._pipeline_with_logprob = pipeline
    generator = torch.Generator().manual_seed(23)

    with adapter._full_sde_generator(generator):
        assert namespace["sde_step_with_logprob"] is not upstream_sde_step
        assert pipeline() is generator

    assert seen == [(False, generator)]
    assert namespace["sde_step_with_logprob"] is upstream_sde_step


def test_sd3_full_wrapper_preserves_explicit_prev_sample_and_generator():
    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    def upstream_sde_step(
        scheduler,
        model_output,
        timestep,
        sample,
        prev_sample=None,
        generator=None,
    ):
        del scheduler, model_output, timestep, sample
        return prev_sample, generator

    namespace = {"sde_step_with_logprob": upstream_sde_step}
    exec(  # noqa: S102 - isolated function globals model the upstream module
        "def pipeline(explicit_prev_sample, explicit_generator):\n"
        "    with_prev_sample = sde_step_with_logprob(\n"
        "        None, None, None, None, prev_sample=explicit_prev_sample\n"
        "    )\n"
        "    with_generator = sde_step_with_logprob(\n"
        "        None, None, None, None, generator=explicit_generator\n"
        "    )\n"
        "    return with_prev_sample, with_generator\n",
        namespace,
    )
    pipeline = torch.no_grad()(namespace["pipeline"])
    adapter = SD3TempFlowAdapter.__new__(SD3TempFlowAdapter)
    adapter._pipeline_with_logprob = pipeline
    bound_generator = torch.Generator().manual_seed(29)
    explicit_generator = torch.Generator().manual_seed(31)
    explicit_prev_sample = object()

    with adapter._full_sde_generator(bound_generator):
        with_prev_sample, with_generator = pipeline(
            explicit_prev_sample,
            explicit_generator,
        )

    assert with_prev_sample == (explicit_prev_sample, None)
    assert with_generator == (None, explicit_generator)
    assert namespace["sde_step_with_logprob"] is upstream_sde_step


def test_sd3_perstep_wrapper_uses_dynamic_branch_count_with_shared_kernel():
    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    seen = []

    def legacy_perstep_kernel(*args, **kwargs):
        raise AssertionError(
            f"legacy per-step kernel should be replaced: args={args}, kwargs={kwargs}"
        )

    def shared_kernel(
        scheduler,
        model_output,
        timestep,
        sample,
        prev_sample=None,
        generator=None,
        determistic=False,
    ):
        del scheduler, prev_sample
        seen.append(
            {
                "model_rows": int(model_output.shape[0]),
                "timestep_rows": int(timestep.shape[0]),
                "sample_rows": int(sample.shape[0]),
                "generator": generator,
                "determistic": determistic,
            }
        )
        rows = sample.shape[0]
        return (
            sample,
            torch.zeros(rows),
            sample,
            torch.ones(rows),
        )

    namespace = {"sde_step_with_logprob": legacy_perstep_kernel}
    exec(  # noqa: S102 - isolated function globals model the upstream module
        "def pipeline(scheduler, model_output, timestep, sample):\n"
        "    ode = sde_step_with_logprob(\n"
        "        scheduler, model_output, timestep, sample, determistic=True\n"
        "    )\n"
        "    branches = sde_step_with_logprob(\n"
        "        scheduler, model_output, timestep, sample, determistic=False\n"
        "    )\n"
        "    return ode, branches\n",
        namespace,
    )
    pipeline = torch.no_grad()(namespace["pipeline"])
    adapter = SD3TempFlowAdapter.__new__(SD3TempFlowAdapter)
    adapter._pipeline_with_logprob_perstep = pipeline
    adapter._sde_step_with_logprob = shared_kernel
    generator = torch.Generator().manual_seed(19)
    parent_rows = 2
    branch_count = 4

    with adapter._perstep_sde_generator(
        generator,
        branch_count=branch_count,
    ):
        ode, branches = pipeline(
            object(),
            torch.zeros(parent_rows, 1, 2, 2),
            torch.tensor([900.0, 900.0]),
            torch.zeros(parent_rows, 1, 2, 2),
        )

    assert ode[0].shape[0] == parent_rows
    assert branches[0].shape[0] == parent_rows * branch_count
    assert seen == [
        {
            "model_rows": parent_rows,
            "timestep_rows": parent_rows,
            "sample_rows": parent_rows,
            "generator": None,
            "determistic": True,
        },
        {
            "model_rows": parent_rows * branch_count,
            "timestep_rows": parent_rows * branch_count,
            "sample_rows": parent_rows * branch_count,
            "generator": generator,
            "determistic": False,
        },
    ]
    assert namespace["sde_step_with_logprob"] is legacy_perstep_kernel


def test_sd3_perstep_wrapper_hard_fails_when_real_kernel_cannot_be_replaced():
    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    adapter = SD3TempFlowAdapter.__new__(SD3TempFlowAdapter)
    adapter._pipeline_with_logprob_perstep = lambda: None
    adapter._sde_step_with_logprob = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="per-step branching SDE kernel"):
        with adapter._perstep_sde_generator(
            torch.Generator().manual_seed(37),
            branch_count=2,
        ):
            pass


def test_sd3_transformer_input_dtype_cast_is_scoped_and_exception_safe():
    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    class RecordingTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor(1.0, dtype=torch.bfloat16)
            )
            self.seen_dtypes = []

        def forward(self, hidden_states=None):
            self.seen_dtypes.append(hidden_states.dtype)
            return hidden_states

    transformer = RecordingTransformer()
    adapter = SD3TempFlowAdapter.__new__(SD3TempFlowAdapter)
    adapter.transformer = transformer
    fp32_hidden_states = torch.ones(1, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="stop inside context"):
        with adapter._transformer_input_dtype():
            output = transformer(hidden_states=fp32_hidden_states)
            positional_output = transformer(fp32_hidden_states)
            assert output.dtype == torch.bfloat16
            assert positional_output.dtype == torch.bfloat16
            raise RuntimeError("stop inside context")

    restored_output = transformer(hidden_states=fp32_hidden_states)
    assert restored_output.dtype == torch.float32
    assert transformer.seen_dtypes == [
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
    ]


def _deferred_sd3_adapter(tmp_path, transformer):
    from types import SimpleNamespace

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    adapter = SD3TempFlowAdapter(
        {
            "model_path": str(tmp_path / "model"),
            "repo_root": str(tmp_path / "TempFlow-GRPO-main"),
            "extra": {"defer_load": True, "device": "cpu", "dtype": "float32"},
        }
    )
    adapter.pipeline = SimpleNamespace(scheduler=object())
    adapter.transformer = transformer
    return adapter


def _shared_prefix_batch(
    torch,
    *,
    guidance_scale=1.0,
    parent_count=2,
    branch_count=3,
):
    from visual_rl.core.types import RolloutBatch

    parent_latents = torch.tensor(
        [
            [[[-0.75, -0.25], [0.25, 0.75]]],
            [[[0.125, 0.375], [0.625, 0.875]]],
        ],
        dtype=torch.float32,
    )[:parent_count]
    parent_prompt_embeds = torch.arange(
        1,
        parent_count + 1,
        dtype=torch.float32,
    ).reshape(parent_count, 1, 1)
    rows = parent_count * branch_count
    parent_indices = [
        parent_index
        for parent_index in range(parent_count)
        for _ in range(branch_count)
    ]
    branch_ids = list(range(branch_count)) * parent_count
    latents = parent_latents.repeat_interleave(branch_count, dim=0)[:, None]
    prompt_embeds = parent_prompt_embeds.repeat_interleave(branch_count, dim=0)
    pooled_prompt_embeds = prompt_embeds[:, 0]
    timesteps = torch.full((rows, 1), 700.0, dtype=torch.float32)

    return RolloutBatch(
        prompts=[f"parent-{parent_index}" for parent_index in parent_indices],
        metadata=[
            {
                "parent_prompt_index": parent_index,
                "branch_id": branch_id,
                "branch_step_index": 0,
                "branch_timestep_value": 700.0,
                "is_main_branch": False,
                "rollout_kind": "tempflow_branching",
            }
            for parent_index, branch_id in zip(
                parent_indices,
                branch_ids,
                strict=True,
            )
        ],
        media=torch.zeros(rows, 3, 2, 2),
        latents=latents,
        next_latents=latents.float() + 0.1,
        timesteps=timesteps,
        old_log_probs=torch.zeros(rows, 1),
        kl=torch.zeros(rows, 1),
        branch_ids=torch.tensor(branch_ids),
        model_metadata={
            "guidance_scale": guidance_scale,
            "branching_mode": "shared_prefix",
            "branch_count": branch_count,
            "branch_step_index": 0,
            "trajectory_step_indices": [0],
        },
        model_tensors={
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": torch.zeros_like(prompt_embeds),
            "negative_pooled_prompt_embeds": torch.zeros_like(
                pooled_prompt_embeds
            ),
        },
    )


def _quadratic_transition_kernel(torch, seen_samples=None):
    def kernel(scheduler, noise_pred, timestep, sample, prev_sample):
        del scheduler, timestep
        if seen_samples is not None:
            seen_samples.append(sample.detach().clone())
        mean = sample + 0.25 * noise_pred
        log_prob = -(
            (prev_sample.detach() - mean).flatten(start_dim=1).square().mean(dim=1)
        )
        return prev_sample, log_prob, mean, torch.ones_like(log_prob)

    return kernel


def test_sd3_branching_canonicalizes_bf16_source_without_rounding_target(
    tmp_path,
):
    from types import SimpleNamespace

    import torch

    from visual_rl.model_adapters.sd3 import SD3TempFlowAdapter

    class StrictBFloat16Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor(1.0, dtype=torch.bfloat16)
            )
            self.seen_dtypes = []

        def forward(self, *, hidden_states):
            self.seen_dtypes.append(hidden_states.dtype)
            if hidden_states.dtype != torch.bfloat16:
                raise RuntimeError("transformer requires bfloat16 hidden states")
            return (hidden_states,)

    repo_root = tmp_path / "TempFlow-GRPO-main"
    repo_root.mkdir()
    adapter = SD3TempFlowAdapter(
        {
            "model_path": str(tmp_path / "model"),
            "repo_root": str(repo_root),
            "extra": {
                "defer_load": True,
                "device": "cpu",
                "dtype": "bfloat16",
            },
        }
    )
    adapter.pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(
            timesteps=torch.tensor([900.0, 600.0, 300.0])
        )
    )
    adapter.transformer = StrictBFloat16Transformer()
    adapter._pipeline_with_logprob_perstep = object()
    adapter._encode_text = lambda prompts: (
        torch.ones(len(prompts), 1, 1, dtype=torch.bfloat16),
        torch.ones(len(prompts), 1, dtype=torch.bfloat16),
    )

    raw_source = torch.tensor(
        [[[[1.0001, -1.0001], [0.3333, -0.3333]]]],
        dtype=torch.float32,
    )
    canonical_source = raw_source.to(torch.bfloat16)
    assert not torch.equal(raw_source, canonical_source.float())
    raw_targets = torch.tensor(
        [
            [[[0.1111, 0.2222], [0.3333, 0.4444]]],
            [[[0.5555, 0.6666], [0.7777, 0.8888]]],
        ],
        dtype=torch.float32,
    )
    main_latents = [
        torch.zeros_like(raw_source, dtype=torch.bfloat16),
        raw_source,
        raw_source + 1.0,
    ]
    sde_latents = [torch.zeros_like(raw_targets), raw_targets]
    log_probs = [torch.zeros(2), torch.tensor([0.25, 0.5])]
    kls = [torch.zeros(1), torch.zeros(1)]
    branch_media = [
        torch.zeros(2, 3, 2, 2),
        torch.ones(2, 3, 2, 2),
    ]
    def fake_perstep_call(**_kwargs):
        adapter.transformer(hidden_states=raw_targets)
        return branch_media, main_latents, sde_latents, log_probs, kls

    adapter._call_pipeline_with_logprob_perstep = fake_perstep_call

    batch = adapter.sample_branching(
        ["a red cube"],
        [{}],
        {
            "branch_step_index": 1,
            "branch_count": 2,
            "num_steps": 3,
            "guidance_scale": 1.0,
            "seed": 17,
            "include_main": False,
            "transition_count": 2,
        },
    )

    expected_sources = canonical_source.repeat_interleave(2, dim=0)
    assert batch.latents.dtype == torch.bfloat16
    assert torch.equal(batch.latents[:, 0], expected_sources)
    assert batch.next_latents.dtype == torch.float32
    assert torch.equal(batch.next_latents[:, 0], raw_targets)
    assert adapter.transformer.seen_dtypes == [torch.bfloat16]


@pytest.mark.parametrize(
    ("guidance_scale", "expected_transformer_batch"),
    [(1.0, 2), (4.5, 4)],
)
def test_sd3_shared_prefix_recompute_forwards_only_parent_batch(
    tmp_path,
    guidance_scale,
    expected_transformer_batch,
):
    import torch

    class RecordingTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))
            self.batch_sizes = []

        def forward(
            self,
            *,
            hidden_states,
            timestep,
            encoder_hidden_states,
            pooled_projections,
            return_dict,
        ):
            del timestep, encoder_hidden_states, pooled_projections, return_dict
            self.batch_sizes.append(int(hidden_states.shape[0]))
            return (hidden_states * self.scale,)

    transformer = RecordingTransformer()
    adapter = _deferred_sd3_adapter(tmp_path, transformer)
    adapter._sde_step_with_logprob = _quadratic_transition_kernel(torch)
    batch = _shared_prefix_batch(torch, guidance_scale=guidance_scale)

    recomputed = adapter.recompute_log_probs(batch)

    assert recomputed.shape == (6, 1)
    assert transformer.batch_sizes == [expected_transformer_batch]


@pytest.mark.parametrize(
    "inconsistent_field",
    ["source", "timestep", "embedding"],
)
def test_sd3_shared_prefix_rejects_inconsistent_parent_transition(
    tmp_path,
    inconsistent_field,
):
    import torch

    class SimpleTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))

        def forward(
            self,
            *,
            hidden_states,
            timestep,
            encoder_hidden_states,
            pooled_projections,
            return_dict,
        ):
            del timestep, encoder_hidden_states, pooled_projections, return_dict
            return (hidden_states * self.scale,)

    adapter = _deferred_sd3_adapter(tmp_path, SimpleTransformer())
    adapter._sde_step_with_logprob = _quadratic_transition_kernel(torch)
    batch = _shared_prefix_batch(torch)
    if inconsistent_field == "source":
        batch.latents[1, 0, 0, 0, 0] += 1.0
    elif inconsistent_field == "timestep":
        batch.timesteps[1, 0] += 1.0
    else:
        batch.model_tensors["prompt_embeds"][1, 0, 0] += 1.0

    with pytest.raises(ValueError):
        adapter.recompute_log_probs(batch)


def test_sd3_shared_prefix_old_new_logprob_parity_preserves_gradient(tmp_path):
    import torch

    class BatchSensitiveTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))
            self.batch_sizes = []

        def forward(
            self,
            *,
            hidden_states,
            timestep,
            encoder_hidden_states,
            pooled_projections,
            return_dict,
        ):
            del timestep, pooled_projections, return_dict
            self.batch_sizes.append(int(hidden_states.shape[0]))
            embedding_signal = encoder_hidden_states.flatten(start_dim=1).mean(
                dim=1
            )
            embedding_signal = embedding_signal.reshape(
                -1,
                *([1] * (hidden_states.ndim - 1)),
            )
            batch_signal = hidden_states.new_tensor(
                hidden_states.shape[0] * 0.01
            )
            return (
                (hidden_states + embedding_signal) * self.scale + batch_signal,
            )

    transformer = BatchSensitiveTransformer()
    adapter = _deferred_sd3_adapter(tmp_path, transformer)
    adapter._sde_step_with_logprob = _quadratic_transition_kernel(torch)
    batch = _shared_prefix_batch(torch, guidance_scale=1.0)
    branch_count = 3

    parent_source = batch.latents[::branch_count, 0]
    parent_embeds = batch.model_tensors["prompt_embeds"][::branch_count]
    embedding_signal = parent_embeds.flatten(start_dim=1).mean(dim=1).reshape(
        -1,
        1,
        1,
        1,
    )
    parent_noise = (
        (parent_source + embedding_signal) * transformer.scale.detach()
        + parent_source.new_tensor(0.02)
    )
    expanded_noise = parent_noise.repeat_interleave(branch_count, dim=0)
    expected_mean = batch.latents[:, 0].float() + 0.25 * expanded_noise.float()
    offsets = torch.tensor(
        [0.1, 0.2, 0.4, 0.15, 0.25, 0.45],
        dtype=torch.float32,
    ).reshape(-1, 1, 1, 1)
    batch.next_latents[:, 0] = expected_mean.detach() + offsets
    batch.old_log_probs[:, 0] = -offsets.flatten(start_dim=1).square().mean(
        dim=1
    )

    recomputed = adapter.recompute_log_probs(batch)
    recomputed.sum().backward()

    assert transformer.scale.grad is not None
    assert torch.isfinite(transformer.scale.grad)
    assert float(transformer.scale.grad.abs()) > 0.0
    assert transformer.batch_sizes == [2]
    assert torch.allclose(
        recomputed,
        batch.old_log_probs,
        atol=1e-7,
        rtol=0.0,
    )


def _full_mixed_precision_fixture(torch, tmp_path):
    from types import SimpleNamespace

    class FullTrajectoryTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(
                torch.tensor(0.25, dtype=torch.bfloat16)
            )
            self.batch_sizes = []

        def forward(
            self,
            *,
            hidden_states,
            timestep,
            encoder_hidden_states,
            pooled_projections,
            return_dict,
        ):
            del timestep, pooled_projections, return_dict
            self.batch_sizes.append(int(hidden_states.shape[0]))
            embedding_signal = encoder_hidden_states.flatten(start_dim=1).mean(
                dim=1
            )
            embedding_signal = embedding_signal.reshape(
                -1,
                *([1] * (hidden_states.ndim - 1)),
            )
            return ((hidden_states + embedding_signal) * self.scale,)

    transformer = FullTrajectoryTransformer()
    adapter = _deferred_sd3_adapter(tmp_path, transformer)
    adapter.pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(
            timesteps=torch.tensor([900.0, 600.0, 300.0]),
            sigmas=torch.tensor([1.0, 0.7, 0.3, 0.0]),
        )
    )
    adapter._sde_step_with_logprob = _quadratic_transition_kernel(torch)
    namespace = {"sde_step_with_logprob": adapter._sde_step_with_logprob}
    exec(  # noqa: S102 - model the decorated TempFlow module globals
        "def pipeline():\n"
        "    return sde_step_with_logprob\n",
        namespace,
    )
    adapter._pipeline_with_logprob = torch.no_grad()(namespace["pipeline"])
    positive_embed = torch.ones(1, 1, 1, dtype=torch.bfloat16)
    positive_pooled = torch.ones(1, 1, dtype=torch.bfloat16)
    negative_embed = torch.zeros_like(positive_embed)
    negative_pooled = torch.zeros_like(positive_pooled)

    def encode_text(prompts):
        if prompts == [""]:
            return negative_embed, negative_pooled
        return positive_embed, positive_pooled

    adapter._encode_text = encode_text
    initial = torch.tensor(
        [[[[1.0001, -1.0001], [0.3333, -0.3333]]]],
        dtype=torch.bfloat16,
    )
    raw_states = [initial]
    old_log_probs = []
    live_source = initial
    for offset in (0.1, 0.2, 0.3):
        noise_pred = (
            (live_source + live_source.new_tensor(1.0))
            * transformer.scale.detach()
        ).to(positive_embed.dtype)
        mean = live_source.float() + 0.25 * noise_pred.float()
        raw_target = mean + float(offset)
        raw_states.append(raw_target)
        old_log_probs.append(torch.tensor([-(float(offset) ** 2)]))
        live_source = raw_target.to(initial.dtype)

    adapter._call_pipeline_with_logprob = lambda **_kwargs: (
        torch.zeros(1, 3, 2, 2),
        raw_states,
        old_log_probs,
        [torch.zeros(1) for _ in old_log_probs],
    )
    batch = adapter.sample(
        ["a red cube"],
        [{}],
        {
            "num_steps": 3,
            "guidance_scale": 1.0,
            "seed": 29,
            "output_type": "pt",
        },
    )
    return adapter, batch, raw_states, transformer


def test_sd3_full_sample_canonicalizes_sources_but_preserves_raw_targets(
    tmp_path,
):
    import torch

    _adapter, batch, raw_states, _transformer = _full_mixed_precision_fixture(
        torch,
        tmp_path,
    )
    expected_sources = torch.stack(
        [state.to(raw_states[0].dtype) for state in raw_states[:-1]],
        dim=1,
    )
    expected_targets = torch.stack(raw_states[1:], dim=1)

    assert raw_states[0].dtype == torch.bfloat16
    assert all(state.dtype == torch.float32 for state in raw_states[1:])
    assert any(
        not torch.equal(state, state.to(torch.bfloat16).float())
        for state in raw_states[1:-1]
    )
    assert batch.latents.dtype == torch.bfloat16
    assert torch.equal(batch.latents, expected_sources)
    assert batch.next_latents.dtype == torch.float32
    assert torch.equal(batch.next_latents, expected_targets)
    assert batch.model_metadata["trajectory_source_dtype"] == "torch.bfloat16"
    assert batch.model_metadata["trajectory_target_dtype"] == "torch.float32"


def test_sd3_full_v2_recompute_rejects_changed_target_dtype(tmp_path):
    import torch

    adapter, batch, _raw_states, transformer = _full_mixed_precision_fixture(
        torch,
        tmp_path,
    )
    assert batch.model_metadata["trajectory_contract_version"] == (
        "sd3_tempflow_v2"
    )
    batch.next_latents = batch.next_latents.to(torch.bfloat16)

    with pytest.raises(
        ValueError,
        match="trajectory target dtype changed after rollout",
    ):
        adapter.recompute_log_probs(batch)

    assert transformer.batch_sizes == []


def test_sd3_full_three_step_old_new_parity_preserves_gradient(tmp_path):
    import torch

    adapter, batch, _raw_states, transformer = _full_mixed_precision_fixture(
        torch,
        tmp_path,
    )

    recomputed = adapter.recompute_log_probs(batch)
    recomputed.sum().backward()

    assert recomputed.shape == (1, 3)
    assert torch.allclose(
        recomputed,
        batch.old_log_probs,
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        recomputed - batch.old_log_probs,
        torch.zeros_like(recomputed),
        atol=1e-7,
        rtol=0.0,
    )
    assert transformer.batch_sizes == [1, 1, 1]
    assert transformer.scale.grad is not None
    assert torch.isfinite(transformer.scale.grad)
    assert float(transformer.scale.grad.abs()) > 0.0


@pytest.mark.parametrize("changed_field", ["timesteps", "sigmas"])
def test_sd3_full_recompute_rejects_changed_scheduler_context(
    tmp_path,
    changed_field,
):
    import torch

    adapter, batch, _raw_states, _transformer = _full_mixed_precision_fixture(
        torch,
        tmp_path,
    )
    live_value = getattr(adapter.pipeline.scheduler, changed_field).clone()
    live_value[0] += 0.125
    setattr(adapter.pipeline.scheduler, changed_field, live_value)

    with pytest.raises(
        ValueError,
        match=rf"scheduler {changed_field} changed after rollout",
    ):
        adapter.recompute_log_probs(batch)
