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
        torch.full((parent_count, 2, 2), float(step)) for step in range(4)
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
    pipeline = namespace["pipeline"]
    adapter = SD3TempFlowAdapter.__new__(SD3TempFlowAdapter)
    adapter._pipeline_with_logprob_perstep = pipeline
    generator = torch.Generator().manual_seed(17)

    with adapter._perstep_sde_generator(generator):
        assert pipeline() is generator

    assert seen == [(False, generator)]
    assert namespace["sde_step_with_logprob"] is upstream_sde_step
