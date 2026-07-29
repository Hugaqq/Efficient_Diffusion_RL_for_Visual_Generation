"""Format-v5 two-rank checkpoint and resume contracts."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import visual_rl as vr
from visual_rl.artifacts.checkpoint import (
    RankState,
    TrainingContract,
    apply_training_state,
    read_and_validate_training_state,
    save_training_state,
)
from visual_rl.core.types import RuntimeBuildContext
from visual_rl.errors import ResumeError
from visual_rl.model_adapters.tiny_diffusion import TinyDiffusionAdapter


ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "tests" / "fixtures" / "configs" / "tiny_grpo.yaml"


def _adapter_optimizer():
    adapter = TinyDiffusionAdapter.from_config(
        {"image_size": 4},
        RuntimeBuildContext(
            rank=0,
            local_rank=0,
            world_size=1,
            backend=None,
            device=torch.device("cpu"),
            precision="fp32",
        ),
    )
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in adapter.named_parameters()],
        lr=1.0e-4,
    )
    adapter.color_bias.grad = torch.tensor([0.5, -0.25, 0.125])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return adapter, optimizer


def _rank_state(rank: int, seed: int) -> RankState:
    python_state = random.Random(seed).getstate()
    numpy_state = np.random.RandomState(seed).get_state()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return RankState.from_rng(
        rank=rank,
        python_state=python_state,
        numpy_state=numpy_state,
        torch_cpu=generator.get_state(),
        torch_cuda=None,
    )


def test_two_rank_format_v5_restores_selected_rank_and_optimizer(
    tmp_path: Path,
) -> None:
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_torch = torch.get_rng_state()
    try:
        adapter, optimizer = _adapter_optimizer()
        saved_parameter = adapter.color_bias.detach().clone()
        saved_optimizer = optimizer.state_dict()
        states = (_rank_state(0, 101), _rank_state(1, 202))
        contract = TrainingContract(algorithm="grpo", version=2)
        checkpoint = tmp_path / "checkpoint_000001"

        metadata = save_training_state(
            checkpoint,
            adapter=adapter,
            optimizer=optimizer,
            scaler=None,
            global_step=1,
            training_contract=contract,
            rank_states=states,
            writer_rank=0,
            writer_device=torch.device("cpu"),
        )

        payload = json.loads(
            (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
        )
        assert metadata.world_size == payload["world_size"] == 2
        assert payload["format_version"] == 5
        assert payload["training_contract"] == {
            "algorithm": "grpo",
            "version": 2,
        }
        assert set(path.name for path in checkpoint.iterdir()) == {
            "adapter",
            "checkpoint.json",
            "training_state.pt",
        }

        with torch.no_grad():
            adapter.color_bias.zero_()
        optimizer.state.clear()
        validated = read_and_validate_training_state(
            checkpoint,
            adapter=adapter,
            optimizer=optimizer,
            scaler=None,
            expected_global_step=1,
            expected_world_size=2,
            expected_training_contract=contract,
        )
        apply_training_state(
            validated,
            adapter=adapter,
            optimizer=optimizer,
            scaler=None,
            optimizer_config=vr.load(TINY).resolve().optimizer,
            rank=1,
        )

        torch.testing.assert_close(adapter.color_bias, saved_parameter)
        assert optimizer.state_dict()["state"].keys() == saved_optimizer["state"].keys()
        assert torch.equal(torch.get_rng_state(), states[1].torch_cpu)
        with pytest.raises(ResumeError, match="already been applied"):
            apply_training_state(
                validated,
                adapter=adapter,
                optimizer=optimizer,
                scaler=None,
                optimizer_config=vr.load(TINY).resolve().optimizer,
                rank=1,
            )
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.set_rng_state(original_torch)


@pytest.mark.parametrize(
    ("world_size", "contract", "message"),
    [
        (1, TrainingContract("grpo", 2), "world_size"),
        (2, TrainingContract("grpo", 1), "training_contract"),
        (2, TrainingContract("flash_grpo", 2), "training_contract"),
    ],
)
def test_resume_rejects_topology_or_training_contract_drift(
    tmp_path: Path,
    world_size: int,
    contract: TrainingContract,
    message: str,
) -> None:
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_torch = torch.get_rng_state()
    try:
        adapter, optimizer = _adapter_optimizer()
        checkpoint = tmp_path / "checkpoint_000001"
        save_training_state(
            checkpoint,
            adapter=adapter,
            optimizer=optimizer,
            scaler=None,
            global_step=1,
            training_contract=TrainingContract("grpo", 2),
            rank_states=(_rank_state(0, 1), _rank_state(1, 2)),
            writer_rank=0,
            writer_device=torch.device("cpu"),
        )

        with pytest.raises(ResumeError, match=message):
            read_and_validate_training_state(
                checkpoint,
                adapter=adapter,
                optimizer=optimizer,
                scaler=None,
                expected_global_step=1,
                expected_world_size=world_size,
                expected_training_contract=contract,
            )
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.set_rng_state(original_torch)
