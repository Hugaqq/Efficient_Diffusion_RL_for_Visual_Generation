def test_plain_pytest_is_scoped_to_project_tests():
    text = open("pyproject.toml", encoding="utf-8").read()
    assert "[tool.pytest.ini_options]" in text
    assert 'testpaths = ["tests"]' in text
    assert "reference_code" in text


def test_strict_rollout_validation_catches_bad_batch_axis():
    import pytest
    import torch

    from visual_rl.core.types import RolloutBatch

    batch = RolloutBatch(
        prompts=["p", "q"],
        metadata=[{}, {}],
        media=torch.zeros(2, 3, 4, 4),
        latents=torch.zeros(1, 2, 3, 4, 4),
        next_latents=torch.zeros(1, 2, 3, 4, 4),
        timesteps=torch.zeros(2, 2),
        old_log_probs=torch.zeros(2, 2),
    )
    with pytest.raises(ValueError, match="latents batch dimension"):
        batch.validate_lightweight(strict=True)


def test_reward_media_hash_uses_numpy_and_pil_content():
    import numpy as np

    from visual_rl.feedback.cache import stable_hash_media

    first = np.zeros((4, 4, 3), dtype=np.uint8)
    second = first.copy()
    second[0, 0, 0] = 255
    assert stable_hash_media(first) != stable_hash_media(second)

    try:
        from PIL import Image
    except ImportError:
        return
    assert stable_hash_media(Image.fromarray(first)) != stable_hash_media(Image.fromarray(second))


def test_grpo_loss_accepts_old_logprob_dtype_mismatch():
    import torch

    from visual_rl.optimizers.grpo import GRPOAlgorithm
    from visual_rl.core.types import RolloutBatch

    batch = RolloutBatch(
        prompts=["p"],
        metadata=[{}],
        media=torch.zeros(1, 3, 4, 4),
        latents=torch.zeros(1, 2, 3, 4, 4),
        next_latents=torch.zeros(1, 2, 3, 4, 4),
        timesteps=torch.zeros(1, 2),
        old_log_probs=torch.zeros(1, 2, dtype=torch.float64),
    )
    new_log_probs = torch.zeros(1, 2, dtype=torch.float32, requires_grad=True)
    loss, info = GRPOAlgorithm().compute_loss(batch, torch.ones(1), new_log_probs)
    assert loss.dtype == torch.float32
    assert info["approx_kl"].dtype == torch.float32
