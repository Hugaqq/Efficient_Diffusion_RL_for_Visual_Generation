def test_reward_router_raw_weighted_and_cache(tmp_path):
    import torch

    import visual_rl.rewards.clients  # noqa: F401
    from visual_rl.rewards.router import RewardRouter

    router = RewardRouter(
        {
            "weights": {"mock": 2.0},
            "clients": {"mock": {"name": "mock", "version": "test", "mode": "prompt_media"}},
            "normalize": "none",
        },
        cache_dir=tmp_path / "reward_cache",
    )
    rewards = router.score(
        media=torch.ones(2, 3, 4, 4),
        prompts=["a prompt", "a prompt"],
        metadata=[{}, {}],
    )
    assert "mock" in rewards.raw
    assert torch.allclose(rewards.weighted["mock"], rewards.raw["mock"] * 2.0)
    assert rewards.valid_mask.all()
    assert list((tmp_path / "reward_cache").glob("*.json"))


def test_reward_router_replays_cache_without_calling_client_again(tmp_path, monkeypatch):
    import numpy as np
    import torch

    from visual_rl.core.registry import REWARD_CLIENTS
    from visual_rl.rewards.router import RewardRouter

    calls = {"count": 0}

    class CountingRewardClient:
        def score(self, media, prompts, metadata):
            del media, metadata
            calls["count"] += 1
            return np.arange(len(prompts), dtype=np.float32), {"calls": calls["count"]}

    class ExplodingRewardClient:
        def score(self, media, prompts, metadata):
            del media, prompts, metadata
            raise AssertionError("cache miss should not call this client")

    monkeypatch.setitem(REWARD_CLIENTS._items, "counting_cache", CountingRewardClient)  # noqa: SLF001
    config = {
        "weights": {"cache_reward": 1.0},
        "clients": {"cache_reward": {"name": "counting_cache", "version": "cache-v1"}},
    }
    cache_dir = tmp_path / "reward_cache"
    first = RewardRouter(config, cache_dir=cache_dir).score(
        media=torch.ones(2, 3, 4, 4),
        prompts=["a", "b"],
        metadata=[{"i": 0}, {"i": 1}],
    )

    monkeypatch.setitem(REWARD_CLIENTS._items, "counting_cache", ExplodingRewardClient)  # noqa: SLF001
    second = RewardRouter(config, cache_dir=cache_dir).score(
        media=torch.ones(2, 3, 4, 4),
        prompts=["a", "b"],
        metadata=[{"i": 0}, {"i": 1}],
    )

    assert calls["count"] == 1
    assert torch.equal(second.raw["cache_reward"], first.raw["cache_reward"])
    assert second.metadata["cache_reward"] == {"calls": 1}


def test_reward_router_cache_key_includes_media_content(tmp_path, monkeypatch):
    import numpy as np
    import torch

    from visual_rl.core.registry import REWARD_CLIENTS
    from visual_rl.rewards.router import RewardRouter

    calls = {"count": 0}

    class CountingRewardClient:
        def score(self, media, prompts, metadata):
            del media, metadata
            calls["count"] += 1
            return np.full(len(prompts), calls["count"], dtype=np.float32), {"calls": calls["count"]}

    monkeypatch.setitem(REWARD_CLIENTS._items, "content_cache", CountingRewardClient)  # noqa: SLF001
    config = {
        "weights": {"cache_reward": 1.0},
        "clients": {"cache_reward": {"name": "content_cache", "version": "cache-v1"}},
    }
    router = RewardRouter(config, cache_dir=tmp_path / "reward_cache")

    first = router.score(torch.zeros(1, 3, 4, 4), ["same prompt"], [{}])
    second = router.score(torch.ones(1, 3, 4, 4), ["same prompt"], [{}])

    assert calls["count"] == 2
    assert first.raw["cache_reward"].item() == 1.0
    assert second.raw["cache_reward"].item() == 2.0
    assert len(list((tmp_path / "reward_cache").glob("*.json"))) == 2


def test_reward_router_invalid_reward_shape_sets_invalid_mask(tmp_path, monkeypatch):
    import numpy as np

    from visual_rl.core.registry import REWARD_CLIENTS
    from visual_rl.rewards.router import RewardRouter

    class BadShapeRewardClient:
        def score(self, media, prompts, metadata):
            del media, prompts, metadata
            return np.asarray([1.0], dtype=np.float32), {"source": "bad_shape"}

    monkeypatch.setitem(REWARD_CLIENTS._items, "bad_shape", BadShapeRewardClient)  # noqa: SLF001
    router = RewardRouter(
        {
            "weights": {"bad": 1.0},
            "clients": {"bad": {"name": "bad_shape"}},
            "fail_policy": "invalid",
        },
        cache_dir=tmp_path / "reward_cache",
    )

    rewards = router.score(media=None, prompts=["a", "b"], metadata=[{}, {}])

    assert rewards.valid_mask.tolist() == [False, False]
    assert rewards.raw["bad"].tolist() == [0.0, 0.0]
    assert "expected shape (2,)" in rewards.metadata["bad"]["error"]
    assert not list((tmp_path / "reward_cache").glob("*.json"))


def test_reward_router_invalid_reward_shape_raises_when_configured(monkeypatch):
    import numpy as np
    import pytest

    from visual_rl.core.registry import REWARD_CLIENTS
    from visual_rl.rewards.router import RewardRouter

    class BadShapeRewardClient:
        def score(self, media, prompts, metadata):
            del media, prompts, metadata
            return np.asarray([1.0], dtype=np.float32), {}

    monkeypatch.setitem(REWARD_CLIENTS._items, "bad_shape_raise", BadShapeRewardClient)  # noqa: SLF001
    router = RewardRouter(
        {
            "weights": {"bad": 1.0},
            "clients": {"bad": {"name": "bad_shape_raise"}},
            "fail_policy": "raise",
        },
        cache_dir=None,
    )

    with pytest.raises(ValueError, match="expected shape"):
        router.score(media=None, prompts=["a", "b"], metadata=[{}, {}])


def test_reward_router_unknown_reward_client_name_raises():
    import pytest

    from visual_rl.rewards.router import RewardRouter

    with pytest.raises(KeyError, match="Unknown reward_client key"):
        RewardRouter(
            {
                "weights": {"missing": 1.0},
                "clients": {"missing": {"name": "does_not_exist"}},
            }
        )


def test_remote_pickle_reward_timeout_retries_and_invalid_policy(monkeypatch):
    import sys
    from types import SimpleNamespace

    from visual_rl.rewards.router import RewardRouter

    calls = []

    def fake_post(url, data, timeout):
        calls.append({"url": url, "data": data, "timeout": timeout})
        raise TimeoutError("timed out")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=fake_post))
    router = RewardRouter(
        {
            "weights": {"remote": 1.0},
            "clients": {
                "remote": {
                    "name": "remote_pickle",
                    "url": "http://reward.invalid/score",
                    "timeout": 0.01,
                    "retries": 2,
                }
            },
            "fail_policy": "invalid",
        },
        cache_dir=None,
    )

    rewards = router.score(media="payload", prompts=["a"], metadata=[{}])

    assert len(calls) == 3
    assert {call["url"] for call in calls} == {"http://reward.invalid/score"}
    assert {call["timeout"] for call in calls} == {0.01}
    assert rewards.valid_mask.tolist() == [False]
    assert rewards.raw["remote"].tolist() == [0.0]
    assert "timed out" in rewards.metadata["remote"]["error"]
