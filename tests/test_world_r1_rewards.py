import pytest


def test_world_r1_reward_helpers_validate_url_and_payload_kind():
    from visual_rl.feedback.world_r1_rewards import reward_3d_client, reward_general_client

    reward_3d = reward_3d_client(" http://127.0.0.1:18080/reward_3d ")
    reward_general = reward_general_client("https://reward.example.test/general")

    assert reward_3d.url == "http://127.0.0.1:18080/reward_3d"
    assert reward_3d.payload_kind == "videos"
    assert reward_3d.timeout == 2000.0
    assert reward_general.payload_kind == "images"
    assert reward_general.timeout == 1000.0


def test_world_r1_reward_general_samples_video_frame_payload(monkeypatch):
    import pickle

    import torch

    import visual_rl.feedback.clients as clients
    from visual_rl.feedback.world_r1_rewards import reward_general_client

    calls = []

    def fake_post_bytes(url, payload, timeout):
        calls.append({"url": url, "payload": pickle.loads(payload), "timeout": timeout})
        return pickle.dumps({"outputs": [0.25, 0.5], "metadata": {"server": "general"}})

    monkeypatch.setattr(clients, "_post_bytes", fake_post_bytes)

    media = torch.arange(2 * 3 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 3, 4, 5)
    client = reward_general_client("http://127.0.0.1:18080/general")
    values, metadata = client.score(media, ["a", "b"], [{}, {}])

    assert values.tolist() == [0.25, 0.5]
    assert calls[0]["payload"]["images"].shape == (2, 3, 4, 5)
    assert torch.equal(calls[0]["payload"]["images"], media[:, 1])
    assert calls[0]["payload"]["prompts"] == ["a", "b"]
    assert metadata["server"] == "general"
    assert metadata["payload_kind"] == "images"
    assert metadata["frame_sampling"]["selected_frame_index"] == 1
    assert metadata["frame_sampling"]["input_shape"] == [2, 3, 3, 4, 5]
    assert metadata["frame_sampling"]["output_shape"] == [2, 3, 4, 5]


def test_world_r1_reward_3d_uses_videos_payload(monkeypatch):
    import pickle

    import torch

    import visual_rl.feedback.clients as clients
    from visual_rl.feedback.world_r1_rewards import reward_3d_client

    calls = []

    def fake_post_bytes(url, payload, timeout):
        calls.append({"url": url, "payload": pickle.loads(payload), "timeout": timeout})
        return pickle.dumps({"outputs": [0.75], "metadata": {"server": "3d"}})

    monkeypatch.setattr(clients, "_post_bytes", fake_post_bytes)

    media = torch.ones(1, 2, 3, 4, 5)
    values, metadata = reward_3d_client("http://127.0.0.1:18080/3d").score(media, ["prompt"], [{}])

    assert values.tolist() == [0.75]
    assert metadata == {"server": "3d"}
    assert calls[0]["payload"]["videos"].shape == (1, 2, 3, 4, 5)
    assert "images" not in calls[0]["payload"]


def test_world_r1_reward_url_validation_rejects_non_http_endpoints():
    from visual_rl.feedback.world_r1_rewards import validate_reward_server_url

    with pytest.raises(ValueError, match="must use http or https"):
        validate_reward_server_url("file:///tmp/reward.pkl", reward_name="reward_3d")

    with pytest.raises(ValueError, match="must include a host"):
        validate_reward_server_url("https:///score", reward_name="reward_general")

    with pytest.raises(ValueError, match="must not embed credentials"):
        validate_reward_server_url("http://user:pass@127.0.0.1:18080/score")


def test_remote_pickle_reward_client_falls_back_to_urllib_without_requests(monkeypatch):
    import pickle
    import sys

    import numpy as np

    import visual_rl.feedback.clients as clients
    from visual_rl.feedback.clients import RemotePickleRewardClient

    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            return False

        def read(self):
            return pickle.dumps({"outputs": [0.75], "metadata": {"transport": "urllib"}})

    def fake_urlopen(request, timeout):
        calls.append({"url": request.full_url, "payload": pickle.loads(request.data), "timeout": timeout})
        return Response()

    monkeypatch.setitem(sys.modules, "requests", None)
    monkeypatch.setattr(clients, "urlopen", fake_urlopen)

    reward = RemotePickleRewardClient(url="http://127.0.0.1:9000/reward", retries=0, timeout=0.5)
    values, metadata = reward.score(np.zeros((1, 3, 2, 2), dtype=np.float32), ["prompt"], [{}])

    assert values.tolist() == [0.75]
    assert metadata == {"transport": "urllib"}
    assert len(calls) == 1
    assert calls[0]["url"] == "http://127.0.0.1:9000/reward"
    assert calls[0]["timeout"] == 0.5
    assert calls[0]["payload"]["prompts"] == ["prompt"]
    np.testing.assert_array_equal(calls[0]["payload"]["images"], np.zeros((1, 3, 2, 2), dtype=np.float32))
