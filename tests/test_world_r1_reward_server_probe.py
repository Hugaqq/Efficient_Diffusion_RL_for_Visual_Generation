from __future__ import annotations


def _install_fake_requests(monkeypatch, post):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))


def test_world_r1_reward_server_probe_success(capsys, monkeypatch, tmp_path):
    import json
    import pickle

    import visual_rl.cli as cli

    side_effect_dir = tmp_path / "should_not_exist"
    calls = []

    class Response:
        content = pickle.dumps({"outputs": [0.25], "metadata": {"server": "mock"}})

        def raise_for_status(self):
            return None

    def fake_post(url, data, timeout):
        calls.append({"url": url, "payload": pickle.loads(data), "timeout": timeout})
        return Response()

    _install_fake_requests(monkeypatch, fake_post)

    exit_code = cli.main(
        [
            "world-r1-reward-server-probe",
            "--reward",
            "reward_general",
            "--url",
            "http://127.0.0.1:9000/reward",
            "--timeout",
            "0.5",
            "--batch-size",
            "1",
            "--height",
            "2",
            "--width",
            "3",
            "--prompt",
            "a probe prompt",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["payload_kind"] == "images"
    assert payload["media_shape"] == [1, 3, 2, 3]
    assert payload["values"] == [0.25]
    assert payload["metadata"] == {"server": "mock"}
    assert payload["side_effects"] == {
        "checkpoint_written": False,
        "output_dir_written": False,
        "trainer_constructed": False,
    }
    assert calls[0]["url"] == "http://127.0.0.1:9000/reward"
    assert calls[0]["timeout"] == 0.5
    assert calls[0]["payload"]["prompts"] == ["a probe prompt"]
    assert calls[0]["payload"]["images"].shape == (1, 3, 2, 3)
    assert not side_effect_dir.exists()


def test_world_r1_reward_server_probe_3d_payload(capsys, monkeypatch):
    import json
    import pickle

    import visual_rl.cli as cli

    class Response:
        content = pickle.dumps({"outputs": [0.5, 0.75], "metadata": {"kind": "3d"}})

        def raise_for_status(self):
            return None

    def fake_post(url, data, timeout):
        del url, timeout
        payload = pickle.loads(data)
        assert payload["videos"].shape == (2, 3, 3, 2, 2)
        return Response()

    _install_fake_requests(monkeypatch, fake_post)

    exit_code = cli.main(
        [
            "world-r1-reward-server-probe",
            "--reward",
            "reward_3d",
            "--url",
            "https://reward.example/3d",
            "--batch-size",
            "2",
            "--frames",
            "3",
            "--height",
            "2",
            "--width",
            "2",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["payload_kind"] == "videos"
    assert payload["media_shape"] == [2, 3, 3, 2, 2]
    assert payload["values"] == [0.5, 0.75]


def test_world_r1_reward_server_probe_timeout_returns_structured_json(capsys, monkeypatch):
    import json

    import visual_rl.cli as cli

    calls = {"count": 0}

    def fake_post(url, data, timeout):
        del url, data, timeout
        calls["count"] += 1
        raise TimeoutError("reward server timed out")

    _install_fake_requests(monkeypatch, fake_post)

    exit_code = cli.main(
        [
            "world-r1-reward-server-probe",
            "--reward",
            "reward_general",
            "--url",
            "http://127.0.0.1:9000/reward",
            "--timeout",
            "0.01",
            "--retries",
            "2",
        ]
    )

    assert exit_code == 1
    assert calls["count"] == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert "timed out" in payload["errors"][0]


def test_world_r1_reward_server_probe_invalid_response_shape(capsys, monkeypatch):
    import json
    import pickle

    import visual_rl.cli as cli

    class Response:
        content = pickle.dumps({"outputs": [0.1], "metadata": {}})

        def raise_for_status(self):
            return None

    def fake_post(url, data, timeout):
        del url, data, timeout
        return Response()

    _install_fake_requests(monkeypatch, fake_post)

    exit_code = cli.main(
        [
            "world-r1-reward-server-probe",
            "--reward",
            "reward_general",
            "--url",
            "http://127.0.0.1:9000/reward",
            "--batch-size",
            "2",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert "expected shape (2,)" in payload["errors"][0]


def test_world_r1_reward_server_probe_rejects_bad_url(capsys):
    import json

    import visual_rl.cli as cli

    exit_code = cli.main(
        [
            "world-r1-reward-server-probe",
            "--reward",
            "reward_general",
            "--url",
            "file:///tmp/reward.sock",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert "http or https" in payload["errors"][0]
