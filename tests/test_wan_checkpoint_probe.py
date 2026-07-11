from __future__ import annotations


def _write_model_index(path, class_name="WanPipeline"):
    import json

    path.mkdir(parents=True)
    (path / "model_index.json").write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")


def test_wan_checkpoint_probe_manifest_only(capsys, tmp_path):
    import json

    from scripts import legacy_cli as cli

    checkpoint = tmp_path / "Wan2.1-T2V-1.3B-Diffusers"
    repo_root = tmp_path / "World-R1-main"
    _write_model_index(checkpoint)
    repo_root.mkdir()

    exit_code = cli.main(
        [
            "wan-checkpoint-probe",
            "--model-path",
            str(checkpoint),
            "--repo-root",
            str(repo_root),
            "--manifest-only",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["manifest_valid"] is True
    assert payload["loaded"] is False
    assert payload["load_requested"] is False
    assert payload["side_effects"]["sample_called"] is False
    assert payload["side_effects"]["checkpoint_written"] is False


def test_wan_checkpoint_probe_rejects_non_wan_manifest(capsys, tmp_path):
    import json

    from scripts import legacy_cli as cli

    checkpoint = tmp_path / "stable-diffusion"
    repo_root = tmp_path / "World-R1-main"
    _write_model_index(checkpoint, class_name="StableDiffusionPipeline")
    repo_root.mkdir()

    exit_code = cli.main(
        [
            "wan-checkpoint-probe",
            "--model-path",
            str(checkpoint),
            "--repo-root",
            str(repo_root),
            "--manifest-only",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["manifest_valid"] is False
    assert "does not look like Wan" in payload["errors"][0]


def test_wan_checkpoint_probe_loads_fake_diffusers_pipeline(capsys, monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace

    from scripts import legacy_cli as cli

    checkpoint = tmp_path / "Wan2.1-T2V-1.3B-Diffusers"
    repo_root = tmp_path / "World-R1-main"
    _write_model_index(checkpoint)
    repo_root.mkdir()
    calls = []

    class FakeTransformer:
        pass

    class FakePipeline:
        def __init__(self):
            self.transformer = FakeTransformer()
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class WanPipeline:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls.append({"model_path": model_path, "kwargs": kwargs})
            return FakePipeline()

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(WanPipeline=WanPipeline))

    exit_code = cli.main(
        [
            "wan-checkpoint-probe",
            "--model-path",
            str(checkpoint),
            "--repo-root",
            str(repo_root),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["loaded"] is True
    assert payload["pipeline_class"] == "FakePipeline"
    assert payload["transformer_class"] == "FakeTransformer"
    assert payload["transformer_present"] is True
    assert calls == [
        {
            "model_path": str(checkpoint),
            "kwargs": {"local_files_only": True, "low_cpu_mem_usage": True},
        }
    ]
