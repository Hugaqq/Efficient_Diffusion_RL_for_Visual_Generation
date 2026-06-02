from __future__ import annotations


def _write_model_index(path, class_name):
    import json

    path.mkdir(parents=True, exist_ok=True)
    (path / "model_index.json").write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")


def test_checkpoint_inventory_classifies_diffusers_model_dirs(tmp_path):
    from visual_rl.experiments.checkpoint_inventory import build_checkpoint_inventory

    _write_model_index(tmp_path / "stable-diffusion-3.5-medium", "StableDiffusion3Pipeline")
    _write_model_index(tmp_path / "FLUX.1-dev", "FluxPipeline")
    _write_model_index(tmp_path / "Qwen-Image", "QwenImagePipeline")
    _write_model_index(tmp_path / "stable-diffusion-v1-5", "StableDiffusionPipeline")
    _write_model_index(tmp_path / "Wan2.1-T2V-1.3B-Diffusers", "WanPipeline")

    payload = build_checkpoint_inventory(
        [tmp_path],
        required_adapters=[
            "sd3_tempflow",
            "world_r1_wan_legacy",
        ],
    )

    assert payload["valid"] is True
    assert payload["missing_adapters"] == []
    assert payload["found_adapters"] == [
        "sd3_tempflow",
        "world_r1_wan_legacy",
    ]
    model_types = {record["model_type"] for record in payload["records"]}
    assert model_types == {"sd3", "flux", "qwenimage", "sd15", "wan"}


def test_checkpoint_inventory_reports_missing_required_adapter(tmp_path, capsys):
    import json

    import visual_rl.cli as cli

    _write_model_index(tmp_path / "stable-diffusion-3.5-medium", "StableDiffusion3Pipeline")

    exit_code = cli.main(
        [
            "checkpoint-inventory",
            str(tmp_path),
            "--require-adapter",
            "sd3_tempflow",
            "--require-adapter",
            "world_r1_wan_legacy",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["found_adapters"] == ["sd3_tempflow"]
    assert payload["missing_adapters"] == ["world_r1_wan_legacy"]
    assert payload["errors"] == []


def test_checkpoint_inventory_ignores_parent_directory_model_keywords(tmp_path):
    from visual_rl.experiments.checkpoint_inventory import build_checkpoint_inventory

    contaminated_root = tmp_path / "Wan-parent"
    _write_model_index(contaminated_root / "stable-diffusion-v1-5", "StableDiffusionPipeline")

    payload = build_checkpoint_inventory([contaminated_root])

    assert payload["valid"] is True
    assert payload["records"][0]["model_type"] == "sd15"
    assert payload["found_adapters"] == []


def test_checkpoint_inventory_rejects_missing_root(tmp_path, capsys):
    import json

    import visual_rl.cli as cli

    missing = tmp_path / "missing"

    exit_code = cli.main(["checkpoint-inventory", str(missing)])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["records"] == []
    assert payload["errors"][0]["root"] == str(missing)
    assert "does not exist" in payload["errors"][0]["message"]
