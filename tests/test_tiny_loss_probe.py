def test_tiny_loss_probe_descends_and_writes_artifacts(tmp_path):
    import json
    from pathlib import Path

    from scripts.loss_probe import TinyLossProbeConfig, run_tiny_loss_probe

    summary = run_tiny_loss_probe(
        TinyLossProbeConfig(
            output_dir=tmp_path,
            steps=60,
            learning_rate=0.1,
            batch_size=4,
            num_steps=4,
            image_size=8,
            seed=123,
        )
    )

    assert summary["loss_end"] < summary["loss_start"] * 0.1
    assert summary["bias_error_end"] < summary["bias_error_start"] * 0.25
    assert summary["grpo_policy_loss_end"] < summary["grpo_policy_loss_start"]

    metrics_path = Path(summary["metrics_path"])
    assert metrics_path.exists()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["step"] == 0
    assert rows[-1]["step"] == 60
    assert rows[-1]["loss"] == summary["loss_end"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "checkpoint_final" / "tiny_diffusion.pt").exists()


def test_tiny_loss_probe_cli_outputs_summary(tmp_path, capsys):
    from scripts import legacy_cli as cli

    exit_code = cli.main(
        [
            "tiny-loss-probe",
            "--output-dir",
            str(tmp_path),
            "--steps",
            "60",
            "--learning-rate",
            "0.1",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"loss_start"' in output
    assert '"loss_end"' in output
    assert '"metrics_path"' in output
