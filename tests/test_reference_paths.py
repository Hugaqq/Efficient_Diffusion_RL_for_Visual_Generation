def test_legacy_repo_resolves_reference_code_layout():
    from visual_rl.third_party.legacy import resolve_legacy_repo

    resolved = resolve_legacy_repo("World-R1-main")
    assert str(resolved).endswith("reference_code/World-R1-main")


def test_legacy_repo_resolves_external_reference_root(monkeypatch, tmp_path):
    from visual_rl.third_party.legacy import resolve_legacy_repo

    reference_root = tmp_path / "reference_code"
    world_r1 = reference_root / "World-R1-main"
    world_r1.mkdir(parents=True)
    monkeypatch.setenv("VISUAL_RL_REFERENCE_CODE_ROOT", str(reference_root))

    assert resolve_legacy_repo("World-R1-main") == world_r1
    assert resolve_legacy_repo("reference_code/World-R1-main") == world_r1
