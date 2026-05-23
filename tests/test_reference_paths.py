def test_legacy_repo_resolves_reference_code_layout():
    from visual_rl.third_party.legacy import resolve_legacy_repo

    resolved = resolve_legacy_repo("World-R1-main")
    assert str(resolved).endswith("reference_code/World-R1-main")
