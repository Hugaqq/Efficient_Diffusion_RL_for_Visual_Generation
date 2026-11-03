"""History lookup is not artifact identity migration (CPU/local Git only)."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import pytest

from scripts import flash_small40_repair as repair
from scripts import h3_checkpoint_upgrade as upgrade
from scripts import launch_flash140 as flash
from scripts import launch_h3_media as media
from scripts.rewritten_history import (
    H3_RECORDED_SHA, SMALL40_RECORDED_SHA, RECORDED_TO_GIT_SHA, git_revision,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("recorded,rewritten", RECORDED_TO_GIT_SHA.items())
def test_exact_pins_and_real_rewritten_ancestry(recorded, rewritten):
    assert git_revision(recorded) == rewritten != recorded
    assert git_revision(rewritten) == rewritten
    assert git_revision(recorded + ":visual_rl") == rewritten + ":visual_rl"
    assert git_revision(recorded[:7]) == recorded[:7]  # never guess prefixes
    assert git_revision("f" * 40) == "f" * 40
    for argument in ("HEAD", "--name-only", "data/prompts", "HEAD:visual_rl"):
        assert git_revision(argument) == argument
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                    rewritten, "HEAD"], check=True, timeout=30)
    assert subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", rewritten + ":visual_rl"],
        text=True, timeout=30).strip()


def test_shared_constants_keep_recorded_identity():
    assert repair.REPAIR_BASE == SMALL40_RECORDED_SHA
    assert repair.REPAIR_GIT_BASE == RECORDED_TO_GIT_SHA[SMALL40_RECORDED_SHA]
    assert upgrade.BASE_SHA == media.BASE_SHA == H3_RECORDED_SHA
    assert upgrade.BASE_GIT_SHA == media.BASE_GIT_SHA == RECORDED_TO_GIT_SHA[H3_RECORDED_SHA]
    fixture = json.loads((ROOT / "tests/fixtures/characterization/v0_6_loss_probe.json").read_text())
    assert fixture["source"]["code_revision"] == "5ce1ef7d91048f7291457269aa8b22e05d32b1d8"
    evidence = json.loads((ROOT / "experiments/flow_pickapic_20260801/evidence/flow_pickapic_c20_stable_v2_seed17.json").read_text())
    assert evidence["candidate"]["commit"] == "b75854029318ea2a0e79ec50d0ab0cc1124623e0"


def test_small40_real_git_never_receives_old_objects(monkeypatch):
    # Refuse old revisions even when unrelated local branches retain the objects.
    original = subprocess.check_output
    calls = []

    def public_git_only(command, **kwargs):
        calls.append(command)
        assert command[0] == "git"
        assert not any(arg.partition(":")[0] in RECORDED_TO_GIT_SHA for arg in command)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", public_git_only)
    repair.git_read(ROOT, "diff", "--name-only", repair.REPAIR_BASE, "HEAD")
    repair.git_read(ROOT, "rev-parse", repair.REPAIR_BASE + ":visual_rl")
    proof = repair.prove_launcher_upgrade(ROOT, repair.REPAIR_BASE, flash.SMALL_CONFIG)
    assert proof["base_sha"] == proof["new_sha"] == SMALL40_RECORDED_SHA
    assert proof["changed_paths"] == []
    assert set(proof["unchanged_git_objects"]) == {"visual_rl", "experiments", "data"}
    assert any("merge-base" in command for command in calls)
    # Mapping the base must not authorize the current, scientifically different tree.
    with pytest.raises(RuntimeError, match="outside the allowed paths"):
        repair.prove_launcher_upgrade(ROOT, "HEAD", flash.SMALL_CONFIG)


def test_small40_owner_identity_is_not_rewritten(tmp_path, monkeypatch):
    output = tmp_path / "run-002"
    output.mkdir()
    monkeypatch.setattr(repair, "REPAIR_OUTPUT", output)
    identity = dict(git_sha="f" * 40, source_config_sha256=hashlib.sha256(b"config").hexdigest())
    proof = dict(base_sha=SMALL40_RECORDED_SHA, source_config_sha256=identity["source_config_sha256"])
    monkeypatch.setattr(repair, "prove_launcher_upgrade", lambda *args: proof)
    owner_file = output / "flash140-owner.json"
    old_owner = dict(identity, git_sha=SMALL40_RECORDED_SHA)
    owner_file.write_text(json.dumps(old_owner))
    before = owner_file.read_bytes()

    def authorize(base):
        return repair.resume_owner(output, identity, output / "checkpoints/step-2",
                                   upgrade_from=base, root=ROOT, config=flash.SMALL_CONFIG)

    assert authorize(SMALL40_RECORDED_SHA) == (old_owner, dict(identity, launcher_upgrade=proof))
    assert owner_file.read_bytes() == before
    with pytest.raises(RuntimeError, match="different source/config"):
        authorize(repair.REPAIR_GIT_BASE)
    owner_file.write_text(json.dumps(dict(identity, git_sha=repair.REPAIR_GIT_BASE)))
    with pytest.raises(RuntimeError, match="different source/config"):
        authorize(SMALL40_RECORDED_SHA)


@pytest.mark.parametrize("kind", ["checkpoint", "observation"])
@pytest.mark.parametrize("failure", [None, "ancestry", "paths", "trees", "dirty"])
def test_h3_git_proofs_use_only_rewritten_base(monkeypatch, kind, failure):
    calls = []
    head = "f" * 40

    def git(*args):
        calls.append(args)
        assert not any(arg.partition(":")[0] in RECORDED_TO_GIT_SHA for arg in args)
        if args[0] == "status":
            return " M unexpected" if failure == "dirty" else ""
        if args == ("rev-parse", "HEAD"):
            return head
        if args[0] == "merge-base":
            assert args == ("merge-base", media.BASE_GIT_SHA, head)
            return "unrelated" if failure == "ancestry" else media.BASE_GIT_SHA
        if args[0] == "diff":
            assert args == ("diff", "--name-only", media.BASE_GIT_SHA, head)
            return "data/changed.json" if failure == "paths" else "scripts/launch_h3_media.py"
        if args[0] == "rev-parse":
            sha, path = args[1].split(":", 1)
            assert sha in (media.BASE_GIT_SHA, head)
            return "different-tree" if failure == "trees" and sha == head else "tree-" + path
        raise AssertionError(args)

    if kind == "checkpoint":
        monkeypatch.setattr(upgrade, "git", lambda root, *args: git(*args))
        prove = lambda: upgrade.prove(ROOT, head)
    else:
        monkeypatch.setattr(media.subprocess, "check_output", lambda cmd, **kw: git(*cmd[3:]))
        prove = media.prove_observation_only
    if failure is not None:
        with pytest.raises(ValueError):
            prove()
    else:
        proof = prove()
        assert proof["base_sha"] == H3_RECORDED_SHA
        assert ("merge-base", media.BASE_GIT_SHA, head) in calls
        assert any(args[0] == "rev-parse" and args[1].startswith(media.BASE_GIT_SHA + ":")
                   for args in calls)


@pytest.mark.parametrize("legacy_head,code_hash,valid", [
    (upgrade.BASE_GIT_SHA, upgrade.OLD_CODE, True),
    (H3_RECORDED_SHA, upgrade.OLD_CODE, False),
    (upgrade.BASE_GIT_SHA, "changed-code", False),
])
def test_legacy_checkout_uses_new_git_pin_but_old_content_hash(
        tmp_path, monkeypatch, legacy_head, code_hash, valid):
    from visual_rl.composition.preflight import artifacts

    monkeypatch.setattr(upgrade, "prove", lambda *args: dict(base_sha=H3_RECORDED_SHA))
    monkeypatch.setattr(upgrade, "git", lambda root, *args: legacy_head if args[0] == "rev-parse" else "")
    monkeypatch.setattr(artifacts, "_hash_artifact", lambda *args, **kw: dict(content_sha256=code_hash))
    kwargs = dict(root=ROOT, legacy_root=tmp_path, expected_sha="f" * 40)
    if valid:
        result = upgrade.H3CodeUpgrade(**kwargs)
        assert result.legacy_code["content_sha256"] == upgrade.OLD_CODE
        assert result.proof["base_sha"] == H3_RECORDED_SHA
        assert result.admitted is False
    else:
        with pytest.raises(ValueError):
            upgrade.H3CodeUpgrade(**kwargs)


@pytest.mark.parametrize("script,recorded", [
    ("launch_flash140.py", SMALL40_RECORDED_SHA),
    ("launch_h3_media.py", H3_RECORDED_SHA),
])
def test_direct_launcher_help_needs_no_pythonpath(script, recorded):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
                            cwd=ROOT, env=env, text=True, capture_output=True,
                            check=True, timeout=30)
    assert recorded in result.stdout
    assert git_revision(recorded) not in result.stdout


@pytest.mark.parametrize("kind,recorded", [("small40", SMALL40_RECORDED_SHA), ("h3", H3_RECORDED_SHA)])
def test_upgrade_cli_choices_still_name_recorded_artifacts(monkeypatch, kind, recorded):
    class ParsedRecordedIdentity(Exception):
        pass

    def stop_before_execution(*args, **kwargs):
        raise ParsedRecordedIdentity

    if kind == "small40":
        monkeypatch.setattr(flash, "git_identity", stop_before_execution)

        def invoke(value):
            flash.main(["preflight", "--preset", "small40", "--expected-sha", "f" * 40,
                        "--output", "unused", "--resume", "unused/checkpoints/step-2",
                        "--upgrade-launcher-from", value])
    else:
        monkeypatch.setattr(upgrade, "H3CodeUpgrade", stop_before_execution)

        def invoke(value):
            monkeypatch.setattr(sys, "argv", ["launch_h3_media", "unused.yaml",
                "--ffmpeg", "unused", "--ffprobe", "unused", "--legacy-source", "unused",
                "--expected-sha", "f" * 40, "--upgrade-from", value])
            media.main()

    with pytest.raises(ParsedRecordedIdentity):
        invoke(recorded)
    with pytest.raises(SystemExit) as rejected:
        invoke(git_revision(recorded))
    assert rejected.value.code == 2
