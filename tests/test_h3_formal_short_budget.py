"""启动预算：实际权重 header、真实 EMA 保存、默认 launcher 与基线对照。"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import pytest

from scripts import launch_h3_formal as formal
from tests.test_h3_mvp_cpu_profile import _fixture, _save
from visual_rl.composition.config.h3_profile import compile_h3_profile, preflight_h3_profile, H3EMAConfig


def args():
    return formal.build_parser().parse_args(['cfg.json', '--expected-sha', 'a'*40,
        '--ffmpeg', 'f', '--ffprobe', 'p', '--monitor-dir', 'm'])


def test_default_launcher_budget_parser_identity_match_7c85751(tmp_path):
    root = Path(__file__).resolve().parents[1]
    old = ModuleType('baseline_formal')
    old.__file__ = str(root/'scripts/launch_h3_formal.py')
    exec(subprocess.check_output(['git', 'show', '7c85751:scripts/launch_h3_formal.py'], cwd=root), old.__dict__)
    argv = ['cfg.json', '--expected-sha', 'a'*40, '--ffmpeg', 'f', '--ffprobe', 'p', '--monitor-dir', 'm']
    assert vars(args()) == vars(old.build_parser().parse_args(argv))
    profile = SimpleNamespace(launch=SimpleNamespace(output_dir=tmp_path, max_optimizer_steps=200))
    assert formal.fresh_record(profile, args()) == old.fresh_record(profile, args())
    for after in (0, 1, 35, 199, 200):
        sizes = dict(total=200, after=after, log_bytes=123456)
        assert formal.disk_need(**sizes) == old.disk_need(**sizes)
    assert formal.profile_ema_bytes(profile) == 0  # 默认不读模型文件


def test_short_entry_strict_target_identity_and_pause_budget(tmp_path):
    path, raw = _fixture(tmp_path)
    raw['overrides']['training'] = dict(group_size=8, num_steps=10, algorithm='dual_flash', group_layout='paired_4x8')
    raw['launch'].update(max_optimizer_steps=3, checkpoint_every_optimizer_steps=5)
    raw['launch']['expected'].update(train_sha256=formal.TRAIN_SHA, eval_sha256=formal.TEST_SHA)
    profile = compile_h3_profile(_save(tmp_path, raw))
    assert formal.validate_formal(profile, max_optimizer_steps=3, target_updates=3) is profile
    with pytest.raises(ValueError, match='formal run requires'):
        formal.validate_formal(profile)
    for target in (0, -1, 200, 201, True, 3.0):
        with pytest.raises(ValueError, match='short test'):
            formal.validate_formal(profile, max_optimizer_steps=target, target_updates=target)
    with pytest.raises(ValueError, match='short test'):
        formal.validate_formal(profile, max_optimizer_steps=2, target_updates=3)
    with pytest.raises(ValueError, match='formal run requires'):
        formal.validate_formal(replace(profile, launch=replace(profile.launch, checkpoint_every_optimizer_steps=1)),
                               max_optimizer_steps=3, target_updates=3)
    a = args(); a.target_updates = a.max_optimizer_steps = 3
    profile.launch.output_dir.mkdir(parents=True)
    (profile.launch.output_dir/formal.IDENTITY_FILE).write_text(json.dumps(formal.fresh_record(profile, a)))
    assert formal.check_resume_identity(profile, a) is None
    changed = replace(profile, launch=replace(profile.launch, max_optimizer_steps=2))
    with pytest.raises(ValueError, match='changed formal-run identity'):
        formal.check_resume_identity(changed, a)
    # step-1 自动保存 + step-2 PAUSE + step-3 终点：不能只预算两个常规保存。
    need = formal.disk_need(total=3, log_bytes=0)
    assert need['checkpoints'] == 3 and need['export_updates'] == 2
    assert formal.disk_need(total=3, after=2, log_bytes=0)['checkpoints'] == 1
    fake = SimpleNamespace(f_bavail=need['need_bytes']-1, f_frsize=1, f_favail=10**6, f_files=10**6)
    with pytest.raises(RuntimeError, match='insufficient disk'):
        formal.disk_preflight(profile, log_bytes=0, statvfs=lambda _: fake)
    monitor = tmp_path/'monitor'; monitor.mkdir()
    (monitor/'health.json').write_text(json.dumps(dict(healthy=True, interval_seconds=5, max_seconds=30000,
        max_bytes=10**9, started_monotonic_ns=0, bytes=0, sample_count=0)))
    assert formal.monitor_budget(monitor, remaining_updates=3, now_ns=0)['need_seconds'] == 21600
    with pytest.raises(RuntimeError, match='cannot cover'):
        formal.monitor_budget(monitor, remaining_updates=200, now_ns=0)


def test_header_ema_budget_and_target_semantics(tmp_path):
    path, raw = _fixture(tmp_path)
    profile = compile_h3_profile(path)
    assert formal.profile_ema_bytes(profile) == 0
    enabled = replace(profile, training=replace(profile.training, ema=H3EMAConfig(enabled=True)))
    # 六个真实 safetensors base weight (4,4)，rank32 的 A=(32,4)、B=(4,32)。
    budget = formal.profile_ema_bytes(enabled)
    assert budget == formal.ema_attachment_budget([(32,4), (4,32)]*6)
    assert budget == 6 * (32*4 + 4*32)*4 + 8*(12*8192 + 65536)
    need = formal.disk_need(total=3, log_bytes=0, ema_bytes=budget)
    assert need['checkpoint_bytes'] == formal.CHECKPOINT_BYTES + budget
    assert need['checkpoint_files'] == formal.CHECKPOINT_FILES + 9
    fake = SimpleNamespace(f_bavail=10**12, f_frsize=1, f_favail=10**6, f_files=10**6)
    assert formal.disk_preflight(enabled, log_bytes=0, statvfs=lambda _: fake)['ema_bytes'] == budget
    old = preflight_h3_profile(profile)
    changed = preflight_h3_profile(replace(profile, launch=replace(profile.launch, max_optimizer_steps=3)))
    assert old.semantic_id != changed.semantic_id
    assert old.semantic_payload['max_optimizer_steps'] == 2
    assert changed.semantic_payload['max_optimizer_steps'] == 3
    for shape in ([], [(0,3)], [(True,3)], [(3,)], [(3,-1)]):
        with pytest.raises(ValueError, match='LoRA shapes'):
            formal.ema_attachment_budget(shape)


def test_bounded_run_budget_uses_cadence_plus_one_pause():
    from scripts import launch_h3_formal as formal
    # 100 更新：step-1、5…100 共 21 份常规存档，加一次 PAUSE 预留。
    assert formal.disk_need(total=100, after=0, log_bytes=0)['checkpoints'] == 22
    assert formal.disk_need(total=100, after=20, log_bytes=0)['checkpoints'] == 17
    assert formal.disk_need(total=1, after=0, log_bytes=0)['checkpoints'] == 1
    # 正式 200 路径不变。
    assert formal.disk_need(total=200, after=0, log_bytes=0)['checkpoints'] == len(
        formal.checkpoint_steps(total=200, after=0))
