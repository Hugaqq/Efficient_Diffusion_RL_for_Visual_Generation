"""Formal H3 Flash-GRPO run (experiment card F1): 200 optimizer updates at 384x256.

Fresh start, or resume of the SAME run (same output dir and opt-in flags; the
lifecycle enforces run/visual_rl-code/topology ownership and the latest complete
checkpoint; a different launcher Git SHA needs --allow-launcher-upgrade and may
change only scripts/ and tests/; such an upgrade can still change training-side
wrappers kept in scripts/ -- iteration/score wrappers, audits -- so every upgrade
must be reviewed before use). A checkpoint at the target is a finalize-only resume
(no updates; terminal validation and SUCCESS). Reuses the two-update acceptance backend:
per-update graph/count/Adam audit, journal and card-monitor health before
every update (degraded is non-fatal; stop only on healthy=false or a health
file older than 30 s), terminal save validation. Refuses to start when the card
monitor's remaining lifetime/bytes or the output filesystem's space/inodes cannot
cover the remaining updates. Adds reward-ranked rollout
video retention at update 1, every 5 updates and the terminal one, a disk preflight, a
per-launch manifest, and optional (default off) per-update weight hashing.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import time

from scripts.launch_h3_one_update import TRAIN_SHA, TEST_SHA
from scripts.launch_h3_resolution_memory import TwoUpdateBackend, add_common_arguments
from scripts import h3_video_retention as retention
from visual_rl.runtime.h3_lifecycle import _phase, _write_json
from visual_rl.runtime.controller import RunController
from visual_rl.runtime import h3_memory as memory
from visual_rl.algorithms.optimization.h3_world import _exchange
from visual_rl.composition.config.h3_profile import compile_h3_profile

ROOT = Path(__file__).resolve().parents[1]
FORMAL_UPDATES = (200,)          # the only authorized target; kept a validated parameter
FORMAL_GEOMETRY = (384, 256, 124, 24, 32, 64)  # width, height, frames, fps, LoRA rank, alpha
CHECKPOINT_EVERY = 5
EXPORT_EVERY = 5
EXPORT_ALSO = (1,)               # update 1: rollout of the untrained base (zero LoRA)
UPGRADE_SCOPE = ('scripts/', 'tests/')  # launcher-only upgrades; visual_rl/ and all else refused
VIDEOS_PER_EXPORT = 12           # member0 + best + worst for each of 4 prompt groups
CHECKPOINT_BYTES = 2_023_338_320  # measured M1 step-1 checkpoint
CHECKPOINT_FILES = 15_025        # measured M1 step-1 checkpoint file count
VIDEO_BYTES = 2 * 1024**2        # conservative per retained MP4 (M1 480x320: about 0.6 MB)
FILES_PER_EXPORT = 64            # 12 x (dir + mp4 + receipt) + step/role dirs + selection.json, rounded up
JOURNAL_PROCESSES = 9            # 8 training ranks + the HPS service, each with its own journal budget
OTHER_LOG_BYTES = 2 * 1024**3    # stdout/stderr and HPS service logs
MONITOR_SECONDS_PER_UPDATE = 7200  # conservative: M1 measured 88 min/update (480x320, incl. save)
MONITOR_ROW_BYTES = 32 * 1024    # conservative card-process.jsonl row; the observed mean is used if larger
DISK_MARGIN = 1.2
IDENTITY_FILE = 'formal-run.json'


def validate_formal(profile, *, max_optimizer_steps=200):
    if type(max_optimizer_steps) is not int or max_optimizer_steps not in FORMAL_UPDATES:
        raise ValueError(f'formal run accepts max_optimizer_steps in {FORMAL_UPDATES} only')
    t, l, c = profile.training, profile.launch, profile.model.config
    math_payload = t.payload()
    if (l.max_optimizer_steps != max_optimizer_steps or l.checkpoint_every_optimizer_steps != CHECKPOINT_EVERY
            or t.num_steps != 10 or t.algorithm != 'dual_flash' or t.group_layout != 'paired_4x8' or t.group_size != 8
            or math_payload.get('lr') != 1e-4 or tuple(math_payload.get('stochastic_indices', ())) != (0, 1, 2, 3, 4)
            or profile.expected['train_sha256'] != TRAIN_SHA or profile.expected['eval_sha256'] != TEST_SHA
            or (c.width, c.height, c.frames, c.fps, c.lora_rank, c.lora_alpha) != FORMAL_GEOMETRY):
        raise ValueError('formal run requires author-data paired_4x8 dual_flash first5of10, lr1e-4, '
                         f'384x256x124@24 LoRA32/64, max{max_optimizer_steps}, cadence{CHECKPOINT_EVERY}')
    return profile


def resume_step(profile):
    """None for a fresh start; else the optimizer step of this run's resume checkpoint."""
    path = profile.launch.resume_from
    if path is None:
        return None
    match = re.fullmatch(r'step-(\d+)', path.name)
    if (match is None or path.parent != profile.launch.output_dir / 'checkpoints'
            or not 0 < int(match.group(1)) <= profile.launch.max_optimizer_steps):
        raise ValueError('resume_from must be <output_dir>/checkpoints/step-N of this run, 0 < N <= target '
                         '(N == target: finalize-only, no updates)')
    return int(match.group(1))


def checkpoint_steps(*, total, after=0):
    """Saves the lifecycle will publish: step 1 (first formal flash), every 5, terminal."""
    return [s for s in range(after + 1, total + 1) if s == 1 or s % CHECKPOINT_EVERY == 0 or s == total]


def disk_need(*, total, log_bytes, after=0, checkpoint_bytes=CHECKPOINT_BYTES, video_bytes=VIDEO_BYTES,
              checkpoint_files=CHECKPOINT_FILES, margin=DISK_MARGIN):
    """Bytes and inodes for the checkpoints/exports AFTER update `after`, plus logs, times margin."""
    checkpoints = len(checkpoint_steps(total=total, after=after))
    exports = len(retention.export_steps(every=EXPORT_EVERY, terminal_step=total, after=after, also=EXPORT_ALSO))
    raw = checkpoints * checkpoint_bytes + exports * VIDEOS_PER_EXPORT * video_bytes + log_bytes
    files = checkpoints * checkpoint_files + exports * FILES_PER_EXPORT
    return dict(checkpoints=checkpoints, export_updates=exports, checkpoint_bytes=checkpoint_bytes,
                checkpoint_files=checkpoint_files, video_bytes=video_bytes, log_bytes=log_bytes, margin=margin,
                need_bytes=math.ceil(margin * raw), need_inodes=math.ceil(margin * files))


def disk_free(path, *, statvfs=os.statvfs):
    """Free bytes/inodes of the filesystem holding path; free_inodes None when not reported."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    stat = statvfs(probe)
    return dict(filesystem_probe=str(probe), free_bytes=stat.f_bavail * stat.f_frsize,
                free_inodes=stat.f_favail if stat.f_files else None)


def disk_preflight(profile, *, after=0, statvfs=os.statvfs, **sizes):
    """Refuse to start when the output filesystem cannot hold the remaining run."""
    need = disk_need(total=profile.launch.max_optimizer_steps, after=after, **sizes)
    report = dict(need, **disk_free(profile.launch.output_dir, statvfs=statvfs))
    short = report['free_bytes'] < need['need_bytes']
    short_inodes = report['free_inodes'] is not None and report['free_inodes'] < need['need_inodes']
    if short or short_inodes:
        raise RuntimeError(f'insufficient disk on {report["filesystem_probe"]}: free {report["free_bytes"]} B / '
                           f'{report["free_inodes"]} inodes < need {need["need_bytes"]} B / {need["need_inodes"]} '
                           f'inodes ({need["checkpoints"]} checkpoints, {need["export_updates"]} video exports, '
                           f'logs, {need["margin"]}x margin)')
    return report


def monitor_budget(directory, *, remaining_updates, seconds_per_update=MONITOR_SECONDS_PER_UPDATE,
                   row_bytes=MONITOR_ROW_BYTES, now_ns=None):
    """Refuse when the card monitor would exhaust its lifetime/byte budget before the run ends."""
    health = json.loads((Path(directory) / 'health.json').read_text())
    if health.get('healthy') is not True:
        raise RuntimeError('card monitor unhealthy; cannot start the formal run')
    keys = ('interval_seconds', 'max_seconds', 'max_bytes', 'started_monotonic_ns', 'bytes', 'sample_count')
    missing = [k for k in keys if k not in health]
    if missing:
        raise RuntimeError(f'card monitor health lacks budget fields {missing}; restart it from this checkout')
    now = time.monotonic_ns() if now_ns is None else now_ns
    remaining_seconds = health['max_seconds'] - (now - health['started_monotonic_ns']) / 1e9
    remaining_bytes = health['max_bytes'] - health['bytes']
    row = max(row_bytes, health['bytes'] / health['sample_count'] if health['sample_count'] else 0)
    need_seconds = remaining_updates * seconds_per_update
    if need_seconds and not health['interval_seconds'] > 0:
        raise RuntimeError('card monitor interval must be positive for a run with remaining updates')
    need_bytes = math.ceil(math.ceil(need_seconds / health['interval_seconds']) * row) if need_seconds else 0
    report = dict(remaining_updates=remaining_updates, seconds_per_update=seconds_per_update, row_bytes=row,
                  remaining_seconds=remaining_seconds, need_seconds=need_seconds,
                  remaining_bytes=remaining_bytes, need_bytes=need_bytes, max_bytes=health['max_bytes'])
    if remaining_seconds < need_seconds or remaining_bytes < need_bytes:
        raise RuntimeError(f'card monitor budget cannot cover {remaining_updates} updates: {report}; '
                           'restart the monitor with larger --max-seconds/--max-bytes (or a longer --interval)')
    return report


def extension_flags(args):
    """Extension point: S1 (rollout batching) and P2 (parallel owners) live on other
    branches and are NOT accepted here. When one is merged and separately measured,
    register its argparse flag next to add_common_arguments() and return it here so
    it becomes part of the resume identity and the launch manifest."""
    return {}


def formal_identity(profile, args):
    """What a resume must not change (observation-only switches excluded). The training
    core (visual_rl code hash, semantics, topology) is enforced by the lifecycle owner check;
    the launcher Git SHA is tracked separately in formal-run.json (see check_resume_identity)."""
    return dict(schema='h3.formal-run.v2',
                max_optimizer_steps=profile.launch.max_optimizer_steps,
                checkpoint_every=CHECKPOINT_EVERY, export_every=EXPORT_EVERY, export_also=list(EXPORT_ALSO),
                retention=retention.POLICY,
                geometry=list(FORMAL_GEOMETRY), flags=dict(
                    reward_pipeline=args.reward_pipeline, reuse_video_vae=args.reuse_video_vae,
                    reuse_rollout_anchor=args.reuse_rollout_anchor,
                    boundary_content_check=args.boundary_content_check,
                    checkpoint_threads=args.checkpoint_threads, **extension_flags(args)))


def fresh_record(profile, args):
    """formal-run.json at the fresh start: identity, launcher SHA, empty upgrade history."""
    return dict(identity=formal_identity(profile, args), launcher_sha=args.expected_sha, launcher_upgrades=[])


def current_launcher_sha(record):
    """The launcher SHA of the last APPLIED upgrade; unapplied attempts never count."""
    applied = [u for u in record['launcher_upgrades'] if u.get('status') == 'applied']
    return applied[-1]['new_sha'] if applied else record['launcher_sha']


def git_changed_files(old, new, *, root=ROOT):
    """Paths differing between two commits, both sides of every rename (--no-renames, NUL-split)."""
    for sha in (old, new):
        if subprocess.run(['git', '-C', str(root), 'cat-file', '-e', sha + '^{commit}'],
                          capture_output=True).returncode:
            raise ValueError(f'commit {sha} is not present in {root}; fetch it before a launcher upgrade')
    raw = subprocess.check_output(['git', '-C', str(root), 'diff', '--no-renames', '--name-only', '-z', old, new])
    return [name.decode() for name in raw.split(b'\0') if name]


def check_resume_identity(profile, args, *, changed_files=git_changed_files, now=None, attempt=None):
    """Refuse a changed identity. A different launcher SHA is refused unless
    --allow-launcher-upgrade with --upgrade-reason, and only when every changed file is
    under scripts/ or tests/. Returns the upgrade history entry to append, or None."""
    path = profile.launch.output_dir / IDENTITY_FILE
    if not path.is_file():
        raise ValueError(f'resume requires {IDENTITY_FILE} written by this run\'s fresh start')
    record = json.loads(path.read_text())
    saved, current = record.get('identity'), formal_identity(profile, args)
    if saved != current:
        keys = set(saved or {}) | set(current)
        changed = sorted(k for k in keys if (saved or {}).get(k) != current.get(k))
        raise ValueError(f'refusing resume across a changed formal-run identity: {changed}')
    old, new = current_launcher_sha(record), args.expected_sha
    if old == new:
        if args.allow_launcher_upgrade or args.upgrade_reason is not None:
            raise ValueError('--allow-launcher-upgrade/--upgrade-reason given but the launcher SHA is unchanged')
        return None
    if not args.allow_launcher_upgrade:
        raise ValueError(f'refusing resume with a different launcher SHA ({old} -> {new}); '
                         'pass --allow-launcher-upgrade with --upgrade-reason for a scripts/-only change')
    if type(args.upgrade_reason) is not str or not args.upgrade_reason.strip():
        raise ValueError('--allow-launcher-upgrade requires a nonempty --upgrade-reason')
    files = sorted(changed_files(old, new))
    outside = [f for f in files if not f.startswith(UPGRADE_SCOPE)]
    if outside:
        raise ValueError(f'launcher upgrade may change only scripts/ and tests/; also changed: {outside}')
    stamp = now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    return dict(old_sha=old, new_sha=new, utc=stamp, reason=args.upgrade_reason.strip(),
                changed_scripts=[f for f in files if f.startswith('scripts/')],
                changed_tests=[f for f in files if f.startswith('tests/')], status='attempt', attempt=attempt)


def record_launcher_upgrade(output, upgrade):
    """Rank0, after the lifecycle admitted the resume: append the upgrade as an 'attempt'."""
    path = output / IDENTITY_FILE
    record = json.loads(path.read_text())
    if current_launcher_sha(record) != upgrade['old_sha'] or upgrade.get('status') != 'attempt':
        raise ValueError('formal-run.json launcher SHA changed since the resume check')
    _write_json(path, dict(record, launcher_upgrades=[*record['launcher_upgrades'], upgrade]))


def mark_launcher_upgrade_applied(output, attempt, *, now=None):
    """Rank0, after restore_bound succeeded: the attempt becomes the current launcher SHA."""
    path = output / IDENTITY_FILE
    record = json.loads(path.read_text())
    rows = record['launcher_upgrades']
    match = [i for i, u in enumerate(rows) if u.get('attempt') == attempt and u.get('status') == 'attempt']
    if len(match) != 1:
        raise ValueError('no unique pending launcher upgrade attempt to apply')
    stamp = now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    rows = [dict(u, status='applied', applied_utc=stamp) if i == match[0] else u for i, u in enumerate(rows)]
    _write_json(path, dict(record, launcher_upgrades=rows))


class FormalBackend(TwoUpdateBackend):
    monitor_max_age = 30

    def __init__(self, *, args, manifest):
        super().__init__(args=args)
        self.audit_weights = args.audit_weights
        self.manifest = manifest
        self.retention = None

    def validate_profile(self, profile):
        return validate_formal(profile, max_optimizer_steps=self.args.max_optimizer_steps)

    def construct_graph(self, preflight, runtime):
        graph = super().construct_graph(preflight, runtime)
        output = graph.profile.launch.output_dir
        def record():
            if runtime.rank != 0:
                return
            if graph.profile.launch.resume_from is None:
                identity = output / IDENTITY_FILE
                if identity.exists():
                    raise FileExistsError('fresh formal run found an existing identity file')
                _write_json(identity, fresh_record(graph.profile, self.args))
            elif self.manifest.get('launcher_upgrade') is not None:
                record_launcher_upgrade(output, self.manifest['launcher_upgrade'])
            (output / 'formal-launches').mkdir(exist_ok=True)
            _write_json(output / 'formal-launches' / f"launch-{self.manifest['launch_stamp']}-pid{os.getpid()}.json",
                        dict(self.manifest, run_id=graph.run_id, semantic_id=preflight.facts.semantic_id,
                             binding_id=preflight.facts.binding_id, code_hash=preflight.code_hash))
        try:
            _phase(runtime, 'formal/manifest', record)
            self._log_disk_at_checkpoints(graph, runtime)
        except BaseException:
            graph.close()
            raise
        return graph

    def restore_bound(self, preflight, runtime, graph, prepared, bound, prepared_restore):
        result = super().restore_bound(preflight, runtime, graph, prepared, bound, prepared_restore)
        upgrade = self.manifest.get('launcher_upgrade')
        if upgrade is not None:
            def applied():
                if runtime.rank == 0:
                    mark_launcher_upgrade_applied(graph.profile.launch.output_dir, upgrade['attempt'])
            _phase(runtime, 'formal/launcher-upgrade-applied', applied)
        return result

    def _log_disk_at_checkpoints(self, graph, runtime):
        launch, save = graph.profile.launch, self.transaction.checkpoint
        def report(when, step, after):
            """Rank0-only observation around a collective save: never raises (peers would hang)."""
            if runtime.rank != 0:
                return
            try:
                free = disk_free(launch.output_dir)
                need = disk_need(total=launch.max_optimizer_steps, after=after, log_bytes=0,
                                 checkpoint_bytes=self.args.checkpoint_bytes, video_bytes=self.args.video_bytes,
                                 checkpoint_files=self.args.checkpoint_files)
                row = dict(free, when=when, step=step, remaining_checkpoints=need['checkpoints'],
                           remaining_need_bytes=need['need_bytes'], remaining_need_inodes=need['need_inodes'],
                           sufficient_for_remaining=free['free_bytes'] >= need['need_bytes'] and (
                               free['free_inodes'] is None or free['free_inodes'] >= need['need_inodes']))
                memory.event('formal_disk', **row)
            except Exception as exc:
                row = dict(when=when, step=step, error=repr(exc))
            print(json.dumps(dict(event='h3_formal_disk', **row)), flush=True)
        def checkpoint(*args, **kwargs):
            step = getattr(kwargs.get('progress'), 'next_optimizer_step', None)
            # In-progress save: its own checkpoint is still part of the remaining need.
            report('before_save', step, step - 1 if type(step) is int else launch.max_optimizer_steps)
            committed = save(*args, **kwargs)
            report('after_save', committed.step, committed.step)
            return committed
        self.transaction.checkpoint = checkpoint

    def install_media(self, bound, graph, runtime):
        launch = graph.profile.launch
        self.retention = retention.RolloutRetention(root=launch.output_dir / 'rollout-videos', rank=runtime.rank,
            ffmpeg=self.args.ffmpeg, ffprobe=self.args.ffprobe, launcher_sha=self.args.expected_sha,
            every=EXPORT_EVERY, also=EXPORT_ALSO, terminal_step=launch.max_optimizer_steps, run_seed=launch.seed,
            identity=lambda sid: self.rollout_identity(bound, graph, sid))
        self.rank = runtime.rank
        bound.stages.score_one = self.retention.observe(bound.stages.score_one, source=graph.source)

    def check_start(self, step, graph, prepared):
        if self.start_step == 0:
            super().check_start(step, graph, prepared)
        else:
            counters = {int(s['step'].item()) for s in prepared.optimizer.state.values()}
            if (step != self.start_step or graph.source.state_dict()['next_optimizer_step'] != step
                    or counters != {step}):
                raise ValueError('restored optimizer/data position differs from the resume checkpoint')
        # Stale staging of a crashed attempt is deleted; its published step dirs are kept aside.
        self.retention.discard()
        if self.rank == 0:
            moved = self.retention.supersede(first_step=step, label='attempt-' + self.manifest['launch_stamp'])
            print(json.dumps(dict(event='h3_formal_superseded_videos', first_step=step, moved=moved)), flush=True)

    def after_update(self, step, runtime):
        if not self.retention.is_export(step + 1):
            return
        try:
            with memory.span('rollout_video_retention', step=step + 1):
                local = _phase(runtime, 'formal-media/staged', lambda: self.retention.local_rows(step + 1))
                tables = _exchange('formal-media/rewards', runtime.rank, runtime.process_group, local, True)
                _phase(runtime, 'formal-media/publish', lambda: self.retention.publish(step + 1, tables))
        finally:
            self.retention.discard()

    def update_failed(self, step):
        if self.retention is not None:
            try:
                self.retention.discard()
            except OSError as exc:
                print(json.dumps(dict(event='h3_formal_staging_cleanup_failed', step=step + 1, error=repr(exc))),
                      flush=True)


def journal_budget(mib):
    if type(mib) is not int or not 64 <= mib <= 16384:
        raise ValueError('--journal-max-mib must be an integer in [64, 16384]')
    value = str(mib * 1024**2)
    if os.environ.get('H3_MEMORY_MAX_BYTES', value) != value:
        raise ValueError('H3_MEMORY_MAX_BYTES differs from --journal-max-mib')
    os.environ['H3_MEMORY_MAX_BYTES'] = value
    return int(value)


def build_parser():
    p = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--max-optimizer-steps', type=int, default=FORMAL_UPDATES[0],
                   help=f'must equal the profile; accepted values {FORMAL_UPDATES}')
    p.add_argument('--audit-weights', action='store_true',
                   help='hash all local weight shards before/after every update (slow; default off)')
    p.add_argument('--journal-max-mib', type=int, default=512, help='per-rank memory journal budget (default 512)')
    p.add_argument('--checkpoint-bytes', type=int, default=CHECKPOINT_BYTES,
                   help='disk preflight: bytes per checkpoint (default: measured M1 step-1)')
    p.add_argument('--video-bytes', type=int, default=VIDEO_BYTES, help='disk preflight: bytes per retained MP4')
    p.add_argument('--checkpoint-files', type=int, default=CHECKPOINT_FILES,
                   help='disk preflight: files (inodes) per checkpoint (default: measured M1 step-1)')
    p.add_argument('--other-log-bytes', type=int, default=OTHER_LOG_BYTES,
                   help='disk preflight: stdout/HPS logs, added to 9 journal budgets and the monitor max-bytes')
    p.add_argument('--monitor-seconds-per-update', type=int, default=MONITOR_SECONDS_PER_UPDATE,
                   help='conservative wall time per update used to check the card monitor lifetime (default 7200)')
    p.add_argument('--monitor-row-bytes', type=int, default=MONITOR_ROW_BYTES,
                   help='conservative bytes per monitor sample (the observed mean is used when larger)')
    p.add_argument('--allow-launcher-upgrade', action='store_true',
                   help='resume only: accept a different launcher Git SHA whose changes are confined to scripts/ and tests/')
    p.add_argument('--upgrade-reason', default=None, help='required with --allow-launcher-upgrade; recorded in formal-run.json')
    return p


def main():
    args = build_parser().parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    started, stamp = now.isoformat(timespec='microseconds'), now.strftime('%Y%m%dT%H%M%S%fZ')
    def git(*argv): return subprocess.check_output(['git', '-C', str(ROOT), *argv], text=True).strip()
    if git('status', '--porcelain') or git('rev-parse', 'HEAD') != args.expected_sha:
        raise ValueError('requires exact clean deployed Git SHA')
    profile = validate_formal(compile_h3_profile(args.config), max_optimizer_steps=args.max_optimizer_steps)
    if not memory.enabled() or not args.ffmpeg.is_file() or not args.ffprobe.is_file():
        raise ValueError('requires memory journal and real media tools')
    start = resume_step(profile)
    upgrade = None
    if start is None and (args.allow_launcher_upgrade or args.upgrade_reason is not None):
        raise ValueError('--allow-launcher-upgrade/--upgrade-reason apply to a resume only')
    if start is not None:
        if args.profile_rollout:
            raise ValueError('--profile-rollout targets update 2 and is accepted on a fresh start only')
        upgrade = check_resume_identity(profile, args, attempt=stamp)
    journal_bytes = journal_budget(args.journal_max_mib)
    done = start or 0
    monitor = monitor_budget(args.monitor_dir, remaining_updates=profile.launch.max_optimizer_steps - done,
                             seconds_per_update=args.monitor_seconds_per_update, row_bytes=args.monitor_row_bytes)
    disk = disk_preflight(profile, after=done, checkpoint_bytes=args.checkpoint_bytes,
                          checkpoint_files=args.checkpoint_files, video_bytes=args.video_bytes,
                          log_bytes=JOURNAL_PROCESSES * journal_bytes + monitor['max_bytes'] + args.other_log_bytes)
    mode = 'fresh' if start is None else ('finalize' if start == profile.launch.max_optimizer_steps else 'resume')
    manifest = dict(schema='h3.formal-launch.v1', started_utc=started, launch_stamp=stamp, git_sha=args.expected_sha,
                    mode=mode, resume_from=str(profile.launch.resume_from), monitor_budget=monitor,
                    resume_step=start, launcher_upgrade=upgrade, config=str(args.config),
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    profile_source_id=profile.source_id, identity=formal_identity(profile, args),
                    audit_weights=args.audit_weights, profile_rollout=args.profile_rollout,
                    journal_max_bytes=journal_bytes, disk_preflight=disk,
                    argv={k: str(v) for k, v in sorted(vars(args).items())})
    print(json.dumps(dict(event='h3_formal_launch', **manifest)), flush=True)
    memory.journal()
    try:
        with memory.span('full_lifecycle'):
            result = RunController(FormalBackend(args=args, manifest=manifest)).run(args.config)
        if result.committed_steps != profile.launch.max_optimizer_steps:
            raise RuntimeError('did not stop at the formal update target')
        print(json.dumps(dict(status='formal_complete', optimizer_updates=result.committed_steps,
                              checkpoint=str(result.authoritative_checkpoint))), flush=True)
    except BaseException as exc:
        try: memory.journal().exception(exc)
        except BaseException: pass
        raise
    finally:
        memory.close()


if __name__ == '__main__':
    main()
