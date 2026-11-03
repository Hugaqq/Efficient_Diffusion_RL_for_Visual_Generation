"""Formal H3 Flash-GRPO run (experiment card F1): 200 optimizer updates at 384x256.

Fresh start, or resume of the SAME run (same output dir, Git SHA and opt-in
flags; the lifecycle additionally enforces run/code/topology ownership and
the latest complete checkpoint). Reuses the two-update acceptance backend:
per-update graph/count/Adam audit, journal and card-monitor health before
every update (degraded is non-fatal; stop only on healthy=false or a health
file older than 30 s), terminal save validation. Adds reward-ranked rollout
video retention every 5 updates plus the terminal one, a disk preflight, a
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
VIDEOS_PER_EXPORT = 12           # member0 + best + worst for each of 4 prompt groups
CHECKPOINT_BYTES = 2_023_338_320  # measured M1 step-1 checkpoint (15,025 files)
VIDEO_BYTES = 2 * 1024**2        # conservative per retained MP4 (M1 480x320: about 0.6 MB)
LOG_BYTES = 14 * 1024**3         # 8 GiB card monitor + 8 x 512 MiB journals + 2 GiB stdout/HPS logs
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
            or not 0 < int(match.group(1)) < profile.launch.max_optimizer_steps):
        raise ValueError('resume_from must be <output_dir>/checkpoints/step-N of this run, 0 < N < target')
    return int(match.group(1))


def checkpoint_steps(*, total, after=0):
    """Saves the lifecycle will publish: step 1 (first formal flash), every 5, terminal."""
    return [s for s in range(after + 1, total + 1) if s == 1 or s % CHECKPOINT_EVERY == 0 or s == total]


def disk_need(*, total, after=0, checkpoint_bytes=CHECKPOINT_BYTES, video_bytes=VIDEO_BYTES,
              log_bytes=LOG_BYTES, margin=DISK_MARGIN):
    checkpoints = len(checkpoint_steps(total=total, after=after))
    exports = len(retention.export_steps(every=EXPORT_EVERY, terminal_step=total, after=after))
    raw = checkpoints * checkpoint_bytes + exports * VIDEOS_PER_EXPORT * video_bytes + log_bytes
    return dict(checkpoints=checkpoints, export_updates=exports, checkpoint_bytes=checkpoint_bytes,
                video_bytes=video_bytes, log_bytes=log_bytes, margin=margin, need_bytes=math.ceil(margin * raw))


def disk_free(path, *, statvfs=os.statvfs):
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    stat = statvfs(probe)
    return str(probe), stat.f_bavail * stat.f_frsize


def disk_preflight(profile, *, after=0, statvfs=os.statvfs, **sizes):
    """Refuse to start when the output filesystem cannot hold the remaining run."""
    need = disk_need(total=profile.launch.max_optimizer_steps, after=after, **sizes)
    probe, free = disk_free(profile.launch.output_dir, statvfs=statvfs)
    report = dict(need, filesystem_probe=probe, free_bytes=free)
    if free < need['need_bytes']:
        raise RuntimeError(f'insufficient disk on {probe}: free {free} B < need {need["need_bytes"]} B '
                           f'({need["checkpoints"]} checkpoints, {need["export_updates"]} video exports, logs, '
                           f'{need["margin"]}x margin)')
    return report


def extension_flags(args):
    """Extension point: S1 (rollout batching) and P2 (parallel owners) live on other
    branches and are NOT accepted here. When one is merged and separately measured,
    register its argparse flag next to add_common_arguments() and return it here so
    it becomes part of the resume identity and the launch manifest."""
    return {}


def formal_identity(profile, args):
    """What a resume must not change (observation-only switches excluded)."""
    return dict(schema='h3.formal-run.v1', expected_sha=args.expected_sha,
                max_optimizer_steps=profile.launch.max_optimizer_steps,
                checkpoint_every=CHECKPOINT_EVERY, export_every=EXPORT_EVERY, retention=retention.POLICY,
                geometry=list(FORMAL_GEOMETRY), flags=dict(
                    reward_pipeline=args.reward_pipeline, reuse_video_vae=args.reuse_video_vae,
                    reuse_rollout_anchor=args.reuse_rollout_anchor,
                    boundary_content_check=args.boundary_content_check,
                    checkpoint_threads=args.checkpoint_threads, **extension_flags(args)))


def check_resume_identity(profile, args):
    path = profile.launch.output_dir / IDENTITY_FILE
    if not path.is_file():
        raise ValueError(f'resume requires {IDENTITY_FILE} written by this run\'s fresh start')
    saved, current = json.loads(path.read_text()), formal_identity(profile, args)
    if saved != current:
        changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        raise ValueError(f'refusing resume across a changed formal-run identity: {changed}')
    return current


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
                _write_json(identity, formal_identity(graph.profile, self.args))
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

    def _log_disk_at_checkpoints(self, graph, runtime):
        launch, save = graph.profile.launch, self.transaction.checkpoint
        def report(when, step):
            if runtime.rank != 0:
                return
            probe, free = disk_free(launch.output_dir)
            need = disk_need(total=launch.max_optimizer_steps, after=step,
                             checkpoint_bytes=self.args.checkpoint_bytes, video_bytes=self.args.video_bytes,
                             log_bytes=0)['need_bytes']
            row = dict(when=when, step=step, filesystem_probe=probe, free_bytes=free,
                       remaining_need_bytes=need, sufficient_for_remaining=free >= need)
            memory.event('formal_disk', **row)
            print(json.dumps(dict(event='h3_formal_disk', **row)), flush=True)
        def checkpoint(*args, **kwargs):
            step = getattr(kwargs.get('progress'), 'next_optimizer_step', None)
            report('before_save', step)
            committed = save(*args, **kwargs)
            report('after_save', committed.step)
            return committed
        self.transaction.checkpoint = checkpoint

    def install_media(self, bound, graph, runtime):
        launch = graph.profile.launch
        self.retention = retention.RolloutRetention(root=launch.output_dir / 'rollout-videos', rank=runtime.rank,
            ffmpeg=self.args.ffmpeg, ffprobe=self.args.ffprobe, launcher_sha=self.args.expected_sha,
            every=EXPORT_EVERY, terminal_step=launch.max_optimizer_steps, run_seed=launch.seed,
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
    p.add_argument('--log-bytes', type=int, default=LOG_BYTES, help='disk preflight: monitor/journal/stdout budget')
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
    if start is not None:
        if args.profile_rollout:
            raise ValueError('--profile-rollout targets update 2 and is accepted on a fresh start only')
        check_resume_identity(profile, args)
    disk = disk_preflight(profile, after=start or 0, checkpoint_bytes=args.checkpoint_bytes,
                          video_bytes=args.video_bytes, log_bytes=args.log_bytes)
    journal_bytes = journal_budget(args.journal_max_mib)
    manifest = dict(schema='h3.formal-launch.v1', started_utc=started, launch_stamp=stamp, git_sha=args.expected_sha,
                    mode='fresh' if start is None else 'resume', resume_from=str(profile.launch.resume_from),
                    resume_step=start, config=str(args.config),
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
