"""Reward-ranked rollout video retention for the formal H3 run.

Only on export updates (every N-th update plus the terminal update) does each
owner rank encode every local sample once, right after the ORIGINAL score call
and while the media lease is still live, into a per-rank staging MP4 (same
encoder as launch_h3_media). Once all 32 rewards are global, every rank keeps
only its selected samples -- member0 (reward independent), and per prompt
group the highest and lowest reward, ties broken by the smaller sample_id --
and deletes the rest. Rewards, scoring order, credit and training math are
untouched; nothing is encoded on other updates.

Layout: rollout-videos/step-NNNN/{member0,best,worst}/<sample_id>/
        {video.mp4,receipt.json} and rollout-videos/step-NNNN/selection.json.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time

from scripts.launch_h3_media import ENCODE_TIMEOUT_S, encode_av, reservation, sha256_file, validate_av

ROLES = ('member0', 'best', 'worst')
POLICY = ('h3-rollout-retention-v1: member0 of each prompt group plus the group max/min reward; '
          'ties broken by the lexicographically smallest sample_id; chosen after all rewards are global')
_SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,199}')
_STEP_DIR = re.compile(r'step-(\d{4,})')


def export_step(step, *, every, terminal_step, also=()):
    """True on updates every, 2*every, ..., on the terminal update and on the extra updates `also`."""
    for name, value in (('step', step), ('every', every), ('terminal_step', terminal_step), *(('also', a) for a in also)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    return step % every == 0 or step == terminal_step or step in also


def export_steps(*, every, terminal_step, after=0, also=()):
    return [step for step in range(after + 1, terminal_step + 1)
            if export_step(step, every=every, terminal_step=terminal_step, also=also)]


def sample_dirname(sample_id):
    return sample_id if _SAFE_NAME.fullmatch(sample_id) else hashlib.sha256(sample_id.encode()).hexdigest()


def select_roles(rows, *, groups=4, group_size=8):
    """rows: (sample_id, group_id, member, reward). Returns ({sid: roles}, {sid: rank in group})."""
    by_group, seen = {}, set()
    for row in rows:
        if type(row) not in (tuple, list) or len(row) != 4:
            raise ValueError('reward row must be (sample_id, group_id, member, reward)')
        sid, gid, member, reward = row
        if (type(sid) is not str or not sid or type(gid) is not str or sid in seen or type(member) is not int
                or type(reward) is not float or not math.isfinite(reward)):
            raise ValueError('duplicate/invalid/nonfinite reward row')
        seen.add(sid)
        by_group.setdefault(gid, []).append((sid, gid, member, reward))
    if len(by_group) != groups or any(len(members) != group_size for members in by_group.values()):
        raise ValueError(f'expected {groups} complete prompt groups of {group_size}')
    roles, ranks = {}, {}
    for members in by_group.values():
        if sorted(member for _, _, member, _ in members) != list(range(group_size)):
            raise ValueError('prompt group members must be exactly 0..group_size-1')
        ranked = sorted(members, key=lambda r: (-r[3], r[0]))
        ranks.update((r[0], index + 1) for index, r in enumerate(ranked))
        chosen = dict(member0=next(r[0] for r in members if r[2] == 0), best=ranked[0][0],
                      worst=min(members, key=lambda r: (r[3], r[0]))[0])
        for role in ROLES:
            roles.setdefault(chosen[role], []).append(role)
    return {sid: tuple(value) for sid, value in roles.items()}, ranks


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RolloutRetention:
    def __init__(self, *, root, rank, ffmpeg, ffprobe, launcher_sha, every, terminal_step,
                 run_seed, identity=lambda sample_id: {}, also=()):
        also = tuple(also)
        export_step(1, every=every, terminal_step=terminal_step, also=also)  # strict ints
        if type(rank) is not int or not 0 <= rank < 8:
            raise ValueError('rank must be in 0..7')
        self.root, self.rank = Path(root), rank
        self.staging = self.root / '.staging' / f'rank-{rank}'
        self.ffmpeg, self.ffprobe, self.launcher_sha = ffmpeg, ffprobe, launcher_sha
        self.every, self.terminal_step, self.run_seed, self.also = every, terminal_step, run_seed, also
        self.identity = identity
        self.pending, self.step = {}, None

    def is_export(self, step):
        return export_step(step, every=self.every, terminal_step=self.terminal_step, also=self.also)

    def observe(self, original, *, source, cancel=lambda: None):
        """Original score first, then stage; resolve this update's cancel event after scoring."""
        def score(*, sample_id, prompt, media):
            row = reservation(source, sample_id)
            if row['prompt'] != prompt:
                raise ValueError('observed prompt differs from reservation')
            reward = original(sample_id=sample_id, prompt=prompt, media=media)
            if self.is_export(row['step']):
                ordinal = [sid for sid, _, _ in source._pending.samples].index(sample_id)
                self.stage(dict(row, reservation_index=ordinal), media=media, reward=reward, cancel=cancel())
            return reward
        return score

    def stage(self, row, *, media, reward, cancel=None):
        from visual_rl.runtime import h3_memory as memory
        if cancel is not None and cancel.is_set():
            raise RuntimeError('parallel owners aborted; no new retention encode')
        step, sid = row['step'], row['sample_id']
        if self.step not in (None, step) or sid in self.pending:
            raise ValueError('staged samples of another update were not finalized')
        if type(reward) is not float or not math.isfinite(reward):
            raise ValueError('reward must be a finite Python float')
        validate_av(media)
        parent = self.staging / f'step-{step:04d}'
        parent.mkdir(parents=True, exist_ok=True)
        self.step = step
        started = time.monotonic()
        temporary = Path(tempfile.mkdtemp(prefix='.encode-', dir=parent))
        try:
            with memory.span('video_stage', sample_id=sid, step=step):
                # Cancellation may arrive during validation/staging preparation.
                # Once admitted here, this one encode finishes under its own
                # subprocess timeouts; subsequent samples must not start one.
                if cancel is not None and cancel.is_set():
                    raise RuntimeError('parallel owners aborted; no new retention encode')
                facts = encode_av(temporary, media=media, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
            final = parent / sample_dirname(sid)
            os.rename(temporary, final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        metadata = dict(row, **self.identity(sid), run_seed=self.run_seed)
        seconds = time.monotonic() - started
        self.pending[sid] = dict(metadata=metadata, reward=reward, facts=facts, directory=final,
                                 encode_seconds=seconds)
        print(json.dumps(dict(event='h3_rollout_video_staged', step=step, sample_id=sid, rank=self.rank,
                              seconds=seconds, bytes=(final / 'video.mp4').stat().st_size)), flush=True)

    def local_rows(self, step):
        if self.step != step or not self.pending:
            raise ValueError('no staged samples for this export update')
        return [[sid, p['metadata']['group_id'], p['metadata']['member'], p['reward']]
                for sid, p in sorted(self.pending.items())]

    def publish(self, step, tables):
        """tables: exchanged per-rank local_rows. Keep this rank's selected files, delete the rest."""
        if type(tables) is not list or len(tables) != 8 or self.step != step:
            raise ValueError('publication requires the eight exchanged reward tables of this update')
        if sorted(tuple(r) for r in tables[self.rank]) != sorted(tuple(r) for r in self.local_rows(step)):
            raise ValueError('exchanged local reward table differs from the staged samples')
        roles, ranks = select_roles([tuple(row) for table in tables for row in table])
        step_dir = self.root / f'step-{step:04d}'
        published = []
        for sid, pending in sorted(self.pending.items()):
            for role in roles.get(sid, ()):
                published.append(self._publish_one(step_dir / role / sample_dirname(sid), pending,
                                                   role=role, roles=roles[sid], group_rank=ranks[sid]))
        if self.rank == 0:
            flat = sorted(tuple(row) for table in tables for row in table)
            summary = dict(step=step, policy=POLICY, launcher_sha=self.launcher_sha,
                           interpretation='rollout for upcoming optimizer update; not a post-update evaluation',
                           samples=[dict(sample_id=sid, group_id=gid, member=member, reward=reward,
                                         group_reward_rank=ranks[sid], roles=list(roles.get(sid, ())))
                                    for sid, gid, member, reward in flat])
            self._write_new(step_dir / 'selection.json', summary)
        self.discard()
        return published

    def _write_new(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'refusing to overwrite {path}')
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        _fsync_dir(path.parent)

    def _publish_one(self, target, pending, *, role, roles, group_rank):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError('refusing to overwrite an existing rollout sample')
        temporary = Path(tempfile.mkdtemp(prefix='.pending-', dir=target.parent))
        try:
            source, video = pending['directory'] / 'video.mp4', temporary / 'video.mp4'
            try:
                os.link(source, video)  # same filesystem: no copy; roles of one sample share the file
            except OSError:
                shutil.copyfile(source, video)
            if sha256_file(video) != pending['facts']['video_sha256']:
                raise ValueError('staged rollout video changed before publication')
            # roles: every role this sample holds (all three when its whole group ties).
            receipt = dict(pending['metadata'], role=role, roles=list(roles), reward=pending['reward'],
                           group_reward_rank=group_rank,
                           selection_policy=POLICY, launcher_sha=self.launcher_sha,
                           encode_seconds=pending['encode_seconds'], **pending['facts'])
            (temporary / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
            for file in (video, temporary / 'receipt.json'):
                with file.open('rb') as stream:
                    os.fsync(stream.fileno())
            os.rename(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        _fsync_dir(target.parent)
        print(json.dumps(dict(event='h3_rollout_video_saved', step=pending['metadata']['step'], role=role,
                              sample_id=pending['metadata']['sample_id'], path=str(target / 'video.mp4'),
                              reward=pending['reward'], group_reward_rank=group_rank)), flush=True)
        return target

    def discard(self):
        """Delete this rank's staging (success, failure, or stale files from a crashed attempt)."""
        self.pending, self.step = {}, None
        if self.staging.exists():
            shutil.rmtree(self.staging)

    def supersede(self, *, first_step, label):
        """Rank0 on (re)start: move step dirs of updates after the restored boundary aside, never delete."""
        moved = []
        if not self.root.is_dir():
            return moved
        for path in sorted(self.root.iterdir()):
            match = _STEP_DIR.fullmatch(path.name)
            if match and path.is_dir() and int(match.group(1)) > first_step:
                destination = self.root / 'superseded' / label / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(path, destination)
                moved.append(path.name)
        return moved
