"""Local-only Markdown reports, bounded sections and crash-safe append-once writes."""
import fcntl
import os
from pathlib import Path

from .common import Store, digest

SECTION_LIMIT = 8000  # Unicode characters, including the dated heading and marker


def append_once(store, name, marker, text):
    """Caller holds the report lock. Whole-file temp+fsync+rename, never truncate history."""
    path = store.path(name)
    previous = path.read_text(encoding='utf-8') if path.exists() else ''
    if marker not in previous.splitlines():
        store.text(name, previous + ('\n' if previous and not previous.endswith('\n') else '') + text)


def append_progress(store, name, line):
    """Lock the actual inode, then append one marker-bearing line with O_APPEND.

    Other appenders need not share our lock: their bytes are never replaced. A
    writer that replaces/truncates the entire file must still coordinate externally.
    """
    path = store.path(name)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'r', encoding='utf-8') as stream:
        fcntl.flock(fd, fcntl.LOCK_EX)
        last = ''
        for previous in stream:
            if previous.rstrip('\n') == line:
                os.fsync(fd)  # replay after an earlier append succeeded but fsync failed
                return
            last = previous
        raw = (('\n' if last and not last.endswith('\n') else '') + line + '\n').encode('utf-8')
        # Regular-file appends are normally one write. Handle a short write without
        # replacing existing bytes; a failed partial line is not a complete marker.
        while raw:
            written = os.write(fd, raw)
            if not written:
                raise OSError('zero-byte progress append')
            raw = raw[written:]
        os.fsync(fd)
    directory = os.open(store.root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class Reports:
    def __init__(self, directory, progress_log_path):
        directory, progress = Path(directory), Path(progress_log_path)
        if not directory.is_absolute() or not progress.is_absolute():
            raise ValueError('report_dir and progress_log_path must be absolute local paths')
        self.store = Store(directory)
        self.progress = Store(progress.parent)
        self.progress_name = progress.name
        self.progress.path(self.progress_name)
        if progress in {directory / name for name in ('DAILY.md', 'INCIDENTS.md', 'reports.lock')}:
            raise ValueError('progress log must be distinct from report files/lock')

    def path(self, daily=False):
        return self.store.path('DAILY.md' if daily else 'INCIDENTS.md')

    def record(self, key, date, summary, diagnosis='', *, daily=False):
        # Daily keys identify a date, so restarts (even a lost state receipt) cannot
        # append two sections for that day. Incidents use config-scoped identities.
        marker = '<!-- h3-report:' + digest(key) + ' -->'
        heading = '\n' + marker + '\n## ' + date + '\n\n'
        body = summary + ('\n\n### 只读诊断\n\n' + diagnosis if diagnosis else '')
        section = heading + body[:SECTION_LIMIT - len(heading) - 2] + '\n\n'
        with self.store.lock('reports.lock'):
            append_once(self.store, self.path(daily).name, marker, section)
            if daily:
                # A separate marker-bearing single line makes a crash between the
                # report, progress and observer-state writes safe to replay.
                line = '- ' + date + ' H3 daily summary saved: ' + str(self.path(True)) + ' ' + marker
                append_progress(self.progress, self.progress_name, line)
        return self.path(daily)
