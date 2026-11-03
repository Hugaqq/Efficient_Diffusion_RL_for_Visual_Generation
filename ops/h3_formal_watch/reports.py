"""Local-only Markdown reports, bounded sections and crash-safe append-once writes."""
from pathlib import Path

from .common import Store, digest

SECTION_LIMIT = 8000  # Unicode characters, including the dated heading and marker


def append_once(store, name, marker, text):
    """Caller holds the report lock. Whole-file temp+fsync+rename, never truncate history."""
    path = store.path(name)
    previous = path.read_text(encoding='utf-8') if path.exists() else ''
    if marker not in previous.splitlines():
        store.text(name, previous + ('\n' if previous and not previous.endswith('\n') else '') + text)


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
                with self.progress.lock('.h3-progress.lock'):
                    append_once(self.progress, self.progress_name, line, line + '\n')
        return self.path(daily)
