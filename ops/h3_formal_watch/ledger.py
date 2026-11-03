"""Disk-backed observer work/outbox and permanent compact dedup tombstones.

Only one observer holds observer.lock. Rows write through before side effects;
iteration streams rows instead of loading an ever-growing JSON document.
"""
from collections import deque
from collections.abc import MutableMapping
import json
import sqlite3


class Row(dict):
    def __init__(self, value, save):
        super().__init__(value)
        self.save = save

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.save(self)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.save(self)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.save(self)

    def pop(self, key, *default):
        result = super().pop(key, *default)
        self.save(self)
        return result

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]


class WorkMap(MutableMapping):
    def __init__(self, ledger, kind):
        self.ledger, self.kind = ledger, kind
        self.db = ledger.db

    def __getitem__(self, key):
        row = self.db.execute('SELECT payload FROM work WHERE kind=? AND key=?', (self.kind, key)).fetchone()
        if row is None:
            raise KeyError(key)
        return Row(json.loads(row[0]), lambda value: self.__setitem__(key, value))

    def __setitem__(self, key, value):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
        self.db.execute('INSERT INTO work(kind,key,payload,done) VALUES(?,?,?,?) '
                        'ON CONFLICT(kind,key) DO UPDATE SET payload=excluded.payload,done=excluded.done',
                        (self.kind, key, raw, int(bool(value.get('finished')))))

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def __delitem__(self, key):
        if not self.db.execute('DELETE FROM work WHERE kind=? AND key=?', (self.kind, key)).rowcount:
            raise KeyError(key)

    def __iter__(self):
        for row in self.db.execute('SELECT key FROM work WHERE kind=? ORDER BY key', (self.kind,)):
            yield row[0]

    def __len__(self):
        return self.db.execute('SELECT count(*) FROM work WHERE kind=?', (self.kind,)).fetchone()[0]

    def seen(self, key):
        return key in self or self.db.execute('SELECT 1 FROM completed WHERE kind=? AND key=?',
                                             (self.kind, key)).fetchone() is not None

    def batch(self, after, limit):
        """Round-robin unfinished jobs; a blocked old job cannot starve new alerts."""
        keys = [r[0] for r in self.db.execute('SELECT key FROM work WHERE kind=? AND done=0 AND key>? '
                    'ORDER BY key LIMIT ?', (self.kind, after, limit))]
        keys += [r[0] for r in self.db.execute('SELECT key FROM work WHERE kind=? AND done=0 AND key<=? '
                    'ORDER BY key LIMIT ?', (self.kind, after, limit - len(keys)))]
        for key in keys:
            yield key, self[key]

    def archive(self, key, row):
        # History may contain a duplicate after a crash; the transaction ensures
        # pending work is never lost and dedup never depends on retained segments.
        self.ledger.store.append_rotated('observer-history.jsonl', dict(kind=self.kind, key=key, value=row))
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO completed(kind,key) VALUES(?,?)', (self.kind, key))
            del self[key]
        for prefix, suffix in (('brief-', '.md'), ('analysis-', '.jsonl')):
            if self.kind == 'jobs':
                from .common import digest
                self.ledger.store.path(prefix + digest(key)[:32] + suffix).unlink(missing_ok=True)

    def compact(self, keep, eligible):
        retained = deque()
        for key, row in self.items():
            if eligible(key, row):
                retained.append((key, row))
                if len(retained) > keep:
                    self.archive(*retained.popleft())


class Ledger:
    def __init__(self, store, identity):
        self.store = store
        self.db = sqlite3.connect(store.path('observer-work.sqlite3'))
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('CREATE TABLE IF NOT EXISTS metadata(identity TEXT PRIMARY KEY);'
            'CREATE TABLE IF NOT EXISTS work(kind TEXT,key TEXT,payload TEXT,done INTEGER,PRIMARY KEY(kind,key));'
            'CREATE TABLE IF NOT EXISTS completed(kind TEXT,key TEXT,PRIMARY KEY(kind,key));')
        saved = self.db.execute('SELECT identity FROM metadata').fetchone()
        if saved and saved[0] != identity:
            raise ValueError('observer ledger identity changed; reconcile explicitly')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO metadata VALUES(?)', (identity,))
        self.maps = {kind: WorkMap(self, kind) for kind in ('jobs', 'notices')}

    def flush(self):
        self.db.commit()

    def migrate(self, kind, rows):
        # SQLite is committed BEFORE removing legacy inline JSON. Repeated migration
        # never overwrites newer delivery/claim state if a crash interrupted it.
        mapping = self.maps[kind]
        for key, value in rows.items():
            if not mapping.seen(key):
                mapping[key] = value
        self.flush()
