"""Standard-library-only, bounded IO and private state storage."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid


def read_json(path, limit=2 * 1024 * 1024):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'oversize JSON: {path}')
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def tail(path, limit=16384):
    with Path(path).open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read(limit).decode('utf-8', errors='replace')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Store:
    """All writes are direct children of an operator-owned, non-symlink state dir.

    Do not put untrusted writers in this directory. Retain all receipts; never unlink.
    Atomic replacement of our own status files is the only overwrite operation.
    """
    def __init__(self, root):
        self.root = Path(os.path.abspath(root))
        if self.root.resolve() != self.root:
            raise ValueError('state directory must not traverse symlinks')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def path(self, name):
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('state name must be a direct child')
        path = self.root / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError('state file must be regular and not a symlink')
        return path

    def text(self, name, text):
        dest = self.path(name)
        temporary = self.path('.write-' + uuid.uuid4().hex)
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, dest)
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def save(self, name, payload):
        self.text(name, json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')

    def load(self, name, default=None):
        path = self.path(name)
        return read_json(path) if path.exists() else default

    def append(self, name, payload):
        fd = os.open(self.path(name), os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def lock(self, name):
        fd = os.open(self.path(name), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        stream = os.fdopen(fd, 'a+')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            stream.close()
            raise
        return stream


def regular(path):
    path = Path(path)
    return path.resolve() == path and stat.S_ISREG(path.lstat().st_mode)
