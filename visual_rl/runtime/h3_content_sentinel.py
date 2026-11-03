"""Opt-in frozen-model stat verification, never a replacement content identity.

Snapshots come ONLY from the successful initial full hash's observers. A
same-size/inode rewrite with both mtime and ctime restored can evade this check
(ctime restoration needs root/clock manipulation). No filesystem lock or
adversarial tamper-proof guarantee is provided; terminal/restore hashes remain
mandatory. Directory records also cover empty directories in all-files.v1.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from visual_rl.composition.preflight.artifacts import (
    ArtifactIdentityError, _file_metadata, _lstat, _node_type,
    _require_directory_access, _require_file_access,
)


def boundary_content_check_mode(value):
    if type(value) is not str or value not in ('full', 'sentinel'):
        raise ValueError('boundary_content_check must be full or sentinel')
    return value


def content_check_gate(mode, *, rank, process_group, phase):
    from visual_rl.algorithms.optimization.h3_world import _exchange
    mode = boundary_content_check_mode(mode)
    # The exchange rejects a different label on ANY peer, before file work.
    _exchange(f'h3-content-check/{phase}/{mode}', rank, process_group, None, True)


@dataclass(frozen=True)
class ModelStatSnapshot:
    root: Path
    records: tuple
    file_count: int
    init_content_sha256: str
    snapshot_sha256: str

    @classmethod
    def from_hash(cls, root, identity, files, directories):
        records = tuple(sorted(
            [(r.relative, 'file', r.metadata) for r in files]
            + [(relative, 'tree', metadata) for relative, metadata in directories]))
        if (identity['node_type'] != 'tree' or identity['content_policy'] != 'all-files.v1'
                or len(files) != identity['file_count']
                or sum(r.size for r in files) != identity['byte_count']
                or len({r[0] for r in records}) != len(records)
                or not any(r[:2] == ('.', 'tree') for r in records)):
            raise ValueError('sentinel requires complete initial full model hash observations')
        digest = hashlib.sha256(json.dumps(records, ensure_ascii=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return cls(Path(os.path.abspath(root)), records, len(files), identity['content_sha256'], digest)

    def verify(self):
        """Every rank traverses and lstats all nodes; never opens model bytes.

        Compare before descending as well as after traversal, fail closed on
        races/access errors/symlinks, and never refresh the accepted snapshot.
        """
        expected = {relative: (kind, metadata) for relative, kind, metadata in self.records}
        seen = set()

        def visit(path, relative):
            before = _lstat(path)
            kind = _node_type(path, before)  # rejects symlinks and special nodes
            if expected.get(relative) != (kind, _file_metadata(before)):
                raise ArtifactIdentityError(f'model stat sentinel mismatch: {relative}')
            seen.add(relative)
            if kind == 'tree':
                _require_directory_access(path, before)
                try:
                    children = sorted(path.iterdir(), key=lambda p: p.name)
                except OSError as exc:
                    raise ArtifactIdentityError(f'cannot enumerate model sentinel: {path}') from exc
                for child in children:
                    visit(child, child.name if relative == '.' else f'{relative}/{child.name}')
            else:
                _require_file_access(path, before)
            if _file_metadata(_lstat(path)) != _file_metadata(before):
                raise ArtifactIdentityError(f'model changed during stat sentinel: {relative}')

        visit(self.root, '.')
        if seen != set(expected):
            raise ArtifactIdentityError('model stat sentinel file/directory coverage mismatch')

    def receipt(self, mode):
        return dict(schema='h3.boundary-content-check.v1', mode=boundary_content_check_mode(mode),
                    configured_mode='sentinel', file_count=self.file_count,
                    snapshot_sha256=self.snapshot_sha256, init_content_sha256=self.init_content_sha256)
