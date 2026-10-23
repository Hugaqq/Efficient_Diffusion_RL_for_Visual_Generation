"""Copy committed checkpoints and receipt-verified videos without loading tensors.

This is a file-copy backup, NOT an immutable or security-verified snapshot.
Run from a separate Git checkout; never modify the active training checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def files(root: Path) -> list[Path]:
    paths = sorted(root.rglob('*'))
    if root.is_symlink() or any(p.is_symlink() for p in paths):
        raise ValueError(f'symlink rejected: {root}')
    return [p for p in paths if p.is_file()]


def copy_checkpoint(source: Path, target: Path) -> bool:
    if target.exists():
        if not (target / 'backup-manifest.json').is_file():
            raise ValueError(f'unverified existing destination: {target}')
        return False
    members = files(source)
    if not (source / 'complete.json').is_file():
        return False
    if shutil.disk_usage(target.parent).free < sum(p.stat().st_size for p in members) + 10 * 1024**3:
        raise OSError('backup filesystem requires 10 GiB spare capacity')
    stage = Path(tempfile.mkdtemp(prefix='.pending-' + source.name + '-', dir=target.parent))
    manifest = {}
    for src in members:
        relative = src.relative_to(source)
        dst = stage / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = src.stat()
        expected = sha(src)
        shutil.copyfile(src, dst)
        after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha(dst) != expected or sha(src) != expected:
            raise ValueError(f'source changed or copy mismatch: {src}')
        manifest[str(relative)] = {'bytes': after.st_size, 'sha256': expected}
    if [p.relative_to(source) for p in files(source)] != [p.relative_to(source) for p in members]:
        raise ValueError('checkpoint membership changed during copy')
    (stage / 'backup-manifest.json').write_text(json.dumps({'source': str(source), 'verified_at': time.time(), 'files': manifest}, indent=2))
    os.rename(stage, target)
    return True


def copy_video(receipt: Path, source: Path, destination: Path) -> bool:
    if receipt.is_symlink():
        raise ValueError('symlink receipt rejected')
    raw = receipt.read_bytes()
    data = json.loads(raw)
    video = Path(data['video_path'])
    if not video.is_absolute():
        video = receipt.parent / video
    if video.is_symlink() or not video.resolve().is_relative_to(source):
        raise ValueError('video escapes run directory')
    expected = data['video_sha256']
    if sha(video) != expected:
        raise ValueError(f'video does not match receipt: {video}')
    target = destination / video.relative_to(source)
    target_receipt = destination / receipt.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target_receipt.exists() and sha(target) == expected and target_receipt.read_bytes() == raw:
        return False
    if shutil.disk_usage(target.parent).free < video.stat().st_size + 10 * 1024**3:
        raise OSError('backup filesystem requires 10 GiB spare capacity')
    temporary = target.with_name('.pending-' + target.name)
    shutil.copyfile(video, temporary)
    if sha(temporary) != expected or receipt.read_bytes() != raw:
        raise ValueError('video/receipt changed during copy')
    os.replace(temporary, target)
    temporary_receipt = target_receipt.with_name('.pending-' + target_receipt.name)
    temporary_receipt.write_bytes(raw)
    os.replace(temporary_receipt, target_receipt)
    return True


def sweep(source: Path, destination: Path, minimum: int) -> dict:
    copied, errors = [], []
    (destination / 'checkpoints').mkdir(parents=True, exist_ok=True)
    for checkpoint in sorted((source / 'checkpoints').glob('step-*')):
        try:
            if int(checkpoint.name.removeprefix('step-')) < minimum:
                continue
            if copy_checkpoint(checkpoint, destination / 'checkpoints' / checkpoint.name):
                copied.append(str(checkpoint))
        except Exception as exc:
            errors.append({'path': str(checkpoint), 'error': str(exc)})
    for receipt in sorted((source / 'evaluations').glob('step-*/video-*.json')):
        try:
            if copy_video(receipt, source, destination):
                copied.append(str(receipt))
        except Exception as exc:
            errors.append({'path': str(receipt), 'error': str(exc)})
    return {'time': time.time(), 'source': str(source), 'destination': str(destination), 'copied': copied, 'errors': errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--min-step', type=int, default=26)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if not source.is_dir() or destination.is_relative_to(source) or source.is_relative_to(destination):
        parser.error('source and destination must be separate non-nested directories')
    if args.interval < 10:
        parser.error('interval must be >=10 seconds')
    destination.mkdir(parents=True, exist_ok=True)
    # Prevent duplicate backup owners on the same destination.
    import fcntl
    with (destination / '.backup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = sweep(source, destination, args.min_step)
            print(json.dumps(result), flush=True)
            temporary = destination / '.backup-status.pending.json'
            temporary.write_text(json.dumps(result, indent=2))
            os.replace(temporary, destination / 'backup-status.json')
            if args.once:
                raise SystemExit(1 if result['errors'] else 0)
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
