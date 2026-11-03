"""Read-only GPU-side payload executed as python3 -c by the VPS.

No imports of project code, no writes, no process/service control. Paths are the
operator's explicit allowlist. All reads and the number of log tails are bounded.
"""
import base64
import json
from pathlib import Path
import re
import sys
import time


def read_file(path, limit, *, from_end=False):
    with Path(path).open('rb') as stream:
        if from_end:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            raw = stream.read(limit)
        else:
            raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ValueError('snapshot file exceeds limit')
    return raw.decode('utf-8', errors='replace')


def snapshot(request):
    if len(request.get('logs', [])) > 4:
        raise ValueError('at most four explicit log paths')
    status = json.loads(read_file(request['status'], 1024 * 1024))
    events, logs, errors = [], {}, {}
    try:
        for line in read_file(request['events'], 65536, from_end=True).split('\n'):
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    events.append(row)
            except ValueError:
                pass  # first/last line may be incomplete; status retains incident IDs
    except OSError as exc:
        errors['events'] = repr(exc)
    paths = list(request.get('logs', []))
    launch_id = (status.get('process') or {}).get('launch_id', '')
    if request.get('include_training_log', True) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', launch_id):
        current = str(Path(request['status']).parent / ('training-' + launch_id + '.log'))
        if current not in paths:
            if len(paths) >= 4:
                raise ValueError('reserve one log slot for current supervised training')
            paths.append(current)
    for path in paths:
        try:
            logs[path] = read_file(path, 16384, from_end=True)
        except OSError as exc:
            errors[path] = repr(exc)
    return dict(status=status, events=events, logs=logs, errors=errors, remote_time=time.time())


if __name__ == '__main__':
    request = json.loads(base64.b64decode(sys.argv[1], validate=True))
    print(json.dumps(snapshot(request), ensure_ascii=False, allow_nan=False))
