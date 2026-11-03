"""Bounded local helper execution. Never a shell; cleanup targets our child only."""
import os
import selectors
import signal
import subprocess
import time


def run_command(argv, *, timeout=30, max_bytes=1024 * 1024, cwd=None):
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=cwd, start_new_session=True)
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ, 'stdout')
            selector.register(child.stderr, selectors.EVENT_READ, 'stderr')
            size = 0
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('helper command deadline exceeded')
                for key, _ in selector.select(min(remaining, 1)):
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    size += len(data)
                    if size > max_bytes:
                        raise ValueError('helper output size limit exceeded')
                    buffers[key.data].extend(data)
            try:
                code = child.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError('helper command deadline exceeded') from exc
        result = {k: bytes(v).decode('utf-8', errors='replace') for k, v in buffers.items()}
        if code:
            raise RuntimeError(f'helper exited {code}: {result["stderr"][-2000:]}')
        return result
    finally:
        if child.poll() is None:
            # All helpers are started in a fresh local session; never a training PID.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)
        child.stdout.close()
        child.stderr.close()
