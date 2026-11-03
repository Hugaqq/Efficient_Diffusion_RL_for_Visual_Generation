"""Bounded nvidia-smi samples; sampled maxima are NOT instantaneous peaks.
Starts before any model load. No GPU context, new service, or allocator changes.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import time


def query(fields, kind):
    result = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields,
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, check=True, timeout=8)
    return [list(map(str.strip, line.split(','))) for line in result.stdout.splitlines() if line.strip()]


def number(value):
    try:
        return float(value)
    except ValueError:
        return None


def process(pid, journal_roles):
    root = Path('/proc') / str(pid)
    result = dict(pid=pid, rank=None, local_rank=None, role=journal_roles.get(pid, 'external'))
    try:
        env = dict(entry.split(b'=', 1) for entry in (root/'environ').read_bytes().split(b'\0') if b'=' in entry)
        result.update(rank=env.get(b'RANK', b'').decode(), local_rank=env.get(b'LOCAL_RANK', b'').decode(),
                      cuda_visible_devices=env.get(b'CUDA_VISIBLE_DEVICES', b'').decode(),
                      command=(root/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')[:2048])
        result['cpu_rss_bytes'] = next(int(line.split()[1])*1024 for line in (root/'status').read_text().splitlines() if line.startswith('VmRSS:'))
    except (OSError, StopIteration) as exc:
        result['proc_read_error'] = repr(exc)
    return result


def sample(journal_dir):
    started = time.monotonic()
    cards = query('index,uuid,memory.used,memory.free,memory.total,utilization.gpu', 'gpu')
    apps = query('gpu_uuid,pid,used_memory,process_name', 'compute-apps')
    roles = {}
    for path in journal_dir.glob('*-rank*-pid*.jsonl'):
        roles[int(path.stem.rsplit('pid', 1)[1])] = path.stem.split('-rank', 1)[0]
    pids = {int(row[1]) for row in apps} | set(roles)
    return dict(schema='h3.card-sample.v1', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                monotonic_ns=time.monotonic_ns(), monitor_pid=os.getpid(),
                observation='sequential nvidia-smi queries, sampled not instantaneous peak; memory bytes',
                query_seconds=time.monotonic()-started,
                gpus=[dict(physical_index=int(i), uuid=u, used_bytes=number(used)*1024**2,
                           free_bytes=number(free)*1024**2, total_bytes=number(total)*1024**2,
                           utilization_percent=number(util)) for i,u,used,free,total,util in cards],
                gpu_processes=[dict(uuid=u,pid=int(pid),used_bytes=None if number(mem) is None else number(mem)*1024**2,
                                    name=name) for u,pid,mem,name in apps],
                processes=[process(pid,roles) for pid in sorted(pids)])


def atomic(path, payload):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(payload, stream);stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def run(directory, journal_dir, *, interval=1., max_bytes=512*1024**2, max_seconds=24*3600,
        max_consecutive_failures=3):
    """A failed nvidia-smi sample is degraded, not fatal, until N in a row fail."""
    if type(max_consecutive_failures) is not int or max_consecutive_failures<1:
        raise ValueError('max_consecutive_failures must be an int >= 1')
    directory.mkdir(parents=True, exist_ok=True)
    started=time.monotonic();size=0;count=0;failures=0
    # Budgets travel with every health file so a long-run launcher can check the remaining lifetime.
    budget=dict(interval_seconds=interval,max_seconds=max_seconds,max_bytes=max_bytes,
                started_monotonic_ns=time.monotonic_ns())
    try:
        with (directory/'card-process.jsonl').open('x',buffering=1) as stream:
            def write(row):
                nonlocal size
                raw=json.dumps(row,allow_nan=False)+'\n'
                size+=len(raw.encode())
                if size>max_bytes:raise RuntimeError('monitor disk byte budget exhausted')
                stream.write(raw);stream.flush()
            while not (directory/'STOP').exists():
                tick=time.monotonic()
                if tick-started>max_seconds:raise RuntimeError('monitor duration budget exhausted')
                try:
                    row=sample(journal_dir)
                except Exception as exc:
                    failures+=1
                    if failures>=max_consecutive_failures:raise
                    now=time.monotonic_ns();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    write(dict(schema='h3.card-sample.v1',utc=utc,monotonic_ns=now,monitor_pid=os.getpid(),
                               degraded=True,error=repr(exc),consecutive_failures=failures))
                    atomic(directory/'health.json',dict(healthy=True,degraded=True,pid=os.getpid(),sample_count=count,
                           consecutive_failures=failures,last_error=repr(exc),monotonic_ns=now,utc=utc,bytes=size,**budget))
                else:
                    failures=0
                    write(row);count+=1
                    atomic(directory/'health.json',dict(healthy=True,degraded=False,pid=os.getpid(),sample_count=count,
                           consecutive_failures=0,monotonic_ns=row['monotonic_ns'],utc=row['utc'],bytes=size,**budget))
                time.sleep(max(0.,interval-(time.monotonic()-tick)))
            os.fsync(stream.fileno())
        atomic(directory/'exit.json',dict(exit_code=0,sample_count=count,bytes=size))
    except BaseException as exc:
        atomic(directory/'health.json',dict(healthy=False,pid=os.getpid(),error=repr(exc),sample_count=count))
        atomic(directory/'exit.json',dict(exit_code=1,error=repr(exc),sample_count=count))
        raise


def failure_limit(value):
    if not isinstance(value,str) or not value.isascii() or not value.isdigit() or int(value)<1:
        raise argparse.ArgumentTypeError('expected an integer >= 1')
    return int(value)


def positive_number(value):
    try:
        number=float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('expected a positive finite number') from None
    if not 0<number<float('inf'):
        raise argparse.ArgumentTypeError('expected a positive finite number')
    return number


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--journal-dir',type=Path,required=True)
    parser.add_argument('--max-consecutive-failures',type=failure_limit,default=3)
    # Defaults keep the short-acceptance budgets; a multi-day run must raise them explicitly.
    parser.add_argument('--interval',type=positive_number,default=1.,help='seconds between samples (default 1)')
    parser.add_argument('--max-seconds',type=positive_number,default=24*3600.,
                        help='monitor lifetime before it exits unhealthy (default 86400)')
    parser.add_argument('--max-bytes',type=failure_limit,default=512*1024**2,
                        help='card-process.jsonl byte budget (default 512 MiB)')
    args=parser.parse_args()
    run(args.directory,args.journal_dir,interval=args.interval,max_bytes=args.max_bytes,
        max_seconds=args.max_seconds,max_consecutive_failures=args.max_consecutive_failures)


if __name__=='__main__':main()
