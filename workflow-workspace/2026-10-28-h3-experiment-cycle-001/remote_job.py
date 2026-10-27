"""Owned remote jobs: bounded execution, durable status, no automatic replay."""
import base64
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path('/home/ubtroot/qifan/h3-experiment-cycle-20261028-001')


def atomic(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def job_path(name):
    if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,95}', name):
        raise ValueError('invalid job id')
    return ROOT / 'runs' / name


def start_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def alive(owner):
    pid = owner['pid']
    return start_ticks(pid) == owner['start_ticks'] and os.getpgid(pid) == pid


def inspect(name):
    path = job_path(name)
    if not path.exists():
        return dict(job_id=name, exists=False)
    spec = json.loads((path/'spec.json').read_text())
    owner = json.loads((path/'owner.json').read_text()) if (path/'owner.json').exists() else None
    running = False
    if owner:
        try: running = alive(owner)
        except ProcessLookupError: pass
    result = json.loads((path/'exit.json').read_text()) if (path/'exit.json').exists() else None
    log = path/'job.log'
    tail = ''
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size-12000)); tail = f.read().decode(errors='replace')
    success = None
    marker = path/'output/SUCCESS'
    if marker.is_file() and marker.stat().st_size < 1024*1024:
        try: success=json.loads(marker.read_text())
        except (ValueError,OSError): pass
    complete = None
    if success is not None:
        relative = Path(success.get('checkpoint_relative_path', ''))
        if not relative.is_absolute() and '..' not in relative.parts:
            receipt = path/'output'/relative/'complete.json'
            if receipt.is_file() and not receipt.is_symlink() and receipt.stat().st_size < 1024*1024:
                complete = json.loads(receipt.read_text())
    heartbeat = json.loads((path/'heartbeat.json').read_text()) if (path/'heartbeat.json').exists() else None
    proof = model = None
    resolved = path/'output/resolved_recipe.json'
    if resolved.is_file() and resolved.stat().st_size < 4*1024*1024:
        data=json.loads(resolved.read_text()); proof=data.get('math',{}); model=data.get('model',{})
    return dict(job_id=name, exists=True, spec=spec, owner=owner, running=running, exit=result,
                success=success, checkpoint_complete=complete, heartbeat=heartbeat, math=proof, model=model,
                output=str(path/'output'), log=str(log),
                log_mtime=log.stat().st_mtime if log.exists() else None, tail=tail)


def launch(spec):
    path=job_path(spec['job_id'])
    sha=spec['sha']
    if not re.fullmatch(r'[0-9a-f]{40}',sha): raise ValueError('full deployed SHA required')
    source=ROOT/'sources'/sha
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=sha:
        raise ValueError('source identity mismatch')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('dirty source; do not repair it remotely')
    if spec.get('kind') not in ('train','prepare','probe','service'):
        raise ValueError('explicit job kind required')
    deadline=spec.get('timeout_seconds')
    if type(deadline) is not int or not 1<=deadline<=172800: raise ValueError('bounded job timeout required (max48h)')
    argv=spec.get('argv')
    if not isinstance(argv,list) or not argv or any(not isinstance(x,str) or '\x00' in x for x in argv):
        raise ValueError('argv list required; no shell launcher')
    python_root='/home/ubtroot/qifan/'
    executable=Path(argv[0])
    if not str(executable).startswith(python_root) or executable.name not in ('python','python3','python3.11','torchrun','pip'):
        raise ValueError('use an explicit dedicated Python environment executable')
    if spec['kind']=='train' and ('--module' not in argv or 'visual_rl.train' not in argv):
        raise ValueError('training must use the actual visual_rl.train module')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.mkdir()  # never overwrite or silently relaunch a job id
    atomic(path/'spec.json',spec)
    if 'config_text' in spec:
        (path/'config.yaml').write_text(spec['config_text'])
    log=(path/'job.log').open('ab',buffering=0)
    worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'execute',spec['job_id']],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    owner=dict(pid=worker.pid,start_ticks=start_ticks(worker.pid),job_id=spec['job_id'],created=time.time())
    atomic(path/'owner.json',owner); log.close()
    return dict(job_id=spec['job_id'],owner=owner,log=str(path/'job.log'),output=str(path/'output'))


def execute(name):
    path=job_path(name);spec=json.loads((path/'spec.json').read_text())
    source=ROOT/'sources'/spec['sha']
    lock=None
    try:
        if spec['kind']!='service':
            lock=(ROOT/'gpu-job.lock').open('a+')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=os.environ.copy()
        supplied=spec.get('env',{})
        allowed={'CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF','OMP_NUM_THREADS',
                 'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TORCH_NCCL_ASYNC_ERROR_HANDLING','WANDB_MODE'}
        if set(supplied)-allowed or any(not isinstance(v,str) for v in supplied.values()):
            raise ValueError('unsupported environment override')
        if spec['kind']=='train' and len(supplied.get('CUDA_VISIBLE_DEVICES','').split(','))!=8:
            raise ValueError('training requires eight explicit UUIDs')
        env.update(supplied,PYTHONPATH=str(source),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        argv=[v.replace('{source}',str(source)).replace('{config}',str(path/'config.yaml'))
              .replace('{output}',str(path/'output')).replace('{job}',str(path)) for v in spec['argv']]
        started=time.time();child=subprocess.Popen(argv,cwd=source,env=env)
        atomic(path/'child.json',dict(pid=child.pid,started=started,argv=argv))
        try:
            deadline=started+spec['timeout_seconds']
            while True:
                remaining=deadline-time.time()
                if remaining<=0:
                    raise subprocess.TimeoutExpired(argv,spec['timeout_seconds'])
                atomic(path/'heartbeat.json',dict(time=time.time(),child_pid=child.pid,elapsed=time.time()-started))
                try:
                    code=child.wait(timeout=min(30,remaining))
                    break
                except subprocess.TimeoutExpired:
                    if time.time()>=deadline: raise
        except subprocess.TimeoutExpired:
            atomic(path/'exit.json',dict(code=124,reason='declared timeout',started=started,ended=time.time()))
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            os.killpg(os.getpgrp(),signal.SIGTERM)
            time.sleep(30)
            os.killpg(os.getpgrp(),signal.SIGKILL)
            return
        atomic(path/'exit.json',dict(code=code,started=started,ended=time.time()))
    except BaseException as exc:
        atomic(path/'exit.json',dict(code=1,error=f'{type(exc).__name__}: {exc}',ended=time.time()))
        raise
    finally:
        if lock: lock.close()


def stop(name):
    path=job_path(name);owner=json.loads((path/'owner.json').read_text())
    if not alive(owner): return dict(job_id=name,stopped=False,reason='not the original live process')
    atomic(path/'stop-request.json',dict(pid=owner['pid'],time=time.time()))
    os.killpg(owner['pid'],signal.SIGTERM)
    return dict(job_id=name,stop_requested=True,pid=owner['pid'])


if __name__=='__main__':
    action=sys.argv[1]
    if action=='execute': execute(sys.argv[2])
    elif action=='launch': print(json.dumps(launch(json.loads(base64.b64decode(sys.argv[2]))),ensure_ascii=False))
    elif action=='inspect': print(json.dumps(inspect(sys.argv[2]),ensure_ascii=False))
    elif action=='stop': print(json.dumps(stop(sys.argv[2]),ensure_ascii=False))
    else: raise SystemExit('unknown operation')
