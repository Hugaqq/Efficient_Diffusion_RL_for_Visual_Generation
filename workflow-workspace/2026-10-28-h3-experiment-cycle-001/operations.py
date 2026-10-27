"""Fixed-host workflow operations with request journals; not a shell sandbox."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

W = Path(__file__).resolve().parent
SOURCE = Path('/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode')
STATE = SOURCE/'.pi/workflow-tmp/h3-experiment-cycle-001'
STATE.mkdir(parents=True,exist_ok=True)
ROOT = '/home/ubtroot/qifan/h3-experiment-cycle-20261028-001'
REPO = '/home/ubtroot/qifan/Efficient_Diffusion_RL_for_Visual_Generation'
BRANCH = 'experiments/h3-cycle-20261028-001'
WORKFLOW = 'wf-2026-10-28-h3-experiment-cycle-001'
REL = 'workflow-workspace/2026-10-28-h3-experiment-cycle-001'
SSH = ['ssh','-i',str(Path.home()/'.ssh/id_ed25519_tailscale_ubtroot_192_168_102_193'),
       '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
       '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','ubtroot@192.168.102.193']
PKG=Path('/Users/qvanium/Documents/Codex/2026-10-19/gei-w/outputs/pi-workflow')
CODE=['visual_rl','scripts','configs','services','pyproject.toml',REL+'/remote_job.py']


def command(argv, timeout=120):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        return dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except subprocess.TimeoutExpired as exc:
        return dict(code=124,stdout=str(exc.stdout or ''),stderr='local timeout; remote effects may be unknown')


def ssh(text, timeout=120):
    return command([*SSH,text],timeout)


def git(*args):
    r=command(['git',*args])
    if r['code']: raise RuntimeError(r['stderr'])
    return r['stdout'].strip()


def remote_python(code):
    return 'python3 -c '+shlex.quote(code)


def worker(sha, action, value):
    if not re.fullmatch('[0-9a-f]{40}',sha): raise ValueError('full SHA required')
    executable=f'{ROOT}/sources/{sha}/{REL}/remote_job.py'
    return ssh('python3 '+shlex.quote(executable)+' '+shlex.quote(action)+' '+shlex.quote(value))


def validate():
    evidence=json.loads((W/'evidence.json').read_text())
    if set(evidence)!={'flow','flash','formal'}: raise ValueError('three real experiment receipts required')
    plan=json.loads((W/'formal-plan.json').read_text())
    target=plan.get('max_optimizer_steps')
    if type(target) is not int or target<1: raise ValueError('formal update target is not user-confirmed')
    for label, record in evidence.items():
        r=worker(record['sha'],'inspect',record['job_id'])
        if r['code']: raise RuntimeError(r)
        live=json.loads(r['stdout'])
        wanted=target if label=='formal' else 5
        if (live.get('running') or not live.get('exit') or live['exit']['code']!=0
                or not live.get('success') or live['success'].get('committed_steps')!=wanted
                or not live.get('checkpoint_complete') or live['checkpoint_complete'].get('step')!=wanted
                or live['spec'].get('kind')!='train'):
            raise ValueError(f'{label}: no real successful {wanted}-update job')
        expected='flow-grpo' if label=='flow' else 'flash-grpo'
        if live['spec'].get('algorithm')!=expected: raise ValueError('algorithm receipt mismatch')
        policy=str((live.get('math') or {}).get('policy_steps','')).lower()
        if label=='flow' and policy!='all_including_terminal_q0': raise ValueError('Flow policy semantics not recorded')
        if label!='flow' and 'single' not in policy: raise ValueError('Flash single-step semantics not recorded')
        if label=='formal':
            for key in ('height','width','frames','fps','lora_rank','lora_alpha'):
                if (live.get('model') or {}).get(key)!=plan[key]: raise ValueError('formal model mismatch: '+key)
            for key in ('global_prompts','samples_per_prompt','global_batch'):
                if (live.get('math') or {}).get(key)!=plan[key]: raise ValueError('formal batch semantics missing/mismatched: '+key)
        if git('diff',record['sha'],'HEAD','--',*CODE):
            raise ValueError(f'{label}: production code changed after GPU validation; revalidate affected tests')
    print(json.dumps({'ok':True,'validation':'remote real-job exits/checkpoints, algorithm semantics and tested code diff'}))


def main():
    if sys.argv[1:] == ['--syntax']:
        count=0
        for name in ('visual_rl','scripts'):
            for path in Path(name).rglob('*.py'):
                ast.parse(path.read_text(),filename=str(path));count+=1
        print(json.dumps({'ok':True,'python_files_parsed':count,'gpu_test':False})); return
    if sys.argv[1:] == ['--validate-evidence']:
        validate();return
    if sys.argv[1:] == ['--wait']:
        state=json.loads((W/'state.json').read_text()) if (W/'state.json').exists() else {}
        jobs=state.get('jobs',[])
        if not jobs:
            time.sleep(60); print('No declared job; wait completed'); return
        job=jobs[-1]
        for _ in range(10):
            result=worker(job['sha'],'inspect',job['job_id'])
            if result['code']:
                print(json.dumps(result,ensure_ascii=False)); return
            live=json.loads(result['stdout'])
            if not live.get('running') or live.get('exit'):
                print(json.dumps(live,ensure_ascii=False)); return
            time.sleep(60)
        print(json.dumps({'job_id':job['job_id'],'still_running':True,'wait_seconds':600})); return
    if sys.argv[1:] == ['--pause']:
        active=json.loads((STATE/'active-run.json').read_text())
        request=STATE/'pause-input.json';request.write_text(json.dumps({'action':'pause','runId':active['runId']}))
        result=command([str(PKG/'node_modules/.bin/tsx'),str(PKG/'src/cli.ts'),str(SOURCE),str(request)])
        if result['code']: raise RuntimeError(result)
        print('Pause requested at step boundary; user input required.');return
    reference_only = sys.argv[1:] == ['--reference']
    if sys.argv[1:] and not reference_only: raise ValueError('unsupported arguments')
    request=json.loads((W/('coder-request.json' if reference_only else 'request.json')).read_text())
    rid=request['id'];action=request['action']
    if reference_only and action!='local_git_read': raise ValueError('Coder reference mode is read-only Git inspection')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}',rid):raise ValueError('unique safe operation id required')
    journal=STATE/'operations';journal.mkdir(exist_ok=True)
    record_file=journal/(rid+'.json')
    digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    if record_file.exists():
        old=json.loads(record_file.read_text())
        if old['digest']!=digest:raise ValueError('operation id reused with different contents')
        print(json.dumps(old,ensure_ascii=False));return
    readonly=action in ('inspect_job','inspect_host','local_git_read') or (action=='exec' and request.get('read_only') is True)
    if not readonly and action!='reconcile':
        for path in journal.glob('*.json'):
            old=json.loads(path.read_text())
            if not old['read_only'] and (old.get('uncertain') or old.get('state')=='started'):
                raise RuntimeError(f'First reconcile unknown operation {old["id"]}; no blind mutation retry')
    record=dict(id=rid,action=action,digest=digest,read_only=readonly,state='started',started=time.time())
    record_file.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    result=None
    try:
        if action=='local_git_read':
            ref=request['ref']
            if not isinstance(ref,str) or not re.fullmatch(r'(?:[0-9a-f]{40}|HEAD|origin/h3-pytorch-training|origin/h3-readable-integration)',ref):
                raise ValueError('only fixed SHA/current or known H3 branches may be inspected')
            path=request.get('path')
            if path is None:
                result=command(['git','ls-tree','-r','--name-only',ref])
            else:
                if not isinstance(path,str) or path.startswith('/') or '..' in Path(path).parts:
                    raise ValueError('repository-relative reference path required')
                result=command(['git','show',ref+':'+path])
        elif action=='inspect_host':
            result=ssh('hostname; nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader; free -h; df -h /home/ubtroot')
        elif action=='bootstrap':
            code=f"from pathlib import Path; import json; p=Path({ROOT!r}); p.mkdir(parents=True,exist_ok=True); f=p/'owner.json'; expected={{'workflow':{WORKFLOW!r}}}; assert not f.exists() or json.loads(f.read_text())==expected; f.write_text(json.dumps(expected))"
            result=ssh(remote_python(code))
        elif action=='exec':
            text=request['command']
            if not isinstance(text,str) or len(text)>24000:raise ValueError('bounded command text required')
            # Scope is enforced by task instructions too; this is not a security sandbox.
            if re.search(r'\b(sudo|pkill|killall|reboot|shutdown)\b|rm\s+-[^\n]*r|git\s+(reset|clean)|push[^\n]*--force|docker\s+(stop|kill)|(^|[;&|])\s*kill\b',text):
                raise ValueError('destructive/global operations forbidden; use stop_job for owned jobs')
            result=ssh('cd '+shlex.quote(ROOT)+' && '+text, min(120,max(1,int(request.get('timeout_seconds',90)))))
        elif action=='publish':
            if git('status','--porcelain','--',*CODE):raise ValueError('production source dirty; return to Coder before publishing')
            sha=git('rev-parse','HEAD')
            result=command(['git','-c','http.version=HTTP/1.1','-c','http.proxy='+os.environ.get('HTTPS_PROXY',''),
                            'push','origin',f'{sha}:refs/heads/{BRANCH}'])
            result['sha']=sha;result['branch']=BRANCH
        elif action=='deploy':
            sha=request['sha']
            if not re.fullmatch('[0-9a-f]{40}',sha):raise ValueError('full source SHA required')
            dest=f'{ROOT}/sources/{sha}'
            code=f'''from pathlib import Path
import subprocess
repo={REPO!r}; dest=Path({dest!r}); sha={sha!r}
def g(*args): return subprocess.check_output(['git','-C',repo,*args],text=True).strip()
assert g('remote','get-url','origin')=='https://github.com/Hugaqq/Efficient_Diffusion_RL_for_Visual_Generation.git'
subprocess.run(['git','-C',repo,'fetch','origin',{BRANCH!r}],check=True)
assert g('rev-parse','FETCH_HEAD')==sha
if not dest.exists():
 dest.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['git','-C',repo,'worktree','add','--detach',str(dest),sha],check=True)
assert subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()==sha
assert not subprocess.check_output(['git','-C',str(dest),'status','--porcelain'],text=True).strip()
print(str(dest),sha)
'''
            result=ssh(remote_python(code))
        elif action=='launch':
            spec=request['spec'];spec=dict(spec)
            if spec.get('kind')=='train':
                import yaml
                stage=spec.get('stage')
                if stage not in ('flow-smoke','flash-smoke','formal'): raise ValueError('explicit training stage required')
                expected_algorithm='flow-grpo' if stage=='flow-smoke' else 'flash-grpo'
                if spec.get('algorithm')!=expected_algorithm: raise ValueError('stage/algorithm mismatch')
                plan=json.loads((W/'formal-plan.json').read_text())
                target=plan.get('max_optimizer_steps') if stage=='formal' else 5
                if type(target) is not int or target<1: raise ValueError('formal update target needs user confirmation')
                config=yaml.safe_load(spec.get('config_text',''))
                if not isinstance(config,dict) or config.get('launch',{}).get('max_optimizer_steps')!=target:
                    raise ValueError('training config must match the authorized update count')
                if stage=='formal':
                    model=config.get('overrides',{}).get('model',{}).get('params',{})
                    for key in ('height','width','frames','fps','lora_rank','lora_alpha'):
                        if model.get(key)!=plan[key]: raise ValueError('formal model config differs from approved plan: '+key)
            if 'config_text' in spec:
                spec['config_text']=spec['config_text'].replace('{output}',ROOT+'/runs/'+spec['job_id']+'/output')
            encoded=base64.b64encode(json.dumps(spec).encode()).decode()
            result=worker(spec['sha'],'launch',encoded)
        elif action in ('inspect_job','stop_job'):
            result=worker(request['sha'],'inspect' if action=='inspect_job' else 'stop',request['job_id'])
        elif action=='reconcile':
            target=journal/(request['operation_id']+'.json');old=json.loads(target.read_text())
            probe=json.loads((journal/(request['probe_id']+'.json')).read_text())
            if not probe['read_only'] or probe.get('state')!='finished' or probe['result']['code']!=0:
                raise ValueError('successful read-only observation required')
            if not isinstance(request.get('evidence'),str) or len(request['evidence'])<30:
                raise ValueError('actual effects/owned PID/output evidence required, not blind retry')
            old.update(uncertain=False,state='reconciled',reconciliation=request['evidence'],probe=request['probe_id'])
            target.write_text(json.dumps(old,ensure_ascii=False,indent=2));result={'code':0,'stdout':'reconciled without replay','stderr':''}
        else:raise ValueError('unknown operation')
        record.update(state='finished',result=result,ended=time.time(),uncertain=not readonly and result['code'] in (124,255))
    except Exception as exc:
        record.update(state='failed',error=f'{type(exc).__name__}: {exc}',ended=time.time(),
                      uncertain=not readonly and result is not None and result.get('code') in (124,255))
        raise
    finally:
        record_file.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    view=dict(record)
    view['result']={k:(v[-22000:] if isinstance(v,str) else v) for k,v in result.items()}
    print(json.dumps(view,ensure_ascii=False))


if __name__=='__main__': main()
