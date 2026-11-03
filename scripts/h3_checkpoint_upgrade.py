"""Explicit, task-scoped checkpoint compatibility; never rewrites legacy identity.

The current binding observes current code. To read a legacy artifact, derive
its expected contract from live resources/parameters and the actually hashed,
Git-verified old source tree. This is not an ignore-code or trust-disk switch.
"""
from pathlib import Path
import json
import subprocess

from scripts.rewritten_history import H3_RECORDED_SHA, git_revision

BASE_SHA=H3_RECORDED_SHA  # recorded artifact / audit identity
BASE_GIT_SHA=git_revision(BASE_SHA)
OLD_CODE='e0a03bfdbe367e13f5585b597485f2cd2812c3f5e844c066bd21fd10798cdd43'
RUN_ID='671b3af57aeb4e599f48c080ec0a65d0'
SEMANTIC='h3.semantic.v1:96ec05a8e1349b8f642cd9003fa73fe44817f7443e2b39cd61dc3fca577c4d65'
OUTPUT=Path('/home/user/user/hpsv3-h3-20261030-001/flash200-run-002/output')
ALLOWED=frozenset({
 'scripts/launch_h3_media.py','scripts/h3_checkpoint_upgrade.py',
 'tests/test_flash_collection_runtime.py',
 'visual_rl/artifacts/checkpoint/h3_state.py',
 'visual_rl/artifacts/checkpoint/h3_transaction.py',
 'visual_rl/runtime/h3_file_phase.py',
 'visual_rl/runtime/h3_lifecycle.py',
 'visual_rl/runtime/h3_binding.py',
})


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True,timeout=30).strip()


def prove(root,expected_sha):
    root=Path(root)
    if git(root,'status','--porcelain') or git(root,'rev-parse','HEAD')!=expected_sha:
        raise ValueError('upgrade requires exact clean expected Git SHA')
    if git(root,'merge-base',BASE_GIT_SHA,expected_sha)!=BASE_GIT_SHA:
        raise ValueError('upgrade lacks approved base ancestry')
    changed=set(git(root,'diff','--name-only',BASE_GIT_SHA,expected_sha).splitlines())
    if not changed<=ALLOWED:raise ValueError('upgrade outside reviewed file allowlist')
    stable={}
    for path in ('data','experiments'):
        old,new=(git(root,'rev-parse',f'{s}:{path}') for s in (BASE_GIT_SHA,expected_sha))
        if old!=new:raise ValueError('data/experiment changed during checkpoint upgrade')
        stable[path]=old
    return dict(base_sha=BASE_SHA,expected_sha=expected_sha,changed_files=sorted(changed),
                unchanged_trees=stable,legacy_code_hash=OLD_CODE)


class H3CodeUpgrade:
    def __init__(self,*,root,legacy_root,expected_sha):
        self.root=Path(root).resolve();self.legacy_root=Path(legacy_root).resolve()
        self.proof=prove(self.root,expected_sha)
        if git(self.legacy_root,'rev-parse','HEAD') not in (BASE_SHA,BASE_GIT_SHA) or git(self.legacy_root,'status','--porcelain'):
            raise ValueError('legacy source must be the clean actual base checkout')
        from visual_rl.composition.preflight.artifacts import _hash_artifact
        self.legacy_code=_hash_artifact(self.legacy_root/'visual_rl',content_policy='python-code.v1')
        if self.legacy_code['content_sha256']!=OLD_CODE:
            raise ValueError('actual legacy code content does not match pinned checkpoint code')
        self.admitted=False

    def admit_owner(self,*,saved,current,profile,code_root):
        if (Path(code_root).resolve()!=self.root/'visual_rl' or
                profile.launch.output_dir.resolve()!=OUTPUT or
                profile.launch.resume_from is None or
                profile.launch.resume_from.parent.resolve()!=OUTPUT/'checkpoints' or
                profile.launch.max_optimizer_steps!=200 or
                profile.training.algorithm!='dual_flash' or profile.training.group_layout!='paired_4x8'):
            raise ValueError('upgrade is limited to the original Flash200 run/topology')
        expected=dict(current,code=OLD_CODE,run_id=RUN_ID)
        if saved!=expected or current['semantic_id']!=SEMANTIC:
            raise ValueError('legacy owner/semantics/topology differ from approved run')
        audit_dir=OUTPUT/'code-upgrades';audit_dir.mkdir(exist_ok=True)
        audit=audit_dir/(self.proof['expected_sha']+'.json')
        evidence=dict(self.proof,legacy_owner=saved,current_content_owner=current,
                      output=str(OUTPUT),policy='legacy artifact read-only, current code for subsequent saves')
        if audit.exists():
            if json.loads(audit.read_text())!=evidence:raise ValueError('code upgrade audit differs')
        else:
            # Rank0 output-lock already excludes another controller.
            from visual_rl.runtime.h3_lifecycle import _write_json
            _write_json(audit,evidence)
        self.admitted=True

    def restore_contract(self,binding,path):
        # admit_owner runs only on rank0; all peers independently check the
        # pinned run owner and actual code before projecting any old contract.
        saved=json.loads((OUTPUT/'h3-run.json').read_text())
        if (saved.get('run_id')!=RUN_ID or saved.get('code')!=OLD_CODE or
                saved.get('semantic_id')!=SEMANTIC or
                Path(path).parent.resolve()!=OUTPUT/'checkpoints'):
            raise ValueError('restore scope/owner mismatch')
        file=Path(path)/'h3_transaction.json'
        if file.is_symlink() or file.stat().st_size>256*1024:raise ValueError('invalid transaction manifest')
        observed=json.loads(file.read_text())['binding']['live']['code']
        if observed==binding.content['code']['content_sha256']:
            return binding.contract,binding._live_ids()
        if observed!=OLD_CODE:raise ValueError('unknown checkpoint code version')
        live=binding._live_ids(code_content=self.legacy_code)
        return binding._contract(live=live),live
