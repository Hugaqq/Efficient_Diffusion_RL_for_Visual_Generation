"""Actually restore a completed step1, verify values/state and EXIT, zero updates.
Does not rename/remove SUCCESS or change any checkpoint/owner/metrics bytes.
"""
from pathlib import Path
import argparse,json,subprocess
from scripts.h3_one_update_audit import weight_fingerprints
from scripts.launch_h3_one_update import TRAIN_SHA,TEST_SHA
from visual_rl.runtime.h3_lifecycle import H3LifecycleBackend,_phase,_write_json
from visual_rl.runtime.controller import RunController
from visual_rl.algorithms.optimization.h3_world import _exchange
ROOT=Path(__file__).resolve().parents[1]
class RestoreValidated(RuntimeError):pass
class InspectBackend(H3LifecycleBackend):
    def __init__(self,receipt):super().__init__(inspection_only=True);self.receipt=receipt
    def compile(self,source):
        profile=super().compile(source);launch=profile.launch
        if (launch.max_optimizer_steps!=1 or launch.resume_from!=launch.output_dir/'checkpoints/step-1'
                or profile.expected['train_sha256']!=TRAIN_SHA or profile.expected['eval_sha256']!=TEST_SHA
                or profile.training.algorithm!='dual_flash' or profile.training.group_layout!='paired_4x8'):
            raise ValueError('read-only restore is restricted to completed one-update author-data step1')
        return profile
    def run(self,bound):
        session=bound.graph.session
        def inspect():
            if self.start_step!=1 or bound.trainer.next_optimizer_step!=1:
                raise ValueError('restored trainer counter is not1')
            if not bound.prepared.optimizer.state or any(int(s['step'].item())!=1 for s in bound.prepared.optimizer.state.values()):
                raise ValueError('restored Adam counter is not1')
            saved=json.loads((bound.graph.profile.launch.output_dir/'one-update-audit.json').read_text())['ranks'][session.rank]
            current=weight_fingerprints(bound.prepared.prepared_root)
            if saved['after']!=current:raise ValueError('restored frozen/LoRA bytes differ from accepted step1')
            return dict(rank=session.rank,weights=current,optimizer_step=1,source=bound.graph.source.state_dict())
        local=_phase(session,'restore-inspection/exact-state',inspect)
        rows=_exchange('restore-inspection/receipts',session.rank,session.process_group,local,True)
        if session.rank==0:
            if self.receipt.exists():raise FileExistsError('refuse replacing restore validation evidence')
            _write_json(self.receipt,dict(status='real_step1_load_pass_no_additional_update',additional_optimizer_updates=0,ranks=rows))
        raise RestoreValidated()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--expected-sha',required=True);p.add_argument('--receipt',required=True,type=Path);a=p.parse_args()
    def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if git('status','--porcelain') or git('rev-parse','HEAD')!=a.expected_sha:raise ValueError('restore requires exact clean source')
    try:RunController(InspectBackend(a.receipt)).run(a.config)
    except RestoreValidated:
        print(json.dumps(dict(status='restore_validated_no_additional_update',additional_optimizer_updates=0)),flush=True);return
    raise RuntimeError('inspection unexpectedly entered normal finalization')
if __name__=='__main__':main()
