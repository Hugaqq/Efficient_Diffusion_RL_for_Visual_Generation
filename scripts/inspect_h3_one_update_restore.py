"""Actually restore a completed step1, verify values/state and EXIT, zero updates.
Does not rename/remove SUCCESS or change any checkpoint/owner/metrics bytes.
Layouts: one-update (11/01, max1, one-update-audit.json) and resolution-memory
(scripts/launch_h3_resolution_memory, max2, update-1-audit.json). The receipt must
live outside the run output tree; restore always runs the full content hash.
"""
from pathlib import Path
import argparse,json,subprocess,time
from scripts.h3_one_update_audit import weight_fingerprints
from scripts.launch_h3_one_update import TRAIN_SHA,TEST_SHA
from visual_rl.runtime.h3_lifecycle import H3LifecycleBackend,_phase,_write_json
from visual_rl.runtime.controller import RunController
from visual_rl.algorithms.optimization.h3_world import _exchange
ROOT=Path(__file__).resolve().parents[1]
LAYOUTS={'one-update':dict(max_steps=1,audit='one-update-audit.json'),
         'resolution-memory':dict(max_steps=2,audit='update-1-audit.json')}
class RestoreValidated(RuntimeError):pass

def check_restore_layout(profile,layout,receipt):
    """Pure compile-time gate: completed author-data step1 of the named layout, receipt outside output."""
    launch=profile.launch;t=profile.training;spec=LAYOUTS[layout]
    if (launch.max_optimizer_steps!=spec['max_steps'] or launch.resume_from!=launch.output_dir/'checkpoints/step-1'
            or profile.expected['train_sha256']!=TRAIN_SHA or profile.expected['eval_sha256']!=TEST_SHA
            or t.algorithm!='dual_flash' or t.group_layout!='paired_4x8'):
        raise ValueError(f'read-only restore is restricted to completed {layout} author-data step1')
    if layout=='resolution-memory':
        c=profile.model.config
        if (launch.checkpoint_every_optimizer_steps!=5 or t.num_steps!=10
                or (c.width,c.height,c.frames,c.fps,c.lora_rank,c.lora_alpha)!=(480,320,124,24,32,64)):
            raise ValueError('resolution-memory restore requires 480x320x124@24, first5of10, cadence5')
    if Path(receipt).resolve().is_relative_to(Path(launch.output_dir).resolve()):
        raise ValueError('restore receipt must not be written inside the run output tree')
    return profile

def check_restored_state(*,rank,start_step,trainer_step,adam_steps,saved_row,current,source_state,cursor):
    """Pure per-rank comparison of the restored live state with the accepted step1 evidence."""
    if start_step!=1 or trainer_step!=1:raise ValueError('restored trainer counter is not1')
    if not adam_steps or any(s!=1 for s in adam_steps):raise ValueError('restored Adam counter is not1')
    if saved_row.get('rank',rank)!=rank or saved_row.get('step',1)!=1:raise ValueError('audit row is not this rank step1')
    if saved_row['after']!=current:raise ValueError('restored frozen/LoRA bytes differ from accepted step1')
    if source_state['next_optimizer_step']!=1 or source_state['global_cursor']!=cursor:
        raise ValueError('restored data cursor differs from checkpoint progress')

class InspectBackend(H3LifecycleBackend):
    def __init__(self,receipt,layout='one-update'):
        super().__init__(inspection_only=True);self.receipt=receipt;self.layout=layout;self.timings={}
    def _timed(self,name,method,*args):
        begin=time.monotonic();value=method(*args);self.timings[name]=round(time.monotonic()-begin,1)
        print(json.dumps(dict(restore_stage=name,seconds=self.timings[name])),flush=True);return value
    def compile(self,source):return check_restore_layout(super().compile(source),self.layout,self.receipt)
    def create_runtime(self,*a):return self._timed('create_runtime',super().create_runtime,*a)
    def construct_graph(self,*a):return self._timed('construct_graph',super().construct_graph,*a)
    def prepare(self,*a):return self._timed('prepare_owner_weight_load',super().prepare,*a)
    def restore_prepared(self,*a):return self._timed('restore_prepared_full_content_hash_dcp',super().restore_prepared,*a)
    def bind(self,*a):return self._timed('bind',super().bind,*a)
    def restore_bound(self,*a):return self._timed('restore_bound',super().restore_bound,*a)
    def run(self,bound):
        session=bound.graph.session;launch=bound.graph.profile.launch
        def inspect():
            progress=bound.transaction.manager.inspect_complete(launch.resume_from).progress
            cursor=dict(progress.source_cursors)[bound.graph.source.source_id]
            saved=json.loads((launch.output_dir/LAYOUTS[self.layout]['audit']).read_text())['ranks'][session.rank]
            current=weight_fingerprints(bound.prepared.prepared_root);source=bound.graph.source.state_dict()
            adam=[int(s['step'].item()) for s in bound.prepared.optimizer.state.values()]
            check_restored_state(rank=session.rank,start_step=self.start_step,trainer_step=bound.trainer.next_optimizer_step,
                                 adam_steps=adam,saved_row=saved,current=current,source_state=source,cursor=cursor)
            return dict(rank=session.rank,weights=current,optimizer_step=1,adam_states=len(adam),source=source,
                        checkpoint_cursor=cursor,model_content_sha256=bound.binding.content['model']['content_sha256'],
                        boundary_content_check=bound.binding.boundary_content_check)
        local=_phase(session,'restore-inspection/exact-state',inspect)
        rows=_exchange('restore-inspection/receipts',session.rank,session.process_group,local,True)
        if session.rank==0:
            if self.receipt.exists():raise FileExistsError('refuse replacing restore validation evidence')
            _write_json(self.receipt,dict(status='real_step1_load_pass_no_additional_update',layout=self.layout,
                additional_optimizer_updates=0,rank0_stage_seconds=self.timings,ranks=rows))
        raise RestoreValidated()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--expected-sha',required=True)
    p.add_argument('--receipt',required=True,type=Path);p.add_argument('--layout',choices=tuple(LAYOUTS),default='one-update');a=p.parse_args()
    def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if git('status','--porcelain') or git('rev-parse','HEAD')!=a.expected_sha:raise ValueError('restore requires exact clean source')
    try:RunController(InspectBackend(a.receipt,a.layout)).run(a.config)
    except RestoreValidated:
        print(json.dumps(dict(status='restore_validated_no_additional_update',layout=a.layout,additional_optimizer_updates=0)),flush=True);return
    raise RuntimeError('inspection unexpectedly entered normal finalization')
if __name__=='__main__':main()
