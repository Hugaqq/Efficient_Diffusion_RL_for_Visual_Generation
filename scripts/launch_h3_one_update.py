"""Fresh author-data/first-five H3 acceptance. Hard maximum ONE update.

--preview-only generates eight local trajectories for WORLD8 participation,
exports/scores two samples of the first prompt, and performs NO update.
Neither mode starts/resumes a formal200 run or rewrites an old checkpoint.
"""
from pathlib import Path
import argparse,json,subprocess,time,hashlib,resource
import torch
from scripts.launch_h3_media import observed_score,save_media,sha256_file
from scripts.h3_one_update_audit import PredictAudit,weight_fingerprints
from visual_rl.runtime.h3_lifecycle import H3LifecycleBackend,_phase,_write_json
from visual_rl.runtime.controller import RunController
from visual_rl.algorithms.optimization.h3_world import _exchange
from visual_rl.composition.config.h3_profile import compile_h3_profile
ROOT=Path(__file__).resolve().parents[1]
TRAIN_SHA='610e16e6b0e8a30bbb1e75cf87413bf592fe491faa74d918d57d914d865e0016'
TEST_SHA='96536b01ab2d091158118c4fd9f548950abce4fa5fa9a7f9ac2ebaae47776da5'


def validate_fresh(profile):
    if (profile.launch.max_optimizer_steps!=1 or profile.launch.resume_from is not None
            or profile.launch.checkpoint_every_optimizer_steps!=5
            or profile.training.num_steps!=10 or profile.training.algorithm!='dual_flash'
            or profile.training.group_layout!='paired_4x8'
            or profile.expected['train_sha256']!=TRAIN_SHA or profile.expected['eval_sha256']!=TEST_SHA):
        raise ValueError('acceptance requires fresh author-data4x8 first5of10, max1 update, cadence5')
    return profile


class PreviewComplete(RuntimeError):pass


class OneUpdateBackend(H3LifecycleBackend):
    def __init__(self,*,args):
        super().__init__();self.args=args
    def compile(self,source):return validate_fresh(super().compile(source))
    def bind(self,preflight,runtime,graph,prepared,transformed):
        bound=super().bind(preflight,runtime,graph,prepared,transformed)
        def attach():
            self.predict_audit=PredictAudit(bound.stages.predict)
            bound.stages.predict=self.predict_audit
            def exporter(**kwargs):
                ctx=bound.stages._active;sid=kwargs['metadata']['sample_id']
                request=ctx.request
                kwargs['metadata']=dict(kwargs['metadata'],stochastic_index=ctx.dual_selection[sid],
                    initial_noise_sha256=hashlib.sha256(request.initial_states[sid].numpy().tobytes()).hexdigest(),
                    model_sha256=graph.profile.expected['model_sha256'])
                return save_media(**kwargs)
            bound.stages.score_one=observed_score(bound.stages.score_one,source=graph.source,
                root=graph.profile.launch.output_dir/'rollout-videos',ffmpeg=self.args.ffmpeg,
                ffprobe=self.args.ffprobe,launcher_sha=self.args.expected_sha,every=5,terminal_step=1,saver=exporter)
        _phase(runtime,'attach-one-update-observer',attach)
        return bound
    def run(self,bound):
        session=bound.graph.session;started=time.monotonic()
        def before():
            if bound.prepared.optimizer.state or self.start_step!=0 or bound.graph.source.state_dict()['next_optimizer_step']!=0:
                raise ValueError('acceptance requires genuinely fresh optimizer/data state')
            return weight_fingerprints(bound.prepared.prepared_root)
        first=_phase(session,'one-update/initial-weights',before)
        torch.cuda.reset_peak_memory_stats(session.device)
        if self.args.preview_only:return self.preview(bound,first)
        result=super().run(bound)
        last=_phase(session,'one-update/final-weights',lambda:weight_fingerprints(bound.prepared.prepared_root))
        def validate():
            audit=self.predict_audit.payload()
            if len(audit['grad_forward_indices'])!=4 or not audit['final_prediction_released']:
                raise ValueError('expected four sequential single-cell backward forwards per rank')
            if first['hashes']['frozen']!=last['hashes']['frozen']:
                raise ValueError('frozen base changed during single update')
            if not bound.prepared.optimizer.state or any(int(s['step'].item())!=1 for s in bound.prepared.optimizer.state.values()):
                raise ValueError('optimizer counter is not exactly one')
            if bound.trainer.next_optimizer_step!=1:
                raise ValueError('trainer did not stop after exactly one update')
            return dict(rank=session.rank,before=first,after=last,graph=audit,
                lora_changed=first['hashes']['trainable']!=last['hashes']['trainable'],
                seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(session.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(session.device),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        local=_phase(session,'one-update/validate',validate)
        rows=_exchange('one-update/audit',session.rank,session.process_group,local,True)
        if not any(row['lora_changed'] for row in rows):raise ValueError('no LoRA value changed')
        if session.rank==0:_write_json(bound.graph.profile.launch.output_dir/'one-update-audit.json',dict(optimizer_updates=1,ranks=rows))
        return result
    def preview(self,bound,first):
        from visual_rl.algorithms.trainer.h3_dual_rollout import collect
        session=bound.graph.session;value=bound.stages.build(0);ctx=bound.stages._context(value);request=ctx.request
        sid,gid,member=request.samples[0]
        bound.prepared.prepared_root.eval()
        records,terminals,choices=collect(dynamics=ctx.dual,samples=((sid,gid),),
            initial_states={sid:request.initial_states[sid]},generators={sid:request.generators[sid]},
            mode='flash',seed=bound.graph.profile.launch.seed,optimizer_step=0,device=session.device,
            predict=lambda sid,x,i:bound.stages._predict(ctx,sid,x,i),
            read_behavior_id=bound.stages.read_behavior_id,behavior_id=request.behavior_id,
            gate=lambda event,**kw:ctx.coordinator.gate('preview/'+event,**kw))
        def score():
            if session.rank>=2:return None
            with bound.binding.decode(terminals[sid]) as media:
                reward=bound.binding.score_one(sample_id=sid,prompt=request.prompts[sid],media=media)
                folder=save_media(root=bound.graph.profile.launch.output_dir/'preview-videos',
                    metadata=dict(step=1,sample_id=sid,group_id=gid,member=member,prompt=request.prompts[sid],
                                  stochastic_index=choices[sid],initial_noise_sha256=hashlib.sha256(request.initial_states[sid].numpy().tobytes()).hexdigest(),
                                  interpretation='no-update preview; not training or post-update evaluation'),
                    media=media,reward=reward,ffmpeg=self.args.ffmpeg,ffprobe=self.args.ffprobe,launcher_sha=self.args.expected_sha)
            return dict(sample_id=sid,reward=reward,path=str(folder),initial_weights=first)
        local=ctx.coordinator.aux_turns(score)
        rows=_exchange('preview/receipts',session.rank,session.process_group,local,True)
        last=_phase(session,'preview/final-weights',lambda:weight_fingerprints(bound.prepared.prepared_root))
        if first!=last or bound.prepared.optimizer.state:
            raise ValueError('preview changed weights/initialized optimizer')
        if session.rank==0:_write_json(bound.graph.profile.launch.output_dir/'preview.json',dict(optimizer_updates=0,generated_trajectories=8,scored_saved_samples=2,rows=rows))
        raise PreviewComplete('no-update preview finished')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('--expected-sha',required=True)
    p.add_argument('--ffmpeg',type=Path,required=True);p.add_argument('--ffprobe',type=Path,required=True)
    p.add_argument('--preview-only',action='store_true');args=p.parse_args()
    def git(*argv):return subprocess.check_output(['git','-C',str(ROOT),*argv],text=True).strip()
    if git('status','--porcelain') or git('rev-parse','HEAD')!=args.expected_sha:
        raise ValueError('launcher requires exact clean expected Git SHA')
    validate_fresh(compile_h3_profile(args.config))
    if not args.ffmpeg.is_file() or not args.ffprobe.is_file():raise ValueError('media tools absent')
    try:result=RunController(OneUpdateBackend(args=args)).run(args.config)
    except PreviewComplete:
        print(json.dumps(dict(status='preview_complete_no_optimizer_updates',optimizer_updates=0)),flush=True);return
    if result.committed_steps!=1:raise RuntimeError('acceptance exceeded one update')
    print(json.dumps(dict(status='one_update_complete',optimizer_updates=1,checkpoint=str(result.authoritative_checkpoint))),flush=True)


if __name__=='__main__':main()
