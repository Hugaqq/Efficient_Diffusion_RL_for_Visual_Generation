"""480x320 fresh TWO-update acceptance with allocator/card evidence. Never 200.
Both steps save pre-update member0 rollouts and resumable checkpoints, cadence5
unchanged. Observation-only synchronization affects timings, not training math.
"""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import os
import subprocess
import time

from scripts.launch_h3_one_update import TRAIN_SHA, TEST_SHA
from scripts.launch_h3_media import observed_score, save_media
from scripts.h3_one_update_audit import PredictAudit, weight_fingerprints
from visual_rl.runtime.h3_lifecycle import H3LifecycleBackend, _phase, _write_json
from visual_rl.runtime.controller import RunController
from visual_rl.runtime import h3_memory as memory
from visual_rl.algorithms.optimization.h3_world import _exchange
from visual_rl.composition.config.h3_profile import compile_h3_profile

ROOT=Path(__file__).resolve().parents[1]


def validate_fresh(profile):
    t=profile.training;l=profile.launch
    if (l.max_optimizer_steps!=2 or l.resume_from is not None or l.checkpoint_every_optimizer_steps!=5
            or t.num_steps!=10 or t.algorithm!='dual_flash' or t.group_layout!='paired_4x8'
            or profile.expected['train_sha256']!=TRAIN_SHA or profile.expected['eval_sha256']!=TEST_SHA
            or (profile.model.config.width,profile.model.config.height,profile.model.config.frames,profile.model.config.fps,
                profile.model.config.lora_rank,profile.model.config.lora_alpha)!=(480,320,124,24,32,64)):
        raise ValueError('requires fresh author-data4x8 first5of10, 480x320x124@24, max2, cadence5')
    return profile


def monitor_healthy(directory, *, now_ns=None, max_age=15):
    health=json.loads((Path(directory)/'health.json').read_text())
    age=((time.monotonic_ns() if now_ns is None else now_ns)-health.get('monotonic_ns',0))/1e9
    if not health.get('healthy') or not 0<=age<max_age or health.get('sample_count',0)<2:
        raise RuntimeError('card/process monitor unhealthy or stale; cannot start another update')
    # Degraded = transient nvidia-smi failures below the monitor's fatal limit.
    return dict(health,degraded=health.get('degraded') is True,consecutive_failures=health.get('consecutive_failures',0))


def phase(name):
    def decorate(method):
        @functools.wraps(method)
        def wrapped(*args,**kwargs):
            with memory.span(name):return method(*args,**kwargs)
        return wrapped
    return decorate


class TwoUpdateBackend(H3LifecycleBackend):
    """Two-update acceptance backend; subclasses override the documented hooks only."""
    monitor_max_age=15  # seconds; health.json older than this stops the next update
    def __init__(self,*,args):
        super().__init__(reward_pipeline=args.reward_pipeline, reuse_video_vae=args.reuse_video_vae,
                         reuse_rollout_anchor=args.reuse_rollout_anchor,
                         checkpoint_threads=args.checkpoint_threads,
                         boundary_content_check=getattr(args, 'boundary_content_check', 'full'))
        # 4 samples x 10 rollout transitions, plus 4 anchor forwards unless the rollout anchor is reused.
        self.nograd_forwards_per_update=40 if self.reuse_rollout_anchor else 44
        from visual_rl.runtime.h3_rollout_profile import configuration
        configuration(args.profile_rollout)  # strict bool; no CUDA probe when off
        self.profile_rollout=args.profile_rollout;self.rollout_profile=None
        self.args=args;self.previous=None;self.audits=[]
        self.audit_weights=True;self.first_step=None
    def validate_profile(self,profile):return validate_fresh(profile)
    def compile(self,source):return self.validate_profile(super().compile(source))
    def rollout_identity(self,bound,graph,sid):
        """Reward-independent seed/index identity of a live rollout sample."""
        ctx=bound.stages._active;request=ctx.request
        return dict(stochastic_index=ctx.dual_selection[sid],
            initial_noise_sha256=hashlib.sha256(request.initial_states[sid].numpy().tobytes()).hexdigest(),
            model_sha256=graph.profile.expected['model_sha256'])
    def install_media(self,bound,graph,runtime):
        """Hook: member0 export at every update of this two-update acceptance."""
        def exporter(**kwargs):
            sid=kwargs['metadata']['sample_id']
            kwargs['metadata']=dict(kwargs['metadata'],**self.rollout_identity(bound,graph,sid),
                diagnostic_save_exception=True,interpretation='pre-update rollout; not post-update evaluation')
            with memory.span('video_save',sample_id=sid,step=kwargs['metadata']['step']):
                return save_media(**kwargs)
        bound.stages.score_one=observed_score(bound.stages.score_one,source=graph.source,
            root=graph.profile.launch.output_dir/'rollout-videos',ffmpeg=self.args.ffmpeg,
            ffprobe=self.args.ffprobe,launcher_sha=self.args.expected_sha,every=1,terminal_step=2,saver=exporter)
    def check_start(self,step,graph,prepared):
        """Hook: state checks before the first update executed by this process."""
        if step!=0 or prepared.optimizer.state or self.start_step!=0 or graph.source.state_dict()['next_optimizer_step']!=0:
            raise ValueError('optimizer/data not fresh')
    def after_update(self,step,runtime):
        """Hook: collective-safe work after a successful update, before its audit."""
    def update_failed(self,step):
        """Hook: local cleanup when an update raised; the error is re-raised."""
    @phase('runtime_initialization')
    def create_runtime(self,*args):return super().create_runtime(*args)
    @phase('graph_cache_preparation')
    def construct_graph(self,*args):return super().construct_graph(*args)
    @phase('model_initialization_loading')
    def prepare(self,*args):return super().prepare(*args)
    @phase('bind')
    def bind(self,preflight,runtime,graph,prepared,transformed):
        bound=super().bind(preflight,runtime,graph,prepared,transformed)
        from visual_rl.runtime.h3_rollout_profile import configuration, RolloutCellProfile
        config=configuration(self.profile_rollout)
        configs=_exchange('rollout-profile/config',runtime.rank,runtime.process_group,config,True)
        if len(configs)!=8 or any(row!=config for row in configs):
            raise ValueError('rank rollout profile configuration mismatch')
        def attach():
            if self.profile_rollout:
                self.rollout_profile=RolloutCellProfile(output_dir=graph.profile.launch.output_dir,
                    rank=runtime.rank,gpu_uuid=runtime.gpu_uuids[runtime.rank],
                    launcher_sha=self.args.expected_sha,process_group=runtime.process_group)
                bound.stages.rollout_cell_scope=self.rollout_profile.cell
            self.predict_audit=PredictAudit(bound.stages.predict)
            bound.stages.predict=self.predict_audit
            self.install_media(bound,graph,runtime)
            original=bound.trainer.run_iteration
            def iteration(step):
                # All ranks check local telemetry and the independent monitor
                # before either iteration; no silent blind second update.
                def check():
                    memory.journal().assert_healthy();monitor_healthy(self.args.monitor_dir,max_age=self.monitor_max_age)
                    if self.first_step is None:
                        self.check_start(step,graph,prepared)
                        if self.audit_weights:self.previous=weight_fingerprints(prepared.prepared_root)
                        self.first_step=step
                    return True
                _phase(runtime,'memory/before-iteration',check)
                memory.set_context(step=step+1)
                try:
                    result=original(step)
                except BaseException:
                    self.update_failed(step)
                    raise
                self.after_update(step,runtime)
                def audit():
                    # PredictAudit counts forwards of THIS process: a resumed run starts at first_step.
                    done=step+1-self.first_step
                    after=weight_fingerprints(prepared.prepared_root) if self.audit_weights else None;before=self.previous
                    graph_audit=self.predict_audit.payload();norm=float(result.value.payload.grad_norm)
                    if self.audit_weights and before['hashes']['frozen']!=after['hashes']['frozen']:
                        raise ValueError('frozen base changed')
                    if self.audit_weights and before['hashes']['trainable']==after['hashes']['trainable']:
                        raise ValueError('local LoRA shard unchanged')
                    if (len(graph_audit['grad_forward_indices'])!=4*done or graph_audit['nograd_forwards']!=self.nograd_forwards_per_update*done
                            or not graph_audit['final_prediction_released']):
                        raise ValueError('single-cell graph/count/release audit failed')
                    if not prepared.optimizer.state or any(int(s['step'].item())!=step+1 for s in prepared.optimizer.state.values()):
                        raise ValueError('Adam counter differs from actual step')
                    import math
                    if not math.isfinite(norm) or norm<=0:raise ValueError('gradient not finite nonzero')
                    # None = not audited (--audit-weights off), never a claim of change/no change.
                    checked=True if self.audit_weights else None
                    memory.event('update_audit',step=step+1,graph=graph_audit,gradient_norm=norm,
                                 lora_changed=checked,frozen_unchanged=checked)
                    self.previous=after
                    return dict(rank=runtime.rank,step=step+1,before=before,after=after,graph=graph_audit,
                                gradient_norm=norm,lora_changed=checked,frozen_unchanged=checked)
                with memory.span('post_update_validation',step=step+1):
                    local=_phase(runtime,'memory/update-audit',audit)
                    rows=_exchange('memory/update-audit',runtime.rank,runtime.process_group,local,True)
                    self.audits.append(local)
                    _phase(runtime,'memory/persist-update-audit',lambda: _write_json(
                        graph.profile.launch.output_dir/f'update-{step+1}-audit.json',dict(optimizer_updates=step+1,ranks=rows)) if runtime.rank==0 else None)
                return result
            bound.trainer.run_iteration=iteration
        _phase(runtime,'attach-two-update-memory-observer',attach)
        return bound
    @phase('terminal_save_validation')
    def checkpoint(self,bound,result):
        if self.profile_rollout:
            def complete():
                if self.rollout_profile is None:
                    raise RuntimeError('enabled rollout profile was not installed')
                self.rollout_profile.assert_complete()
            _phase(bound.graph.session,'rollout-profile/complete',complete)
        return super().checkpoint(bound,result)


def add_common_arguments(p):
    """Required inputs and default-off performance opt-ins shared by the H3 launchers."""
    p.add_argument('config',type=Path);p.add_argument('--expected-sha',required=True)
    p.add_argument('--ffmpeg',type=Path,required=True);p.add_argument('--ffprobe',type=Path,required=True)
    p.add_argument('--monitor-dir',type=Path,required=True)
    p.add_argument('--reward-pipeline',action='store_true',
                   help='opt-in owner-local pipeline; requires separately approved independent-HPS experiment')
    p.add_argument('--reuse-video-vae',action='store_true',
                   help='opt-in verified owner-local video VAE reuse; requires approved experiment')
    p.add_argument('--reuse-rollout-anchor',action='store_true',
                   help='opt-in: reuse the rollout-time old score instead of a second anchor no-grad forward')
    p.add_argument('--boundary-content-check', choices=('full', 'sentinel'), default='full',
                   help='frozen-base boundary verification (default: full); sentinel uses init-hash stat snapshots '
                        'only before non-terminal saves; terminal/restore still fully hash')
    p.add_argument('--profile-rollout',action='store_true',
                   help='opt-in one-cell CPU/CUDA trace; requires separately approved experiment')
    p.add_argument('--checkpoint-threads',type=int,default=1,
                   help='opt-in DCP writer and rank0 state-tree hashing threads (1 = unchanged serial path); requires approved experiment')
    return p


def main():
    p=add_common_arguments(argparse.ArgumentParser(description=__doc__))
    args=p.parse_args()
    def git(*argv):return subprocess.check_output(['git','-C',str(ROOT),*argv],text=True).strip()
    if git('status','--porcelain') or git('rev-parse','HEAD')!=args.expected_sha:
        raise ValueError('requires exact clean deployed Git SHA')
    validate_fresh(compile_h3_profile(args.config))
    if not memory.enabled() or not args.ffmpeg.is_file() or not args.ffprobe.is_file():
        raise ValueError('requires memory journal and real media tools')
    memory.journal()
    try:
        with memory.span('full_lifecycle'):
            result=RunController(TwoUpdateBackend(args=args)).run(args.config)
        if result.committed_steps!=2:raise RuntimeError('did not stop at exactly two updates')
        print(json.dumps(dict(status='two_update_complete',optimizer_updates=2,checkpoint=str(result.authoritative_checkpoint))),flush=True)
    except BaseException as exc:
        try:memory.journal().exception(exc)
        except BaseException:pass
        raise
    finally:memory.close()


if __name__=='__main__':main()
