#!/usr/bin/env python3
"""Bounded capacity preflight using real VisualRL Wan model ports.

NOT a full trainer/G3/reward experiment. The scalar loss is solely a backward
memory probe, not an RL reward. No model/checkpoint is saved or reused in training.
Run each batch in a fresh process; only escalate after inspecting the prior receipt.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,resource,statistics,subprocess,sys,threading,time,traceback
from pathlib import Path

GIB=2**30

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--conditioning-bundle',type=Path,required=True,help='Trusted saved experiment embeddings, same Wan text checkpoint/max length512')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--reserve-gib',type=float,default=16.)
    a=p.parse_args()
    assert 1<=a.batch<=16 and a.repeats>=3 and a.reserve_gib>=12
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    report={'status':'running','batch':a.batch,'geometry':{'height':480,'width':832,'frames':81,'denoising_steps':20,'cfg':4.5,'max_sequence_length':512},'scope':'actual VisualRL model ports capacity preflight only','full_trainer_g3_exercised':False,'reward_evaluated':False,'formal_training_started':False,'probe_loss':'mean square prediction, only for backward/optimizer memory allocation','phases':[]}
    stop=threading.Event();samples=[];start=time.monotonic()
    def save():
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    def sample_gpu():
        while not stop.is_set():
            try:
                text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                row=[float(v.strip()) for v in text.splitlines()[0].split(',')]
                samples.append({'elapsed':time.monotonic()-start,'used_mib':row[0],'free_mib':row[1],'utilization':row[2]})
            except Exception:
                pass
            stop.wait(1)
    monitor=threading.Thread(target=sample_gpu,daemon=True)
    try:
        import torch
        from accelerate import Accelerator
        from diffusers import UniPCMultistepScheduler
        from visual_rl.core.contracts import ComputePrecision,LatentLayout
        from visual_rl.models.catalog import WanConfig
        from visual_rl.models.implementations.wan import WanT2VAdapter,WanConditioning
        from visual_rl.models.interface import ModelInput,ModelLatentSpec
        from visual_rl.models.lifecycle.components import ComponentManager
        assert torch.cuda.device_count()==1,'Expose only the target GPU'
        meminfo={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line and line.split()[1].isdigit()}
        assert meminfo.get('MemAvailable',0)>=48*GIB,'Insufficient available host RAM; do not disturb other jobs'
        free,total=torch.cuda.mem_get_info()
        reserve=max(a.reserve_gib*GIB,.2*total)
        budget=free-reserve-2*GIB  # Additional non-allocator/context allowance.
        assert budget>20*GIB,'Insufficient GPU headroom'
        torch.cuda.set_per_process_memory_fraction(min(.8,budget/total))
        torch.set_num_threads(2);torch.manual_seed(42)
        report.update(initial_free_gib=free/GIB,total_gib=total/GIB,allocator_budget_gib=budget/GIB,reserve_gib=reserve/GIB,versions={'torch':torch.__version__})
        import visual_rl,diffusers,peft
        report['versions'].update(diffusers=diffusers.__version__,peft=peft.__version__,visualrl_file=visual_rl.__file__)
        report['visualrl_adapter_sha256']=hashlib.sha256(Path(sys.modules[WanT2VAdapter.__module__].__file__).read_bytes()).hexdigest()
        report['phase']='load';save();monitor.start()
        targets=('add_k_proj','add_q_proj','add_v_proj','to_add_out','to_k','to_out.0','to_q','to_v')
        config=WanConfig(artifact_ref='main',lora_rank=16,lora_alpha=32,lora_target_modules=targets,gradient_checkpointing=True,guidance_scale=4.5,height=480,width=832,frames=81,max_sequence_length=512,vae_tiling=False)
        adapter=WanT2VAdapter(config,artifact_path=a.model,precision=ComputePrecision.BF16,model_loader=None)
        manager=ComponentManager(adapter,execution_device='cuda',offload_device='cpu')
        components=manager.load()
        def configure(items):
            for value in items.binding('transformer').component.parameters():
                if value.requires_grad:value.data=value.data.float()
        manager.configure(configure)
        params=manager.parameter_state.parameters()
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
        accelerator=Accelerator(mixed_precision='no')
        handle=manager.prepare(accelerator=accelerator,optimizer=optimizer)
        model=components.binding('transformer').component
        decoder=components.binding('decoder').component
        # No forged RuntimeBoundContract: this is the model-port test path, not a production runtime.
        trusted=torch.load(a.conditioning_bundle,map_location='cpu',weights_only=False)
        if isinstance(trusted,dict) and trusted.get('schema')=='wan_capacity_conditioning_v1':
            assert trusted['model']==str(a.model) and trusted['max_sequence_length']==512
            pos,neg,prompt=trusted['positive'],trusted['negative'],trusted['prompt']
        else:
            pos,neg=trusted[0]['embeds'];prompt=trusted[0]['prompt']
        del trusted
        report['conditioning_encode_seconds']=0.
        if pos.shape[1]!=512:
            print('ENCODE_REAL_CONDITIONING_512_ON_CPU',flush=True)
            encoder=components.binding('prompt_encoder').component;t0=time.monotonic()
            with torch.no_grad():pos,neg=encoder.encode((prompt,),512,4.5)
            report['conditioning_encode_seconds']=time.monotonic()-t0
            torch.save({'schema':'wan_capacity_conditioning_v1','model':str(a.model),'max_sequence_length':512,'positive':pos.cpu(),'negative':neg.cpu(),'prompt':prompt},a.output/'conditioning-512.pt')
        assert pos.shape[1]==512 and neg is not None and neg.shape==pos.shape,'Capacity inputs must have real512-length conditioning'
        ids=tuple(f'capacity-row-{i}' for i in range(a.batch))
        conditioning=WanConditioning(pos.expand(a.batch,-1,-1).contiguous(),neg.expand(a.batch,-1,-1).contiguous(),ids)
        report.update(prompt=prompt,conditioning_from=str(a.conditioning_bundle),text_encoder_timing_excluded=True,cold_load_seconds=time.monotonic()-start)
        shape=(a.batch,16,21,60,104)
        spec=ModelLatentSpec(shape=shape,layout=LatentLayout.BCTHW,axis_semantics=('batch','channel','time','height','width'),device=torch.device('cuda:0'),dtype=torch.float32,spatial_stride=(8,8),temporal_stride=4)
        def predict(x,t):
            inp=ModelInput(latents=x.detach(),timestep=t.detach().reshape(1),conditioning=conditioning,guidance=None,latent_spec=spec,condition_identity=ids,guidance_identity=tuple('cfg:4.5' for _ in ids))
            return adapter.predict(inp).value
        def phase(label,fn):
            report['phase']=label;save();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();initial_allocated=torch.cuda.memory_allocated()/GIB;t0=time.monotonic()
            value=fn();torch.cuda.synchronize()
            row={'phase':label,'seconds':time.monotonic()-t0,'initial_allocated_gib':initial_allocated,'peak_allocated_gib':torch.cuda.max_memory_allocated()/GIB,'peak_reserved_gib':torch.cuda.max_memory_reserved()/GIB}
            report['phases'].append(row);save();print(json.dumps(row),flush=True)
            return value
        def backward_probe(x,t):
            model.train();handle.optimizer.zero_grad(set_to_none=True)
            prediction=predict(x,t);loss=prediction.float().square().mean()
            assert torch.isfinite(loss)
            accelerator.backward(loss)
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
            norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
            handle.optimizer.step();handle.optimizer.zero_grad(set_to_none=True)
            return float(norm)
        pre=torch.randn(shape,device='cuda',dtype=torch.float32)
        phase('backward_safety_preflight',lambda:backward_probe(pre,torch.tensor(900.,device='cuda')));del pre
        for repeat in range(a.repeats):
            model.eval();state={}
            def rollout():
                scheduler=UniPCMultistepScheduler.from_pretrained(str(a.model),subfolder='scheduler',local_files_only=True);scheduler.set_timesteps(20,device='cuda')
                x=torch.randn(shape,device='cuda',dtype=torch.float32,generator=torch.Generator(device='cuda').manual_seed(42+repeat))
                with torch.no_grad():
                    for i,t in enumerate(scheduler.timesteps):
                        if i==2:state['replay_x']=x.detach().cpu();state['replay_t']=t.detach().cpu()
                        x=scheduler.step(predict(x,t),t,x,return_dict=False)[0]
                return x.detach()
            final=phase(f'rollout-{repeat}',rollout)
            def decode():
                decoder.to('cuda')
                one_spec=ModelLatentSpec(shape=(1,*shape[1:]),layout=spec.layout,axis_semantics=spec.axis_semantics,device=spec.device,dtype=spec.dtype,spatial_stride=(8,8),temporal_stride=4)
                with torch.no_grad():
                    for i in range(a.batch):
                        media=adapter.decode(final[i:i+1],one_spec);assert media.tensor.shape[0]==1 and media.tensor.shape[1]==81;del media
                decoder.to('cpu')
            phase(f'decode1-{repeat}',decode);del final;gc.collect();torch.cuda.empty_cache()
            replay_x=state['replay_x'].cuda();replay_t=state['replay_t'].cuda();del state
            phase(f'backward-{repeat}',lambda:backward_probe(replay_x,replay_t));del replay_x,replay_t
        roll=[r['seconds'] for r in report['phases'] if r['phase'].startswith('rollout-')]
        decode_times=[r['seconds'] for r in report['phases'] if r['phase'].startswith('decode1-')]
        stable=roll[1:] or roll;stable_decode=decode_times[1:] or decode_times
        report.update(status='capacity_preflight_pass',median_rollout_seconds=statistics.median(stable),median_decode_seconds=statistics.median(stable_decode),rollout_only_videos_per_minute=60*a.batch/statistics.median(stable),rollout_decode_videos_per_minute=60*a.batch/(statistics.median(stable)+statistics.median(stable_decode)),discarded_probe_optimizer_steps=a.repeats+1)
        manager.close();adapter.close();accelerator.free_memory()
    except Exception as e:
        report.update(status='capacity_rejected_or_failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        stop.set()
        if monitor.is_alive():monitor.join(timeout=6)
        (a.output/'gpu_samples.json').write_text(json.dumps(samples))
        report.update(total_seconds=time.monotonic()-start,gpu_sample_count=len(samples),gpu_samples_path=str(a.output/'gpu_samples.json'),host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)/GIB)
        if samples:
            report['observed_min_free_gib']=min(s['free_mib'] for s in samples)/1024
            report['observed_card_peak_used_gib']=max(s['used_mib'] for s in samples)/1024
            if report['status']=='capacity_preflight_pass' and report['observed_min_free_gib']<report['reserve_gib']:
                report['status']='capacity_completed_but_insufficient_headroom'
        save();print(report['status'],flush=True)
    return 0 if report['status']=='capacity_preflight_pass' else 1
if __name__=='__main__':raise SystemExit(main())
