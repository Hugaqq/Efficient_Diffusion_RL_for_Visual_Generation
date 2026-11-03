# 已核对的P1恢复点（不是已完成代码）

原attempt在七项CPU命令抄错时被白名单拒绝，命令未执行。主会话随后运行正确原七项：5 passed/2 failed（43.26s），错误为SimpleNamespace缺reward_pipeline及identity必须为IterationIdentity；完整输出附后。

为遵守原子commit，未通过测试的源码没有提交为功能commit，而是逐文件hash备份并保存为recovery-003.patch；生产源文件已恢复到原HEAD315d71a，工具将只保存恢复材料快照。没有丢弃P1工作。

继续本轮时先git status/diff确认补丁是否已应用；未应用则执行允许的git apply --check及git apply恢复一次，已应用禁止重复。修复真实问题，使用固定短入口bash workflow-workspace/2026-11-03-h3-training-speed-001/inputs/run-cpu-gate.sh执行同一七项（完整环境/测试集合不变）。不得接受错误长命令、修改门禁或跳过测试。7/7后按原PLAN完成一个功能+回归原子commit，交Simplifier；后续将此恢复记录标记resolved。

F.....F                                                                  [100%]
=================================== FAILURES ===================================
___ test_real_controller_collects384_then_two192_and_resumes_without_scoring ___

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f1901e8ff50>
tmp_path = PosixPath('/tmp/pytest-of-user/pytest-22/test_real_controller_collects30')
caplog = <_pytest.logging.LogCaptureFixture object at 0x7f1901e8e810>

    def test_real_controller_collects384_then_two192_and_resumes_without_scoring(monkeypatch, tmp_path, caplog):
        from dataclasses import replace
        from visual_rl.artifacts.checkpoint import manager as checkpoint_manager
        from visual_rl.artifacts.checkpoint.h3_state import _save_dcp
        import torch.distributed.checkpoint as dcp
        from pathlib import Path
        import hashlib
    
        # Real tree hashing/verification must never materialize a checkpoint shard.
        root = tmp_path / "bounded-tree"
        root.mkdir()
        payload = b"checkpoint" * 250000
        shard = root / "shard.bin"
        shard.write_bytes(payload)
        with monkeypatch.context() as patch:
            def reject_read_bytes(self):
                raise AssertionError("unbounded checkpoint read")
            patch.setattr(Path, "read_bytes", reject_read_bytes)
            tree = checkpoint_manager._build_state_tree(root)
            assert tree['files'][0]['sha256'] == hashlib.sha256(payload).hexdigest()
            checkpoint_manager._verify_state_tree(root, tree)
            with shard.open('r+b') as stream:
                stream.write(b'X')
            with pytest.raises(ValueError, match='digest mismatch'):
                checkpoint_manager._verify_state_tree(root, tree)
        # Check the production writer configuration without pretending this is GPU DCP.
        with monkeypatch.context() as patch:
            calls = []
            patch.setattr(dcp, 'FileSystemWriter', lambda path, **kw: calls.append(kw))
            patch.setattr(dcp, 'save', lambda *args, **kw: None)
            _save_dcp({}, root / 'dcp', None)
            assert calls == [dict(overwrite=False, sync_files=True,
                                  single_file_per_rank=False, thread_count=1)]
        # New first-five protocol: stateless group selection, no terminal noise,
        # and an explicit one-step window for the old two-transition tiny fixture.
        from visual_rl.algorithms.trainer.h3_dual_rollout import selected_step,collect,SELECTION_VERSION
        from visual_rl.composition.config.h3_profile import H3Training
        from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
        from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
        from visual_rl.models.h3_joint import H3JointLayout
        picks=[selected_step(seed=i,optimizer_step=0,group_id='g',num_steps=10) for i in range(5000)]
        assert set(picks)==set(range(5)) and all(850<picks.count(k)<1150 for k in range(5))
        assert selected_step(seed=42,optimizer_step=1,group_id='g',num_steps=10)==selected_step(seed=42,optimizer_step=1,group_id='g',num_steps=10)
        with pytest.raises(ValueError,match='window'):
            selected_step(seed=42,optimizer_step=0,group_id='g',num_steps=2)
        assert selected_step(seed=42,optimizer_step=0,group_id='g',num_steps=2,window_size=1)==0
        with pytest.raises(ValueError,match='window'): H3Training(algorithm='dual_flash',num_steps=5)
        assert H3Training(algorithm='dual_flash').payload()['selection']==SELECTION_VERSION
        assert 'stochastic_window_size' not in H3Training(algorithm='dual_flow',num_steps=2).payload()
        dynamics=H3DualFlowSDE(H3JointLayout(1,2,1,3),H3ScheduleSnapshot.build(10))
        def rollout(mode):
            def predict(sid,x,index):
                assert not torch.is_grad_enabled()
                return torch.zeros_like(x)
            return collect(dynamics=dynamics,samples=(('s0','group'),('s1','group')),
                initial_states={sid:torch.full((1,5),.25,dtype=torch.float16) for sid in ['s0','s1']},
                generators={sid:(torch.Generator().manual_seed(i*2),torch.Generator().manual_seed(i*2+1)) for i,sid in enumerate(['s0','s1'])},
                mode=mode,seed=42,optimizer_step=0,device='cpu',predict=predict,
                read_behavior_id=lambda:'behavior',behavior_id='behavior',gate=lambda *a,**kw:None)
        records,terminals,choices=rollout('flash')
        assert len(records)==2 and choices['s0']==choices['s1'] and all(0<=r.index<5 for r in records)
        assert all(not x.requires_grad for x in terminals.values())
        assert len(rollout('flow')[0])==20
        from visual_rl.algorithms.optimization.h3_dual_anchor import H3DualReplayAnchor
        keys=tuple((record.sample_id,record.index) for record in records)
        anchor=H3DualReplayAnchor(dynamics=dynamics,behavior_id='behavior',expected_keys=keys,max_records=2)
        anchor.capture(records,device='cpu',predict=lambda sid,x,index:torch.zeros_like(x),
            read_behavior_id=lambda:'behavior',gate=lambda *a,**kw:None)
        for record in records:assert torch.equal(anchor.get(record.sample_id,record.index)[0],record.old_joint)
        bad_records=(replace(records[0],old_joint=records[0].old_joint+1),records[1])
        bad_anchor=H3DualReplayAnchor(dynamics=dynamics,behavior_id='behavior',expected_keys=keys,max_records=2)
        with pytest.raises(ValueError,match='sampled/replayed old score mismatch'):
            bad_anchor.capture(bad_records,device='cpu',predict=lambda sid,x,index:torch.zeros_like(x),
                read_behavior_id=lambda:'behavior',gate=lambda *a,**kw:None)
        from tests.test_h3_dual_rollout import test_stable_group_selection_and_full_vs_single_step
        test_stable_group_selection_and_full_vs_single_step()
        # Independent Gaussian derivative for each stream (not two calls to a
        # shared likelihood helper), followed by element weights and rectification.
        rec=records[0];prediction=torch.full_like(rec.x,.125,dtype=torch.float32,requires_grad=True)
        joint,components,coefs=dynamics.evaluate(rec.x,rec.action,prediction,rec.index)
        means=(torch.tensor(2.),torch.tensor(3.))
        rectified=dynamics.rectified_score(joint,components,coefs,means).sum()
        rectified.backward()
        offset=0;manual_score=0.;manual_grad=[]
        import math
        for width,grid,coef,normalizer in zip([2,3],[dynamics.schedule.video_sigmas,dynamics.schedule.audio_sigmas],coefs,means):
            s=float(grid[rec.index]);u=float(grid[rec.index+1]);maximum=float(grid[1]);dt=u-s
            diffusion=maximum*s;variance=diffusion**2*(-dt)
            velocity=(1+diffusion**2*(1-s)/(2*s))*dt
            x=rec.x.double()[:,offset:offset+width];a=rec.action.double()[:,offset:offset+width]
            mu=x*(1+diffusion**2/(2*s)*dt)+.125*velocity
            coefficient=1/(math.sqrt(-dt)/diffusion+diffusion*math.sqrt(-dt)*(1-s)/(2*s))
            assert abs(float(coef)-coefficient)<1e-4
            expected=(a-mu)/variance*velocity/5*(coefficient/float(normalizer))
            manual_grad.append(expected)
            manual_score+=float((-.5*(a-mu).square()/variance-.5*math.log(variance)-.5*math.log(2*math.pi)).sum())/5
            assert torch.allclose(prediction.grad[:,offset:offset+width],expected.float(),rtol=3e-4,atol=3e-4)
            offset+=width
        from visual_rl.algorithms.optimization.h3_dual_credit import policy_loss,global_stream_means
        assert abs(float(joint.detach())-manual_score)<1e-4
        p2=torch.full_like(rec.x,.125,dtype=torch.float32,requires_grad=True);audit_values=[]
        loss=policy_loss(dynamics=dynamics,record=rec,prediction=p2,old_joint=torch.tensor(manual_score),
            advantage=torch.tensor(.7),mode='flash',world_means=means,clip_range=.005,scale=.25,
            audit=lambda delta,ratio:audit_values.append((delta,ratio)))
        loss.backward()
        assert audit_values[0][0]<1e-4 and abs(audit_values[0][1]-1)<1e-4
        assert torch.allclose(p2.grad,torch.cat(manual_grad,dim=1).float()*(-.7*.25),rtol=4e-4,atol=4e-4)
        with monkeypatch.context() as reduction_patch:
            reduction_patch.setattr(torch.distributed,'all_reduce',lambda value,**kw:value.add_(torch.tensor([525.,1080.,30.,0.],dtype=torch.float64)))
            actual=global_stream_means(((torch.tensor(1.),torch.tensor(3.)),(torch.tensor(2.),torch.tensor(5.))),process_group=None,device='cpu')
            assert [float(v) for v in actual]==[16.5,34.0]
        from visual_rl.algorithms.optimization.h3_world import h3_slot_loss_scale
        scale=h3_slot_loss_scale(n_slot=1,global_count=32,reducer_divisor=8)
        gradients=torch.arange(32,dtype=torch.float64)
        assert torch.equal((gradients*scale).sum()/8,gradients.mean())
        # Fresh single-update launcher denies formal200/resume; output observers
        # use cadence5 plus the explicit terminal1 exception without updating twice.
        from scripts.launch_h3_one_update import validate_fresh,TRAIN_SHA,TEST_SHA
        from scripts.launch_h3_media import selection
        from scripts.h3_one_update_audit import PredictAudit,weight_fingerprints
        from types import SimpleNamespace
        fresh=SimpleNamespace(launch=SimpleNamespace(max_optimizer_steps=1,resume_from=None,checkpoint_every_optimizer_steps=5),
            training=H3Training(group_size=8,algorithm='dual_flash',group_layout='paired_4x8'),expected={'train_sha256':TRAIN_SHA,'eval_sha256':TEST_SHA})
        assert validate_fresh(fresh) is fresh
        # Exercise the real compiler, not only the launcher's already-typed guard.
        from tests.test_h3_mvp_cpu_profile import _fixture as h3_profile_fixture,_save as save_h3_profile
        from visual_rl.composition.config.h3_profile import compile_h3_profile
        profile_path,raw=h3_profile_fixture(tmp_path/'terminal-profile')
        raw['overrides']['training']=dict(group_size=8,num_steps=10,algorithm='dual_flash',group_layout='paired_4x8')
        raw['launch'].update(max_optimizer_steps=1,checkpoint_every_optimizer_steps=5)
        raw['launch']['expected'].update(train_sha256=TRAIN_SHA,eval_sha256=TEST_SHA)
        typed=compile_h3_profile(save_h3_profile(profile_path.parent,raw))
        assert validate_fresh(typed).launch.checkpoint_every_optimizer_steps==5
        # Same seven core tests: extend internally for the authorized TWO-update
        # geometry and lossless peak/exception journal, never a fake GPU claim.
        from scripts.launch_h3_resolution_memory import validate_fresh as validate_two, monitor_healthy
        from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry
        from visual_rl.core.protocols.flash_hpsv3 import VideoContract, aggregate_frame_scores, validate_health, digest
        raw['overrides']['model']=dict(params=dict(width=480,height=320,frames=124,fps=24,lora_rank=32,lora_alpha=64))
        raw['launch']['max_optimizer_steps']=2
        two=compile_h3_profile(save_h3_profile(profile_path.parent,raw))
        assert validate_two(two) is two
        geometry=H3T2VAGeometry(width=480,height=320,num_frames=124,fps=24)
        assert geometry.width==480 and geometry.height==320
        contract=VideoContract(124,320,480)
        assert aggregate_frame_scores(list(range(124)),contract)[1]==tuple(range(123,86,-1))
        for bad_steps in (1,3,200):
            raw['launch']['max_optimizer_steps']=bad_steps
            if bad_steps==3:
                with pytest.raises(ValueError):compile_h3_profile(save_h3_profile(profile_path.parent,raw))
            else:
                with pytest.raises(ValueError,match='max2'):validate_two(compile_h3_profile(save_h3_profile(profile_path.parent,raw)))
        raw['launch']['max_optimizer_steps']=2
        for w,h in ((384,256),(480,256)):
            raw['overrides']['model']['params'].update(width=w,height=h)
            with pytest.raises(ValueError):validate_two(compile_h3_profile(save_h3_profile(profile_path.parent,raw)))
        raw['overrides']['model']['params'].update(width=480,height=320)
        raw['launch']['resume_from']=str(tmp_path/'old-step1')
        with pytest.raises(ValueError,match='max2'):validate_two(compile_h3_profile(save_h3_profile(profile_path.parent,raw)))
        raw['launch']['resume_from']=None
        from visual_rl.runtime.h3_memory import Journal, TorchCounters
        # Actual PyTorch2.8 HPS telemetry exposed UUID text WITHOUT GPU-. Keep
        # strict runtime/NVML identity spelling compatible, before any update.
        with monkeypatch.context() as mp:
            mp.setattr(torch.cuda,'is_initialized',lambda:True)
            mp.setattr(torch.cuda,'memory_allocated',lambda device:10)
            mp.setattr(torch.cuda,'memory_reserved',lambda device:20)
            mp.setattr(torch.cuda,'max_memory_allocated',lambda device:30)
            mp.setattr(torch.cuda,'max_memory_reserved',lambda device:40)
            for observed in ('0e1cd165-126d-e889-2b87-240d8faddddb','GPU-0e1cd165-126d-e889-2b87-240d8faddddb'):
                mp.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(uuid=observed))
                assert TorchCounters(0).snapshot()['device_uuid']=='GPU-0e1cd165-126d-e889-2b87-240d8faddddb'
            mp.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(uuid=None))
            with pytest.raises(RuntimeError,match='UUID'):TorchCounters(0).snapshot()
        class Counters:
            current=10;peak=10;resets=0
            def snapshot(self):
                self.peak=max(self.current,self.peak)
                return dict(allocated=self.current,reserved=self.current+20,max_allocated=self.peak,max_reserved=self.peak+20,device_uuid='GPU-test')
            def reset(self):
                # A durable snapshot must precede EVERY reset, including nesting
                # and exceptions. Read actual disk, not the in-memory history.
                rows=[json.loads(line) for line in journal.path.read_text().splitlines()]
                assert rows[-1]['kind']=='before_unified_reset'
                assert rows[-1]['whole_run_high_water']['allocated']>=self.peak
                self.peak=self.current;self.resets+=1
            def synchronize(self):pass
        counters=Counters();journal=Journal(tmp_path/'memory-events',rank=0,role='test',device=0,counters=counters)
        with journal.span('outer',step=1):
            counters.current=100
            with pytest.raises(RuntimeError,match='simulated allocation failure'):
                with journal.span('inner',sample_id='s',selected_index=2):
                    counters.current=250
                    raise RuntimeError('simulated allocation failure')
            counters.current=10
        journal.assert_healthy();assert journal.hwm==dict(allocated=250,reserved=270)
        journal.exception(RuntimeError('simulated allocation failure'))
        journal.close()
        rows=[json.loads(line) for line in journal.path.read_text().splitlines()]
        assert [row['seq'] for row in rows]==list(range(len(rows)))
        assert max(row['counters']['max_allocated'] for row in rows)==250
        assert rows[-1]['whole_run_high_water']['allocated']==250
        assert any(row['kind']=='phase_exception' and row['context']['selected_index']==2 for row in rows)
        required={'utc','monotonic_ns','pid','rank','role','device','counters','context','reset_id','whole_run_high_water','cpu_rss_bytes'}
        assert all(required<=row.keys() for row in rows)
        assert journal.path.with_suffix('.exception.json').is_file() and counters.resets==4
        counters=Counters();journal=Journal(tmp_path/'memory-budget',rank=0,role='test',device=0,counters=counters)
        journal.limit=journal.bytes
        with pytest.raises(OSError,match='budget'):journal._reset('must-not-reset')
        assert counters.resets==0 and not journal.healthy
        journal.limit=64*1024**2;journal.close()
        monitor=tmp_path/'monitor';monitor.mkdir()
        (monitor/'health.json').write_text(json.dumps(dict(healthy=True,monotonic_ns=1000000000,sample_count=2)))
        monitor_healthy(monitor,now_ns=2000000000)
        with pytest.raises(RuntimeError,match='stale'):monitor_healthy(monitor,now_ns=30000000000)
        from scripts import monitor_h3_memory as card_monitor
        with monkeypatch.context() as mp:
            mp.setattr(card_monitor,'query',lambda fields,kind: [['0','GPU-test','12','52','64','9']] if kind=='gpu' else [['GPU-test',str(os.getpid()),'10','python']])
            row=card_monitor.sample(tmp_path/'memory-events')
            assert row['gpus'][0]['physical_index']==0 and row['gpus'][0]['used_bytes']==12*1024**2
            assert row['gpu_processes'][0]['pid']==os.getpid() and row['processes'][0]['cpu_rss_bytes']>0
            assert row['schema']=='h3.card-sample.v1'
        # Exercise the real new bind/iteration/audit wrapper twice with tiny CPU
        # model/optimizer/world leaves. No fabricated real-GPU memory evidence.
        from scripts import launch_h3_resolution_memory as resolution_entry
        from visual_rl.runtime import h3_memory
        import time
        fake_root=tmp_path/'two-wrapper';fake_root.mkdir()
        model=torch.nn.Module()
        model.register_parameter('lora',torch.nn.Parameter(torch.ones(2)))
        model.register_parameter('frozen',torch.nn.Parameter(torch.ones(2),requires_grad=False))
        optimizer=torch.optim.AdamW([model.lora],lr=1e-4)
        prepared=SimpleNamespace(optimizer=optimizer,prepared_root=model)
        stages=SimpleNamespace(predict=lambda **kw:model.lora*2,score_one=lambda **kw:1.)
        def tiny_iteration(step):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(44):
                with torch.no_grad():stages.predict(index=2)
            for _ in range(4):
                value=stages.predict(index=2);value.sum().backward();del value
            norm=float(model.lora.grad.norm());optimizer.step();optimizer.zero_grad(set_to_none=True)
            return SimpleNamespace(value=SimpleNamespace(payload=SimpleNamespace(grad_norm=norm)))
        bound=SimpleNamespace(stages=stages,trainer=SimpleNamespace(run_iteration=tiny_iteration))
        graph=SimpleNamespace(source=SimpleNamespace(state_dict=lambda:dict(next_optimizer_step=0)),
            profile=SimpleNamespace(launch=SimpleNamespace(output_dir=fake_root)))
        runtime=SimpleNamespace(rank=0,process_group=None)
        counters=Counters();journal=Journal(tmp_path/'wrapper-memory',rank=0,role='test',device=0,counters=counters)
        (monitor/'health.json').write_text(json.dumps(dict(healthy=True,monotonic_ns=time.monotonic_ns(),sample_count=2)))
        with monkeypatch.context() as mp:
            mp.setattr(h3_memory,'journal',lambda:journal)
            mp.setattr(resolution_entry.H3LifecycleBackend,'bind',lambda *a:bound)
            mp.setattr(resolution_entry,'_phase',lambda session,label,work:work())
            mp.setattr(resolution_entry,'_exchange',lambda *a:[a[3]])
>           backend=resolution_entry.TwoUpdateBackend(args=SimpleNamespace(ffmpeg=Path('ffmpeg'),ffprobe=Path('ffprobe'),expected_sha='test',monitor_dir=monitor))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_flash_collection_runtime.py:325: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <scripts.launch_h3_resolution_memory.TwoUpdateBackend object at 0x7f1902c16ad0>
args = namespace(ffmpeg=PosixPath('ffmpeg'), ffprobe=PosixPath('ffprobe'), expected_sha='test', monitor_dir=PosixPath('/tmp/pytest-of-user/pytest-22/test_real_controller_collects30/monitor'))

    def __init__(self,*,args):
>       super().__init__(reward_pipeline=args.reward_pipeline)
                                         ^^^^^^^^^^^^^^^^^^^^
E       AttributeError: 'types.SimpleNamespace' object has no attribute 'reward_pipeline'

scripts/launch_h3_resolution_memory.py:56: AttributeError
----------------------------- Captured stdout call -----------------------------
{"event": "h3_rollout_replay_score_check", "sample_id": "s0", "index": 1, "max_abs_delta": 0.0}
{"event": "h3_rollout_replay_score_check", "sample_id": "s1", "index": 1, "max_abs_delta": 0.0}
_______________________ test_approved140_manifest_counts _______________________

tmp_path = PosixPath('/tmp/pytest-of-user/pytest-22/test_approved140_manifest_coun0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f18ff48fc90>

    def test_approved140_manifest_counts(tmp_path, monkeypatch):
        # Formal H3 WORLD8 four prompt occurrences / eight members. Tiny adapter
        # leaves only; real source, plan validation, credit and restore identities.
        from tests.test_h3_mvp_cpu_source import _fixture, _same_request, TRAIN
        from visual_rl.data.h3_source import H3PromptSource
        from visual_rl.composition.config.h3_profile import H3Training
        from visual_rl.algorithms.optimization import h3_world
        formal = H3Training(group_size=8, algorithm='dual_flash', group_layout='paired_4x8')
        assert formal.local_samples == 4 and formal.global_samples == 32
        assert 'group_layout' not in H3Training().payload()  # default CPS payload unchanged
        for algorithm in ('cps', 'dual_flow'):
            with pytest.raises(ValueError):
                H3Training(algorithm=algorithm, group_layout='paired_4x8')
        with monkeypatch.context() as patch:
            (tmp_path / 'paired').mkdir()
            adapter, manager, _, args = _fixture(tmp_path / 'paired', patch)
            width = args['dynamics'].layout.packed_width
            args.update(group_size=8, group_layout='paired_4x8', max_samples=4,
                        max_cells=12, max_trajectory_bytes=12*(4*width+4)+8*width)
            sources = [H3PromptSource(**(args | dict(rank=r))) for r in range(8)]
            requests = [s.reserve(0) for s in sources]
            tables = tuple(r.samples for r in requests)
            plan = h3_world.H3WorldPlan('c','b',8,3,'g','schedule','score',tables,'paired_4x8')
            assert len(plan.samples) == len({s[0] for s in plan.samples}) == 32
            assert len({s[1] for s in plan.samples}) == 4
            for rank, request in enumerate(requests):
                assert list(request.prompts.values()) == [TRAIN[rank//2]] * 4
                assert tuple(s[2] for s in request.samples) == tuple(range(rank%2*4, rank%2*4+4))
            from dataclasses import replace
            with pytest.raises(ValueError): replace(plan, group_layout='rank_local')
            bad = list(tables); bad[1] = bad[0]
            with pytest.raises(ValueError): replace(plan, samples_by_rank=tuple(bad))
            bad = list(tables); bad[1], bad[3] = bad[3], bad[1]
            with pytest.raises(ValueError): replace(plan, samples_by_rank=tuple(bad))
            values = torch.arange(32, dtype=torch.float64)
            reward_tables = [[plan.plan_id,'c','b',[(sid,gid,m,float(values[i*4+j]),True)
                for j,(sid,gid,m) in enumerate(rows)]] for i,rows in enumerate(tables)]
            patch.setattr(h3_world, '_world_rank', lambda group: 0)
            patch.setattr(h3_world, '_exchange', lambda *a: reward_tables)
            credit = h3_world.gather_h3_credit(plan,process_group=None,collection_id='c',behavior_id='b',rewards=reward_tables[0][3])
            expected = (values.reshape(4,8)-values.reshape(4,8).mean(1,keepdim=True)).flatten()/(values.std(correction=0)+1e-4)
            assert torch.allclose(torch.stack([credit[sid] for sid,_,_ in plan.samples]),expected.float())
            for source, request in zip(sources, requests): source.commit(request)
            state = sources[0].state_dict()
            assert state['global_cursor'] == 4
            # Exercise the actual post-tail checkpoint boundary, not just source
            # cursor serialization. Only trainer/prepared-model leaves are tiny.
            from types import SimpleNamespace
            from visual_rl.algorithms.trainer.h3_stages import H3IterationStages
            def boundary(source, last_collection):
                stages = object.__new__(H3IterationStages)
                stages.data = source
                stages._active = None
                stages._poisoned = False
                stages._last_collection = last_collection
                stages._expected_behavior = source.read_behavior_id()
                parameter = torch.nn.Parameter(torch.ones(1))
                stages.prepared = SimpleNamespace(trainable_named_parameters=lambda: (('tiny',parameter),))
                trainer = SimpleNamespace(_prelude=stages,next_optimizer_step=1)
                assert stages.checkpoint_boundary(trainer) == source.state_dict()
                original = source.state_dict
                with monkeypatch.context() as bp:
                    wrong = original() | dict(global_cursor=8 if original()['global_cursor']==4 else 4)
                    bp.setattr(source,'state_dict',lambda: wrong)
                    with pytest.raises(RuntimeError,match='boundary mismatch'):
                        stages.checkpoint_boundary(trainer)
                parameter.grad = torch.ones_like(parameter)
                with pytest.raises(RuntimeError,match='cleared optimizer gradients'):
                    stages.checkpoint_boundary(trainer)
            for source,request in zip(sources,requests): boundary(source,request.collection_id)
            local = H3PromptSource(**(args | dict(group_layout='rank_local',group_size=2)))
            local_request = local.reserve(0); local.commit(local_request)
            assert local.state_dict()['global_cursor'] == 8
            boundary(local,local_request.collection_id)
            restored = H3PromptSource(**args); restored.load_state_dict(state)
            _same_request(sources[0].reserve(1), restored.reserve(1))
            with pytest.raises(ValueError):
                H3PromptSource(**(args | dict(group_layout='rank_local',max_samples=8,max_cells=24,max_trajectory_bytes=24*(4*width+4)+16*width))).load_state_dict(state)
            manager.close()
        manifest = json.loads((Path(__file__).parents[1] / "experiments/flash_wan_capacity_20261022/workload-140.json").read_text())
        assert manifest["optimizer_commits_target"] == 140
        assert len(manifest["collections"]) == 70
        assert sum(c["training_video_slots"] for c in manifest["collections"]) == 26880
        assert all(c["training_video_slots"] == 2 * c["nominal_rows_per_commit"] == 384 for c in manifest["collections"])
        assert [e["after_optimizer_commits"] for e in manifest["evaluations"]] == [0, 36, 76, 116]
        assert all(e["prompt_slots"] == 300 and e["sampling_steps"] == 50 for e in manifest["evaluations"])
        assert manifest["evaluation_videos_target"] == 1200
        assert not manifest["additional_final_evaluation"]
        from visual_rl.algorithms.trainer.flash_presets import SMALL40, APPROVED140
        from visual_rl.algorithms.trainer.evaluation import load_evaluation_prompts, select_small40_prompts
        assert APPROVED140.optimizer_steps == 140 and APPROVED140.evaluation_steps == (0, 36, 76, 116)
        assert SMALL40.optimizer_steps == 40
        assert SMALL40.optimizer_steps // 2 == 20
        assert SMALL40.optimizer_steps // 2 * SMALL40.collection_rows == 640
        assert SMALL40.collection_rows == 2 * SMALL40.rows_per_commit == 32
        assert SMALL40.evaluation_steps == (0, 20, 40)
        assert SMALL40.prompt_count * len(SMALL40.evaluation_steps) == 48
        assert SMALL40.sampling_steps == 20 and SMALL40.save_videos
        author = _author_data_dir()
        selected = select_small40_prompts(load_evaluation_prompts(author / "test.txt"), author / "train.txt")
        assert len(set(selected)) == 16
        train = set((author / "train.txt").read_text().splitlines())
        assert not set(selected).intersection(train)
        small = json.loads((Path(__file__).parents[1] / "experiments/flash_wan_capacity_20261022/workload-small40.json").read_text())
        assert small["optimizer_commits_target"] == small["collection_count"] * small["optimizer_commits_per_collection"] == 40
        assert small["training_videos_target"] == small["collection_count"] * small["training_videos_per_collection"] == 640
        assert small["training_videos_per_collection"] == small["prompts_per_collection"] * small["group_size"] == 32
        assert small["evaluation_steps"] == [0, 20, 40] and small["terminal_evaluation_required"]
        assert small["evaluation_videos_target"] == small["evaluation_prompts"] * 3 == 48
        assert small["logical_hps_frame_scores_target"] == (640 + 48) * 81
    
        # I1: actual binding/artifact/source/tail checks, with only the existing
        # tiny-model, CUDA/FSDP and HTTP leaves substituted by the shared fixture.
        from tests.test_h3_mvp_cpu_binding import _fixture as binding_fixture
        from tests.test_h3_mvp_cpu_checkpoint_state import _snapshot, _equal
        from visual_rl.algorithms.trainer.h3_stages import H3IterationStages
        from visual_rl.algorithms.trainer.interface import PrepareRunContext
        from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot
        from types import SimpleNamespace
        with monkeypatch.context() as patch:
            assets_path = tmp_path / 'checkpoint-snapshot'
            assets_path.mkdir()
            # Shared fixture predates the file-IO heartbeat transport. Keep real
            # file work, replacing only CPU-test broadcast/CUDA wait leaves.
            from visual_rl.runtime import h3_file_phase
            from visual_rl.models.implementations import h3_loading
            def admit_tiny_cpu_weights(budget):
                assert 0 < budget < 10**7  # no production 24GiB CUDA-host reserve here
            patch.setattr(h3_loading, '_admit_host_memory', admit_tiny_cpu_weights)
            patch.setattr(torch.distributed, 'broadcast', lambda *args, **kwargs: None)
            patch.setattr(h3_file_phase, 'file_phase', lambda action, **kwargs: action())
            h = binding_fixture(assets_path, patch)
            from tests.test_h3_mvp_cpu_stages import _CoordinationLeaf
            from visual_rl.models.lifecycle.h3_coordination import H3Coordination
            patch.setattr(H3Coordination, 'aux_turns', _CoordinationLeaf.aux_turns)
            from tests.test_h3_mvp_cpu_decode import _decoder
            h.adapter._aux._decoder = _decoder(h.profile.launch.model, h.adapter.geometry)
            b = h.binding
            stages = H3IterationStages(data=h.source, prepared=h.prepared, predict=b.predict,
                decode=b.decode, score_one=b.score_one, coordination_factory=b.coordination_factory,
                process_group=h.pg, rank=0, device='cpu', read_behavior_id=h.source.read_behavior_id)
            trainer = stages.trainer()
            trainer.prepare_run(PrepareRunContext('snapshot-test', 'h3', 0))
            validations = []
            real_validate = b.validate_live
            def validate():
                live = real_validate()
                validations.append(live)
                return live
            patch.setattr(b, 'validate_live', validate)
            try:
                with pytest.raises(RuntimeError, match='tail'):
                    b.checkpoint_snapshot(stages, trainer)
                with pytest.raises(ValueError, match='foreign'):
                    b.checkpoint_snapshot(SimpleNamespace(data=object(), prepared=h.prepared), trainer)
                assert validations == []
                for step in range(2):
                    trainer.run_iteration(step)
                    before = _snapshot(h.prepared)
                    source_before = h.source.state_dict()
                    rng_before = RankRNGSnapshot.capture_current(0)
                    count = len(validations)
                    boundary, live = b.checkpoint_snapshot(stages, trainer)
                    assert len(validations) == count + 1 and live is validations[-1]
                    assert boundary[0].next_optimizer_step == step + 1
                    assert boundary[2]['payload'] == source_before
                    assert b.checkpoint_boundary(stages, trainer) == boundary
                    assert len(validations) == count + 2  # legacy API is self-validating
                    again, again_live = b.checkpoint_snapshot(stages, trainer)
                    assert again == boundary and again_live == live
                    assert len(validations) == count + 3  # no same-boundary memoization
                    assert RankRNGSnapshot.capture_current(0) == rng_before == boundary[3]
                    _equal(_snapshot(h.prepared), before)
                    assert h.source.state_dict() == source_before
                p = next(iter(h.prepared.trainable_named_parameters()))[1]
                p.grad = torch.zeros_like(p)
                with pytest.raises(RuntimeError, match='gradients'):
                    b.checkpoint_snapshot(stages, trainer)
                p.grad = None
                with patch.context() as fault:
                    fault.setattr(h.prepared.scheduler, 'last_epoch', 1)
                    with pytest.raises(ValueError, match='scheduler'):
                        b.checkpoint_snapshot(stages, trainer)
                with patch.context() as fault:
                    saved = h.prepared.optimizer.state[p]
                    fault.setitem(saved, 'step', torch.tensor(3.))
                    with pytest.raises(ValueError, match='scheduler|Adam'):
                        b.checkpoint_snapshot(stages, trainer)
                artifact = h.profile.launch.model / 'vae/diffusion_pytorch_model.safetensors'
                original_bytes = artifact.read_bytes()
                try:
                    artifact.write_bytes(original_bytes + b' ')
                    with pytest.raises(ValueError, match='model content'):
                        b.checkpoint_snapshot(stages, trainer)
                finally:
                    artifact.write_bytes(original_bytes)
                b.checkpoint_snapshot(stages, trainer)  # no stale failed receipt/cache
                # Modify live source only AFTER a successful real scan. Read-only
                # tail recheck must reject both advancement and a new reservation.
                for mode in ('advance', 'reservation'):
                    with patch.context() as fault:
                        def drift():
                            result = validate()
                            if mode == 'advance':
                                fault.setattr(h.source, '_next_step', 3)
                            else:
                                h.source.reserve(2)  # real new reservation; deliberately not rolled back
                            return result
                        fault.setattr(b, 'validate_live', drift)
                        with pytest.raises(RuntimeError, match='boundary'):
                            b.checkpoint_snapshot(stages, trainer)
                    if mode == 'advance':
                        assert h.source.state_dict() == source_before
                    else:
                        assert h.source._pending is not None
                        h.source.abort(h.source._pending)  # test teardown, leaves source poisoned
            finally:
                trainer.close(); b.close(); h.scorer.close(); h.adapter.close(); h.manager.close()
    
        # Exercise the real lifecycle save branch and _phase error propagation.
        # Trainer/transaction are CPU leaves; no publication or CUDA execution.
        from visual_rl.runtime import h3_lifecycle as lifecycle
        from visual_rl.algorithms.optimization import h3_world
        with monkeypatch.context() as patch:
            exchanges, snapshots, saves = [], [], []
            def exchange(label, rank, group, payload, ok):
                exchanges.append((label, ok))
                if not ok:
                    raise ValueError('peer phase rejected')
                if label == 'lifecycle/pause-request':
                    return [False] + [None] * 7
                return [payload] * 8
            patch.setattr(h3_world, '_exchange', exchange)
            output = tmp_path / 'lifecycle-snapshot'
            output.mkdir()
            fake_trainer = SimpleNamespace(next_optimizer_step=0)
            def iteration(step):
                fake_trainer.next_optimizer_step = step + 1
                return SimpleNamespace(value=SimpleNamespace(payload=SimpleNamespace(
                    scaled_loss=1., local_cells=4, grad_norm=.5)))
            fake_trainer.run_iteration = iteration
            fake_binding = SimpleNamespace(contract=object(), state_identity=object())
            fake_stages, prepared = object(), object()
            def forbidden():
                pytest.fail('lifecycle must not call validate_live after snapshot')
            fake_binding.validate_live = forbidden
            def snapshot(stages, trainer):
                assert stages is fake_stages and trainer is fake_trainer
                step = trainer.next_optimizer_step
                boundary = (object(), object(), {'step': step}, object())
                live = {'step': step}
                snapshots.append((boundary, live))
                return boundary, live
            fake_binding.checkpoint_snapshot = snapshot
            def save(actual_prepared, **kwargs):
                boundary, live = snapshots[-1]
                assert actual_prepared is prepared
                assert kwargs['bindings'] is live
                assert kwargs['progress'] is boundary[0] and kwargs['safe_point'] is boundary[1]
                assert kwargs['data_state'] is boundary[2]
                assert kwargs['contract'] is fake_binding.contract
                assert kwargs['identity'] is fake_binding.state_identity
                saves.append(kwargs)
                return SimpleNamespace(step=live['step'])
            bound = SimpleNamespace(trainer=fake_trainer, stages=fake_stages, prepared=prepared,
                binding=fake_binding, transaction=SimpleNamespace(checkpoint=save),
                graph=SimpleNamespace(session=SimpleNamespace(rank=0, process_group=object()),
                    profile=SimpleNamespace(training=SimpleNamespace(global_samples=32, group_layout='paired_4x8'),
                        launch=SimpleNamespace(output_dir=output, max_optimizer_steps=2,
                                               checkpoint_every_optimizer_steps=1))))
            backend = lifecycle.H3LifecycleBackend()
            committed, _ = backend.run(bound)
            assert committed.step == 2 and len(snapshots) == len(saves) == 2
            assert [live['step'] for _, live in snapshots] == [1, 2]
            assert exchanges.count(('lifecycle/checkpoint-boundary', True)) == 2
            def fail_snapshot(*args):
                raise RuntimeError('snapshot rejected')
            fake_binding.checkpoint_snapshot = fail_snapshot
            with pytest.raises(RuntimeError, match='snapshot rejected'):
                backend.run(bound)
            assert len(saves) == 2 and exchanges[-1] == ('lifecycle/checkpoint-boundary', False)
    
        # P1 ownership: real tiny CPU decoder, original lease locking and storage.
        from tests.test_h3_mvp_cpu_decode import _artifacts, _leaf_setup, _decoder, _terminal
        from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry
        from visual_rl.models.implementations.h3_decode import H3AVMedia, H3OwnedAVMedia
        from contextlib import contextmanager
        from threading import Event, Thread
        import threading
        from visual_rl.algorithms.trainer import h3_reward_pipeline as pipeline
        from visual_rl.runtime import h3_memory as memory
        with monkeypatch.context() as patch:
            root = tmp_path / 'pipeline-decoder'; root.mkdir()
            geometry = H3T2VAGeometry(32, 32, 124)
            _artifacts(root); _leaf_setup(patch, geometry)
            decoder = _decoder(root, geometry)
            terminal = _terminal(geometry)
            before_terminal = terminal.clone()
            with decoder.decode_one(terminal, geometry) as leased:
                video_ptr, audio_ptr, metadata = leased.video.data_ptr(), leased.audio.data_ptr(), leased.metadata
                with pytest.raises(RuntimeError, match='one media lease'):
                    with decoder.decode_one(terminal, geometry):
                        pytest.fail('normal lease lock was weakened')
                owned = leased.handoff()
                assert owned.video.data_ptr() == video_ptr and owned.audio.data_ptr() == audio_ptr
                assert owned.metadata == metadata
                with pytest.raises(RuntimeError, match='expired'):
                    leased.handoff()
            with decoder.decode_one(terminal, geometry) as next_media:
                assert next_media.video.shape == owned.video.shape  # previous owner still alive
            with pytest.raises(RuntimeError, match='expired'):
                _ = leased.video
            with pytest.raises(RuntimeError, match='again'):
                owned.handoff()
            video_ref, audio_ref = weakref.ref(owned.video), weakref.ref(owned.audio)
            with pytest.raises(RuntimeError, match='consumer exception'):
                with owned:
                    raise RuntimeError('consumer exception')
            owned.close()
            assert video_ref() is None and audio_ref() is None
            with pytest.raises(RuntimeError, match='expired'):
                _ = owned.audio
            assert torch.equal(terminal, before_terminal)
    
        samples = tuple(pipeline.RewardSample(i, f'pipe-{i}', 'g', i, f'prompt {i}', i) for i in range(4))
        # Deterministic capacity/overlap and cancellation: score0 blocks, decode1
        # completes, packet1 queues, producer attempts permit3 but cannot decode2.
        from visual_rl.models.implementations import h3_loading
        for failure in (None, 'score', 'nonfinite', 'saver', 'decode', 'transfer', 'cleanup'):
            with monkeypatch.context() as patch:
                budgets = []
                patch.setattr(h3_loading, '_admit_host_memory', lambda n: budgets.append(n))
                decoded, scored, tensor_refs, owners = [], [], [], []
                score_started, second_queued, third_permit, release = Event(), Event(), Event(), Event()
                real_queue, real_semaphore = pipeline.Queue, pipeline.Semaphore
                class QueueSpy(real_queue):
                    def put(self, packet, **kwargs):
                        assert self.maxsize == 1
                        super().put(packet, **kwargs)
                        if packet[0].ordinal == 1: second_queued.set()
                class Permits:
                    def __init__(self, count):
                        assert count == 2
                        self.semaphore = real_semaphore(count)
                        self.attempts = 0
                    def acquire(self, **kwargs):
                        self.attempts += 1
                        if self.attempts >= 3: third_permit.set()
                        return self.semaphore.acquire(**kwargs)
                    def release(self): self.semaphore.release()
                patch.setattr(pipeline, 'Queue', QueueSpy)
                patch.setattr(pipeline, 'Semaphore', Permits)
                real_handoff = H3AVMedia.handoff
                def handoff(media):
                    if failure == 'transfer' and len(decoded) == 2:
                        raise RuntimeError('transfer failed')
                    result = real_handoff(media)
                    owners.append(weakref.ref(result))
                    if failure == 'cleanup':
                        def close():
                            result._release()
                            raise RuntimeError('cleanup failed')
                        result.close = close
                    return result
                patch.setattr(H3AVMedia, 'handoff', handoff)
                @contextmanager
                def decode(sample):
                    decoded.append(sample.ordinal)
                    if sample.ordinal == 1:
                        assert score_started.wait(5), 'first score never started'
                        if failure == 'decode': raise RuntimeError('decode failed')
                    video, audio = torch.zeros(2, 3, 2, 2), torch.zeros(4, 2)
                    tensor_refs.extend((weakref.ref(video), weakref.ref(audio)))
                    media = H3AVMedia(video, audio, 24, 32000)
                    del video, audio
                    try: yield media
                    finally: media._release()
                def score(sample, media):
                    scored.append(sample.ordinal)
                    assert media.video.device.type == 'cpu'
                    if sample.ordinal == 0:
                        score_started.set()
                        assert release.wait(5), 'test did not release score'
                        if failure in ('score', 'saver'): raise RuntimeError(f'{failure} failed')
                        if failure == 'nonfinite': return float('nan')
                    return float(sample.ordinal)
                result, errors = [], []
                def run_pipeline():
                    try:
                        result.append(pipeline.reward_pipeline(samples, decode=decode, score=score,
                                                              media_bytes=1024, decode_bytes=2048))
                    except BaseException as exc: errors.append(exc)
                runner = Thread(target=run_pipeline, name='cpu-pipeline-test')
                runner.start()
                try:
                    assert score_started.wait(5)
                    if failure not in ('decode', 'transfer'):
                        assert second_queued.wait(5)
                        assert third_permit.wait(5)
                        assert decoded == [0, 1] and scored == [0]
                    # Even on decode/transfer failure, join must wait for the
                    # already-running operation, not close its active owner.
                    assert runner.is_alive()
                finally:
                    release.set(); runner.join(10)
                assert not runner.is_alive()
                assert not any(t.name == 'h3-reward-consumer' for t in threading.enumerate())
                assert budgets == [3072]
                if failure is None:
                    assert not errors and decoded == scored == list(range(4))
                    assert result[0] == tuple((s.sample_id, s.group_id, s.member, float(s.ordinal), True) for s in samples)
                else:
                    assert not result and len(errors) == 1
                    assert failure in str(errors[0]) or (failure == 'nonfinite' and 'finite' in str(errors[0]))
                    # No new POST once the blocked first operation fails.
                    if failure in ('score', 'nonfinite', 'saver', 'cleanup'): assert scored == [0]
                assert all(ref() is None for ref in tensor_refs)
                assert all(ref() is None or ref()._video is None for ref in owners)
    
        # Real dual reward -> original H3HPSReward/JPEG/validation/observed saver.
        # No helper/reward mock: only decode model/HTTP/coordination leaves.
        from tests.test_h3_mvp_cpu_hps import _scorer, _media
        from visual_rl.algorithms.trainer.h3_dual_stages import H3DualIterationStages
        from visual_rl.algorithms.trainer.interface import StageValue
        from visual_rl.core.protocols.flash_hpsv3 import VideoContract
        from scripts.launch_h3_media import observed_score
        from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot
        with monkeypatch.context() as patch:
            patch.setattr(h3_loading, '_admit_host_memory', lambda n: None)
            contract = VideoContract(124, 32, 32)
            scorer, transport, _ = _scorer(tmp_path / 'pipeline-hps', patch, contract)
            scorer.acquire()
            source = SimpleNamespace()
            request = SimpleNamespace(identity=SimpleNamespace(optimizer_step=0), group_layout='paired_4x8',
                samples=tuple((s.sample_id, s.group_id, s.member) for s in samples),
                prompts={s.sample_id: s.prompt for s in samples}, collection_id='c', behavior_id='b')
            source._pending = request
            tensors = []
            for i in range(4):
                item = _media(contract, offset=i*.01)
                tensors.append((item.video, item.audio)); item._release()
            terminals = {s.sample_id: torch.full((1, 2), float(s.ordinal), dtype=torch.float16) for s in samples}
            original_terminals = {sid: t.clone() for sid, t in terminals.items()}
            context = SimpleNamespace(request=request, dual_terminals=terminals,
                dual_selection={s.sample_id: s.selected_index for s in samples})
            gates, aux, exported = [], [], []
            class Coordinator:
                mismatch = False
                def gate(self, event, **kwargs):
                    gates.append(event)
                    if self.mismatch and event.startswith('dual/reward-pipeline/'):
                        raise ValueError('configuration mismatch')
                    if kwargs.get('local_error'): raise ValueError('local pipeline admission rejected')
                def aux_turns(self, work):
                    aux.append(True)
                    return work()
            context.coordinator = Coordinator()
            stage = object.__new__(H3DualIterationStages)
            stage._active = context; stage.rank = 0; stage.mode = 'dual_flash'
            stage.media_bytes = 10**7; stage.decode_bytes = 2*10**7
            stage.read_behavior_id = lambda: 'b'
            @contextmanager
            def decode(terminal):
                video, audio = tensors[int(terminal[0, 0])]
                media = H3AVMedia(video, audio, 24, 32000)
                try: yield media
                finally: media._release()
            def saver(**kwargs):
                # Export sees original active storage, not an MP4 readback/copy.
                sid = kwargs['metadata']['sample_id']; i = int(sid.split('-')[1])
                assert kwargs['media'].video.data_ptr() == tensors[i][0].data_ptr()
                assert kwargs['media'].audio.data_ptr() == tensors[i][1].data_ptr()
                with memory.span('video_save', sample_id=sid): exported.append(sid)
            stage.decode = decode
            stage.score_one = observed_score(scorer.score_one, source=source, root=tmp_path / 'view',
                ffmpeg='unused', ffprobe='unused', launcher_sha='cpu', every=1, saver=saver)
            before_rng = RankRNGSnapshot.capture_current(0)
            stage.reward_pipeline = False
>           value = StageValue(request.identity, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_flash_collection_runtime.py:1915: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
<string>:5: in __init__
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = StageValue(identity=namespace(optimizer_step=0), payload=namespace(request=namespace(identity=namespace(optimizer_step...=<tests.test_flash_collection_runtime.test_approved140_manifest_counts.<locals>.Coordinator object at 0x7f18f5547d10>))

    def __post_init__(self) -> None:
        if not isinstance(self.identity, IterationIdentity):
>           raise TypeError("identity must be an IterationIdentity")
E           TypeError: identity must be an IterationIdentity

visual_rl/algorithms/trainer/interface.py:91: TypeError
----------------------------- Captured stdout call -----------------------------
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0008109889458864927, "largest_embedding_bytes": 61440}
{"event": "h3_owner_load_begin", "rank": 0, "timeout_seconds": 3600}
{"event": "h3_owner_load_wait", "seconds": 0.0006380819249898195, "done": false}
{"event": "h3_owner_load_end", "seconds": 1.0012208019616082, "ok": true, "error": null}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0008280909387394786, "largest_embedding_bytes": 61440}
{"event": "h3_vae_load_begin", "component": "vae"}
{"event": "h3_vae_load_end", "component": "vae", "seconds": 0.0003364639123901725}
{"event": "h3_vae_decode_begin", "component": "vae"}
{"event": "h3_vae_decode_exit", "component": "vae", "seconds": 0.0026206490583717823}
{"event": "h3_vae_load_begin", "component": "audio_vae"}
{"event": "h3_vae_load_end", "component": "audio_vae", "seconds": 0.00030921201687306166}
{"event": "h3_vae_decode_begin", "component": "audio_vae"}
{"event": "h3_vae_decode_exit", "component": "audio_vae", "seconds": 0.0014579769922420382}
{"event": "h3_vae_load_begin", "component": "vae"}
{"event": "h3_vae_load_end", "component": "vae", "seconds": 0.00042890897020697594}
{"event": "h3_vae_decode_begin", "component": "vae"}
{"event": "h3_vae_decode_exit", "component": "vae", "seconds": 0.002469334052875638}
{"event": "h3_vae_load_begin", "component": "audio_vae"}
{"event": "h3_vae_load_end", "component": "audio_vae", "seconds": 0.0003516220021992922}
{"event": "h3_vae_decode_begin", "component": "audio_vae"}
{"event": "h3_vae_decode_exit", "component": "audio_vae", "seconds": 0.0015545400092378259}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0011313209543004632, "largest_embedding_bytes": 61440}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0008068500319495797, "largest_embedding_bytes": 61440}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0007318799616768956, "largest_embedding_bytes": 61440}
{"event": "h3_vae_load_begin", "component": "vae"}
{"event": "h3_vae_load_end", "component": "vae", "seconds": 0.000385797000490129}
{"event": "h3_vae_decode_begin", "component": "vae"}
{"event": "h3_vae_decode_exit", "component": "vae", "seconds": 0.001827846048399806}
{"event": "h3_vae_load_begin", "component": "audio_vae"}
{"event": "h3_vae_load_end", "component": "audio_vae", "seconds": 0.0005915849469602108}
{"event": "h3_vae_decode_begin", "component": "audio_vae"}
{"event": "h3_vae_decode_exit", "component": "audio_vae", "seconds": 0.0014311270788311958}
{"event": "h3_vae_load_begin", "component": "vae"}
{"event": "h3_vae_load_end", "component": "vae", "seconds": 0.00034260598476976156}
{"event": "h3_vae_decode_begin", "component": "vae"}
{"event": "h3_vae_decode_exit", "component": "vae", "seconds": 0.002762255957350135}
{"event": "h3_vae_load_begin", "component": "audio_vae"}
{"event": "h3_vae_load_end", "component": "audio_vae", "seconds": 0.0003466940252110362}
{"event": "h3_vae_decode_begin", "component": "audio_vae"}
{"event": "h3_vae_decode_exit", "component": "audio_vae", "seconds": 0.0013139970833435655}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0008889849996194243, "largest_embedding_bytes": 61440}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0008009900338947773, "largest_embedding_bytes": 61440}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.000653863069601357, "largest_embedding_bytes": 61440}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0009164069779217243, "largest_embedding_bytes": 61440}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0010663500288501382, "largest_embedding_bytes": 61440}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.001235760049894452, "largest_embedding_bytes": 61440}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0014510949840769172, "largest_embedding_bytes": 61440}
{"event": "h3_cache_finite_scan_begin", "train_rows": 8, "eval_rows": 1}
{"event": "h3_cache_finite_scan_end", "seconds": 0.0009394900407642126, "largest_embedding_bytes": 61440}
{"event": "h3_vae_load_begin", "component": "vae"}
{"event": "h3_vae_load_end", "component": "vae", "seconds": 0.0002609930234029889}
{"event": "h3_vae_decode_begin", "component": "vae"}
{"event": "h3_vae_decode_exit", "component": "vae", "seconds": 0.0018068770878016949}
{"event": "h3_vae_load_begin", "component": "audio_vae"}
{"event": "h3_vae_load_end", "component": "audio_vae", "seconds": 0.0002821929519996047}
{"event": "h3_vae_decode_begin", "component": "audio_vae"}
{"event": "h3_vae_decode_exit", "component": "audio_vae", "seconds": 0.0016189010348170996}
{"event": "h3_vae_load_begin", "component": "vae"}
{"event": "h3_vae_load_end", "component": "vae", "seconds": 0.00020765000954270363}
{"event": "h3_vae_decode_begin", "component": "vae"}
{"event": "h3_vae_decode_exit", "component": "vae", "seconds": 0.001941908965818584}
{"event": "h3_vae_load_begin", "component": "audio_vae"}
{"event": "h3_vae_load_end", "component": "audio_vae", "seconds": 0.00020736095029860735}
{"event": "h3_vae_decode_begin", "component": "audio_vae"}
{"event": "h3_vae_decode_exit", "component": "audio_vae", "seconds": 0.001249002991244197}
=========================== short test summary info ============================
FAILED tests/test_flash_collection_runtime.py::test_real_controller_collects384_then_two192_and_resumes_without_scoring
FAILED tests/test_flash_collection_runtime.py::test_approved140_manifest_counts
2 failed, 5 passed in 43.26s
