"""CPU integration only: real controller/stages/kernel, fake model/reward leaves."""
import json
import os
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch
import yaml

from tests import test_default_composition_e2e as e2e
from tests import test_v2_demo_default_composition as demo
from visual_rl.runtime.composition import create_default_run_controller


def _author_data_dir():
    configured = os.environ.get("VISUALRL_FLASH_AUTHOR_DATA")
    root = (Path(configured).expanduser() if configured else
            Path(__file__).resolve().parents[1] / "reference_code/flash_grpo/dataset/video")
    missing = [name for name in ("train.txt", "test.txt") if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Flash-GRPO author data at {root}: {', '.join(missing)}. "
            "Set VISUALRL_FLASH_AUTHOR_DATA to the original dataset/video directory "
            "or copy train.txt and test.txt to reference_code/flash_grpo/dataset/video."
        )
    return root


def config(root, artifacts, output, *, resume=None, steps=2, rows=192):
    root.mkdir()
    case = next(c for c in demo._DEMO_CASES if c.recipe_id == "flash_grpo_v1")
    path = demo._write_smoke_config(root, case, output_dir=output, artifacts=artifacts,
                                     resume_from=resume, max_optimizer_steps=steps)
    raw = yaml.safe_load(path.read_text())
    raw["overrides"]["training"].update({
        "global_prompt_batch_size": rows * 2 // 8,
        "collection": {"collection_rows_per_commit": rows, "collection_commits": 2,
                       "collection_chunk_rows": 16},
        "policy_recompute": {"row_microbatch_size": 4, "transition_window_size": 1},
    })
    raw["overrides"]["execution"].update({"group_size": 8, "rollout": {
        "forward_microbatch_size": 4, "decode_microbatch_size": 1,
        "trajectory_storage_device": "cpu"}})
    path.write_text(yaml.safe_dump(raw))
    return path


def setup(monkeypatch, tmp_path):
    import visual_rl
    from visual_rl.algorithms.trainer import collection
    checkout = Path(__file__).resolve().parents[1]
    assert Path(visual_rl.__file__).resolve().parent == checkout / "visual_rl"
    assert Path(collection.__file__).resolve() == checkout / "visual_rl/algorithms/trainer/collection.py"
    # These tests compare multiple fresh controllers in ONE process. Diffusers'
    # lazy HTTP/Rich import initializes a style id using Python's global RNG.
    # Prime that one-time import before either controller seeds its run, as in
    # test_default_composition_continuation_matches_continuous_two_step_run.
    # Keep the exact RNG/checkpoint assertions; do not mask restore differences.
    import importlib
    importlib.import_module("diffusers")
    e2e._patch_default_runtime(monkeypatch)
    case = next(c for c in demo._DEMO_CASES if c.recipe_id == "flash_grpo_v1")
    artifacts = demo._write_fake_artifacts(tmp_path / "artifacts", case)
    code = tmp_path / "code"
    code.mkdir()
    (code / "identity.py").write_text("UNIT_TEST = True\n")
    return artifacts, code


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
    # Opt-in parallel hashing must fold to the same ordered tree and digest.
    (root / "nested").mkdir()
    for index in range(9):
        (root / "nested" / f"item-{index}.bin").write_bytes(bytes([index]) * ((index + 1) * 1000))
    import threading
    expected_paths = [f'nested/item-{i}.bin' for i in range(9)] + ['shard.bin']
    with monkeypatch.context() as patch:
        def reject_read_bytes(self):
            raise AssertionError("unbounded checkpoint read")
        patch.setattr(Path, "read_bytes", reject_read_bytes)
        fsync_threads = []
        real_fsync = os.fsync
        def observed_fsync(descriptor):
            fsync_threads.append(threading.get_ident())
            return real_fsync(descriptor)
        patch.setattr(os, "fsync", observed_fsync)
        tree = checkpoint_manager._build_state_tree(root)
        assert [entry['path'] for entry in tree['files']] == expected_paths
        assert tree['files'][-1]['sha256'] == hashlib.sha256(payload).hexdigest()
        assert fsync_threads == [threading.get_ident()] * len(expected_paths)
        fsync_threads.clear()
        parallel = checkpoint_manager._build_state_tree(root, threads=4)
        assert parallel == tree
        assert checkpoint_manager._payload_digest(parallel) == checkpoint_manager._payload_digest(tree)
        assert len(fsync_threads) == len(expected_paths) and threading.get_ident() not in fsync_threads
        for bad in (True, 4.0, '4', None):
            with pytest.raises(TypeError, match='int'):
                checkpoint_manager._build_state_tree(root, threads=bad)
        for bad in (0, -1, 65):
            with pytest.raises(ValueError, match='1..64'):
                checkpoint_manager._build_state_tree(root, threads=bad)
        # threads==1 keeps the original interleaved loop (hash+fsync file k
        # before validating file k+1); threads>1 validates every path first.
        reserved = root / 'state_tree.json'
        reserved.write_bytes(b'{}')
        for threads, expected_fsyncs in ((1, len(expected_paths)), (4, 0)):
            fsync_threads.clear()
            with pytest.raises(ValueError, match='reserved file'):
                checkpoint_manager._build_state_tree(root, threads=threads)
            assert len(fsync_threads) == expected_fsyncs
        reserved.unlink()
        checkpoint_manager._verify_state_tree(root, tree)
        with shard.open('r+b') as stream:
            stream.write(b'X')
        with pytest.raises(ValueError, match='digest mismatch'):
            checkpoint_manager._verify_state_tree(root, tree)
    # A hashing worker failure propagates like the serial loop and nothing is
    # published; the unchanged restore/validation path accepts a parallel publish.
    from tests.test_checkpoint_coordinator import _contract as publish_contract, _progress as publish_progress, _selection_policy
    contract = publish_contract()
    progress = publish_progress(1, _selection_policy())
    real_digest = checkpoint_manager._file_digest_size
    def broken_digest(path):
        if path.name == 'item-3.bin':
            raise OSError('injected hash failure')
        return real_digest(path)
    def write_tree(directory):
        (directory / 'nested').mkdir()
        for index in range(9):
            (directory / 'nested' / f'item-{index}.bin').write_bytes(bytes([index]) * ((index + 1) * 1000))
        (directory / 'shard.bin').write_bytes(payload)
    parallel_manager = checkpoint_manager.AtomicCheckpointManager(tmp_path / 'publish-parallel', checkpoint_threads=4)
    serial_manager = checkpoint_manager.AtomicCheckpointManager(tmp_path / 'publish-serial')
    assert (parallel_manager.checkpoint_threads, serial_manager.checkpoint_threads) == (4, 1)
    with pytest.raises(TypeError, match='int'):
        checkpoint_manager.AtomicCheckpointManager(tmp_path / 'publish-bad', checkpoint_threads=True)
    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_manager, '_file_digest_size', broken_digest)
        for threads in (1, 4):
            with pytest.raises(OSError, match='injected hash failure'):
                checkpoint_manager._build_state_tree(root, threads=threads)
        with pytest.raises(OSError, match='injected hash failure'):
            parallel_manager.commit(1, contract, write_tree, progress=progress)
    assert list(parallel_manager.root.iterdir()) == [] and parallel_manager.latest_complete() is None
    published = parallel_manager.commit(1, contract, write_tree, progress=progress)
    serial = serial_manager.commit(1, contract, write_tree, progress=progress)
    assert published.state_tree_id == serial.state_tree_id
    assert json.loads((published.path / 'state_tree.json').read_text()) == json.loads((serial.path / 'state_tree.json').read_text())
    assert [entry['path'] for entry in json.loads((published.path / 'state_tree.json').read_text())['tree']['files']] == expected_paths
    assert parallel_manager.inspect_complete(published.path, expected_contract=contract).committed.state_tree_id == published.state_tree_id
    # Check the production writer configuration without pretending this is GPU DCP.
    default_hook = threading.excepthook
    with monkeypatch.context() as patch:
        calls = []
        class RecordingWriter:
            def __init__(self, path, **kw):
                calls.append(kw)
            def write_data(self, plan, planner):
                raise AssertionError('writer must not run under the no-op save')
        patch.setattr(dcp, 'FileSystemWriter', RecordingWriter)
        patch.setattr(dcp, 'save', lambda *args, **kw: None)
        _save_dcp({}, root / 'dcp', None)
        _save_dcp({}, root / 'dcp', None, thread_count=4)
        assert calls == [dict(overwrite=False, sync_files=True,
                              single_file_per_rank=False, thread_count=1),
                         dict(overwrite=False, sync_files=True,
                              single_file_per_rank=False, thread_count=4)]
        assert threading.excepthook is default_hook
        with pytest.raises(TypeError, match='int'):
            _save_dcp({}, root / 'dcp', None, thread_count=True)
        with pytest.raises(ValueError, match='1..64'):
            _save_dcp({}, root / 'dcp', None, thread_count=0)
    # Real tiny CPU DCP through the same leaf: four writer threads produce the
    # same one-item-per-file set with identical item bytes and equal metadata
    # keys/entries. The .metadata bytes differ between saves by DCP's per-save
    # uuid and, for threads>1, by storage_data insertion (thread completion)
    # order, which the key-based reader ignores. Not GPU/WORLD8 evidence.
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.api import CheckpointException
    from visual_rl.artifacts.checkpoint.h3_state import _disk_schema
    def tiny_state():
        generator = torch.Generator().manual_seed(7)
        return {'model': {f'layer{i}.weight': torch.randn(16, 8, generator=generator) for i in range(6)},
                'buffers': {'rope.inv_freq': torch.arange(4.)}, 'groups': [{'lr': 1e-4}]}
    serial_dcp, parallel_dcp = tmp_path / 'dcp-threads-1', tmp_path / 'dcp-threads-4'
    _save_dcp(tiny_state(), serial_dcp, None)
    _save_dcp(tiny_state(), parallel_dcp, None, thread_count=4)
    names = sorted(entry.name for entry in serial_dcp.iterdir())
    assert names == sorted(entry.name for entry in parallel_dcp.iterdir()) and len(names) == 9 and '.metadata' in names
    assert all((serial_dcp / name).read_bytes() == (parallel_dcp / name).read_bytes() for name in names if name != '.metadata')
    serial_meta, parallel_meta = (FileSystemReader(path).read_metadata() for path in (serial_dcp, parallel_dcp))
    assert serial_meta.state_dict_metadata == parallel_meta.state_dict_metadata
    assert serial_meta.storage_data == parallel_meta.storage_data
    for path in (serial_dcp, parallel_dcp):
        _disk_schema(path, tiny_state())  # chunk coverage accepts complete saves of either kind
    restored = tiny_state()
    for value in restored['model'].values():
        value.zero_()
    dcp.load(restored, checkpoint_id=parallel_dcp, process_group=None)
    assert all(torch.equal(restored['model'][name], value) for name, value in tiny_state()['model'].items())
    # Restore refuses metadata whose storage entries do not cover every chunk.
    import pickle
    metadata_file = serial_dcp / '.metadata'
    complete_metadata = metadata_file.read_bytes()
    damaged = pickle.loads(complete_metadata)
    del damaged.storage_data[next(iter(damaged.storage_data))]
    metadata_file.write_bytes(pickle.dumps(damaged))
    with pytest.raises(ValueError, match='do not cover'):
        _disk_schema(serial_dcp, tiny_state())
    metadata_file.write_bytes(complete_metadata)
    _disk_schema(serial_dcp, tiny_state())
    # Writer-thread failures must fail a threads>1 save loudly: torch only
    # joins its raw writer threads, so the hook-captured OSError is raised
    # inside write_data and exchanged by DCP like a serial-path error; no
    # .metadata is written, the hook is restored and no writer thread outlives
    # the call. The second case fails on the calling thread while workers are
    # still writing, which the bounded join must absorb before returning.
    calling = threading.get_ident()
    real_torch_save = torch.save
    worker_failed, calling_failed = threading.Event(), threading.Event()
    def worker_fails(obj, stream, *args, **kwargs):
        if threading.get_ident() == calling:
            worker_failed.wait(5)
        elif not worker_failed.is_set():
            worker_failed.set()
            raise OSError(28, 'No space left on device (injected)')
        return real_torch_save(obj, stream, *args, **kwargs)
    def calling_fails(obj, stream, *args, **kwargs):
        if threading.get_ident() == calling:
            calling_failed.set()
            raise OSError(5, 'Input/output error (injected)')
        calling_failed.wait(5)
        return real_torch_save(obj, stream, *args, **kwargs)
    for label, flaky in (('worker', worker_fails), ('calling', calling_fails)):
        target = tmp_path / f'dcp-{label}-failure'
        before = set(threading.enumerate())
        with monkeypatch.context() as patch:
            patch.setattr(torch, 'save', flaky)
            with pytest.raises(CheckpointException) as failure:
                _save_dcp(tiny_state(), target, None, thread_count=4)
        assert 'injected' in str(failure.value)
        assert threading.excepthook is default_hook
        assert not [thread for thread in threading.enumerate() if thread not in before and thread.is_alive()]
        assert not (target / '.metadata').exists()
    # Through the manager's publish path the failure leaves nothing published.
    def failing_writer(directory):
        worker_failed.clear()
        with monkeypatch.context() as patch:
            patch.setattr(torch, 'save', worker_fails)
            _save_dcp(tiny_state(), directory / 'dcp', None, thread_count=4)
    with pytest.raises(CheckpointException):
        parallel_manager.commit(2, contract, failing_writer, progress=publish_progress(2, _selection_policy()))
    assert sorted(entry.name for entry in parallel_manager.root.iterdir()) == ['latest.json', 'step-1']
    assert parallel_manager.latest_complete().step == 1
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
    # O1: opt-in anchor reuse performs zero no-grad forwards and reproduces the
    # recompute path bit-for-bit (rollout-time old score, schedule-only coefficients).
    forwards=[];anchor_gates=[]
    def counted_predict(sid,x,index):
        forwards.append((sid,index));return torch.zeros_like(x)
    def captured(rows=records,version='behavior',**options):
        owner=H3DualReplayAnchor(dynamics=dynamics,behavior_id='behavior',expected_keys=keys,max_records=2,**options)
        owner.capture(rows,device='cpu',predict=counted_predict,read_behavior_id=lambda:version,
            gate=lambda event,**kw:anchor_gates.append((event,kw.get('local_error'))))
        return owner
    recompute=captured();assert len(forwards)==2 and recompute.reuse_rollout is False
    assert anchor_gates==[(f'anchor/{i}/{name}',err) for i in range(2) for name,err in (('forward',None),('prepare',False),('result',False))]
    del anchor_gates[:]
    reused=captured(reuse_rollout=True);assert len(forwards)==2 and reused.reuse_rollout is True
    assert anchor_gates==[(f'anchor/{i}/reuse',False) for i in range(2)]
    for record in records:
        (score,anchor_coefs),(old,reuse_coefs)=recompute.get(record.sample_id,record.index),reused.get(record.sample_id,record.index)
        assert torch.equal(score,old) and torch.equal(old,record.old_joint) and (score.dtype,score.shape)==(old.dtype,old.shape)
        assert len(reuse_coefs)==2 and all(torch.equal(a,b) and (a.dtype,a.shape)==(b.dtype,b.shape) for a,b in zip(anchor_coefs,reuse_coefs))
        assert all(torch.equal(a,b) for a,b in zip(reuse_coefs,dynamics.coefficients(record.index,device='cpu')))
        old.fill_(99);assert torch.equal(reused.get(record.sample_id,record.index)[0],record.old_joint)
    with pytest.raises(TypeError,match='bool'):captured(reuse_rollout=1)
    with pytest.raises(ValueError,match='old score missing'):
        captured(rows=(replace(records[0],old_joint=None),records[1]),reuse_rollout=True)
    assert anchor_gates[-1]==('anchor/0/reuse',True)
    with pytest.raises(ValueError,match='behavior changed'):captured(version='other',reuse_rollout=True)
    assert len(forwards)==2
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
    # Transient nvidia-smi timeouts degrade the monitor; only N consecutive are fatal.
    import subprocess
    def scripted_monitor(directory, script, captured):
        calls=iter(script)
        def query(fields, kind):
            if kind!='gpu': return [['GPU-test',str(os.getpid()),'10','python']]
            step=next(calls,'stop')
            if step=='stop':
                (directory/'STOP').touch();return [['0','GPU-test','12','52','64','9']]
            if step=='health':
                health=json.loads((directory/'health.json').read_text())
                captured.append((health,monitor_healthy(directory,now_ns=health['monotonic_ns']+1)))
            if step=='fail': raise subprocess.TimeoutExpired(['nvidia-smi'],8)
            return [['0','GPU-test','12','52','64','9']]
        return query
    with monkeypatch.context() as mp:
        captured=[];degraded=tmp_path/'monitor-degraded'
        mp.setattr(card_monitor,'query',scripted_monitor(degraded,['ok','ok','fail','fail','health','ok'],captured))
        card_monitor.run(degraded,tmp_path/'memory-events',interval=0.)
        rows=[json.loads(line) for line in (degraded/'card-process.jsonl').read_text().splitlines()]
        assert [r.get('consecutive_failures') for r in rows if r.get('degraded')]==[1,2]
        assert all(r['schema']=='h3.card-sample.v1' and 'TimeoutExpired' in r['error'] for r in rows if r.get('degraded'))
        mid,seen=captured[0]
        assert mid['healthy'] is True and mid['degraded'] is True and mid['consecutive_failures']==2 and 'TimeoutExpired' in mid['last_error']
        assert seen['degraded'] is True and seen['consecutive_failures']==2
        final=json.loads((degraded/'health.json').read_text())
        assert final['healthy'] is True and final['degraded'] is False and final['consecutive_failures']==0
        assert monitor_healthy(degraded,now_ns=final['monotonic_ns']+1)['degraded'] is False
        assert json.loads((degraded/'exit.json').read_text())['exit_code']==0
        fatal=tmp_path/'monitor-fatal'
        mp.setattr(card_monitor,'query',scripted_monitor(fatal,['ok','ok','fail','fail','fail','ok'],[]))
        with pytest.raises(subprocess.TimeoutExpired):card_monitor.run(fatal,tmp_path/'memory-events',interval=0.)
        assert json.loads((fatal/'health.json').read_text())['healthy'] is False
        assert json.loads((fatal/'exit.json').read_text())['exit_code']==1
        with pytest.raises(RuntimeError,match='unhealthy'):monitor_healthy(fatal)
        for bad in (0,True,1.5):
            with pytest.raises(ValueError):card_monitor.run(tmp_path/'monitor-bad',tmp_path/'memory-events',max_consecutive_failures=bad)
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
        mp.setattr(resolution_entry,'_exchange',lambda *a:[a[3]]*8)
        backend=resolution_entry.TwoUpdateBackend(args=SimpleNamespace(ffmpeg=Path('ffmpeg'),ffprobe=Path('ffprobe'),expected_sha='test',monitor_dir=monitor,reward_pipeline=False,profile_rollout=False,reuse_video_vae=False,reuse_rollout_anchor=False, checkpoint_threads=1))
        backend.bind(None,runtime,graph,prepared,None)
        bound.trainer.run_iteration(0);bound.trainer.run_iteration(1)
        assert len(backend.audits)==2
        assert backend.audits[1]['graph']['nograd_forwards']==88
        assert backend.audits[1]['graph']['final_prediction_released']
        assert len(backend.audits[1]['graph']['grad_forward_indices'])==8
        assert json.loads((fake_root/'update-2-audit.json').read_text())['optimizer_updates']==2
    journal.close()
    raw['launch']['checkpoint_every_optimizer_steps']=6
    with pytest.raises(ValueError,match='terminal exception'):
        compile_h3_profile(save_h3_profile(profile_path.parent,raw))
    fresh.launch.max_optimizer_steps=200
    with pytest.raises(ValueError,match='max1'): validate_fresh(fresh)
    fresh.launch.max_optimizer_steps=1;fresh.launch.resume_from=Path('old/step-3')
    with pytest.raises(ValueError,match='max1'): validate_fresh(fresh)
    fresh.launch.resume_from=None
    media_source=SimpleNamespace(_pending=SimpleNamespace(identity=SimpleNamespace(optimizer_step=0),
        group_layout='paired_4x8',samples=(('s','g',0),),prompts={'s':'prompt'},collection_id='c',behavior_id='b'))
    assert selection(media_source,'s',every=5,terminal_step=1)['step']==1
    assert selection(media_source,'s',every=5) is None
    media_source._pending.identity.optimizer_step=4
    assert selection(media_source,'s',every=5)['step']==5
    module=torch.nn.Module();module.register_parameter('frozen',torch.nn.Parameter(torch.ones(2),requires_grad=False));module.register_parameter('lora',torch.nn.Parameter(torch.ones(2)))
    before=weight_fingerprints(module)
    watcher=PredictAudit(lambda *a,**kw:module.lora*2)
    prediction=watcher(index=2)
    with pytest.raises(RuntimeError,match='retained'):watcher(index=2)
    prediction.sum().backward();del prediction
    assert watcher.payload()['final_prediction_released']
    with torch.no_grad():module.lora.add_(.1)
    after=weight_fingerprints(module)
    assert before['hashes']['frozen']==after['hashes']['frozen'] and before['hashes']['trainable']!=after['hashes']['trainable']
    from visual_rl.runtime.h3_lifecycle import _pause_requested,H3LifecycleBackend
    with pytest.raises(RuntimeError,match='inspection-only backend'):
        H3LifecycleBackend(inspection_only=True).run(None)
    from visual_rl.algorithms.optimization import h3_world as pause_world
    pause_root=tmp_path/'pause-root';pause_root.mkdir()
    with monkeypatch.context() as pp:
        pp.setattr(pause_world,'_exchange',lambda label,rank,group,local,valid:[local]+[None]*7)
        session=SimpleNamespace(rank=0,process_group=None)
        assert not _pause_requested(session,pause_root)
        (pause_root/'PAUSE_REQUESTED').touch()
        assert _pause_requested(session,pause_root)
    data_root=Path(__file__).resolve().parents[1]/'data/h3_author_video'
    train_bytes=(data_root/'train.txt').read_bytes();test_bytes=(data_root/'test.txt').read_bytes()
    assert hashlib.sha256(train_bytes).hexdigest()==TRAIN_SHA and hashlib.sha256(test_bytes).hexdigest()==TEST_SHA
    assert len(train_bytes.splitlines())==19700 and len(test_bytes.splitlines())==300
    assert not set(train_bytes.splitlines())&set(test_bytes.splitlines())
    # Full cache finite scan visits each packed shard once, keeps before/after
    # hash/index guards and still rejects legitimate-hash nonfinite payloads.
    from tests import test_h3_mvp_cpu_text_cache as cache_fixture
    from contextlib import contextmanager
    cache_root=tmp_path/'packed-cache';data,tensors=cache_fixture._fixture(cache_root)
    reader=cache_fixture._reader(cache_root);visited=[];checked=reader._checked_shard
    with monkeypatch.context() as cp:
        @contextmanager
        def observed_shard(name):
            visited.append(name)
            with checked(name) as handle:yield handle
        cp.setattr(reader,'_checked_shard',observed_shard)
        cp.setattr(reader,'get',lambda prompt: (_ for _ in ()).throw(AssertionError('per-prompt rehash regression')))
        assert reader.validate_embeddings(['cat',' cat','heldout\n'])==3*5120*4
    assert len(visited)==2 and len(set(visited))==2
    tensors['embeddings_00000.safetensors'][cache_fixture._key('cat')][1,0]=float('nan')
    cache_fixture._replace_shard(cache_root,data,tensors,'embeddings_00000.safetensors')
    with pytest.raises(ValueError,match='finite'):
        cache_fixture._reader(cache_root).validate_embeddings(['cat',' cat','heldout\n'])
    # Actual preflight entry on real tiny files (not only the scan method) so
    # progress-log imports and integration seams execute in the CPU gate.
    from visual_rl.composition.config.h3_profile import preflight_h3_profile
    integration_path,_=h3_profile_fixture(tmp_path/'full-preflight-entry')
    facts=preflight_h3_profile(compile_h3_profile(integration_path))
    assert facts.semantic_id.startswith('h3.semantic.v1:')
    # A slow file worker must keep main-thread collectives alive; worker
    # exceptions and a separate total deadline are preserved, never hidden.
    from visual_rl.runtime import h3_file_phase
    import threading, time
    with monkeypatch.context() as heartbeat_patch:
        zeros = torch.zeros
        ticks=[]
        heartbeat_patch.setattr(h3_file_phase.torch,'zeros',lambda *a,**kw: zeros((),dtype=torch.int32))
        def reduce(signal,**kw):
            assert threading.current_thread() is threading.main_thread()
            ticks.append(int(signal.item())); signal.add_(7)
        heartbeat_patch.setattr(h3_file_phase.dist,'all_reduce',reduce)
        def slow_file():
            assert threading.current_thread() is not threading.main_thread()
            time.sleep(.04);return 'hashed'
        assert h3_file_phase.file_phase(slow_file,rank=0,group=None,label='cpu-test',timeout_s=1,interval_s=.005)=='hashed'
        assert ticks.count(0)>1 and ticks[-1]==1
        with pytest.raises(TimeoutError,match='file phase exceeded'):
            h3_file_phase.file_phase(slow_file,rank=0,group=None,label='bounded',timeout_s=.005,interval_s=.005)
        def broken_file(): raise ValueError('disk-failure')
        with pytest.raises(ValueError,match='disk-failure'):
            h3_file_phase.file_phase(broken_file,rank=0,group=None,label='error',timeout_s=1,interval_s=.005)
    # Real disk DCP: new adapter-only payload and legacy full-state restore.
    # Only WORLD8/CUDA transport leaves use the established tiny CPU fixture.
    from tests import test_h3_mvp_cpu_checkpoint_state as h3_fixture
    from visual_rl.artifacts.checkpoint import h3_state as h3_codec
    with monkeypatch.context() as hp:
        from visual_rl.models.implementations import h3_loading
        # Tiny ~4MB CPU fixture, not admission for the real 33B model; the
        # VPS has 15GiB, below the production-only 24GiB loading reserve.
        hp.setattr(h3_loading,'_admit_host_memory',lambda budget: None)
        group, _ = h3_fixture._cpu_transport(hp)
        prepared = h3_fixture._prepared(tmp_path/'h3-source',hp)
        h3_fixture._step(prepared); h3_fixture._step(prepared)
        saved_state = h3_fixture._snapshot(prepared)
        original_manifest = h3_codec._manifest
        for version in ('legacy','adapter'):
            checkpoint = tmp_path/f'h3-{version}';checkpoint.mkdir()
            if version == 'legacy':
                with monkeypatch.context() as legacy_patch:
                    legacy_patch.setattr(h3_codec,'_manifest',lambda *a,**kw: original_manifest(*a,schema=h3_codec._LEGACY_SCHEMA))
                    h3_fixture._save(prepared,checkpoint,group)
            else:
                h3_fixture._save(prepared,checkpoint,group)
            manifest = json.loads((checkpoint/'h3-state/manifest.json').read_text())
            if version == 'adapter':
                assert manifest['schema']==h3_codec._SCHEMA
                assert manifest['frozen_base']['parameters']
                assert not set(manifest['model']) & set(manifest['frozen_base']['parameters'])
                assert set(manifest['trainable']) <= set(manifest['model'])
            else:
                assert manifest['schema']==h3_codec._LEGACY_SCHEMA
                assert 'frozen_base' not in manifest
            restored=h3_fixture._prepared(tmp_path/f'h3-restored-{version}',hp)
            h3_fixture._load(restored,checkpoint,group)
            h3_fixture._equal(h3_fixture._snapshot(restored),saved_state)
        bad_manifest = json.loads((tmp_path/'h3-adapter/h3-state/manifest.json').read_text())
        bad_manifest['frozen_base']['model_id']='f'*64
        (tmp_path/'h3-adapter/h3-state/manifest.json').write_text(json.dumps(bad_manifest))
        rejected=h3_fixture._prepared(tmp_path/'h3-rejected',hp)
        with pytest.raises(ValueError,match='load/preflight rejected local input') as rejected_error:
            h3_fixture._load(rejected,tmp_path/'h3-adapter',group)
        assert isinstance(rejected_error.value.__cause__, ValueError)
        assert 'schema mismatch' in str(rejected_error.value.__cause__)
        assert not rejected.optimizer.state
    # Scoped code upgrade preserves the old owner and refuses other outputs,
    # semantics or unrecognized checkpoint code. Only tiny live-contract leaves.
    from scripts import h3_checkpoint_upgrade as upgrade_module
    from types import SimpleNamespace
    with monkeypatch.context() as up:
        output=tmp_path/'upgrade-output';output.mkdir()
        checkpoint=output/'checkpoints/step-1';checkpoint.mkdir(parents=True)
        up.setattr(upgrade_module,'OUTPUT',output)
        upgrade=object.__new__(upgrade_module.H3CodeUpgrade)
        upgrade.root=tmp_path/'new-source'
        upgrade.proof=dict(expected_sha='f'*40)
        upgrade.legacy_code={'content_sha256':upgrade_module.OLD_CODE}
        current=dict(schema_version=1,semantic_id=upgrade_module.SEMANTIC,code='f'*64,gpu_uuids=['gpu'])
        saved=dict(current,code=upgrade_module.OLD_CODE,run_id=upgrade_module.RUN_ID)
        owner=output/'h3-run.json';owner.write_text(json.dumps(saved))
        before_owner=owner.read_bytes()
        profile=SimpleNamespace(launch=SimpleNamespace(output_dir=output,resume_from=checkpoint,max_optimizer_steps=200),
                                training=SimpleNamespace(algorithm='dual_flash',group_layout='paired_4x8'))
        upgrade.admit_owner(saved=saved,current=current,profile=profile,code_root=upgrade.root/'visual_rl')
        assert owner.read_bytes()==before_owner
        assert (output/'code-upgrades'/('f'*40+'.json')).is_file()
        with pytest.raises(ValueError,match='owner/semantics/topology'):
            upgrade.admit_owner(saved=dict(saved,semantic_id='different'),current=current,profile=profile,code_root=upgrade.root/'visual_rl')
        live_calls=[]
        def live_ids(**kw):
            live_calls.append(kw);return {'code':kw.get('code_content',{'content_sha256':'f'*64})['content_sha256']}
        binding=SimpleNamespace(content={'code':{'content_sha256':'f'*64}},contract='current-contract',
                                _live_ids=live_ids,_contract=lambda *,live: ('legacy-contract',live))
        manifest=checkpoint/'h3_transaction.json'
        manifest.write_text(json.dumps({'binding':{'live':{'code':upgrade_module.OLD_CODE}}}))
        contract,live=upgrade.restore_contract(binding,checkpoint)
        assert live['code']==upgrade_module.OLD_CODE and contract[0]=='legacy-contract'
        manifest.write_text(json.dumps({'binding':{'live':{'code':'f'*64}}}))
        assert upgrade.restore_contract(binding,checkpoint)==('current-contract',{'code':'f'*64})
        manifest.write_text(json.dumps({'binding':{'live':{'code':'unknown'}}}))
        with pytest.raises(ValueError,match='unknown checkpoint code'):
            upgrade.restore_contract(binding,checkpoint)
        assert owner.read_bytes()==before_owner
    # Observation-only export: cadence, original score, media/RNG and real CPU codec.
    from scripts.launch_h3_media import observed_score, save_media
    from visual_rl.models.implementations.h3_decode import H3AVMedia
    from types import SimpleNamespace
    import shutil, random
    samples = tuple((f's{g}-{m}',f'g{g}',m) for g in range(4) for m in range(8))
    request = SimpleNamespace(identity=SimpleNamespace(optimizer_step=0),
        group_layout='paired_4x8',samples=samples,prompts={sid:'a prompt' for sid,_,_ in samples},
        collection_id='collection',behavior_id='behavior')
    source = SimpleNamespace(_pending=request)
    saved,scored = [],[]
    marker = object()
    def original(**kw):
        assert kw['media'] is marker
        scored.append(kw['sample_id'])
        return 1.25
    callback = observed_score(original,source=source,root=tmp_path/'media',
        ffmpeg='ffmpeg',ffprobe='ffprobe',launcher_sha='test',saver=lambda **kw:saved.append(kw))
    for step in (1,9,10,20):
        request.identity.optimizer_step=step-1
        count=len(saved)
        for sid,_,_ in samples:
            assert callback(sample_id=sid,prompt='a prompt',media=marker)==1.25
        assert len(saved)-count == (4 if step%10==0 else 0)
    assert len(scored)==128
    assert {x['metadata']['member'] for x in saved}=={0}
    assert {x['metadata']['group_id'] for x in saved}=={'g0','g1','g2','g3'}
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
    assert ffmpeg and ffprobe, 'CPU media gate requires ffmpeg and ffprobe'
    video=torch.linspace(0,1,3*3*16*16).reshape(3,3,16,16)
    audio=torch.zeros(8000,2);media=H3AVMedia(video,audio,12,32000)
    before_video,before_audio=video.clone(),audio.clone()
    rng,python_rng=torch.get_rng_state().clone(),random.getstate()
    folder=save_media(root=tmp_path/'real-encode',metadata=saved[0]['metadata'],
        media=media,reward=1.25,ffmpeg=ffmpeg,ffprobe=ffprobe,launcher_sha='test')
    receipt=json.loads((folder/'receipt.json').read_text())
    assert receipt['frames']==3 and receipt['reward']==1.25 and (folder/'video.mp4').stat().st_size>0
    assert torch.equal(video,before_video) and torch.equal(audio,before_audio)
    assert torch.equal(torch.get_rng_state(),rng) and random.getstate()==python_rng
    with pytest.raises(FileExistsError):
        save_media(root=tmp_path/'real-encode',metadata=saved[0]['metadata'],
            media=media,reward=1.25,ffmpeg=ffmpeg,ffprobe=ffprobe,launcher_sha='test')
    from scripts.launch_flash140 import SafePointPause, observe_checkpoints, inspect_checkpoint
    caplog.set_level("INFO", logger="visual_rl.flash140")
    from visual_rl.algorithms.optimization.recompute import PolicyRecomputer
    artifacts, code = setup(monkeypatch, tmp_path)
    replay_calls = []
    compute = PolicyRecomputer.compute_current_slot
    def counted(self, request, slot, **kwargs):
        replay_calls.append(slot.row_indices)
        return compute(self, request, slot, **kwargs)
    monkeypatch.setattr(PolicyRecomputer, "compute_current_slot", counted)
    original_score = demo._FakeRewardResource.score
    def shifted_score(self, *, batch):
        result = original_score(self, batch=batch)
        return replace(result, values=result.values + self.score_calls * 10)
    monkeypatch.setattr(demo._FakeRewardResource, "score", shifted_score)
    rewards = demo._FakeRewardResourceFactory()
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                               reward_resource_factory=rewards)
    checkpoints = []
    observe_checkpoints(controller, pause_after=1, on_checkpoint=checkpoints.append)
    output = tmp_path / "out"
    with pytest.raises(SafePointPause) as paused:
        controller.run(config(tmp_path / "first", artifacts, output))
    assert paused.value.record["status"] == "paused_not_success"
    assert paused.value.record["committed_steps"] == 1
    assert checkpoints[0]["metrics"]["active_transition_count"] == 192
    assert rewards.resources[0].score_calls == 24
    assert not (output / "SUCCESS").exists()
    checkpoint = output / "checkpoints/step-1"
    inspected_first = inspect_checkpoint(checkpoint)
    assert inspected_first["pending_rows"] == 384 and inspected_first["commit_cursor"] == 1
    assert inspected_first["trainable_parameter_count"] == inspected_first["ema_parameter_count"] > 0
    inspection, saved = e2e._checkpoint_snapshot(checkpoint)
    state = saved.component_state("data_plane").payload["runtime_state"]
    assert state["committed_steps"] == state["commit_cursor"] == 1
    assert state["schema_version"] == 2
    assert "behavior" not in state and "sampling_fingerprint" not in state
    assert len(replay_calls) == 48
    assert all(len(rows) == 4 for rows in replay_calls)
    assert inspection.progress.ema_state_saved
    assert state["pool"]["execution"].trajectory.media is None
    assert state["pool"]["execution"].trajectory.batch_size == 384
    reward = state["pool"]["reward"]["weighted_total"]
    grouping = state["pool"]["advantage"].grouping
    expected = np.zeros(384)
    for group in dict.fromkeys(grouping.group_ids):
        rows = np.array([i for i, g in enumerate(grouping.group_ids) if g == group])
        expected[rows] = (reward[rows] - reward[rows].mean()) / (reward.std() + 1e-4)
    np.testing.assert_allclose(state["pool"]["advantage"].values.numpy(), expected, rtol=1e-5, atol=1e-6)
    assert saved.component_state("run_checkpoint_summary").payload.last_metrics["active_transition_count"] == 192
    resumed_rewards = demo._FakeRewardResourceFactory()
    resumed = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                            reward_resource_factory=resumed_rewards)
    result = resumed.run(config(tmp_path / "resume", artifacts, output, resume=checkpoint))
    assert result.committed_steps == 2
    assert resumed_rewards.resources[0].score_calls == 0
    assert result.last_metrics["active_transition_count"] == 192
    assert result.last_metrics["logprob_delta_max"] > 0
    assert result.last_metrics["gradient_norm_pre_clip"] > 0
    assert len(replay_calls) == 96  # 48 per commit, no behavior-policy prepass
    inspected_second = inspect_checkpoint(result.authoritative_checkpoint)
    assert inspected_second["pending_rows"] == 0 and inspected_second["commit_cursor"] == 0
    assert inspected_first["trainable_sha256"] != inspected_second["trainable_sha256"]
    assert inspected_first["ema_sha256"] == inspected_second["ema_sha256"]
    _, final = e2e._checkpoint_snapshot(result.authoritative_checkpoint)
    final_state = final.component_state("data_plane").payload["runtime_state"]
    assert final_state["pool"] is None
    assert final_state["committed_steps"] == 2
    final_inspection, _ = e2e._checkpoint_snapshot(result.authoritative_checkpoint)
    assert inspection.progress.source_cursors == final_inspection.progress.source_cursors
    assert dict(final_inspection.progress.source_cursors)["main"] == 48
    continuous = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                               reward_resource_factory=demo._FakeRewardResourceFactory())
    reference = continuous.run(config(tmp_path / "continuous", artifacts, tmp_path / "continuous-out"))
    e2e._assert_checkpoint_state_exact(e2e._checkpoint_snapshot(reference.authoritative_checkpoint),
                                      e2e._checkpoint_snapshot(result.authoritative_checkpoint),
                                      include_run_summary=False)
    assert len(replay_calls) == 192  # another 2 x 48 in continuous reference
    events = [json.loads(r.message)["event"] for r in caplog.records if r.name == "visual_rl.flash140"]
    assert events.count("collection_chunk_durable") == 48
    assert events.count("collection_start") == 2  # none on odd resume
    assert "checkpoint_durable" in events and "optimizer_commit" in events

    # Same real controller/PPO/std/resume regression at SMALL40's 32 ->16+16.
    replay_calls.clear()
    small_out = tmp_path / "small-out"
    small_rewards = demo._FakeRewardResourceFactory()
    small = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                          reward_resource_factory=small_rewards)
    observe_checkpoints(small, pause_after=1)
    with pytest.raises(SafePointPause):
        small.run(config(tmp_path / "small-first", artifacts, small_out, rows=16))
    assert small_rewards.resources[0].score_calls == 2
    assert len(replay_calls) == 4  # four row4 slots, not an extra prepass
    _, small_saved = e2e._checkpoint_snapshot(small_out / "checkpoints/step-1")
    small_state = small_saved.component_state("data_plane").payload["runtime_state"]
    small_pool = small_state["pool"]
    assert small_pool["execution"].trajectory.batch_size == 32
    reward = small_pool["reward"]["weighted_total"]
    expected = np.zeros(32)
    grouping = small_pool["advantage"].grouping
    for group in dict.fromkeys(grouping.group_ids):
        indices = np.array([i for i, g in enumerate(grouping.group_ids) if g == group])
        expected[indices] = (reward[indices] - reward[indices].mean()) / (reward.std() + 1e-4)
    np.testing.assert_allclose(small_pool["advantage"].values.numpy(), expected, rtol=1e-5, atol=1e-6)
    resumed_rewards = demo._FakeRewardResourceFactory()
    small_resume = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                 reward_resource_factory=resumed_rewards)
    final_small = small_resume.run(config(tmp_path / "small-resume", artifacts, small_out, rows=16,
                                           resume=small_out / "checkpoints/step-1"))
    assert resumed_rewards.resources[0].score_calls == 0
    assert final_small.last_metrics["active_transition_count"] == 16
    assert final_small.last_metrics["logprob_delta_max"] > 0
    assert len(replay_calls) == 8
    continuous_small = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                      reward_resource_factory=demo._FakeRewardResourceFactory())
    reference_small = continuous_small.run(config(tmp_path / "small-continuous", artifacts,
                                                   tmp_path / "small-continuous-out", rows=16))
    e2e._assert_checkpoint_state_exact(e2e._checkpoint_snapshot(reference_small.authoritative_checkpoint),
                                      e2e._checkpoint_snapshot(final_small.authoritative_checkpoint),
                                      include_run_summary=False)


def test_collection_disk_chunks_survive_failure_before_first_commit(monkeypatch, tmp_path, caplog):
    from visual_rl.algorithms.trainer.collection import CollectionState
    artifacts, code = setup(monkeypatch, tmp_path)
    fingerprints = []
    begin = CollectionState.begin
    def observed_begin(self):
        begin(self)
        fingerprints.append(self.sampling_fingerprint)
    monkeypatch.setattr(CollectionState, "begin", observed_begin)
    first_rewards = demo._FakeRewardResourceFactory()
    original = demo._FakeRewardResource.score
    def score_or_fail(self, *, batch):
        if self.score_calls == 3:
            raise RuntimeError("during fourth reward chunk")
        return original(self, batch=batch)
    monkeypatch.setattr(demo._FakeRewardResource, "score", score_or_fail)
    output = tmp_path / "out"
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=first_rewards)
    with pytest.raises(RuntimeError, match="fourth reward chunk"):
        controller.run(config(tmp_path / "first", artifacts, output))
    assert len(list((output / "collection-cache/collection-0").glob("*.pt"))) == 3
    assert not (output / "SUCCESS").exists()
    monkeypatch.setattr(demo._FakeRewardResource, "score", original)
    def changed_begin(self):
        with torch.no_grad():
            self.parameters[0].add_(0.5)
        observed_begin(self)
    monkeypatch.setattr(CollectionState, "begin", changed_begin)
    changed_rewards = demo._FakeRewardResourceFactory()
    changed = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=changed_rewards)
    with pytest.raises(ValueError, match="disk chunk behavior/row identity mismatch"):
        changed.run(config(tmp_path / "changed", artifacts, output))
    assert changed_rewards.resources[0].score_calls == 0
    assert fingerprints[0] != fingerprints[1]
    monkeypatch.setattr(CollectionState, "begin", observed_begin)
    rewards = demo._FakeRewardResourceFactory()
    resumed = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=rewards)
    result = resumed.run(config(tmp_path / "retry", artifacts, output))
    assert result.committed_steps == 2
    assert fingerprints[0] == fingerprints[2]
    assert rewards.resources[0].score_calls == 21  # 48 logical videos already durable
    assert not (output / "collection-cache/collection-0").exists()
    attempts = list((output / "physical-attempts/collection-0").glob("*.json"))
    assert len(attempts) == 24
    assert sum(json.loads(p.read_text())["physical_attempts_started"] for p in attempts) == 25

    # The supported resource-factory seam runs after managed rollout exit, with
    # detached media and no autograd at empty_cache. CUDA counters here are
    # explicit fake leaves: ordering/logging evidence, NOT reclaimed GPU bytes.
    from contextlib import contextmanager
    from dataclasses import replace
    from scripts.flash_small40_repair import CacheHandoffRewardFactory, CacheHandoffHpsClient
    from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract
    from visual_rl.models.lifecycle.components import ComponentManager
    caplog.set_level("INFO", logger="visual_rl.flash140")
    active = []
    released = []
    managed = ComponentManager.execution
    @contextmanager
    def tracking(self, policy):
        with managed(self, policy):
            active.append(policy)
            try:
                yield
            finally:
                active.pop()
    monkeypatch.setattr(ComponentManager, "execution", tracking)
    # Fake CUDA is ONLY active inside release_reward_cache; RNG capture must
    # continue to observe the actual CPU-only runtime.
    from scripts import flash_small40_repair as repair
    release = repair.release_reward_cache
    def handoff(phase):
        assert not active
        with monkeypatch.context() as cuda:
            cuda.setattr(torch.cuda, "synchronize", lambda _: None)
            cuda.setattr(torch.cuda, "mem_get_info", lambda _: (10, 100))
            cuda.setattr(torch.cuda, "memory_allocated", lambda _: 20)
            cuda.setattr(torch.cuda, "memory_reserved", lambda _: 60)
            def empty():
                assert not torch.is_grad_enabled()
                released.append(phase)
            cuda.setattr(torch.cuda, "empty_cache", empty)
            release(phase)
    def enabled_handoff(phase):
        with monkeypatch.context() as cuda:
            cuda.setattr(torch.cuda, "is_available", lambda: True)
            handoff(phase)
    monkeypatch.setattr(repair, "release_reward_cache", enabled_handoff)
    class FakeHps(demo._FakeRewardResource, Hpsv3VideoClient):
        fail = False
        def __init__(self):
            demo._FakeRewardResource.__init__(self)
            self.service_identity, self.contract = "1" * 64, VideoContract(5, 2, 2)
        def score(self, *, batch):
            assert released[-1] == "collection_chunk" and not active
            if self.fail:
                raise RuntimeError("ordinary HPS503")
            return original(self, batch=batch)
        def score_video(self, **kwargs):
            assert released[-1] == "evaluation_video"
            return 0.5, {"real_leaf": False}
    clients = []
    facts = demo._FakeRewardResourceFactory()
    def leaf(request):
        acquired = facts(request)
        acquired.resource.close()
        client = FakeHps()
        clients.append(client)
        return replace(acquired, resource=client)
    wrapped_factory = CacheHandoffRewardFactory(leaf)
    before = len(released)
    cache_controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                      reward_resource_factory=wrapped_factory)
    cache_result = cache_controller.run(config(tmp_path / "handoff", artifacts, tmp_path / "handoff-out", rows=16))
    assert len(released) - before == 2  # one release per generated16 chunk, not per optimizer/slot
    assert clients[-1].score_calls == 2 and clients[-1].close_calls == 1
    assert clients[-1].activate_calls == 1
    baseline = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                               reward_resource_factory=demo._FakeRewardResourceFactory())
    baseline_result = baseline.run(config(tmp_path / "handoff-baseline", artifacts, tmp_path / "handoff-reference", rows=16))
    e2e._assert_checkpoint_state_exact(e2e._checkpoint_snapshot(cache_result.authoritative_checkpoint),
                                      e2e._checkpoint_snapshot(baseline_result.authoritative_checkpoint),
                                      include_run_summary=False)
    FakeHps.fail = True
    failing = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                            reward_resource_factory=wrapped_factory)
    with pytest.raises(RuntimeError, match="ordinary HPS503"):
        failing.run(config(tmp_path / "handoff-failed", artifacts, tmp_path / "handoff-failed-out", rows=16))
    assert not (tmp_path / "handoff-failed-out/SUCCESS").exists()
    assert clients[-1].close_calls == 1
    adapted = CacheHandoffHpsClient(FakeHps())
    assert isinstance(adapted, Hpsv3VideoClient)
    assert adapted.score_video(sample_id="fake") == (0.5, {"real_leaf": False})
    adapted.close()
    events = [json.loads(r.message) for r in caplog.records if r.name == "visual_rl.flash140"]
    before_events = [r for r in events if r["event"] == "reward_cuda_handoff_before"]
    after_events = [r for r in events if r["event"] == "reward_cuda_handoff_after"]
    assert len(before_events) == len(after_events) == len(released)
    assert all(e["allocated_bytes"] == 20 and e["reserved_bytes"] == 60 for e in after_events)


def test_step0_gate_media_release_and_resumed_step1_safety(monkeypatch, tmp_path):
    import gc
    from dataclasses import replace
    from visual_rl.algorithms.optimization.recompute import PolicyRecomputer
    from visual_rl.algorithms.rewards.clients.input import resolve_pointwise_reward_input
    artifacts, code = setup(monkeypatch, tmp_path)
    media_refs = []
    score = demo._FakeRewardResource.score
    def observed_score(self, *, batch):
        media_refs.append(weakref.ref(resolve_pointwise_reward_input(batch).media))
        return score(self, batch=batch)
    monkeypatch.setattr(demo._FakeRewardResource, "score", observed_score)
    compute = PolicyRecomputer.compute_current_slot
    mode = "initial-mismatch"
    def injected(self, request, slot, **kwargs):
        if mode == "initial-mismatch":
            # Finite but wrong saved behavior probability; step0 must reject.
            request.rollout.trajectory.old_log_probs.add_(0.05)
        stats = compute(self, request, slot, **kwargs)
        if mode == "nan":
            return replace(stats, current_log_probs=stats.current_log_probs * float("nan"))
        if mode == "later-ratio":
            # Beyond both the initial tolerance and PPO clip; retains a real
            # differentiable forward and unclipped negative-advantage gradients.
            return replace(stats, current_log_probs=stats.current_log_probs + 0.01)
        return stats
    monkeypatch.setattr(PolicyRecomputer, "compute_current_slot", injected)
    output = tmp_path / "bad-initial"
    def controller(rewards=None):
        return create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
            reward_resource_factory=rewards or demo._FakeRewardResourceFactory())
    with pytest.raises(RuntimeError, match="on-policy old/current ratio parity gate"):
        controller().run(config(tmp_path / "first", artifacts, output))
    assert len(media_refs) == 24
    gc.collect()
    assert all(ref() is None for ref in media_refs)
    assert not (output / "SUCCESS").exists()
    assert not (output / "checkpoints/step-1").exists()

    mode = "normal"
    output = tmp_path / "resume-out"
    first = controller()
    sink = first._backend.checkpoint_sink
    checkpoint_safe_point = sink.checkpoint_safe_point
    def stop(request):
        receipt = checkpoint_safe_point(request)
        raise RuntimeError("after durable step1")
    sink.checkpoint_safe_point = stop
    with pytest.raises(RuntimeError, match="after durable step1"):
        first.run(config(tmp_path / "durable", artifacts, output))
    checkpoint = output / "checkpoints/step-1"
    mode = "nan"
    nan_rewards = demo._FakeRewardResourceFactory()
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        controller(nan_rewards).run(config(tmp_path / "nan", artifacts, output, resume=checkpoint))
    assert nan_rewards.resources[0].score_calls == 0
    assert not (output / "checkpoints/step-2").exists()
    assert not (output / "SUCCESS").exists()
    mode = "later-ratio"
    resumed_rewards = demo._FakeRewardResourceFactory()
    result = controller(resumed_rewards).run(config(tmp_path / "valid", artifacts, output, resume=checkpoint))
    assert result.committed_steps == 2
    assert resumed_rewards.resources[0].score_calls == 0
    assert result.last_metrics["logprob_delta_max"] > 1e-4
    assert result.last_metrics["clipfrac"] > 0
    assert np.isfinite(result.last_metrics["gradient_norm_pre_clip"])
    assert result.last_metrics["gradient_norm_pre_clip"] > 0
    gc.collect()
    assert all(ref() is None for ref in media_refs)


def test_cpu_ema_updates_only_each_eighth_commit_and_restores_on_error():
    from visual_rl.algorithms.trainer.collection import CollectionState
    from visual_rl.algorithms.trainer.config import GRPOTrainerConfig
    from visual_rl.runtime.checkpoint_binding import _validate_data_payload_schema
    parameter = torch.nn.Parameter(torch.tensor(0.))
    state = CollectionState(config=GRPOTrainerConfig(192, 2, 16), parameters=(parameter,),
                            request_factory=None)
    original_id = id(parameter)
    with torch.no_grad():
        parameter.fill_(8.)
    for _ in range(7):
        state.commit()
    assert state.ema[0].item() == 0
    state.commit()
    assert state.ema[0].item() == pytest.approx(8 * (1 - 8 / 17))
    with pytest.raises(RuntimeError):
        with state.parameter_context(state.ema):
            assert parameter.item() != 8
            raise RuntimeError("evaluation failed")
    assert parameter.item() == 8
    assert id(state.parameters[0]) == original_id
    state.begin()
    saved = state.state_dict()
    assert saved["schema_version"] == 2
    assert "behavior" not in saved and "sampling_fingerprint" not in saved
    state.load_state_dict(saved)
    assert state.sampling_fingerprint is None
    assert id(state.parameters[0]) == original_id
    assert state.committed_steps == 8 and state.commit_cursor == 0
    assert torch.equal(state.ema[0], saved["ema"][0])
    outer = dict(schema_version=1, kind="data_plane_prelude_state", prelude_id="test",
                 placement_contract_id="test", phase_schedule_state={}, sampler_state={}, runtime_state=saved)
    _validate_data_payload_schema(outer)
    old = dict(saved, schema_version=1, behavior=[parameter.detach().clone()])
    with pytest.raises(ValueError, match="schema"):
        state.load_state_dict(old)
    with pytest.raises(ValueError, match="collection runtime state"):
        _validate_data_payload_schema(dict(outer, runtime_state=old))
    for bad in (dict(saved, commit_cursor=1), dict(saved, committed_steps=9, commit_cursor=1),
                dict(saved, pool={}), dict(saved, commit_cursor=False)):
        with pytest.raises(ValueError, match="cursor|pool"):
            state.load_state_dict(bad)


def test_real_ode_evaluation_stage_runs300x50_and_reuses_durable_receipts(monkeypatch, tmp_path):
    from dataclasses import replace
    from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract
    from visual_rl.algorithms.trainer import evaluation
    from visual_rl.algorithms.rollout.collector import RolloutCollector
    from visual_rl.artifacts.checkpoint import RankRNGSnapshot
    artifacts, code = setup(monkeypatch, tmp_path)
    evaluation_calls = []
    evaluate = evaluation.FlashEvaluationStage.__call__
    def checked_evaluation(self, step):
        before_rng = RankRNGSnapshot.capture_current(0)
        parameters = self.collection.parameters
        before_weights = [p.detach().clone() for p in parameters]
        try:
            return evaluate(self, step)
        finally:
            assert RankRNGSnapshot.capture_current(0) == before_rng
            assert all(p is old for p, old in zip(self.collection.parameters, parameters, strict=True))
            assert all(torch.equal(p, old) for p, old in zip(parameters, before_weights, strict=True))
            evaluation_calls.append(step)
    monkeypatch.setattr(evaluation.FlashEvaluationStage, "__call__", checked_evaluation)
    # Explicit fake leaves only. Production loader enforces the author SHA.
    monkeypatch.setattr(evaluation, "load_evaluation_prompts", lambda path: tuple(f"eval prompt {i}" for i in range(300)))
    class Client(Hpsv3VideoClient):
        fail = True
        def __init__(self):
            self.service_identity = "1" * 64
            self.contract = VideoContract(5, 2, 2)
            self.score_calls = self.eval_calls = self.close_calls = self.activate_calls = 0
        activate = demo._FakeRewardResource.activate
        close = demo._FakeRewardResource.close
        score = demo._FakeRewardResource.score
        def score_video(self, *, sample_id, prompt, frames_base64):
            import random
            random.random()
            np.random.random()
            torch.rand(1)  # exercise global RNG restoration, including failure
            if type(self).fail and self.eval_calls == 3:
                raise RuntimeError("evaluation interruption")
            assert len(frames_base64) == 5
            self.eval_calls += 1
            return 0.5, {"scored_frame_indices": list(range(5)), "selected_frame_indices": [0], "frame_scores": [0.5] * 5}
    clients = []
    base_factory = demo._FakeRewardResourceFactory()
    def factory(request):
        acquired = base_factory(request)
        client = Client()
        clients.append(client)
        return replace(acquired, resource=client)
    steps = []
    original = RolloutCollector.evaluate_deterministic
    def counted(self, request, *, num_steps):
        steps.append(num_steps)
        return original(self, request, num_steps=num_steps)
    monkeypatch.setattr(RolloutCollector, "evaluate_deterministic", counted)
    output = tmp_path / "out"
    def path(name):
        p = config(tmp_path / name, artifacts, output)
        raw = yaml.safe_load(p.read_text())
        raw["overrides"]["training"]["flash_evaluation"] = True
        p.write_text(yaml.safe_dump(raw))
        return p
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=factory)
    with pytest.raises(RuntimeError, match="evaluation interruption"):
        controller.run(path("first"))
    assert len(list((output / "evaluations/step-0").glob("video-*.json"))) == 3
    Client.fail = False
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(), reward_resource_factory=factory)
    result = controller.run(path("retry"))
    assert clients[-1].eval_calls == 297
    assert evaluation_calls == [0, 0]
    assert steps == [50] * 301  # one physical failed RPC attempt, 300 logical IDs
    summary = json.loads((output / "evaluations/step-0/summary.json").read_text())
    assert summary["videos_scored"] == 300
    assert summary["physical_video_attempts_started"] == 301
    _, snapshot = e2e._checkpoint_snapshot(result.authoritative_checkpoint)
    assert snapshot.component_state("data_plane").payload["runtime_state"]["evaluations_completed"] == [0]

    # Real SMALL40 dispatch/controller/terminal hook; only heavy model geometry,
    # scorer and MP4 encoder are explicit CPU fake leaves. Production guards,
    # author source hashes, fixed selection, 40 commits and 0/20/40 stay real.
    from visual_rl.models.implementations.wan import WanT2VAdapter
    from visual_rl.algorithms.trainer.flash_presets import SMALL40
    from scripts.launch_flash140 import SMALL_CONFIG
    author = _author_data_dir()
    assert author.is_dir()  # task's local author data, not downloaded fixtures
    # Restore the actual loader used in production (the earlier300 case used synthetic prompts).
    monkeypatch.undo()
    e2e._patch_default_runtime(monkeypatch)
    monkeypatch.setattr(evaluation.FlashEvaluationStage, "__call__", checked_evaluation)
    tiny_config = lambda cfg: replace(cfg, height=16, width=16, frames=5)
    latent = WanT2VAdapter.latent_spec_for_batch
    validate = WanT2VAdapter._validate_latent_spec
    def tiny_latent(self, *args, **kwargs):
        original_config = self.config
        self.config = tiny_config(original_config)
        try:
            return latent(self, *args, **kwargs)
        finally:
            self.config = original_config
    def tiny_validate(self, *args, **kwargs):
        original_config = self.config
        self.config = tiny_config(original_config)
        try:
            return validate(self, *args, **kwargs)
        finally:
            self.config = original_config
    monkeypatch.setattr(WanT2VAdapter, "latent_spec_for_batch", tiny_latent)
    monkeypatch.setattr(WanT2VAdapter, "_validate_latent_spec", tiny_validate)
    saved_media = []
    def fake_encoder(path, media, contract):
        assert media.tensor.shape[1] == contract.frames == 81  # not21 latent frames
        assert media.tensor.shape[-2:] == (2, 2)  # explicit CPU fake model
        saved_media.append(weakref.ref(media.tensor))
        path.write_bytes(b"CPU fake MP4 encoder output")
    monkeypatch.setattr(evaluation, "write_evaluation_video", fake_encoder)
    evaluation_calls.clear()
    steps.clear()
    seeds = []
    def small_counted(self, request, *, num_steps):
        seeds.append((request.samples.prompts[0], request.generator.initial_seed()))
        steps.append(num_steps)
        return original(self, request, num_steps=num_steps)
    monkeypatch.setattr(RolloutCollector, "evaluate_deterministic", small_counted)
    class SmallClient(Client):
        fail_terminal = True
        def __init__(self):
            super().__init__()
            self.contract = VideoContract(81, 2, 2)
            self.terminal_calls = 0
        def score_video(self, *, sample_id, prompt, frames_base64):
            assert len(frames_base64) == 81
            if "step-40-" in sample_id:
                if type(self).fail_terminal and self.terminal_calls == 3:
                    raise RuntimeError("terminal40 evaluation interruption")
                self.terminal_calls += 1
            self.eval_calls += 1
            # Descriptive nonconstant values permit checking paired differences.
            score = 0.4 + int(sample_id.split("step-")[1].split("-")[0]) / 1000
            return score, {"scored_frame_indices": list(range(81)),
                           "selected_frame_indices": list(range(24)), "frame_scores": [score] * 81}
    small_clients = []
    def small_factory(request):
        acquired = base_factory(request)
        client = SmallClient()
        small_clients.append(client)
        return replace(acquired, resource=client)
    small_out = tmp_path / "small40-out"
    def small_path(name, resume=None):
        p = config(tmp_path / name, artifacts, small_out, steps=40, rows=16, resume=resume)
        raw = yaml.safe_load(p.read_text())
        approved = yaml.safe_load(SMALL_CONFIG.read_text())["overrides"]
        raw["overrides"]["training"] = approved["training"]
        raw["overrides"]["algorithm"]["params"].update(approved["algorithm"]["params"])
        raw["overrides"]["model"]["params"].update(approved["model"]["params"])
        raw["launch"]["checkpoint_every_optimizer_steps"] = 2
        raw["launch"]["artifacts"]["datasets"]["main"] = str(author / "train.txt")
        raw["launch"]["artifacts"]["evaluation_prompts"] = str(author / "test.txt")
        p.write_text(yaml.safe_dump(raw))
        return p
    from scripts.launch_flash140 import SafePointPause, observe_checkpoints
    from scripts.flash_small40_repair import CacheHandoffRewardFactory, validate_small40_resume
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    observe_checkpoints(controller, pause_after=2)  # test-only interruption, production RUN remains40
    with pytest.raises(SafePointPause):
        controller.run(small_path("small40-first"))
    assert small_clients[0].score_calls == 2 and small_clients[0].eval_calls == 16
    step2 = small_out / "checkpoints/step-2"
    restored = validate_small40_resume(small_out, step2)
    assert restored["committed_steps"] == 2 and restored["evaluations_completed"] == [0]
    latest_file = step2.parent / "latest.json"
    latest_bytes = latest_file.read_bytes()
    latest_file.write_text(json.dumps(dict(json.loads(latest_bytes), step=0)))
    with pytest.raises(RuntimeError, match="latest.json"):
        validate_small40_resume(small_out, step2)
    latest_file.write_bytes(latest_bytes)
    newer = step2.parent / "step-4"
    newer.mkdir()
    with pytest.raises(RuntimeError, match="latest checkpoint"):
        validate_small40_resume(small_out, step2)
    newer.rmdir()  # test-only empty collision, never a real checkpoint
    baseline_receipts = {p.name: p.read_bytes() for p in (small_out / "evaluations/step-0").iterdir()}
    # Actual immutable contract decoding rejects a changed code artifact; no
    # owner upgrade can suppress the production prepared/full restore checks.
    changed_code = tmp_path / "changed-code"
    changed_code.mkdir()
    (changed_code / "identity.py").write_text("DIFFERENT = True\n")
    incompatible = create_default_run_controller(code_root=changed_code, model_loader=demo._FakeModelLoader(),
                                                  reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    from visual_rl.errors import ResumeError
    with pytest.raises(ResumeError, match="recipe_id differs"):
        incompatible.run(small_path("small40-wrong-source", step2))
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    with pytest.raises(RuntimeError, match="terminal40 evaluation interruption"):
        controller.run(small_path("small40-resume2", step2))
    assert {p.name: p.read_bytes() for p in (small_out / "evaluations/step-0").iterdir()} == baseline_receipts
    assert not (small_out / "SUCCESS").exists()
    assert not (small_out / "checkpoints/step-40").exists()
    assert (small_out / "checkpoints/step-38").is_dir()
    assert small_clients[0].score_calls + small_clients[-1].score_calls == 40  # no recollection at step2
    assert evaluation_calls == [0, 20, 40]
    assert steps == [20] * 36  #16+16+3 durable+1 interrupted
    assert [seed for _, seed in seeds[:16]] == list(SMALL40.noise_seeds)
    assert seeds[:16] == seeds[16:32]  # identical prompt noise across EMA rounds
    assert len(list((small_out / "evaluations/step-40").glob("video-*.json"))) == 3
    SmallClient.fail_terminal = False
    controller = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                                reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    small_result = controller.run(small_path("small40-retry", small_out / "checkpoints/step-38"))
    assert small_result.committed_steps == 40 and (small_out / "SUCCESS").is_file()
    assert evaluation_calls == [0, 20, 40, 40]  # no baseline/step20 redo
    assert small_clients[-1].eval_calls == 13  #3 terminal receipts reused
    assert steps == [20] * 49  #48 logical videos +1 interrupted physical RPC
    small_inspection, small_snapshot = e2e._checkpoint_snapshot(small_result.authoritative_checkpoint)
    assert dict(small_inspection.progress.source_cursors)["main"] == 80  #20 x4 prompts, no extra logical collection
    terminal_state = small_snapshot.component_state("data_plane").payload["runtime_state"]
    assert terminal_state["evaluations_completed"] == [0, 20, 40]
    assert terminal_state["committed_steps"] == 40 and terminal_state["commit_cursor"] == 0
    assert terminal_state["pool"] is None
    assert sorted(int(p.name.removeprefix("step-")) for p in (small_out / "checkpoints").iterdir()
                  if p.name.removeprefix("step-").isdigit()) == list(range(2, 41, 2))
    for step in (0, 20, 40):
        root = small_out / f"evaluations/step-{step}"
        assert len(list(root.glob("*.mp4"))) == 16
        summary = json.loads((root / "summary.json").read_text())
        assert summary["sampling_steps"] == 20 and summary["noise_seeds"] == list(SMALL40.noise_seeds)
        assert summary["decoded_frames"] == 81 and summary["fps"] == 15
        assert summary["videos_scored"] == len(summary["per_prompt"]) == 16
        assert summary["physical_video_attempts_started"] == (17 if step == 40 else 16)
        if step:
            assert summary["paired_differences"]["step-0"]["mean"] == pytest.approx(step / 1000)
    import gc
    gc.collect()
    assert all(ref() is None for ref in saved_media)
    # Fresh continuous reference: exact prepared owners/RNG/sampler/EMA/cursors
    # versus the actual step2 continuation (and later terminal receipt retry).
    resumed_out = small_out
    small_out = tmp_path / "small40-continuous"
    reference = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                              reward_resource_factory=CacheHandoffRewardFactory(small_factory))
    reference_result = reference.run(small_path("small40-continuous-config"))
    e2e._assert_checkpoint_state_exact(e2e._checkpoint_snapshot(reference_result.authoritative_checkpoint),
                                      e2e._checkpoint_snapshot(small_result.authoritative_checkpoint),
                                      include_run_summary=False)
    small_out = resumed_out
    # Even already-completed earlier rounds must retain their videos before
    # finalization; a saved evaluations_completed cursor is not enough.
    terminal_summary = json.loads((small_out / "evaluations/step-40/summary.json").read_text())
    evaluation.validate_small40_evaluation_artifacts(small_out, terminal_summary["recipe_id"])
    baseline_video = small_out / "evaluations/step-0/video-000-seed-42000.mp4"
    baseline_bytes = baseline_video.read_bytes()
    baseline_video.unlink()
    with pytest.raises(ValueError, match="video missing/corrupt"):
        evaluation.validate_small40_evaluation_artifacts(small_out, terminal_summary["recipe_id"])
    baseline_video.write_bytes(baseline_bytes)
    # Existing receipt without its associated MP4 must never be silently reused.
    # Reuse the terminal evaluator before close is not possible; exercise the
    # same real journal path by another even-checkpoint restore.
    (small_out / "evaluations/step-40/video-000-seed-42000.mp4").unlink()
    stale = create_default_run_controller(code_root=code, model_loader=demo._FakeModelLoader(),
                                          reward_resource_factory=small_factory)
    stale_out = tmp_path / "stale-small40"
    # Clone only evaluation journals to a new owned test output (no SUCCESS).
    import shutil
    shutil.copytree(small_out / "evaluations", stale_out / "evaluations")
    original_out = small_out
    small_out = stale_out
    with pytest.raises(ValueError, match="video missing/corrupt"):
        stale.run(small_path("stale-small40-case", original_out / "checkpoints/step-38"))
    assert not (stale_out / "SUCCESS").exists()


def test_approved140_candidate_compiles_typed_cadence(monkeypatch, tmp_path):
    from scripts import launch_flash140 as launch
    with pytest.raises(SystemExit) as help_result:
        launch.main(["--help"])
    assert help_result.value.code == 0
    from visual_rl.composition.config.compiler import compile_recipe_v2
    from visual_rl.composition.config.source import load_source_recipe
    from visual_rl.composition.config.bootstrap import bootstrap_recipe_v2
    e2e._patch_default_runtime(monkeypatch)
    path = Path(__file__).parents[1] / "experiments/flash_wan_capacity_20261022/approved140-runtime.yaml"
    source = load_source_recipe(path)
    recipe = compile_recipe_v2(source)
    assert recipe.training.collection.collection_rows_per_commit == 192
    assert recipe.training.collection.collection_commits == 2
    assert recipe.training.collection.collection_chunk_rows == 16
    assert recipe.execution_policy.group_size == 8
    assert recipe.execution_policy.rollout.forward_microbatch_size == 4
    assert recipe.execution_policy.rollout.decode_microbatch_size == 1
    assert recipe.training.policy_recompute.row_microbatch_size == 4
    assert recipe.training.policy_recompute.transition_window_size == 1
    assert recipe.training.update_safety.max_initial_logprob_delta == 1e-4
    assert recipe.training.update_safety.require_initial_clipfrac_zero
    assert recipe.training.global_prompt_batch_size * recipe.execution_policy.group_size == 384
    assert recipe.training.max_optimizer_steps == 140
    assert recipe.training.flash_evaluation
    assert bootstrap_recipe_v2(source).require_launch().artifacts.evaluation_prompts.name == "test.txt"
    raw = yaml.safe_load(path.read_text())
    rendered = launch.render_config(raw, tmp_path / "run", tmp_path / "run/checkpoints/step-1")
    assert rendered["overrides"] == raw["overrides"]
    assert rendered["launch"]["artifacts"] == raw["launch"]["artifacts"]
    assert raw["launch"]["resume_from"] is None
    assert rendered["overrides"]["training"]["max_optimizer_steps"] == 140
    launch_path = tmp_path / "rendered.yaml"
    launch_path.write_text(yaml.safe_dump(rendered))
    assert compile_recipe_v2(load_source_recipe(launch_path)).training == recipe.training
    small_path = launch.preset_config("small40")
    assert launch.preset_config("approved140") == path.resolve()
    with pytest.raises(ValueError, match="unknown"):
        launch.preset_config("unapproved")
    small_source = load_source_recipe(small_path)
    small_recipe = compile_recipe_v2(small_source)
    small_raw = yaml.safe_load(small_path.read_text())
    assert small_recipe.training.flash_preset == "small40"
    assert small_recipe.training.max_optimizer_steps == 40
    assert small_recipe.training.global_prompt_batch_size == 4
    assert small_recipe.training.collection.collection_rows_per_commit == 16
    assert small_recipe.training.collection.collection_commits == 2
    assert small_recipe.training.collection.collection_chunk_rows == 16
    assert small_recipe.training.flash_evaluation
    assert small_recipe.training.policy_recompute == recipe.training.policy_recompute
    assert small_recipe.training.update_safety == recipe.training.update_safety
    assert small_recipe.execution_policy == recipe.execution_policy
    assert small_raw["overrides"]["model"] == raw["overrides"]["model"]
    assert small_raw["overrides"]["algorithm"] == raw["overrides"]["algorithm"]
    assert small_raw["overrides"]["rewards"] == raw["overrides"]["rewards"]
    assert small_raw["launch"]["artifacts"] == raw["launch"]["artifacts"]
    assert small_raw["launch"]["checkpoint_every_optimizer_steps"] == 2
    small_render = launch.render_config(small_raw, tmp_path / "small40-run", None)
    assert small_render["overrides"] == small_raw["overrides"]
    launch.check_preset_mode("small40", "run", None, False)
    launch.check_preset_mode("small40", "preflight", None, False)
    for mode in ("preflight", "run", "inspect"):
        launch.check_preset_mode("small40", mode, Path("owned/checkpoints/step-2"), False)
    for mode, resume, retry in (("validate", None, False), ("run", None, True)):
        with pytest.raises(RuntimeError, match="explicit owned latest"):
            launch.check_preset_mode("small40", mode, resume, retry)
    resumed_render = launch.render_config(small_raw, tmp_path / "small40-run", Path("owned/checkpoints/step-2"))
    assert resumed_render["overrides"] == small_raw["overrides"]
    assert resumed_render["launch"]["resume_from"].endswith("owned/checkpoints/step-2")
    import copy
    from visual_rl.errors import ConfigError
    for target in (2, 39, 140):
        bad = copy.deepcopy(small_raw)
        bad["overrides"]["training"]["max_optimizer_steps"] = target
        with pytest.raises(RuntimeError, match="endpoint"):
            launch.render_config(bad, tmp_path / "bad", None)
        p = tmp_path / f"invalid-small-{target}.yaml"
        p.write_text(yaml.safe_dump(bad))
        with pytest.raises(ConfigError, match="SMALL40"):
            compile_recipe_v2(load_source_recipe(p))
    safe = {"gpu_free_gib": 68.03, "host_available_gib": 88, "disk_free_gib": 1024}
    launch.check_budget(safe, admission=True, disk_required_gib=1024)
    for key in safe:
        with pytest.raises(RuntimeError, match="resource safety"):
            launch.check_budget({**safe, key: safe[key] - 0.01}, admission=True, disk_required_gib=1024)
    launch.check_budget(dict(gpu_free_gib=16, host_available_gib=24, disk_free_gib=64))
    with pytest.raises(RuntimeError, match="resource safety"):
        launch.check_budget({**safe, "gpu_free_gib": float("nan")})
    monitor = launch.ResourceMonitor(tmp_path, tmp_path / "resources.json", gpu_reserved_limit=60)
    observed = {**safe, "torch_peak_reserved_gib": 52.03, "process_peak_rss_gib": 10}
    monkeypatch.setattr(launch, "resource_sample", lambda _: observed)
    monitor.sample()  # explicit CPU sample leaf, never described as GPU evidence
    assert json.loads(monitor.report.read_text())["sampled_minima"]["gpu_free_gib"] == 68.03
    monkeypatch.setattr(launch, "resource_sample", lambda _: {**observed, "gpu_free_gib": 15})
    with pytest.raises(RuntimeError, match="resource safety"):
        monitor.sample()
    assert json.loads(monitor.report.read_text())["sampled_minima"]["gpu_free_gib"] == 15
    monkeypatch.setattr(launch, "resource_sample", lambda _: {**observed, "torch_peak_reserved_gib": 61})
    with pytest.raises(RuntimeError, match="peak reservation"):
        monitor.sample()
    # Explicit user-approved GPU policy: no headroom or reserved-peak abort;
    # host/disk and invalid telemetry still fail, and CUDA errors aren't caught.
    launch.check_budget({**safe, "gpu_free_gib": 0}, admission=True, gpu_memory_policy="oom")
    oom_monitor = launch.ResourceMonitor(tmp_path, tmp_path / "oom-resources.json",
                                       gpu_reserved_limit=60, gpu_memory_policy="oom")
    monkeypatch.setattr(launch, "resource_sample", lambda _: {**observed, "gpu_free_gib": 1,
                                                            "torch_peak_reserved_gib": 77})
    oom_monitor.sample()
    assert oom_monitor.peaks["torch_peak_reserved_gib"] == 77
    for key, low in (("host_available_gib", 23), ("disk_free_gib", 63)):
        with pytest.raises(RuntimeError, match="resource safety"):
            launch.check_budget({**safe, key: low}, gpu_memory_policy="oom")
    with pytest.raises(RuntimeError, match="invalid GPU"):
        launch.check_budget({**safe, "gpu_free_gib": float("nan")}, gpu_memory_policy="oom")
    lock = tmp_path / "launch.lock"
    with launch.exclusive_launch(lock, {"pid": 123}):
        with pytest.raises(RuntimeError, match="duplicate"):
            with launch.exclusive_launch(lock, {"pid": 456}):
                pytest.fail("second owner entered")
    with launch.exclusive_launch(lock, {"pid": 789}):
        assert json.loads(lock.read_text())["pid"] == 789
    identity = {"git_sha": "a" * 40, "source_config_sha256": "b" * 64}
    real_root = launch.ROOT
    rejected_output = real_root / "test-output-must-not-be-created"
    assert not rejected_output.exists()
    with pytest.raises(RuntimeError, match="outside the clean source checkout"):
        launch.prepare_output(rejected_output, identity=identity)
    assert not rejected_output.exists()
    # Workflow tmp_path may itself be inside the real checkout. Model the
    # deployment's sibling source/output layout only for output admission.
    source_root = tmp_path / "source-checkout"
    with monkeypatch.context() as output_patch:
        output_patch.setattr(launch, "ROOT", source_root)
        for rejected in (source_root, source_root / "nested-output"):
            with pytest.raises(RuntimeError, match="outside the clean source checkout"):
                launch.prepare_output(rejected, identity=identity)
            assert not rejected.exists()
        assert not source_root.exists()  # rejection creates neither root nor owner
        out = launch.prepare_output(tmp_path / "owned", identity=identity)
        with pytest.raises(FileExistsError):
            launch.prepare_output(out, identity=identity)
        launch.prepare_output(out, identity=identity, retry_incomplete=True)
        with pytest.raises(RuntimeError, match="different source/config"):
            launch.prepare_output(out, identity={**identity, "git_sha": "c" * 40}, retry_incomplete=True)
        checkpoint = out / "checkpoints/step-1"
        checkpoint.mkdir(parents=True)
        launch.prepare_output(out, identity=identity, resume=checkpoint)
        with pytest.raises(RuntimeError, match="explicit --resume"):
            launch.prepare_output(out, identity=identity, retry_incomplete=True)
        (out / "checkpoints/step-2").mkdir()
        with pytest.raises(RuntimeError, match="latest"):
            launch.prepare_output(out, identity=identity, resume=checkpoint)
        (out / "checkpoints/step-3.tmp-orphan").mkdir()
        launch.prepare_output(out, identity=identity, resume=out / "checkpoints/step-2")
        assert (out / "checkpoints/step-3.tmp-orphan").exists()
        (out / "SUCCESS").write_text("terminal")
        with pytest.raises(RuntimeError, match="terminal"):
            launch.prepare_output(out, identity=identity, resume=out / "checkpoints/step-2")
    assert launch.ROOT == real_root

    from scripts import flash_small40_repair as repair
    import hashlib
    # Historical launcher-only upgrades must reject this H3 runtime branch.
    # Still verify the historical base's actual Git objects/config bytes below.
    current_sha = repair.git_read(launch.ROOT, 'rev-parse', 'HEAD')
    changed = repair.git_read(launch.ROOT, 'diff', '--name-only', repair.REPAIR_BASE, current_sha)
    if not set(changed.splitlines()) <= repair.ALLOWED_CHANGES:
        with pytest.raises(RuntimeError, match='outside the allowed paths'):
            repair.prove_launcher_upgrade(launch.ROOT, current_sha, small_path)
    else:
        repair.prove_launcher_upgrade(launch.ROOT, current_sha, small_path)
    real_proof = repair.prove_launcher_upgrade(launch.ROOT, repair.REPAIR_BASE, small_path)
    assert set(real_proof["unchanged_git_objects"]) == {"visual_rl", "experiments", "data"}
    new_sha = "f" * 40
    def git(*args):
        if args[0] == "merge-base":
            return repair.REPAIR_BASE
        if args[0] == "diff":
            return "scripts/launch_flash140.py\nscripts/flash_small40_repair.py\ndocs/flash_small40_repair.md"
        if args[0] == "hash-object":
            return "config-blob"
        if args[0] == "rev-parse":
            path = args[1].split(":", 1)[1]
            return "config-blob" if path.endswith(".yaml") else "tree-" + path
        raise AssertionError(args)
    proof = repair.prove_launcher_upgrade(launch.ROOT, new_sha, small_path, git=git)
    assert proof["base_sha"] == repair.REPAIR_BASE
    for failure in ("ancestry", "paths", "runtime", "config"):
        def bad_git(*args):
            if failure == "ancestry" and args[0] == "merge-base":
                return "unrelated"
            if failure == "paths" and args[0] == "diff":
                return "visual_rl/algorithms/trainer/collection.py"
            if failure == "runtime" and args == ("rev-parse", f"{new_sha}:visual_rl"):
                return "different-runtime-tree"
            if failure == "config" and args[0] == "hash-object":
                return "changed-config-bytes"
            return git(*args)
        with pytest.raises(RuntimeError, match="ancestry|runtime|content|config"):
            repair.prove_launcher_upgrade(launch.ROOT, new_sha, small_path, git=bad_git)
    monkeypatch.setattr(repair, "git_read", lambda root, *args: git(*args))
    original = tmp_path / "run-002"
    monkeypatch.setattr(repair, "REPAIR_OUTPUT", original)
    original.mkdir()
    step2 = original / "checkpoints/step-2"
    step2.mkdir(parents=True)
    new_identity = dict(git_sha=new_sha, source_config_sha256=hashlib.sha256(small_path.read_bytes()).hexdigest())
    old_identity = dict(new_identity, git_sha=repair.REPAIR_BASE)
    owner_file = original / "flash140-owner.json"
    launch.publish(owner_file, old_identity)
    def authorize(**kwargs):
        return repair.resume_owner(original, kwargs.pop("identity", new_identity), kwargs.pop("resume", step2),
            upgrade_from=kwargs.pop("upgrade_from", repair.REPAIR_BASE), root=launch.ROOT, config=small_path, **kwargs)
    before_owner = owner_file.read_bytes()
    previous, next_owner = authorize()
    assert previous == old_identity and next_owner == {**new_identity, "launcher_upgrade": proof}
    assert owner_file.read_bytes() == before_owner  # preflight/read-only authorization
    for kw in ({"upgrade_from": None}, {"upgrade_from": "a" * 40},
               {"identity": dict(new_identity, source_config_sha256="a" * 64)},
               {"resume": original / "checkpoints/step-0"}):
        with pytest.raises(RuntimeError, match="different source/config"):
            authorize(**kw)
    with monkeypatch.context() as output_patch:
        output_patch.setattr(launch, "ROOT", source_root)
        launch.prepare_output(original, identity=previous, resume=step2)
    assert launch.ROOT == real_root
    launch.publish(owner_file, next_owner)  # same single atomic owner+audit publication as main
    assert authorize(upgrade_from=None) == (next_owner, next_owner)
    with monkeypatch.context() as output_patch:
        output_patch.setattr(launch, "ROOT", source_root)
        launch.prepare_output(original, identity=next_owner, resume=step2)
    assert launch.ROOT == real_root
    launch.publish(owner_file, {**next_owner, "launcher_upgrade": dict(proof, new_sha="forged")})
    with pytest.raises(RuntimeError, match="audit/source"):
        authorize()
    assert list(original.glob("checkpoints/step-*")) == [step2]  # no reset/conversion


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
        # S1: the source binding declares the rollout microbatch, so a B4 source
        # has a new data identity and refuses a B1 checkpoint; sample IDs, groups,
        # prompts, initial latents and generator streams are unchanged.
        batched_source = H3PromptSource(**(args | dict(rollout_microbatch=4)))
        assert batched_source.data_identity != restored.data_identity
        with pytest.raises(ValueError, match='identity'):
            H3PromptSource(**(args | dict(rollout_microbatch=4))).load_state_dict(state)
        plain_request, batched_request = H3PromptSource(**args).reserve(0), batched_source.reserve(0)
        assert plain_request.samples == batched_request.samples and plain_request.prompts == batched_request.prompts
        assert all(torch.equal(plain_request.initial_states[s], batched_request.initial_states[s]) for s in plain_request.prompts)
        assert all(torch.equal(a.get_state(), b.get_state()) for s in plain_request.prompts
                   for a, b in zip(plain_request.generators[s], batched_request.generators[s]))
        for bad in (2, True):
            with pytest.raises(ValueError, match='rollout microbatch'): H3PromptSource(**(args | dict(rollout_microbatch=bad)))
        with pytest.raises(ValueError, match='rollout microbatch'):
            H3PromptSource(**(args | dict(group_layout='rank_local', group_size=2, rollout_microbatch=4)))
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

    # C1s: initial full hash observes ALL model files in the same byte pass;
    # only non-terminal opted-in boundaries avoid the model content stream.
    from visual_rl.runtime import h3_binding as binding_module
    from visual_rl.runtime.h3_content_sentinel import ModelStatSnapshot, content_check_gate
    from visual_rl.composition.preflight import artifacts as artifact_module
    for content_mode in ('full', 'sentinel'):
        with monkeypatch.context() as patch:
            root = tmp_path / ('content-check-' + content_mode); root.mkdir()
            patch.setattr(h3_loading, '_admit_host_memory', admit_tiny_cpu_weights)
            patch.setattr(torch.distributed, 'broadcast', lambda *args, **kwargs: None)
            patch.setattr(h3_file_phase, 'file_phase', lambda action, **kwargs: action())
            h = binding_fixture(root, patch, bind=False)
            patch.setattr(H3Coordination, 'aux_turns', _CoordinationLeaf.aux_turns)
            h.adapter._aux._decoder = _decoder(h.profile.launch.model, h.adapter.geometry)
            reads, gates = [], []
            hash_file, exchange = artifact_module._hash_file_record, h3_world._exchange
            def counted_hash(digest, path, relative, metadata, totals, **kwargs):
                reads.append((path, metadata.st_size))
                return hash_file(digest, path, relative, metadata, totals, **kwargs)
            def observed_exchange(kind, *args):
                gates.append(kind)
                if kind.startswith('h3-content-check/'):
                    # Only opted-in fixtures need the new transport leaf;
                    # default full must use the original WORLD stub unchanged.
                    assert content_mode == 'sentinel' and args[-1]
                    return [args[-2]]*8
                return exchange(kind, *args)
            patch.setattr(artifact_module, '_hash_file_record', counted_hash)
            patch.setattr(h3_world, '_exchange', observed_exchange)
            b = binding_module.H3RuntimeBinding(**h.kwargs, boundary_content_check=content_mode)
            def model_reads():
                return [size for path, size in reads if path.is_relative_to(h.profile.launch.model)]
            assert len(model_reads()) == b.content['model']['file_count']
            assert sum(model_reads()) == b.content['model']['byte_count']  # no second bytes pass
            assert gates == (['h3-content-check/bind/sentinel'] if content_mode == 'sentinel' else [])
            assert b.boundary_content_check == content_mode and b.checkpoint_content_receipt is None
            full_live = b.validate_live()  # restore/default API remains full even when opted in
            assert len(model_reads()) == 2 * b.content['model']['file_count']
            contract, identity = b.contract, b.state_identity
            if content_mode == 'sentinel':
                snapshot = b._model_stat_snapshot
                assert snapshot.file_count == b.content['model']['file_count']
                assert {r[0] for r in snapshot.records if r[1] == 'file'} == {
                    p.relative_to(h.profile.launch.model).as_posix()
                    for p in h.profile.launch.model.rglob('*') if p.is_file()}
                assert snapshot.init_content_sha256 == h.profile.expected['model_sha256']
                with patch.context() as mode_only:
                    mode_only.setattr(b, '_boundary_content_check', 'full')
                    assert b._contract() == contract and b._live_ids() == full_live
            stages = H3IterationStages(data=h.source, prepared=h.prepared, predict=b.predict,
                decode=b.decode, score_one=b.score_one, coordination_factory=b.coordination_factory,
                process_group=h.pg, rank=0, device='cpu', read_behavior_id=h.source.read_behavior_id)
            trainer = stages.trainer(); trainer.prepare_run(PrepareRunContext('sentinel-test', 'h3', 0))
            try:
                for step in range(2):
                    trainer.run_iteration(step)
                    reads.clear()
                    video_evidence = b._video_files
                    boundary, live = b.checkpoint_snapshot(stages, trainer)
                    assert live == full_live and b.contract == contract and b.state_identity == identity
                    assert boundary[0].next_optimizer_step == step + 1
                    use_sentinel = content_mode == 'sentinel' and step == 0
                    assert len(model_reads()) == (0 if use_sentinel else b.content['model']['file_count'])
                    assert any(path.is_relative_to(h.code_root) for path, _ in reads)  # code still streamed
                    if content_mode == 'full':
                        assert b.checkpoint_content_receipt is None
                        assert not any(kind.startswith('h3-content-check/') for kind in gates)
                    else:
                        receipt = b.checkpoint_content_receipt
                        expected_mode = 'sentinel' if use_sentinel else 'full'
                        assert receipt == dict(snapshot.receipt(expected_mode), rank=0, optimizer_step=step+1,
                            checkpoint_contract_id=contract.checkpoint_contract_id,
                            checkpoint_relative_path=f'checkpoints/step-{step+1}')
                        assert f'h3-content-check/boundary/{step+1}/{expected_mode}/sentinel' in gates
                        if use_sentinel:
                            assert b._video_files is video_evidence  # no new decode credential semantics
                            # A safe pause below the configured target is also terminal.
                            reads.clear()
                            paused, paused_live = b.checkpoint_snapshot(stages, trainer, terminal=True)
                            assert paused == boundary and paused_live == live
                            assert len(model_reads()) == snapshot.file_count
                            assert b.checkpoint_content_receipt['mode'] == 'full'
                            reads.clear()
                            assert b.checkpoint_snapshot(stages, trainer)[1] == live
                            assert model_reads() == []  # no sticky terminal/full fallback
                            # Code and cache validation still execute on the sentinel path.
                            for path in (h.code_root / 'implementation.py', h.profile.launch.text_cache / 'index.json'):
                                original = path.read_bytes()
                                try:
                                    path.write_bytes(original + b' ')
                                    with pytest.raises(ValueError): b.checkpoint_snapshot(stages, trainer)
                                    assert b.checkpoint_content_receipt is None
                                finally:
                                    path.write_bytes(original)
                            assert b.checkpoint_snapshot(stages, trainer)[1] == live
                            # A real non-terminal binding failure never rehashes,
                            # refreshes, or accepts even a byte-restored file.
                            target = h.profile.launch.model / 'vae/diffusion_pytorch_model.safetensors'
                            original = target.read_bytes(); target.write_bytes(original + b' ')
                            reads.clear()
                            with pytest.raises(artifact_module.ArtifactIdentityError):
                                b.checkpoint_snapshot(stages, trainer)
                            assert reads == [] and b.checkpoint_content_receipt is None
                            target.write_bytes(original)
                            with pytest.raises(artifact_module.ArtifactIdentityError):
                                b.checkpoint_snapshot(stages, trainer)
                            assert reads == [] and b._model_stat_snapshot is snapshot
                            # Continue only this synthetic fixture to independently
                            # exercise the mandatory terminal full path below.
                # A changed base is never accepted by the terminal full hash.
                target = h.profile.launch.model / 'vae/diffusion_pytorch_model.safetensors'
                raw = target.read_bytes(); target.write_bytes(raw + b' ')
                with pytest.raises(ValueError, match='model content'):
                    b.checkpoint_snapshot(stages, trainer)
                assert b.checkpoint_content_receipt is None
            finally:
                trainer.close(); b.close(); h.scorer.close(); h.adapter.close(); h.manager.close()

    # Real filesystem faults, including empty directories (covered by the
    # content codec) and preserved-mtime rewrites; no stat or hash mocks here.
    import os
    for fault in ('size', 'mtime', 'ctime', 'missing', 'extra', 'directory', 'symlink',
                  'mode', 'inode', 'empty-directory', 'missing-directory', 'root-symlink'):
        root = tmp_path / ('sentinel-fault-' + fault); root.mkdir()
        (root / 'empty').mkdir()
        target = root / 'base.bin'; target.write_bytes(b'frozen base')
        files, directories = [], []
        observed = artifact_module._hash_artifact(root, file_observer=files.append,
            directory_observer=lambda relative, metadata: directories.append((relative, metadata)))
        assert observed == artifact_module._hash_artifact(root)  # identical content codec
        snapshot = ModelStatSnapshot.from_hash(root, observed, files, directories)
        with pytest.raises(ValueError, match='complete initial full'):
            ModelStatSnapshot.from_hash(root, observed, files[:-1], directories)
        with pytest.raises(ValueError, match='complete initial full'):
            ModelStatSnapshot.from_hash(root, observed, files, [])
        with monkeypatch.context() as stat_only:
            def no_open(*args, **kwargs): pytest.fail('stat sentinel opened file bytes')
            stat_only.setattr(Path, 'open', no_open)
            snapshot.verify()
        before = target.stat()
        if fault == 'size': target.write_bytes(b'longer frozen base')
        elif fault == 'mtime': os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        elif fault == 'ctime':
            target.write_bytes(b'changedbase')  # same size and inode; restore mtime only
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert target.stat().st_ino == before.st_ino and target.stat().st_size == before.st_size
        elif fault == 'missing': target.unlink()
        elif fault == 'extra': (root / 'extra.bin').write_bytes(b'new')
        elif fault == 'directory': target.unlink(); target.mkdir()
        elif fault == 'symlink': target.unlink(); target.symlink_to('empty')
        elif fault == 'mode': target.chmod(before.st_mode ^ 0o100)
        elif fault == 'inode':
            replacement = root / 'replacement'; replacement.write_bytes(target.read_bytes())
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns)); replacement.replace(target)
        elif fault == 'empty-directory': (root / 'new-empty').mkdir()
        elif fault == 'missing-directory': (root / 'empty').rmdir()
        else:
            moved = root.with_name(root.name + '-moved'); root.rename(moved); root.symlink_to(moved)
        with pytest.raises(artifact_module.ArtifactIdentityError): snapshot.verify()
        with pytest.raises(artifact_module.ArtifactIdentityError): snapshot.verify()  # never refresh-and-accept

    # Real WORLD8 receipt decoder rejects mode differences, not a mocked gate.
    with monkeypatch.context() as patch:
        mismatch = False
        def gather_content_modes(outputs, raw, *, group):
            row = json.loads(raw)
            outputs[:] = [json.dumps(row[:2] + [rank] + row[3:]).encode() for rank in range(8)]
            if mismatch:
                peer = json.loads(outputs[7]); peer[1] = peer[1].replace('/sentinel', '/full')
                outputs[7] = json.dumps(peer).encode()
        patch.setattr(h3_world.dist, 'all_gather_object', gather_content_modes)
        for rank in range(8):
            content_check_gate('sentinel', rank=rank, process_group=object(), phase='bind')
        mismatch = True
        for phase in ('bind', 'boundary/1/sentinel', 'boundary/2/full'):
            with pytest.raises(ValueError, match='inconsistent phase'):
                content_check_gate('sentinel', rank=0, process_group=object(), phase=phase)

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
        # Opt-in sidecars are durable before publication, separate from the
        # exact live/contract/data arguments. Pauses request a full check too.
        backend = lifecycle.H3LifecycleBackend(boundary_content_check='sentinel')
        def sentinel_snapshot(stages, trainer, *, terminal):
            result = snapshot(stages, trainer)
            step = trainer.next_optimizer_step
            fake_binding.checkpoint_content_receipt = dict(schema='h3.boundary-content-check.v1',
                configured_mode='sentinel', mode='full' if terminal or step == 2 else 'sentinel',
                file_count=3, snapshot_sha256='a'*64, init_content_sha256='b'*64,
                rank=0, optimizer_step=step, checkpoint_contract_id='contract',
                checkpoint_relative_path=f'checkpoints/step-{step}')
            return result
        fake_binding.checkpoint_snapshot = sentinel_snapshot
        def receipt_save(actual_prepared, **kwargs):
            saved = save(actual_prepared, **kwargs)
            saved.path = output / f'checkpoints/step-{saved.step}'
            sidecar = output / f'h3-content-check-step-{saved.step}-rank-0.json'
            assert json.loads(sidecar.read_text()) == fake_binding.checkpoint_content_receipt
            return saved
        bound.transaction.checkpoint = receipt_save
        committed, _ = backend.run(bound)
        assert committed.step == 2
        assert json.loads((output / 'h3-content-check-step-1-rank-0.json').read_text())['mode'] == 'sentinel'
        assert json.loads((output / 'h3-content-check-step-2-rank-0.json').read_text())['mode'] == 'full'
        assert exchanges.count(('lifecycle/checkpoint-content-receipt', True)) == 2
        (output / 'PAUSE_REQUESTED').write_text('pause')
        def paused_exchange(label, rank, group, payload, ok):
            if label == 'lifecycle/pause-request': return [True] + [None]*7
            return exchange(label, rank, group, payload, ok)
        patch.setattr(h3_world, '_exchange', paused_exchange)
        # The lifecycle consumes the all-rank pause, not a rank-local decision.
        with pytest.raises(lifecycle.H3SafeBoundaryPause): backend.run(bound)
        assert json.loads((output / 'h3-content-check-step-1-rank-0.json').read_text())['mode'] == 'full'
        saved_count = len(saves)
        def no_receipt(stages, trainer, **kwargs):
            result = snapshot(stages, trainer); fake_binding.checkpoint_content_receipt = None
            return result
        fake_binding.checkpoint_snapshot = no_receipt
        with pytest.raises(ValueError, match='missing checkpoint content'):
            backend.run(bound)
        assert len(saves) == saved_count  # cannot publish without provenance

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

    # V1v: owner-local video VAE reuse on the real tiny decoder. Only the two
    # VAE leaves, CUDA RNG/free telemetry and the memory journal are
    # synthetic; credentials come from the real full hasher over real files.
    import hashlib, random, time
    from contextlib import nullcontext
    from safetensors.torch import save_file
    from visual_rl.models.implementations import h3_decode
    from visual_rl.models.implementations import h3_decode_reuse as reuse
    from visual_rl.models.implementations.h3 import _H3Aux
    from visual_rl.models.implementations.h3_decode import H3CPUDecoder
    from visual_rl.composition.preflight.artifacts import _hash_artifact
    with monkeypatch.context() as patch:
        root = (tmp_path / 'reuse-decoder').resolve(); root.mkdir()
        geometry = H3T2VAGeometry(32, 32, 124)
        configs = _artifacts(root)
        for component in ('vae', 'audio_vae'):
            save_file({'weight': torch.tensor(1.)}, str(root / component / 'diffusion_pytorch_model.safetensors'))
        (root / 'vae/empty-shard-dir').mkdir()  # hashed directory that records no file
        def credential():
            records = []
            identity = _hash_artifact(root, content_policy='all-files.v1',
                file_observer=lambda r: records.append(r) if r.relative.startswith('vae/') else None)
            assert identity == _hash_artifact(root, content_policy='all-files.v1')  # observer never alters identity
            return reuse._video_files(root, identity['content_sha256'], records)
        evidence = credential()
        assert sorted(r.relative for r in evidence.records) == ['vae/config.json', 'vae/diffusion_pytorch_model.safetensors']
        assert all(r.sha256 == hashlib.sha256((root / r.relative).read_bytes()).hexdigest() for r in evidence.records)
        events, loads, decodes, finalized = [], [], [], []
        knobs = dict(rng=False, drift=None, probe=False, processor=None)
        patch.setattr(memory, 'event', lambda kind, **fields: events.append((kind, fields)))
        for module, name in ((torch, 'set_rng_state'), (random, 'setstate'), (np.random, 'set_state')):
            patch.setattr(module, name, lambda *a, **k: pytest.fail('video reuse must observe RNG read-only'))
        class Processor:  # mirrors MiniMaxH3VideoAttnProcessor: plain callable object without tensor state
            _attention_backend = None
            _parallel_config = None
            def __call__(self, module, hidden_states): return hidden_states
        class Attention(torch.nn.Module):  # mirrors AttentionModuleMixin.set_processor: object kept in vars(module)
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Parameter(torch.ones(1))  # deterministic: the loader must not consume RNG
                self.processor = Processor()
                if knobs['processor'] == 'tensor': self.processor.cache = torch.zeros(1)
                if knobs['processor'] == 'cycle': self.processor.owner = self.processor
        def resolve(component):
            class Leaf(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.tensor(1.))
                    self.register_buffer('filter', torch.tensor([.5]))
                    self.config = config; self.hop_length = 800
                    if component == 'vae': self.attn = Attention()
                    if knobs['probe'] and component == 'vae': self.flags = {None, 'mixed'}  # probe sorted() TypeError
                @classmethod
                def from_pretrained(cls, path, *, subfolder, torch_dtype, local_files_only):
                    assert subfolder == component and torch_dtype == torch.float32 and local_files_only is True
                    if knobs['rng'] and component == 'vae': random.random(); np.random.rand(); torch.rand(1)
                    loads.append(component)
                    model = cls(json.loads((Path(path) / component / 'config.json').read_text()))
                    weakref.finalize(model, finalized.append, component)
                    return model
                def decode(self, latents, *, return_dict):
                    assert return_dict is False and not self.training and not torch.is_grad_enabled()
                    decodes.append(component)
                    if component != 'vae':
                        return (latents[:, :1].repeat_interleave(800, dim=2) * self.weight,)
                    if knobs['drift'] == 'buffer': self.filter.add_(1.)  # registered state changed in place
                    if knobs['drift'] == 'attribute': self.scratch = torch.ones(1)  # unregistered tensor state
                    if knobs['drift'] == 'processor': self.attn.processor.calls = 1  # nested plain state changed
                    t = torch.arange(geometry.num_frames) % latents.shape[2]
                    h = torch.arange(geometry.height) % latents.shape[3]
                    w = torch.arange(geometry.width) % latents.shape[4]
                    return (latents[:, :3].index_select(2, t).index_select(3, h).index_select(4, w) * self.weight,)
            if component == 'vae':  # the only class the reuse gate accepts
                Leaf.__module__ = 'diffusers.models.autoencoders.autoencoder_kl_minimax_h3'
                Leaf.__name__ = Leaf.__qualname__ = 'AutoencoderKLMiniMaxH3'
            return Leaf
        patch.setattr(h3_decode, '_resolve_vae_class', resolve)
        terminals = (_terminal(geometry), _terminal(geometry)); terminals[1][0, 7] = 3.  # second sample differs
        def rng(): return (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().numpy().tobytes())
        def run(decoder, scoped):
            del loads[:], decodes[:], events[:]
            random.seed(11); np.random.seed(11); torch.manual_seed(11)
            outputs, start = [], len(finalized)
            with (decoder.video_reuse_scope(evidence) if scoped else nullcontext()):
                for terminal in terminals:
                    with decoder.decode_one(terminal, geometry) as media:
                        assert media.video.dtype == media.audio.dtype == torch.float32
                        outputs.append((media.video.numpy().tobytes(), media.audio.numpy().tobytes(), media.metadata))
                assert (decoder._reuse is not None) is scoped
            assert decoder._reuse is None and finalized[start:].count('vae') == loads.count('vae')  # no model survives
            return outputs, list(loads), list(decodes), rng()
        decoder = _decoder(root, geometry)
        off, on = run(decoder, False), run(decoder, True)
        assert on[0] == off[0] and on[3] == off[3] and on[2] == off[2] == ['vae', 'audio_vae'] * 2
        assert off[1] == ['vae', 'audio_vae'] * 2 and on[1] == ['vae', 'audio_vae', 'audio_vae']  # one video load, audio never cached
        assert [k for k, _ in events] == ['video_reuse_scope_begin', 'video_reuse_miss', 'video_reuse_hit', 'video_reuse_drop']
        assert events[0][1]['version'] == reuse.REUSE_VERSION and events[-1][1]['reason'] == 'owner-scope-release'
        knobs['rng'] = True  # loader consumes Python/NumPy/Torch RNG: no cache, no replay/restore, same sequence as off
        off, on = run(decoder, False), run(decoder, True)
        assert on[0] == off[0] and on[3] == off[3] and on[1] == off[1] == ['vae', 'audio_vae'] * 2
        assert [f['reason'] for k, f in events if k == 'video_reuse_ineligible'] == ['loader-global-RNG-or-unobservable']
        assert 'video_reuse_hit' not in [k for k, _ in events] and not decoder._poisoned
        knobs['rng'], knobs['probe'] = False, True  # a TypeError inside the eligibility probe is advisory, not fatal
        off, on = run(decoder, False), run(decoder, True)
        assert on[0] == off[0] and on[1] == off[1] == ['vae', 'audio_vae'] * 2 and not decoder._poisoned
        assert [f['reason'][:9] for k, f in events if k == 'video_reuse_ineligible'] == ['TypeError']
        knobs['probe'] = False
        for drift in ('buffer', 'attribute', 'processor'):  # decode changed registered/unregistered/nested state: media kept, cache dropped
            knobs['drift'] = drift
            off, on = run(decoder, False), run(decoder, True)
            assert on[0] == off[0] and on[1] == off[1] == ['vae', 'audio_vae'] * 2 and not decoder._poisoned
            reasons = [f['reason'] for k, f in events if k == 'video_reuse_drop']
            assert reasons[0].startswith('decode-state-drift') and reasons[-1] == 'owner-scope-release'
            assert 'video_reuse_hit' not in [k for k, _ in events]
        knobs['drift'] = None
        for hidden, reason in (('tensor', 'ValueError: unregistered tensor state'), ('cycle', 'ValueError: cyclic module state')):
            knobs['processor'] = hidden  # processor object hiding tensor/cyclic state: ineligible, off path
            off, on = run(decoder, False), run(decoder, True)
            assert on[0] == off[0] and on[1] == off[1] == ['vae', 'audio_vae'] * 2 and not decoder._poisoned
            assert [f['reason'] for k, f in events if k == 'video_reuse_ineligible'] == [reason]
        knobs['processor'] = None
        class Plain: pass
        nested = Plain(); nested.inner = Plain(); nested.inner.flag = 'x'
        name = Plain.__module__ + '.' + Plain.__qualname__
        assert reuse._plain(nested) == ['object', name, {'inner': ['object', name, {'flag': 'x'}]}]
        deep = []
        for _ in range(reuse._NEST_LIMIT + 1): deep = [deep]
        for bad, message in ((deep, 'nested too deeply'), (lambda: None, 'callable'), (Plain, 'callable'),
                             (torch.nn.Linear(1, 1), 'module state'), (object(), 'uncontrolled Python')):
            with pytest.raises(ValueError, match=message): reuse._plain(bad)
        # Credential is an invalidation sentinel over the hashed file set/stat.
        shard = root / 'vae/diffusion_pytorch_model.safetensors'
        original, stamp = shard.read_bytes(), shard.stat()
        time.sleep(.02)
        shard.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))  # same size, mtime restored, ctime moves
        os.utime(shard, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert (shard.stat().st_size, shard.stat().st_mtime_ns) == (stamp.st_size, stamp.st_mtime_ns)
        del loads[:]
        for restore in (False, True):
            if restore: shard.write_bytes(original)  # bytes back, but the full hash was not rerun
            with pytest.raises(ValueError, match='changed since full verification'):
                with decoder.video_reuse_scope(evidence): pytest.fail('tampered shard accepted')
        evidence = credential()
        with decoder.video_reuse_scope(evidence): pass
        (root / 'vae/late-empty-dir').mkdir()  # directories without files never gate reuse
        with decoder.video_reuse_scope(evidence): pass
        extra = root / 'vae/extra.bin'; extra.write_bytes(b'x')
        with pytest.raises(ValueError, match='changed since full verification'):
            with decoder.video_reuse_scope(evidence): pytest.fail('extra file accepted')
        extra.unlink(); shard.rename(root / 'shard.bak')
        with pytest.raises(ValueError, match='changed since full verification'):
            with decoder.video_reuse_scope(evidence): pytest.fail('missing file accepted')
        (root / 'shard.bak').rename(shard)
        assert loads == [] and decoder._reuse is None and not decoder._poisoned
        evidence = credential(); start = len(finalized)
        with pytest.raises(ValueError, match='changed since full verification'):  # mid-scope drift fails closed
            with decoder.video_reuse_scope(evidence):
                with decoder.decode_one(terminals[0], geometry): pass
                extra.write_bytes(b'x')
                with decoder.decode_one(terminals[1], geometry): pytest.fail('drifted file list accepted mid-scope')
        extra.unlink()
        assert decoder._poisoned and decoder._reuse is None and loads == ['vae', 'audio_vae']
        assert finalized[start:] == ['audio_vae', 'vae']  # cached video model released only at scope exit
        with pytest.raises(RuntimeError, match='closed or poisoned'):
            with decoder.decode_one(terminals[0], geometry): pytest.fail('poisoned decoder decoded')
        decoder.close(); decoder.close()  # idempotent after poison
        # A foreign thread (main thread after an aux timeout) closes without
        # raising or dropping; the producer's own scope exit releases the model.
        decoder = _decoder(root, geometry)
        inside, release, worker_errors = Event(), Event(), []
        def producer():
            try:
                with decoder.video_reuse_scope(evidence):
                    with decoder.decode_one(terminals[0], geometry):
                        inside.set(); assert release.wait(5), 'test never released the lease'
                    with pytest.raises(RuntimeError, match='closed or poisoned'):
                        with decoder.decode_one(terminals[1], geometry): pytest.fail('lease after close')
            except BaseException as exc: worker_errors.append(exc)
        worker = Thread(target=producer, name='reuse-producer'); worker.start()
        try:
            assert inside.wait(5)
            del loads[:]; start = len(finalized)
            decoder.close()
            assert decoder._closed and decoder._poisoned and decoder._reuse is not None and decoder._reuse.model is not None
            with pytest.raises(RuntimeError, match='closed or poisoned'):
                with decoder.video_reuse_scope(evidence): pytest.fail('scope after close')
        finally:
            release.set(); worker.join(10)
        assert not worker.is_alive() and not worker_errors and loads == []
        assert decoder._reuse is None and finalized[start:] == ['vae']
        decoder.close()
        decoder = _decoder(root, geometry)  # same-thread misuse keeps raising and leaves the decoder open
        with decoder.video_reuse_scope(evidence):
            with pytest.raises(RuntimeError, match='active video producer scope'): decoder.close()
            with decoder.decode_one(terminals[0], geometry):
                with pytest.raises(RuntimeError, match='active decode lease'): decoder.close()
        assert not decoder._closed and not decoder._poisoned
        aux = object.__new__(_H3Aux); aux.closed = False; aux._cache = object(); aux._decoder = decoder
        aux.close(); aux.close()
        assert aux.closed and aux._decoder is None and aux._cache is None and decoder._closed
        # Synthetic rank-local CUDA telemetry (no GPU): warm entry below 40GiB
        # drops its own cache first, re-reads the real free, then cold or refuse.
        class Card:
            def __init__(self): self._root, self._device, self.free, self.releases = root, torch.device('cuda', 0), 100., 0
            def _card_free(self): return self.free
            def _allocated(self): return 0
            def _release_device_cache(self): self.releases += 1
        card, cached = Card(), resolve('vae')(configs['vae'])
        state = reuse.VideoReuse(card, evidence)
        def warm(): state.model, state.signature, state.geometry, state.bytes, state.allowed = cached, 'sig', 'key', 8, True
        warm(); card.free = 39.; del events[:]
        with pytest.raises(BlockingIOError, match='40GiB'): state.admission()
        assert state.model is None and not state.allowed and card.releases == 1 and state.bytes == 0
        warm(); reads = []
        card._card_free = lambda: reads.append(len(reads)) or (39. if len(reads) == 1 else 64.)
        state.admission(audio=True)  # evict the idle video cache, then the real re-read admits the cold audio load
        assert state.model is None and not state.allowed and reads == [0, 1] and card.releases == 2
        assert [f['reason'] for k, f in events if k == 'video_reuse_drop'] == ['warm-headroom', 'audio-headroom']
        cold = state.borrow(geometry, configs['vae'], lambda: resolve('vae')(configs['vae']))
        assert isinstance(cold, torch.nn.Module) and state.model is None and state.bytes == 0  # cold path never caches
        card.free = 100.; warm(); state.admission()
        assert state.model is cached and state.allowed  # enough real free: the warm model stays
        class SyntheticCard(H3CPUDecoder):  # real decoder path; only device/free telemetry are synthetic
            free = 39.
            def __init__(self, *a, **k): super().__init__(*a, **k); self._device = torch.device('cuda', 0)
            def _card_free(self): return self.free
            def _allocated(self): return 0
            def _release_device_cache(self): pass
        media_bytes, workspace = H3CPUDecoder.estimate_bytes(geometry)
        synthetic = SyntheticCard(root, max_media_bytes=media_bytes, max_decode_bytes=workspace)
        del loads[:], events[:]
        with pytest.raises(BlockingIOError, match='40GiB'):
            with synthetic.video_reuse_scope(evidence):
                with synthetic.decode_one(terminals[0], geometry): pytest.fail('audio cold load admitted below 40GiB')
        assert loads == ['vae'] and synthetic._poisoned and synthetic._reuse is None  # audio refused before its load
        assert [f['reason'] for k, f in events if k == 'video_reuse_ineligible'] == ['loader-global-RNG-or-unobservable']

    samples = tuple(pipeline.RewardSample(i, f'pipe-{i}', 'g', i, f'prompt {i}', i) for i in range(4))
    # Deterministic capacity/overlap and cancellation: score0 blocks, decode1
    # completes, packet1 queues, producer attempts permit3 but cannot decode2.
    from visual_rl.models.implementations import h3_loading
    for failure in (None, 'score', 'nonfinite', 'saver', 'decode', 'transfer', 'cleanup', 'decode-cleanup'):
        with monkeypatch.context() as patch:
            budgets = []
            patch.setattr(h3_loading, '_admit_host_memory', lambda n: budgets.append(n))
            decoded, scored, tensor_refs, owners = [], [], [], []
            score_started, second_queued, third_permit, release = Event(), Event(), Event(), Event()
            real_queue, real_semaphore, real_event = pipeline.Queue, pipeline.Semaphore, pipeline.Event
            signals = []
            def pipeline_event():
                signal = real_event(); signals.append(signal)
                return signal
            patch.setattr(pipeline, 'Event', pipeline_event)
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
                if failure in ('cleanup', 'decode-cleanup'):
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
                    if failure in ('decode', 'decode-cleanup'): raise RuntimeError('decode failed')
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
                if failure not in ('decode', 'transfer', 'decode-cleanup'):
                    assert second_queued.wait(5)
                    assert third_permit.wait(5)
                    assert decoded == [0, 1] and scored == [0]
                else:
                    assert signals[0].wait(5), 'producer never reported cancellation'
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
                assert failure.split('-')[0] in str(errors[0]) or (failure == 'nonfinite' and 'finite' in str(errors[0]))
                if failure == 'decode-cleanup':
                    assert 'decode failed' in str(errors[0])
                    assert any('cleanup failed' in note for note in errors[0].__notes__)
                # No new POST once the blocked first operation fails.
                if failure in ('score', 'nonfinite', 'saver', 'cleanup'): assert scored == [0]
            assert all(ref() is None for ref in tensor_refs)
            assert all(ref() is None or ref()._video is None for ref in owners)

    # Real dual reward -> original H3HPSReward/JPEG/validation/observed saver.
    # No helper/reward mock: only decode model/HTTP/coordination leaves.
    from tests.test_h3_mvp_cpu_hps import _scorer, _media
    from visual_rl.algorithms.trainer.h3_dual_stages import H3DualIterationStages
    from visual_rl.algorithms.trainer.interface import StageValue, IterationIdentity
    from visual_rl.core.protocols.flash_hpsv3 import VideoContract
    from scripts.launch_h3_media import observed_score
    from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot
    with monkeypatch.context() as patch:
        patch.setattr(h3_loading, '_admit_host_memory', lambda n: None)
        contract = VideoContract(124, 32, 32)
        scorer, transport, _ = _scorer(tmp_path / 'pipeline-hps', patch, contract)
        scorer.acquire()
        source = SimpleNamespace()
        identity = IterationIdentity(0, 'source', 'train', tuple(s.sample_id for s in samples),
                                     tuple(s.group_id for s in samples), tuple(s.member for s in samples))
        request = SimpleNamespace(identity=identity, group_layout='paired_4x8',
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
        stage.reuse_video_vae = False; stage.video_reuse_scope = None
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
        value = StageValue(request.identity, context)
        stage.reward(value); serial = context.rewards
        serial_payloads = [call[2]['data'] for call in transport.calls if call[0] == 'POST']
        stage.reward_pipeline = True
        stage.reward(value)
        payloads = [call[2]['data'] for call in transport.calls if call[0] == 'POST']
        assert context.rewards == serial and payloads[4:] == serial_payloads
        assert len(payloads) == 8 and all(len(json.loads(p)['jpeg_frames']) == 124 for p in payloads)
        assert scorer.identity.video_contract.top_k == 37
        assert exported == [samples[0].sample_id]*2  # existing member0-only selection
        assert RankRNGSnapshot.capture_current(0) == before_rng
        assert all(torch.equal(t, original_terminals[sid]) for sid, t in terminals.items())
        assert len(aux) == 2 and gates[-1].startswith('dual/reward-pipeline/1/queue1/total2/')
        # V1v: the binding scope wraps the whole P1 owner work on the owner
        # worker; it is released only after the consumer joined and before
        # owner-end, and its rank gate precedes the reward-pipeline gate.
        order = []
        @contextmanager
        def reuse_scope():
            order.append(('enter', threading.current_thread().name))
            try: yield
            finally:
                assert not any(t.name == 'h3-reward-consumer' for t in threading.enumerate()), 'scope released before consumer join'
                order.append(('exit', threading.current_thread().name))
        class OwnerCoordinator(Coordinator):
            def aux_turns(self, work):
                order.append(('owner-begin', None)); box = []
                def execute():
                    try: box.append(('ok', work()))
                    except BaseException as exc: box.append(('error', exc))
                runner = Thread(target=execute, name='h3-aux-0'); runner.start(); runner.join(30)
                assert not runner.is_alive() and box
                order.append(('owner-end', None))
                if box[0][0] == 'error': raise box[0][1]
                return box[0][1]
        real_score_one = stage.score_one
        stage.score_one = lambda **kw: float(int(kw['sample_id'].split('-')[1]))
        stage.reuse_video_vae, stage.video_reuse_scope = True, reuse_scope
        for pipelined in (False, True):
            stage.reward_pipeline = pipelined; del order[:]; del gates[:]
            context.coordinator = OwnerCoordinator()
            stage.reward(value)
            assert order == [('owner-begin', None), ('enter', 'h3-aux-0'), ('exit', 'h3-aux-0'), ('owner-end', None)]
            assert context.rewards == tuple((s.sample_id, s.group_id, s.member, float(s.ordinal), True) for s in samples)
            video_gate = f'dual/video-reuse/1/{reuse.REUSE_VERSION}/{reuse.VIDEO_CAP}'
            assert gates.index(video_gate) < gates.index(gates[-1]) and gates[-1].startswith('dual/reward-pipeline/')
        assert RankRNGSnapshot.capture_current(0) == before_rng
        stage.reuse_video_vae, stage.video_reuse_scope, stage.score_one = False, None, real_score_one
        stage.reward_pipeline = True; context.coordinator = Coordinator()
        context.coordinator.mismatch = True
        with pytest.raises(ValueError, match='mismatch'): stage.reward(value)
        assert len(aux) == 2 and len(payloads) == 8
        context.coordinator.mismatch = False
        request.group_layout = 'rank_local'
        with pytest.raises(ValueError, match='admission'): stage.reward(value)
        assert len(aux) == 2
        request.group_layout = 'paired_4x8'
        # Real gate and receipt decoder reject a peer's differing mode/budget
        # BEFORE aux work, not merely a fake coordinator raising on demand.
        from visual_rl.models.lifecycle import h3_coordination
        with patch.context() as world:
            world.setattr(h3_coordination, '_world_rank', lambda group: 0)
            world.setattr(h3_coordination, '_cuda_device', lambda device: torch.device('cpu'))
            for peer_config in ('dual/reward-pipeline/0/queue1/total2/10000000/20000000',
                                'dual/reward-pipeline/1/queue1/total2/10000001/20000000'):
                def gather(output, raw, **kwargs):
                    for rank in range(8):
                        row = json.loads(raw); row[2] = rank
                        if rank == 7 and row[4][2].startswith('dual/reward-pipeline/1/'):
                            row[4][2] = peer_config
                        output[rank] = json.dumps(row).encode()
                world.setattr(torch.distributed, 'all_gather_object', gather)
                coordinator = h3_coordination.H3Coordination(plan, process_group=object(), device='cuda:0')
                world.setattr(coordinator, 'aux_turns', lambda work: pytest.fail('mismatched rank entered aux'))
                context.coordinator = coordinator
                with pytest.raises(h3_coordination.H3CoordinationError, match='gate failed'):
                    stage.reward(value)
                assert coordinator._poisoned
        # A real trainer cannot reach credit/update or the next owner after an
        # observed export fails. The first fake HTTP operation finishes normally;
        # its saver then fails, and every consumer is joined before propagation.
        from visual_rl.algorithms.trainer.grpo import GRPOTrainer
        owner_events, aborts = [], []
        context.coordinator = Coordinator()
        def owner_turn(work):
            owner_events.append('owner0')
            result = work()
            owner_events.append('next-owner')
            return result
        context.coordinator.aux_turns = owner_turn
        def failed_saver(**kwargs): raise RuntimeError('actual observed saver failed')
        stage.score_one = observed_score(scorer.score_one, source=source, root=tmp_path / 'bad-view',
            ffmpeg='unused', ffprobe='unused', launcher_sha='cpu', every=1, saver=failed_saver)
        def no_update(value): pytest.fail('reward failure reached credit/update')
        trainer = GRPOTrainer(prelude=SimpleNamespace(build=lambda step: value,
            abort_iteration=lambda identity: aborts.append(identity)), rollout=lambda v: v,
            reward=stage.reward, advantage=no_update, credit=no_update, optimize=no_update)
        trainer.prepare_run(PrepareRunContext('pipeline-failure', 'h3', 0))
        with pytest.raises(RuntimeError, match='actual observed saver failed'): trainer.run_iteration(0)
        assert owner_events == ['owner0'] and aborts == [request.identity]
        assert trainer.next_optimizer_step == 0
        assert len([c for c in transport.calls if c[0] == 'POST']) == 9
        assert not any(t.name == 'h3-reward-consumer' for t in threading.enumerate())
        trainer.close(); scorer.close()

    # Concurrent observation: CPU score encloses GPU decode in wall time but
    # must NOT enclose it in the GPU stack or reset/synchronize its counters.
    class Counters:
        def __init__(self): self.resets = self.syncs = self.snapshots = 0
        def snapshot(self):
            assert threading.current_thread().name not in ('cpu-observer', 'h3-reward-consumer')
            self.snapshots += 1
            return dict(allocated=1, reserved=2, max_allocated=3, max_reserved=4, device_uuid='GPU-cpu')
        def reset(self): self.resets += 1
        def synchronize(self): self.syncs += 1
    counters = Counters()
    journal = memory.Journal(tmp_path / 'concurrent-memory', rank=0, role='test', device=0, counters=counters)
    entered, decode_done = Event(), Event()
    observer_errors = []
    try:
        with journal.span('reward', step=1):
            scope = journal.capture_cpu_scope()
            def observe():
                try:
                    with scope():
                        with journal.span('reward_score', synchronize=True, sample_id='score-0'):
                            entered.set()
                            assert decode_done.wait(5)
                            with journal.span('video_save', sample_id='score-0'): pass
                except BaseException as exc: observer_errors.append(exc)
            observer = Thread(target=observe, name='cpu-observer'); observer.start()
            try:
                assert entered.wait(5)
                with journal.span('vae_decode', sample_id='decode-1'): pass
            finally:
                decode_done.set(); observer.join(10)
            assert not observer.is_alive() and not observer_errors
        events = [json.loads(line) for line in journal.path.read_text().splitlines()]
        cpu_events = [e for e in events if e.get('observation') == 'cpu-consumer']
        assert [e['kind'] for e in cpu_events] == ['cpu_span_begin', 'cpu_span_begin', 'cpu_span_end', 'cpu_span_end']
        assert all(e['context']['sample_id'] == 'score-0' and e['context']['step'] == 1
                   and e['counters'] is None and not e['synchronized'] for e in cpu_events)
        assert cpu_events[1]['ancestors'] == ['reward', 'reward_score', 'video_save']
        assert all(e['thread'] != 'cpu-observer' for e in events if 'unified_reset' in e['kind'])
        assert counters.resets == 4 and counters.syncs == 0 and journal.stack == []
        assert all(e['context'].get('sample_id') != 'score-0' for e in events if e['thread'] != 'cpu-observer')
        with journal.capture_cpu_scope()():
            with pytest.raises(OSError, match='byte budget'):
                old_limit = journal.limit; journal.limit = journal.bytes
                try:
                    with journal.span('reward_score'): pass
                finally: journal.limit = old_limit
        assert not journal.healthy
    finally: journal.close()

    # Journal and payload failures propagate through the actual pipeline and
    # leave neither a media owner nor consumer behind.
    for fault_kind in ('journal', 'payload'):
        with monkeypatch.context() as patch:
            patch.setattr(h3_loading, '_admit_host_memory', lambda n: None)
            journal = memory.Journal(tmp_path / f'pipeline-{fault_kind}-log', rank=0,
                                     role='test', device=0, counters=Counters())
            patch.setattr(memory, 'journal', lambda: journal)
            media_refs, posts = [], []
            @contextmanager
            def decode_one(sample):
                media = H3AVMedia(torch.zeros(2, 3, 2, 2), torch.zeros(4, 2), 24, 32000)
                media_refs.append(weakref.ref(media.video))
                try: yield media
                finally: media._release()
            def consume_one(sample, media):
                with memory.span('reward_score', sample_id=sample.sample_id): posts.append(sample.sample_id)
                return 1.
            if fault_kind == 'journal': journal.limit = journal.bytes
            try:
                with pytest.raises((OSError, ValueError), match='byte budget'):
                    pipeline.reward_pipeline(samples[:1], decode=decode_one, score=consume_one,
                        media_bytes=1 if fault_kind == 'payload' else 1024, decode_bytes=2048)
                assert not posts and all(ref() is None for ref in media_refs)
                assert not any(t.name == 'h3-reward-consumer' for t in threading.enumerate())
            finally:
                journal.limit = 64*1024**2; journal.close()

    # Backend/launcher wiring is offline: no controller.run or GPU entrypoint.
    from visual_rl.runtime.h3_lifecycle import H3LifecycleBackend
    from scripts import launch_h3_resolution_memory as two_update
    assert H3LifecycleBackend().reward_pipeline is False
    for bad in (1, 'yes', None):
        with pytest.raises(TypeError, match='bool'): H3LifecycleBackend(reward_pipeline=bad)
    # --checkpoint-threads: strict int, default 1 (serial publish), wired to
    # both the manager (rank0 tree hashing) and the transaction (DCP writer).
    assert H3LifecycleBackend().checkpoint_threads == 1
    for bad in (True, 4.0, '4', None):
        with pytest.raises(TypeError, match='int'): H3LifecycleBackend(checkpoint_threads=bad)
    for bad in (0, -1, 65):
        with pytest.raises(ValueError, match='1..64'): H3LifecycleBackend(checkpoint_threads=bad)
    assert H3LifecycleBackend(checkpoint_threads=4).checkpoint_threads == 4
    from visual_rl.artifacts.checkpoint import h3_transaction as h3_tx
    with monkeypatch.context() as patch:
        patch.setattr(h3_tx, '_world_rank', lambda group: 0)
        threaded_manager = h3_tx.AtomicCheckpointManager(tmp_path / 'threaded-checkpoints', checkpoint_threads=4)
        assert h3_tx.H3CheckpointTransaction(threaded_manager, process_group=object()).checkpoint_threads == 1
        assert h3_tx.H3CheckpointTransaction(threaded_manager, process_group=object(), checkpoint_threads=4).checkpoint_threads == 4
        with pytest.raises(TypeError, match='int'):
            h3_tx.H3CheckpointTransaction(threaded_manager, process_group=object(), checkpoint_threads=True)
    enabled_backend = H3LifecycleBackend(reward_pipeline=True)
    for algorithm, layout in (('cps', 'rank_local'), ('dual_flow', 'rank_local'), ('dual_flash', 'rank_local')):
        with pytest.raises(ValueError, match='paired_4x8'):
            enabled_backend._check_reward_pipeline(SimpleNamespace(training=SimpleNamespace(
                algorithm=algorithm, group_layout=layout)))
    enabled_backend._check_reward_pipeline(SimpleNamespace(training=SimpleNamespace(
        algorithm='dual_flash', group_layout='paired_4x8')))
    for bad in (1, 'yes', None):
        with pytest.raises(TypeError, match='bool'):
            H3DualIterationStages(mode='dual_flash', seed=42, reward_pipeline=bad)
        with pytest.raises(TypeError, match='bool'):
            H3DualIterationStages(mode='dual_flash', seed=42, reuse_rollout_anchor=bad)
        with pytest.raises(TypeError, match='bool'): H3LifecycleBackend(reuse_rollout_anchor=bad)
    with pytest.raises(ValueError, match='dual_flash'):
        H3DualIterationStages(mode='dual_flow', seed=42, reward_pipeline=True, media_bytes=1024, decode_bytes=2048)
    assert two_update.TwoUpdateBackend(args=SimpleNamespace(reward_pipeline=True,profile_rollout=False,reuse_video_vae=False,reuse_rollout_anchor=False, checkpoint_threads=1)).reward_pipeline is True
    assert two_update.TwoUpdateBackend(args=SimpleNamespace(reward_pipeline=False,profile_rollout=False,reuse_video_vae=False,reuse_rollout_anchor=False, checkpoint_threads=1)).reward_pipeline is False
    # C1s CLI/backend mode validation and lifecycle-to-binding threading.
    assert H3LifecycleBackend().boundary_content_check == 'full'
    for mode in ('full', 'sentinel'):
        assert two_update.TwoUpdateBackend(args=SimpleNamespace(reward_pipeline=False,
            profile_rollout=False, reuse_video_vae=False, reuse_rollout_anchor=False, checkpoint_threads=1, boundary_content_check=mode)).boundary_content_check == mode
    for bad in (None, True, 1, '', 'FULL', 'sentinel ', []):
        with pytest.raises(ValueError, match='full or sentinel'):
            H3LifecycleBackend(boundary_content_check=bad)
        with pytest.raises(ValueError, match='full or sentinel'):
            two_update.TwoUpdateBackend(args=SimpleNamespace(reward_pipeline=False,
                profile_rollout=False, reuse_video_vae=False, reuse_rollout_anchor=False, checkpoint_threads=1, boundary_content_check=bad))
    with monkeypatch.context() as patch:
        constructed = []
        def binding_leaf(**kwargs):
            constructed.append(kwargs)
            return SimpleNamespace(close=lambda: None)
        patch.setattr(binding_module, 'H3RuntimeBinding', binding_leaf)
        patch.setattr(h3_world, '_exchange', lambda label, rank, group, payload, ok: [payload]*8)
        backend = H3LifecycleBackend(boundary_content_check='sentinel')
        graph = SimpleNamespace(profile=SimpleNamespace(launch=SimpleNamespace(resume_from=None)),
            facts=object(), adapter=object(), source=object(), scorer=object(), code_root=tmp_path, code_hash='c'*64)
        runtime = SimpleNamespace(rank=0, process_group=object(), device='cpu', gpu_uuids=())
        backend.restore_prepared(None, runtime, graph, object())
        assert constructed[0]['boundary_content_check'] == 'sentinel'
    # V1v flag: strict bool, default off, paired_4x8 dual_flash only, real binding scope factory required.
    assert H3LifecycleBackend().reuse_video_vae is False
    for bad in (1, 'yes', None):
        with pytest.raises(TypeError, match='bool'): H3LifecycleBackend(reuse_video_vae=bad)
        with pytest.raises(TypeError, match='bool'): H3DualIterationStages(mode='dual_flash', seed=42, reuse_video_vae=bad)
        with pytest.raises(TypeError, match='bool'):
            two_update.TwoUpdateBackend(args=SimpleNamespace(reward_pipeline=False, profile_rollout=False, reuse_video_vae=bad, reuse_rollout_anchor=False, checkpoint_threads=1))
    with pytest.raises(ValueError, match='binding scope factory'):
        H3DualIterationStages(mode='dual_flash', seed=42, reuse_video_vae=True)
    with pytest.raises(ValueError, match='dual_flash'):
        H3DualIterationStages(mode='dual_flow', seed=42, reuse_video_vae=True, video_reuse_scope=lambda: None)
    reuse_backend = H3LifecycleBackend(reuse_video_vae=True)
    for algorithm, layout in (('cps', 'rank_local'), ('dual_flow', 'paired_4x8'), ('dual_flash', 'rank_local')):
        with pytest.raises(ValueError, match='video reuse requires paired_4x8'):
            reuse_backend._check_reward_pipeline(SimpleNamespace(training=SimpleNamespace(algorithm=algorithm, group_layout=layout)))
    assert two_update.TwoUpdateBackend(args=SimpleNamespace(reward_pipeline=False,profile_rollout=False,reuse_video_vae=True,reuse_rollout_anchor=False, checkpoint_threads=1)).reuse_video_vae is True
    with pytest.raises(ValueError, match='dual_flash'):
        H3DualIterationStages(mode='dual_flow', seed=42, reuse_rollout_anchor=True)
    with pytest.raises(ValueError, match='dual_flash'):
        H3LifecycleBackend(reuse_rollout_anchor=True, checkpoint_threads=1)._check_reward_pipeline(SimpleNamespace(training=SimpleNamespace(
            algorithm='dual_flow', group_layout='paired_4x8')))
    # O1 audit constant: 44 no-grad forwards per update by default (40 rollout + 4 anchor), 40 when reused.
    for reuse, expected in ((False, 44), (True, 40)):
        backend = two_update.TwoUpdateBackend(args=SimpleNamespace(reward_pipeline=False, profile_rollout=False,
                                                                   reuse_video_vae=False, reuse_rollout_anchor=reuse, checkpoint_threads=1))
        assert backend.reuse_rollout_anchor is reuse and backend.nograd_forwards_per_update == expected
    # O1: the real iteration() audit wrapper (not only the constant) with the flag
    # on. Exactly 40 no-grad predicts per update pass for two updates (40, 80);
    # 44 with the flag on, or 40 with it off, is rejected by the same audit.
    import time
    def audited_nograd_counts(*, reuse, nograd, batched=0, rollout_batch=1, selfcheck=False):
        model = torch.nn.Module()
        model.register_parameter('lora', torch.nn.Parameter(torch.ones(2)))
        model.register_parameter('frozen', torch.nn.Parameter(torch.ones(2), requires_grad=False))
        optimizer = torch.optim.AdamW([model.lora], lr=1e-4)
        prepared = SimpleNamespace(optimizer=optimizer, prepared_root=model)
        stages = SimpleNamespace(predict=lambda **kw: model.lora*2, score_one=lambda **kw: 1.)
        if batched: stages.predict_batch = lambda **kw: (model.lora*2,)*4  # S1 batched rollout leaf
        def tiny_iteration(step):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(batched + (1 if step == 0 and selfcheck else 0)):
                with torch.no_grad(): stages.predict_batch(index=2)
            for _ in range(4 if step == 0 and selfcheck else 0):  # S1 startup self-check B=1 forwards
                with torch.no_grad(): stages.predict(index=0)
            for _ in range(nograd):
                with torch.no_grad(): stages.predict(index=2)
            for _ in range(4):
                value = stages.predict(index=2); value.sum().backward(); del value
            norm = float(model.lora.grad.norm()); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            return SimpleNamespace(value=SimpleNamespace(payload=SimpleNamespace(grad_norm=norm)))
        bound = SimpleNamespace(stages=stages, trainer=SimpleNamespace(run_iteration=tiny_iteration))
        root = tmp_path / f'o1-audit-{int(reuse)}-{nograd}-{batched}-{rollout_batch}-{int(selfcheck)}'; monitor = root / 'monitor'; monitor.mkdir(parents=True)
        (monitor / 'health.json').write_text(json.dumps(dict(healthy=True, monotonic_ns=time.monotonic_ns(), sample_count=2)))
        graph = SimpleNamespace(source=SimpleNamespace(state_dict=lambda: dict(next_optimizer_step=0)),
                                profile=SimpleNamespace(launch=SimpleNamespace(output_dir=root)))
        journal = memory.Journal(root / 'memory', rank=0, role='test', device=0, counters=Counters())
        try:
            with monkeypatch.context() as mp:
                mp.setattr(memory, 'journal', lambda: journal)
                mp.setattr(two_update.H3LifecycleBackend, 'bind', lambda *a: bound)
                mp.setattr(two_update, '_phase', lambda session, label, work: work())
                mp.setattr(two_update, '_exchange', lambda *a: [a[3]]*8)
                backend = two_update.TwoUpdateBackend(args=SimpleNamespace(ffmpeg=Path('ffmpeg'), ffprobe=Path('ffprobe'),
                    expected_sha='test', monitor_dir=monitor, reward_pipeline=False, profile_rollout=False,
                    reuse_video_vae=False, reuse_rollout_anchor=reuse, checkpoint_threads=1, rollout_batch=rollout_batch))
                backend.bind(None, SimpleNamespace(rank=0, process_group=None), graph, prepared, None)
                if batched:  # the audit wrapper counts each batched call once and is no-grad only
                    with pytest.raises(RuntimeError, match='no-grad'): stages.predict_batch(index=2)
                bound.trainer.run_iteration(0); bound.trainer.run_iteration(1)
                assert len(backend.audits) == 2 and all(row['graph']['final_prediction_released'] for row in backend.audits)
                assert len(backend.audits[-1]['graph']['grad_forward_indices']) == 8  # cumulative PredictAudit list
                assert json.loads((root / 'update-2-audit.json').read_text())['optimizer_updates'] == 2
                return [row['graph']['nograd_forwards'] for row in backend.audits]
        finally:
            journal.close()
    assert audited_nograd_counts(reuse=True, nograd=40) == [40, 80]
    assert audited_nograd_counts(reuse=False, nograd=44) == [44, 88]
    for reuse, nograd in ((True, 44), (False, 40)):
        with pytest.raises(ValueError, match='graph/count/release audit'):
            audited_nograd_counts(reuse=reuse, nograd=nograd)
    # S1: 10 batched rollout forwards + 4 B=1 anchor forwards = 14 per update
    # through the same real audit wrapper; B1 constants reject 14, B4 rejects 44.
    # Plus the mandatory once-per-run S1 row self-check (1 batched + 4 B=1): [19, 33].
    assert audited_nograd_counts(reuse=False, nograd=4, batched=10, rollout_batch=4, selfcheck=True) == [19, 33]
    for nograd, batched, rollout_batch, selfcheck in ((44, 0, 4, False), (40, 4, 4, False), (4, 10, 1, False),
                                                      (4, 10, 4, False), (4, 10, 1, True)):
        with pytest.raises(ValueError, match='graph/count/release audit'):
            audited_nograd_counts(reuse=False, nograd=nograd, batched=batched, rollout_batch=rollout_batch, selfcheck=selfcheck)
    launcher_args = dict(reward_pipeline=True, profile_rollout=False, reuse_video_vae=False, reuse_rollout_anchor=False, checkpoint_threads=1)
    assert two_update.TwoUpdateBackend(args=SimpleNamespace(**launcher_args)).reward_pipeline is True
    assert two_update.TwoUpdateBackend(args=SimpleNamespace(**dict(launcher_args, reward_pipeline=False))).reward_pipeline is False
    assert two_update.TwoUpdateBackend(args=SimpleNamespace(**launcher_args)).checkpoint_threads == 1
    assert two_update.TwoUpdateBackend(args=SimpleNamespace(**dict(launcher_args, reuse_rollout_anchor=False, checkpoint_threads=4))).checkpoint_threads == 4
    with monkeypatch.context() as patch:
        patch.setattr(h3_world, '_exchange', lambda label, rank, group, payload, ok: [payload]*8)
        source = SimpleNamespace(reserve=lambda step: None, commit=lambda r: None, abort=lambda r: None,
                                 read_behavior_id=lambda: 'b')
        leaf = lambda *args, **kwargs: None
        backend = H3LifecycleBackend(reward_pipeline=True)
        backend.binding = SimpleNamespace(predict=leaf, decode=leaf, score_one=leaf, coordination_factory=leaf)
        backend.transaction = object()
        graph = SimpleNamespace(source=source, profile=SimpleNamespace(training=SimpleNamespace(
            algorithm='dual_flash', group_layout='paired_4x8'), launch=SimpleNamespace(seed=42,
                budgets=SimpleNamespace(media_bytes=1024, decode_bytes=2048))))
        runtime = SimpleNamespace(rank=0, process_group=object(), device='cpu')
        bound = backend.bind(None, runtime, graph, object(), None)
        assert bound.stages.reward_pipeline is True
        assert (bound.stages.media_bytes, bound.stages.decode_bytes) == (1024, 2048)
        assert bound.stages.score_one is backend.binding.score_one
        assert bound.stages.reuse_rollout_anchor is False
        bound.close()
        backend = H3LifecycleBackend(reward_pipeline=True, reuse_video_vae=True)
        backend.binding = SimpleNamespace(predict=leaf, decode=leaf, score_one=leaf, coordination_factory=leaf, video_reuse_scope=leaf)
        backend.transaction = object()
        bound = backend.bind(None, runtime, graph, object(), None)
        assert bound.stages.reuse_video_vae is True and bound.stages.video_reuse_scope is leaf
        bound.close()
        reuse_backend = H3LifecycleBackend(reward_pipeline=True, reuse_rollout_anchor=True)
        reuse_backend.binding = backend.binding; reuse_backend.transaction = backend.transaction
        reuse_bound = reuse_backend.bind(None, runtime, graph, object(), None)
        assert reuse_bound.stages.reuse_rollout_anchor is True and reuse_bound.stages.reward_pipeline is True
        reuse_bound.close()
        # S1 bind threading: B=4 stages get the binding's batch predict and the
        # declared tolerance; the default bind passes neither (unchanged B1 stages).
        batch_leaf = lambda *args, **kwargs: None
        s1_bind = H3LifecycleBackend(rollout_batch=4, rollout_batch_replay_tolerance=1e-3)
        s1_bind.binding = SimpleNamespace(predict=leaf, decode=leaf, score_one=leaf, coordination_factory=leaf,
                                          predict_batch=batch_leaf)
        s1_bind.transaction = object()
        s1_graph = SimpleNamespace(source=source, profile=SimpleNamespace(training=SimpleNamespace(
            algorithm='dual_flash', group_layout='paired_4x8', rollout_batch=4, rollout_batch_replay_tolerance=1e-3),
            launch=graph.profile.launch))
        s1_bound = s1_bind.bind(None, runtime, s1_graph, object(), None)
        assert (s1_bound.stages.rollout_batch, s1_bound.stages.rollout_batch_replay_tolerance) == (4, 1e-3)
        assert s1_bound.stages.predict_batch is batch_leaf and s1_bound.stages.reuse_rollout_anchor is False
        s1_bound.close()
        assert (bound.stages.rollout_batch, bound.stages.predict_batch) == (1, None)
        # Real CLI parser, help exits before any Git/preflight/GPU side effect.
        import sys
        patch.setattr(sys, 'argv', ['launch_h3_resolution_memory', '--reward-pipeline', '--profile-rollout', '--reuse-video-vae', '--reuse-rollout-anchor', '--help'])
        with pytest.raises(SystemExit) as help_result: two_update.main()
        assert help_result.value.code == 0
        import io
        from contextlib import redirect_stdout
        help_output = io.StringIO()
        patch.setattr(sys, 'argv', ['launch_h3_resolution_memory', '--boundary-content-check', 'sentinel', '--help'])
        with redirect_stdout(help_output), pytest.raises(SystemExit) as sentinel_help:
            two_update.main()
        assert sentinel_help.value.code == 0
        assert '--boundary-content-check {full,sentinel}' in help_output.getvalue()
        assert 'default: full' in help_output.getvalue()
        patch.setattr(sys, 'argv', ['launch_h3_resolution_memory', '--boundary-content-check', 'unknown'])
        with pytest.raises(SystemExit) as invalid_mode: two_update.main()
        assert invalid_mode.value.code == 2
        # The parsed namespace carries the strict int default; the spy exits
        # before main()'s Git/preflight/GPU side effects.
        import argparse
        parsed = []
        real_parse_args = argparse.ArgumentParser.parse_args
        def spy_parse_args(self, *args, **kwargs):
            parsed.append(real_parse_args(self, *args, **kwargs))
            raise SystemExit(0)
        patch.setattr(argparse.ArgumentParser, 'parse_args', spy_parse_args)
        required = ['launch_h3_resolution_memory', 'profile.yaml', '--expected-sha', 'x',
                    '--ffmpeg', 'f', '--ffprobe', 'p', '--monitor-dir', 'm']
        for argv in (required, required + ['--checkpoint-threads', '4'],
                     required + ['--rollout-batch', '4', '--rollout-batch-replay-tolerance', '1e-3']):
            patch.setattr(sys, 'argv', argv)
            with pytest.raises(SystemExit): two_update.main()
        assert [row.checkpoint_threads for row in parsed] == [1, 4, 1]
        assert [(row.rollout_batch, row.rollout_batch_replay_tolerance) for row in parsed] == [(1, 1e-4), (1, 1e-4), (4, 1e-3)]
    with monkeypatch.context() as patch:
        def no_memory(n): raise BlockingIOError('host pipeline budget insufficient')
        patch.setattr(h3_loading, '_admit_host_memory', no_memory)
        with pytest.raises(BlockingIOError, match='budget'):
            pipeline.reward_pipeline(samples, decode=lambda s: pytest.fail('decode before admission'),
                score=lambda *a: pytest.fail('score before admission'), media_bytes=1024, decode_bytes=2048)
    assert not any(t.name == 'h3-reward-consumer' for t in threading.enumerate())

    # F0: real rollout math with only CUDA/CUPTI profiler and file-wait leaves
    # substituted. These synthetic activity events are NOT GPU evidence.
    from visual_rl.runtime import h3_rollout_profile as profiling
    from visual_rl.algorithms.trainer.h3_dual_rollout import collect
    from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
    from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
    from visual_rl.models.h3_joint import H3JointLayout
    import hashlib
    uuid = 'GPU-0e1cd165-126d-e889-2b87-240d8faddddb'
    trace_raw = '{"test_only_synthetic_activity_leaf":true,"traceEvents":[]}'
    profile_calls, profile_gates = [], []
    fault = None
    class ProfilerLeaf:
        def start(self):
            assert threading.current_thread() is threading.main_thread()
            profile_calls.append('start')
            if fault == 'start': raise RuntimeError('start failed')
        def stop(self):
            assert threading.current_thread() is threading.main_thread()
            profile_calls.append('stop')
            if fault in ('stop', 'body-stop'): raise RuntimeError('stop failed')
        def events(self):
            assert threading.current_thread() is threading.main_thread()
            kinds = [torch.autograd.DeviceType.CPU]
            if fault != 'cpu-only': kinds.append(torch.autograd.DeviceType.CUDA)
            return [SimpleNamespace(device_type=k) for k in kinds]
        def export_chrome_trace(self, path):
            assert threading.current_thread().name == 'cpu-profile-export'
            profile_calls.append('export')
            Path(path).write_text(trace_raw)
            if fault == 'export': raise OSError('export failed')
    def factory():
        assert threading.current_thread() is threading.main_thread()
        profile_calls.append('factory')
        return ProfilerLeaf()
    def file_worker(action, **kwargs):
        assert threading.current_thread() is threading.main_thread()
        results, errors = [], []
        def execute():
            try: results.append(action())
            except BaseException as exc: errors.append(exc)
        worker = Thread(target=execute, name='cpu-profile-export'); worker.start(); worker.join(10)
        assert not worker.is_alive()
        if errors: raise errors[0]
        return results[0]
    def profile_gate(event, **kwargs):
        profile_gates.append((event, kwargs.get('local_error', False)))
        if kwargs.get('local_error'): raise RuntimeError('profile peer rejected')
        if (fault, event) in (('peer-start', 'rollout-profile/start'), ('peer-stop', 'rollout-profile/stop')):
            raise RuntimeError('another rank failed profiling')
    def recorder(label):
        output = tmp_path / label; output.mkdir()
        return profiling.RolloutCellProfile(output_dir=output, rank=0, gpu_uuid=uuid,
            launcher_sha='e'*40, process_group=object())
    target = dict(optimizer_step=1, row=1, index=0, sample_id='sample-second', selected_index=3,
                  gate=profile_gate)
    with monkeypatch.context() as patch:
        patch.setattr(torch.profiler, 'supported_activities', lambda: {
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA})
        patch.setattr(profiling, '_new_profiler', factory)
        patch.setattr(profiling, 'file_phase', file_worker)
        dynamics = H3DualFlowSDE(H3JointLayout(1, 2, 1, 3), H3ScheduleSnapshot.build(10))
        initial = {s: torch.full((1, 5), .25, dtype=torch.float16) for s in ('a', 'b')}
        def run_rollouts(scope):
            generators = {s: (torch.Generator().manual_seed(i*2), torch.Generator().manual_seed(i*2+1))
                          for i, s in enumerate(('a', 'b'))}
            outputs, calls, gates = [], [], []
            def predict(sid, x, index):
                calls.append((sid, index)); return torch.zeros_like(x)
            def gate(event, **kwargs):
                gates.append((event, kwargs.get('local_error', False)))
                profile_gate(event, **kwargs)
            for step in range(2):
                outputs.append(collect(dynamics=dynamics, samples=(('a', 'g'), ('b', 'g')),
                    initial_states=initial, generators=generators, mode='flash', seed=42, optimizer_step=step,
                    device='cpu', predict=predict, read_behavior_id=lambda: 'behavior', behavior_id='behavior',
                    gate=gate, cell_scope=scope))
            states = {sid: tuple(g.get_state() for g in pair) for sid, pair in generators.items()}
            return outputs, states, calls, gates
        def forbidden_factory(): pytest.fail('disabled profile constructed/probed profiler')
        with patch.context() as off:
            off.setattr(profiling, '_new_profiler', forbidden_factory)
            off.setattr(torch.profiler, 'supported_activities', forbidden_factory)
            before_rng = RankRNGSnapshot.capture_current(0)
            baseline = run_rollouts(None)
            assert RankRNGSnapshot.capture_current(0) == before_rng
            assert not (tmp_path / 'disabled-profile').exists()
        rec = recorder('profile-success')
        before_rng = RankRNGSnapshot.capture_current(0)
        observed = run_rollouts(rec.cell)
        assert RankRNGSnapshot.capture_current(0) == before_rng
        assert profile_calls == ['factory', 'start', 'stop', 'export']
        assert rec.hits == 1
        rec.assert_complete()
        for (records_a, terminal_a, choices_a), (records_b, terminal_b, choices_b) in zip(baseline[0], observed[0]):
            assert choices_a == choices_b and len(records_a) == len(records_b) == 2
            for a, b in zip(records_a, records_b):
                assert (a.sample_id, a.index) == (b.sample_id, b.index)
                assert all(torch.equal(getattr(a, k), getattr(b, k)) for k in ('x', 'action', 'old_joint'))
            assert all(torch.equal(terminal_a[s], terminal_b[s]) for s in initial)
        assert baseline[2] == observed[2] and len(observed[2]) == 40
        assert baseline[3] == [g for g in observed[3] if not g[0].startswith('rollout-profile/')]
        assert all(torch.equal(a, b) for sid in initial for a, b in zip(baseline[1][sid], observed[1][sid]))
        receipt = json.loads(rec.receipt.read_text())
        assert receipt['actual']['sample_id'] == 'b' and receipt['target'] == profiling.TARGET
        assert receipt['admitted_gpu_uuid'] == uuid and receipt['launcher_sha'] == 'e'*40
        assert receipt['trace_sha256'] == hashlib.sha256(trace_raw.encode()).hexdigest()
        assert receipt['trace_bytes'] == len(trace_raw) and receipt['observed_device_events'] == {'CPU': 1, 'CUDA': 1}
        assert receipt['settings'] == profiling.SETTINGS and not receipt['coverage']['row_initial_h2d']
        assert set(p.name for p in rec.directory.iterdir()) == {rec.trace.name, rec.receipt.name}
        with pytest.raises(RuntimeError, match='already attempted'):
            with rec.cell(**target): pytest.fail('duplicate target replayed')
        with pytest.raises(RuntimeError, match='exactly one'): rec.assert_complete()
        with pytest.raises(FileExistsError):
            profiling.RolloutCellProfile(output_dir=rec.directory.parent.parent, rank=0, gpu_uuid=uuid,
                                         launcher_sha='e'*40, process_group=object())
        missed = recorder('profile-missed')
        with missed.cell(**dict(target, optimizer_step=0)): pass
        with pytest.raises(RuntimeError, match='exactly one'): missed.assert_complete()
        # Public support failure cannot become an automatic CPU fallback.
        with patch.context() as unsupported:
            unsupported.setattr(torch.profiler, 'supported_activities', lambda: {torch.profiler.ProfilerActivity.CPU})
            with pytest.raises(RuntimeError, match='no CPU fallback'): recorder('profile-unsupported')
        for fault in ('start', 'stop', 'peer-start', 'peer-stop', 'body-stop', 'cpu-only', 'export', 'oversize', 'receipt-size', 'receipt-io'):
            profile_calls.clear(); profile_gates.clear()
            rec = recorder('profile-failure-' + fault)
            body, after = [], []
            with patch.context() as fail_patch:
                if fault == 'oversize': fail_patch.setattr(profiling, 'TRACE_LIMIT', 1)
                if fault == 'receipt-size': fail_patch.setattr(profiling, 'RECEIPT_LIMIT', 1)
                if fault == 'receipt-io':
                    link = profiling.os.link
                    def failed_link(src, dst, **kwargs):
                        if str(dst).endswith('.receipt.json'): raise OSError('receipt IO failed')
                        return link(src, dst, **kwargs)
                    fail_patch.setattr(profiling.os, 'link', failed_link)
                with pytest.raises((RuntimeError, ValueError, OSError)) as error:
                    with rec.cell(**target):
                        body.append(True)
                        if fault == 'body-stop': raise ValueError('original model error')
                    after.append(True)
                assert not after and len(body) == (0 if fault in ('start', 'peer-start') else 1)
                assert profile_calls.count('stop') == 1  # cleanup never retries/replays
                if fault == 'body-stop':
                    assert isinstance(error.value, ValueError) and 'original model error' in str(error.value)
                    assert any('stop failed' in note for note in error.value.__notes__)
                    assert profile_gates == [('rollout-profile/start', False)]  # no rescue collective
                if fault == 'start': assert profile_gates[0] == ('rollout-profile/start', True)
                if fault in ('stop', 'cpu-only'): assert profile_gates[-1] == ('rollout-profile/stop', True)
                assert not rec.receipt.exists() and not rec.trace.exists()
                assert list(rec.directory.iterdir()) == []
                assert rec.receipt_sha is None
        fault = None
        for symlink in (False, True):
            rec = recorder('profile-existing-' + str(symlink))
            unknown = tmp_path / ('unknown-profile-' + str(symlink)); unknown.write_text('preserve me')
            if symlink: rec.trace.symlink_to(unknown)
            else: rec.trace.write_text('preserve me')
            with pytest.raises(FileExistsError):
                with rec.cell(**target): pass
            assert rec.trace.read_text() == unknown.read_text() == 'preserve me'
            assert not rec.receipt.exists()
        output = tmp_path / 'profile-symlink-root'; output.mkdir()
        outside = tmp_path / 'preserved-profile-directory'; outside.mkdir()
        (output / 'rollout-profile').symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match='symlink'):
            profiling.RolloutCellProfile(output_dir=output, rank=0, gpu_uuid=uuid,
                                         launcher_sha='e'*40, process_group=object())
        assert list(outside.iterdir()) == []
        # No large trace read_bytes: publication already streamed/hash-checked;
        # final receipt read is bounded and detects tampering.
        rec = recorder('profile-no-readbytes')
        with patch.context() as bounded:
            bounded.setattr(Path, 'read_bytes', lambda path: pytest.fail('unbounded profile read_bytes'))
            with rec.cell(**target): pass
            rec.assert_complete()
        assert {'rollout_batch', 'sample_ids'}.isdisjoint(json.loads(rec.receipt.read_text())['actual'])
        rec.receipt.write_text('{}')
        with pytest.raises(ValueError, match='integrity'): rec.assert_complete()
        # S1: a batched target cell names its batch and every traced sample.
        rec = recorder('profile-batched-cell')
        batch_ids = ('s0', target['sample_id'], 's2', 's3')
        with rec.cell(**target, batch_sample_ids=batch_ids): pass
        rec.assert_complete()
        actual = json.loads(rec.receipt.read_text())['actual']
        assert (actual['rollout_batch'], actual['sample_ids'], actual['row']) == (4, list(batch_ids), target['row'])
        rec = recorder('profile-batched-bad')
        with pytest.raises(ValueError, match='batched sample identities'):
            with rec.cell(**target, batch_sample_ids=('s0', 's2')): pass
        # Only the main thread may enter the target; no body execution/replay.
        rec = recorder('profile-thread')
        errors = []
        def wrong_thread():
            try:
                with rec.cell(**target): pytest.fail('profile ran in file/consumer thread')
            except BaseException as exc: errors.append(exc)
        worker = Thread(target=wrong_thread); worker.start(); worker.join(5)
        assert not worker.is_alive() and len(errors) == 1 and 'main training thread' in str(errors[0])
    # Start/stop failures at the actual collect target never advance to the
    # following prediction. No replay or extra warmup/model call is introduced.
    with monkeypatch.context() as patch:
        patch.setattr(torch.profiler, 'supported_activities', lambda: {
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA})
        patch.setattr(profiling, '_new_profiler', factory)
        for fault, expected in (('start', 10), ('stop', 11)):
            rec = recorder('collect-profile-' + fault)
            predictions = []
            def predict(sid, x, index):
                predictions.append((sid, index)); return torch.zeros_like(x)
            generators = {s: (torch.Generator().manual_seed(i*2), torch.Generator().manual_seed(i*2+1))
                          for i, s in enumerate(('a', 'b'))}
            with pytest.raises(RuntimeError, match=f'{fault} failed'):
                collect(dynamics=dynamics, samples=(('a', 'g'), ('b', 'g')),
                    initial_states=initial, generators=generators, mode='flash', seed=42, optimizer_step=1,
                    device='cpu', predict=predict, read_behavior_id=lambda: 'behavior', behavior_id='behavior',
                    gate=profile_gate, cell_scope=rec.cell)
            assert len(predictions) == expected
        fault = None
        patch.setattr(profiling, 'file_phase', file_worker)
        rec = recorder('profile-dual-stage')
        predictions = []
        data = SimpleNamespace(reserve=lambda step: None, commit=lambda r: None, abort=lambda r: None)
        prepared = SimpleNamespace(prepared_root=torch.nn.Identity())
        def predict(prepared, *, sample_id, x, index, **kwargs):
            predictions.append((sample_id, index)); return torch.zeros_like(x)
        stage = H3DualIterationStages(mode='dual_flash', seed=42, data=data, prepared=prepared,
            predict=predict, decode=lambda t: None, score_one=lambda **kw: 1.,
            coordination_factory=lambda plan: None, process_group=None, rank=0, device='cpu',
            read_behavior_id=lambda: 'behavior', rollout_cell_scope=rec.cell)
        identity = IterationIdentity(1, 'source', 'train', ('a', 'b'), ('g', 'g'), (0, 1))
        request = SimpleNamespace(identity=identity, samples=(('a', 'g', 0), ('b', 'g', 1)),
            initial_states=initial, generators={s: (torch.Generator().manual_seed(i*2),
            torch.Generator().manual_seed(i*2+1)) for i, s in enumerate(('a', 'b'))},
            behavior_id='behavior', conditions={s: torch.zeros(1) for s in initial}, dynamics=dynamics)
        ctx = SimpleNamespace(request=request, dual=dynamics, coordinator=SimpleNamespace(gate=profile_gate))
        stage._active = ctx
        stage.rollout(StageValue(identity, ctx))
        rec.assert_complete()
        assert rec.hits == 1 and len(predictions) == 20 and len(ctx.dual_records) == 2
        # O1: the real optimize stage with the real anchor on the records above.
        # Off path: one anchor no-grad forward per record before the grad forward;
        # on path: zero, with a bit-identical loss/update, the rank-consistent
        # 'dual/anchor/reuse/1' gate and the unchanged initial-ratio audit.
        import io, contextlib
        patch.setattr(torch.distributed, 'all_reduce', lambda value, **kwargs: value)
        def optimize_once(reuse):
            lora = torch.nn.Parameter(torch.zeros(5)); optimizer = torch.optim.AdamW([lora], lr=1e-2)
            calls, gates, out = [], [], io.StringIO()
            def predict_lora(prepared, *, sample_id, x, index, **kwargs):
                calls.append((sample_id, index, torch.is_grad_enabled())); return x.float()*0 + lora
            trained = H3DualIterationStages(mode='dual_flash', seed=42, data=data, predict=predict_lora,
                prepared=SimpleNamespace(optimizer=optimizer, prepared_root=torch.nn.Identity(), reducer_divisor=8, scheduler=None),
                decode=lambda t: None, score_one=lambda **kw: 1., coordination_factory=lambda plan: None,
                process_group=None, rank=0, device='cpu', read_behavior_id=lambda: 'behavior', reuse_rollout_anchor=reuse)
            run = SimpleNamespace(request=SimpleNamespace(identity=identity, samples=request.samples, behavior_id='behavior',
                    conditions=request.conditions, dynamics=dynamics, max_cells=2, collection_id='c', next_behavior_id='n'),
                dual=dynamics, dual_records=ctx.dual_records, plan=SimpleNamespace(samples=request.samples),
                advantages={'a': torch.tensor(.5), 'b': torch.tensor(-.5)},
                coordinator=SimpleNamespace(gate=lambda event, **kw: gates.append((event, kw.get('local_error', False))),
                                            clip_grad_norm=lambda prepared, max_norm: float(lora.grad.norm())))
            trained._active = run
            with contextlib.redirect_stdout(out): result = trained.optimize(StageValue(identity, run))
            return calls, gates, out.getvalue(), result.payload, lora.detach().clone()
        off, on = optimize_once(False), optimize_once(True)
        nograd = lambda calls: [(sid, i) for sid, i, grad in calls if not grad]
        assert nograd(off[0]) == [(r.sample_id, r.index) for r in ctx.dual_records] and nograd(on[0]) == []
        assert [c for c in off[0] if c[2]] == [c for c in on[0] if c[2]] and len(on[0]) == 2
        assert ('dual/anchor/reuse/0', False) in off[1] and ('dual/anchor/reuse/1', False) in on[1]
        per_record = lambda gates: [e for e, _ in gates if e.startswith('dual/anchor/') and e.split('/')[2].isdigit()]
        assert per_record(off[1]) == [f'dual/anchor/{i}/{name}' for i in range(2) for name in ('forward', 'prepare', 'result')]
        assert per_record(on[1]) == ['dual/anchor/0/reuse', 'dual/anchor/1/reuse']
        assert [e for e, _ in off[1] if e.startswith('dual/current/')] == [e for e, _ in on[1] if e.startswith('dual/current/')]
        assert off[2].count('h3_initial_policy_ratio') == 2 == on[2].count('h3_initial_policy_ratio')
        assert 'h3_rollout_replay_score_check' in off[2] and 'h3_rollout_replay_score_check' not in on[2]
        assert off[3].scaled_loss == on[3].scaled_loss and off[3].grad_norm == on[3].grad_norm and torch.equal(off[4], on[4])
        assert off[3].local_cells == 2 and off[3].grad_norm > 0

    # Actual two-update binding, including off/no-probe, peer agreement,
    # installation and missed-window final gate. Parent model construction
    # and transport alone are CPU leaves; bind itself is not mocked.
    with monkeypatch.context() as patch:
        parent_calls, configs = [], []
        def parent_bind(self, preflight, runtime, graph, prepared, transformed):
            return SimpleNamespace(graph=graph, stages=SimpleNamespace(predict=lambda **kw: None,
                score_one=lambda **kw: 1., rollout_cell_scope=None),
                trainer=SimpleNamespace(run_iteration=lambda step: None))
        patch.setattr(two_update.H3LifecycleBackend, 'bind', parent_bind)
        patch.setattr(two_update.H3LifecycleBackend, 'checkpoint', lambda *args: parent_calls.append('finalized'))
        patch.setattr(two_update, '_phase', lambda session, label, work: work())
        def exchange(label, rank, group, payload, ok):
            configs.append(payload)
            return [payload]*8
        patch.setattr(two_update, '_exchange', exchange)
        runtime = SimpleNamespace(rank=0, process_group=object(), gpu_uuids=(uuid,)*8)
        for enabled in (False, True):
            output = tmp_path / ('profile-backend-' + str(enabled)); output.mkdir()
            graph = SimpleNamespace(session=runtime, source=SimpleNamespace(),
                                    profile=SimpleNamespace(launch=SimpleNamespace(output_dir=output)))
            args = SimpleNamespace(reward_pipeline=True, profile_rollout=enabled, reuse_video_vae=False, reuse_rollout_anchor=False, checkpoint_threads=1, expected_sha='e'*40,
                ffmpeg=Path('unused'), ffprobe=Path('unused'), monitor_dir=tmp_path / 'unused')
            backend = two_update.TwoUpdateBackend(args=args)
            assert backend.reward_pipeline is True
            with patch.context() as capability:
                if not enabled:
                    capability.setattr(torch.profiler, 'supported_activities', forbidden_factory)
                else:
                    capability.setattr(torch.profiler, 'supported_activities', lambda: {
                        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA})
                bound = backend.bind(None, runtime, graph, object(), None)
                assert configs[-1] == profiling.configuration(enabled)
                if not enabled:
                    assert bound.stages.rollout_cell_scope is None and not (output / 'rollout-profile').exists()
                    backend.checkpoint(bound, None)
                else:
                    assert bound.stages.rollout_cell_scope.__self__ is backend.rollout_profile
                    assert isinstance(bound.stages.predict, two_update.PredictAudit)
                    with pytest.raises(RuntimeError, match='exactly one'): backend.checkpoint(bound, None)
        assert parent_calls == ['finalized']
        # A peer's different flag must fail before attach/profiler support.
        def disagree(label, rank, group, payload, ok):
            return [payload]*7 + [dict(payload, enabled=not payload['enabled'])]
        patch.setattr(two_update, '_exchange', disagree)
        patch.setattr(torch.profiler, 'supported_activities', forbidden_factory)
        with pytest.raises(ValueError, match='configuration mismatch'):
            backend.bind(None, runtime, graph, object(), None)
    for bad in (1, 'yes', None):
        with pytest.raises(TypeError, match='bool'): profiling.configuration(bad)
        with pytest.raises(TypeError, match='bool'):
            two_update.TwoUpdateBackend(args=SimpleNamespace(reward_pipeline=False, profile_rollout=bad, reuse_video_vae=False, reuse_rollout_anchor=False, checkpoint_threads=1))

    # H1: actual backend constructor/identity/_score and offline Handler. Only
    # the heavy inferencer, CUDA observations/transfers/clock/host admission
    # and HTTP socket are leaves. No GPU or service request occurs here.
    from scripts import serve_hpsv3_video as service
    from visual_rl.core.protocols.flash_hpsv3 import canonical_json, validate_health, validate_score, aggregate_frame_scores
    from io import BytesIO
    from PIL import Image
    import base64, sys, types
    hps_root = tmp_path / 'hps-residency'; hps_root.mkdir()
    checkpoint, config = hps_root / 'tiny.safetensors', hps_root / 'config.json'
    checkpoint.write_bytes(b'CPU inferencer leaf'); config.write_text('{}')
    module_file = hps_root / '__init__.py'; module_file.write_text('# CPU-only HPS heavy-model leaf\n')
    active, clock = [None], [0.]
    class TinyHpsModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fp32 = torch.nn.Parameter(torch.tensor([.25], dtype=torch.float32))
            self.fp16 = torch.nn.Parameter(torch.tensor([.5], dtype=torch.float16))
            self.logical = 'cpu'; self.moves = []; self.fail_move = None
        def parameters(self, recurse=True):
            for p in super().parameters(recurse=recurse):
                yield p if self.logical == 'cpu' else SimpleNamespace(
                    device=torch.device('cuda:0'), dtype=p.dtype, requires_grad=p.requires_grad)
        def buffers(self, recurse=True): return iter(())
        def to(self, device):
            assert device in ('cuda', 'cpu')  # no dtype/cast/batch/CPU inference fallback
            self.moves.append(device)
            if self.fail_move == device: raise RuntimeError('model transfer ' + device + ' failed')
            self.logical = device
            if device == 'cpu' and hasattr(self, 'controller'): self.controller.low_until_offload = False
            return self
    class InferencerLeaf:
        def __init__(self, *, device, checkpoint_path, config_path):
            assert device == 'cpu'
            self.device = device; self.model = TinyHpsModel(); self.calls = []; self.failure = None
            self.processor = SimpleNamespace(tokenizer=SimpleNamespace(
                backend_tokenizer=SimpleNamespace(to_str=lambda: '{}'), special_tokens_map={}),
                chat_template='test', image_processor=SimpleNamespace(to_dict=lambda: {'cpu_test': True}))
            self.use_special_tokens = False; self.data_config = SimpleNamespace(test=True)
        def reward(self, *, prompts, image_paths):
            assert self.device == 'cuda' and len(prompts) == len(image_paths) == 1
            assert torch.is_inference_mode_enabled() and not self.model.training
            self.calls.append((prompts[0], image_paths[0].tobytes()))
            if self.failure == 'reward': raise ValueError('original reward failed')
            if self.failure == 'monitor':
                assert self.model.controller.monitor_checked.wait(5)
            if self.failure == 'margin': self.model.controller.free_gpu = 15.
            if self.failure == 'runtime-telemetry': self.model.controller.bad_memory = True
            value = float(image_paths[0].getpixel((0, 0))[0]) / 255
            return (torch.tensor([[value]], dtype=torch.float32) + self.model.fp32 + self.model.fp16.float(),)
    class CudaLeaf:
        def __init__(self, backend):
            self.backend = backend; self.free_cpu = 60.; self.free_gpu = 48.
            self.low_until_offload = False; self.empty = 0; self.bad_memory = False
            self.uuid = uuid; self.device = 0; self.monitor_error = False; self.monitor_checked = Event()
        def memory(self):
            if threading.current_thread().name == 'hps-request-memory' and self.monitor_error:
                self.monitor_checked.set(); raise RuntimeError('monitor observation failed')
            if self.bad_memory: return float('nan'), 64*1024**3
            free = self.free_cpu if self.backend.inferencer.model.logical == 'cpu' else (
                39. if self.low_until_offload else self.free_gpu)
            return int(free*1024**3), 64*1024**3
        def empty_cache(self): self.empty += 1
    with monkeypatch.context() as hp:
        hp.setenv('CUDA_VISIBLE_DEVICES', uuid)
        hp.setenv('HF_HUB_OFFLINE', '1'); hp.setenv('TRANSFORMERS_OFFLINE', '1')
        module = types.ModuleType('hpsv3'); module.__file__ = str(module_file)
        module.HPSv3RewardInferencer = InferencerLeaf
        hp.setitem(sys.modules, 'hpsv3', module)
        read_text = Path.read_text
        def host_memory(path, *args, **kwargs):
            if path == Path('/proc/meminfo'): return 'MemAvailable: 104857600 kB\n'
            return read_text(path, *args, **kwargs)
        hp.setattr(Path, 'read_text', host_memory)
        hp.setattr(service.time, 'monotonic', lambda: clock[0])
        hp.setattr(torch.cuda, 'mem_get_info', lambda: active[0].cuda.memory())
        hp.setattr(torch.cuda, 'current_device', lambda: active[0].cuda.device)
        hp.setattr(torch.cuda, 'get_device_properties', lambda n: SimpleNamespace(uuid=active[0].cuda.uuid))
        hp.setattr(torch.cuda, 'synchronize', lambda: None)
        hp.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
        hp.setattr(torch.cuda, 'max_memory_allocated', lambda: 1024**3)
        hp.setattr(torch.cuda, 'empty_cache', lambda: active[0].cuda.empty_cache())
        contract = VideoContract(124, 32, 32)
        def new_backend(enabled):
            clock[0] = 0.
            backend = service.HpsBackend(checkpoint, config, contract, 40., gpu_resident=enabled,
                                         resident_gpu_uuid=uuid if enabled else None)
            backend.cuda = CudaLeaf(backend); backend.inferencer.model.controller = backend.cuda
            active[0] = backend
            return backend
        jpeg_frames = []
        for i in range(124):
            stream = BytesIO(); Image.new('RGB', (32, 32), (i%31, 10, 20)).save(stream, format='JPEG')
            jpeg_frames.append(base64.b64encode(stream.getvalue()).decode())
        def post(backend, *, identity=None):
            active[0] = backend
            data = dict(protocol=service.PROTOCOL, protocol_version=service.VERSION,
                service_identity=identity or backend.health['service_identity'], sample_id='sample',
                prompt=' exact prompt ', jpeg_frames=jpeg_frames)
            raw = canonical_json(data)
            handler = object.__new__(service.make_handler(backend))
            handler.path = '/score'; handler.headers = {'Content-Type': 'application/json', 'Content-Length': str(len(raw))}
            handler.rfile = BytesIO(raw); handler.wfile = BytesIO()
            handler.connection = SimpleNamespace(settimeout=lambda seconds: None)
            statuses = []
            handler.send_response = statuses.append; handler.send_header = lambda *a: None
            handler.end_headers = lambda: None
            handler.do_POST()
            return statuses[0], json.loads(handler.wfile.getvalue())
        off = new_backend(False)
        off_health = canonical_json(off.health)
        expected_hash = service.loaded_tensor_sha(off.inferencer.model)
        before = {n: p.clone() for n, p in off.inferencer.model.state_dict().items()}
        rng = RankRNGSnapshot.capture_current(0)
        status, serial1 = post(off); assert status == 200
        status, serial2 = post(off); assert status == 200
        assert off.inferencer.model.moves == ['cuda', 'cpu', 'cuda', 'cpu'] and off.cuda.empty == 2
        assert off._onloads == off._offloads == 2 and len(off.inferencer.calls) == 248
        assert canonical_json(off.health) == off_health
        with hp.context() as idle_off:
            idle_off.setattr(torch.cuda, 'mem_get_info', lambda: pytest.fail('off idle/close probed CUDA'))
            off.idle_housekeeping(); off.close(); off.close()
        warm = new_backend(True)
        warm_health = canonical_json(warm.health)
        status, resident1 = post(warm); assert status == 200
        status, resident2 = post(warm); assert status == 200
        assert warm.inferencer.model.moves == ['cuda'] and warm._hits == 1
        assert warm._onloads == 1 and warm._offloads == 0 and warm.cuda.empty == 2
        assert warm._onload_attempts == 1 and warm._offload_attempts == 0
        assert len(warm.inferencer.calls) == 248 and warm.inferencer.calls == off.inferencer.calls
        assert RankRNGSnapshot.capture_current(0) == rng
        for response, backend in ((serial1, off), (serial2, off), (resident1, warm), (resident2, warm)):
            score, indices = validate_score(response, sample_id='sample', identity=backend._health['service_identity'], contract=contract)
            assert response['frame_scores'] == resident1['frame_scores']
            import struct
            assert b''.join(struct.pack('!f', x) for x in response['frame_scores']) == b''.join(
                struct.pack('!f', x) for x in resident1['frame_scores'])
            assert struct.pack('!f', score) == struct.pack('!f', resident1['video_score'])
            assert (score, indices) == aggregate_frame_scores(resident1['frame_scores'], contract)
        assert len(indices) == 37 and len(set(resident1['frame_scores'])) < 124  # includes ties
        assert all(torch.equal(before[n], p) and before[n].dtype == p.dtype
                   for n, p in warm.inferencer.model.state_dict().items())
        assert warm.health['identity_facts']['loaded_model_state_sha256'] == expected_hash
        assert warm.health['service_identity'] != off._health['service_identity']
        validate_health(warm.health, warm.health['service_identity'], contract)
        assert canonical_json(warm.health) == warm_health
        assert post(warm, identity=off._health['service_identity'])[0] == 400
        assert len(warm.inferencer.calls) == 248
        # Actual server housekeeping hook on the current HTTP thread, fake clock.
        server = object.__new__(service.HpsHTTPServer); server.backend = warm
        clock[0] = 59.; server.service_actions(); assert warm._offloads == 0
        warm._active = True; clock[0] = 70.; server.service_actions(); assert warm._offloads == 0
        warm._active = False; clock[0] = 60.; server.service_actions()
        assert warm._offloads == 1 and warm._location == 'cpu'
        assert canonical_json(warm.health) == warm_health
        warm.close(); warm.close(); assert warm._offloads == 1
        # Expiry is rechecked at request entry even if an idle tick was missed.
        expired = new_backend(True); assert post(expired)[0] == 200
        clock[0] = 60.; assert post(expired)[0] == 200
        assert expired.inferencer.model.moves == ['cuda', 'cpu', 'cuda']
        expired.close(); assert expired.inferencer.model.moves[-1] == 'cpu'
        # Low warm headroom first releases its real weights, then reads real free.
        fallback = new_backend(True); assert post(fallback)[0] == 200
        fallback.cuda.low_until_offload = True
        assert post(fallback)[0] == 200
        assert fallback.inferencer.model.moves == ['cuda', 'cpu', 'cuda']
        assert len(fallback.inferencer.calls) == 248  # not a POST/inference retry
        fallback.cuda.low_until_offload = True; fallback.cuda.free_cpu = 35.
        assert post(fallback)[0] == 503 and len(fallback.inferencer.calls) == 248
        assert fallback._location == 'cpu'; fallback.close()
        cold = new_backend(True); cold.cuda.free_cpu = 39.
        assert post(cold)[0] == 503 and not cold.inferencer.calls and not cold.inferencer.model.moves
        cold.close()
        low_idle = new_backend(True); assert post(low_idle)[0] == 200
        low_idle.cuda.low_until_offload = True
        server.backend = low_idle; server.service_actions()
        assert low_idle._location == 'cpu' and low_idle._offloads == 1
        low_idle.close()
        # Identity declaration/observation errors poison; future health/POST must
        # not claim ready or reuse an unknown GPU position.
        for problem in ('uuid', 'current', 'cvd', 'telemetry', 'stringable'):
            backend = new_backend(True)
            if problem == 'uuid': backend.cuda.uuid = 'GPU-4a964c9b-4342-56a1-e7d7-84067d4a83ab'
            if problem == 'current': backend.cuda.device = 1
            if problem == 'telemetry': backend.cuda.bad_memory = True
            if problem == 'stringable':
                class UUIDStringable:
                    def __str__(self): return uuid
                backend.cuda.uuid = UUIDStringable()
            with hp.context() as env:
                if problem == 'cvd': env.setenv('CUDA_VISIBLE_DEVICES', '0')
                assert post(backend)[0] == 500
                assert not backend.ready and not backend.inferencer.calls
                assert post(backend)[0] == 503
                handler = object.__new__(service.make_handler(backend)); handler.path = '/healthz'
                replies = []; handler.reply = lambda status, data: replies.append((status, data))
                handler.do_GET(); assert replies[0][0] == 503 and replies[0][1]['status'] == 'unready'
                with pytest.raises(ValueError, match='unready'):
                    validate_health(backend.health, backend.health['service_identity'], contract)
                backend.close()
        for lifecycle in ('warm', 'idle', 'close'):
            backend = new_backend(True); assert post(backend)[0] == 200
            backend.cuda.uuid = 'GPU-4a964c9b-4342-56a1-e7d7-84067d4a83ab'
            if lifecycle == 'warm': assert post(backend)[0] == 500
            else:
                clock[0] = 61.
                with pytest.raises(ValueError, match='UUID'):
                    backend.idle_housekeeping() if lifecycle == 'idle' else backend.close()
            assert not backend.ready and backend.inferencer.model.moves == ['cuda']
            assert post(backend)[0] == 503  # no wrong-device transfer or reuse
            if not backend._closed:
                with pytest.raises(RuntimeError, match='release failed'): backend.close()
            backend.close()
        # Known PyTorch CUuuid representation is normalized, but arbitrary
        # stringable observations above were rejected.
        class KnownCUuuid:
            def __str__(self): return uuid.removeprefix('GPU-')
        with hp.context() as observation:
            observation.setattr(torch._C, '_CUuuid', KnownCUuuid, raising=False)
            backend = new_backend(True); backend.cuda.uuid = KnownCUuuid()
            assert post(backend)[0] == 200
            backend.close()
        # All request failure paths stop/join the actual monitor. Model transfers
        # are observable CPU leaves; failed .to(cpu) is never blindly replayed.
        for failure in ('reward', 'onload', 'offload', 'both', 'margin', 'monitor', 'runtime-telemetry', 'journal'):
            backend = new_backend(True)
            if failure in ('reward', 'both'): backend.inferencer.failure = 'reward'
            if failure == 'onload': backend.inferencer.model.fail_move = 'cuda'
            if failure in ('offload', 'both'): backend.inferencer.model.fail_move = 'cpu'
            if failure == 'offload': backend.cuda.free_gpu = 39.
            if failure in ('margin', 'runtime-telemetry'): backend.inferencer.failure = failure
            if failure == 'monitor':
                backend.cuda.monitor_error = True; backend.inferencer.failure = 'monitor'
            images = [Image.new('RGB', (32, 32)) for _ in range(124)]
            with hp.context() as journal_fault:
                if failure == 'journal':
                    assert post(backend)[0] == 200  # warm before logger fails at score scope entry
                    @contextmanager
                    def broken_span(*args, **kwargs):
                        raise OSError('journal failed')
                        yield
                    journal_fault.setattr(memory, 'span', broken_span)
                with pytest.raises((RuntimeError, ValueError, BlockingIOError, OSError)) as error:
                    backend.score('failure', 'prompt', images)
            if failure == 'both':
                assert isinstance(error.value, ValueError) and 'original reward' in str(error.value)
                assert any('cpu failed' in note for note in error.value.__notes__)
            if failure == 'journal': assert 'journal failed' in str(error.value)
            assert not backend.ready and not backend._active
            assert not any(t.name == 'hps-request-memory' for t in threading.enumerate())
            assert post(backend)[0] == 503
            moves = list(backend.inferencer.model.moves)
            if backend._release_failed:
                with pytest.raises(RuntimeError, match='release failed'): backend.close()
                assert backend.inferencer.model.moves == moves
            else: backend.close()
            backend.close()
        for business_failure in (False, True):
            backend = new_backend(True)
            assert post(backend)[0] == 200
            empty_before = backend.cuda.empty
            with hp.context() as logging_fault:
                original_event = memory.event
                def event(kind, **fields):
                    if kind == 'empty_cache_before': raise OSError('cache journal failed')
                    return original_event(kind, **fields)
                logging_fault.setattr(memory, 'event', event)
                if business_failure:
                    backend.inferencer.failure = 'reward'
                    with pytest.raises(ValueError, match='original reward') as primary:
                        backend.score('failure', 'prompt', [Image.new('RGB', (32, 32))]*124)
                    assert any('cache journal failed' in note for note in primary.value.__notes__)
                else:
                    with pytest.raises(OSError, match='cache journal failed'): backend.close()
            assert backend.cuda.empty == empty_before + 1 and backend._location == 'cpu'
            assert not backend.ready and not any(t.name == 'hps-request-memory' for t in threading.enumerate())
            backend.close()
        for context in ('idle', 'close'):
            backend = new_backend(True); assert post(backend)[0] == 200
            backend.inferencer.model.fail_move = 'cpu'; clock[0] = 61.
            with pytest.raises(RuntimeError, match='cpu failed'):
                backend.idle_housekeeping() if context == 'idle' else backend.close()
            assert not backend.ready and backend._release_failed
            assert backend.inferencer.model.moves == ['cuda', 'cpu']
            if context == 'idle':
                with pytest.raises(RuntimeError, match='release failed'): backend.close()
            backend.close()
        # Original client acquisition rejects a new policy identity when its
        # immutable artifact expects another one; no validator code changed.
        scorer, transport, _ = _scorer(tmp_path / 'hps-policy-mismatch', hp, contract)
        transport.health = warm._health
        with pytest.raises(ValueError, match='identity'): scorer.acquire()
        assert not any(t.name == 'hps-request-memory' for t in threading.enumerate())

        # CLI checks run before heavy loading/socket; main finally/idle wiring
        # is exercised with explicit fake server/backend, never a listening port.
        for argv in (['--help'], ['--checkpoint', str(checkpoint), '--config', str(config),
            '--artifact-output', str(hps_root/'unused'), '--gpu-resident'],
            ['--checkpoint', str(checkpoint), '--config', str(config), '--artifact-output', str(hps_root/'unused'),
             '--resident-gpu-uuid', uuid],
            ['--checkpoint', str(checkpoint), '--config', str(config), '--artifact-output', str(hps_root/'unused'),
             '--min-free-gib', '39']):
            with hp.context() as cli:
                cli.setattr(service, 'HpsBackend', lambda *a, **kw: pytest.fail('invalid/help CLI loaded model'))
                cli.setattr(sys, 'argv', ['serve_hpsv3_video', *argv])
                with pytest.raises(SystemExit): service.main()
        with hp.context() as cli:
            cli.setenv('CUDA_VISIBLE_DEVICES', '0')
            cli.setattr(service, 'HpsBackend', lambda *a, **kw: pytest.fail('wrong CVD reached heavy loading'))
            cli.setattr(sys, 'argv', ['serve_hpsv3_video', '--checkpoint', str(checkpoint), '--config', str(config),
                '--artifact-output', str(hps_root/'unused'), '--gpu-resident', '--resident-gpu-uuid', uuid])
            with pytest.raises(SystemExit): service.main()
        for bad in (None, 1, 'yes'):
            with pytest.raises(TypeError, match='bool'): service.residency_policy(bad)
        for bad_uuid in ('0', uuid.removeprefix('GPU-'), 'GPU-00000000-0000-0000-0000-000000000000'):
            with pytest.raises((ValueError, RuntimeError)): service.residency_policy(True, bad_uuid)
        lifecycle_events = []
        close_failure = False
        idle_failure = False
        idle_action = service.HpsHTTPServer.service_actions
        class BackendLeaf:
            def __init__(self, *args, **kwargs):
                assert kwargs == dict(gpu_resident=True, resident_gpu_uuid=uuid)
                lifecycle_events.append('construct'); self.health = warm._health
            def idle_housekeeping(self):
                lifecycle_events.append('idle')
                if idle_failure: raise RuntimeError('main idle failed')
            def close(self):
                lifecycle_events.append('backend-close')
                if close_failure: raise OSError('backend close failed')
        class ServerLeaf:
            def __init__(self, *args): lifecycle_events.append('socket-leaf')
            def serve_forever(self, *, poll_interval):
                assert poll_interval == .5
                idle_action(self); raise KeyboardInterrupt('fake loop exit')
            def server_close(self):
                lifecycle_events.append('server-close')
                if close_failure: raise OSError('server close failed')
        with hp.context() as cli:
            cli.setattr(service, 'HpsBackend', BackendLeaf); cli.setattr(service, 'HpsHTTPServer', ServerLeaf)
            artifact = hps_root / 'new-manifest.json'
            cli.setattr(sys, 'argv', ['serve_hpsv3_video', '--checkpoint', str(checkpoint), '--config', str(config),
                '--artifact-output', str(artifact), '--gpu-resident', '--resident-gpu-uuid', uuid])
            with pytest.raises(KeyboardInterrupt, match='fake loop exit'): service.main()
            assert lifecycle_events == ['construct', 'socket-leaf', 'idle', 'server-close', 'backend-close']
            artifact.write_text('old manifest preserved')
            lifecycle_events.clear()
            with pytest.raises(ValueError, match='refusing to overwrite'): service.main()
            assert artifact.read_text() == 'old manifest preserved'
            assert lifecycle_events == ['construct', 'socket-leaf', 'server-close', 'backend-close']
            cli.setattr(sys, 'argv', ['serve_hpsv3_video', '--checkpoint', str(checkpoint), '--config', str(config),
                '--artifact-output', str(hps_root / 'cleanup-manifest'), '--gpu-resident', '--resident-gpu-uuid', uuid])
            lifecycle_events.clear(); close_failure = True
            with pytest.raises(KeyboardInterrupt, match='fake loop exit') as primary:
                service.main()
            assert len(primary.value.__notes__) == 2
            assert lifecycle_events[-2:] == ['server-close', 'backend-close']
            lifecycle_events.clear(); close_failure = False; idle_failure = True
            with pytest.raises(RuntimeError, match='main idle failed'): service.main()
            assert lifecycle_events == ['construct', 'socket-leaf', 'idle', 'server-close', 'backend-close']

    # Formal 200-update run: long-run observation budgets are explicit opt-ins;
    # unset environment / default CLI keep the short-acceptance limits.
    from visual_rl.runtime import h3_memory as budget_memory
    from scripts import monitor_h3_memory as budget_monitor
    import sys
    with monkeypatch.context() as patch:
        patch.delenv('H3_MEMORY_MAX_BYTES', raising=False)
        assert budget_memory.max_bytes_from_environment() == 64 * 1024**2
        patch.setenv('H3_MEMORY_MAX_BYTES', str(512 * 1024**2))
        assert budget_memory.max_bytes_from_environment() == 512 * 1024**2
        for bad in ('0', '-1', '1e9', ' 5', str(2**40 + 1)):
            patch.setenv('H3_MEMORY_MAX_BYTES', bad)
            with pytest.raises(ValueError, match='H3_MEMORY_MAX_BYTES'): budget_memory.max_bytes_from_environment()
        runs = []
        patch.setattr(budget_monitor, 'run', lambda *a, **kw: runs.append(kw))
        base = ['monitor_h3_memory', str(tmp_path / 'm'), '--journal-dir', str(tmp_path / 'j')]
        patch.setattr(sys, 'argv', base); budget_monitor.main()
        patch.setattr(sys, 'argv', base + ['--interval', '5', '--max-seconds', '1728000', '--max-bytes', str(8 * 1024**3)])
        budget_monitor.main()
        assert runs == [dict(interval=1., max_bytes=512 * 1024**2, max_seconds=86400., max_consecutive_failures=3),
                        dict(interval=5., max_bytes=8 * 1024**3, max_seconds=1728000., max_consecutive_failures=3)]
        for bad in (['--interval', '0'], ['--max-seconds', 'inf'], ['--max-bytes', '0'], ['--interval', 'nan']):
            patch.setattr(sys, 'argv', base + bad)
            with pytest.raises(SystemExit): budget_monitor.main()

    # Formal reward-ranked video retention: cadence, tie-break, real CPU encode
    # on 8 simulated owner ranks, global selection, and temp cleanup.
    from scripts import h3_video_retention as retention
    from visual_rl.models.implementations.h3_decode import H3AVMedia
    import shutil
    assert retention.export_steps(every=5, terminal_step=200) == list(range(5, 201, 5))
    assert retention.export_steps(every=5, terminal_step=203)[-2:] == [200, 203]
    assert retention.export_steps(every=5, terminal_step=200, after=20)[0] == 25
    assert retention.export_steps(every=5, terminal_step=200, also=(1,))[:3] == [1, 5, 10]
    assert len(retention.export_steps(every=5, terminal_step=200, also=(1,))) == 41
    assert retention.export_steps(every=5, terminal_step=200, after=1, also=(1,))[0] == 5
    with pytest.raises(ValueError, match='also'): retention.export_step(1, every=5, terminal_step=200, also=(0,))
    for bad in ((0, 5, 200), (5, 0, 200), (5, 5, True), (1.0, 5, 200)):
        with pytest.raises(ValueError): retention.export_step(bad[0], every=bad[1], terminal_step=bad[2])
    tie_rows = [(f's{g}-{m}', f'g{g}', m, float(m % 3)) for g in range(4) for m in range(8)]
    roles, ranks = retention.select_roles(tie_rows)
    # Members 2 and 5 tie at the max (2.0), members 0/3/6 tie at the min (0.0): smallest sample_id wins.
    assert roles['s0-0'] == ('member0', 'worst') and roles['s0-2'] == ('best',)
    assert ranks['s0-2'] == 1 and ranks['s0-5'] == 2 and ranks['s0-6'] == 8
    assert sum(len(v) for v in roles.values()) == 12
    flat_rows = [(f's{g}-{m}', f'g{g}', m, 1.5) for g in range(4) for m in range(8)]
    assert retention.select_roles(flat_rows)[0]['s1-0'] == ('member0', 'best', 'worst')
    for broken in (tie_rows[:-1], tie_rows[:-1] + [tie_rows[0]], tie_rows[:-1] + [('x', 'g3', 7, float('nan'))],
                   tie_rows[:-1] + [('x', 'g3', 7, 1)], tie_rows[:-1] + [('x', 'g3', 0, 1.0)]):
        with pytest.raises(ValueError): retention.select_roles(broken)
    assert retention.sample_dirname('h3-sample:ab') == 'h3-sample:ab'
    assert retention.sample_dirname('../x') == hashlib.sha256(b'../x').hexdigest()
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    assert ffmpeg and ffprobe, 'CPU media gate requires ffmpeg and ffprobe'
    video_root = tmp_path / 'formal-videos'
    reward_of = {f'h3-sample:{g}-{m}': float((m * 7 + g) % 8) / 4 for g in range(4) for m in range(8)}
    def owner(rank, step):
        g, half = rank // 2, rank % 2
        local = tuple((f'h3-sample:{g}-{m}', f'h3-group:{g}', m) for m in range(4 * half, 4 * half + 4))
        request = SimpleNamespace(identity=SimpleNamespace(optimizer_step=step - 1), group_layout='paired_4x8',
            samples=local, prompts={sid: f'prompt {g}' for sid, _, _ in local}, collection_id='c', behavior_id='b')
        return SimpleNamespace(_pending=request)
    media = H3AVMedia(torch.linspace(0, 1, 3 * 3 * 16 * 16).reshape(3, 3, 16, 16), torch.zeros(8000, 2), 12, 32000)
    scored = []
    def original(*, sample_id, prompt, media):
        scored.append(sample_id); return reward_of[sample_id]
    def owners(step, ffmpeg_path=ffmpeg):
        rows = []
        for rank in range(8):
            keep = retention.RolloutRetention(root=video_root, rank=rank, ffmpeg=ffmpeg_path, ffprobe=ffprobe,
                launcher_sha='test', every=5, terminal_step=200, run_seed=42, also=(1,),
                identity=lambda sid: dict(stochastic_index=3, initial_noise_sha256='n' * 64))
            source = owner(rank, step)
            callback = keep.observe(original, source=source)
            for sid, _, _ in source._pending.samples:
                assert callback(sample_id=sid, prompt=source._pending.prompts[sid], media=media) == reward_of[sid]
            rows.append(keep)
        return rows
    before = media.video.clone()
    quiet = owners(4)  # non-export update: scored, never encoded or staged
    assert len(scored) == 32 and not video_root.exists() and all(not k.pending for k in quiet)
    keepers = owners(5)
    assert len(scored) == 64 and torch.equal(media.video, before)
    tables = [k.local_rows(5) for k in keepers]
    published = [path for k in keepers for path in k.publish(5, tables)]
    roles, ranks = retention.select_roles([tuple(r) for t in tables for r in t])
    assert len(published) == sum(len(v) for v in roles.values()) == 12
    step_dir = video_root / 'step-0005'
    assert sorted(p.name for p in step_dir.iterdir()) == ['best', 'member0', 'selection.json', 'worst']
    for role in retention.ROLES:
        assert len(list((step_dir / role).iterdir())) == 4
    best = [sid for sid, value in roles.items() if 'best' in value]
    for sid in best:
        group = sid.split(':')[1].split('-')[0]
        assert reward_of[sid] == max(v for s, v in reward_of.items() if s.split(':')[1].split('-')[0] == group)
        receipt = json.loads((step_dir / 'best' / sid / 'receipt.json').read_text())
        assert (receipt['role'], receipt['reward'], receipt['group_reward_rank'], receipt['run_seed']) == ('best', reward_of[sid], 1, 42)
        assert receipt['roles'] == list(roles[sid])
        assert receipt['prompt'].startswith('prompt ') and receipt['stochastic_index'] == 3 and receipt['frames'] == 3
        assert receipt['video_sha256'] == hashlib.sha256((step_dir / 'best' / sid / 'video.mp4').read_bytes()).hexdigest()
    selection_rows = json.loads((step_dir / 'selection.json').read_text())['samples']
    assert len(selection_rows) == 32 and sum(len(r['roles']) for r in selection_rows) == 12
    assert not (video_root / '.staging').exists() or not any((video_root / '.staging').rglob('*.mp4'))
    with pytest.raises(FileExistsError):  # an export step is never silently overwritten
        again = owners(5)
        for k in again: k.publish(5, [k2.local_rows(5) for k2 in again])
    for k in again: k.discard()
    assert not any((video_root / '.staging').rglob('*.mp4'))
    # Update 1 (untrained base) is exported too; an all-equal group keeps one sample in all three roles.
    distinct = dict(reward_of)
    reward_of.update((sid, 1.5) for sid in reward_of)
    tied = owners(1)
    tied_tables = [k.local_rows(1) for k in tied]
    assert sum(len(k.publish(1, tied_tables)) for k in tied) == 12
    reward_of.update(distinct)
    for g in range(4):
        sid = f'h3-sample:{g}-0'
        for role in retention.ROLES:
            tied_receipt = json.loads((video_root / 'step-0001' / role / sid / 'receipt.json').read_text())
            assert tied_receipt['role'] == role and tied_receipt['roles'] == ['member0', 'best', 'worst']
            assert tied_receipt['step'] == 1 and tied_receipt['reward'] == 1.5
        assert all(len(list((video_root / 'step-0001' / role).iterdir())) == 4 for role in retention.ROLES)
    # Failure: a broken encoder fails the score call AFTER the original reward; no temp survives.
    with pytest.raises(Exception):
        owners(10, ffmpeg_path=str(tmp_path / 'missing-ffmpeg'))
    assert not any((video_root / '.staging').rglob('.encode-*')) and not (video_root / 'step-0010').exists()
    retention.RolloutRetention(root=video_root, rank=0, ffmpeg=ffmpeg, ffprobe=ffprobe, launcher_sha='t',
        every=5, terminal_step=200, run_seed=42).discard()
    assert not (video_root / '.staging' / 'rank-0').exists()
    # Resume: step dirs after the restored boundary move aside (kept), earlier ones stay.
    keeper = retention.RolloutRetention(root=video_root, rank=0, ffmpeg=ffmpeg, ffprobe=ffprobe, launcher_sha='t',
        every=5, terminal_step=200, run_seed=42)
    (video_root / 'step-0010').mkdir()
    assert keeper.supersede(first_step=5, label='attempt-x') == ['step-0010']
    assert (video_root / 'step-0005').is_dir() and (video_root / 'superseded' / 'attempt-x' / 'step-0010').is_dir()

    # Formal 200-update launcher: validation, resume identity, disk preflight,
    # 30 s monitor staleness with degraded non-fatal, and the real backend
    # hooks around tiny CPU iterations (no GPU, no controller.run).
    from scripts import launch_h3_formal as formal
    from scripts import launch_h3_resolution_memory as two_entry
    from scripts.launch_h3_one_update import TRAIN_SHA as FORMAL_TRAIN, TEST_SHA as FORMAL_TEST
    from tests.test_h3_mvp_cpu_profile import _fixture as formal_fixture, _save as formal_save
    from visual_rl.composition.config.h3_profile import compile_h3_profile as formal_compile
    from visual_rl.runtime import h3_lifecycle as formal_lifecycle
    from visual_rl.runtime import h3_memory as formal_memory
    import io, math, threading, time
    from contextlib import contextmanager, redirect_stdout
    profile_path, raw = formal_fixture(tmp_path / 'formal-profile')
    raw['overrides']['training'] = dict(group_size=8, num_steps=10, algorithm='dual_flash', group_layout='paired_4x8')
    raw['overrides']['model'] = dict(params=dict(width=384, height=256, frames=124, fps=24, lora_rank=32, lora_alpha=64))
    raw['launch'].update(max_optimizer_steps=200, checkpoint_every_optimizer_steps=5)
    raw['launch']['expected'].update(train_sha256=FORMAL_TRAIN, eval_sha256=FORMAL_TEST)
    def compiled(**changes):
        data = json.loads(json.dumps(raw))
        for key, value in changes.items():
            if key in ('width', 'height'): data['overrides']['model']['params'][key] = value
            else: data['launch'][key] = value
        return formal_compile(formal_save(profile_path.parent, data))
    accepted = compiled()
    assert formal.validate_formal(accepted) is accepted
    for changes in (dict(width=480, height=320), dict(width=384, height=320), dict(max_optimizer_steps=2),
                    dict(max_optimizer_steps=199), dict(max_optimizer_steps=205),
                    dict(checkpoint_every_optimizer_steps=10), dict(checkpoint_every_optimizer_steps=1)):
        with pytest.raises(ValueError, match='formal run requires'): formal.validate_formal(compiled(**changes))
    for bad in (199, 2, '200', 200.0, True):
        with pytest.raises(ValueError, match='accepts max_optimizer_steps'):
            formal.validate_formal(accepted, max_optimizer_steps=bad)
    with pytest.raises(ValueError, match='max2'): two_entry.validate_fresh(accepted)  # old launcher unchanged
    # Resume target: only this run's checkpoints/step-N with 0 < N <= 200 (200 = finalize-only).
    out = tmp_path / 'formal-out'
    def at(path):
        return SimpleNamespace(launch=SimpleNamespace(resume_from=path, output_dir=out, max_optimizer_steps=200))
    assert formal.resume_step(at(None)) is None and formal.resume_step(at(out / 'checkpoints' / 'step-20')) == 20
    assert formal.resume_step(at(out / 'checkpoints' / 'step-200')) == 200
    for bad in (out / 'checkpoints' / 'step-201', out / 'checkpoints' / 'latest', tmp_path / 'other' / 'checkpoints' / 'step-5',
                out / 'checkpoints' / 'step-0'):
        with pytest.raises(ValueError, match='resume_from'): formal.resume_step(at(bad))
    # Finalize-only: the lifecycle run loop executes no update and returns the restored checkpoint.
    finalize_out = tmp_path / 'formal-finalize'; finalize_out.mkdir()
    (finalize_out / 'h3-last-metrics.json').write_text(json.dumps(dict(step=199, loss=0.5)))
    finalize_bound = SimpleNamespace(graph=SimpleNamespace(profile=SimpleNamespace(launch=SimpleNamespace(
            max_optimizer_steps=200, resume_from=finalize_out / 'checkpoints' / 'step-200', output_dir=finalize_out))),
        transaction=SimpleNamespace(manager=SimpleNamespace(inspect_complete=lambda path: SimpleNamespace(checkpoint=('ck', path.name)))),
        trainer=SimpleNamespace(run_iteration=lambda step: pytest.fail('finalize-only resume must not update')))
    finalizer = formal.FormalBackend.__new__(formal.FormalBackend)
    finalizer.inspection_only, finalizer.start_step = False, 200
    assert formal.FormalBackend.run(finalizer, finalize_bound) == (('ck', 'step-200'), dict(step=199, loss=0.5))
    args = formal.build_parser().parse_args(['cfg.json', '--expected-sha', 'a' * 40, '--ffmpeg', 'f',
                                             '--ffprobe', 'p', '--monitor-dir', 'm'])
    assert (args.max_optimizer_steps, args.audit_weights, args.reward_pipeline, args.reuse_video_vae,
            args.reuse_rollout_anchor, args.profile_rollout, args.checkpoint_threads,
            args.boundary_content_check, args.journal_max_mib) == (200, False, False, False, False, False, 1, 'full', 512)
    identity_profile = SimpleNamespace(launch=SimpleNamespace(output_dir=out, max_optimizer_steps=200))
    with pytest.raises(ValueError, match='formal-run.json'): formal.check_resume_identity(identity_profile, args)
    out.mkdir()
    (out / formal.IDENTITY_FILE).write_text(json.dumps(formal.fresh_record(identity_profile, args)))
    assert formal.check_resume_identity(identity_profile, args) is None
    for name, value in (('reuse_video_vae', True), ('checkpoint_threads', 8)):
        changed = SimpleNamespace(**dict(vars(args), **{name: value}))
        with pytest.raises(ValueError, match='changed formal-run identity'): formal.check_resume_identity(identity_profile, changed)
    observed = SimpleNamespace(**dict(vars(args), audit_weights=True))  # observation-only: not identity
    assert formal.check_resume_identity(identity_profile, observed) is None
    # Launcher upgrade: a new SHA is refused by default; allowed only with flag + reason and
    # only for scripts/ and tests/ changes (visual_rl/ is the lifecycle-hashed training core).
    def with_args(**kw): return SimpleNamespace(**dict(vars(args), **kw))
    def diff(*files): return lambda old, new: (old, new) == ('a' * 40, 'b' * 40) and list(files) or pytest.fail('wrong range')
    scripts_only = diff('scripts/launch_h3_formal.py', 'tests/test_flash_collection_runtime.py')
    with pytest.raises(ValueError, match='different launcher SHA'):
        formal.check_resume_identity(identity_profile, with_args(expected_sha='b' * 40), changed_files=scripts_only)
    with pytest.raises(ValueError, match='upgrade-reason'):
        formal.check_resume_identity(identity_profile, with_args(expected_sha='b' * 40, allow_launcher_upgrade=True),
                                     changed_files=scripts_only)
    for outside in ('visual_rl/runtime/h3_lifecycle.py', 'docs/notes.md', 'experiments/x.yaml'):
        with pytest.raises(ValueError, match='only scripts/ and tests/'):
            formal.check_resume_identity(identity_profile, with_args(expected_sha='b' * 40, allow_launcher_upgrade=True,
                upgrade_reason='fix'), changed_files=diff('scripts/a.py', outside))
    with pytest.raises(ValueError, match='unchanged'):
        formal.check_resume_identity(identity_profile, with_args(allow_launcher_upgrade=True, upgrade_reason='x'))
    with pytest.raises(ValueError, match='unchanged'):
        formal.check_resume_identity(identity_profile, with_args(upgrade_reason='x'))
    upgrade = formal.check_resume_identity(identity_profile, with_args(expected_sha='b' * 40, allow_launcher_upgrade=True,
        upgrade_reason=' export fix '), changed_files=scripts_only, now='2026-11-04T00:00:00+00:00', attempt='A1')
    assert upgrade == dict(old_sha='a' * 40, new_sha='b' * 40, utc='2026-11-04T00:00:00+00:00', reason='export fix',
                           changed_scripts=['scripts/launch_h3_formal.py'], changed_tests=['tests/test_flash_collection_runtime.py'],
                           status='attempt', attempt='A1')
    formal.record_launcher_upgrade(out, upgrade)
    history = json.loads((out / formal.IDENTITY_FILE).read_text())
    assert history['launcher_sha'] == 'a' * 40 and history['launcher_upgrades'] == [upgrade]
    # An unapplied attempt (e.g. restore failed) never changes the current launcher SHA.
    assert formal.check_resume_identity(identity_profile, args) is None
    with pytest.raises(ValueError, match='different launcher SHA'):
        formal.check_resume_identity(identity_profile, with_args(expected_sha='b' * 40))
    formal.mark_launcher_upgrade_applied(out, 'A1', now='2026-11-04T00:05:00+00:00')
    applied = json.loads((out / formal.IDENTITY_FILE).read_text())['launcher_upgrades']
    assert applied == [dict(upgrade, status='applied', applied_utc='2026-11-04T00:05:00+00:00')]
    assert formal.check_resume_identity(identity_profile, with_args(expected_sha='b' * 40)) is None  # now current
    with pytest.raises(ValueError, match='different launcher SHA'): formal.check_resume_identity(identity_profile, args)
    with pytest.raises(ValueError, match='changed since'): formal.record_launcher_upgrade(out, upgrade)
    with pytest.raises(ValueError, match='pending'): formal.mark_launcher_upgrade_applied(out, 'A1')
    # Real Git: --no-renames lists both sides of a rename, so moving a protected file into
    # scripts/ is refused; a scripts-only change passes; an absent old commit refuses cleanly.
    import subprocess
    repo = tmp_path / 'upgrade-repo'; repo.mkdir()
    def run_git(*argv): return subprocess.check_output(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t',
                                                         '-c', 'commit.gpgsign=false', *argv], text=True).strip()
    run_git('init', '-q')
    (repo / 'experiments').mkdir(); (repo / 'scripts').mkdir()
    (repo / 'experiments' / 'x.yaml').write_text('lr: 1e-4\n' * 20); (repo / 'scripts' / 'a.py').write_text('A = 1\n')
    run_git('add', '-A'); run_git('commit', '-q', '-m', 'base'); base_sha = run_git('rev-parse', 'HEAD')
    (repo / 'scripts' / 'a.py').write_text('A = 2\n'); run_git('commit', '-qam', 'scripts only')
    scripts_sha = run_git('rev-parse', 'HEAD')
    run_git('mv', 'experiments/x.yaml', 'scripts/x.yaml'); run_git('commit', '-qm', 'rename'); renamed_sha = run_git('rev-parse', 'HEAD')
    in_repo = lambda old, new: formal.git_changed_files(old, new, root=repo)
    assert in_repo(scripts_sha, renamed_sha) == ['experiments/x.yaml', 'scripts/x.yaml']
    git_out = tmp_path / 'formal-git-out'; git_out.mkdir()
    git_profile = SimpleNamespace(launch=SimpleNamespace(output_dir=git_out, max_optimizer_steps=200))
    (git_out / formal.IDENTITY_FILE).write_text(json.dumps(formal.fresh_record(git_profile, with_args(expected_sha=base_sha))))
    upgrade_args = dict(allow_launcher_upgrade=True, upgrade_reason='review fix')
    with pytest.raises(ValueError, match="experiments/x.yaml"):
        formal.check_resume_identity(git_profile, with_args(expected_sha=renamed_sha, **upgrade_args), changed_files=in_repo)
    allowed = formal.check_resume_identity(git_profile, with_args(expected_sha=scripts_sha, **upgrade_args), changed_files=in_repo)
    assert allowed['changed_scripts'] == ['scripts/a.py'] and allowed['old_sha'] == base_sha
    with pytest.raises(ValueError, match='not present'): in_repo('0' * 40, scripts_sha)
    assert formal.git_changed_files('HEAD', 'HEAD') == []  # this deployed checkout
    # Disk: 41 checkpoints (step1, every 5, terminal), 41 video exports (update 1, every 5), bytes and inodes.
    assert formal.checkpoint_steps(total=200)[:3] == [1, 5, 10] and len(formal.checkpoint_steps(total=200)) == 41
    logs = formal.JOURNAL_PROCESSES * 512 * 1024**2 + 8 * 1024**3 + formal.OTHER_LOG_BYTES
    need = formal.disk_need(total=200, log_bytes=logs)
    assert (need['checkpoints'], need['export_updates']) == (41, 41)
    assert need['need_bytes'] == math.ceil(1.2 * (41 * formal.CHECKPOINT_BYTES + 492 * formal.VIDEO_BYTES + logs))
    assert need['need_inodes'] == math.ceil(1.2 * (41 * 15025 + 41 * formal.FILES_PER_EXPORT))
    assert formal.disk_need(total=200, after=100, log_bytes=0)['checkpoints'] == 20
    finalize_need = formal.disk_need(total=200, after=200, log_bytes=0)
    assert (finalize_need['checkpoints'], finalize_need['export_updates'], finalize_need['need_bytes']) == (0, 0, 0)
    def fs(free, inodes=10**7, total_inodes=10**8):
        return lambda path: SimpleNamespace(f_bavail=free // 4096, f_frsize=4096, f_favail=inodes, f_files=total_inodes)
    deep = SimpleNamespace(launch=SimpleNamespace(output_dir=tmp_path / 'missing' / 'run', max_optimizer_steps=200))
    report = formal.disk_preflight(deep, statvfs=fs(160 * 10**9), log_bytes=logs)
    assert report['filesystem_probe'] == str(tmp_path) and report['free_bytes'] <= 160 * 10**9
    with pytest.raises(RuntimeError, match='insufficient disk'): formal.disk_preflight(deep, statvfs=fs(100 * 10**9), log_bytes=logs)
    with pytest.raises(RuntimeError, match='inodes'): formal.disk_preflight(deep, statvfs=fs(160 * 10**9, inodes=500_000), log_bytes=logs)
    assert formal.disk_preflight(deep, statvfs=fs(160 * 10**9, inodes=0, total_inodes=0), log_bytes=logs)['free_inodes'] is None
    assert formal.disk_preflight(deep, after=150, statvfs=fs(50 * 10**9), log_bytes=logs)['checkpoints'] == 10
    # Card-monitor lifetime/bytes must cover the remaining updates; budgets come from health.json.
    budget_dir = tmp_path / 'formal-monitor-budget'; budget_dir.mkdir()
    def budget_health(**fields):
        (budget_dir / 'health.json').write_text(json.dumps(dict(dict(healthy=True, sample_count=100, bytes=100 * 4096,
            interval_seconds=5., max_seconds=1728000., max_bytes=12 * 1024**3, started_monotonic_ns=0, monotonic_ns=0), **fields)))
    budget_health()
    covered = formal.monitor_budget(budget_dir, remaining_updates=200, now_ns=0)
    assert covered['need_seconds'] == 200 * 7200 and covered['row_bytes'] == formal.MONITOR_ROW_BYTES
    budget_health(max_seconds=86400., max_bytes=512 * 1024**2, interval_seconds=1.)  # old defaults: 24 h
    with pytest.raises(RuntimeError, match='cannot cover 200 updates'): formal.monitor_budget(budget_dir, remaining_updates=200, now_ns=0)
    assert formal.monitor_budget(budget_dir, remaining_updates=10, now_ns=0, row_bytes=1024)['need_seconds'] == 72000
    with pytest.raises(RuntimeError, match='cannot cover'):  # elapsed lifetime counts
        formal.monitor_budget(budget_dir, remaining_updates=10, now_ns=20000 * 10**9, row_bytes=1024)
    assert formal.monitor_budget(budget_dir, remaining_updates=0, now_ns=0)['need_bytes'] == 0  # finalize-only
    for slow in (10.5, 30.):  # a longer interval would let the 30 s per-update health check go stale
        budget_health(interval_seconds=slow)
        for remaining in (200, 0):
            with pytest.raises(RuntimeError, match='exceeds 10.0 s'):
                formal.monitor_budget(budget_dir, remaining_updates=remaining, now_ns=0)
    budget_health(interval_seconds=10., max_bytes=512 * 1024**2)
    with pytest.raises(RuntimeError, match=r'larger --max-seconds/--max-bytes \(keep --interval <= 10 s\)') as hint:
        formal.monitor_budget(budget_dir, remaining_updates=200, now_ns=0)
    assert 'longer --interval' not in str(hint.value)
    budget_health(interval_seconds=1., max_seconds=86400., max_bytes=512 * 1024**2)
    budget_health(bytes=100 * 10**6)  # observed rows (1 MB) larger than the default estimate are used
    with pytest.raises(RuntimeError, match='cannot cover'): formal.monitor_budget(budget_dir, remaining_updates=200, now_ns=0)
    (budget_dir / 'health.json').write_text(json.dumps(dict(healthy=True, sample_count=9, monotonic_ns=0)))
    with pytest.raises(RuntimeError, match='budget fields'): formal.monitor_budget(budget_dir, remaining_updates=1)
    budget_health(healthy=False)
    with pytest.raises(RuntimeError, match='unhealthy'): formal.monitor_budget(budget_dir, remaining_updates=1)
    card_budget_dir = tmp_path / 'formal-card-monitor'
    from scripts import monitor_h3_memory as card_budget_monitor
    with monkeypatch.context() as patch:
        patch.setattr(card_budget_monitor, 'sample', lambda journal: (card_budget_dir / 'STOP').touch() or dict(
            monotonic_ns=time.monotonic_ns(), utc='u'))
        card_budget_monitor.run(card_budget_dir, tmp_path / 'j', interval=0., max_seconds=1728000., max_bytes=8 * 1024**3)
    written_health = json.loads((card_budget_dir / 'health.json').read_text())
    assert (written_health['interval_seconds'], written_health['max_seconds'], written_health['max_bytes']) == (0., 1728000., 8 * 1024**3)
    assert formal.monitor_budget(card_budget_dir, remaining_updates=0)['remaining_bytes'] > 0
    # Monitor: degraded is non-fatal; stale beyond 30 s or healthy=false stops the next update.
    health_dir = tmp_path / 'formal-monitor'; health_dir.mkdir()
    def health(**fields):
        (health_dir / 'health.json').write_text(json.dumps(dict(dict(healthy=True, monotonic_ns=10**12, sample_count=9), **fields)))
    health(degraded=True, consecutive_failures=2)
    assert two_entry.monitor_healthy(health_dir, now_ns=10**12 + 20 * 10**9, max_age=formal.FormalBackend.monitor_max_age)['degraded'] is True
    with pytest.raises(RuntimeError, match='stale'): two_entry.monitor_healthy(health_dir, now_ns=10**12 + 20 * 10**9)
    with pytest.raises(RuntimeError, match='stale'):
        two_entry.monitor_healthy(health_dir, now_ns=10**12 + 31 * 10**9, max_age=formal.FormalBackend.monitor_max_age)
    health(healthy=False)
    with pytest.raises(RuntimeError, match='unhealthy'): two_entry.monitor_healthy(health_dir, now_ns=10**12, max_age=30)
    # Journal budget is explicit and must agree with any pre-set environment.
    with monkeypatch.context() as patch:
        patch.delenv('H3_MEMORY_MAX_BYTES', raising=False)
        assert formal.journal_budget(512) == 512 * 1024**2 == int(os.environ['H3_MEMORY_MAX_BYTES'])
        with pytest.raises(ValueError, match='differs'): formal.journal_budget(256)
        for bad in (63, 16385, 512.0, True):
            with pytest.raises(ValueError, match='journal-max-mib'): formal.journal_budget(bad)
    # Backend hooks over tiny CPU iterations: fresh 0..4 exports at 5, then a resume at 5.
    class FakeJournal:
        lock = threading.RLock(); base = {}
        def assert_healthy(self): pass
        def event(self, *a, **kw): pass
        @contextmanager
        def span(self, *a, **kw): yield self
    calls = []
    class SpyRetention:
        def __init__(self, **kw): self.kw = kw; calls.append(('init', kw['rank'], kw['terminal_step'], kw['every']))
        def observe(self, original, *, source): return original
        def is_export(self, step): return formal.retention.export_step(step, every=5, terminal_step=200, also=self.kw['also'])
        def local_rows(self, step): calls.append(('rows', step)); return [['s', 'g', 0, 1.0]]
        def publish(self, step, tables): calls.append(('publish', step, len(tables)))
        def discard(self): calls.append(('discard',))
        def supersede(self, *, first_step, label): calls.append(('supersede', first_step)); return []
    def tiny_backend(start_step, audit_weights, adam_step=0, rollout_batch=1):
        model = torch.nn.Module()
        model.register_parameter('lora', torch.nn.Parameter(torch.ones(2)))
        model.register_parameter('frozen', torch.nn.Parameter(torch.ones(2), requires_grad=False))
        optimizer = torch.optim.AdamW([model.lora], lr=1e-4)
        if adam_step:
            model.lora.grad = torch.ones(2)
            for _ in range(adam_step): optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        prepared = SimpleNamespace(optimizer=optimizer, prepared_root=model)
        stages = SimpleNamespace(predict=lambda **kw: model.lora * 2, score_one=lambda **kw: 1.)
        if rollout_batch == 4: stages.predict_batch = lambda **kw: (model.lora * 2,) * 4
        selfcheck = [rollout_batch == 4]  # S1 row self-check: first rollout of THIS process only
        def tiny_iteration(step):
            if step == 7: raise RuntimeError('injected update failure')
            optimizer.zero_grad(set_to_none=True)
            if rollout_batch == 4:
                for _ in range(10 + selfcheck[0]):
                    with torch.no_grad(): stages.predict_batch(index=2)
                for _ in range(4 + 4 * selfcheck[0]):
                    with torch.no_grad(): stages.predict(index=0)
                selfcheck[0] = False
            for _ in range(44 if rollout_batch == 1 else 0):
                with torch.no_grad(): stages.predict(index=2)
            for _ in range(4):
                value = stages.predict(index=2); value.sum().backward(); del value
            norm = float(model.lora.grad.norm()); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            return SimpleNamespace(value=SimpleNamespace(payload=SimpleNamespace(grad_norm=norm)))
        bound = SimpleNamespace(stages=stages, trainer=SimpleNamespace(run_iteration=tiny_iteration))
        graph = SimpleNamespace(source=SimpleNamespace(state_dict=lambda: dict(next_optimizer_step=start_step)),
            profile=SimpleNamespace(launch=SimpleNamespace(output_dir=tmp_path / f'formal-wrapper-{start_step}-{rollout_batch}',
                max_optimizer_steps=200, seed=42), expected={'model_sha256': 'm'}))
        graph.profile.launch.output_dir.mkdir(exist_ok=True)
        options = dict(vars(args), monitor_dir=health_dir, audit_weights=audit_weights, rollout_batch=rollout_batch)
        backend = formal.FormalBackend(args=SimpleNamespace(**options), manifest=dict(launch_stamp='t'))
        backend.start_step = start_step
        bound_holder[0] = bound
        backend.bind(None, SimpleNamespace(rank=0, process_group=None), graph, prepared, None)
        return backend, bound, graph
    with monkeypatch.context() as patch:
        patch.setattr(formal_memory, 'journal', lambda: FakeJournal())
        bound_holder = [None]
        patch.setattr(formal_lifecycle.H3LifecycleBackend, 'bind', lambda self, *a: bound_holder[0])
        for module in (formal, two_entry):
            patch.setattr(module, '_phase', lambda session, label, work: work())
            patch.setattr(module, '_exchange', lambda *a: [a[3]] * 8)
        patch.setattr(formal.retention, 'RolloutRetention', SpyRetention)
        health(degraded=True, consecutive_failures=2, monotonic_ns=time.monotonic_ns())  # degraded, fresh (<30 s)
        fresh_backend, fresh_bound, fresh_graph = tiny_backend(0, False)
        assert calls[0] == ('init', 0, 200, 5) and fresh_backend.audit_weights is False
        for step in range(5): fresh_bound.trainer.run_iteration(step)
        # Start: stale staging deleted and later step dirs set aside; export only after update 5.
        assert calls[1:3] == [('discard',), ('supersede', 0)]
        assert [c for c in calls if c[0] in ('rows', 'publish')] == [('rows', 1), ('publish', 1, 8), ('rows', 5), ('publish', 5, 8)]
        assert calls[-1] == ('discard',) and len(fresh_backend.audits) == 5
        assert fresh_backend.audits[-1]['lora_changed'] is None and fresh_backend.audits[-1]['after'] is None
        assert json.loads((fresh_graph.profile.launch.output_dir / 'update-5-audit.json').read_text())['optimizer_updates'] == 5
        # Resume at 5 with weight audit on: counts restart for this process; Adam counter must match.
        calls.clear()
        resumed, resumed_bound, _ = tiny_backend(5, True, adam_step=5)
        resumed_bound.trainer.run_iteration(5); resumed_bound.trainer.run_iteration(6)
        assert resumed.audits[-1]['graph']['nograd_forwards'] == 88 and resumed.audits[-1]['lora_changed'] is True
        assert ('supersede', 5) in calls and not any(c[0] == 'publish' for c in calls)
        calls.clear()
        with pytest.raises(RuntimeError, match='injected'): resumed_bound.trainer.run_iteration(7)
        assert calls == [('discard',)]  # failed update: staging removed, error re-raised
        # S1 x resume: no-grad audit = (10 batched + 4 anchor) x updates of THIS process
        # + 5 once-per-process self-check forwards: 14*(step+1-first_step)+5 -> [19, 33], not 14*(step+1)+5.
        s1_resumed, s1_bound, _ = tiny_backend(5, False, adam_step=5, rollout_batch=4)
        s1_bound.trainer.run_iteration(5); s1_bound.trainer.run_iteration(6)
        assert [row['graph']['nograd_forwards'] for row in s1_resumed.audits] == [19, 33]
        assert (s1_resumed.nograd_forwards_per_update, s1_resumed.nograd_forwards_once) == (14, 5)
        calls.clear()
        mismatch, mismatch_bound, _ = tiny_backend(10, False, adam_step=9)
        with pytest.raises(ValueError, match='resume checkpoint'): mismatch_bound.trainer.run_iteration(10)
        fresh_wrong, fresh_wrong_bound, _ = tiny_backend(0, False, adam_step=1)
        with pytest.raises(ValueError, match='not fresh'): fresh_wrong_bound.trainer.run_iteration(0)
        health(healthy=False)
        stopped, stopped_bound, _ = tiny_backend(0, False)
        with pytest.raises(RuntimeError, match='unhealthy'): stopped_bound.trainer.run_iteration(0)
        # Construct: identity + per-launch manifest after admission; disk logged around each save.
        graph_out = tmp_path / 'formal-graph'; graph_out.mkdir()
        closed = []
        fake_graph = SimpleNamespace(run_id='r', close=lambda: closed.append(1), profile=SimpleNamespace(
            launch=SimpleNamespace(output_dir=graph_out, resume_from=None, max_optimizer_steps=200)))
        patch.setattr(formal_lifecycle.H3LifecycleBackend, 'construct_graph', lambda self, *a: fake_graph)
        maker = formal.FormalBackend(args=SimpleNamespace(**dict(vars(args), monitor_dir=health_dir)),
                                     manifest=dict(launch_stamp='20261103T000000Z', started_utc='x'))
        maker.transaction = SimpleNamespace(checkpoint=lambda *a, **kw: SimpleNamespace(step=5))
        preflight = SimpleNamespace(facts=SimpleNamespace(semantic_id='sem', binding_id='bind'), code_hash='code')
        assert maker.construct_graph(preflight, SimpleNamespace(rank=0)) is fake_graph
        written = json.loads((graph_out / formal.IDENTITY_FILE).read_text())
        assert written['identity']['max_optimizer_steps'] == 200 and written['launcher_upgrades'] == []
        launches = list((graph_out / 'formal-launches').iterdir())
        assert len(launches) == 1 and json.loads(launches[0].read_text())['semantic_id'] == 'sem'
        printed = io.StringIO()
        with redirect_stdout(printed):
            assert maker.transaction.checkpoint(progress=SimpleNamespace(next_optimizer_step=5)).step == 5
        disk_rows = [json.loads(line) for line in printed.getvalue().splitlines()]
        assert [r['when'] for r in disk_rows] == ['before_save', 'after_save'] and disk_rows[1]['free_bytes'] > 0
        assert disk_rows[1]['remaining_need_bytes'] == formal.disk_need(total=200, after=5, log_bytes=0)['need_bytes']
        assert (disk_rows[0]['remaining_checkpoints'], disk_rows[1]['remaining_checkpoints']) == (40, 39)  # in-progress save counted
        # Rank0-only reporting around the collective save never raises (peers would hang).
        patch.setattr(formal, 'disk_free', lambda path: (_ for _ in ()).throw(OSError('statvfs failed')))
        printed = io.StringIO()
        with redirect_stdout(printed):
            assert maker.transaction.checkpoint(progress=SimpleNamespace(next_optimizer_step=10)).step == 5
        assert [json.loads(line)['error'] for line in printed.getvalue().splitlines()] == ["OSError('statvfs failed')"] * 2
        # Launcher upgrade: recorded as an attempt at construct, applied only after restore_bound succeeded.
        upgrade_out = tmp_path / 'formal-upgrade-graph'; upgrade_out.mkdir()
        (upgrade_out / formal.IDENTITY_FILE).write_text(json.dumps(formal.fresh_record(identity_profile, args)))
        pending = dict(upgrade, attempt='U1')
        upgraded_graph = SimpleNamespace(run_id='r', close=lambda: None, profile=SimpleNamespace(launch=SimpleNamespace(
            output_dir=upgrade_out, resume_from=upgrade_out / 'checkpoints' / 'step-5', max_optimizer_steps=200)))
        patch.setattr(formal_lifecycle.H3LifecycleBackend, 'construct_graph', lambda self, *a: upgraded_graph)
        upgrader = formal.FormalBackend(args=SimpleNamespace(**dict(vars(args), monitor_dir=health_dir, expected_sha='b' * 40)),
                                        manifest=dict(launch_stamp='U1', started_utc='z', launcher_upgrade=pending))
        upgrader.transaction = SimpleNamespace(checkpoint=lambda *a, **kw: None)
        upgrader.construct_graph(preflight, SimpleNamespace(rank=0))
        assert [u['status'] for u in json.loads((upgrade_out / formal.IDENTITY_FILE).read_text())['launcher_upgrades']] == ['attempt']
        patch.setattr(formal_lifecycle.H3LifecycleBackend, 'restore_bound', lambda self, *a: (_ for _ in ()).throw(ValueError('restore failed')))
        with pytest.raises(ValueError, match='restore failed'):
            upgrader.restore_bound(None, SimpleNamespace(rank=0), upgraded_graph, None, None, None)
        assert json.loads((upgrade_out / formal.IDENTITY_FILE).read_text())['launcher_upgrades'][0]['status'] == 'attempt'
        patch.setattr(formal_lifecycle.H3LifecycleBackend, 'restore_bound', lambda self, *a: None)
        upgrader.restore_bound(None, SimpleNamespace(rank=0), upgraded_graph, None, None, None)
        assert json.loads((upgrade_out / formal.IDENTITY_FILE).read_text())['launcher_upgrades'][0]['status'] == 'applied'
        patch.setattr(formal_lifecycle.H3LifecycleBackend, 'construct_graph', lambda self, *a: fake_graph)
        with pytest.raises(FileExistsError, match='identity'):
            formal.FormalBackend(args=SimpleNamespace(**dict(vars(args), monitor_dir=health_dir)),
                                 manifest=dict(launch_stamp='later', started_utc='y')).construct_graph(preflight, SimpleNamespace(rank=0))
        assert closed == [1]

    # ---- S1: opt-in batched rollout of the four paired_4x8 local samples ----
    # Default B=1 untouched; B=4 batches only the no-grad rollout transformer
    # forward (10 instead of 40 calls/round), keeps per-sample RNG/records,
    # is a different run identity, and is rejected with rollout-anchor reuse.
    import importlib.util, io, contextlib, math
    from dataclasses import replace as dc_replace
    from types import SimpleNamespace
    from visual_rl.algorithms.dynamics.h3_dual_flow_sde import H3DualFlowSDE
    from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
    from visual_rl.algorithms.optimization.h3_dual_anchor import H3DualReplayAnchor
    from visual_rl.algorithms.trainer import h3_dual_rollout as s1_rollout
    from visual_rl.algorithms.trainer.h3_dual_stages import H3DualIterationStages
    from visual_rl.algorithms.trainer.interface import StageValue, IterationIdentity
    from visual_rl.models.h3_joint import H3JointLayout
    from visual_rl.models.implementations.h3_forward import h3_t2va_forward
    from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry
    from visual_rl.runtime.h3_lifecycle import H3LifecycleBackend
    from scripts import launch_h3_resolution_memory as two_update
    # (1) Training contract/identity: B=4 adds payload keys only when enabled.
    paired = H3Training(group_size=8, algorithm='dual_flash', group_layout='paired_4x8')
    batched_training = dc_replace(paired, rollout_batch=4)
    assert paired.rollout_batch == 1 and 'rollout_microbatch' not in paired.payload()
    s1_payload = batched_training.payload()
    assert {k: v for k, v in s1_payload.items() if k not in ('rollout_microbatch', 'rollout_batch',
            'rollout_batch_replay_tolerance')} == paired.payload()
    assert (s1_payload['rollout_microbatch'], s1_payload['rollout_batch'], s1_payload['rollout_batch_replay_tolerance'],
            s1_payload['microbatch']) == (4, s1_rollout.ROLLOUT_BATCH_VERSION, 1e-4, 1)
    assert dc_replace(batched_training, rollout_batch_replay_tolerance=1e-3).payload()['rollout_batch_replay_tolerance'] == 1e-3
    for bad in (dict(rollout_batch=True), dict(rollout_batch=4.0), dict(rollout_batch='4'), dict(rollout_batch=2),
                dict(rollout_batch_replay_tolerance=1), dict(rollout_batch_replay_tolerance=float('nan')),
                dict(rollout_batch_replay_tolerance=.1), dict(rollout_batch_replay_tolerance=0.)):
        with pytest.raises(ValueError): dc_replace(batched_training, **bad)
    with pytest.raises(ValueError, match='only declared for batched'):
        dc_replace(paired, rollout_batch_replay_tolerance=1e-3)
    for algorithm, layout in (('dual_flash', 'rank_local'), ('dual_flow', 'rank_local'), ('cps', 'rank_local')):
        with pytest.raises(ValueError, match='paired_4x8 dual_flash'):
            H3Training(algorithm=algorithm, group_layout=layout, rollout_batch=4)
    # Real compiler + preflight: the backend records B=4 in the semantic ID
    # (owner/resume admission and checkpoint contract math), default does not.
    from tests.test_h3_mvp_cpu_profile import _fixture as s1_profile_fixture, _save as s1_save_profile
    from visual_rl.composition.config.h3_profile import compile_h3_profile, preflight_h3_profile
    s1_profile_path, s1_raw = s1_profile_fixture(tmp_path / 's1-profile')
    s1_raw['overrides']['training'] = dict(group_size=8, num_steps=10, algorithm='dual_flash', group_layout='paired_4x8')
    s1_profile_path = s1_save_profile(s1_profile_path.parent, s1_raw)
    plain = compile_h3_profile(s1_profile_path)
    assert H3LifecycleBackend().compile(s1_profile_path) == plain
    batched_profile = H3LifecycleBackend(rollout_batch=4).compile(s1_profile_path)
    assert batched_profile.training == batched_training and batched_profile.launch == plain.launch
    plain_facts, batched_facts = preflight_h3_profile(plain), preflight_h3_profile(batched_profile)
    assert plain_facts.semantic_id != batched_facts.semantic_id
    assert batched_facts.semantic_payload['math']['rollout_microbatch'] == 4
    assert 'rollout_microbatch' not in plain_facts.semantic_payload['math']
    tolerant = preflight_h3_profile(H3LifecycleBackend(rollout_batch=4, rollout_batch_replay_tolerance=1e-3).compile(s1_profile_path))
    assert tolerant.semantic_id not in (plain_facts.semantic_id, batched_facts.semantic_id)
    # Backend/stage/CLI validation; B=4 + rollout-anchor reuse is rejected everywhere.
    assert (H3LifecycleBackend().rollout_batch, H3LifecycleBackend().rollout_batch_replay_tolerance) == (1, 1e-4)
    for bad in (True, 4.0, '4', 2, 0):
        with pytest.raises(ValueError, match='1 or 4'): H3LifecycleBackend(rollout_batch=bad)
        with pytest.raises(ValueError, match='1 or 4'): H3DualIterationStages(mode='dual_flash', seed=42, rollout_batch=bad)
    for bad in (1, float('inf'), .5):
        with pytest.raises(ValueError): H3LifecycleBackend(rollout_batch=4, rollout_batch_replay_tolerance=bad)
    with pytest.raises(ValueError, match='only declared for batched'):
        H3LifecycleBackend(rollout_batch_replay_tolerance=1e-3)
    with pytest.raises(ValueError, match='anchor reuse is rejected'):
        H3LifecycleBackend(rollout_batch=4, reuse_rollout_anchor=True)
    with pytest.raises(ValueError, match='anchor reuse is rejected'):
        H3DualIterationStages(mode='dual_flash', seed=42, rollout_batch=4, reuse_rollout_anchor=True, predict_batch=lambda **kw: None)
    with pytest.raises(ValueError, match='batch predict'):
        H3DualIterationStages(mode='dual_flash', seed=42, rollout_batch=4)
    with pytest.raises(ValueError, match='dual_flash'):
        H3DualIterationStages(mode='dual_flow', seed=42, rollout_batch=4, predict_batch=lambda **kw: None)
    with pytest.raises(ValueError, match='only declared for batched'):
        H3DualIterationStages(mode='dual_flash', seed=42, predict_batch=lambda **kw: None)
    with pytest.raises(ValueError, match='compiled rollout batch'):
        H3LifecycleBackend(rollout_batch=4)._check_reward_pipeline(SimpleNamespace(training=paired))
    with pytest.raises(ValueError, match='compiled rollout batch'):
        H3LifecycleBackend()._check_reward_pipeline(SimpleNamespace(training=batched_training))
    s1_args = dict(reward_pipeline=False, profile_rollout=False, reuse_video_vae=False, reuse_rollout_anchor=False, checkpoint_threads=1)
    assert two_update.TwoUpdateBackend(args=SimpleNamespace(**s1_args)).nograd_forwards_per_update == 44
    s1_backend = two_update.TwoUpdateBackend(args=SimpleNamespace(**s1_args, rollout_batch=4, rollout_batch_replay_tolerance=1e-3))
    assert (s1_backend.rollout_batch, s1_backend.rollout_batch_replay_tolerance, s1_backend.nograd_forwards_per_update) == (4, 1e-3, 14)
    assert (s1_backend.nograd_forwards_once, two_update.TwoUpdateBackend(args=SimpleNamespace(**s1_args)).nograd_forwards_once) == (5, 0)
    with pytest.raises(ValueError, match='anchor reuse is rejected'):
        two_update.TwoUpdateBackend(args=SimpleNamespace(**dict(s1_args, reuse_rollout_anchor=True), rollout_batch=4))
    # (2) World admission: B4 only for paired_4x8 and only if every rank agrees.
    rank_rows = lambda r: [[f's{r}-{m}', f'g{r//2}', (r % 2)*4 + m] for m in range(4)]
    s1_admission = dict(collection_id='c', behavior_id='b', samples=tuple(map(tuple, rank_rows(0))), group_size=8,
                        num_steps=10, geometry_id='geo', schedule_id='sched', score_profile='score', group_layout='paired_4x8')
    with monkeypatch.context() as patch:
        patch.setattr(h3_world, '_world_rank', lambda group: 0)
        tables = []
        def s1_exchange(kind, rank, group, payload, valid):
            if not valid: raise ValueError('H3 admission rejected local input')
            tables.append(json.loads(json.dumps(payload)))  # transport round trip, as in _exchange
            profiles = [tables[-1][0]]*8 if len(tables) % 2 else [tables[-1][0]]*7 + [tables[-2][0]]
            return [[profile, rank_rows(r)] for r, profile in enumerate(profiles)]
        patch.setattr(h3_world, '_exchange', s1_exchange)
        b1 = h3_world.admit_h3_world(process_group=None, **s1_admission)  # odd call: all agree
        with pytest.raises(ValueError, match='profile mismatch'):  # even call: rank7 still B1
            h3_world.admit_h3_world(process_group=None, **s1_admission, rollout_microbatch=4)
        b4 = h3_world.admit_h3_world(process_group=None, **s1_admission, rollout_microbatch=4)
        assert b1 == b4 and tables[0][0][-1] == 'paired_4x8' and tables[2][0][-1] == 'rollout-microbatch-4'
        for bad in (dict(rollout_microbatch=2), dict(rollout_microbatch=4, group_layout='rank_local',
                    samples=tuple((f's{m}', 'g', m) for m in range(8)))):
            tables.append(None)  # keep the odd/even agreement schedule aligned
            with pytest.raises(ValueError, match='rejected local input'):
                h3_world.admit_h3_world(process_group=None, **(s1_admission | bad))
    # (3) Model port: one batched forward over one shared condition/sigma pair.
    # Batch-invariant tiny forward -> bitwise; the real H3 transformer
    # architecture (tiny fp32) -> per-item equal only to float tolerance.
    geometry = H3T2VAGeometry(height=32, width=32, num_frames=124)
    layout = geometry.joint_layout
    states = [torch.randn(1, layout.packed_width, generator=torch.Generator().manual_seed(70+i)).half() for i in range(4)]
    text = torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(9))
    sigmas = dict(video_sigma=torch.tensor(.7), audio_sigma=torch.tensor(.6))
    seen = []
    def invariant_forward(*, hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices, **attention):
        seen.append((hidden_states.shape[0], encoder_hidden_states.shape, encoder_hidden_states.stride(0)))
        assert timestep_indices.shape == attention['token_tags'].shape and attention['position_ids'].ndim == 2
        context = encoder_hidden_states[:, :1, :1]  # per-item, no cross-item reduction
        return (torch.sin(hidden_states.float()*1.3 + context) + timestep[0], audio_hidden_states.float().roll(1, dims=1) * context)
    with torch.no_grad():
        single = [h3_t2va_forward(invariant_forward, x=x, geometry=geometry, text_embeddings=text, **sigmas) for x in states]
        stacked = h3_t2va_forward(invariant_forward, x=torch.cat(states), geometry=geometry, text_embeddings=text, batch_size=4, **sigmas)
    assert all(torch.equal(stacked[i:i+1], single[i]) for i in range(4))
    assert seen[-1][0] == 4 and seen[-1][1] == (4, 3, 16) and seen[-1][2] != 0  # owned replicated condition
    with pytest.raises(ValueError, match='no-grad'):
        h3_t2va_forward(invariant_forward, x=torch.cat(states), geometry=geometry, text_embeddings=text, batch_size=4, **sigmas)
    with torch.no_grad(), pytest.raises(ValueError, match='declared batch'):
        h3_t2va_forward(invariant_forward, x=torch.cat(states[:3]), geometry=geometry, text_embeddings=text, batch_size=4, **sigmas)
    reference_path = Path(__file__).resolve().parents[1] / ('workflow-workspace/2026-10-27-h3-8gpu-mvp-dev-001/reference/'
                                                           'UniRL/unirl/models/minimax_h3/vendor/transformer_minimax_h3.py')
    try:  # the deployed diffusers module when present, else the tracked vendored reference
        from diffusers.models.transformers import transformer_minimax_h3 as h3_reference
    except ImportError:
        spec = importlib.util.spec_from_file_location('s1_h3_reference', reference_path)
        h3_reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(h3_reference)
    torch.manual_seed(0)
    tiny_h3 = h3_reference.MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=96, hidden_size=32,
        num_layers=2, num_refiner_layers=1, ffn_dim=48, text_dim=16, freq_dim=16, time_embed_hidden_dim=32,
        time_embed_dim=16).float().eval()
    with torch.no_grad():
        real_single = [h3_t2va_forward(tiny_h3, x=x, geometry=geometry, text_embeddings=text, **sigmas) for x in states]
        real_stacked = h3_t2va_forward(tiny_h3, x=torch.cat(states), geometry=geometry, text_embeddings=text, batch_size=4, **sigmas)
    real_delta = max(float((real_stacked[i:i+1] - real_single[i]).abs().max()) for i in range(4))
    assert real_delta <= 1e-5 and not torch.equal(real_single[0], real_single[1])
    print(json.dumps(dict(event='s1_reference_cpu_fp32_batch_delta', max_abs=real_delta)))
    # Real H3Adapter.predict_batch and H3RuntimeBinding.predict_batch on the tiny loaded adapter.
    from tests.test_h3_mvp_cpu_adapter import _adapter as s1_adapter, _cpu_materialize_and_bind, _input as s1_input
    from visual_rl.models.lifecycle.components import ComponentManager
    from visual_rl.models.h3_interface import H3PromptInput, H3Generators
    from visual_rl.models.interface import ModelPortError
    from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
    from visual_rl.runtime import h3_binding as s1_binding_module
    with monkeypatch.context() as patch:
        (tmp_path / 's1-adapter').mkdir()
        from visual_rl.models.implementations import h3_loading as s1_loading
        patch.setattr(s1_loading, '_admit_host_memory', lambda budget: None)  # 24 GiB host reserve is a GPU-host leaf
        adapter, _, _ = s1_adapter(tmp_path / 's1-adapter', patch)
        manager = ComponentManager(adapter, execution_device='cpu'); components = manager.load()
        _cpu_materialize_and_bind(adapter, components)
        condition = adapter.encode(H3PromptInput('cat'))
        spec = adapter.latent_spec_for_batch(H3PromptInput('cat'), device='cpu', dtype=torch.float16)
        latents = [adapter.prepare_latents(spec, generator=H3Generators(torch.Generator().manual_seed(2*i+1),
                   torch.Generator().manual_seed(2*i+2))) for i in range(4)]
        inputs = tuple(s1_input(adapter, condition, x=x) for x in latents)
        with torch.no_grad():
            port_single = [adapter.predict(item).value for item in inputs]
            port_batched = adapter.predict_batch(inputs)
        assert len(port_batched) == 4 and all(p.shape == item.latents.shape for p, item in zip(port_batched, inputs))
        assert all(torch.allclose(a, b, rtol=0, atol=1e-5) for a, b in zip(port_batched, port_single))
        with pytest.raises(ModelPortError, match='no-grad'): adapter.predict_batch(inputs)
        with torch.no_grad():
            with pytest.raises(TypeError, match='at least two'): adapter.predict_batch(inputs[:1])
            with pytest.raises(ModelPortError, match='shared condition'):
                adapter.predict_batch((inputs[0], dc_replace(inputs[1], timestep=torch.tensor(.7))))
            other = adapter.encode(H3PromptInput(' cat'))
            with pytest.raises(ModelPortError, match='shared condition'):
                adapter.predict_batch((inputs[0], s1_input(adapter, other, x=latents[1])))
        bind = object.__new__(s1_binding_module.H3RuntimeBinding)
        bind.prepared = object(); bind.dynamics = H3CPSDynamics(layout, H3ScheduleSnapshot.build(10))
        bind.adapter = adapter; bind.facts = SimpleNamespace(cache_identity=condition.cache_identity)
        bind.source = SimpleNamespace(conditioning_for=lambda sid: condition)
        bind.scope = lambda mode: contextlib.nullcontext()
        ids = tuple(f'p{i}' for i in range(4)); conds = tuple(condition.embeddings for _ in ids)
        with torch.no_grad():
            via_binding = bind.predict_batch(bind.prepared, sample_ids=ids, x=torch.cat(latents), index=2,
                                             conditions=conds, dynamics=bind.dynamics)
            via_single = [bind.predict(bind.prepared, sample_id=sid, x=x, index=2, condition=c, dynamics=bind.dynamics)
                          for sid, x, c in zip(ids, latents, conds)]
            assert all(torch.allclose(a, b, rtol=0, atol=1e-5) for a, b in zip(via_binding, via_single))
            for bad in (dict(sample_ids=ids[:3]), dict(sample_ids=('p0',)*4), dict(conditions=conds[:3]),
                        dict(conditions=(torch.zeros_like(conds[0]),) + conds[1:])):
                with pytest.raises(ValueError):
                    bind.predict_batch(bind.prepared, **(dict(sample_ids=ids, x=torch.cat(latents), index=2,
                                       conditions=conds, dynamics=bind.dynamics) | bad))
        with pytest.raises(ValueError, match='no-grad'):
            bind.predict_batch(bind.prepared, sample_ids=ids, x=torch.cat(latents), index=2, conditions=conds, dynamics=bind.dynamics)
        adapter.close(); manager.close()
    # (4) collect: identical records/terminals/choices with 10 instead of 40
    # model calls; per-sample generators, selection and row order unchanged.
    dual = H3DualFlowSDE(H3JointLayout(1, 2, 1, 3), H3ScheduleSnapshot.build(10))
    ids = tuple(f'sample-{i}' for i in range(4))
    initial = {sid: torch.randn(1, 5, generator=torch.Generator().manual_seed(40+i)).half() for i, sid in enumerate(ids)}
    def generators():
        return {sid: (torch.Generator().manual_seed(2*i), torch.Generator().manual_seed(2*i+1)) for i, sid in enumerate(ids)}
    def invariant(x, index):  # elementwise/roll only: exactly batch-invariant on CPU fp32
        return torch.sin(x.float()*1.7 + .1*index) * .5 + x.float().roll(1, dims=1) * .25
    def run(batch, *, admit=None, groups=None):
        calls, gates = [], []
        def predict(sid, x, index):
            assert not torch.is_grad_enabled() and x.shape[0] == 1; calls.append(('b1', sid, index)); return invariant(x, index)
        def predict_batch(sids, x, index):
            assert not torch.is_grad_enabled() and x.shape == (4, 5) and sids == ids
            calls.append(('b4', sids, index)); out = invariant(x, index); return tuple(out[i:i+1] for i in range(4))
        options = {} if batch == 1 else dict(rollout_batch=4, predict_batch=predict_batch,
            admit=admit or (lambda index: s1_rollout.rollout_batch_admission('cpu', batch=4, layout=dual.layout)))
        result = s1_rollout.collect(dynamics=dual, samples=tuple(zip(ids, groups or ('g',)*4)), initial_states=initial,
            generators=generators(), mode='flash', seed=42, optimizer_step=3, device='cpu', predict=predict,
            read_behavior_id=lambda: 'behavior', behavior_id='behavior',
            gate=lambda event, **kw: gates.append((event, kw.get('local_error'))), **options)
        return result, calls, gates
    (r1, t1, c1), calls1, gates1 = run(1)
    (r4, t4, c4), calls4, gates4 = run(4)
    assert [c[0] for c in calls1] == ['b1']*40 and [c[0] for c in calls4] == ['b4']*10
    assert [c[2] for c in calls4] == list(range(10))
    assert c1 == c4 and len(set(c4.values())) == 1 and list(t1) == list(t4) == list(ids)
    assert all(torch.equal(t1[s], t4[s]) for s in ids)
    assert [(r.sample_id, r.index) for r in r1] == [(r.sample_id, r.index) for r in r4] and len(r4) == 4
    assert all(torch.equal(a.x, b.x) and torch.equal(a.action, b.action) and torch.equal(a.old_joint, b.old_joint)
               for a, b in zip(r1, r4))
    assert [g for g in gates1 if 'prepare' in g[0]] == [g for g in gates4 if g[0].endswith('/prepare') and 'batch' not in g[0]]
    assert sorted(g for g in gates1 if g[0].endswith('/result')) == sorted(g for g in gates4 if g[0].endswith('/result'))
    assert [g for g in gates4 if g[0].startswith('rollout/batch/0/')] == [
        ('rollout/batch/0/admission', False), ('rollout/batch/0/forward', None), ('rollout/batch/0/split', False)]
    # Admission fails closed BEFORE the first batched forward (no B=1 fallback).
    def refuse(index):
        return s1_rollout.rollout_batch_admission('cpu', batch=4, layout=H3JointLayout(5550, 96, 414, 32), free_bytes=15*1024**3)
    with pytest.raises(RuntimeError, match='memory admission failed'):
        run(4, admit=refuse)
    receipt = s1_rollout.rollout_batch_admission('cpu', batch=4, layout=H3JointLayout(5550, 96, 414, 32), free_bytes=17*1024**3)
    assert receipt['checked'] and receipt['required_bytes'] == 16*1024**3 and 'warning' not in receipt
    assert (receipt['driver_free_bytes'], receipt['reserved_unallocated_bytes']) == (17*1024**3, 0)
    cached = s1_rollout.rollout_batch_admission('cpu', batch=4, layout=H3JointLayout(5550, 96, 414, 32),
                                                free_bytes=40*1024**3, driver_free_bytes=10*1024**3)
    assert cached['reserved_unallocated_bytes'] == 30*1024**3 and 'driver-free below' in cached['warning']
    with pytest.raises(ValueError, match='driver-free <= allocatable'):
        s1_rollout.rollout_batch_admission('cpu', batch=4, layout=H3JointLayout(5550, 96, 414, 32),
                                           free_bytes=1, driver_free_bytes=2)
    with pytest.raises(ValueError, match='no declared'):
        s1_rollout.rollout_batch_admission('cpu', batch=4, layout=dual.layout, free_bytes=10**12)
    assert s1_rollout.rollout_batch_admission('cpu', batch=4, layout=dual.layout)['checked'] is False
    refused_calls = []
    with pytest.raises(RuntimeError, match='memory admission failed'):
        s1_rollout.collect(dynamics=dual, samples=tuple(zip(ids, ('g',)*4)), initial_states=initial, generators=generators(),
            mode='flash', seed=42, optimizer_step=3, device='cpu', predict=lambda *a: pytest.fail('B=1 fallback'),
            read_behavior_id=lambda: 'behavior', behavior_id='behavior', gate=lambda event, **kw: refused_calls.append((event, kw.get('local_error'))),
            rollout_batch=4, predict_batch=lambda *a: pytest.fail('batched forward after refused admission'), admit=refuse)
    assert refused_calls[-1] == ('rollout/batch/0/admission', True)
    with pytest.raises(ValueError, match='one prompt group'):
        run(4, groups=('g', 'g', 'g', 'h'))
    # Mandatory startup row self-check: rows must map to their own B=1 sample;
    # a swapped batch output fails closed at a rank-consistent result gate.
    def selfcheck(order, *, admit=lambda index: {'checked': False}):
        events, forwards = [], []
        def batch(sids, x, index):
            forwards.append('b4'); out = invariant(x, index); return tuple(out[i:i+1] for i in order)
        def single(sid, x, index):
            assert not torch.is_grad_enabled(); forwards.append('b1'); return invariant(x, index)
        result = s1_rollout.rollout_batch_selfcheck(samples=tuple(zip(ids, ('g',)*4)), initial_states=initial,
            device='cpu', predict=single, predict_batch=batch, admit=admit,
            gate=lambda event, **kw: events.append((event, kw.get('local_error'))))
        return result, events, forwards
    good, good_events, good_forwards = selfcheck((0, 1, 2, 3))
    assert good['max_abs_delta'] == [0.]*4 and min(good['nearest_other_sample_delta']) > 0
    assert good['extra_nograd_forwards'] == 5 and good_forwards == ['b4'] + ['b1']*4
    assert good_events == [('rollout/batch/selfcheck/admission', False), ('rollout/batch/selfcheck/forward', None),
                           ('rollout/batch/selfcheck/result', False)]
    swapped_events = []
    with pytest.raises(ValueError, match='permuted'):
        s1_rollout.rollout_batch_selfcheck(samples=tuple(zip(ids, ('g',)*4)), initial_states=initial, device='cpu',
            predict=lambda sid, x, index: invariant(x, index), admit=lambda index: {},
            predict_batch=lambda sids, x, index: tuple(invariant(x, index)[i:i+1] for i in (1, 0, 2, 3)),
            gate=lambda event, **kw: swapped_events.append((event, kw.get('local_error'))))
    assert swapped_events[-1] == ('rollout/batch/selfcheck/result', True)
    with pytest.raises(RuntimeError, match='memory admission failed'):
        selfcheck((0, 1, 2, 3), admit=refuse)
    for bad in (dict(rollout_batch=2), dict(rollout_batch=4, mode='flow'), dict(rollout_batch=4, predict_batch=None)):
        with pytest.raises(ValueError):
            s1_rollout.collect(**(dict(dynamics=dual, samples=tuple(zip(ids, ('g',)*4)), initial_states=initial,
                generators=generators(), mode='flash', seed=42, optimizer_step=3, device='cpu', predict=lambda *a: None,
                read_behavior_id=lambda: 'behavior', behavior_id='behavior', gate=lambda *a, **kw: None,
                predict_batch=lambda *a: None, admit=lambda i: {}) | bad))
    # (5) Real dual stages: rollout + optimize. B=4 keeps the B=1 anchor
    # forward (old score with current-forward numerics) and checks the
    # rollout-vs-anchor delta against the DECLARED tolerance.
    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, 'all_reduce', lambda value, **kwargs: value)
        identity = IterationIdentity(1, 'source', 'train', ids, ('g',)*4, (0, 1, 2, 3))
        def stage_run(batch, tolerance=1e-4, perturb=0.):
            lora = torch.nn.Parameter(torch.zeros(5)); optimizer = torch.optim.AdamW([lora], lr=1e-2)
            calls, gates, out = [], [], io.StringIO()
            def predict(prepared, *, sample_id, x, index, **kwargs):
                calls.append(('b1', torch.is_grad_enabled())); return invariant(x, index) + lora
            def predict_batch(prepared, *, sample_ids, x, index, conditions, dynamics):
                assert len(conditions) == 4 and dynamics is dual
                calls.append(('b4', torch.is_grad_enabled())); value = invariant(x, index) + lora
                return tuple(value[i:i+1] for i in range(4))
            extra = {} if batch == 1 else dict(rollout_batch=4, predict_batch=predict_batch, rollout_batch_replay_tolerance=tolerance)
            stage = H3DualIterationStages(mode='dual_flash', seed=42, predict=predict,
                data=SimpleNamespace(reserve=lambda s: None, commit=lambda r: None, abort=lambda r: None),
                prepared=SimpleNamespace(optimizer=optimizer, prepared_root=torch.nn.Identity(), reducer_divisor=8, scheduler=None),
                decode=lambda t: None, score_one=lambda **kw: 1., coordination_factory=lambda plan: None,
                process_group=None, rank=0, device='cpu', read_behavior_id=lambda: 'behavior', **extra)
            request = SimpleNamespace(identity=identity, samples=tuple((s, 'g', m) for m, s in enumerate(ids)),
                initial_states=initial, generators=generators(), behavior_id='behavior',
                conditions={s: torch.zeros(1, 1, 1) for s in ids}, dynamics=dual, max_cells=40,
                collection_id='c', next_behavior_id='n')
            ctx = SimpleNamespace(request=request, dual=dual, plan=SimpleNamespace(samples=request.samples),
                advantages={s: torch.tensor(v) for s, v in zip(ids, (.5, -.5, .25, -.25))},
                coordinator=SimpleNamespace(gate=lambda event, **kw: gates.append((event, kw.get('local_error', False))),
                                            clip_grad_norm=lambda prepared, max_norm: float(lora.grad.norm())))
            stage._active = ctx
            with contextlib.redirect_stdout(out):
                stage.rollout(StageValue(identity, ctx))
                if perturb:
                    ctx.dual_records = (dc_replace(ctx.dual_records[0], old_joint=ctx.dual_records[0].old_joint + perturb),
                                        *ctx.dual_records[1:])
                result = stage.optimize(StageValue(identity, ctx))
            return calls, gates, out.getvalue(), result.payload, lora.detach().clone(), ctx.dual_records
        off, on = stage_run(1), stage_run(4)
        nograd = lambda calls: [kind for kind, grad in calls if not grad]
        # 14 per update, plus the once-per-run row self-check (1 batched + 4 B=1) before the first rollout.
        assert nograd(off[0]) == ['b1']*44 and nograd(on[0]) == ['b4'] + ['b1']*4 + ['b4']*10 + ['b1']*4
        assert [c for c in off[0] if c[1]] == [c for c in on[0] if c[1]] == [('b1', True)]*4
        assert ('dual/rollout-batch/4/0.0001', False) in on[1] and not any('rollout-batch' in g for g, _ in off[1])
        assert ('dual/anchor/reuse/0', False) in on[1]
        assert off[2].count('h3_initial_policy_ratio') == on[2].count('h3_initial_policy_ratio') == 4
        events = [json.loads(line) for line in on[2].splitlines() if line.startswith('{')]
        replay = [e for e in events if e.get('event') == 'h3_rollout_replay_score_check']
        assert len(replay) == 4 and all(e['rollout_batch'] == 4 and e['declared_tolerance'] == 1e-4
                                        and e['old_score_source'] == 'b1_anchor_forward' and e['within_tolerance'] for e in replay)
        assert [e['transition_index'] for e in events if e.get('event') == 'h3_rollout_batch_admission'] == [0] + list(range(10))
        checks = [e for e in events if e.get('event') == 'h3_rollout_batch_selfcheck']
        assert len(checks) == 1 and checks[0]['max_abs_delta'] == [0.]*4 and checks[0]['sample_ids'] == list(ids)
        assert 'h3_rollout_batch_selfcheck' not in off[2]
        assert all('declared_tolerance' not in json.loads(line) for line in off[2].splitlines()
                   if 'h3_rollout_replay_score_check' in line)
        assert all(torch.equal(a.old_joint, b.old_joint) for a, b in zip(off[5], on[5]))
        assert off[3].scaled_loss == on[3].scaled_loss and off[3].grad_norm == on[3].grad_norm and torch.equal(off[4], on[4])
        with pytest.raises(ValueError, match='beyond the declared batched-rollout tolerance'):
            stage_run(4, perturb=5e-4)
        loosened = stage_run(4, tolerance=1e-3, perturb=5e-4)
        assert ('dual/rollout-batch/4/0.001', False) in loosened[1]
        assert torch.equal(loosened[4], on[4])  # ratio uses the B=1 anchor score, not the perturbed rollout score
        with pytest.raises(ValueError, match='sampled/replayed old score mismatch'):
            stage_run(1, perturb=5e-4)
    with pytest.raises(ValueError, match='anchor reuse is rejected'):
        H3DualReplayAnchor(dynamics=dual, behavior_id='b', expected_keys=(('s', 0),), max_records=1,
                           reuse_rollout=True, batched_rollout_tolerance=1e-4)
    with pytest.raises(ValueError, match='tolerance'):
        H3DualReplayAnchor(dynamics=dual, behavior_id='b', expected_keys=(('s', 0),), max_records=1, batched_rollout_tolerance=1)
