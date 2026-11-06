"""独立解释器 CPU Flash 探针；只共享测试叶，不加载另一版本的生产模块。"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import struct
import sys


def run(source, fixtures, work, output):
    # -I 禁止环境 PYTHONPATH；测试辅助模块与被测生产源码分开搜索。
    sys.path[:0] = [str(fixtures), str(source)]
    import numpy as np
    import pytest
    import torch
    import safetensors.torch as st
    import yaml
    torch.set_num_threads(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    with pytest.MonkeyPatch.context() as patch:
        save_file = st.save_file
        def stable_input(tensors, filename, metadata=None):
            # 只规范化测试输入构造：safetensors metadata 键序不应改变两侧输入身份。
            save_file(tensors, filename, metadata=metadata)
            path = Path(filename)
            raw = path.read_bytes()
            size = struct.unpack('<Q', raw[:8])[0]
            header = json.dumps(json.loads(raw[8:8+size]), sort_keys=True, separators=(',', ':')).encode()
            header += b' ' * (-len(header) % 8)
            path.write_bytes(struct.pack('<Q', len(header)) + header + raw[8+size:])
        patch.setattr(st, 'save_file', stable_input)
        from tests.test_h3_mvp_cpu_entry import _controller_case, _config, _run
        from tests.test_h3_mvp_cpu_checkpoint_state import _snapshot
        from tests.test_h3_mvp_cpu_hps import _Transport
        from visual_rl.algorithms.trainer import h3_stages, h3_dual_stages
        from visual_rl.models.lifecycle import h3_fsdp
        from visual_rl.algorithms.rewards.clients import hpsv3
        from visual_rl.composition.config.h3_profile import compile_h3_profile, preflight_h3_profile
        from visual_rl.core.serialization import to_plain_dict
        case = _controller_case(work, patch)
        # 不使用 fixture 的 implementation.py 占位身份：校验真正被测的完整生产树。
        case.assets.kwargs['code_root'] = source/'visual_rl'
        from visual_rl.runtime.h3_binding import H3RuntimeBinding
        bindings = {}
        initialize = H3RuntimeBinding.__init__
        def observed_binding(self, *args, **kwargs):
            initialize(self, *args, **kwargs)
            bindings[self.rank] = dict(content=deepcopy(self.content), score=asdict(self.scorer.identity))
        patch.setattr(H3RuntimeBinding, '__init__', observed_binding)
        patch.setattr(st, 'save_file', save_file)  # 输入构造完毕；生产执行不规范化序列化。
        captures = {rank: dict(gradients=[], states=[], rollouts=[], credit=[], updates=[]) for rank in range(8)}
        prepare = h3_fsdp.prepare_h3_fsdp
        def observed_prepare(*args, **kwargs):
            prepared = prepare(*args, **kwargs)
            row = captures[case.world.tls.rank]
            row['initial'] = _snapshot(prepared)
            def before(opt, args, kwargs):
                row['gradients'].append({n: p.grad.detach().clone() for n,p in prepared.trainable_named_parameters()})
            def after(opt, args, kwargs):
                row['states'].append(_snapshot(prepared))
            prepared.optimizer.register_step_pre_hook(before)
            prepared.optimizer.register_step_post_hook(after)
            return prepared
        patch.setattr(h3_fsdp, 'prepare_h3_fsdp', observed_prepare)
        rollout = h3_dual_stages.H3DualIterationStages.rollout
        advantage = h3_stages.H3IterationStages.advantage
        optimize = h3_dual_stages.H3DualIterationStages.optimize
        def observed_rollout(self, value):
            result = rollout(self, value)
            captures[self.rank]['rollouts'].append(dict(
                records=[asdict(r) for r in result.payload.dual_records],
                terminals=deepcopy(result.payload.dual_terminals)))
            return result
        def observed_advantage(self, value):
            result = advantage(self, value)
            captures[self.rank]['credit'].append(dict(rewards=deepcopy(result.payload.rewards),
                                                       advantages=deepcopy(result.payload.advantages)))
            return result
        def observed_optimize(self, value):
            result = optimize(self, value)
            captures[self.rank]['updates'].append(asdict(result.payload))
            return result
        patch.setattr(h3_dual_stages.H3DualIterationStages, 'rollout', observed_rollout)
        patch.setattr(h3_stages.H3IterationStages, 'advantage', observed_advantage)
        patch.setattr(h3_dual_stages.H3DualIterationStages, 'optimize', observed_optimize)
        from visual_rl.runtime import h3_lifecycle
        metrics = []
        write_json = h3_lifecycle._write_json
        def observed_json(path, payload):
            result = write_json(path, payload)
            if path.name == 'h3-last-metrics.json':
                metrics.append(deepcopy(payload))
            return result
        patch.setattr(h3_lifecycle, '_write_json', observed_json)
        transports = []
        def transport():
            leaf = _Transport(case.assets.transport.health)
            transports.append(leaf)
            return leaf
        patch.setattr(hpsv3, 'requests_session', transport)
        path = _config(case, 'default', output=work/'run')
        raw = yaml.safe_load(path.read_text())
        raw['overrides']['training'].update(algorithm='dual_flash', num_steps=10)
        path.write_text(yaml.safe_dump(raw))
        profile = compile_h3_profile(path)
        facts = preflight_h3_profile(profile)
        inputs = {str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in work.rglob('*') if p.is_file()}
        results, _ = _run(case, path)
        for result in results:
            if isinstance(result, Exception):
                raise result
            assert result.committed_steps == 2
        assert [row['step'] for row in metrics] == [0, 1]
        for row in captures.values():
            assert all(len(row[k]) == 2 for k in ('gradients', 'states', 'rollouts', 'credit', 'updates'))
            assert any(bool(g.count_nonzero()) for step in row['gradients'] for g in step.values())
            for step, state in enumerate(row['states'], 1):
                assert state['optimizer']['state']
                assert all(float(s['step']) == step for s in state['optimizer']['state'].values())
            assert any(not torch.equal(row['initial']['model'][n], row['states'][-1]['model'][n])
                       for n in row['initial']['model'])
        requests = sorted((method, url, kwargs['data']) for t in transports
                          for method,url,kwargs in t.calls if method == 'POST')
        assert len(requests) == 32
        checkpoints = {}
        from visual_rl.artifacts.checkpoint.manager import AtomicCheckpointManager
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        for step in (1, 2):
            root = work/'run'/'checkpoints'/f'step-{step}'
            AtomicCheckpointManager(root.parent).inspect_complete(root)
            dcp_to_torch_save(root/'h3-state'/'dcp', work/f'dcp-{step}.pt')
            checkpoints[step] = dict(
                files=sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()),
                json={str(p.relative_to(root)): json.loads(p.read_text()) for p in root.rglob('*.json')},
                pt={str(p.relative_to(root)): torch.load(p, weights_only=True) for p in root.rglob('*.pt')},
                dcp=torch.load(work/f'dcp-{step}.pt', weights_only=True))
        imports = {name: str(Path(module.__file__).resolve().relative_to(source))
                   for name,module in sys.modules.items() if name == 'visual_rl' or name.startswith('visual_rl.')
                   if getattr(module, '__file__', None)}  # relative_to 拒绝跨源码树导入
        assert all(Path(m.__file__).resolve().is_relative_to(fixtures) for n,m in sys.modules.items()
                   if (n == 'tests' or n.startswith('tests.')) and getattr(m, '__file__', None))
        torch.save(dict(inputs=inputs, semantic_id=facts.semantic_id, semantic_payload=to_plain_dict(facts.semantic_payload),
                        training=profile.training.payload(), bindings=bindings, captures=captures, final=case.saved,
                        checkpoints=checkpoints, metrics=metrics,
                        terminal_metrics=json.loads((work/'run'/'h3-last-metrics.json').read_text()),
                        hps_requests=requests, imports=imports), output)


if __name__ == '__main__':
    run(*(Path(p).resolve() for p in sys.argv[1:]))
