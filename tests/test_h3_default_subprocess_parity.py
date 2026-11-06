"""两个 git archive、两个独立解释器；共同 CPU 外部叶不含 visual_rl 实现。"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tarfile

import torch


FIXTURE_MODULES = ('test_h3_mvp_cpu', 'test_h3_mvp_cpu_entry', 'test_h3_mvp_cpu_bound_checkpoint',
                   'test_h3_mvp_cpu_binding', 'test_h3_mvp_cpu_profile', 'test_h3_mvp_cpu_decode',
                   'test_h3_mvp_cpu_text_cache', 'test_h3_mvp_cpu_hps', 'test_h3_mvp_cpu_prepare',
                   'test_h3_mvp_cpu_coordination', 'test_h3_mvp_cpu_checkpoint_state',
                   'test_h3_mvp_cpu_stages', 'test_h3_mvp_cpu_pool')


def _equal(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8)), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            _equal(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x,y) in enumerate(zip(a,b)):
            _equal(x, y, f'{path}/{i}')
    elif isinstance(a, float):
        assert type(b) is float and struct.pack('!d', a) == struct.pack('!d', b), path
    else:
        assert type(a) is type(b) and a == b, (path, a, b)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def _separate_code_and_storage_ids(result):
    """显式字段白名单；独立重算派生身份，不泛化忽略 hash/id 字段。"""
    value = deepcopy(result)
    separate = {}
    def take(mapping, key, path):
        separate[f'{path}/{key}'] = mapping[key]
        mapping[key] = f'<{key}: separately verified>'

    binding = value['bindings'][0]
    for rank in range(8):
        _equal(binding, value['bindings'][rank])
    code = binding['content']['code']['content_sha256']
    for step, checkpoint in value['checkpoints'].items():
        prefix = f'checkpoints/{step}/json'
        files = checkpoint['json']
        envelope = files['checkpoint_contract.json']
        contract = envelope['contract']
        transaction = files['h3_transaction.json']
        live = transaction['binding']['live']
        assert live['code'] == contract['code_identity'] == code
        assert live['resources'] == _digest(dict(content=binding['content'], cache=live['cache'], score=binding['score']))
        assert contract['recipe_id'] == 'materialized-recipe.v2:' + _digest(dict(h3=live['semantics'], resources=live['resources']))
        assert contract['algorithm_materialization_spec_id'] == 'algorithm-materialization-spec.v1:' + _digest(dict(
            implementation='visual_rl.algorithms.trainer.h3_dual_stages:H3DualIterationStages',
            math=value['training'], dynamics=live['dynamics'], code=code))
        runtime = _digest(dict(live=live, projection=contract['model_state_projection'], numerics=contract['model_execution_numerics']))
        assert contract['runtime_bound_contract_id'] == contract['components'][0]['runtime_bound_contract_id'] == runtime
        assert envelope['checkpoint_contract_id'] == transaction['binding']['contract_id'] == files['complete.json']['checkpoint_contract_id'] == _digest(contract)
        tree = files['state_tree.json']
        assert tree['state_tree_id'] == files['complete.json']['state_tree_id'] == _digest(tree['tree'])
        for row in tree['tree']['files']:
            # complete/tree 已在各子进程读取真实文件验证；这里只分离物理序列化差异。
            if row['path'] in ('h3-state/dcp/.metadata', 'h3_transaction.json'):
                take(row, 'sha256', f'{prefix}/state_tree.json/{row["path"]}')
        for key in ('code_identity', 'recipe_id', 'algorithm_materialization_spec_id', 'runtime_bound_contract_id'):
            take(contract, key, f'{prefix}/checkpoint_contract.json/contract')
        take(contract['components'][0], 'runtime_bound_contract_id', f'{prefix}/checkpoint_contract.json/contract/components/0')
        take(envelope, 'checkpoint_contract_id', f'{prefix}/checkpoint_contract.json')
        take(transaction['binding'], 'contract_id', f'{prefix}/h3_transaction.json/binding')
        for key in ('code', 'resources'):
            take(live, key, f'{prefix}/h3_transaction.json/binding/live')
        take(tree, 'state_tree_id', f'{prefix}/state_tree.json')
        for key in ('state_tree_id', 'checkpoint_contract_id'):
            take(files['complete.json'], key, f'{prefix}/complete.json')
    for rank in range(8):
        for key in ('content_sha256', 'file_count', 'byte_count'):
            take(value['bindings'][rank]['content']['code'], key, f'bindings/{rank}/content/code')
    separate['import_inventory'] = value.pop('imports')
    assert 'visual_rl.runtime.h3_binding' in separate['import_inventory']
    return value, separate


def test_default_flash_two_updates_from_complete_isolated_git_archives(tmp_path):
    root = Path(__file__).resolve().parents[1]
    revisions = {'baseline': '7c85751', 'current': 'HEAD'}
    revisions = {label: subprocess.check_output(['git', 'rev-parse', revision], cwd=root, text=True).strip()
                 for label, revision in revisions.items()}
    (tmp_path/'source-revisions.json').write_text(json.dumps(revisions, indent=2))
    sources = {}
    for label, revision in revisions.items():
        archive = tmp_path/f'{label}.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git', 'archive', revision], cwd=root, stdout=stream, check=True, timeout=30)
        sources[label] = tmp_path/label
        sources[label].mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(sources[label], filter='data')
    fixtures = tmp_path/'fixtures'
    (fixtures/'tests').mkdir(parents=True)
    for module in ('__init__', *FIXTURE_MODULES):
        shutil.copyfile(sources['current']/'tests'/f'{module}.py', fixtures/'tests'/f'{module}.py')
    worker = root/'tests/support/h3_default_parity_worker.py'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    work = tmp_path/'identical-input-path'
    results = {}
    for label, source in sources.items():
        work.mkdir()
        with (tmp_path/f'{label}.log').open('w') as log:
            result = subprocess.run([sys.executable, '-I', '-B', str(worker), str(source), str(fixtures),
                                     str(work), str(tmp_path/f'{label}.pt')], cwd=source, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, (tmp_path/f'{label}.log').read_text()[-12000:]
        results[label] = torch.load(tmp_path/f'{label}.pt', weights_only=True)
        shutil.rmtree(work)
    a, old_ids = _separate_code_and_storage_ids(results['baseline'])
    b, new_ids = _separate_code_and_storage_ids(results['current'])
    (tmp_path/'separate-fields.json').write_text(json.dumps(dict(baseline=old_ids, current=new_ids), indent=2))
    assert old_ids['bindings/0/content/code/content_sha256'] != new_ids['bindings/0/content/code/content_sha256']
    _equal(a, b)
