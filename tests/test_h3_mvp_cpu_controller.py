"""Controller prerequisite regression: cooperative prepare errors precede collectives.

No controller implementation is claimed here. This intentionally exposes the
W8 owner-boundary blocker; no skip/xfail and no real PG/CUDA/network is used.
"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import torch
from safetensors.torch import save_file

from tests.test_h3_mvp_cpu import _loading_fixture
from visual_rl.models.implementations import h3_loading
from visual_rl.models.lifecycle import h3_fsdp


def test_h3_controller_prepare_owner_read_failure_stops_all_ranks_before_mesh(tmp_path, monkeypatch):
    """A real nonfinite checkpoint rejection must not strand peers in prepare.

    Only WORLD/CUDA observations, metadata transport and the mesh entry are
    leaves. Actual meta/PEFT, API admission, metadata digest and owner weight
    reading/finite validation execute. The mesh sentinel avoids any actual
    collective hang; reaching it is the observed unsafe ordering, not a GPU
    deadlock claim.
    """
    first, weights, _ = _loading_fixture(tmp_path, monkeypatch)
    models = [first] + [h3_loading.build_h3_meta(tmp_path, rank=2, alpha=4) for _ in range(7)]
    weights[next(iter(weights))].fill_(float('nan'))
    save_file(weights, str(first.transformer_dir / 'diffusion_pytorch_model.safetensors'))
    # The file is structurally valid, with unchanged keys/shapes/dtypes.
    locations, headers = h3_loading._weight_headers(first.transformer_dir)
    assert set(locations) == set(first.base_to_model)
    assert all(tuple(headers[k]['shape']) == first.state_specs[v].shape
               for k, v in first.base_to_model.items())

    h3_fsdp._require_apis()
    pg, tls = object(), threading.local()
    barrier = threading.Barrier(8, timeout=10)
    receipts, entered_mesh = [None] * 8, []
    monkeypatch.setattr(h3_fsdp.dist, 'group', SimpleNamespace(WORLD=pg))
    monkeypatch.setattr(h3_fsdp.dist, 'is_available', lambda: True)
    monkeypatch.setattr(h3_fsdp.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(h3_fsdp.dist, 'get_backend', lambda group: 'nccl')
    monkeypatch.setattr(h3_fsdp.dist, 'get_world_size', lambda group: 8)
    monkeypatch.setattr(h3_fsdp.dist, 'get_rank', lambda group: tls.rank)

    def cuda(device):
        assert device == torch.device(f'cuda:{tls.rank}')
        return torch.device('cpu')

    def gather(output, receipt, *, group):
        assert group is pg
        receipts[tls.rank] = receipt
        barrier.wait()
        output[:] = receipts
        barrier.wait()

    class PeerAdvanced(RuntimeError):
        pass

    def mesh(group):
        assert group is pg
        entered_mesh.append(tls.rank)
        raise PeerAdvanced('peer entered mesh before owner initialization succeeded')

    monkeypatch.setattr(h3_fsdp, '_cuda_device', cuda)
    monkeypatch.setattr(h3_fsdp.dist, 'all_gather_object', gather)
    monkeypatch.setattr(h3_fsdp, '_world_mesh', mesh)

    def prepare(rank):
        tls.rank = rank
        try:
            h3_fsdp.prepare_h3_fsdp(models[rank], process_group=pg, device=f'cuda:{rank}',
                rank=rank, local_rank=rank, local_world_size=8,
                host_weight_limit_bytes=10**8, init_seed=42, constant_scheduler=True)
        except Exception as exc:
            return exc
        raise AssertionError('invalid checkpoint unexpectedly prepared')

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(prepare, rank) for rank in range(8)]
        errors = [future.result(timeout=20) for future in futures]
    assert isinstance(errors[0], ValueError) and 'nonfinite checkpoint tensor' in str(errors[0])
    # Required W8 safety invariant. Currently fails: all seven peers reach mesh.
    assert entered_mesh == [], (
        'rank0 read failure is not cooperatively exchanged inside prepare; '
        f'peers entered mesh: {sorted(entered_mesh)}; '
        f'errors: {[type(error).__name__ for error in errors]}'
    )
    assert all(isinstance(error, ValueError) for error in errors)
