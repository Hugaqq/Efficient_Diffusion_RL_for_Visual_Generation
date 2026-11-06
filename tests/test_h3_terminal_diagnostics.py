from types import SimpleNamespace as NS

import pytest

from visual_rl.runtime.h3_lifecycle import H3LifecycleBackend
from visual_rl.composition.config.h3_profile import H3EMAConfig


def terminal(tmp_path, metrics, enabled=True):
    training = NS(algorithm='dual_flash', global_samples=32, ema=H3EMAConfig())
    checkpoint = NS(step=3, path=tmp_path/'checkpoints/step-3', checkpoint_contract_id='c',
                    progress_id='p', state_tree_id='t')
    checkpoint.path.mkdir(parents=True, exist_ok=True)
    manager = NS(inspect_complete=lambda *a, **kw: None)
    graph = NS(profile=NS(launch=NS(max_optimizer_steps=3, output_dir=tmp_path, gpu_uuids=()),
                          training=training), session=NS(rank=0), run_id='child',
               facts=NS(semantic_payload={}))
    bound = NS(graph=graph, trainer=NS(next_optimizer_step=3),
               binding=NS(diagnostics_enabled=enabled, contract=None),
               transaction=NS(manager=manager, finalize_files=lambda f: f()))
    return H3LifecycleBackend().checkpoint(bound, (checkpoint, metrics))


def test_terminal_signed_policy_loss_and_default(tmp_path):
    basic = dict(step=2, sample_count=32, active_transition_count=32, loss=-2.79e-9, gradient_norm=.1)
    terminal(tmp_path, basic, enabled=False)
    metrics = dict(basic, policy_loss=-2.79e-9, kl_loss=.01, kl_video=.1, kl_audio=.2, kl=.15, lora_norm=1.)
    terminal(tmp_path, metrics)
    for key in ('policy_loss', 'kl', 'kl_audio', 'kl_video', 'kl_loss', 'lora_norm'):
        for bad in (float('nan'), float('inf'), True, -1.):
            if key == 'policy_loss' and bad == -1.:
                continue
            with pytest.raises(ValueError):
                terminal(tmp_path, dict(metrics, **{key: bad}))
