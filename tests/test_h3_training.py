"""CPU tests only: real small-tensor math/autograd, explicit fake model/reward.

These do NOT load H3 or demonstrate GPU capacity, PEFT/CUDA compatibility,
real HPS rewards or model quality. The original seven tests remain separate.
"""
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from visual_rl.algorithms.dynamics.h3_joint import (joint_transition, flow_mean_std,
    weighted_mean, validate_target_layout, NAMES)
from visual_rl.algorithms.trainer.h3 import (H3Diagnostic, H3GRPODiagnostic, H3Transition,
    collect_sample, replay, aligned_advantages, optimize_group, hps_preprocess, atomic_checkpoint)
from visual_rl.models.h3_training import H3TrainingAdapter, H3PreparedRoot
from visual_rl.models.h3 import H3Result

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    state = {"video": torch.tensor([[[0.2, -0.4], [0.8, 0.1], [-0.2, 0.4]]]),
             "audio": torch.tensor([[[0.4, -0.1], [0.6, 0.2]]])}
    velocity = {n: (x*0.13).requires_grad_() for n, x in state.items()}
    masks = {n: torch.ones_like(x, dtype=torch.bool) for n, x in state.items()}
    return state, velocity, {"video": 0.8, "audio": 0.4}, {"video": 0.7, "audio": 0.2}, {"video": 0.97, "audio": 0.91}, masks


def test_formula_sign_clocks_effective_std_gradient_and_independent_oracle():
    state, velocities, sigmas, following, maximum, masks = inputs()
    actions = {n: x+0.03 for n, x in state.items()}
    _, density = joint_transition(state, velocities, sigmas, following, maximum, masks, noise=0.7, action=actions)
    expectations = {}
    for n in NAMES:
        # Independent explicit FF formula; never calls tested helper for expected.
        x, v = state[n], velocities[n]
        s, t = torch.tensor(sigmas[n]), torch.tensor(following[n])
        dt = t-s
        g2 = s/(1-s) * 0.7**2
        mean = x * (1 + g2/(2*s)*dt) - v * (1 + g2*(1-s)/(2*s))*dt
        sd = torch.sqrt(g2) * torch.sqrt(-dt)
        logp = (-0.5*((actions[n]-mean)/sd)**2 - torch.log(sd)-0.5*math.log(2*math.pi)).mean()
        torch.testing.assert_close(density.means[n], mean, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(density.stds[n], sd, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(density.component[n][0], logp, rtol=1e-6, atol=1e-7)
        expectations[n] = logp
    expected = (6*expectations["video"]+4*expectations["audio"])/10
    torch.testing.assert_close(density.aggregate[0], expected)
    density.aggregate.sum().backward()
    for value in velocities.values():
        assert value.grad is not None and torch.isfinite(value.grad).all() and (value.grad != 0).any()
    assert not torch.equal(density.stds["video"], density.stds["audio"])


def test_equal_unequal_counts_masks_and_anchor_rejection():
    values = {"video": torch.tensor([2.]), "audio": torch.tensor([8.])}
    assert weighted_mean(values, {"video": 3, "audio": 1}).item() == 3.5
    assert weighted_mean(values, {"video": 5, "audio": 5}).item() == 5
    state, v, s, nxt, mx, mask = inputs()
    mask["video"][0, 0, :] = False
    action, density = joint_transition(state, v, s, nxt, mx, mask, noise=0.7,
                                      generator=torch.Generator().manual_seed(4))
    assert density.counts == {"video": 4, "audio": 4}
    assert torch.equal(action["video"][~mask["video"]], state["video"][~mask["video"]])
    altered = {n: x.clone() for n, x in action.items()}
    altered["video"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="anchors"):
        joint_transition(state, v, s, nxt, mx, mask, noise=0.7, action=altered)
    mask["audio"].zero_()
    with pytest.raises(ValueError, match="omit"):
        joint_transition(state, v, s, nxt, mx, mask, noise=0.7, action=action)


@pytest.mark.parametrize("sigma,following,noise", [(1., .5, .7), (.8, 0., .7), (.8, .8, .7), (.8, .7, 0.), (.8, .7, float('nan'))])
def test_no_density_for_endpoints_or_zero_noise(sigma, following, noise):
    x = torch.ones(1, 2, 2)
    with pytest.raises(ValueError, match="density"):
        flow_mean_std(x, x, sigma, following, .9, noise)


def test_sample_fixed_action_exact_parity_detach_and_rng_preservation():
    args = inputs()
    generator = torch.Generator().manual_seed(42)
    before = generator.get_state().clone()
    action, old = joint_transition(*args, noise=.7, generator=generator)
    assert not torch.equal(before, generator.get_state())
    states = (torch.get_rng_state().clone(), generator.get_state().clone(), random.getstate(), np.random.get_state())
    fixed = {n: x.clone().requires_grad_() for n, x in action.items()}
    _, new = joint_transition(*args, noise=.7, generator=generator, action=fixed)
    for n in NAMES:
        assert torch.equal(old.component[n], new.component[n])
    assert torch.equal(old.aggregate, new.aggregate)
    assert torch.equal(states[0], torch.get_rng_state()) and torch.equal(states[1], generator.get_state())
    assert states[2] == random.getstate() and np.array_equal(states[3][1], np.random.get_state()[1])
    new.aggregate.sum().backward()
    assert all(x.grad is None for x in fixed.values())
    detached = old.detached()
    assert not detached.aggregate.requires_grad and detached.aggregate.grad_fn is None


class TinyJointLoRA(torch.nn.Module):
    """Explicit fake H3 compute leaf, actual differentiable rank-one update."""
    def __init__(self):
        super().__init__()
        self.register_buffer("frozen", torch.tensor(0.13))
        self.lora_A = torch.nn.Parameter(torch.tensor(0.2))
        self.lora_B = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden_states, audio_hidden_states, **kwargs):
        w = self.frozen + self.lora_A * self.lora_B
        return hidden_states*w+0.1, audio_hidden_states*(w+0.2)-0.05


class TinyAdapter(H3TrainingAdapter):
    def __init__(self):
        super().__init__("/unused", "/unused", {}, device="cpu")
        self.calls = []

    def encode(self, request):
        with torch.inference_mode():
            encoded = torch.ones(1, 2, 3)
        return encoded.clone()

    def prepare_policy(self, **_):
        self.root = H3PreparedRoot(TinyJointLoRA()).eval()
        self.trainables = tuple((n, p) for n, p in self.root.named_parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW([p for _, p in self.trainables], lr=.003, weight_decay=0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.assert_owner()

    def prepare_latents(self, request, prompt):
        g = torch.Generator().manual_seed(request.seed)
        state = {"video": torch.randn(1, 3, 2, generator=g), "audio": torch.randn(1, 2, 2, generator=g)}
        layout = {"text_indices": torch.tensor([0, 1]), "audio_indices": torch.tensor([2, 3]),
                  "video_indices": torch.tensor([4, 5, 6]), "token_tags": torch.tensor([1, 1, 2, 2, 0, 0, 0]),
                  "position_ids": torch.zeros(7, 3)}
        return state, layout, (1, 1, 1, 1)

    def predict(self, state, prompt, layout, video_time, audio_time):
        self.calls.append((float(video_time), float(audio_time), torch.is_grad_enabled()))
        video, audio = self.transformer(hidden_states=state["video"], audio_hidden_states=state["audio"])
        return {"video": video, "audio": audio}

    def decode(self, request, state, geometry):
        assert self.transformer is None and not self.graph_live
        assert all(p.device.type == 'cpu' for p in self.root.parameters())
        pixels = torch.zeros(request.frames, request.height, request.width, 3, dtype=torch.uint8)
        wave = torch.ones(2, round(request.frames/24*40)*800)*.01
        return H3Result(request, pixels, wave)


def tiny_group(config=None):
    config = config or H3Diagnostic()
    adapter = TinyAdapter()
    adapter.encoded_prompt = adapter.encode(config.request(0, 0))
    assert not adapter.encoded_prompt.is_inference()
    adapter.prepare_policy()
    rows = [collect_sample(adapter, config.request(0, i), config, 'group0', lambda *a, **k: None)[2] for i in range(2)]
    return adapter, config, rows


def test_rollout_paired_record_root_gradient_real_tiny_optimizer_and_owner():
    a, c, rows = tiny_group()
    assert len(a.calls) == 78 and all(not grad for _, _, grad in a.calls)
    assert all(len(row) == 1 and row[0].index == 20 for row in rows)
    assert all(row[0].old.counts == {'video': 6, 'audio': 4} for row in rows)
    assert a.calls[20][0] != a.calls[20][1]
    assert not torch.equal(rows[0][0].action['audio'], rows[1][0].action['audio'])
    ids = [r[0].sample_id for r in rows]
    objects = [id(p) for _, p in a.trainables]
    audit = optimize_group(a, rows, ids, [1., 2.], c, lambda *a, **k: None)
    assert audit['trainable_before'] != audit['trainable_after'] and audit['gradient_norm_pre_clip'] > 0
    assert audit['gradient_norm_post_clip'] > 0 and audit['lora_delta_l2'] > 0 and audit['lora_delta_max'] > 0
    assert len(a.calls) == 80 and all(grad for _, _, grad in a.calls[-2:])
    assert audit['gradient_nonzero_names'] == ['dit.lora_B']
    assert all(t['ratio'] == 1 and t['delta'] == {'video': 0., 'audio': 0., 'aggregate': 0.} for t in audit['transitions'])
    a.park_policy()
    assert [id(p) for _, p in a.trainables] == objects
    with a.policy(gradients=True):
        with pytest.raises(RuntimeError, match='active'):
            a.park_policy()
    a.optimizer.param_groups[0]['params'] = []
    with pytest.raises(RuntimeError, match='ownership'):
        a.assert_owner()


def test_parameter_conversion_scope_preserves_objects_and_restores_flag_on_failure(monkeypatch):
    a = TinyAdapter()
    a.prepare_policy()
    original = a.root.to
    previous = torch.__future__.get_swap_module_params_on_conversion()
    ids = [id(p) for _, p in a.trainables]
    def conversion(*args, **kwargs):
        assert torch.__future__.get_swap_module_params_on_conversion() is True
        return original(*args, **kwargs)
    monkeypatch.setattr(a.root, 'to', conversion)
    a.park_policy()
    assert [id(p) for _, p in a.trainables] == ids
    assert torch.__future__.get_swap_module_params_on_conversion() == previous
    def failed(*args, **kwargs):
        assert torch.__future__.get_swap_module_params_on_conversion() is True
        raise RuntimeError('explicit fake device failure')
    monkeypatch.setattr(a.root, 'to', failed)
    with pytest.raises(RuntimeError, match='device failure'):
        a._move_policy('cpu')
    assert torch.__future__.get_swap_module_params_on_conversion() == previous
    assert [id(p) for _, p in a.trainables] == ids


def test_wrong_group_reward_alignment_degenerate_and_no_fake_update():
    a, c, rows = tiny_group()
    ids = [r[0].sample_id for r in rows]
    before = [p.detach().clone() for _, p in a.trainables]
    for rewards in ([1., 1.], [1., float('nan')]):
        with pytest.raises(ValueError, match='degenerate'):
            optimize_group(a, rows, ids, rewards, c, lambda *a, **k: None)
    with pytest.raises(ValueError, match='identity'):
        aligned_advantages(rows, list(reversed(ids)), [1., 2.])
    bad = [(replace(rows[0][0], group_id='other'),), rows[1]]
    with pytest.raises(ValueError, match='same-prompt'):
        aligned_advantages(bad, ids, [1., 2.])
    assert all(torch.equal(old, p) for old, (_, p) in zip(before, a.trainables))
    assert not a.optimizer.state


def test_density_corruption_nan_failure_leaves_optimizer_untouched():
    a, c, rows = tiny_group()
    ids = [r[0].sample_id for r in rows]
    bad_old = replace(rows[1][0].old, component={**rows[1][0].old.component, 'audio': rows[1][0].old.component['audio'] + .01})
    bad = [rows[0], (replace(rows[1][0], old=bad_old),)]
    with pytest.raises(ValueError, match='audio density mismatch'):
        optimize_group(a, bad, ids, [1., 2.], c, lambda *a, **k: None)
    assert not a.optimizer.state and all(p.grad is None for _, p in a.trainables)
    action = {n: x.clone() for n, x in rows[0][0].action.items()}
    action['audio'][0, 0, 0] = float('nan')
    with a.policy(gradients=True), pytest.raises(ValueError, match='nonfinite'):
        replay(a, replace(rows[0][0], action=action))


def test_layout_rejects_prefix_padding_duplicate_and_bad_mask():
    a, c, rows = tiny_group()
    r = rows[0][0]
    for layout in ({**r.layout, 'audio_indices': torch.tensor([2])},
                   {**r.layout, 'text_indices': torch.tensor([0, 0])},
                   {**r.layout, 'token_tags': torch.zeros(7, dtype=torch.long)}):
        with pytest.raises(ValueError):
            validate_target_layout(r.state, r.prompt, layout)
    with pytest.raises(ValueError, match='masks'):
        joint_transition(r.state, r.state, r.sigmas, r.following, r.sigma_max,
                         {**r.masks, 'video': r.masks['video'][..., :1]}, noise=.7, action=r.action)


def test_reward_preprocessing_explicit_selection_resize_and_original_unchanged():
    # Each frame's color encodes source index. No real HPS request in CPU test.
    video = torch.arange(124, dtype=torch.uint8)[:, None, None, None].expand(124, 4, 6, 3).clone()
    original = video.clone()
    frames, receipt = hps_preprocess(video)
    expected_indices = [round(i*123/80) for i in range(81)]
    assert receipt['source_frame_indices'] == expected_indices
    assert frames.shape == (81, 480, 832, 3) and frames.dtype == torch.uint8
    assert frames[:, 0, 0, 0].tolist() == expected_indices
    assert receipt['source_shape'] == [124, 4, 6, 3] and receipt['scorer_top_k'] == 24
    assert receipt['audio_reward'] is False and torch.equal(video, original)
    with pytest.raises(ValueError):
        hps_preprocess(video[:80])


def test_real_reward_client_wiring_uses_original_contract_not_h3_geometry(monkeypatch):
    from visual_rl.algorithms.trainer.h3 import H3VideoReward
    import visual_rl.algorithms.rewards.clients.hpsv3 as clients
    calls = []
    class SpyClient:
        # Explicit fake HTTP leaf; actual preprocessing/JPEG encoding runs.
        def __init__(self, **kwargs):
            self.contract = kwargs['contract']
            assert self.contract.payload() == {'frames': 81, 'height': 480, 'width': 832, 'top_k': 24}
            calls.append('construct')
        def healthcheck(self):
            calls.append('health')
        def score_video(self, **kwargs):
            assert len(kwargs['frames_base64']) == 81
            assert kwargs['sample_id'] == 'update-1-sample-0'
            assert kwargs['prompt'] == H3Diagnostic().prompt
            assert all(isinstance(x, str) and len(x) > 0 for x in kwargs['frames_base64'])
            calls.append('score')
            return 0.123, {'frame_scores': [0.123]*81, 'selected_frame_indices': list(range(24))}
        def close(self):
            calls.append('close')
    monkeypatch.setattr(clients, 'Hpsv3VideoClient', SpyClient)
    a = TinyAdapter()
    a.prepare_policy()
    result = a.decode(H3Diagnostic().request(0, 0), {}, ())
    reward = H3VideoReward('http://127.0.0.1:18820', 'a'*64)
    score, metadata = reward(result)
    reward.close()
    assert score == .123 and metadata['source_shape'] == [124, 64, 96, 3]
    assert metadata['scorer_shape'] == [81, 480, 832, 3] and metadata['audio_reward'] is False
    assert calls == ['construct', 'health', 'score', 'close']


def test_checkpoint_real_tiny_state_rng_atomic_and_explicit_no_resume(tmp_path):
    a, c, rows = tiny_group()
    ids = [r[0].sample_id for r in rows]
    audit = optimize_group(a, rows, ids, [1., 2.], c, lambda *a, **k: None)
    checkpoint = atomic_checkpoint(a, tmp_path, 1, c, audit, {'git_sha': 'cpu_fake'})
    data = torch.load(checkpoint/'training.pt', weights_only=False)
    assert data['step'] == 1 and data['resume_supported'] is False and data['optimizer']['state']
    assert data['lr'] == a.lr_scheduler.state_dict() and torch.equal(data['rng']['torch'], torch.get_rng_state())
    for n, p in a.trainables:
        assert torch.equal(data['lora'][n], p) and data['lora'][n].device.type == 'cpu'
    assert not (tmp_path/'step-1.tmp').exists()
    with pytest.raises(FileExistsError):
        atomic_checkpoint(a, tmp_path, 1, c, audit, {})
    command = subprocess.run([sys.executable, '-m', 'visual_rl.train_h3', '--expected-sha', 'a'*40,
        '--model-root', '/unused', '--converted-root', '/unused', '--model-revision', 'b'*40,
        '--output', '/unused', '--resume', 'x'], cwd=ROOT, text=True, capture_output=True)
    assert command.returncode != 0 and 'unrecognized arguments: --resume' in command.stderr
    assert 'SUCCESS' not in command.stdout


def test_algorithm_one_and_five_actual_cpu_updates_fake_heavy_leaves(tmp_path, monkeypatch):
    import visual_rl.algorithms.trainer.h3 as module
    monkeypatch.setattr(module, 'export_result', lambda result, folder: {'cpu_fake_encoder': True})
    def publish(path, payload):
        Path(path).write_text(json.dumps(payload))
    for count in (1, 5):
        out = tmp_path/str(count)
        out.mkdir()
        a = TinyAdapter()
        def fake_reward(result):
            return float(result.request.seed % 2), {'cpu_fake_reward': True}
        algorithm = H3GRPODiagnostic(a, fake_reward, H3Diagnostic(), out, {'cpu_fake': True}, publish, lambda: None)
        result = algorithm.run(count)
        assert result['committed_steps'] == count and not result['normal_capacity_verified']
        hashes = []
        for step in range(1, count+1):
            audit = json.loads((out/f'group-{step}'/'update.json').read_text())
            assert audit['gradient_norm_pre_clip'] > 0 and audit['trainable_before'] != audit['trainable_after']
            assert (out/f'step-{step}'/'training.pt').is_file()
            trajectory = torch.load(out/f'group-{step}'/'trajectory.pt', weights_only=False)
            assert all(not row[0].old.aggregate.requires_grad for row in trajectory)
            hashes.append(audit['trainable_after'])
        assert len(set(hashes)) == count
        assert len(a.calls) == count * 80
        assert not (out/'SUCCESS.json').exists()  # CLI alone finalizes after close/safety


def test_native_training_encoder_unwraps_inference_tensor_and_live_close_rejects(monkeypatch):
    from visual_rl.models.h3_components import H3InferenceAdapter
    with torch.inference_mode():
        original = torch.ones(1, 2, 3)
    monkeypatch.setattr(H3InferenceAdapter, 'encode', lambda *args: original)
    a = H3TrainingAdapter('/unused', '/unused', {}, device='cpu')
    encoded = a.encode(H3Diagnostic().request(0, 0))
    assert not encoded.is_inference() and torch.equal(encoded, original)
    linear = torch.nn.Linear(3, 2)
    linear(encoded).sum().backward()
    assert linear.weight.grad is not None
    tiny = TinyAdapter()
    tiny.prepare_policy()
    with tiny.policy(gradients=True):
        with pytest.raises(RuntimeError, match='live training graph'):
            tiny.close()
    tiny.close()
    assert tiny.root is None and tiny.optimizer is None and tiny.closed


def test_real_preparation_control_path_with_explicit_fake_diffusers_peft(monkeypatch, tmp_path):
    # API/control-path fake, NOT proof the real H3/PEFT/CUDA stack can train.
    import types
    import visual_rl.models.h3_training as model_module
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            for name in ('to_q', 'to_k', 'to_v'):
                setattr(self.attn, name, torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16))
            self.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)])
            self.ff = torch.nn.Module()
            self.ff.net = torch.nn.ModuleList([torch.nn.Linear(2, 2).to(torch.bfloat16)])
    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = torch.nn.Linear(2, 2)
            self.transformer_blocks = torch.nn.ModuleList([Block()])
            self.token_refiner = torch.nn.Module()
            self.token_refiner.refiner_blocks = torch.nn.ModuleList([Block()])
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs['local_files_only'] and kwargs['use_safetensors']
            assert kwargs['torch_dtype'] == torch.bfloat16
            return cls()
        def add_adapter(self, config):
            assert config.r == 4 and config.lora_alpha == 8 and config.lora_dropout == 0
            assert config.target_modules == ['to_q', 'to_k', 'to_v', 'to_out.0']
            for block in [*self.transformer_blocks, *self.token_refiner.refiner_blocks]:
                block.attn.to_q.lora_A = torch.nn.Parameter(torch.ones(1, 2, dtype=torch.bfloat16))
                block.attn.to_q.lora_B = torch.nn.Parameter(torch.zeros(2, 1, dtype=torch.bfloat16))
        def enable_gradient_checkpointing(self):
            self.gradient_checkpointing = self.token_refiner.gradient_checkpointing = True
    monkeypatch.setitem(sys.modules, 'diffusers', types.SimpleNamespace(MiniMaxH3Transformer3DModel=Native))
    monkeypatch.setitem(sys.modules, 'peft', types.SimpleNamespace(LoraConfig=lambda **kwargs: types.SimpleNamespace(**kwargs)))
    monkeypatch.setattr(model_module.importlib.metadata, 'version', lambda name: '0.18.1')
    folder = tmp_path / 'transformer'
    folder.mkdir()
    (folder / 'model.safetensors').write_bytes(b'explicit fake weight; stat only')
    a = H3TrainingAdapter('/unused', tmp_path, {}, device='cpu',
                          resource_probe=lambda: {'gpu_free_gib': 0.1, 'host_available_gib': 128.})
    a.prepare_policy(rank=4, alpha=8, learning_rate=1e-4)
    assert isinstance(a.root, H3PreparedRoot) and len(a.trainables) == 4
    assert all('lora_' in n and p.dtype == torch.float32 and p.requires_grad for n, p in a.trainables)
    assert not a.root.dit.training and a.root.dit.gradient_checkpointing
    assert all(not p.requires_grad for n, p in a.root.named_parameters() if 'lora_' not in n)
    assert a.root.dit.proj_in.weight.dtype == torch.float32
    assert a.root.dit.transformer_blocks[0].attn.to_k.weight.dtype == torch.bfloat16
    assert not hasattr(a.root.dit, '_diffusers_hook')
    a.assert_owner()
    with pytest.raises(RuntimeError, match='once'):
        a.prepare_policy(rank=4, alpha=8, learning_rate=1e-4)


def test_failure_no_durable_or_success_and_original_candidate_imports(tmp_path, monkeypatch):
    import visual_rl
    import visual_rl.algorithms.trainer.h3 as module
    assert Path(visual_rl.__file__).resolve().parent == ROOT/'visual_rl'
    assert Path(module.__file__).resolve() == ROOT/'visual_rl/algorithms/trainer/h3.py'
    monkeypatch.setattr(module, 'export_result', lambda *args: {'cpu_fake_encoder': True})
    a = TinyAdapter()
    def broken_reward(result):
        raise RuntimeError('explicit fake scorer failure')
    algorithm = H3GRPODiagnostic(a, broken_reward, H3Diagnostic(), tmp_path, {},
        lambda p, data: Path(p).write_text(json.dumps(data)), lambda: None)
    with pytest.raises(RuntimeError, match='scorer'):
        algorithm.run(1)
    assert algorithm.committed_steps == 0 and not (tmp_path/'step-1').exists()
    assert not (tmp_path/'SUCCESS.json').exists() and not a.optimizer.state
    with pytest.raises(ValueError):
        H3Diagnostic(width=1344, height=768)
    with pytest.raises(ValueError):
        H3Diagnostic(stochastic_indices=(0, 38))


def test_no_training_gpu_reserve_in_stages_events_and_heartbeat(tmp_path, monkeypatch):
    from visual_rl.models import h3_components as components
    from scripts import launch_flash140 as launch
    from visual_rl.models.h3_components import H3InferenceAdapter, stage_budget
    assert H3TrainingAdapter.gpu_memory_policy == 'oom'
    assert H3InferenceAdapter.gpu_memory_policy == 'reserve'
    sample = {'gpu_free_gib': 0.1, 'host_available_gib': 128., 'disk_free_gib': 128.,
              'torch_peak_reserved_gib': 10000., 'process_peak_rss_gib': 12.}
    policy = H3TrainingAdapter.gpu_memory_policy
    for gpu in (0., .1, 15.963806, 16., 31.99):
        observed = {**sample, 'gpu_free_gib': gpu}
        launch.check_budget(observed, gpu_memory_policy=policy)
        assert stage_budget(1024, 1024, observed, gpu_memory_policy=policy) > 24
        # Existing Wan/inference defaults still reject low GPU headroom.
        if gpu < 16:
            with pytest.raises(RuntimeError, match='gpu_free'):
                launch.check_budget(observed)
        with pytest.raises(RuntimeError, match='GPU 32'):
            stage_budget(1024, 1024, observed)
    for invalid in (-1., float('nan'), float('inf')):
        with pytest.raises(RuntimeError, match='telemetry'):
            launch.check_budget({**sample, 'gpu_free_gib': invalid}, gpu_memory_policy=policy)
        with pytest.raises(RuntimeError, match='telemetry'):
            stage_budget(1, 1, {**sample, 'gpu_free_gib': invalid}, gpu_memory_policy=policy)
    for field in ('host_available_gib', 'disk_free_gib'):
        with pytest.raises(RuntimeError, match=field):
            launch.check_budget({**sample, field: 0.}, gpu_memory_policy=policy)
    with pytest.raises(RuntimeError, match='host'):
        stage_budget(1, 1, {**sample, 'host_available_gib': 24.}, gpu_memory_policy=policy)

    # Inherited encoder/video-decoder/audio-decoder phase gates must use the
    # training policy, not the inference default; no weight bytes are read.
    folder = tmp_path/'component'
    folder.mkdir()
    (folder/'model.safetensors').write_bytes(b'explicit CPU fake asset')
    monkeypatch.setattr(components, 'release_memory', lambda: None)
    a = H3TrainingAdapter('/unused', '/unused', {}, device='cpu', resource_probe=lambda: dict(sample))
    for phase in ('encoder', 'video_decoder', 'audio_decoder'):
        with a._phase(phase, folder):
            assert a.phase == phase
        assert a.phase is None
    inference = H3InferenceAdapter('/unused', '/unused', {}, device='cpu', resource_probe=lambda: dict(sample))
    with pytest.raises(RuntimeError, match='GPU 32'):
        with inference._phase('encoder', folder):
            pytest.fail('inference default must still reject low GPU headroom')

    monkeypatch.setattr(launch, 'resource_sample', lambda _: dict(sample))
    monitor = launch.ResourceMonitor(tmp_path, tmp_path/'resources.json', gpu_memory_policy=policy,
                                     gpu_reserved_limit=.01)
    # Even an explicit low peak limit is inactive in existing 'oom' policy.
    algorithm = H3GRPODiagnostic(a, None, H3Diagnostic(), tmp_path, {}, launch.publish, monitor.sample)
    algorithm.event('replay_forward')  # same callback used at every real event
    saved = json.loads((tmp_path/'resources.json').read_text())
    assert saved['sample']['gpu_free_gib'] == .1  # measured telemetry never padded
    assert saved['peaks']['torch_peak_reserved_gib'] == 10000.
    class Once:
        def __init__(self):
            self.calls = 0
        def wait(self, seconds):
            self.calls += 1
            return self.calls > 1
        def is_set(self):
            return False
    monitor.stop = Once()
    monkeypatch.setattr(launch.os, 'kill', lambda *args: pytest.fail('no low-GPU heartbeat signal allowed'))
    monitor._run()  # real heartbeat path, bounded by explicit fake wait
    assert monitor.error is None
    sample['host_available_gib'] = 23.
    with pytest.raises(RuntimeError, match='host_available'):
        monitor.sample()


def test_cli_uses_no_gpu_reserve_and_preserves_native_oom_failure(tmp_path, monkeypatch):
    # Actual CLI/monitor/algorithm/objective/checkpoint; heavy compute/HTTP/media
    # and telemetry leaves are explicit CPU fakes. Injected OOM tests propagation,
    # NOT an actual CUDA allocation failure or training capacity experiment.
    import types
    import importlib.metadata
    from scripts import launch_flash140 as launch
    from visual_rl import train_h3
    from visual_rl.models import h3_components as components
    import visual_rl.algorithms.trainer.h3 as trainer
    from visual_rl.models.h3_training import H3TrainingAdapter
    raw, converted = tmp_path/'raw', tmp_path/'converted'
    for folder in (raw/'FL2VA/text_encoder', converted/'transformer', converted/'vae', converted/'audio_vae'):
        folder.mkdir(parents=True)
        (folder/'model.safetensors').write_bytes(b'explicit CPU fake asset; metadata only')
    sample = {'gpu_free_gib': .1, 'host_available_gib': 128., 'disk_free_gib': 128.,
              'torch_peak_reserved_gib': 10000., 'process_peak_rss_gib': 12.}
    monkeypatch.setattr(launch, 'LOCK', tmp_path/'owned.lock')
    monkeypatch.setattr(launch, 'git_identity', lambda sha: sha)
    monkeypatch.setattr(launch, 'resource_sample', lambda _: dict(sample))
    monkeypatch.setattr(components, 'audit_runtime', lambda: {'explicit_CPU_fake_modules': True})
    monkeypatch.setattr(torch, '__version__', '2.10.0+cu128')
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: {'peft': '0.18.1',
        'transformers': '5.19.0'}.get(name) or original_version(name))
    monkeypatch.setitem(sys.modules, 'peft', types.SimpleNamespace(__file__='explicit_CPU_fake_peft'))
    monkeypatch.setattr(H3TrainingAdapter, 'load', classmethod(lambda cls, **kwargs: TinyAdapter()))
    class FakeReward:
        def __init__(self, *_):
            pass
        def __call__(self, result):
            return float(result.request.seed % 2), {'explicit_CPU_fake_reward': True}
        def close(self):
            pass
    monkeypatch.setattr(trainer, 'H3VideoReward', FakeReward)
    monkeypatch.setattr(trainer, 'export_result', lambda *args: {'explicit_CPU_fake_encoder': True})
    def args(out):
        return ['--config', str(ROOT/'configs/h3/training-diagnostic.json'), '--expected-sha', 'a'*40,
                '--model-root', str(raw), '--converted-root', str(converted), '--model-revision', 'b'*40,
                '--output', str(out), '--updates', '1']
    out = tmp_path/'one'
    assert train_h3.main(args(out)) == 0
    for name in ('owner.json', 'environment.json', 'SUCCESS.json'):
        record = json.loads((out/name).read_text())
        assert record['gpu_memory_policy'] == 'oom' and record['gpu_reserve_enabled'] is False
        assert record['gpu_reserved_limit_gib'] is None
    assert json.loads((out/'SUCCESS.json').read_text())['committed_steps'] == 1
    assert json.loads((out/'resources.json').read_text())['sample']['gpu_free_gib'] == .1

    original_predict = TinyAdapter.predict
    def native_oom(self, *args, **kwargs):
        if torch.is_grad_enabled():
            raise torch.cuda.OutOfMemoryError('explicit CPU-injected native OOM during replay')
        return original_predict(self, *args, **kwargs)
    monkeypatch.setattr(TinyAdapter, 'predict', native_oom)
    failed = tmp_path/'oom'
    with pytest.raises(torch.cuda.OutOfMemoryError, match='native OOM'):
        train_h3.main(args(failed))
    record = json.loads((failed/'FAILED.json').read_text())
    assert record['failure_kind'] == 'cuda_oom' and record['notify_user_capacity'] is True
    assert record['gpu_memory_policy'] == 'oom' and not record['gpu_reserve_enabled']
    assert record['phase'] == 'replay_forward' and record['committed_steps'] == 0
    assert 'OutOfMemoryError' in record['traceback'] and record['resources']['gpu_free_gib'] == .1
    assert not (failed/'SUCCESS.json').exists() and not (failed/'step-1').exists()
