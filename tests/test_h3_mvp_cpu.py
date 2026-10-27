"""H3 contract CPU tests; no model weights, GPU, or distributed claims."""

from dataclasses import FrozenInstanceError
import math
from pathlib import Path

import pytest
import torch

import visual_rl
from visual_rl.algorithms.dynamics import h3_cps, h3_schedule
from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics, H3CPSRecord
from visual_rl.models import h3_joint
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.models.h3_joint import H3JointLayout
from visual_rl.models.implementations import h3_forward, h3_geometry, h3_loading
from visual_rl.models.implementations.h3_forward import h3_t2va_forward
from visual_rl.models.implementations.h3_geometry import H3T2VAGeometry


def test_h3_imports_are_from_this_worktree():
    checkout = Path(__file__).resolve().parents[1]
    assert Path(visual_rl.__file__).resolve().parent == checkout / "visual_rl"
    assert Path(h3_joint.__file__).resolve() == checkout / "visual_rl/models/h3_joint.py"
    assert Path(h3_schedule.__file__).resolve() == checkout / "visual_rl/algorithms/dynamics/h3_schedule.py"
    assert Path(h3_cps.__file__).resolve() == checkout / "visual_rl/algorithms/dynamics/h3_cps.py"
    assert Path(h3_geometry.__file__).resolve() == checkout / "visual_rl/models/implementations/h3_geometry.py"
    assert Path(h3_forward.__file__).resolve() == checkout / "visual_rl/models/implementations/h3_forward.py"
    assert Path(h3_loading.__file__).resolve() == checkout / "visual_rl/models/implementations/h3_loading.py"


def test_joint_roundtrip_batch_order_dtype_and_gradients():
    layout = H3JointLayout(video_rows=2, video_channels=3, audio_rows=4, audio_channels=5)
    assert (layout.video_numel, layout.audio_numel, layout.packed_width) == (6, 20, 26)
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        # Noncontiguous inputs exercise row ordering without requiring contiguous().
        video = torch.arange(12, dtype=dtype).reshape(2, 3, 2).transpose(1, 2).requires_grad_()
        audio = (100 + torch.arange(40, dtype=dtype)).reshape(2, 5, 4).transpose(1, 2).requires_grad_()
        packed = layout.pack(video, audio)
        expected = torch.tensor([
            [0, 2, 4, 1, 3, 5, 100, 104, 108, 112, 116, 101, 105, 109, 113, 117,
             102, 106, 110, 114, 118, 103, 107, 111, 115, 119],
            [6, 8, 10, 7, 9, 11, 120, 124, 128, 132, 136, 121, 125, 129, 133, 137,
             122, 126, 130, 134, 138, 123, 127, 131, 135, 139],
        ], dtype=dtype)
        assert torch.equal(packed, expected)
        v, a = layout.unpack(packed)
        assert torch.equal(v, video) and torch.equal(a, audio)
        assert packed.dtype == v.dtype == a.dtype == dtype
        assert packed.device == v.device == a.device == video.device == audio.device
        (v.sum() * 2 + a.sum() * 3).backward()
        assert torch.equal(video.grad, torch.full_like(video, 2))
        assert torch.equal(audio.grad, torch.full_like(audio, 3))
        # pack owns new storage; unpack intentionally aliases only that storage.
        with torch.no_grad():
            packed[0, 0] = -7
        assert video[0, 0, 0].item() == 0
        assert v[0, 0, 0].item() == -7
        assert torch.equal(a, audio)
    with pytest.raises(FrozenInstanceError):
        layout.audio_rows = 0


def test_joint_rejects_invalid_geometry_shapes_and_implicit_conversion():
    for field in ("video_rows", "video_channels", "audio_rows", "audio_channels"):
        for bad in (0, -1, True, 2.0, "2"):
            values = dict(video_rows=2, video_channels=3, audio_rows=4, audio_channels=5)
            values[field] = bad
            with pytest.raises(ValueError, match="positive integer"):
                H3JointLayout(**values)
    layout = H3JointLayout(2, 3, 4, 5)
    video, audio = torch.zeros(2, 2, 3), torch.zeros(2, 4, 5)
    for bad in (torch.zeros(2, 6), torch.zeros(0, 2, 3), torch.zeros(2, 3, 2)):
        with pytest.raises(ValueError, match="video"):
            layout.pack(bad, audio)
    for bad in (torch.zeros(1, 4, 5), torch.zeros(2, 0, 5), torch.zeros(2, 5, 4)):
        with pytest.raises(ValueError, match="audio"):
            layout.pack(video, bad)
    for bad in (audio.double(), torch.empty(2, 4, 5, device="meta")):
        with pytest.raises(ValueError, match="dtype and device"):
            layout.pack(video, bad)
    for bad in (torch.zeros(2, 25), torch.zeros(0, 26), torch.zeros(2, 2, 13)):
        with pytest.raises(ValueError, match="packed"):
            layout.unpack(bad)
    with pytest.raises(TypeError, match="tensors"):
        layout.pack(video, None)
    with pytest.raises(TypeError, match="tensor"):
        layout.unpack([0] * 26)


def test_dual_schedule_exact_default_grids_and_indexed_terminal():
    schedule = H3ScheduleSnapshot.build(10)
    base = torch.linspace(1, 0, 11, dtype=torch.float32, device="cpu")
    expected_video = 12 * base / (1 + 11 * base)
    expected_audio = 3 * base / (1 + 2 * base)
    assert schedule.num_steps == 10
    assert torch.equal(schedule.video_sigmas, expected_video)
    assert torch.equal(schedule.audio_sigmas, expected_audio)
    assert schedule.video_sigmas.dtype == schedule.audio_sigmas.dtype == torch.float32
    assert schedule.video_sigmas.device.type == schedule.audio_sigmas.device.type == "cpu"
    assert schedule.video_sigmas[5].item() == torch.tensor(12 / 13, dtype=torch.float32).item()
    assert schedule.audio_sigmas[5].item() == 0.75
    assert bool((schedule.video_sigmas[1:-1] > schedule.audio_sigmas[1:-1]).all())
    # Non-sequential/repeated reads cannot advance a hidden scheduler cursor.
    for i in (9, 0, 5, 0, 9):
        actual = schedule.transition(i)
        expected = (expected_video[i], expected_video[i + 1], expected_audio[i], expected_audio[i + 1])
        assert all(x.shape == () and x.dtype == torch.float32 for x in actual)
        assert all(torch.equal(x, y) for x, y in zip(actual, expected))
    vs, vu, aus, au = schedule.transition(9)
    assert vs > 0 and aus > 0 and vu == 0 and au == 0
    one = H3ScheduleSnapshot.build(1)
    assert tuple(x.item() for x in one.transition(0)) == (1, 0, 1, 0)
    unshifted = H3ScheduleSnapshot.build(10, video_shift=1, audio_shift=1)
    assert torch.equal(unshifted.video_sigmas, base)
    assert torch.equal(unshifted.audio_sigmas, base)


def test_schedule_owns_exact_values_and_returns_independent_tensors():
    video = torch.tensor([1, 0.9876543, 0.1234567, 0], dtype=torch.float32, requires_grad=True)
    audio = torch.tensor([1, 0.8, 0.2, 0], dtype=torch.float32)
    expected_v, expected_a = video.detach().clone(), audio.clone()
    schedule = H3ScheduleSnapshot(video, audio)
    with torch.no_grad():
        video.fill_(0)
        audio.fill_(0)
    schedule.video_sigmas.fill_(0)
    schedule.audio_sigmas.fill_(0)
    for scalar in schedule.transition(1):
        scalar.fill_(0)
    assert torch.equal(schedule.video_sigmas.view(torch.int32), expected_v.view(torch.int32))
    assert torch.equal(schedule.audio_sigmas.view(torch.int32), expected_a.view(torch.int32))
    assert not schedule.video_sigmas.requires_grad
    restored = H3ScheduleSnapshot(schedule.video_sigmas, schedule.audio_sigmas)
    assert restored == schedule
    with pytest.raises(FrozenInstanceError):
        schedule._video = (1, 0)


def test_schedule_rejects_invalid_grids_counts_shifts_and_indices():
    good = torch.tensor([1, 0.5, 0], dtype=torch.float32)
    invalid = (
        [], [1], [1, float("nan"), 0], [1, float("inf"), 0],
        [1, -0.1, 0], [1, 1.1, 0], [1, 1, 0], [1, 0, 0.5, 0],
        [0.9, 0.5, 0], [1, 0.5, 0.1], [1, 0],
    )
    for values in invalid:
        bad = torch.tensor(values, dtype=torch.float32)
        for grids in ((bad, good), (good, bad)):
            with pytest.raises(ValueError):
                H3ScheduleSnapshot(*grids)
    for bad in (good[None], good.double(), good.half(), good.to(torch.int64), good.to("meta")):
        with pytest.raises(ValueError):
            H3ScheduleSnapshot(bad, good)
    with pytest.raises(TypeError):
        H3ScheduleSnapshot([1, 0.5, 0], good)
    for bad in (0, -1, True, 10.0, "10"):
        with pytest.raises(ValueError, match="num_steps"):
            H3ScheduleSnapshot.build(bad)
    for bad in (0, -1, float("nan"), float("inf"), True, "3"):
        for key in ("video_shift", "audio_shift"):
            with pytest.raises(ValueError, match="shifts"):
                H3ScheduleSnapshot.build(10, **{key: bad})
    # Large finite shift collapses distinct base values in FP32: never shorten N.
    with pytest.raises(ValueError, match="strictly decreasing"):
        H3ScheduleSnapshot.build(10, video_shift=1e10)
    schedule = H3ScheduleSnapshot.build(10)
    for bad in (-1, 10, 11):
        with pytest.raises(IndexError):
            schedule.transition(bad)
    for bad in (True, 0.0, 0.5, "0", torch.tensor(0)):
        with pytest.raises(TypeError, match="index"):
            schedule.transition(bad)


def _cps_case():
    layout = H3JointLayout(1, 2, 1, 3)
    schedule = H3ScheduleSnapshot(torch.tensor([1., .8, .4, 0.]), torch.tensor([1., .6, .1, 0.]))
    dynamics = H3CPSDynamics(layout, schedule)
    x = (torch.arange(10).reshape(2, 5) / 7 - .3).half()
    prediction = torch.linspace(-.7, .9, 10).reshape(2, 5)
    return dynamics, x, prediction


def _generators(video_seed=123, audio_seed=456):
    return dict(video_generator=torch.Generator().manual_seed(video_seed),
                audio_generator=torch.Generator().manual_seed(audio_seed))


def test_cps_fp32_math_element_weighted_proxy_and_quantized_sampling():
    dynamics, x, prediction = _cps_case()
    stats = dynamics.stats(x, prediction, 1)
    means, qs = [], []
    # Hand-specified independent grids at index 1, not extracted from stats.
    for columns, s, u in ((slice(0, 2), .8, .4), (slice(2, 5), .6, .1)):
        s, u = torch.tensor(s), torch.tensor(u)
        q = u * math.sin(.7 * math.pi / 2)
        state, pred = x[:, columns].float(), prediction[:, columns]
        means.append((1 - u) * (state - s * pred) + torch.sqrt(u*u - q*q) * (state + (1-s)*pred))
        qs.append(q)
    expected_mean = torch.cat(means, dim=1)
    assert torch.equal(stats.mean, expected_mean)
    assert stats.mean.dtype == stats.video_q.dtype == stats.audio_q.dtype == torch.float32
    assert torch.equal(stats.video_q, qs[0]) and torch.equal(stats.audio_q, qs[1])
    assert stats.video_q != stats.audio_q
    torch.testing.assert_close(stats.video_dt, torch.tensor(-.4))
    torch.testing.assert_close(stats.audio_dt, torch.tensor(-.5))
    generators = _generators()
    record = dynamics.sample(x, prediction, 1, **generators)
    reference_rng = _generators()
    zv = torch.randn(2, 1, 2, generator=reference_rng['video_generator']).flatten(1)
    za = torch.randn(2, 1, 3, generator=reference_rng['audio_generator']).flatten(1)
    expected_action = torch.cat((means[0] + qs[0]*zv, means[1] + qs[1]*za), dim=1).half()
    assert torch.equal(record.action, expected_action)
    assert record.score_family == "cps_proxy_negative_mse" and record.quantization == "fp16"
    residual = record.action.float() - expected_mean
    video_score, audio_score = -residual[:, :2].square().mean(1), -residual[:, 2:].square().mean(1)
    expected_score = (2 * video_score + 3 * audio_score) / 5
    score = dynamics.evaluate(record, prediction)
    torch.testing.assert_close(score, expected_score)
    assert torch.equal(score, record.rollout_score)
    assert not torch.allclose(score, (video_score + audio_score) / 2)
    q = torch.cat((qs[0].expand(2), qs[1].expand(3)))
    gaussian = (-residual.square() / (2*q.square()) - q.log() - .5*math.log(2*math.pi)).mean(1)
    assert not torch.allclose(score, gaussian)
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        low = prediction.to(dtype)
        assert torch.equal(dynamics.stats(x, low, 1).mean, dynamics.stats(x, low.float(), 1).mean)


def test_cps_explicit_rng_reproducibility_modality_isolation_and_terminal_draws():
    dynamics, x, prediction = _cps_case()
    global_before = torch.random.get_rng_state().clone()
    first = dynamics.sample(x, prediction, 1, **_generators())
    repeated = dynamics.sample(x, prediction, 1, **_generators())
    assert torch.equal(first.action, repeated.action)
    changed_video = dynamics.sample(x, prediction, 1, **_generators(video_seed=789))
    changed_audio = dynamics.sample(x, prediction, 1, **_generators(audio_seed=789))
    assert not torch.equal(first.action[:, :2], changed_video.action[:, :2])
    assert torch.equal(first.action[:, 2:], changed_video.action[:, 2:])
    assert torch.equal(first.action[:, :2], changed_audio.action[:, :2])
    assert not torch.equal(first.action[:, 2:], changed_audio.action[:, 2:])
    for index in range(dynamics.schedule.num_steps):
        generators, reference_rng = _generators(), _generators()
        before = [g.get_state().clone() for g in generators.values()]
        record = dynamics.sample(x, prediction, index, **generators)
        torch.randn(2, 1, 2, generator=reference_rng['video_generator'])
        torch.randn(2, 1, 3, generator=reference_rng['audio_generator'])
        for old, actual, expected in zip(before, generators.values(), reference_rng.values()):
            assert not torch.equal(old, actual.get_state())
            assert torch.equal(actual.get_state(), expected.get_state())
        after = [g.get_state().clone() for g in generators.values()]
        assert torch.isfinite(dynamics.evaluate(record, prediction)).all()
        assert all(torch.equal(g.get_state(), state) for g, state in zip(generators.values(), after))
    # Sharing an explicitly supplied stream also pins video-before-audio draw order.
    shared, ref = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    record = dynamics.sample(x, prediction, 1, video_generator=shared, audio_generator=shared)
    stats = dynamics.stats(x, prediction, 1)
    expected_v = stats.mean[:, :2] + stats.video_q * torch.randn(2, 1, 2, generator=ref).flatten(1)
    expected_a = stats.mean[:, 2:] + stats.audio_q * torch.randn(2, 1, 3, generator=ref).flatten(1)
    assert torch.equal(record.action, torch.cat((expected_v, expected_a), dim=1).half())
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_cps_fixed_record_ownership_and_current_prediction_only_gradients():
    dynamics, x, prediction = _cps_case()
    prediction.requires_grad_()
    record = dynamics.sample(x, prediction, 1, **_generators())
    stored_x, stored_action, stored_score = record.x, record.action, record.rollout_score
    assert all(not t.requires_grad and t.grad_fn is None for t in (stored_x, stored_action, stored_score))
    assert prediction.grad is None
    # Constructor and readers both own copies, including externally supplied action.
    reconstructed = H3CPSRecord(dynamics, 1, stored_x, stored_action, stored_score)
    x.fill_(9)
    for tensor in (stored_x, stored_action, stored_score, record.x, record.action, record.rollout_score):
        tensor.fill_(9)
    assert torch.equal(record.x, reconstructed.x)
    assert torch.equal(record.action, reconstructed.action)
    assert torch.equal(record.rollout_score, reconstructed.rollout_score)
    with pytest.raises(FrozenInstanceError):
        record.index = 0
    current = (prediction.detach() + .17).requires_grad_()
    rng = torch.random.get_rng_state().clone()
    score = dynamics.evaluate(record, current)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not torch.equal(score.detach(), record.rollout_score)
    stats = dynamics.stats(record.x, current, 1)
    coefficients = []
    for s, u in ((.8, .4), (.6, .1)):
        s, u = torch.tensor(s), torch.tensor(u)
        q = u * math.sin(.7 * math.pi / 2)
        coefficients.append(-(1-u)*s + torch.sqrt(u*u-q*q)*(1-s))
    coefficient = torch.cat((coefficients[0].expand(2), coefficients[1].expand(3)))
    expected_gradient = 2 * (record.action.float() - stats.mean.detach()) * coefficient / 5
    score.sum().backward()
    torch.testing.assert_close(current.grad, expected_gradient)
    assert bool((current.grad[:, :2] != 0).any()) and bool((current.grad[:, 2:] != 0).any())
    assert prediction.grad is None


def test_cps_terminal_q0_quantization_residual_has_analytic_gradient():
    dynamics, x, prediction = _cps_case()
    # Non-FP16-exact predictions leave quantization residual even without noise.
    current = (prediction + .01234567).requires_grad_()
    index = dynamics.schedule.num_steps - 1
    stats = dynamics.stats(x, current, index)
    assert stats.video_q.item() == stats.audio_q.item() == 0
    record = dynamics.sample(x, current, index, **_generators())
    assert torch.equal(record.action, stats.mean.detach().half())
    score = dynamics.evaluate(record, current)
    residual = record.action.float() - stats.mean.detach()
    assert bool((residual != 0).any()) and bool((score < 0).all())
    s = torch.tensor([.4, .4, .1, .1, .1])
    expected_gradient = -2 * residual * s / 5
    score.sum().backward()
    torch.testing.assert_close(current.grad, expected_gradient)
    assert bool((current.grad[:, :2] != 0).any()) and bool((current.grad[:, 2:] != 0).any())
    assert torch.isfinite(current.grad).all()


def test_cps_does_not_relax_real_gaussian_positive_variance_contract():
    from visual_rl.algorithms.dynamics.interface import DynamicsContractError, TransitionInput, TransitionMeanStd

    dynamics, x, prediction = _cps_case()
    transition = TransitionInput(
        x_t=x.float(), model_prediction=prediction, t=torch.ones(2), t_next=torch.zeros(2),
        mask=torch.ones(2, dtype=torch.bool), transition_index=torch.zeros(2, dtype=torch.int64),
        condition_identity=("c", "c"), guidance_identity=("g", "g"),
        storage_dtype_identity=("torch.float32", "torch.float32"), quantization_identity=("fp32", "fp32"),
    )
    terminal = dynamics.stats(x, prediction, 2)
    with pytest.raises(DynamicsContractError, match="strictly positive"):
        TransitionMeanStd(terminal.mean, torch.zeros_like(terminal.mean), torch.full((2,), -.1)).validate_against(transition)
    positive = dynamics.stats(x, prediction, 1)
    std = torch.cat((positive.video_q.expand(2), positive.audio_q.expand(3))).expand(2, -1)
    TransitionMeanStd(positive.mean, std, torch.full((2,), -.1)).validate_against(transition)


def test_cps_rejects_invalid_inputs_contracts_underflow_and_overflow():
    dynamics, x, prediction = _cps_case()
    for eta in (True, 0, -1, 1.1, float('nan'), float('inf'), '0.7'):
        with pytest.raises(ValueError, match="eta"):
            H3CPSDynamics(dynamics.layout, dynamics.schedule, eta)
    assert torch.isfinite(H3CPSDynamics(dynamics.layout, dynamics.schedule, 1).stats(x, prediction, 1).mean).all()
    for index in (-1, 3, True, 1.0, torch.tensor(1)):
        with pytest.raises((IndexError, TypeError)):
            dynamics.sample(x, prediction, index, **_generators())
    for bad in (x.float(), x.clone().requires_grad_(), x[:, :-1], x[:0], x.to('meta'),
                torch.full_like(x, float('nan')), torch.full_like(x, float('inf'))):
        with pytest.raises(ValueError):
            dynamics.stats(bad, prediction, 1)
    for bad in (prediction[:, :1], prediction[:1], prediction.to('meta'), prediction.int(),
                torch.full_like(prediction, float('nan')), torch.full_like(prediction, float('inf'))):
        with pytest.raises(ValueError):
            dynamics.stats(x, bad, 1)
    with pytest.raises(TypeError):
        dynamics.stats(x, None, 1)
    with pytest.raises(TypeError, match="Generator"):
        dynamics.sample(x, prediction, 1, video_generator=None, audio_generator=torch.Generator())
    record = dynamics.sample(x, prediction, 1, **_generators())
    conflicts = (
        H3CPSDynamics(dynamics.layout, dynamics.schedule, .5),
        H3CPSDynamics(H3JointLayout(1, 3, 1, 2), dynamics.schedule),
        H3CPSDynamics(dynamics.layout, H3ScheduleSnapshot.build(3)),
    )
    for conflict in conflicts:
        with pytest.raises(ValueError, match="contract mismatch"):
            conflict.evaluate(record, prediction)
    for bad in (record.action.float(), record.action.requires_grad_(), record.action[:, :-1],
                record.action[:1], record.action.to('meta'), torch.full_like(x, float('nan')),
                torch.full_like(x, float('inf'))):
        with pytest.raises(ValueError):
            H3CPSRecord(dynamics, 1, x, bad, record.rollout_score)
    with pytest.raises(ValueError, match="rollout_score"):
        H3CPSRecord(dynamics, 1, x, record.action, torch.tensor([float('nan'), 0.]))
    with pytest.raises(ValueError, match="underflow"):
        H3CPSDynamics(dynamics.layout, dynamics.schedule, 1e-50).stats(x, prediction, 1)
    with pytest.raises(ValueError, match="quantization overflow"):
        dynamics.sample(x, torch.full_like(prediction, 1e6), 1, **_generators())
    with pytest.raises(ValueError, match="prediction in FP32"):
        dynamics.stats(x, torch.full(prediction.shape, 1e100, dtype=torch.float64), 2)
    with pytest.raises(ValueError, match="CPS proxy score"):
        dynamics.evaluate(record, torch.full_like(prediction, 1e20))


def test_h3_t2va_geometry_fixed_family_and_rejections():
    geometry = H3T2VAGeometry(256, 384, 124)
    assert geometry.latent_shape == (24, 37, 16, 24)
    assert geometry.audio_latents == 207
    assert geometry.joint_layout == H3JointLayout(3552, 96, 414, 32)
    assert geometry.joint_layout.packed_width == 354240
    for change in (dict(num_frames=121), dict(num_frames=107), dict(num_frames=362),
                   dict(height=0), dict(width=-32), dict(height=257), dict(width=2048),
                   dict(height=1024, width=1024), dict(height=True), dict(num_frames=124.),
                   dict(fps=25), dict(fps=24.), dict(task="ref2va"), dict(task="fl2va"),
                   dict(guidance=4.5), dict(guidance=True)):
        values = dict(height=256, width=384, num_frames=124)
        with pytest.raises(ValueError):
            H3T2VAGeometry(**(values | change))
    for count in (0, -1, True, 3.):
        with pytest.raises(ValueError):
            geometry.attention_inputs(count, device=torch.device("cpu"))
    with pytest.raises(FrozenInstanceError):
        geometry.width = 768


def test_h3_attention_positions_clock_stereo_grid_and_coverage():
    plan = H3T2VAGeometry(256, 384, 124).attention_inputs(3, device=torch.device("cpu"))
    text, audio, video = (plan[key] for key in ("text_indices", "audio_indices", "video_indices"))
    assert torch.equal(text, torch.arange(3))
    assert torch.equal(audio, torch.arange(3, 417))
    assert torch.equal(video, torch.arange(417, 3969))
    assert torch.equal(torch.cat((text, audio, video)), torch.arange(3969))
    tags, pos = plan["token_tags"], plan["position_ids"]
    assert tags.dtype == torch.int64 and pos.dtype == torch.float64
    assert bool((tags[text] == 1).all() & (tags[audio] == 2).all() & (tags[video] == 0).all())
    assert torch.equal(pos[text], torch.tensor([[0., 0, 0], [1., 0, 0], [2., 0, 0]], dtype=torch.float64))
    # Hand-derived rectangular grid: sqrt(16*24), 8 height and 12 width patches.
    h0, w0 = 16 * (1 - math.sqrt(2/3)), 16 * (1 - math.sqrt(3/2))
    dh = dw = 64 / math.sqrt(384)
    torch.testing.assert_close(pos[video[0]], torch.tensor([3, h0, w0], dtype=torch.float64))
    torch.testing.assert_close(pos[video[95]], torch.tensor([3, h0+7*dh, w0+11*dw], dtype=torch.float64))
    assert pos[video[95], 1] < h0 + 8*dh and pos[video[95], 2] < w0 + 12*dw
    # First short span, end of first 5-frame cycle, next short span, final frame.
    for frame, time in ((1, 3+5/3), (4, 3+65/3), (5, 3+85/3),
                        (6, 33), (36, 203)):
        assert pos[video[frame*96], 0].item() == pytest.approx(time, abs=1e-12)
    for row, time, width in ((0, 3, w0), (206, 209, w0),
                             (207, 3, w0+11*dw), (413, 209, w0+11*dw)):
        torch.testing.assert_close(pos[audio[row]], torch.tensor([time, 0, width], dtype=torch.float64))
    # New calls own tensors; consumers cannot corrupt subsequent plans.
    pos.fill_(0)
    assert H3T2VAGeometry(256, 384, 124).attention_inputs(3, device=torch.device("cpu"))["position_ids"].abs().sum() > 0


class _TinyH3Transformer(torch.nn.Module):
    """Only the heavy transformer leaf is replaced, never geometry/bridge/CPS."""

    def __init__(self):
        super().__init__()
        self.video_weight = torch.nn.Parameter(torch.tensor(.31))
        self.audio_weight = torch.nn.Parameter(torch.tensor(-.27))
        self.calls = []

    def forward(self, *, hidden_states, audio_hidden_states, encoder_hidden_states,
                timestep, timestep_indices, token_tags, position_ids, video_indices,
                audio_indices, text_indices, return_dict):
        self.calls.append({key: value for key, value in locals().items() if key != "self"})
        assert return_dict is False
        row_t = timestep[timestep_indices]
        condition = encoder_hidden_states.float().mean()
        video = self.video_weight * (hidden_states.float() + condition + row_t[video_indices][None, :, None])
        audio = self.audio_weight * (audio_hidden_states.float() + condition + row_t[audio_indices][None, :, None])
        return video, audio


def _forward_case():
    geometry = H3T2VAGeometry(32, 32, 124)
    x = torch.linspace(-.2, .8, geometry.joint_layout.packed_width).reshape(1, -1).half()
    return dict(x=x, geometry=geometry, text_embeddings=torch.ones(1, 3, 4),
                video_sigma=torch.tensor(.75), audio_sigma=torch.tensor(.25))


def test_h3_forward_single_callable_double_t_sign_dtype_and_gradients():
    leaf, case = _TinyH3Transformer(), _forward_case()
    routed = []

    def prepared(**kwargs):
        routed.append(True)
        return leaf(**kwargs)

    result = h3_t2va_forward(prepared, **case)
    assert len(routed) == len(leaf.calls) == 1
    call = leaf.calls[0]
    assert call["hidden_states"].dtype == call["audio_hidden_states"].dtype == torch.float16
    assert call["encoder_hidden_states"] is case["text_embeddings"]
    assert torch.equal(call["timestep"], torch.tensor([.25, .75]))
    row_t = call["timestep"][call["timestep_indices"]]
    for key, time in (("text_indices", .25), ("video_indices", .25), ("audio_indices", .75)):
        assert bool((row_t[call[key]] == time).all())
    assert call["position_ids"].dtype == torch.float64
    assert call["position_ids"].shape == (454, 3)
    layout = case["geometry"].joint_layout
    v, a = layout.unpack(case["x"])
    expected = layout.pack(-.31*(v.float()+1.25), .27*(a.float()+1.75))
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(leaf.video_weight.grad, -(v.float()+1.25).sum())
    torch.testing.assert_close(leaf.audio_weight.grad, -(a.float()+1.75).sum())
    assert leaf.training and all(p.dtype == torch.float32 for p in leaf.parameters())
    with torch.no_grad():
        repeated = h3_t2va_forward(prepared, **case)
    assert torch.equal(repeated, result.detach()) and not repeated.requires_grad
    # Equal sigma collapses to one unique time, not duplicate condition branches.
    h3_t2va_forward(prepared, **(case | {"audio_sigma": case["video_sigma"]}))
    assert torch.equal(leaf.calls[-1]["timestep"], torch.tensor([.25]))
    assert not bool(leaf.calls[-1]["timestep_indices"].any())


def test_h3_real_bridge_cps_sample_fixed_action_replay_backward():
    leaf, case = _TinyH3Transformer(), _forward_case()
    dynamics = H3CPSDynamics(case["geometry"].joint_layout, H3ScheduleSnapshot.build(3))
    vs, _, aus, _ = dynamics.schedule.transition(1)
    case.update(video_sigma=vs, audio_sigma=aus)
    with torch.no_grad():
        prediction = h3_t2va_forward(leaf, **case)
        record = dynamics.sample(case["x"], prediction, 1, **_generators())
    fixed_action = record.action
    current = h3_t2va_forward(leaf, **(case | {"x": record.x}))
    score = dynamics.evaluate(record, current)
    assert torch.equal(score.detach(), record.rollout_score)
    assert len(leaf.calls) == 2
    (-score.sum()).backward()
    for parameter in leaf.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad) and parameter.grad != 0
    assert torch.equal(record.action, fixed_action) and not fixed_action.requires_grad


def test_h3_forward_rejects_malformed_inputs_before_call_and_bad_outputs():
    leaf, case = _TinyH3Transformer(), _forward_case()
    x, text = case["x"], case["text_embeddings"]
    changes = [dict(geometry=None), dict(x=x[:, :-1]), dict(x=x.expand(2, -1)),
               dict(x=x.float()), dict(x=torch.full_like(x, float("nan"))), dict(x=x.to("meta")),
               dict(text_embeddings=text[:, :0]), dict(text_embeddings=text.expand(2, -1, -1)),
               dict(text_embeddings=text[0]), dict(text_embeddings=text[:, :, :0]),
               dict(text_embeddings=text.int()), dict(text_embeddings=text.to("meta")),
               dict(text_embeddings=torch.full_like(text, float("inf")))]
    for key in ("video_sigma", "audio_sigma"):
        changes.extend({key: value} for value in (0.5, torch.tensor([.5]), torch.tensor(.5).half(),
                       torch.tensor(float("nan")), torch.tensor(float("inf")), torch.tensor(-.1),
                       torch.tensor(1.1), torch.tensor(.5, requires_grad=True), torch.tensor(.5, device="meta")))
    for change in changes:
        with pytest.raises((ValueError, TypeError)):
            h3_t2va_forward(leaf, **(case | change))
    assert len(leaf.calls) == 0
    v, a = case["geometry"].joint_layout.unpack(x)
    for outputs in ((v,), [v, a], (None, a), (v[:, :-1], a), (v, a[:, :-1]),
                    (v.to("meta"), a), (v.int(), a), (v, a.float()),
                    (torch.full_like(v, float("nan")), a), (v, torch.full_like(a, float("inf")))):
        with pytest.raises((ValueError, TypeError)):
            h3_t2va_forward(lambda **kwargs: outputs, **case)


def _tiny_loading_class():
    # Only heavy class resolution is replaced. ConfigMixin, Accelerate, PEFT,
    # safetensors, meta construction, mapping and initialization are all real.
    from diffusers.configuration_utils import ConfigMixin, register_to_config

    class Block(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.attn = torch.nn.Module()
            for name in ("to_q", "to_k", "to_v"):
                setattr(self.attn, name, torch.nn.Linear(width, width))
            self.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(width, width)])
            proj = torch.nn.Module()
            proj.proj = torch.nn.Linear(width, width)
            self.ff = torch.nn.Module()
            self.ff.net = torch.nn.ModuleList([proj, torch.nn.Identity(), torch.nn.Linear(width, width)])
            self.gradient_checkpointing = True

        def forward(self, x):
            y = self.attn.to_q(x) + self.attn.to_k(x) + self.attn.to_v(x)
            y = self.attn.to_out[0](y)
            return x + y + self.ff.net[2](torch.nn.functional.silu(self.ff.net[0].proj(x)))

    class TinyH3(torch.nn.Module, ConfigMixin):
        config_name = "config.json"
        _keep_in_fp32_modules = ["proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope"]
        constructed_on_meta = []

        @register_to_config
        def __init__(self, in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2),
                     text_dim=4, hidden_size=4, num_layers=2):
            super().__init__()
            self.proj_in = torch.nn.Linear(in_channels * math.prod(patch_size), hidden_size)
            self.audio_proj_in = torch.nn.Linear(audio_in_channels, hidden_size)
            self.context_embedder = torch.nn.Linear(text_dim, hidden_size)
            self.time_embedder = torch.nn.Linear(hidden_size, hidden_size)
            self.time_embedder.register_buffer("scale", torch.tensor(.123456789))
            self.proj_out = torch.nn.Linear(hidden_size, 96)
            self.audio_proj_out = torch.nn.Linear(hidden_size, 32)
            self.rope = torch.nn.Module()
            self.rope.register_buffer("inv_freq", 1 / (10000. ** (torch.arange(4).float() / 4)), persistent=False)
            self.transformer_blocks = torch.nn.ModuleList([Block(hidden_size) for _ in range(num_layers)])
            self.token_refiner = Block(hidden_size)
            self.gradient_checkpointing = True
            type(self).constructed_on_meta.append(all(p.is_meta for p in self.parameters()))

        def forward(self, video, audio):
            def run(x, proj, out):
                h = proj(x.float()).bfloat16()
                h = self.token_refiner(h)
                for block in self.transformer_blocks:
                    h = block(h)
                return out(h.float())
            return run(video, self.proj_in, self.proj_out), run(audio, self.audio_proj_in, self.audio_proj_out)

    return TinyH3


def _loading_fixture(tmp_path, monkeypatch):
    import json
    from safetensors.torch import save_file

    cls = _tiny_loading_class()
    directory = tmp_path / "transformer"
    directory.mkdir()
    config = dict(_class_name="MiniMaxH3Transformer3DModel", in_channels=24, audio_in_channels=32,
                  patch_size=[1, 2, 2], text_dim=4, hidden_size=4, num_layers=2)
    (directory / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(h3_loading, "_resolve_h3_class", lambda: cls)
    rng = torch.random.get_rng_state().clone()
    meta = h3_loading.build_h3_meta(tmp_path, rank=2, alpha=4)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert cls.constructed_on_meta == [True]
    generator = torch.Generator().manual_seed(778)
    source = {name: torch.randn(meta.state_specs[target].shape, generator=generator) * .1
              for name, target in meta.base_to_model.items()}
    save_file(source, str(directory / "diffusion_pytorch_model.safetensors"))
    return meta, source, cls


def _load_cpu(meta, **kwargs):
    return h3_loading.load_h3_cpu_state(meta, **(dict(is_init_owner=True, host_weight_limit_bytes=10**8,
                                                    init_seed=42) | kwargs))


def test_h3_loader_real_meta_peft_scope_precision_and_cpu_backward(tmp_path, monkeypatch):
    meta, source, _ = _loading_fixture(tmp_path, monkeypatch)
    model = meta.model
    assert meta.block_names == ("transformer_blocks.0", "transformer_blocks.1")
    assert len(meta.target_names) == 12 and len(meta.lora_names) == 24
    assert all(p.is_meta for p in model.parameters())
    assert not any(getattr(m, "gradient_checkpointing", False) for m in model.modules())
    for name, p in model.named_parameters():
        assert p.requires_grad == (name in meta.lora_names)
        assert p.dtype == (torch.float32 if name in meta.lora_names or name.split('.')[0] in
                           {"proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"} else torch.bfloat16)
    assert not hasattr(model.token_refiner.attn.to_q, "lora_A")
    rope = torch.tensor([1., .1, .01, .001])
    torch.testing.assert_close(model.rope.inv_freq, rope, rtol=0, atol=0)
    assert model.rope.inv_freq.dtype == torch.float32 and not model.rope.inv_freq.is_meta
    assert set(meta.nonpersistent_buffers) == {"rope.inv_freq"}
    assert "time_embedder.scale" in meta.base_to_model
    rng = torch.random.get_rng_state().clone()
    loaded, repeated, different = _load_cpu(meta), _load_cpu(meta), _load_cpu(meta, init_seed=43)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(loaded.state) == set(model.state_dict()) == set(meta.state_specs)
    assert set(loaded.file_sha256) == {"config.json", "diffusion_pytorch_model.safetensors"}
    for name, value in loaded.state.items():
        assert value.device.type == "cpu" and value.shape == meta.state_specs[name].shape
        assert value.dtype == meta.state_specs[name].dtype and not value.requires_grad
        assert torch.equal(value, repeated.state[name])
        if name in meta.lora_names:
            if ".lora_A." in name:
                assert bool(value.ne(0).any()) and not torch.equal(value, different.state[name])
                assert value.abs().max() <= 1 / math.sqrt(value.shape[1])
            else:
                assert not bool(value.any())
        else:
            original = next(key for key, target in meta.base_to_model.items() if target == name)
            assert torch.equal(value, source[original].to(value.dtype))
    # CPU-only materialization, NOT an FSDP setter. assign preserves module
    # requires_grad and retains real constructor buffers without to_empty.
    model.load_state_dict(loaded.state, strict=True, assign=True)
    assert all(not p.is_meta for p in model.parameters())
    v, a = model(torch.ones(1, 2, 96), torch.ones(1, 3, 32))
    assert torch.isfinite(v).all() and torch.isfinite(a).all()
    for output in (v, a):
        model.zero_grad(set_to_none=True)
        output.square().mean().backward(retain_graph=True)
        assert any(p.grad is not None and bool(p.grad.ne(0).any()) for n, p in model.named_parameters() if '.lora_B.' in n)
        assert all(p.grad is None for n, p in model.named_parameters() if n not in meta.lora_names)
    assert torch.equal(model.rope.inv_freq, loaded.nonpersistent_buffers["rope.inv_freq"])


def test_h3_loader_two_shards_and_strict_keys_shapes_dtypes(tmp_path, monkeypatch):
    import json
    from safetensors.torch import save_file

    meta, source, _ = _loading_fixture(tmp_path, monkeypatch)
    directory = meta.transformer_dir
    single = directory / "diffusion_pytorch_model.safetensors"
    reference = _load_cpu(meta)
    single.unlink()
    keys = list(source)
    pieces = [dict((key, source[key]) for key in keys[::2]), dict((key, source[key]) for key in keys[1::2])]
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    mapping = {key: f"part{i}.safetensors" for i, part in enumerate(pieces) for key in part}
    index.write_text(json.dumps(dict(weight_map=mapping)))
    for i, part in enumerate(pieces):
        save_file(part, str(directory / f"part{i}.safetensors"))
    loaded = _load_cpu(meta)
    assert all(torch.equal(value, loaded.state[name]) for name, value in reference.state.items())
    # Each failure goes through real safetensors files and strict loader parsing.
    first = keys[0]
    for changed in (pieces[0] | {"unknown": torch.ones(1)},
                    {k: v for k, v in pieces[0].items() if k != first},
                    pieces[0] | {first: torch.ones(1)},
                    pieces[0] | {first: source[first].double()},
                    pieces[0] | {first: torch.full_like(source[first], float('nan'))},
                    pieces[0] | {keys[1]: source[keys[1]]}):
        save_file(changed, str(directory / "part0.safetensors"))
        with pytest.raises(ValueError):
            _load_cpu(meta)
    save_file(pieces[0], str(directory / "part0.safetensors"))
    for changed in (mapping | {first: "../outside.safetensors"}, mapping | {first: "/abs.safetensors"},
                    mapping | {first: "missing.safetensors"}, mapping | {first: "part1.safetensors"}):
        index.write_text(json.dumps(dict(weight_map=changed)))
        with pytest.raises(ValueError):
            _load_cpu(meta)
    index.write_text('{"weight_map":{},"weight_map":{}}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_cpu(meta)
    index.write_text(json.dumps(dict(weight_map=mapping)))
    (directory / "part0.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        _load_cpu(meta)


def test_h3_loader_owner_budget_and_duplicate_header_admission(tmp_path, monkeypatch):
    import json
    import safetensors

    meta, _, _ = _loading_fixture(tmp_path, monkeypatch)
    original = safetensors.safe_open
    reads = []

    class MetadataOnly:
        def __init__(self, *args, **kwargs):
            self.handle = original(*args, **kwargs)
        def __enter__(self):
            self.handle.__enter__()
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def keys(self):
            return self.handle.keys()
        def get_slice(self, key):
            return self.handle.get_slice(key)
        def get_tensor(self, key):
            reads.append(key)
            raise AssertionError("budget rejection must precede tensor payload access")

    monkeypatch.setattr(safetensors, "safe_open", MetadataOnly)
    with pytest.raises(PermissionError, match="owner"):
        _load_cpu(meta, is_init_owner=False)
    for limit in (0, -1, True, 1):
        with pytest.raises(ValueError, match="host"):
            _load_cpu(meta, host_weight_limit_bytes=limit)
    assert reads == []
    # Duplicate safetensors JSON keys must not be hidden by the Rust parser.
    file = meta.transformer_dir / "diffusion_pytorch_model.safetensors"
    entry = json.dumps(dict(dtype="F32", shape=[1], data_offsets=[0, 4]))
    header = ('{"x":' + entry + ',"x":' + entry + '}').encode()
    file.write_bytes(len(header).to_bytes(8, 'little') + header + b'\0' * 4)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_cpu(meta)


def test_h3_loader_config_scope_path_and_dependency_rejections(tmp_path, monkeypatch):
    import json

    # Real default resolver is an optional dependency, never a reference import.
    resolver = h3_loading._resolve_h3_class
    try:
        actual = resolver()
    except ImportError as error:
        assert "diffusers.models.transformers.transformer_minimax_h3" in str(error)
    else:
        assert actual.__name__ == "MiniMaxH3Transformer3DModel"
    meta, _, cls = _loading_fixture(tmp_path, monkeypatch)
    config_path = meta.transformer_dir / "config.json"
    original_config = config_path.read_text()
    with pytest.raises(ValueError, match="local"):
        h3_loading.build_h3_meta(tmp_path / 'missing-repo')
    for change in (dict(_class_name="Wan"), dict(in_channels=16), dict(audio_in_channels=2),
                   dict(patch_size=[1, 1, 1]), dict(text_dim=0), dict(num_layers=True)):
        config_path.write_text(json.dumps(json.loads(original_config) | change))
        with pytest.raises(ValueError, match="config|patch"):
            h3_loading.build_h3_meta(tmp_path)
    with pytest.raises(ValueError, match="config changed"):
        _load_cpu(meta)
    config_path.write_text('{"text_dim":4,"text_dim":8}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        h3_loading.build_h3_meta(tmp_path)
    config_path.write_text(original_config)
    for change in (dict(rank=0), dict(alpha=True), dict(alpha=float('inf'))):
        with pytest.raises(ValueError, match="LoRA"):
            h3_loading.build_h3_meta(tmp_path, **change)
    from_config = cls.from_config
    for defect in ('nonlinear', 'alias', 'precision', 'projection', 'missing'):
        def broken(config, defect=defect):
            model = from_config(config)
            if defect == 'nonlinear':
                model.transformer_blocks[0].attn.to_q = torch.nn.Identity()
            elif defect == 'alias':
                model.transformer_blocks[0].attn.to_k = model.transformer_blocks[0].attn.to_q
            elif defect == 'precision':
                model.rope.inv_freq = model.rope.inv_freq.bfloat16()
            elif defect == 'missing':
                del model.transformer_blocks[0].attn.to_v
            else:
                model.proj_in = torch.nn.Linear(1, 4)
            return model
        monkeypatch.setattr(cls, 'from_config', broken)
        with pytest.raises(ValueError):
            h3_loading.build_h3_meta(tmp_path)
    monkeypatch.setattr(cls, 'from_config', from_config)
    monkeypatch.setattr(h3_loading, '_resolve_h3_class', resolver)
    real_import = h3_loading.importlib.import_module

    def missing_h3(name, *args, **kwargs):
        if name == 'diffusers.models.transformers.transformer_minimax_h3':
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(h3_loading.importlib, 'import_module', missing_h3)
    with pytest.raises(ImportError, match="H3 requires optional.*no reference/fake fallback"):
        h3_loading.build_h3_meta(tmp_path)


def test_h3_loader_single_file_missing_extra_format_and_symlink_rejections(tmp_path, monkeypatch):
    import json
    from safetensors.torch import save_file

    meta, source, _ = _loading_fixture(tmp_path, monkeypatch)
    directory = meta.transformer_dir
    single = directory / "diffusion_pytorch_model.safetensors"
    key = next(iter(source))
    for changed in ({k: v for k, v in source.items() if k != key}, source | {'extra': torch.ones(1)}):
        save_file(changed, str(single))
        with pytest.raises(ValueError, match="exactly cover"):
            _load_cpu(meta)
    for seed in (True, -1, 2**63, .5):
        with pytest.raises(ValueError, match="init_seed"):
            _load_cpu(meta, init_seed=seed)
    single.unlink()
    # Outside transformer but still within pytest tmp_path: never access any
    # external artifact. Resolve-before-read rejects this symlink escape.
    outside = tmp_path / 'outside.safetensors'
    save_file(source, str(outside))
    single.symlink_to(outside)
    with pytest.raises(ValueError, match="escape"):
        _load_cpu(meta)
    single.unlink()
    (directory / 'diffusion_pytorch_model.bin').write_bytes(b'not a pickle fallback')
    with pytest.raises(ValueError, match="no bin fallback"):
        _load_cpu(meta)
    save_file(source, str(single))
    index = directory / 'diffusion_pytorch_model.safetensors.index.json'
    index.write_text(json.dumps(dict(weight_map={key: single.name})))
    with pytest.raises(ValueError, match="exactly one"):
        _load_cpu(meta)
