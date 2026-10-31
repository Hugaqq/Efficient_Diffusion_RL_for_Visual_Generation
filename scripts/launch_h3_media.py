"""Audited H3 observation-only launcher, preserving the production code tree.

Uses the real RunController/H3LifecycleBackend and an owned score callback.
No optimizer, sampler, checkpoint, owner identity or RNG mutation. Selected
media is copied while its lease is live, after the original scoring call.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import wave

BASE_SHA = 'e835fe39f6d208272a28afb3555f408854e14ca2'
ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def selection(source, sample_id):
    request = source._pending
    if request is None:
        raise ValueError('media observation requires a live source reservation')
    step = request.identity.optimizer_step + 1
    if request.group_layout != 'paired_4x8':
        raise ValueError('media launcher requires the explicit paired_4x8 layout')
    matches = [(gid, member) for sid, gid, member in request.samples if sid == sample_id]
    if len(matches) != 1:
        raise ValueError('sample absent or duplicated in current reservation')
    group, member = matches[0]
    if step % 10 or member != 0:
        return None
    return dict(step=step, sample_id=sample_id, group_id=group, member=member,
                prompt=request.prompts[sample_id], collection_id=request.collection_id,
                behavior_id=request.behavior_id,
                interpretation='rollout for upcoming optimizer update; not a post-update evaluation')


def save_media(*, root, metadata, media, reward, ffmpeg, ffprobe, launcher_sha):
    """Bounded CPU copy/encode, ffprobe verification, then atomic directory rename."""
    import torch
    video, audio, info = media.video, media.audio, media.metadata
    if (video.device.type != 'cpu' or audio.device.type != 'cpu' or
            video.dtype != torch.float32 or audio.dtype != torch.float32 or
            video.requires_grad or audio.requires_grad or video.ndim != 4 or video.shape[1] != 3 or
            audio.ndim != 2 or audio.shape[1] != 2 or
            video.shape[0] != info.frame_count or audio.shape[0] != info.sample_count or
            info.fps <= 0 or info.sample_rate <= 0 or not math.isfinite(reward)):
        raise ValueError('invalid detached CPU AV media/reward')
    if not bool(torch.isfinite(video).all()) or not bool(torch.isfinite(audio).all()):
        raise ValueError('nonfinite media')
    if video.numel()*video.element_size() + audio.numel()*audio.element_size() > 1024**3:
        raise ValueError('media exceeds one GiB export budget')
    parent = Path(root) / f"step-{metadata['step']:04d}"
    parent.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(metadata['sample_id'].encode()).hexdigest()
    target = parent / key
    if target.exists():
        raise FileExistsError('refusing to overwrite an existing rollout sample')
    with tempfile.TemporaryDirectory(prefix='.pending-', dir=parent) as tmp:
        staging = Path(tmp)
        raw, wav, mp4 = staging/'video.rgb', staging/'audio.wav', staging/'video.mp4'
        with raw.open('xb') as stream:
            for frame in video:
                # No in-place operation: scoring/training media remains unchanged.
                pixels = frame.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
                stream.write(pixels.permute(1,2,0).contiguous().numpy().tobytes())
        pcm = audio.detach().clamp(-1,1).mul(32767).round().to(torch.int16).numpy().astype('<i2',copy=False)
        with wave.open(str(wav), 'wb') as stream:
            stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(info.sample_rate)
            stream.writeframes(pcm.tobytes())
        cmd = [str(ffmpeg), '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{video.shape[3]}x{video.shape[2]}',
               '-r', str(info.fps), '-i', str(raw), '-i', str(wav),
               '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-threads', '1',
               '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
               '-movflags', '+faststart', str(mp4)]
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
        probe = subprocess.run([str(ffprobe), '-v', 'error', '-count_frames',
                                '-show_streams', '-of', 'json', str(mp4)], check=True,
                               capture_output=True, text=True, timeout=60)
        streams = json.loads(probe.stdout)['streams']
        videos = [s for s in streams if s['codec_type']=='video']
        audios = [s for s in streams if s['codec_type']=='audio']
        if (len(videos)!=1 or len(audios)!=1 or
                int(videos[0]['nb_read_frames'])!=info.frame_count or
                videos[0]['width']!=video.shape[3] or videos[0]['height']!=video.shape[2] or
                int(audios[0]['sample_rate'])!=info.sample_rate or audios[0]['channels']!=2):
            raise ValueError('encoded AV geometry/frame-count mismatch')
        numerator,denominator=map(int,videos[0]['avg_frame_rate'].split('/'))
        if numerator != info.fps * denominator:
            raise ValueError('encoded frame-rate mismatch')
        receipt = dict(metadata, reward=reward, launcher_sha=launcher_sha,
                       video_sha256=sha256_file(mp4), fps=info.fps, frames=info.frame_count,
                       audio_sample_rate=info.sample_rate, audio_samples=info.sample_count,
                       encoding='h264-crf18+aac; lossy viewing copy, not original HPS input',
                       ffprobe=streams)
        (staging/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        raw.unlink(); wav.unlink()
        for file in (mp4,staging/'receipt.json'):
            with file.open('rb') as stream: os.fsync(stream.fileno())
        os.rename(staging,target)
        fd=os.open(parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    print(json.dumps(dict(event='h3_rollout_video_saved',step=metadata['step'],sample_id=metadata['sample_id'],path=str(target/'video.mp4'),reward=reward)),flush=True)
    return target


def observed_score(original, *, source, root, ffmpeg, ffprobe, launcher_sha, saver=save_media):
    def score(*, sample_id, prompt, media):
        chosen = selection(source,sample_id)
        if chosen is not None and chosen['prompt'] != prompt:
            raise ValueError('observed prompt differs from reservation')
        reward = original(sample_id=sample_id,prompt=prompt,media=media)
        if chosen is not None:
            saver(root=root,metadata=chosen,media=media,reward=reward,
                  ffmpeg=ffmpeg,ffprobe=ffprobe,launcher_sha=launcher_sha)
        return reward
    return score


def prove_observation_only():
    def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if git('status','--porcelain'):
        raise ValueError('dirty source checkout')
    head=git('rev-parse','HEAD')
    if git('merge-base',BASE_SHA,head) != BASE_SHA:
        raise ValueError('media launcher requires audited base ancestry')
    allowed={'scripts/launch_h3_media.py','tests/test_flash_collection_runtime.py'}
    changed=set(git('diff','--name-only',BASE_SHA,head).splitlines())
    if not changed <= allowed:
        raise ValueError('upgrade changes files outside observation-only allowlist')
    trees={}
    for path in ('visual_rl','data','experiments'):
        old,new=(git('rev-parse',f'{sha}:{path}') for sha in (BASE_SHA,head))
        if old!=new:raise ValueError('training/model/data/config code changed')
        trees[path]=old
    return dict(base_sha=BASE_SHA,launcher_sha=head,unchanged_trees=trees,
                launcher_file_sha256=sha256_file(__file__),cadence=10,samples_per_prompt=1,
                selection='member0 of each of four prompt groups; independent of reward')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path)
    parser.add_argument('--ffmpeg',required=True,type=Path)
    parser.add_argument('--ffprobe',required=True,type=Path)
    args=parser.parse_args()
    proof=prove_observation_only()
    if not args.ffmpeg.is_file() or not args.ffprobe.is_file():raise ValueError('media binaries absent')
    from visual_rl.runtime.controller import RunController
    from visual_rl.runtime.h3_lifecycle import H3LifecycleBackend, _phase
    class ObservedBackend(H3LifecycleBackend):
        def bind(self,preflight,runtime,graph,prepared,transformed):
            bound=super().bind(preflight,runtime,graph,prepared,transformed)
            def attach():
                if graph.profile.training.algorithm!='dual_flash' or graph.profile.training.group_layout!='paired_4x8':
                    raise ValueError('observation launcher only supports formal paired Flash')
                root=graph.profile.launch.output_dir/'rollout-videos'
                if runtime.rank==0:
                    root.mkdir(exist_ok=True)
                    audit=root/f"launcher-{proof['launcher_sha']}.json"
                    evidence=dict(proof,run_id=graph.run_id,production_code_hash=graph.code_hash,
                                  ffmpeg_sha256=sha256_file(args.ffmpeg),ffprobe_sha256=sha256_file(args.ffprobe))
                    if audit.exists() and json.loads(audit.read_text())!=evidence:
                        raise ValueError('observation audit identity differs')
                    if not audit.exists():audit.write_text(json.dumps(evidence,indent=2))
                bound.stages.score_one=observed_score(bound.stages.score_one,source=graph.source,
                    root=root,ffmpeg=args.ffmpeg,ffprobe=args.ffprobe,launcher_sha=proof['launcher_sha'])
            _phase(runtime,'attach-media-observer',attach)
            return bound
    result=RunController(ObservedBackend()).run(args.config)
    print(json.dumps(dict(status='ok',committed_steps=result.committed_steps,
                          checkpoint=str(result.authoritative_checkpoint))),flush=True)


if __name__=='__main__':main()
