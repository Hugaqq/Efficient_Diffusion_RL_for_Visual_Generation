# H3-M2a3b-decode — W2 Coding B

## 判定与工作区

- `implementation_ready: true`，仅当前原子项实现/本分支CPU门禁完成，交simplifier-b；不是W2汇合批准或完整MVP。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-f234ad70-77ae-4489-b4e7-27a97c6218c8`。
- 开始/结束HEAD：`8e674e8894c2af61188f7523933cff585a6812d6`；接收工作区干净。未切分支、手工commit/merge/push；引擎snapshot待生成。
- 已读当前task、plan、assignment B、wave-02、已有本配对approved反馈及实际geometry/joint。无changes返工项。

## 唯一改动/下游接口

新 `visual_rl/models/implementations/h3_decode.py`、新 `tests/test_h3_mvp_cpu_decode.py` 及本交付记录。无旧文件修改，不引用A/C本波新模块，不修改原44项或原Flash七项。

```python
decoder = H3CPUDecoder(
    local_model_dir,
    max_media_bytes=explicit_positive_limit,
    max_decode_bytes=explicit_positive_limit,
)
with decoder.decode_one(cpu_detached_fp16_terminal, geometry) as media:
    video = media.video  # CPU FP32 contiguous [T,3,H,W]
    audio = media.audio  # CPU FP32 contiguous [L,2]
    info = media.metadata
    # 未来RewardStage在lease内部评分，不长期保存tensor别名
```

一个decoder最多一条在途lease（nonblocking lock）；同lease重入或另起第二条拒绝。退出（含消费者异常）清空media owner对tensor的引用，并拒绝再次访问media.video/audio；decoder可新开下一条lease。外部消费者故意保存的tensor别名无法被Python撤销，必须遵守lease生命周期，不宣称强制RSS回收。

`H3AVMetadata`记录实际`frame_count/sample_count/fps/sample_rate`，以属性给出两个实际duration。124帧为124/24秒；207 audio latents产生165600 samples即5.175秒，二者不同被原样记录，不trim/pad以伪装相等。

## 实际生产路径

1. CPU detached FP16 B1 joint terminal经既有geometry.joint_layout.unpack；shape/device/dtype/finite/gradient检查及预算在load/decode之前执行。
2. 两组件的本地config准入：存在目录/JSON、明确正确class、全部24/32个numeric finite mean/std且std>0、audio明确32000采样率。拒duplicate JSON key、缺配置、symlink与shard index路径逃逸。不使用scalar scaling_factor或默认mean0/std1。
3. 默认lazy公开路径：
   - `diffusers.models.autoencoders.autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3`
   - `diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio.AutoencoderKLMiniMaxH3Audio`
4. 实际调用`from_pretrained(local_root, subfolder='vae'/'audio_vae', torch_dtype=torch.float32, local_files_only=True)`；构造处于CPU device上下文，无remote code/下载/reference/tiny fallback。缺符号/API给明确错误。
5. 模型eval、requires_grad false；全部参数/浮点buffer核验CPU FP32，拒已被降精度的buffer而不是把BF16静默转回。加载模型的latent stats与读到的本地config逐值核验。decode作用域no_grad并禁CPU autocast，不泄漏到消费者with-body。
6. video rows从(time,patch_y,patch_x)及feature(channel,dy,dx)独立反patch到`[1,24,T_lat,H_lat,W_lat]`；audio channel-major rows到`[2,32,A_lat]`。FP32逐通道`latent*std+mean`并检查溢出。
7. 真实`decode(latents, return_dict=False)`。video严格`[1,3,num_frames,height,width]`、CPU FP32 finite；逆ImageNet pixel mean/std并clamp[0,1]，转换为[T,3,H,W]。audio严格`[2,1,A_lat*800]`、CPU FP32 finite；同时核验模型config采样率32000及实际hop_length800，再转[L,2]，不额外clip/删静音。
8. video helper局部模型在返回前释放，随后才加载audio。失败不发布半套AV、不将模型或完整collection媒体保存在decoder；无GC/GPU内存回落日志或保证。

## 容量边界

`estimate_bytes(geometry)`返回：
- 单条AV payload：`4*(3*F*H*W + 2*A_lat*800)`。
- 保守tensor工作集：`14*joint_width + 3*media_bytes`，即FP16输入+三份FP32 latent+三份AV输出/转换副本。

分别对显式`max_media_bytes/max_decode_bytes`准入，不缩画布/改帧数/降dtype。**不包含VAE权重、加载器副本及模型内部activation/workspace，不是总RSS保证**。真实host资源上限仍须另行测量；跨rank aux轮转并发1由未来阶段负责，本项不建PG/服务/调度器。

## 来源与许可

读UniRL LICENSE全文（Apache-2.0）及固定`8cfbac17765004afb68fef08a4e011a4c8daf555`的vae.py、vendor packing的unpatchify/unpack、两个autoencoder文件头与实际config/decode合同。独立实现必要数学/调用，不复制vendor源码。

本轮直接读取vendor/VENDOR_COMMIT.txt：记录Diffusers PR14355提交`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`，明确两autoencoder在`models/autoencoders/`下的原路径。本记录覆盖旧会话对该文件缺失的推定；未修改历史报告或参考资料。该历史路径证据不是当前安装版本/真实权重兼容证明。

## 真实CPU测试

六项新feature测试：
1. 当前worktree导入身份；真实geometry/packing及逐位置独立unpatchify参考、非均匀stats、stereo顺序、像素逆norm/clamp、严格124帧/165600 samples/真实duration与无额外audio clip。
2. 两leaf释放顺序（weakref）、单lease/重入/第二条拒绝、消费者异常后owner释放和后续新lease。
3. media/工作集超限和非法terminal在模型加载前拒绝。
4. 非法class/stat/config/rate、缺文件、symlink及shard-index逃逸在load前拒绝。
5. 错帧数/音频长度/stereo/FP32/finite/tuple、错误hop/rate/loaded stats/buffer与leaf异常无半媒体交付、释放leaf；有限输入反归一化overflow拒绝。
6. 默认公开resolver缺可选依赖给明确ImportError，无fallback。

仅两重VAE模型类/模型叶子在测试替换；tiny输出真实依赖传入latent，生产unpatchify/反归一化/转换/预算/生命周期代码实际执行。没有运行真实H3 VAE from_pretrained权重加载、GPU或媒体质量验证。

最后源码编辑后实际运行新版两门禁，均exit0，无skip/xfail/全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 13.09s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 50 passed in 3.58s（既有44 + 本项6；不含尚未汇合A/C W2新项）
```

`git diff`为空（仅三个授权新文件未跟踪），`git diff --check` exit0。获准`git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py`无输出、exit0；原七项完整文件与历史基线一致。空tracked diff不代替新文件审核。

## 后续/风险

交simplifier-b审本项；changes回B修复并重跑两集。引擎汇合不是批准，W2全部配对裁决后orchestrator需亲跑完整通配符/Flash。

未来RewardStage必须在完整rollout后逐条消费本lease，绑定真实模型资源和rank-turn并发。真实reward、ModelAdapter/registry/controller、world/phase/global norm、tail/checkpoint/fresh恢复仍须完成；本项不做ffmpeg/媒体写盘/质量宣称。无GPU/服务器/部署/下载/环境升级/push。**MVP未完成，GPU_PENDING。**
