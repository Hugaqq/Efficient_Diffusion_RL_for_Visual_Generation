# Assignment B — W2

- active: true
- task_id: H3-M2a3b-decode
- pair: coding-b → simplifier-b
- execution: parallel W2；decisions/wave-02.md
- integration_parent: `47546f94e7490dafcc3fdbc2eb57279f0839899d`
- dependencies: 既有geometry/forward/loader及W1 text cache approved；源码汇合467c8fd929804aa291bf01a5fa7cda372cc507db。当前无changes。

## 唯一目标/文件

实现**真实本地双VAE逐条CPU解码**，输出音视频联合媒体供后续reward消费；不实现reward、不造服务/runner。

只写新 `visual_rl/models/implementations/h3_decode.py`、新 `tests/test_h3_mvp_cpu_decode.py`、W/coding-b/H3-M2a3b-decode.md。已批准geometry/joint/loader/cache/测试只读，不引用A/C本波新模块。

## 生产调用与数据合同

- 本地artifact目录→真实lazy公开Diffusers `AutoencoderKLMiniMaxH3` / `AutoencoderKLMiniMaxH3Audio`（从已读vendor对应真实模块路径核对），`from_pretrained(...,subfolder='vae'/'audio_vae',torch_dtype=float32,local_files_only=True)`。只接受本地存在目录，路径逃逸/缺config/无API清楚拒绝；无下载、remote-code、reference导入或tiny fallback。缺可选依赖分支明确错误，不能硬编码返回假media。
- 第一版固定**CPU**非prepared aux，无额外GPU；模型eval+requires_grad false，所有浮点参数/相关buffer保持FP32，不能把audio VAE BF16。每次只解一个sample，video VAE与audio VAE顺序加载/用完释放，不长期保存两套重模型或全collection媒体；上层未来负责rank-turn并发1。本项测试只替重量级类解析/模型叶子。
- `decode_one(terminal, geometry, ...)` 接受既有 H3T2VAGeometry 和CPU detached FP16 `[1,packed_width]`；layout.unpack拆两流。使用真实unpatchify顺序将video `[V,96]`还原`[1,24,T_lat,H_lat,W_lat]`，audio channel-major `[2*A_lat,32]`还原`[2,32,A_lat]`。不删音频/伪造batch/dummy frame。
- 读取VAE config中各24/32通道 `latents_mean/latents_std`，要求完整finite且std>0，FP32反归一化`latent*std+mean`；不能用scalar scaling_factor或缺配置默认0/1。调用真实decode(return_dict=False)。
- video输出必须`[1,3,num_frames,height,width]`、有限；逆ImageNet pixel norm(mean .485/.456/.406,std .229/.224/.225)，clamp[0,1]后转CPU FP32 `[T,3,H,W]`。帧数必须与124等明确geometry精确一致，不改121/删尾帧。
- audio decoder真实返回`[2,1,L]`，转CPU FP32 `[L,2]`；校验config sampling_rate=32000、模型hop_length=800，L=audio_latents*800，不能缺值默认一个采样率。保留真实幅度，不默默clip/删静音。audio rounding造成的时长与视频轻微不同要在媒体元数据实际记录，不暗trim/pad以假装严格相等。
- 返回最小AV媒体结果，含video/audio、fps24/sample_rate32000及实际frame/sample count。可用context-managed单条lease，在消费者退出后释放owner引用并拒重入/第二条在途，禁止返回永久媒体列表。没有完整pool的decode循环。
- 显式max_media_bytes及必要decode工作空间预算/单条在途上限；预估反归一化latent+输出/转换副本在load/decode之前检查，超过上限不能缩画布或改dtype。该估算不是VAE权重/内部workspace/总RSS保证，要清楚分列模型资源预算仍需实测。失败释放已载leaf引用且不产半套AV；不靠GC日志宣称GPU内存已回落。

## 来源/验收

先读UniRL LICENSE和 `minimax_h3/vae.py`、vendor `packing.py` 的unpatchify/unpack、两个autoencoder的真实config/decode API及文件头。只独立实现必要数学/调用；若需复制非平凡源码及license路径扩大先报告，不整vendor。

目标测试用真实geometry/packing/反归一化/输出转换与lazy调用控制，只重VAE叶子tiny：逐位置独立参考unpatchify，非均匀mean/std，stereo顺序、pixel反归一化，严格124帧/采样率/hop/actual lengths；leaf断言FP32 eval/no_grad/local_files_only。两leaf生命周期顺序及失败不返回半媒体、超限在load/decode前拒绝、第二lease拒绝、退出释放引用；非法dtype/shape/finite/config/输出长度明确拒绝。无恒定伪媒体作为生产实现；tiny输出依赖输入以证明路径真实。

保持原44项及断言，独立feature文件核验worktree导入。最后编辑后跑plan固定Flash七项和H3通配符，无skip/xfail/全量；报告branch/HEAD、命令/exit/数量/时间、diff/check及精确下游接口。引擎commit，不自行操作Git发布。

## 非目标

不改text cache、ModelAdapter/registry/controller、FSDP/world/pool，不执行ffmpeg/媒体写盘/真实reward。解码必须后续由RewardStage在完整rollout后调用，不能据本项称实际视频质量验证。Simplifier B只审本项数学/真实依赖/有界生命周期，changes回B重测。GPU_PENDING。
