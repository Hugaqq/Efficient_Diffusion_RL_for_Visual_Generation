# H3-M2a3b-decode — Simplifier B / W2

verdict=approved

仅批准本配对双VAE解码原子项。outcome done表示review结束，不批准A/C任务或完整MVP。

## 身份与范围

- 当前HEAD：`ced9a871df0ee6bfe8b560e90da03bc12d4dda0e`。
- 当前分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-8a86fa45-e9cc-467c-aa62-f6a5cc01c567`；接收工作区干净。
- 已读取当前task、plan、assignment B、wave-02、Coding B交付、完整`h3_decode.py`和六项feature测试；对照UniRL LICENSE全文、vae.py、packing逆变换、两VAE文件头/真实decode与config合同、VENDOR_COMMIT。
- 只审本任务及其回归；未修改源码、测试或其他角色报告。

## 核验结果

1. `visual_rl/models/implementations/h3_decode.py:32,170–221`：存在默认公开Diffusers双VAE解析和`from_pretrained`路径，使用本地目录、明确subfolder、CPU上下文、FP32及`local_files_only=True`。config/class/逐通道stats/音频采样率、symlink与shard路径准入明确；缺依赖无reference或tiny fallback。模型eval/frozen，参数与浮点buffer必须CPU FP32。
2. 同文件`:233–265`：video的(time,patch_y,patch_x)/(channel,dy,dx)逆patch顺序与参考一致；audio恢复channel-major stereo为[2,32,A]。两者均显式逐通道FP32反归一化，拒缺失/非法std和overflow，不用scalar scaling_factor替代。
3. 同文件`:223–265`：真实`decode(return_dict=False)`后严格验证shape/device/FP32/finite；video逆ImageNet norm并clamp到[0,1]，输出[T,3,H,W]；audio输出[L,2]且不再clip，严格采样率32000/hop800/长度。124帧与165600音频采样的实际duration分别记录，不trim/pad伪造相等。参考audio VAE内部本身有clamp，本项“不额外clip”不误称绕过该模型行为。
4. 同文件`:112–167,267–288`：单条lease锁、重入/第二条拒绝，预算与terminal校验在模型加载前；video用完丢引用后才加载audio，没有长期保留两套VAE或全collection媒体。失败不公布半套AV、锁释放；退出清空媒体owner引用。外部消费者保存的tensor别名不可撤销，报告没有作强制RSS回收承诺。
5. `tests/test_h3_mvp_cpu_decode.py:109–291`：六项使用真实geometry、packing、stats、转换与lease控制；只替重VAE叶子。独立逐位置unpatchify、非均匀stats、stereo、124帧/真实音频长度与幅度、FP32/eval/no_grad/local_files_only、预算和负例均有断言；weakref验证成功/失败叶子及lease媒体owner引用释放。没有mock整个decoder。

## 简化结论

无必须返工项，返工清单为空。decoder、最小AV结果/元数据和lease是本项所需的窄边界，没有新runner、服务、通用媒体平台或同波模块依赖。每条顺序加载较慢但属于明确最小方案，不要求本项添加缓存/并发优化。

预算仅约束单条media和明确tensor工作集，排除VAE权重、loader副本及内部activation/workspace；不是总RSS或实际host准入保证。后续不能把该预算直接当作整机安全余量。跨rank aux并发1由未来phase/RewardStage实现，不是此对象局部锁已经证明的能力。

源码为公开API与必要数学独立实现，无vendor/reference导入。已核对VENDOR_COMMIT明确记录原模块路径及`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`；这仅是历史接口来源，不证明本机安装版本或真实权重兼容。

## 本轮独立CPU证据

实际运行最新版两门禁，均exit0，无skip/xfail，未跑全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 65 passed in 3.83s（W1 44 + W2 pool6/decode6/world及reducer9）

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.59s
```

`git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py`无输出、exit0，原七项完整文件/断言与授权基线一致。写报告前`git diff`无输出、`git diff --check` exit0；当前clean不代替逐coding提交归属审计。

## 后续及未验证

- Orchestrator记录本组snapshot/approved，结合A/C各自裁决并亲跑汇合门禁；65项通过不代替兄弟源码审核。
- 当前生产搜索仅有decoder定义，尚无RewardStage消费者。后续须在完整rollout后逐条lease评分、跨rank aux轮转，并接真实reward/ModelAdapter/controller。
- world阶段健康门/global norm、完整tail/checkpoint/fresh恢复仍需完成；模型权重加载及内部资源峰值仍待真实环境准入。
- 本轮未操作GPU/服务器/网络部署/下载/环境升级/push；未运行真实VAE权重、ffmpeg或质量验证。

**MVP未完成，GPU_PENDING。**
