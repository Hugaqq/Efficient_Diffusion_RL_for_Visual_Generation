# H3-M1a-joint-schedule — Coding A 交付

## 身份与范围

- active=true，按 assignments/a.md，仅本原子项；本轮无已有 simplifier-a 反馈目录。
- 实际工作区：`/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.pi/workflows/worktrees/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`。
- 开始 HEAD：`f8e4d0e7f126cbd654ebe43c157902d506a0d37b`；source dependency `076635a61263ffd5e7c5be4cc64d77ef99242c51`。初始 status 干净，未切分支。
- 引擎交付 snapshot SHA：pending（本角色不提交、不暂存、不 merge/push）。

## 实际新增文件与合同

1. `visual_rl/models/h3_joint.py`：冻结的 `H3JointLayout(video_rows, video_channels, audio_rows, audio_channels)`，全为正整数，拒绝 bool/小数/空模态。`pack(video, audio)` 将 `[B,V,Cv]` 与 `[B,A,Ca]` 变为 video-first `[B,V*Cv+A*Ca]`；检查形状、共同 B、相同 dtype/device，无 cast/detach。`unpack(packed)` 恢复两流并保留梯度；返回 view 故意与 packed 共享存储，pack 本身不与输入共享存储。公开 video_numel/audio_numel/packed_width 供下一项元素计数使用。不是 attention token 排列，也不推断画布合法性。
2. `visual_rl/algorithms/dynamics/h3_schedule.py`：`H3ScheduleSnapshot(video_sigmas, audio_sigmas)` 接受精确 CPU FP32 一维 tensor，不静默 cast/flatten/transfer；验证同长、至少两点、有限、范围、严格递减及起1终0。内部存 immutable tuple，FP32→Python float→FP32 精确往返；读取产生新 CPU FP32 tensor，输入/输出 mutation 不会污染快照。`build(num_steps, video_shift=12.0, audio_shift=3.0)` 使用 CPU FP32 linspace N+1 与规定 shift 公式，拒绝 FP32 碰撞而非 deduplicate。`num_steps` 与 `transition(index)` 返回显式四个 FP32 scalar `(video_s,video_u,audio_s,audio_u)`，无 cursor，不把 sigma/index/model time 混用。
3. `tests/test_h3_mvp_cpu.py`：六个 CPU 测试，实际调用新实现；导入路径归属当前 worktree；不等 AV 几何、B2、非 contiguous、逐元素顺序、FP16/BF16/FP32/FP64 往返、双流梯度、alias 语义；非法几何/shape/width/空音频、dtype 与 meta-device mismatch；默认10步11点/独立 shift/手算中点/末步/单步；精确显式快照与 mutation；非有限/越界/不递减/错误端点/长度/dtype/count/shift/index 及碰撞拒绝。meta tensor 仅作不同 device 拒绝负例，无 GPU 分配。
4. 本交付记录。

没有修改任何已有跟踪源码、旧配置、原七项测试及断言。没有 CPS、adapter、registry、runtime、模型加载、FSDP2、评分或 checkpoint 实现。

## 来源与许可

已读 task.md、plan.md、assignment、当前 AGENTS/README/pyproject 与实际 models/interface.py、dynamics/{interface,flow_schedule}.py 相关合同、原七项测试代码。已读 UniRL LICENSE（Tencent Apache-2.0）、固定提交 `8cfbac17765004afb68fef08a4e011a4c8daf555` 的 diffusion.py dual packing、packing.py 及 vendor/scheduling_minimax_h3.py（文件头含 MiniMax/HuggingFace Apache-2.0）。按给定数学/形状合同独立实现，并在两个新模块 docstring 标明语义来源；未复制 vendor 实现、未 import reference，无新增外部依赖。

## 真实门禁

在上述 worktree 根目录，最后一次源码编辑（更正来源 SHA 文本）后，原样执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
```

- 原七项：exit 0，`7 passed in 11.98s`。
- H3 单独目标集：exit 0，`6 passed in 0.65s`。
- 更正来源 SHA 文本前也实跑两集：H3 6 passed in 0.87s，Flash 7 passed in 12.31s；最终以上述重测为准。
- 无 skip/xfail，无全量测试。
- `git diff` 无输出，既有跟踪文件未改；`git diff --check` exit 0。新增文件尚未暂存，因此该命令不覆盖 untracked 内容；新增内容由本次写入和真实 import/test 验证，待引擎 snapshot 后可审新增 diff。

## 交给 Simplifier A

请检查最小形状合同与 immutable ownership 是否足够下一项 CPS/adapter 使用、generated AV 元素数与存储顺序、N+1 独立 sigma/精确恢复/末步、返回值 alias 约定，以及无旧 Gaussian/Flash 改动。CPU FP32 tensor-only 显式构造是刻意的窄接口，不接受自动 round/cast 的列表或 FP64；后续消费者应提供精确 grid。

本项暂时仅由目标测试消费，生产接线留给已规划后续任务，不能宣称 MVP 完成。真实 H3 权重、视频/音频、GPU/FSDP2/NCCL、8 UUID/窗口与资源准入均未验证。无服务器、网络部署、模型下载、环境升级或 push。**GPU_PENDING**。
