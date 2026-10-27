# H3-M1b-cps — Coding A 交付

## 当前身份 / 范围

- 本轮 assignment active=true，仅实现 joint CPS sample/evaluate/record。M1a 已 approved，无待修 changes；原交付及审查记录未覆盖。
- 实际工作区：`/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.pi/workflows/worktrees/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`。
- 实际 branch：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`。
- 开始 HEAD：`42a11a8c3fc6473aa52238473e20bc002e66db4c`；初始工作区干净。前置功能 snapshot `7b29f4bf8d990794d4b9bc826a82bd390861d530` 已由配对批准。
- 本项引擎 snapshot SHA：pending；未切分支、暂存、commit、merge 或 push。

## 实际改动

1. 新 `visual_rl/algorithms/dynamics/h3_cps.py`：独立于旧 Gaussian ABI 的窄 CPS 内核，实际消费已批准 joint/schedule 合同。
2. 修改 `tests/test_h3_mvp_cpu.py`：保留原六函数及全部断言，新增模块导入身份断言、两个 tiny fixture helper 和六项 CPS 测试。目标集现在共12项。
3. 本交付记录。

未修改 h3_joint.py/h3_schedule.py、原七项 Flash 测试、旧 Gaussian 源码、配置或 runtime。`git diff` 实际显示唯一既有跟踪文件变动为 H3 测试：214 insertions、1 deletion；一处删除仅为扩展模块 import。旧六项断言无删除或改写。新内核未暂存，不在该 diff/stat 内。

## 锁定供下游使用的接口

```python
dynamics = H3CPSDynamics(layout, schedule, eta=0.7)
stats = dynamics.stats(x, prediction, index)
record = dynamics.sample(x, prediction, index,
                         video_generator=video_rng, audio_generator=audio_rng)
current_score = dynamics.evaluate(record, current_prediction)  # FP32 [B]
next_state = record.action  # detached FP16 owned read copy
```

- `H3CPSDynamics` 冻结绑定 `layout: H3JointLayout`、`schedule: H3ScheduleSnapshot`、有限 `0 < eta <= 1`。不继承旧 Dynamics，不翻转 prediction 符号（必须已是 noise-ward）。x/action 必须 detached FP16、有限、正确 `[B,packed_width]`，仅 CPU/CUDA；prediction 可 FP16/BF16/FP32/FP64，shape/device 与 x 精确一致，无广播。FP64 转 FP32 后也检查有限性。
- `stats` 是采样与 replay 共用的唯一数学实现。对独立双 `(s,u)` 计算 FP32 q、x0、e_hat、mu，无 epsilon/clamp；非末步 q 下溢到0拒绝。返回 `H3CPSStats(mean, video_q, audio_q, video_dt, audio_dt)`：packed mean 为 FP32，q/dt 为 FP32 scalar，mean 保留 prediction 梯度。
- sample 在 no_grad 内使用显式、同 device 的两个 torch.Generator，先 video 后 audio 各采样一次 FP32 noise。q=0 末步也消费两份 draw；全局 RNG 不参与。支持调用者显式共享同一 generator，但不会默认共享/选择全局 RNG。FP16 action 量化后检查 overflow；score 为 joint negative MSE，并检查 finite。
- `H3CPSRecord(dynamics, index, x, action, rollout_score)` 拥有 detached clone；公开 `x/action/rollout_score` 返回副本。record 内的 dynamics 只是冻结数值合同，无模型、generator 或计算图。`score_family='cps_proxy_negative_mse'`，`quantization='fp16'`，完整 layout/schedule/eta 位于 `record.dynamics`，index 显式保存。调用方不得改写内部下划线缓冲。
- evaluate 比较 record.dynamics 的 layout/schedule/eta 精确合同后，只重算 mean 和 score，不抽 noise、不再生成 action；固定 action/x 不留图，仅当前 prediction 参与梯度。
- joint mean 对所有 generated 元素取平均，数学上正是 `(Dv*score_video+Da*score_audio)/(Dv+Da)`，不是模态等权、Gaussian density 或 exact KL。`rollout_score` 只作采样诊断，**不是更新前冻结 replay old anchor**。本项未实现旧策略锚。

## 真实 CPU 证据

最后一次源码/测试编辑后，在上述 worktree 根目录原样运行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

- H3：exit0，**12 passed in 1.13s**。
- Flash：exit0，**7 passed in 12.12s**。
- `git diff --check` exit0。此命令覆盖跟踪测试的 diff，不冒称覆盖尚未暂存的新文件；新内核已读回并实际 import/测试。
- 未跑全量，无 skip/xfail。未变更原七项文件及断言。

新增六项实质覆盖：

1. 非零 B2 输入、Dv2/Da3、手给独立 sigma 的 FP32 mean/q/dt 与 action；joint 元素权重而非0.5/0.5，CPS 与正 q Gaussian 数值不同；各支持 prediction dtype。
2. 显式 generator 重现、单独更换模态 seed 只影响对应模态、全局 RNG 不变；所有 index 可评分；逐步 generator state 对照两个真实 randn（含末步），共享显式 stream 额外验证 video-before-audio 顺序。
3. 采样分数与初始 evaluate 精确一致，record 构造/读取 mutation 不污染；当前预测改变 score，两模态解析梯度非零，原采样 prediction 无梯度；evaluate 不消耗 RNG。
4. 双 q0 末步 finite score、FP16 residual 非零、解析梯度 `-2*residual*s/D` 与实际 backward 一致；没有用全零例掩盖。
5. 直接真实调用旧 `TransitionMeanStd.validate_against`：零 std 被 strictly-positive 校验拒绝，正 std 通过；旧 Gaussian 代码未修改。
6. eta/index/shape/dtype/device、非有限 state/prediction/action、record 合同冲突、非法 generator、非末步 q 下溢、FP16 overflow、FP64→FP32 overflow、score 平方 overflow 负例均明确拒绝。

## 来源与许可

本轮重新读取 task/plan/assignment、M1a approved 审查、当前两合同/目标测试与旧 Gaussian 校验，重读 UniRL LICENSE（Tencent Apache-2.0）、固定提交 `8cfbac17765004afb68fef08a4e011a4c8daf555` 的 kernels.py（CPSSDEStrategy、FP32 denoise 与量化 finalize）及 H3 diffusion.py 双流代码。按 assignment 指定数学独立实现，模块 docstring 标明语义来源，未复制通用 registry/框架，未 import reference，未加依赖。

## 审查重点 / 未完成边界

请 simplifier-a 检查单一数学实现、固定 action detach/ownership、双 schedule 与 generated 元素权重、q0/RNG/解析梯度、精度和 finite/overflow 拒绝、旧 Gaussian 不被放松，以及 record.dynamics 仅含冻结数值合同而不是模型 owner。

本项数值实现只由 CPU 目标测试消费；生产 rollout/replay/adapter、旧策略锚、FSDP2、评分、world 数学、完整 tail/fresh checkpoint 尚未由本项实现。真实 CUDA generator compatibility 分支、H3 权重/音视频质量、8卡 NCCL/FSDP2 均未执行；meta 仅用于 CPU device 拒绝负例。本轮无服务器/GPU/网络部署/下载/环境升级/push。**GPU_PENDING，MVP 未完成。**
