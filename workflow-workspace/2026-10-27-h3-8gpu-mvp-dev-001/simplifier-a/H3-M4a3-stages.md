# H3-M4a3-stages — Simplifier A / W3

verdict=approved

仅批准 A 的现六阶段接线原子项；返工清单：无。不代替 B/C 配对审核、W3 三对象集成验收、RunController 装配或完整 MVP 验收。

## 审核身份与材料

- 当前 HEAD：`5397c46662db30ef535afc0fb4b192df019c4d62`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-02e82c02-70c2-4da0-89bd-345cab93ca57`；审核开始 status 干净。
- 已读当前 task/plan/assignment A/wave-03、Coding A 交付、完整 h3_stages.py 及14个 CPU cases、原 BaseTrainer/GRPOTrainer、pool/anchor 新 hook、真实 world 数学及 prepared/score/coordinator 相关接口。读取均来自当前 worktree。
- 未改源码、测试或他人报告。

## 实现及边界核对

1. `visual_rl/algorithms/trainer/h3_stages.py::trainer/build` 实际返回现 GRPOTrainer，复用原 BaseTrainer 的六阶段调用、identity 检查、abort/commit 和 next_optimizer_step。没有另写 run_iteration 或新训练 runner。Prelude reserve/输入验证失败转换为 local_error 进入真实 world admission；schedule ID 从实际 layout/grid/eta/CPS 合同派生，coordinator 在共同 plan 之后构造。
2. `rollout/reward/advantage/credit`：prepared EVAL 后走真实 pool，全 N 步采样完成才通过 aux_turns 逐条 decode lease/score。只保小 reward 表，真实 gather_h3_credit 后独立 Credit stage 广播本 rank 信用到 N 格；没有以假 StageValue 填充空阶段或常量生产 reward。
3. `optimize`：先一次真实 H3ReplayAnchor.capture，再 anchor/complete gate 后 current replay；明确有序闭包把 sample 与 index 对应到条件。每格现 CPS.evaluate→既有 ClippedSurrogateObjective，clip .005/beta0，乘一次 `plan.loss_scale(n_slot=1,reducer_divisor=8)`，立即 backward 和删除该格图。不重写 PPO，不 no_sync，不再除 world/GA/N，q0 末格保留。全部格后一次 global norm 接口、AdamW、可选 scheduler、zero_grad。
4. `h3_pool.py::collect` / `h3_anchor.py::capture` 可选 hook 保留既有默认调用；rank-independent ordinal 事件在 forward 前及 local CPS result 后发送。数值/finite 失败报告 local_error；raw forward/AC/backward 不包在恢复 collective 的 finally 中。`_checked` 调用位置限于局部工作，未包住 world gather 或 raw backward。
5. `commit_iteration/abort_iteration/close`：原 BaseTrainer 在 optimize 返回后才调用 data tail；tail 完成及新 behavior 验证/共同 gate 后才 pool.release 和解除 poison，BaseTrainer 单独推进 next step。部分 Adam 或部分 data commit 不假回滚、不允许再次更新。失败池在 close 释放，不冒充成功 safe point；本项未提供 checkpoint。
6. 上下文/请求/结果仅承载此次阶段所需字段；未添加通用事务、服务、旧权重副本或 Flash 专属 EMA/cadence。scorer/coordinator 是必需能力，没有生产 no-op 默认值。预检与 pool 的部分重复是 admission 前协作拒绝所需，不能直接删除后让错误落入不同 rank 模型序列。

## 测试证据与对象接口

`tests/test_h3_mvp_cpu_stages.py` 实际执行原 BaseTrainer、guarded prepared bundle、pool/CPS/anchor/world 算法、双 VAE 解码控制、既有 objective 和 AdamW/LambdaLR。两次更新覆盖精确三轮全格 forward、no_grad/grad、每格 gate/backward、媒体/图 weakref 释放、双流非零梯度与参数变化、data/behavior/next-step 恰一次推进和 tail 后 release。零信用仍真实 backward/Adam/tail。12类故障覆盖预检、identity/condition、score/rollout/anchor/current、版本漂移、raw forward/backward、部分 step/tail；无无效更新或就地重试。

已静态核对汇合后的 `H3HPSReward.score_one(*,sample_id,prompt,media)` 与 `H3Coordination.gate/aux_turns/clip_grad_norm` 签名、rank-independent 有界事件兼容 A 调用；HPS 所需媒体来自已有 decoder lease。**A 的阶段测试仍使用显式 coordinator 和 media-dependent scorer 叶子，未实际把 B/C 两个对象一起装入阶段链。**按 wave-03，该三对象集成测试仍由 orchestrator 接收时安排，不能把本次93项通过等同于该组合已执行。

## 当前汇合树独立门禁

在当前 worktree 原样执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
```

- Flash：exit0，**7 passed in 13.27s**。
- H3 通配符：exit0，**93 passed in 4.26s**（65+A14+B6+C8）。不是各分支79/71/73的历史局部结果替代。
- H3 目标文件无 skip/xfail 匹配，未跑全量。
- `git diff --check` exit0。
- 原 Flash 完整文件相对 `d818b8a231b52a9a024fa2f522057acdbb7bf099` 的获准历史 diff 无输出、exit0，七项及断言不变。

## 仍未完成

- 当前源码搜索只有 H3IterationStages 定义，尚无生产 RunController/recipe/ModelAdapter 装配；真实 data port、geometry/condition/artifact 身份绑定仍是下一步必需项。
- 完整 checkpoint safe-point 与同拓扑 fresh optimizer/RNG/data 恢复及下一步对照尚未实现，不能以本地 tail callback 测试替代。
- h3_hpsv3_video_only_v1 是显式 video-only reward，不证明音频语义/AV对齐质量；联合 AV CPS 未删音频。
- 无真实 HTTP、CUDA/FSDP2/NCCL、权重加载或八卡准入验证；无服务器/GPU/下载/升级/push。本次 CPU 证据不代表 GPU 或质量通过。**GPU_PENDING，MVP 未完成。**
