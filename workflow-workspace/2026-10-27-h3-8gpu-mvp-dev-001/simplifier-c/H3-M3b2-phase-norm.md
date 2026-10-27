# H3-M3b2-phase-norm — Simplifier C / W3

- verdict: **approved**
- 审核 HEAD：`5397c46662db30ef535afc0fb4b192df019c4d62`（W3 暂存代码汇合）。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-547d9e5f-26b8-473b-abd2-6c9ed12d3b30`。
- 仅批准 Coding C 当前协调原子项；不代替 A/B 配对审核或 W3 组合验收。初始工作区干净，本次未修改源码、测试或他人报告。

## 实际审查与裁决依据

读取当前 task.md、plan.md、assignments/c.md、decisions/wave-03.md、Coding C 本项交付；完整检查 `visual_rl/models/lifecycle/h3_coordination.py` 和 `tests/test_h3_mvp_cpu_coordination.py`，核对实际 h3_world/h3_fsdp/prepared 依赖。读取汇合的 h3_stages.py 核对冻结调用接口，但不批准 A 的实现。

1. **共同健康门**：复用既有有界 receipt/WORLD 检查，绑定 plan/collection/behavior、event 和单调序号；非法本地 event/status 参加同次错误汇合。收到不一致或失败后永久 poison，后续调用不再 collective；没有 finally barrier 或 PG 创建/销毁。模型内部与 transport 故障不承诺救活。
2. **aux 单 owner**：固定 rank0..7 begin/end 门，只有 owner 执行本地 work，其他 rank 等待；不传结果/媒体。普通 work 异常以同轮 end 的失败状态传播，owner 保留原异常 cause，失败后不进入后续 turn。finally 只清本地标志。close 仅本地释放。
3. **梯度覆盖**：读取真实 H3PreparedModel 最终 trainable 参数，对照完整 requires_grad 路径/对象并复用 optimizer 参数集合检查；无重复/遗漏，None 不当零。生产观察器明确要求参数与梯度均为真实 DTensor，ordered CUDA WORLD8 单维 mesh、同一 PG、Shard(0)、FP32 且 local shape 恰为该 rank 的非 padding 元素；拒复制、Partial、其他 mesh 和普通 Tensor。没有 full_tensor 或全参数 gather。
4. **唯一 norm 数学**：本地 no_grad FP64 sum(g²)，有限性/覆盖/阈值问题先共同 gate；成功事件摘要绑定参数名/global shape/max_norm。随后一次 scalar WORLD SUM，sqrt 为唯一 global norm，不平均各 rank norm、不再除8，也不二次归约 gradient。aggregate 非有限/负值在修改梯度前拒绝；scale 为 norm0 时1，否则 min(1,max_norm/norm)，同一系数原地 clip，无 epsilon。模块不 step/zero_grad/scheduler。
5. **测试证据**：八个 CPU 线程仅模拟设备/WORLD/传输和 DTensor local 观测；真实 gate、aux、prepared 参数身份校验、norm 与 clip 执行。验证错序/错身份/错误状态共同拒绝和 poison、aux 顺序/最大并发1/异常链、不同且不等长/empty local 梯度与独立完整向量参考、零梯度合法、缺失/NaN/Inf/mesh/placement/shape/参数覆盖/阈值错误及 aggregate 错误不 clip。真实 CPU backward 产生 local gradients，非恒定 norm 输出或 mock 整个协调器。
6. **最小范围**：沿用已有传输和 prepared guard/参数校验，没有新 runner、通用 actor、第二套 PG owner 或 CPU 生产 fallback；未依赖同波 A/B 新模块。代码中的三个公共方法签名与 A 汇合源码调用一致。

未发现本原子项必须返工的问题。返工清单：无。

## 独立真实门禁

在本审核 worktree 原样运行，均 exit0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.48s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 93 passed in 5.25s（原65 + W3 A14/B6/C8）
```

H3 目标文件 grep 无 skip/xfail，未跑全量。`git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` exit0/无输出，原七项完整文件含断言不变。审核时 git diff 无输出，git diff --check exit0；此工作区检查不冒充单个 coding snapshot 的历史范围 diff。

## 必需后续与证据边界

- Orchestrator 登记本项 coding snapshot/approved，并结合 A/B 各自裁决亲跑汇合门禁。
- **93项不证明 A+真实B/C 组合执行**：当前 stages 测试仍注入 `_CoordinationLeaf`（其 norm 使用本地 clip_grad_norm_），而 C 独立测试才执行实际 H3Coordination。静态签名一致不替代 W3 决策要求的真实对象组合测试；应由 orchestrator 分派集成验证，再接唯一 RunController/runtime。
- 后续仍须完整 tail safe point、checkpoint 与同拓扑 fresh 恢复/下一步对照。正常 FSDP backward 已完成归约是本 norm 的前置条件，本项不能替代该 GPU 验证。
- 本次未执行 GPU、真实 PG、多进程网络、服务器、HTTP、部署、下载、环境升级或 push。真实 CUDA DTensor/FSDP2/NCCL、权重和音视频质量仍未验证。**MVP 未完成，GPU_PENDING。**
