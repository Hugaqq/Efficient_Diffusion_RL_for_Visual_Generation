# H3-M3b2-phase-norm — Coding C / W3

## 身份、范围与判定

- `implementation_ready: true`，仅当前协调原子项交 simplifier-c；不是 W3 汇合批准或完整 MVP。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-fe15678f-3a80-47ac-9511-ee893a0d0607`。
- 起始/结束 HEAD：`282539fa94c9ca2ec7fda8602e2cc2980ada6d62`，初始工作区干净；assignment integration_parent 为 e436c192778a6bbf24e2bee7ebef777332138f9a。未自行切分支、提交或 merge。
- 仅新增 `visual_rl/models/lifecycle/h3_coordination.py`、`tests/test_h3_mvp_cpu_coordination.py` 和本报告。
- 已读最新 task/plan/assignment C/wave-03、配对 W2 approved 反馈及 A 冻结结构接口；复读实际 h3_world/h3_fsdp/prepared 与既有测试。现 prepare/world/旧 tests、A/B 并行文件均未改，也未 import A/B 本波模块。独立实现数学/控制，未移植 vendor。

## 公共接口及实现

### `H3Coordination(plan, *, process_group, device)`

绑定真实 H3WorldPlan 的 plan_id/collection_id/behavior_id；复用已有 WORLD8/NCCL rank 前置检查和 CUDA 当前设备观测，要求显式 cuda:rank。不创建/销毁 PG，不创建 Accelerator，不访问 UUID。构造时所有 rank 参与序号0的 `coordination/init` gate，验证共同身份；可协作的 plan/device 输入错误进入该 gate 的错误 receipt。没有可用 PG/设备通信条件、死亡或 collective 内部故障仍需上层 attempt timeout/退出，不能承诺救活。

### `gate(event, *, local_error=False)`

复用 W2 有界 `_exchange`，不再实现一套传输框架。每次仅交换 JSON bytes，含 rank/node/schema 与共同 plan/collection/behavior、单调 sequence、event、status。event 为最长256 UTF-8 bytes 的 canonical 字符串，sequence 限于63位；由固定字段构造的正常 gate receipt 小于8KiB，底层仍有现成硬字节上限。无模型、轨迹或张量 payload。

非法本地 event/status 转成有限错误 receipt，同一次汇合后所有存活 rank 拒绝；event/sequence/identity 不一致也拒绝。成功才推进 sequence；失败永久 poison，随后训练方法立即失败且不再 collective。事件名由调用方使用 rank-independent local cell ordinal；各 rank 放入不同 sample ID 会被一致性门拒绝。该门不代替 A 的完整格数/语义阶段控制。

### `aux_turns(work) -> T`

固定 logical rank0..7，每轮相同 begin/end gate，仅 owner 调用一次自己的 work。其他 rank 等待共同 end，因此不会越过未完成的 heavy aux 进入下一模型阶段。只返回本 rank 的结果，类本身不保留 work/result，也不汇合媒体。调用者按冻结合同返回小 reward 表，并在 work 内完成逐条 lease 消费；work 不应进入模型/PG gates。

work 的普通异常转为该轮 end 的失败状态；owner 的最终异常保留原 work 异常链，其余 rank 共同失败；后续 turn 不运行、不重试或换样。中断/death/transport 故障不尝试 rescue，finally 只清本地标志，没有补充 barrier。

### `clip_grad_norm(prepared, *, max_norm) -> float`

- 必须是真实 H3PreparedModel 类型，divisor 与既有显式8设置一致。读取最终 trainable_named_parameters，与 prepared root 的完整 requires_grad 路径/对象一一对照，检查名字/对象唯一，并复用 optimizer 参数集合核验。None gradient 不视为零。
- 生产默认观察器严格要求 parameter **和 gradient** 都是真实 DTensor；只用 `.to_local().detach()`，无 full_tensor/all-gather。逐项检查 CUDA、一维、ordered WORLD8 mesh、同一调用者 PG、精确 Shard(0)；拒绝普通/复制/Partial/其他mesh。检查 FP32 local shape/device/布局及参数/梯度 global shape 一致，local 尺寸必须正好是该 rank 的非 padding shard（末 rank 可以为空）。
- 在 no_grad 下逐 local shard 计算 FP64 sum(g²)，不复制完整模型或保留计算图。覆盖/非有限/max_norm 等本地错误先走同一 preflight gate。成功 gate 的事件摘要还绑定参数名/global shape 和 max_norm，避免 rank 间阈值或覆盖不一致。
- 全部成功后，只有**一次 scalar FP64 WORLD SUM all_reduce**；sqrt 为唯一参数元素的 global norm。不再除8、不平均各 rank norm，也不再归约 gradient。aggregate NaN/Inf/负值拒绝并 poison，尚未修改 gradient。
- norm0 时 scale1，否则 `min(1,max_norm/norm)`，同一系数原地乘各 local gradient，无 epsilon；返回裁剪前 Python float norm。没有 optimizer/zero_grad/scheduler 调用。该梯度必须来自既有正常同步 FSDP backward，本能力不证明或补救模型内部未完成的归约。

`close()` 幂等，只清本地 PG/device 引用并关闭能力，不调用 collective 或关调用者 PG。

## CPU 验证与明确替身

新增独立八项。测试使用8个**同进程 CPU 线程**和 condition 实现内存中的阻塞 collective 叶子；没有 PG、多进程、socket、网络或 CUDA。只替 WORLD/CUDA 观测、all_gather/all_reduce 传输及 DTensor 的 mesh/placement/global-shape/local-view 观测；gate/轮转/prepared 参数检查/norm 数学与 clipping 均是真实现。没有生产 CPU fallback 或伪 DTensor。

实际验证：

1. 当前 worktree 导入身份、构造共同门、多个相同事件/序号正常前进；close 后拒绝且不 collective。
2. event/sequence/plan/behavior 错配、本地状态错误、非 bool status、Tensor event、超长 event 与设备参数错误统一拒绝并 poison；后续 model 门不产生新 collective。
3. 8 rank work 恰按0..7各一次、实际内存并发计数最大1、16个 begin/end 顺序准确；每 rank 获得自己的结果，receipt 不含结果/媒体。
4. owner3 work 抛错时仅0..3运行，保留其原异常对象为 cause，失败轮 end 后无后续 turn/barrier；非法 callable 共同早拒，transport 异常后无 rescue gate。
5. 真实 CPU backward 生成两组不同全局向量的八份不等长 local gradients（包含 empty），使用实际 H3PreparedModel/PreparedModelBundle/AdamW 参数所有权对象；仅 DTensor 观测替换。FP64 SUM-of-squares/global norm 与独立拼接完整向量参考一致，clip 后逐值对照通过，证明不是 mean rank norms；权重与 optimizer state 未改变。
6. 全零真实梯度返回0、保留非None/finite/zero梯度，仍执行 gate+一次SUM；无 epsilon 或 require-nonzero。
7. None/NaN/Inf、重复对象/缺参数、普通Tensor、Replicate/Partial、其他mesh/顺序/维度、错误shard轴/shape/padding、非法或不一致max_norm全部在SUM前统一拒绝。
8. 生产观察器真实拒普通Tensor；aggregate非有限/负值失败不clip，且没有补偿 collective。

当前线程模拟只证明 CPU 控制和数学，不是8进程/FSDP/CUDA/NCCL证据；state-only prepared 也不是权重加载或真实分片验证。

## 最后源码编辑后的门禁

原样执行，均 exit0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.31s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 73 passed in 3.60s（基线65 + C新增8；未包含同波A/B尚未汇合项）
```

无 skip/xfail 或全量测试。已 read 复读最终实现；git diff --stat 为空（本项均新文件，尚未暂存，不以此冒充新增内容审计）；status 仅指定源码/测试及本报告。git diff --check exit0。原Flash完整文件相对 d818b8a231b52a9a024fa2f522057acdbb7bf099 的获准历史 diff 无输出，当前也无该文件修改，保留原七项及断言。

## 待审与边界

交 simplifier-c 审单owner/同序错误传播、真实 DTensor local-view 检查、唯一global norm及无额外归约/cleanup；changes回C修复并重跑两集。引擎snapshot待生成，由orchestrator登记。

W3汇合后仍须核验A实际stage与本对象/真实B scorer的兼容并跑完整通配符；后续runtime提供真实PG/CUDA和prepared，继续controller/完整tail checkpoint/fresh恢复。当前无训练loop、资源launcher或checkpoint实现。无GPU、服务器、HTTP/网络、部署、下载、升级、push或手工Git提交/合并。**MVP未完成，GPU_PENDING。**
