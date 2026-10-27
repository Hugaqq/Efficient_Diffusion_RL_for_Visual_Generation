# H3-M3b1-world-credit — Coding C / W2

## 身份与判定

- `implementation_ready: true`：仅本原子项交 simplifier-c 审核；不是 W2 汇合批准、完整 MVP 或 GPU 验证。
- 实际隔离分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-b7d33a5a-3bad-4618-a835-5d982eddc4cb`。
- 开始/结束 HEAD：`8e674e8894c2af61188f7523933cff585a6812d6`；初始工作区干净。当前 assignment 的 integration_parent 是 `47546f94e7490dafcc3fdbc2eb57279f0839899d`；未自行提交/切分支/merge。
- 已读本 worktree 最新 task/plan/assignment C/wave-02、既有 simplifier-c prepare approved 反馈、实际 prepare 源码/测试，以及 `GroupZScoreAdvantageProcessor`、reward identity/result 和 PPO objective 源码。

## 改动范围

1. 新增 `visual_rl/algorithms/optimization/h3_world.py`。
2. 新增独立 `tests/test_h3_mvp_cpu_world.py` 八项。
3. 仅为显式 reducer divisor，精确修改 `visual_rl/models/lifecycle/h3_fsdp.py`：API 准入、每 block setter、prepared 只读 property。
4. `tests/test_h3_mvp_cpu_prepare.py` 保留旧十项全部断言，添加一项真实 setter 叶子测试及 prepared property 断言。
5. 本报告。

未修改原七项、共享 H3 基础测试、其他配对文件或旧 single/Flash 路径；未 import A/B 本波 pool/decode 模块。

## 公开接口 / 后续阶段接线合同

### `admit_h3_world(...) -> H3WorldPlan`

必需参数：调用者已初始化的 `process_group`、`collection_id`、`behavior_id`、有序 `samples=(sample_id,group_id,member_index)` 表、`group_size`、`num_steps`、`geometry_id`、`schedule_id`、`score_profile`。

可选参数默认且只接受：`sp=tp=rollout_microbatch=replay_microbatch=chunk_size=1`、`guidance=1`、`beta=0`。`local_error=True` 允许调用者报告已发现的协作预检失败。

- 实际 WORLD/NCCL/world_size8/rank 边界，不创建/销毁 PG。receipt 带真实 rank 与 hostname SHA，单节点和 rank 顺序需一致。无 PG/错误 communicator 是不能参与健康汇合的调用前置错误；死亡或 collective 内部失败由上层 timeout/整 attempt 退出处理。
- 有效组为 prompt occurrence ID，不接收 prompt 文本作分组键。每个 rank 样本数严格同正数；每组成员 0..K-1 恰齐，K>=2；world sample 唯一，组不可跨 rank 拆分或复用；不以 total 能整除替代逐 rank 校验。
- collection/behavior/N/geometry/双 schedule/score 标识一致才返回；实际模型几何、schedule、score 内容与 ID 的绑定由后续 runtime 负责，不能用任意共同字符串替代真实指纹绑定。
- `H3WorldPlan` 是 frozen、深层 tuple 所有权的最小计划；`samples_by_rank` 保留各 rank 输入顺序，`samples` 是 rank-major 固定全局顺序；`plan_id` 绑定这些身份和顺序，`global_count = world_unique_samples * num_steps`，包含末步 q=0。

### 协作失败与通信上限

- admission / credit 各自只有一次 `all_gather_object`；发送的是已校验基本类型编码成的 ASCII JSON **bytes**，不发送 tensor、权重、latent、媒体或模型对象。
- 本地非法/空表/非有限 reward/显式 local_error 等在函数内捕获，变成同次汇合中的固定小错误 receipt；不在本地输入检查失败后让其他 rank 进入 model collective。所有存活 rank 收到相同表后统一拒绝，无隐藏 barrier/重试/补救 collective。
- 硬界限：world<=4096 样本、每 rank<=512；N<=4096；identity<=256 UTF-8 bytes；每 receipt<=256KiB，八 rank JSON payload 总计<=2MiB（不声称这是 Python/NCCL/RSS 总内存）。Unicode JSON 膨胀也先检查，超限转换错误 receipt。收到的 peer bytes 再检查大小/schema/rank/phase/node。

### `gather_h3_credit(plan, *, process_group, collection_id, behavior_id, rewards, local_error=False)`

`rewards` 每行固定 `(sample_id, group_id, member_index, reward:float, valid:bool)`。reward 是 scorer 已按固定权重合成的 Python finite float，不在此做评分/二次权重。必须 valid is True、每个本 rank 样本恰一次、三元组与 plan 完全一致；允许行重排，不允许换组/成员/rank、漏样、重复、非有限、无效行。汇合再次绑定 plan_id/collection/behavior/所属 rank，完整覆盖后才归一化。

返回 `dict[sample_id, detached CPU FP32 scalar Tensor]`，按 plan 顺序插入、每值拥有副本；调用者持有对应 plan，不保隐式全局状态。

数学是唯一窄实现：CPU float64 的组均值、**完整唯一 world collection** population std（ddof0），`(r-group_mean)/(global_std+1e-4)` 后转 FP32。无优势 clip、无无效行过滤/重归一化；统计溢出明确失败。同值奖励特殊分支给精确零信用（也覆盖 .1 及 1e308），不执行或跳过 optimizer。

已检查既有 processor：它能使用相同 batch-std 数学，但输入强制 RewardResult 的 row/trajectory/condition/resource 身份。当前 W2 冻结三元组没有这些完整来源，故不在生产伪造身份只为适配旧 processor；H3 小表数学仅这一处。测试使用完整显式合成 fixture 与真实 `GroupZScoreAdvantageProcessor(epsilon=1e-4,std_domain='batch')` 及独立手算交叉验证，未创建假 trajectory/media/condition 生产对象或第二通用 normalizer。

### `h3_slot_loss_scale(n_slot, global_count, reducer_divisor)` / `plan.loss_scale(...)`

系数严格 `d*n_slot/N_world_cells`，作用于当前 objective 的 **local cell mean**；无额外 GA/world/J 除法。数量必须正整数且不超过边界，divisor 必须有限正数；plan 方法额外约束 slot 格数不超过该 rank 的全部格数，global_count 直接取完整 plan。上层必须使用 objective 的真实格数、每个计划格只消费一次，不把此函数当阶段执行器。

正常 WORLD8、每 rank K8 样本、N10，global_count640，一格系数8/640=1/80；FSDP 再按显式 d8 归约后每全局格贡献1/640。通用数值函数的 d 是显式参数，并非内部硬猜8；H3 生产调用必须传 `prepared.reducer_divisor`。

## 真实 reducer divisor 证据

已安装 PyTorch 的旧公开 `FSDPModule.set_reduce_scatter_divide_factor` 在实际 CPU probe 中发出弃用提示，并实际转发 `self.set_gradient_divide_factor(factor)`。采用新公开 `set_gradient_divide_factor`，不依赖默认 AVG、不猜默认版本行为：

- `_require_apis` 检查新 setter 存在且含 factor 参数；不支持时在权重/包装/重 collective 前 fail closed，无静默 legacy/default fallback。
- `_shard_block` 在真实 fully_shard 及 FSDPModule/DTensor 后置检查之后，每个主 block 显式 `set_gradient_divide_factor(8.0)`；不改变 wrapping/AC/dtype/mesh，仍单维 WORLD sharding，无 HSDP 额外归约。
- 成功 prepare 返回的 `H3PreparedModel.reducer_divisor` 暴露该设置值，只读且要求 handle open。
- CPU 测试真实调用公开 setter，替换的仅是无 collective 的 state leaf，核验两个 param groups 的 `gradient_divide_factor` 均为8；再通过实际生产 `_shard_block` 和空 CPU state carrier 验证 setter 接线。未构造伪 DTensor，未把 carrier 称作真实分片模型。缺 setter 的能力拒绝也实测。

API 探查曾出现 fixture 失败（旧 setter 转发、实际 state 字段为 `_fsdp_param_groups`、FSDP 动态类构造剥离），均按实际输出修正为当前真实 API/state-only fixture；最终全门禁通过。未改断言掩盖归约缺口。此证据仅证明 API 调用/状态设置，**不证明真实 CUDA/NCCL 已按8归约**。

## 真实 CPU 测试

新增 world 八项覆盖：

- 本 worktree 导入、全八 rank 同一 immutable plan、固定顺序、发生组身份和上限。
- 只替 WORLD/collective 叶子：先捕获八个由实际函数生成的本地 receipt，再向各模拟 rank 重放同一汇合表；每次函数调用恰一次 collective。不用线程、PG、网络；不 mock admission/credit/数学。
- 本地坏输入、空/不完整/重复表、K/N/B1/SP/TP/guidance/beta错误、调用者错误及 Unicode byte 超限：所有模拟存活 rank 相同拒绝，下游 model callback 零次。
- uneven、world 重复、跨 rank 共享组、profile/collection/behavior/grid不一致、伪 rank/node/phase receipt 拒绝。
- 完整 world std 与每 rank std 不同；独立手算/既有 processor 交叉一致；重复 prompt 不并组，admission/reward 列顺序重排不改 sample credit，输出副本 CPU FP32 detached。
- reward 缺/重/换身份/非法rank归属/NaN/Inf/valid非真/plan和版本错配均拒；不丢样重归一化。
- 真实 `ClippedSurrogateObjective` / `PolicyLossInputs` beta0，八个真实 tiny Linear 副本逐格累计全部N（含末格），按显式 d8 模拟梯度平均，与单进程完整 global cell mean 的 loss/grad/真实 SGD update 在1e-12容差内一致；未复制 PPO 公式。
- .1和1e308常数奖励精确零 credit，仍逐格真实 backward、有限图连接零梯度并执行 step；128样本实例优势>5证明无旧clip。K8/N10系数及错误count/divisor边界通过。

最后一次源码编辑后的固定命令，均 exit0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 13.09s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 53 passed in 5.25s（本分支原44 + world8 + reducer1；不含W2兄弟新增）
```

无 skip/xfail，无全量测试。`git diff --check` exit0；已看实际 tracked diff，仅 prepare 源码20行差异与测试新增31行（49 insertions/2 deletions，两个删除是import/required tuple扩展，旧断言保留）。新增 world 两文件尚未暂存，不计入该 tracked stat，已 read 复读实现。

授权历史 `git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` exit0/无输出，当前该文件无修改；原七项完整文件保持不变。

## 待审 / 未完成

交 simplifier-c 重点审本地错误同次健康汇合、完整group/sample身份、world std及 d*n/N、真实 setter 能力锁和 CPU 证据边界。changes 回 C 修复重跑两集。引擎 snapshot 待生成，由 orchestrator 登记；W2汇合后仍需包含A/B新增的通配符复测与各自审查。

本项没有实际 RolloutStage/OptimizeStage 同序阶段接线、全局 grad norm/clip、真实 reward 服务、optimizer执行器、tail/checkpoint/fresh恢复或 controller。下一步须消费计划并保证实际模型事件顺序与完整格消费；本项表准入不承诺任意collective故障可恢复。没有服务器/GPU/PG初始化/多进程网络/下载/升级/push或手工Git提交。**MVP未完成，GPU_PENDING。**
