# H3-M4a1-anchor — Coding A / W1

## 判定与实际身份

- `implementation_ready: true`：仅指本原子项本分支实现和指定CPU门禁已完成，可交simplifier-a；不是审核批准、run汇合验收或main发布。
- 实际worktree：`/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.pi/workflows/worktrees/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-d89f7883-d6ab-414c-8b71-0ba4187b0aa0`。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-d89f7883-d6ab-414c-8b71-0ba4187b0aa0`。
- 开始HEAD/本分支基线：`44918392c5903cf54c9d8af389317814b60d3e04`，初始干净。assignment的integration parent为`b7f938506f16034ac5d2896ddfa44c16eb4af83d`；当前基线含W1协调更新。
- 引擎snapshot SHA：pending。没有切分支、暂存、手工commit/merge/push，也没有修改或读取其他worktree。
- 已读task、最新plan/assignment A/wave-01、前置M1b approved报告、当前CPS/joint/schedule与现objective。当前任务无changes反馈需返工。

## 唯一改动范围

1. 新 `visual_rl/algorithms/optimization/h3_anchor.py`。
2. 新 `tests/test_h3_mvp_cpu_anchor.py`。
3. 本报告。

没有改动任何既有源码/测试、B/C新模块、task/reference/AGENTS或工作流文件。不import B/C同波新模块。独立实现本地控制合同，无上游源码复制或新增依赖。

## 接口与调用合同

```python
owner = H3ReplayAnchor(
    collection_id=collection_id, behavior_id=behavior_id,
    sample_ids=expected_sample_ids, dynamics=cps_dynamics,
    max_cells=max_cells, max_trajectory_bytes=max_trajectory_bytes,
)
owner.capture(
    ordered_cells,  # tuple[(sample_id, H3CPSRecord), ...]
    collection_id=collection_id, behavior_id=behavior_id,
    device=device, predict=predict_record,
    read_behavior_id=read_policy_version,
)
old = owner.score(
    collection_id=collection_id, behavior_id=behavior_id,
    sample_id=sample_id, index=index,
)  # owned detached CPU FP32 [1]
```

- 构造时绑定完整expected sample IDs与既有冻结CPS dynamics。要求非空唯一ID tuple；不根据prompt分组。显式expected IDs使整条样本轨迹缺失也被检测，不只是缺一个transition。
- capture接受非空有序tuple，仅CPU B1记录；先检查collection/behavior身份、最大格数、轨迹payload bytes、完整格数、sample/index唯一且在域内、共同layout/schedule/eta及CPS/FP16标记，在任何predict调用前完成。完整覆盖包括最后q0格。
- bytes按每个合法B1记录 `2*packed_width*2 + 4` 计算：FP16 x/action与FP32 rollout diagnostic；是输入轨迹tensor payload界限，不是整个进程RSS界限，也不计Python对象/共享schedule metadata。先查长度和预算，再读取任何公开tensor副本。每个记录原有dtype/shape/finite与所有权由已批准H3CPSRecord构造合同保证。
- owner第一次调用capture就消耗一次性资格，包括preflight错误；无重试、刷新、re-anchor API。scores仅在全部格子成功且最终version匹配后原子发布，中途失败不能读出任何局部分数。
- 在no_grad内按输入固定顺序逐格：用record公开x/action/rollout_score复制到指定CPU/CUDA device，通过原H3CPSRecord重建临时格；一次predict，随后调用真实H3CPSDynamics.evaluate。没有读取下划线buffer，没有使用rollout_score充当anchor。每次只保一个临时设备record，退出该格前显式删除临时record/prediction/score。
- 捕获结果检查有限FP32[1]并冻结到CPU副本。owner只保ID/keys、共享不可变数值合同、上限与小scores，不保输入records、callback、模型或参数图。读取核对collection/behavior/sample/index，返回clone。
- capture前后通过必需`read_behavior_id`回调验证同一版本；版本变化拒绝整次结果。模型EVAL、参数不变、credit已完成是上层前置，不在本owner隐式切mode/复制旧权重/创建optimizer。
- 原CPS record不携带collection/behavior或sample来源元数据，因此batch显式identity和sample标签是未来pool提供的可信来源；本项验证标签一致性与完整集合，不声称证明原轨迹来源。版本必须由上层在每次policy mutation时更新，前后检查不能替代跨rank冻结门或防止未申报的权重修改。

## 新增六项CPU覆盖

- 当前worktree导入身份。
- 真实CPS、tiny trainable双流prediction，非顺序输入固定遍历、恰N*rows次no_grad callback、完整含q0；临时record weakref证明前一格在下一次调用前释放。篡改rollout diagnostic加37仍得到逐格真实evaluate参考值。捕获不消耗全局RNG，二次capture拒绝。
- 修改当前模型参数后old不变/current改变；直接把冻结old与真实current送进现`PolicyLossInputs`/`ClippedSurrogateObjective`（beta0），loss/backward有限，两模态参数都有非零梯度。没有另写PPO。
- 缺格、整个样本缺失、重复sample/index、未知sample、合同冲突、B2、非法结构与max_cells/max_bytes提前拒绝，predict零次；验证失败owner不能重试。
- 非有限prediction在第二格失败，不公开第一格的score；版本在capture前或capture中变化均拒绝，collection/behavior不匹配、未知sample/非法index读取拒绝。
- 输入原buffer、record公开读取和score读取的mutation不会改变固定结果；释放输入record与callback模型后weakref全释放，owner仍可读取完整old scores。

## 真实CPU门禁（最终源码状态）

在上面隔离worktree根目录执行，均exit0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

- H3通配符：**28 passed in 2.84s**（本分支既有22+本项6；不包含尚未汇合的B/C同波feature测试）。
- 原Flash七项：**7 passed in 12.83s**。
- 无skip/xfail，无全量测试，无原断言修改。
- `git diff`无输出，全部改动为指定新增文件；`git diff --check` exit0。该命令不覆盖未暂存新文件，已读回新实现并实际import/测试，不冒称已审引擎snapshot diff。
- 新授权 `git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` exit0、无输出；本基线原七项整个文件与授权历史起点一致，本轮也未改它。

## 交Simplifier A / 待办

请审核真实replay而非rollout alias、完整键域/输入上限、一次性失败原子性、只保CPU scores、固定读取及objective实际消费。CPU端未运行CUDA搬运、FSDP2/NCCL或全rank冻结门；本项不接controller/OptimizeStage，不实现credit/真实评分/pool/world计划/checkpoint。

等待对应simplifier-a裁决；changes回本coding修复并两集重测。引擎汇合后orchestrator仍需运行完整W1通配符目标集，不用本分支28项代替集成验收。本轮无GPU/服务器/网络部署/下载/环境升级/push。**GPU_PENDING，整体MVP未完成。**
