# H3-M4c1-runtime-binding-r1 — Simplifier A

verdict=approved

**A-BIND-01 已修复并通过原配对复审，原 runtime-binding 唯一返工项解除。**可交 orchestrator 接收该 binding 原子项；不代替 B/C 本波审核或完整 MVP 验收。原 changes 报告保留。

## 审核对象

- HEAD：`6a7616aa26a63215cdae2b1cdd091dc33204f9e4`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-f735fe9e-c362-454d-85ad-098f411954c7`；开始工作区干净。
- 已读当前 task/plan/assignment A、原 changes 报告、r1交付、最新 binding 观察/scope/live/checkpoint 路径与新增/扩展测试。本轮只审核修复及其回归影响，未修改源码或测试。

## Issue 关闭依据

`visual_rl/runtime/h3_binding.py::_observe_scheduler` 由 `_observe_model` 统一调用，初始构造、受控 scope/forward、validate_live 和 checkpoint 的 live复核均消费：

1. 精确 LambdaLR 类型并要求 `scheduler.optimizer is prepared.optimizer`；首次观察记录实际 scheduler/optimizer 引用，之后任一实例被替换都拒绝。相同参数集合或相同state不再替代对象所有权。
2. 只读检查固定 lr/initial_lr/base_lrs/_last_lr、非负整数 last_epoch、`_step_count=last_epoch+1` 及codec支持的无状态LR callable表示；冷Adam要求epoch0，warm Adam要求完整最终参数集合和有限scalar step均与epoch相符。
3. 没有调用LR callable、scheduler.step或optimizer.step作验证，没有dummy update、额外scheduler框架或新owner。校验范围是现prepare产生的constant路径及其可观察状态，不冒称形式化证明任意Python callable未来行为。
4. 原checkpoint完整tail/源进度/梯度清空等要求保留；新观察不把source计数加入prepared-state恢复前置，从而不破坏两恢复门顺序。

`tests/test_h3_mvp_cpu_binding.py` 两项新测试实际覆盖初次外部optimizer scheduler、绑定后scheduler.optimizer引用漂移、同类型同state实例替换、base/current/initial lr、计数与Adam step漂移；live、scope、typed predict及post-tail checkpoint均拒绝。step失败spy和模型调用计数证明检查无更新且拒绝发生在模型前；恢复被篡改值后合法边界逐值不变。

原合法stage测试扩展至真实连续两轮：Adam/scheduler/data/trainer step2、cursor16、scheduler `_step_count=3`、固定lr1e-4一致。cold、active、poison、optimize结束但tail未完成、未清grad保护和旧断言保留。无新增必须返工项。

## 独立真实 CPU 门禁

当前汇合 worktree 原样运行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
```

- Flash exit0：**7 passed in 12.29s**。
- H3通配符 exit0：**146 passed, 22 warnings in 22.09s**，含本轮全部汇合feature，不以A分支137项代替。warnings来自明确单进程DCP save/load与损坏fixture覆盖提示，原样保留。
- 无skip/xfail匹配，未跑全量；`git diff --check` exit0。
- 原Flash完整文件相对授权 `d818b8a231b52a9a024fa2f522057acdbb7bf099` 历史diff无输出、exit0，七项及全部断言不变。

## 结论边界

本报告解除A-BIND-01并批准修复后的runtime-binding原子项，不批准兄弟session/恢复组合任务。默认唯一RunController/default entry装配与最终candidate仍由后续任务和orchestrator验收。无GPU/服务器/真实HTTP/PG创建/下载/环境升级/push；CPU不证明真实H3权重、CUDA/FSDP2/NCCL或质量。**GPU_PENDING，MVP未完成。**
