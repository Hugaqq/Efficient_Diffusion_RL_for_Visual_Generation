# H3-M4c1-runtime-binding — Simplifier A / W6

verdict=changes

退回 Coding A 修复 **A-BIND-01** 后重跑两集并复审。本轮门禁通过不覆盖该所有权缺口；不批准本项，也不替代 B/C 裁决。

## 审核身份 / 范围

- 当前 HEAD：`c8422b3a780d933564e94dd58c52a5c8efd8f9dc`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-e20a6aae-21d9-4d20-b976-956a6f2023df`；开始 status 干净。
- 已读 task/plan/assignment A、交付、完整 h3_binding.py 与四项目标测试；核对基类数值scope、ParameterStateManager/projection、实际prepare/scheduler创建、prepared handle、stage只读边界及codec scheduler约束。
- 只审不改：未修改源码、测试或其他角色报告。

## 必须返工 A-BIND-01：scheduler 未绑定到已验证的 optimizer

### 位置与事实

- `visual_rl/runtime/h3_binding.py::_observe_model`，约221–222行，仅检查 scheduler 的精确类型是 LambdaLR 及 reducer divisor；没有检查 `scheduler.optimizer is self.prepared.optimizer`。
- `_contract` 约385行据此声明 `lr_scheduler_schema="h3.constant.v1"`。
- `checkpoint_boundary` 约402–408行分别检查 prepared optimizer 的 Adam step 与 scheduler.last_epoch，但没有核验二者属于同一个优化器链。
- 现 `PreparedComponentHandle.__init__` 只存储传入 scheduler，并不提供该所有权保证。
- 相邻实际消费者 `artifacts/checkpoint/h3_state.py::_inventory` 约162–163行明确要求 `scheduler.optimizer is optimizer`，说明当前binding准入与codec准入不一致。

### 可复现输入 / 影响（基于源码路径，未冒称已执行新增复现）

在现binding fixture里保留真实prepared optimizer及全部模型/文件事实，仅把其LambdaLR的公开 `optimizer` 引用指向另一个合法AdamW，或传入一个同类型、绑定别的optimizer的scheduler。当前`_observe_model`不读取该引用，因而仍可生成同一合同并通过`validate_live`。阶段随后对prepared optimizer执行Adam step，却由scheduler更新另一个optimizer；只要scheduler的last_epoch正常推进，当前tail边界也不能识别这个错配。后续codec才会拒绝它。

这违反本项“实际最终对象严格彼此绑定/真实checkpoint合同”边界，不能等到checkpoint写入才发现，也不能把LambdaLR类型当成正确scheduler所有权的证据。已有测试只覆盖optimizer mode/hyperparameter、参数与buffer漂移，没有该负例。

### 最小修复

在本角色允许的 `h3_binding.py` 中统一验证scheduler与prepared optimizer的对象关系，在初次绑定、forward scope/live复核和checkpoint边界均复用。至少要求 `scheduler.optimizer is prepared.optimizer`；绑定后不能通过替换scheduler为另一个同类型实例静默换掉已观察的链。保持现prepare、算法与codec接口不变，不新增通用scheduler注册或新owner。

同时核对已声明constant路径的现有可观察状态（实际optimizer lr与scheduler base/current lr及提交计数），不得仅凭class/last_epoch形成成功合同；不要为了验证而调用scheduler.step或optimizer.step改变训练状态。若确需扩大到其他owner文件，应先报告具体最小接口请求。

### 验证要求

新增精确负例：
1. 初始绑定时scheduler指向外部optimizer，必须在模型调用/更新前拒绝。
2. 合法绑定后改变scheduler.optimizer或替换scheduler实例，`validate_live`及受控forward必须拒绝；不得返回checkpoint边界成功凭据。
3. 合法常数scheduler仍走真实一轮/后续轮，Adam/scheduler/data/trainer计数及lr一致；保留cold、active、optimize后tail前、poison和未清grad断言。
4. 无dummy step、无删除旧断言；最后修改后原Flash七项及完整H3通配符重跑，再交同simplifier-a。

## 已核对的有效部分

- 完整model/VAE与code内容核验复用既有filesystem artifact hash，实际比较profile/cache声明，不以header facts或verified布尔值放行。
- 最终模型FQN/shape/storage dtype、精确LoRA集合、persistent/nonpersistent buffer与rope、参数对象漂移有实际观察；FSDP组检查遍历全部 `_fsdp_param_groups`。
- ParameterStateManager/projection在最终对象上建立；CURRENT-only typed numerics与禁root autocast scope经adapter基类 `_forward_prepared`真实消费，未整体cast模型。
- typed predict使用source真实condition和显式双sigma，decode/HPS/coordination使用真实端口，无新runner/no-op fallback。
- 新stage只读边界位于完整BaseTrainer tail之后，source/step/cursor/behavior/清grad与本rank真实RNG读取有测试；不调用尚待相应审核的publisher。
- 四项目标测试真实执行文件hash/profile/source/loader/PEFT/prepare/guard/HPS与六阶段消费者；CUDA/FSDP观察等叶子明确替换。除此缺口未发现必须新增的框架或重复owner。

## 本轮独立 CPU 门禁

当前 worktree 原样执行 plan 的两条允许命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
```

- Flash exit0：**7 passed in 15.86s**。
- H3通配符 exit0：**135 passed, 18 warnings in 16.09s**。warnings是checkpoint单进程DCP及损坏fixture覆盖提示，未过滤；不代表A-BIND-01已被测试。
- 无skip/xfail匹配，未跑全量；`git diff --check` exit0。
- 原Flash完整文件相对授权 `d818b8a231b52a9a024fa2f522057acdbb7bf099` 历史diff无输出、exit0，七项及断言保留。

## 仍未完成

默认唯一RunController装配、真实data/profile/publisher组合和两阶段整run fresh恢复/下一步对照仍需后续实现。C-PUBLISH-01仅由C对应复审裁决，本报告不批准兄弟任务。未执行GPU/服务器/真实HTTP/PG创建/下载/环境升级/push；CPU不证明实际FSDP2/NCCL/权重或音视频质量。**GPU_PENDING**。
