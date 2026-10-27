# H3-M1b-cps — Simplifier A

## 裁决：approved（仅 CPS 数值原子项）

审核 HEAD：`725f48f4d67c67d90c60cc3536f4bc40f12802f2`；当前分支 `pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`。已重新读取 task、plan、assignment A、对应 Coding A 交付、当前 joint/schedule/CPS 与完整 H3 测试，并核对实际旧 Gaussian 校验及 UniRL CPS 源码。前轮已完整阅读 UniRL Apache-2.0 LICENSE。

未发现需退回 Coding A 修复的实现、边界或过度设计问题。本角色未修改源码或测试。

## 具体检查

- `visual_rl/algorithms/dynamics/h3_cps.py::H3CPSDynamics.stats`：唯一双流数学实现；使用精确双 schedule，state/prediction 转 FP32，分别计算 q/x0/e_hat/mean，保留双 dt；没有混用 sigma/index、额外翻符号、epsilon 或 clamp。非末步 q 下溢明确拒绝。
- `H3CPSDynamics._score`：packed generated 元素的负 MSE 等价于按 Dv/Da 加权，非模态等权，非 Gaussian likelihood。finite 保护覆盖 mean、prediction、量化 action 与 score。
- `H3CPSDynamics.sample`：显式 generator、video-before-audio 两次 FP32 draw；末步 q0 仍抽两份 noise；no_grad 内量化为 FP16，返回 record，未依赖全局 RNG。
- `H3CPSRecord` / `evaluate`：输入 state/action 与读取结果均拥有副本；record 只绑定冻结数值合同，不持有模型、generator 或图。evaluate 校验 layout/schedule/eta 精确一致，复用 stats，对固定 action 评分，不重抽 noise 或重建 action；梯度仅经当前 prediction。下划线 tensor 是内部缓冲，不是公开可变接口。
- `tests/test_h3_mvp_cpu.py`：前六项合同仍在；新增六项实际验证非零 B2、不等元素数与独立 schedule 的 FP32 数学、FP16 action、按元素加权、显式 RNG 可复现与模态隔离、末步 draw、record mutation、解析梯度与末步量化残差、Gaussian 正方差拒绝及输入/配置/overflow/underflow 负例。没有 mock 所测内核，无 skip/xfail。
- `visual_rl/algorithms/dynamics/interface.py::TransitionMeanStd.validate_against`：当前真实实现仍拒绝 std<=0；H3 测试直接调用这一端口验证 q0 拒绝与正 std 接受。
- 对照 UniRL `sde/kernels.py::CPSSDEStrategy`，所选 q、mean 和 stopgrad action 负平方误差语义一致；此实现没有引入上游通用 registry/框架。

## 简化结论

stats/sample/evaluate/record 分工是本项所需的最小数值与所有权边界。sample 和公开 record 构造重复检查部分 state 是边界校验，并未复制数学公式；不建议为减少代码删除。无新 runner、factory、manifest、哈希或池抽象。暂时没有生产消费者，符合本原子项明确范围，不构成完整 MVP。

无最小修复要求。

## 本轮独立门禁

在当前 worktree 原样执行 task.md 两条命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
```

- Flash：exit0，**7 passed in 12.35s**。七个原命名测试函数仍在，grep 无 skip/xfail 匹配。
- H3：exit0，**12 passed in 1.03s**，含当前 worktree 模块导入身份核验。
- 审核开始 status 干净，`git diff` 无输出，`git diff --check` exit0。这些不证明历史提交范围未越界；历史 snapshot diff 未在允许命令中，沿用 plan/ledger 的主会话/引擎审计待办，不冒称核验原七项全部历史断言未变。

## 后续边界

固定 action 不等于冻结 replay old anchor；rollout_score 仅诊断。后续仍必须实现真实 adapter/rollout/replay 消费、旧锚、world 数学、评分、完整 tail/checkpoint/fresh 恢复并逐项审核。GPU/CUDA generator、FSDP2/NCCL、H3 权重及音视频质量未验证。未跑全量、未操作服务器/GPU/部署/下载/升级/push。**GPU_PENDING，MVP 未完成。**
