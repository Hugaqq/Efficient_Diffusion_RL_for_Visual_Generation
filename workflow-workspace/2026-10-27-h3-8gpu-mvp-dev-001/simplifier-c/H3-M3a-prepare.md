# H3-M3a-prepare — Simplifier C

- verdict: **approved**
- 审核范围：仅 Coding C 的 WORLD8 prepare 原子项及其依赖/回归；不批准 A/B 本波功能，不代表 MVP 完成。
- 审核 HEAD：`467c8fd929804aa291bf01a5fa7cda372cc507db`（W1 暂存汇合）。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-49386bc3-79aa-41ed-9651-40cfbef318e5`。审核开始 status 干净。

## 核验依据

完整读取 task.md、plan.md、assignments/c.md、coding-c/H3-M3a-prepare.md，以及实际 `visual_rl/models/lifecycle/h3_fsdp.py`、`tests/test_h3_mvp_cpu_prepare.py`。核对已批准 `implementations/h3_loading.py`、既有 `lifecycle/prepared.py`、tiny fixture；读取 UniRL Apache-2.0 LICENSE 及实际 `train/backend/fsdp/{wrap,state}.py`，未导入参考代码。

1. 实际 WORLD/NCCL/rank/device 准入与单节点摘要一致性检查在权重/包装前执行。rank0 独占真实 CPU loader，peer 空 state 仍调用同一 setter；无 PG 创建/销毁、CPU 分片 fallback 或额外 runner。
2. 主 blocks 先真实 nonreentrant checkpoint wrapper，再 FSDP2；单 WORLD mesh，BF16 forward/FP32 reduce、不 cast 输入、reshard 开启。root/refiner 保持冻结复制；全部 trainable 必须是主 blocks LoRA。生产有 FSDPModule/DTensor 后置核验。
3. 分片后才 to_empty，随后全 rank DCP strict/full_state_dict/broadcast_from_rank0 setter；canonical state FQN 不放宽。显式恢复并逐值检查 rope.inv_freq，完整 state shape/device/dtype 与 trainable 范围复核后才绑定真实 AdamW。未提前创建 ParameterStateManager 或 optimizer。
4. 复用 PreparedModelBundle/PreparedComponentHandle guard 与 optimizer 参数集合检查；窄 accumulate 能力没有伪造完整 Accelerator。单 callable 可供既有 H3 bridge 使用；close 仅释放本地 guard/引用，失败不做隐藏 collective cleanup。
5. 独立十项测试实际执行 loader/PEFT/AC/to_empty/AdamW/guard/bridge forward-backward-step。明确替换 CUDA、WORLD/metadata、mesh、sharding、collective setter 和重模型类叶子；CPU assign=True 只在测试 setter 使用。测试验证最终参数身份、owner/peer、顺序、buffer 恢复及异常拒绝。dtype metadata probe 不被当作实际 FSDP 执行。
6. 实现为窄 prepare 入口，未重写 runner、重复 guard 框架、改旧 single 路径或引入同波 A/B 模块。当前无 controller 消费是明确的后续集成项，不以本项批准冒充完整闭环。

未发现本原子项必须退回的正确性或过度开发问题。返工清单：无。

## 独立真实 CPU 门禁

在本审核 worktree 原样执行，均 exit 0：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.08s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 44 passed in 2.37s（汇合后的22项基础 + A6 + B6 + C10）
```

H3 目标文件 grep 无 skip/xfail；未跑全量。获准历史命令 `git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` exit0/无输出，原七项完整文件含断言不变。累计 name-status 仅见 H3 新源码/测试和 workflow-workspace 资料新增，没有旧源码/配置净修改；该累计比较不冒充单个 coding snapshot 的精确 diff。`git diff --check` exit0。

## 接收与未验证边界

由 orchestrator 登记 C 的引擎 snapshot 和本项批准，继续 world 数学/phase、controller 接线与完整 tail/checkpoint/fresh 恢复；W1 接收仍须其余配对各自裁决及 orchestrator 汇合门禁。

真实 H3 权重兼容、CUDA DTensor 物化、FSDP2 mixed precision/AC、NCCL 广播和八卡 forward/backward 均未执行，后续必须另获授权验证。CPU 44项通过不证明这些行为或媒体质量。未修改源码、测试、他人报告；无服务器/GPU/网络部署/下载/环境升级/push。**MVP 未完成，GPU_PENDING。**
