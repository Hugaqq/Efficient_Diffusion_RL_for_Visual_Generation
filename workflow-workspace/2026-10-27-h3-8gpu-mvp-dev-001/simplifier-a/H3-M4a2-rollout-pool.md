# H3-M4a2-rollout-pool — Simplifier A / W2

verdict=approved

仅批准 A 的 rank-local CPS rollout / CPU latent-only 池原子项。返工清单：无。不是 B/C 配对审核、W2 整体接受或完整 MVP 验收。

## 审核对象

- HEAD：`ced9a871df0ee6bfe8b560e90da03bc12d4dda0e`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-add54f4b-0fbd-4165-a0a9-a8973ccd54d8`；开始工作区干净。
- 已读当前 task、plan、assignment A、wave-02、Coding A 交付，完整 `h3_pool.py` 和六项 feature 测试；核对当前 CPS、anchor、forward 及既有 objective 消费边界。所有读取均来自当前 worktree，不沿用旧工作区路径。
- 未改源码、测试或他人报告。

## 实现 / 边界 / 简化

1. `visual_rl/algorithms/trainer/h3_pool.py::H3RolloutPool.__init__`：绑定 collection/behavior 与精确 CPS 合同；显式 sample/group/member 三元组保证 sample 唯一、完整局部 K 组及合法 member 域。max_samples、max_cells、完整 N 步 x/action/score 加 terminal 的 retained tensor payload 在采样前明确检查。公式 `samples*N*(4*D+4)+samples*2*D` 与测试实际 payload 相符。
2. `collect`：先验证全部初态及显式 generator 的精确样本集合、CPU FP16 B1/detached/finite、目标设备与对象独立性，再拥有全部初态快照。最后一条非法输入不会在前面样本已调用模型后才暴露；callback 修改调用方原初态不影响后续样本。
3. 真实采样是 sample-major/index0..N-1、no_grad、每格一次 predict 和现 CPS.sample，action 继续为下一 x，含 q0 末格。逐格经公开接口构造独立 CPU record，设备临时记录不跨步累计；无 CPS 公式复制、强制 decode、假媒体、模型持有或新 runner。
4. transient collecting 防重入；只有完整生成且 behavior 前后版本匹配才发布 ready。异常设 failed、没有部分池可读、不重试/重置 seed。release 幂等清除池引用并禁止读取；没有伪称 trainer tail 已提交。
5. `records/samples/terminal` 核对 collection/behavior；记录为已批准 CPS 固定所有权格式，terminal 返回 clone 且恰等最后 action，未知 sample 拒绝。借出的 records 仍归读者持有，release 不保证释放调用方自己的引用，这一点代码/交付已明确。
6. `tests/test_h3_mvp_cpu_pool.py` 六项实测真实 forward→CPS→pool→既有 anchor→既有 beta0 PPO/backward。多 sample、多步、双流连续、顺序、CPU/dtype/detach、seed/模态隔离与末步 RNG reference、预算与完整组/全部初态拒绝、失败原子性、版本错误、mutation/weakref/release 均有实际证据。只有重模型叶子 tiny；没有 mock 池/CPS/锚/目标函数。

无需为减少代码删除上述边界检查。未见额外通用事务、池抽象层、重复 PPO 或兄弟新模块依赖。

## 当前汇合代码独立门禁

在上述当前 worktree 原样执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
```

- Flash：exit0，**7 passed in 13.74s**。
- H3 通配符：exit0，**65 passed in 3.20s**，不是 Coding A 分支50项的历史结果替代。目标文件无 skip/xfail 匹配，未跑全量。
- `git diff` 无输出，`git diff --check` exit0。
- `git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py` 无输出、exit0；原七项完整文件和断言与授权历史基线一致。

## 非阻塞但后续必需边界

- payload 上限不包含 caller-owned 输入、公开读取副本、Python metadata 或瞬时模型 workspace；后续资源准入必须合计这些开销，不能称为进程 RSS 上限。
- caller 仍须保证 EVAL、固定参数及每次 policy mutation 更新 behavior version；callback 不保留设备输入。前后版本检查不是全 rank 健康门，也不检测未申报参数修改。
- 源码搜索目前只有池定义，没有生产 stage/controller 消费，符合本项明确非目标。下一步必须接现 Rollout/Reward/Optimize 阶段、真实评分/credit、世界同步与完整 tail/fresh 恢复；不能继续用独立 helper 数量冒充闭环。
- CPU tiny 测试不证明真实 H3 权重、CUDA 搬运、FSDP2/NCCL、8卡资源或音视频质量。无服务器/GPU/下载/部署/环境升级/push。**GPU_PENDING，MVP 未完成。**
