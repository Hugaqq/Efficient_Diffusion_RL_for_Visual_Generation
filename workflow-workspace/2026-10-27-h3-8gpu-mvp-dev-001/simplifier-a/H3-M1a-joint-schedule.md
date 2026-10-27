# H3-M1a-joint-schedule — Simplifier A

## 裁决：approved（仅本原子项）

审核对象：当前 HEAD `7b29f4bf8d990794d4b9bc826a82bd390861d530`，分支 `pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`。已读 task.md、plan.md、assignments/a.md、Coding A 交付、实际三个新增文件、AGENTS.md、UniRL LICENSE 与相关上游 packing/schedule 语义。

未发现需要 Coding A 返工的实现或过度开发问题。没有直接修改源码或测试。

## 实现与边界

- `visual_rl/models/h3_joint.py::H3JointLayout`：正整数双模态几何；pack 检查 B/shape/dtype/device，video-first flatten+cat；unpack 按独立元素数恢复。无 cast/detach，梯度保留，pack 新存储与 unpack 共享 packed 的约定明确。没有把 storage packing 混同 attention tokens，没有引入 mask/registry/model owner。
- `visual_rl/algorithms/dynamics/h3_schedule.py::H3ScheduleSnapshot`：CPU FP32 精确输入、有限/范围/端点/严格递减/同长校验；tuple 保存独立所有权，读取重新构造 FP32 tensor。build 采用 N+1 点和独立 shift12/3，碰撞拒绝；transition 显式整数索引、四个独立 sigma scalar，无 cursor、平均 dt 或 Gaussian ABI 混用。检查集中在构造函数，build 复用，不存在重复校验框架。
- `tests/test_h3_mvp_cpu.py`：六项真实合同测试；逐元素不等 AV 几何、B2、非连续输入、四种浮点 dtype、双流梯度、alias、形状与 dtype/device 负例；双 grid 手算中点/末步/单步、精确位模式恢复、输入输出 mutation、不合法 grid/count/shift/index、FP32 碰撞；导入路径核验当前 worktree。未用 mock 替代被测 packing/schedule，无 skip/xfail。
- 上游 `diffusion.py::pack_dual_streams/unpack_dual_streams` 与 generate 的 N evaluations/N+1 points 注释支持本合同；vendor scheduler 的点数/去重语义没有被不加区分地搬入。独立窄实现未引入 vendor 依赖或通用框架。

## 独立 CPU 复核

在当前 worktree 原样执行 task.md 两条固定命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
```

- Flash：exit 0，**7 passed in 12.19s**；当前文件中七个原命名测试函数均在，无 skip/xfail 匹配。
- H3：exit 0，**6 passed in 0.58s**。
- 审核开始 `git status --short --branch` 干净；`git diff` 无输出；`git diff --check` exit 0。这些只证明当前工作区相对 HEAD 状态，不冒称已核验历史提交 diff。Coding 记录称仅新增允许文件；orchestrator 应通过引擎 snapshot diff 再核对相对前置提交的路径与原七项断言未变。

## 非阻塞边界 / 后续

当前生产源码搜索只有这两个类的定义，没有生产消费者；这是 assignment 明确允许的前置合同，不是完成 H3 MVP。后续 CPS/adapter 必须实际消费，随后补齐旧锚、world 归约、完整 tail 与 fresh 恢复，不能用本次批准替代这些任务审核。

无本项最小修复要求。未运行全量、服务器、GPU、网络部署、下载、环境升级或 push。CPU 不证明 H3 权重、CUDA/FSDP2/NCCL、音视频质量；**GPU_PENDING**。
