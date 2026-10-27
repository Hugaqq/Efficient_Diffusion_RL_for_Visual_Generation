# H3 MVP 执行计划

## M0 已核对事实

- 唯一总需求：本目录 task.md；reference 是证据和候选，不是要求全量照搬的框架。
- 当前引擎 worktree 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`，实际起点 `076635a61263ffd5e7c5be4cc64d77ef99242c51`。task.md 中 d818b8a 为历史起点，不改写 task。初始 `git status --short --branch` 无改动。
- 首次真实固定 baseline：exit 0，`7 passed in 12.90s`；`git diff --check` exit 0。未跑全量、未运行 H3 集（尚不存在），未操作服务器/GPU/网络/环境。
- 已读 AGENTS.md、README.md、pyproject.toml、reference/{architecture-review,hardware,topology,upstream-comparison}.md、sources.json、UniRL LICENSE 及相关 packing/diffusion/CPS/scheduler 源码。
- 当前 `models/catalog.py` 只有 SD3/Wan；`core/contracts/model.py` 尚无 T2VA/audio；`runtime/lifecycle.py` 明确 single-process 并先构造 optimizer 再 prepare；`dynamics/interface.py` 的 mean/std 与 record 都是 Gaussian 正方差合同；`rollout/collector.py::_finalize` 强制 decode；`optimization/kernel.py` 的 slot 缩放还不能当 world 归一化；`trainer/grpo.py` 在 optimize 后才提交 data/collection/next-step。不能仅添加 YAML 或删 guard 称为 H3 支持。

## 当前轮更新

M1a与M1b均已实现且 simplifier-a approved；CPS coding snapshot `725f48f4d67c67d90c60cc3536f4bc40f12802f2`，本轮 HEAD `8ed954f2562cf9d1f5432a7bb84ada181cdae6a9`。Orchestrator 读CPS及完整目标测试、交付/审查，并独立复跑 Flash7/7（12.78s）和 H3十二项（0.88s），均 exit0。数值前置合同通过，但无真实模型/生产controller消费者或旧锚。

本轮唯一 active 为 B/H3-M2a1-forward；A/C idle。将原M2a完整adapter进一步拆成forward、loader、AV decode/cache三个原子项，防止一次大提交。完整链、GPU 与最终 candidate 均未通过。历史 snapshot diff 目前不在命令白名单内，精确历史范围审计待主会话/引擎材料补足，见 merge-ledger；不能以 clean 工作区替代。

## 权限与完成边界

仅本引擎 worktree、本地自动 snapshot/最终 candidate merge；不自行 commit/reset/rebase/push/切分支。保留原七项测试及断言、旧 Flash 配置/语义。Orchestrator 只写指定协调文件，simplifier 只审不改。需返工归原 coding。run 内暂存合并不是批准。

硬件按用户历史观测：单节点目标 8×64GiB，无 NVLink，双 NUMA；10 个可见 UUID 不代表授权集合或空闲。8 UUID/窗口、依赖/权重/数据/评分资源及真实 FSDP2/NCCL 均待确认。最终只能交 CPU candidate，标记 **GPU_PENDING**。

## 最小接口方向

- 保留 `python -m visual_rl.train CONFIG` → 现 RunController → 六阶段 trainer；不另造 Runner。
- 首版仅 T2VA、SP1/TP1、rollout/replay B1、同模型、完整 collection 一次更新、固定等量 rank 工作。禁止音频被删、假媒体、隐含 mock、常量 reward、零 loss 占位。
- 数学选择 UniRL `8cfbac17765004afb68fef08a4e011a4c8daf555` 同模型路径：独立 video/audio sigma；noise-ward prediction 在 adapter 符号转换一次；CPS FP32 负 MSE，按 generated 元素数合并 AV，固定 FP16 action；全 policy 步含 q=0 末步。不是 Gaussian density、不是 VeRL 0.5/0.5 权重，也不是三上游严格复现。
- 双 schedule N 次调用必须 N+1 sigma，默认 video shift12/audio shift3；固定画布候选256×384×124@24，不暗改121。首原子项只锁 generated 状态 packing，不造完整 attention packing 框架。
- 后续必须接实际消费者：latent-only bounded pool → 真 decode/score → world reward 表/advantage → 更新前一次冻结 replay anchor → current replay/backward/一次 optimizer → trainer 完整 tail → checkpoint → 同拓扑 fresh restore/下一步。
- 上游许可仅核验 UniRL Apache-2.0 与已读文件头；不据此假定所有 vendor、权重、reward 许可都相同。先读文件实际许可再最小移植，绝不整体 vendor。

## 原子队列（后续均未授权执行，逐项细化 assignment）

| ID | 配对 | 单步目标 / 预期边界 | 依赖与验收 |
|---|---|---|---|
| H3-M1a-joint-schedule | A | 已实现 `models/h3_joint.py` generated AV packing；`algorithms/dynamics/h3_schedule.py` 精确双 sigma；六项目标测试 | approved；coding snapshot7b29f4b；仅前置合同 |
| H3-M1b-cps | A | 已实现 CPS sample/evaluate/record，消费 M1a；Gaussian未放松 | approved；coding snapshot725f48f；H3目标12项 |
| H3-M2a1-forward | B | T2VA真实几何/attention packing、双t、单次callable、符号转换，连接CPS目标测试 | 本轮唯一active；依赖M1a/b；tiny仅替重transformer叶子，不称完整adapter |
| H3-M2a2-loader | B | 本地artifact/API准入、meta构造/精度岛/主blocks LoRA，给C准备FQN输入 | forward稳定；选真实公开依赖接口或已审最小移植，无安装下载，不先全模型上GPU |
| H3-M2a3-aux | B | 严格离线text cache消费、真实video/audio VAE decode与AV结果 | loader和forward稳定；双VAE FP32、有界逐条媒体；如仍过大再拆，不造服务平台 |
| H3-M2b-profile | B | H3 独立声明/recipe/registry 接入现 capability 与入口 | adapter 接口稳定；未知配置/不支持组合早拒，不改 Flash |
| H3-M3a-prepare | C | 单 owner WORLD8 FSDP2 prepare/最终参数绑定，再 optimizer；真实 lazy API | 模型 block/FQN 合同稳定；tiny 验证调用顺序、严格加载覆盖；CPU 不证明 CUDA |
| H3-M3b-world | C | 固定同序 phase、全局 reward/advantage 与 d*n/N loss、global norm | prepare 合同稳定；不等/空 rank 模型前拒绝，数学与事件序列 CPU 验证 |
| H3-M4a-pool | A | 现 stages 的 latent-only bounded AV collection，真实评分接口与冻结旧锚 | adapter/数学/phase 稳定；资源上界、实际锚消费者；必要时拆 decode/reward 与 anchor 两项 |
| H3-M4b-tail-checkpoint | C | 完整 tail safe point，现 manager 的全rank codec/单writer，同拓扑 fresh optimizer 恢复 | 上述闭环稳定；模型/Adam/scheduler/data/RNG/语义覆盖，下一步对照 |
| H3-M4c-production-integration | B | 补齐现 controller/唯一入口的生产装配、最小配置和 torchrun 说明 | 上述逐项 approved；tiny 端到端走真实控制链，不用 mock controller |
| H3-M5-candidate | Orchestrator | 查全调用链/审核/边界，亲跑两集，final | 所有必须项批准后才 complete；引擎独立 candidate 再验证/安全发布 |

表中是依赖规划，不是一次大授权。每轮只派一个小项；实现暴露必要拆分时先更新 assignment，不顺手写整套架构。无消费者基础合同只能暂时存在于队列前置，最终不允许未消费接口冒充闭环。

## 串并行与审查规则

本轮顺序 B；A/C idle。所有新增 H3 CPU 测试集中同一文件，因此当前不存在完全独立的并行写边界。后续只有确认文件和输入输出合同均独立才 parallel。每次读取 coding 实际 diff/测试与对应 simplifier，记录 branch/前置HEAD/引擎 snapshot SHA/裁决/集成基线；未获真实 SHA 就 pending，不编造。changes 回同 slot 并重测重审。

## 固定 CPU 门禁

每个源码原子项完成后，两条原样运行（第一项保留七函数，不更改断言、不 skip/xfail；第二项随功能增加，不跑全量）：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
```

最终目标集必须覆盖 joint/双schedule、CPS/Gaussian边界、冻结旧锚、world 数学、真实事件序列及 uneven 拒绝、完整 checkpoint/fresh 恢复、真实 controller tiny 闭环；Flash baseline 不证明 H3；当前十二项目标集只证明已批准 joint/双schedule/CPS 合同，尚未覆盖全部完成条件。
