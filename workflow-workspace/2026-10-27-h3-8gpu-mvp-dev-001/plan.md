# H3 MVP 执行计划

## M0 已核对事实

- 唯一总需求：本目录 task.md；reference 是证据和候选，不是要求全量照搬的框架。
- 当前引擎 worktree 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`，实际起点 `076635a61263ffd5e7c5be4cc64d77ef99242c51`。task.md 中 d818b8a 为历史起点，不改写 task。初始 `git status --short --branch` 无改动。
- 首次真实固定 baseline：exit 0，`7 passed in 12.90s`；`git diff --check` exit 0。未跑全量、未运行 H3 集（尚不存在），未操作服务器/GPU/网络/环境。
- 已读 AGENTS.md、README.md、pyproject.toml、reference/{architecture-review,hardware,topology,upstream-comparison}.md、sources.json、UniRL LICENSE 及相关 packing/diffusion/CPS/scheduler 源码。
- 当前 `models/catalog.py` 只有 SD3/Wan；`core/contracts/model.py` 尚无 T2VA/audio；`runtime/lifecycle.py` 明确 single-process 并先构造 optimizer 再 prepare；`dynamics/interface.py` 的 mean/std 与 record 都是 Gaussian 正方差合同；`rollout/collector.py::_finalize` 强制 decode；`optimization/kernel.py` 的 slot 缩放还不能当 world 归一化；`trainer/grpo.py` 在 optimize 后才提交 data/collection/next-step。不能仅添加 YAML 或删 guard 称为 H3 支持。

## 当前轮更新

W4三项均获对应simplifier approved；代码汇合 `fb085fb35fb239cdd4c151d39decbd7fae268c9d`，当前HEAD `6f5e46fce212e00d547a3ca07fa5524018bef816`。Orchestrator读实际adapter/types、DCP codec、组合测试/三审查，亲跑Flash7/7（13.79s）、H3通配符108/108（8.54s，9条no-dist/损坏fixture warnings保留），均exit0；原Flash历史完整diff仍为空。**正式关闭W3三真实对象CPU组合证据缺口**；该仿真仍不证明GPU或默认RunController装配。

**W5并行**：A真实本地data reserve/commit/边界恢复，B单一H3 profile编译/本地事实准入，C现AtomicCheckpointManager共同staging/全rank codec+rank RNG/data/单writer发布恢复。三项接口/文件独立，见decisions/wave-05.md。结果批准后优先实际唯一RunController默认装配，不另造Runner或用raw DCP目录当完整checkpoint。

明确首个reward profile=h3_hpsv3_video_only_v1（现all-124/top37 HPS，权重1）；不声称CLAP/ImageBind复现或音频质量，joint AV CPS元素权重不变。124帧服务资源尚未准入，不启动/访问旧81帧服务。task允许现有评分接口，这一选择避免另造平台或未核验许可下载。后续仍必须接ModelAdapter/recipe/唯一RunController与完整tail checkpoint/fresh恢复；不能把测试注入callback当生产装配已完成。

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
| H3-M2a1-forward | B | 已实现真实T2VA几何、attention packing、双t与单次callable，连接CPS目标测试 | approved；snapshot edd5c87；目标17项，不称完整adapter |
| H3-M2a2-loader | B | 已实现本地meta/LoRA/精度岛/严格CPU输入 | approved；snapshot919b6ab；H3目标22项 |
| H3-M2a3a-text-cache | B | 已实现严格本地layer50 cache、覆盖、有界读取 | W1 approved；独立feature6项 |
| H3-M2a3b-decode | B | 已实现双VAE CPU真实调用与严格AV转换/lease | W2 approved；独立feature6项 |
| H3-M2c-hps-reward | B | 已实现原HPS factory/client/JPEG验证链适配 | approved；W4同stage真实对象CPU组合已通过 |
| H3-M2b1-model-adapter | B | 已注册ModelAdapter，真实模型域端口及封闭类型 | W4 approved；4项目标；G3/runtime仍待绑定 |
| H3-M2b2-profile | B | 固定H3 schema2 profile编译/本地前置事实检查 | W5 active；不假造runtime/G3 success |
| H3-M3a-prepare | C | 已实现WORLD8 prepare/最终optimizer及单guarded callable | W1 approved；独立feature10项；不是CUDA证据 |
| H3-M3b1-world-credit | C | 已实现world准入/完整credit/d*n/N及公开divisor8设置 | W2 approved；world8项+reducer1项 |
| H3-M3b2-phase-norm | C | 已实现同序gate/aux轮转/global唯一norm | approved；W4同stage真实对象CPU组合已通过 |
| H3-M4a1-anchor | A | 已实现一次性完整rank-local replay冻结锚 | W1 approved；独立feature6项；全rank阶段接线待完成 |
| H3-M4a2-rollout-pool | A | 已实现真实CPS采样→CPU轨迹/终态池 | W2 approved；独立feature6项 |
| H3-M4a3-stages | A | 已复用原BaseTrainer六stage与tail | W3原子approved；无默认noop，尚无RunController绑定 |
| H3-M4a4-object-integration | A | 已验证真实stages+HPS+Coordination八线程同链 | W4 approved；4cases，CPU非GPU |
| H3-M4a5-data-source | A | 实际本地prompt/cache请求、预留提交、确定流与边界恢复 | W5 active；不再用测试Data充当生产源 |
| H3-M4b1-state-codec | C | 已实现allrank DCP model/Adam/scheduler/buffer及fresh空Adam恢复 | W4 approved；7项，未publish complete |
| H3-M4b2-checkpoint-publish | C | 现manager共享staging/纯writer、rank RNG/data与两阶段恢复 | W5 active；不造第二manager |
| H3-M4c-production-integration | B | 补齐现 controller/唯一入口的生产装配、最小配置和 torchrun 说明 | 上述逐项 approved；tiny 端到端走真实控制链，不用 mock controller |
| H3-M5-candidate | Orchestrator | 查全调用链/审核/边界，亲跑两集，final | 所有必须项批准后才 complete；引擎独立 candidate 再验证/安全发布 |

表中是依赖规划，不是一次大授权。每配对每轮只派一个小项；实现暴露必要拆分时先更新 assignment，不顺手写整套架构。无消费者基础合同只能暂时存在于队列前置，最终不允许未消费接口冒充闭环。

## 串并行与审查规则

优先寻找并行小项；本轮W5三个配对均active，独立worktree及分离feature测试。原test_h3_mvp_cpu.py只读，不为拆分测试制造重构。只有实际未稳定接口才顺序/idle，并写明具体依赖。每次读取coding源码/diff/测试与对应simplifier，记录实际branch/前置HEAD/snapshot/裁决。run内暂存汇合不是批准；changes回同slot重测重审。汇合后亲跑完整H3通配符及Flash，不用各分支局部通过代替集成通过。

## 固定 CPU 门禁

每个源码原子项完成后，两条原样运行（第一项保留七函数，不更改断言、不 skip/xfail；第二项随功能增加，不跑全量）：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
```

最终目标集必须覆盖 joint/双schedule、CPS/Gaussian边界、冻结旧锚、world 数学、真实事件序列及 uneven 拒绝、完整 checkpoint/fresh 恢复、真实 controller tiny 闭环；Flash baseline 不证明 H3；当前108项目标集为此前93+W4组合4/adapter4/statecodec7；含三对象CPU同链和fresh模型/Adam磁盘恢复，仍缺唯一controller/data/profile/完整checkpoint发布与整run恢复。最终必须跑通配符集合（旧单文件命令仅保留可用，不作最终替代）。
