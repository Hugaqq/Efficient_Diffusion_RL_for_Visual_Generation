# VisualRL v0.8 结构收敛与真实链路验证总计划

> 日期：2026-08-03
> 分支：`refactor/flow-factory-modular-core`
> 状态（更新于 2026-08-08）：**M0–M4 的结构迁移已完成，runtime 顶层已收敛到计划中的 graph/model/algorithm/preprocess/reward/checkpoint/terminal owners；M5–M6 的最终内存治理、clean wheel 与冻结尚未完成。按用户要求，M7 已提前进入真实 GPU engineering bring-up，但当前结果不计为 final A7：Flow-GRPO×SD3 第一次在 preprocess 发生 30.8 GiB CUDA OOM，根因是 prepared transformer 与全部冻结文本编码器同时驻留；SD3/Wan 已改为模型层 CPU-static prompt encoder，并只在单步模型 port 搬运 conditioning tensor。修复后的 Flow-GRPO×SD3 已完成 20/20 optimizer commits，生成 SUCCESS 与完整 step-20 checkpoint，driver peak 21,039 MiB、无 OOM；TempFlow-GRPO×SD3 也已完成 20/20 optimizer commits 并生成 SUCCESS。Flow-GRPO×Wan 的 `row_microbatch_size=1` 已消除四行同时 recompute 的 OOM；attempt 4 进一步证明 reward 有四个唯一值、112/112 active advantages 非零、480/480 参数得到 grad tensor，但数值全部恰为零。根因是 Wan 模型 scheduler 默认 `stochastic_sampling=False`，旧 Dynamics 继承该设置后存储 `action == mean`，使旧策略处的 score-function 梯度必然为零。采样模式现已成为显式 Dynamics contract/config/replay identity，canonical Wan GRPO 投影固定为 stochastic，mode drift 会 fail closed；158 个直接回归测试通过，GPU 3 的 attempt 6 正在做真实 1-step 验证。旧 general reward 虽取得真实有限分数，却在声明为 FP32 时内部启用了 CUDA autocast；新 revision `world-r1-8e46b1b63498` 已移除 autocast，并通过真实 HPS smoke 返回 `0.23295444250106812`，checkpoint/health/score receipt 已哈希落盘。Wan 的 `/dev/shm` 副本已按文件名、文件数和字节数核对完成；Flash-GRPO×Wan 的 20-step bring-up也已启动。3D reward 的 tmpfs 部署已修复 `ninja` PATH，真实 DA3 重建、Gaussian Splatting 渲染与 Qwen3-VL 评分链路已对 4 帧请求返回有限分数 `1.9351264214596995`，健康/score receipt 已哈希落盘。**
> 最新状态（2026-08-08 19:38 CST）：**结构与本地 M6 已完成；code `56507f6e…`、wheel `6f1533ef…`、六份 final config、general marker `648cebf…`、3D marker `fbceb863…` 已由 freeze record `a6c961fc…` 统一绑定。Flow×SD3、TempFlow×SD3、Flow×Wan、Flash×Wan 四条 final 20-step 正在 GPU 3/6/7/2 运行，World-R1 core/release 等待任一卡释放；所有 final checkpoint 先写 tmpfs，避免已实证的 NFS `folio_wait_bit_common` 阻塞。**
> 验收工具状态（2026-08-08 19:56 CST）：**截至 19:49 的一次关键节点检查，四个首批 supervisor 仍存活、无终态退出码，两条 World-R1 一次性队列仍在等待；此后未轮询远端。离线 20-step 终态演练发现并修复了 `audit_a7_route.py` 对 `progress.json` 嵌套 payload 的读取错误，修复版 `13c325bb…` 已完整接受该样本；六路矩阵演练也得到 `accepted=true`。`finalize_a7_acceptance.sh` 将只在终态原子生成逐路/矩阵收据，`archive_a7_acceptance.sh` 将只在六路 accepted 后把 tmpfs 证据校验复制并原子提交到 NFS。**
> 用户触发状态快照（2026-08-08 19:59 CST）：**四条首批 trainer 仍存活且尚无 exitcode/SUCCESS：Flow×SD3、TempFlow×SD3、Flow×Wan、Flash×Wan 分别已运行约 31/30/27/23 分钟，单点 driver memory 为 24,892/24,998/16,868/7,614 MiB，均低于 32,607 MiB；前三者快照利用率为 100%/66%/100%，Flash 快照时为 0% 但进程处于 `Rl` 且 CPU 活跃，单点不足以判定停滞。两条 World-R1 尚未启动，一次性队列仍在等待对应 SD3 路线 exit 0 + SUCCESS；无逐步日志轮询。**
> 事件驱动收尾（2026-08-08 20:01 CST）：**为落实“不用一直监视”，已启动唯一 one-shot watcher PID `1704242`（脚本 `f98073c4…`）。它只等待六个已知 supervisor PID 的退出事件，不读取 live step log；两条 SD3 退出后自动用 `/proc` 固化 World-R1 的命令/CWD/GPU/config/freeze 启动收据，每次终态调用 fail-closed finalize，六路 matrix accepted 后才调用 NFS 原子归档。`flock` 禁止重复 watcher；启动检查显示 `waiting_for_pid_events`，因此后续无需 Agent 轮询。**
> 关键节点检查（2026-08-08 20:10 CST）：**watcher 仍无退出码，事件日志仍停留在等待 Flow×SD3/TempFlow×SD3 supervisor；acceptance 目录尚无逐路收据，两条 World-R1 启动收据尚未出现。这只证明首批路线尚未产生终态，不据此读取 step 日志或判断成功/失败；后续继续由 one-shot watcher 接管。**
> 首条正式验收（2026-08-08 20:52 CST）：**Flow-GRPO×SD3 已由 watcher 自动审计为 accepted：同一 freeze 下 20/20 fresh optimizer commits、exit 0、SUCCESS、canonical step-20 checkpoint、最终 pre/post-clip gradient norm `0.003381970804184675`，显存序列 223 行（222 行 live）且峰值 24,892/32,607 MiB；acceptance SHA-256 `bc5f5d85…`。成功队列随后在 GPU 3 启动 World-R1 core，live procfs 收据把 trainer PID `1708533`、supervisor `1696016`、config `74cb229f…`、code/wheel freeze 与物理 GPU 3 绑定；其终态仍待 20-step 验收。**
> 第二条正式验收（2026-08-08 21:44 CST）：**TempFlow-GRPO×SD3 已由同一 watcher 审计为 accepted：20/20 fresh commits、exit 0、SUCCESS、最终 pre/post-clip gradient norm `0.0005376166081987321`，显存序列 523 行（522 行 live）且峰值 25,148/32,607 MiB；acceptance SHA-256 `bcba2ecb…`，两条 SD3 route 的本地 `reward_quality` content identity 同为 `c352220c…`。成功队列随后在 GPU 6 启动 World-R1 release-surrogate，procfs 收据绑定 trainer PID `1756567`、supervisor `1696017`、config `6f7feacc…` 与同一 code/wheel freeze。当前 2/6 route accepted，两条 World-R1 均已真实启动。**
> Wan 长时运行快照（2026-08-08 22:36 CST）：**四条 Wan trainer 尚无终态但均非空转：Flow×Wan、Flash×Wan、World-R1 core/release 的进程状态分别为 `Sl/Rl/Sl/Rl`，CPU 均持续活跃；物理 GPU 7/2/3/6 的单点 utilization 全部为 100%，显存分别为 17,060/17,136/17,194/17,002 MiB（总量 32,607 MiB）。该快照只用于排除明显停滞，不替代终态 OOM/20-step 审计。**
> 终态与恢复状态（2026-08-09 11:09 CST）：**Flow×SD3、TempFlow×SD3、Flow×Wan、Flash×Wan 已全部在同一 freeze 下通过正式 20/20 验收；Flow×Wan/Flash×Wan 的显存峰值分别为 17,568/22,555 MiB，最终 pre-clip gradient norm 分别为 `0.00047204949078150094`/`0.0005254881107248366`，acceptance SHA-256 为 `2d35e86d…`/`35759b33…`。World-R1 core 已在 tmpfs 原子提交完整 step-10 checkpoint，GPU 3 仍持续计算，证明长时运行不是初始化死锁。World-R1 release-surrogate 首次 final run 在产生 checkpoint 前收到外部 SIGTERM（exit 143）；日志无 OOM/Traceback，显存序列终止时仍远低于上限，但该 run 不计入验收。一次性恢复 watcher PID `2270592` 已等待 core supervisor：只在 core SUCCESS 且 route accepted 后，把 exit-143 的全部 run/log/launch receipt/checksum 证据保留到 tmpfs 与 NFS，再在释放的 GPU 3 上从同一 frozen source/wheel/config fresh 重跑 release 20-step；它不读取 live step log，也不覆盖失败证据。**
> 剩余路线容量检查（2026-08-09 11:12 CST）：**`/dev/shm` 总计 126 GiB、尚余 58 GiB，inode 使用率 1%；整个 A7 evidence root 当前 2.2 GiB，World-R1 core step-10 run 271 MiB，失败 release 仅 76 KiB。主机内存 available 112 GiB；虽然 8 GiB swap 已用满，但当前 RAM/tmpfs 余量足以同时保留失败证据、core step-20 和 fresh release checkpoints，不需要在 live optimizer/checkpoint 路径重新引入 NFS。core trainer 当前 RSS/HWM 为 33.9/40.3 GiB；所属 cgroup `memory.max/high=max`、`memory.events{oom,oom_kill,oom_group_kill}=0`、memory pressure 10/60/300 秒窗口均为 0，未观察到 CPU/cgroup OOM。**
> 第五条正式验收（2026-08-09 19:17 CST）：**World-R1 core×Wan 已在同一 freeze 下完成 20/20 fresh commits、exit 0、SUCCESS 与 canonical step-20 checkpoint；最终 pre/post-clip gradient norm 为 `0.003636404639109969`，显存序列含 5,429 个 live sample、峰值 18,664/32,607 MiB，route acceptance SHA-256 为 `b11a0c48…`。至此 5/6 route accepted，尚缺 World-R1 release-surrogate 的 fresh 20-step 重跑。**
> 恢复链路审计（2026-08-09 19:22 CST）：**远端冻结恢复器已接受 core 前置条件，但在启动 retry 前同步失败现场到 NFS，`sync -f` 进入 `folio_wait_bit_common`，GPU 3 因而空闲；这不是模型训练或 OOM。为避免 NFS durability 占据 GPU 重启关键路径，仓库版恢复器已改为先把失败现场迁入 checksummed tmpfs namespace、立即启动 retry、并行归档不可变失败证据，最终 matrix 验收前再 join NFS archive。该修正只作用于仓库后续版本，未替换远端已冻结 A7 脚本或 code/wheel identity。**
> 权威性：本文件是本分支后续结构修改、接口取舍和验证顺序的唯一计划来源。

## 0. 如何使用本计划

1. 实现前先查本文件的目标目录、唯一 owner、依赖矩阵和迁移表。
2. 本文件没有覆盖的结构问题，不直接在代码里临场决定；先在第 16 节增加决策记录，再实现。
3. 代码和本文件冲突时，以本文件为目标，但必须先更新“当前进度”和受影响的验收门禁。
4. 不用“测试暂时通过”替代职责判断。目录移动必须同时改变 import 方向、构造权和数据所有权。
5. 本文件取代 2026-08-02 版本中互相冲突的多阶段描述；历史内容由 Git 历史保留。
6. 最终 A7/A8 证据仍必须来自冻结后的同一源码/wheel；冻结前允许按用户要求进行真实 GPU engineering bring-up，用于暴露 OOM 和运行时错误，但必须明确标为 diagnostic 并在最终源码上重跑。

## 1. 最终目标与范围

VisualRL v0.8 只公开两个可替换轴：

```text
ModelAdapter × AlgorithmModule
```

- `ModelAdapter` 负责模型制品、条件编码、latent 几何、单步预测、解码、参数视图和模型状态。
- `AlgorithmModule` 负责一种完整后训练算法的蓝图与要求；它可以组合内部 rollout、dynamics、
  reward、advantage、credit、objective、recompute、update 和 trainer，但不能知道具体模型类。
- `Runtime` 是唯一同时看见具体 Model 与具体 Algorithm 的 composition root。
- “可组合”表示兼容组合能在不改另一侧源码的前提下绑定；不表示任意笛卡尔积都合法。
- 不兼容组合必须在模型权重和 reward 服务加载前给出结构化原因。

本轮必须支持并验证五种模型—算法组合、六条版本化 route：

1. Flow-GRPO × SD3.5；
2. Flow-GRPO × Wan2.1 T2V；
3. TempFlow-GRPO × SD3.5；
4. Flash-GRPO × Wan2.1 T2V；
5. World-R1 recipe × Wan2.1 T2V（内部复用 Flow-GRPO，不新增 policy optimizer），包含
   `world_r1_core_v1` 与 `world_r1_release_surrogate_v1` 两条 route。

World-R1 core 与 release-surrogate 是同一模型—算法组合的两种 integration recipe；release 的
100-step main / 50-step dynamic 都是 **optimizer phase steps**，不是 diffusion timesteps。

本轮不实现 FSDP、DeepSpeed、多节点、新的非 GRPO trainer family，也不承诺独立内部组件是长期
公共 SDK。内部拆分的目的首先是单一职责、可测试和不重复，不是制造更多公共插件轴。

## 2. 三条不可妥协的结构原则

### 2.1 模型层和算法层彻底分开

```text
models  ───────────────┐
                       ├──> runtime composition ──> bound training run
algorithms ────────────┘
```

- `models` 不得 import `algorithms`，也不得构造 Dynamics、Rollout、Reward、Credit 或 Objective。
- `algorithms` 只依赖模型的 import-safe port，不得 import `SD3Adapter`、`WanT2VAdapter` 或 Diffusers
  pipeline 实现。
- model 名称、algorithm 名称和 recipe 名称都不得成为对方模块中的条件分支。
- model 只暴露不可变 scheduler artifact blueprint；具体 replay factory 和 transition kernel 由
  algorithm/runtime 绑定。

### 2.2 一个能力只有一个 owner

禁止同时保留以下双栈：

- `models` / `model_adapters`；
- `algorithms` / `algorithm_modules` / `optimizers` / `advantage`；
- `rollout` / `rollouts`；
- `feedback` / `rewards`；
- `artifacts` / `checkpoint` 中两套 checkpoint；
- `runtime` / `training` / `trainers` 中重复的 composition 或 loop owner。

兼容 facade 只能在一个迁移 slice 内短暂存在；最终 wheel 对旧 namespace 必须是物理零文件，而不只是
production 主线不 import。

### 2.3 先固定数据链，再移动文件

每次迁移必须同时写清：

1. 输入和输出类型；
2. 谁构造对象；
3. 谁拥有可变状态；
4. 谁保存/恢复状态；
5. 哪个 identity 进入 recipe/checkpoint；
6. 出错在哪个 gate 被拒绝；
7. 迁移前后数值与 RNG 如何证明等价。

## 3. Flow-Factory 固定源码参考

参考快照固定为官方仓库 commit
[`d21f4bac4467d0b555d9838387022e6e471cf04f`](https://github.com/X-GenGroup/Flow-Factory/tree/d21f4bac4467d0b555d9838387022e6e471cf04f)，
提交时间为 2026-08-01。后续不使用漂移的 `main` 作为架构证据。

### 3.1 直接学习的部分

| Flow-Factory 机制 | 固定源码位置 | VisualRL 决策 |
| --- | --- | --- |
| 每类能力有 registry + loader | `models/registry.py`、`trainers/registry.py`、`scheduler/registry.py`、`rewards/registry.py` | 各域提供无重依赖 descriptor/catalog fragment；通用 registry、总 catalog 和解析唯一归 composition；重资源延迟到 environment gate 后 |
| 模型实现按 family 放置 | `models/stable_diffusion/`、`models/wan/`、`models/flux/` | SD3/Wan 放入 `models/implementations/`，不再为算法复制模型类 |
| Scheduler 与 Model/Trainer 分开 | `scheduler/` | transition/replay 归算法 dynamics；模型只提供不可变 scheduler blueprint |
| Pointwise/Groupwise Reward | `rewards/abc.py`、`reward_processor.py` | 保留 typed reward 分类、row/group 对齐与物理资源去重 |
| 单样本与 batch 分离 | `samples/samples.py` | 单样本记录和 batched trajectory 明确分型，禁止隐式 batch 维度 |
| 六阶段在线 RL 工作流 | `guidance/workflow.md`、`trainers/grpo.py` | 其原始阶段是 preprocessing→K-repeat→trajectory→reward→advantage→optimization；VisualRL 在最后一段内部再显式拆 credit/objective/update |
| 单一 distributed model root | `models/model_bundle.py` | 继续使用唯一 prepared model root，所有 current-policy forward 必须经过它 |

### 3.2 明确不复制的部分

- `models/abc.py` 约 2300 行，同时拥有 pipeline、scheduler、参数 swap、LoRA、checkpoint、offload、
  preprocess 等生命周期职责，并定义抽象 `inference()`；完整 denoising loop 则重复实现在各具体 Adapter。
  VisualRL 必须同时避免 God base class 和 per-model rollout loop 复制。
- `trainers/abc.py` 同时初始化 model、reward、data、optimizer、distributed 和 acceleration；VisualRL
  的 runtime composition 与算法 trainer 不得合并成 God class。
- Flow-Factory 的具体 Adapter 仍拥有完整 denoising loop；VisualRL 的 model 只做单步 `predict()`。
- Scheduler 是 run 级可变对象并持有 seed/mode/cursor；VisualRL 使用每次 rollout 独立、可重放的状态。
- 动态 class path、`filter_kwargs` 和 `extra_kwargs` 不替代 typed contract。
- config 不在运行时按 world size 或对象状态原地修正；resolved recipe 必须不可变。

### 3.3 Flow-Factory 的真实边界与数据链

Flow-Factory 没有独立 `rollout/` 包，也没有同时维护 `models/` 与 `model_adapters/`。它的实际主链是：

```text
YAML -> Arguments -> load_trainer -> Accelerator + ModelAdapter -> BaseTrainer

DataLoader
  -> BaseTrainer.generate_samples
  -> concrete Adapter.inference
       -> [Adapter.forward -> Scheduler.step] x T
       -> BaseSample
  -> RewardBuffer / RewardProcessor
  -> AdvantageProcessor
  -> Trainer.optimize
       -> Adapter.forward
       -> algorithm loss / backward / optimizer.step
```

因此本项目学习它的单一模型 namespace、分域 registry、typed Sample/Reward 和单 distributed root；但不
复制“Adapter 拥有完整 rollout”“Adapter.forward 直接返回 scheduler transition”“一个 Trainer/Adapter
承载多个生命周期域”这些粗粒度耦合。

## 4. 2026-08-03 全包审计快照

当前 `visual_rl` 约 7.8 万行 Python。主要重复域如下：

| 能力 | v0.8 路径 | 旧/重复路径 | 结论 |
| --- | ---: | ---: | --- |
| Model | `models` 约 7.3k LOC | `model_adapters` 约 4.0k LOC | production 已切新路径，但旧代码仍打入 wheel |
| Rollout | `rollouts` 约 3.0k | `rollout` 约 1.3k | 新旧 loop 并存 |
| Reward | `rewards` 约 3.0k | `feedback` 约 2.5k | 新层仍通过 bridge 调旧 client contract |
| Checkpoint | `checkpoint` 约 4.2k | `artifacts` 约 4.2k | v0.7 commit-chain 与 v0.8 checkpoint 并存 |
| Algorithm | 新域合计约 2.3k | `optimizers` 约 2.3k | objective/loss type 仍借旧 namespace |

### 4.1 已完成的安全点

- canonical numerical leaves 已建立：
  - `models/diffusers_common.py`；
  - `dynamics/diffusion_transition.py`；
  - `conditioners/world_r1_camera_math.py`。
- v0.8 production 对 `visual_rl.model_adapters` 的 import 已为 0。
- 该 slice 的 canonical/architecture 85 tests、legacy regression 110 tests、Ruff、compileall 和
  `git diff --check` 已通过。
- slot-streaming 已保存 detached `[B,T] new_log_probs`，每个 slot 单独重算、立即 backward，不跨
  timestep 保存 autograd graph。
- 真实服务器实验已暂停；旧 source-e 的实跑只作诊断，不是最终结构的 release evidence。

### 4.2 必须在最终冻结前解决的 P1

1. **模型仍构造具体 Dynamics replay factory。** `models/sd3.py`、`models/wan.py` 依赖具体
   Dynamics 状态类型；这违反 model→algorithm 单向隔离。
2. **AlgorithmBlueprint 不是编译期唯一来源。** recipe 仍重复声明 trainer/dynamics/rollout/credit，
   compiler 独立解析，直到模型 prepare 后才比较 blueprint。
3. **缺少静态 Model–Dynamics ABI。** 当前主要靠运行时精确 Python replay-state type 相等；必须新增
   可序列化 `scheduler_blueprint_schema`、`dynamics_binding_family`、`replay_state_schema_id`。
4. **`row_microbatch_size` 未真正约束 forward batch。** recompute 为保持低精度数值等价会重放完整
   rollout forward partition；若 partition=8、row slot=1，峰值仍是 8 且计算可放大到 O(B²)。
5. **视频 host memory 瞬时双份。** row-wise `TrajectoryStep` 与最终 `[B,T,...]` collated tensor 同时
   存在；Wan 需要预分配 batched trajectory builder。
6. **Reward 仍经旧 bridge。** `RewardBatchView → RolloutBatch → RewardClient → RewardVector → NumPy`
   多次转换，camera 还发生 FP32→FP64。
7. **运行可观测性不足。** terminal metrics 只有最后一行，没有 slot 数、forward replay 放大、GPU
   allocated/reserved peak、CPU RSS、trajectory bytes 和 partition width。
8. **ownership gate 不覆盖完整主线。** `configs`、`preflight`、`inspection.py` 必须进入 import gate；
   wheel 必须显式拒绝旧 namespace。
9. **`runtime/production.py`、`runtime/checkpointing.py`、`runtime/binding.py` 是大文件。** 它们需要按
   lifecycle、binding、algorithm materialization、checkpoint integration 拆分。
10. **普通 Flow×Wan 泄漏了 World-R1 dynamics 名称。** 当前 `flow_grpo_wan.yaml` 选择
    `kernel_variant=world_r1`，虽然没有 camera conditioner，但这让普通跨模型组合借用了 World-R1 语义。
    最终应提供命名中立的 Wan flow-SDE exact-action kernel；camera hook 只属于 World-R1 route。
11. **Registry 静态解析仍 import/构造 concrete component。** 当前 `ComponentResolver.resolve()` 会加载
    config/component class 并调用 `describe()`，`ComponentLoader` 又与它同包构造实例；必须拆成
    import-safe descriptor/provider 解析与 runtime component loading，否则 artifact gate 前仍会触碰重实现。

### 4.3 计划冻结时的历史本地测试基线

- 计划冻结时以 `PYTHONPATH=. .venv/bin/python -m pytest -q` 运行：
  `1490 passed, 5 skipped, 1 failed`；该结果仅作为历史快照，当前证据以第 15 节为准。
- 唯一失败是新增 `services/world_r1_strict/qwen_loader.py` 后，冻结的 service revision digest 尚未级联；
  这是 M6 final source freeze 的显式待办，不是算法数值失败。
- 直接执行 `.venv/bin/pytest` 会因为项目根未进 `sys.path` 产生 8 个 collection error；这属于调用方式，
  不计为代码回归，但 M6 应固定唯一测试命令。

### 4.4 M2 实施前算法域三路只读审计（2026-08-03）

审计覆盖 `algorithm_modules/algorithms/advantage/optimizers/rollouts/dynamics/conditioners/trainers/training`
及 production compiler/runtime 调用链，未修改源码、未执行真实训练。结论是 M2 必须迁移控制权，不能只
移动目录：

1. **存在三份内部组件事实来源。** `AlgorithmBlueprint`、recipe 的 `components` 和
   `training/algorithm_plan.py` 各自决定 trainer/dynamics/rollout/credit/objective；blueprint 目前只在
   runtime 最后做一致性断言。
2. **新静态 provider 链尚未进入 production。** production compiler 仍使用旧 `ComponentResolver`，在
   artifact/environment gate 前 import config 与具体 implementation 并调用 `describe()`；随后
   `training/assembly.py` 又按 `semantic_config` 重解析并构造组件。
3. **`AlgorithmMaterializationSpec` 尚未被 production 消费。** active `AlgorithmExecutionPlan` 仍从
   `ResolvedRecipe` 重新推导，且复制了 `TransitionSelectionKind`、`ReplayTarget` 等 core 枚举。
4. **optimization 只有一条新主线值得保留。** production 使用新 advantage/credit、slot-streaming
   recompute/update；legacy `optimizers/{grpo,tempflow_grpo,flash_grpo,update_engine}` 无 production
   importer，应删除。仅 `PolicyLossInputs`、clipped-surrogate 和 reference regularizer 数值叶子需要合并
   到新 objective owner。
5. **全 T 兼容入口必须退出 production。** `PolicyRecomputer.compute()` 与
   `PolicyUpdateKernel.step()` 会重新允许一次保留所有 timestep graph；等价 oracle 迁入 tests/native
   harness 后，production 只保留逐 slot recompute→loss→backward。
6. **当前 OOM 修复证据有明确边界。** 逐 slot graph lifetime 已有单测，但 recompute 仍按原始
   `forward_partitions` 重放；`row_microbatch_size=1` 不保证实际 forward batch 为 1。M2 保持数值/RNG
   顺序，M5 再统一 UpdateSlotPlan 与 ModelForwardReplayPlan，M7 用 CUDA profile 验证。

审计时 optimization/trainer 相关 20 个 test 文件基线为 `164 passed`；该结果只证明 CPU fake/contract，
不作为 CUDA OOM 或真实训练证据。

## 5. 最终文件树

顶层只保留七个能力域，加一个薄入口和统一错误：

```text
visual_rl/
├── __init__.py
├── train.py
├── errors.py
├── core/
│   ├── immutable.py
│   ├── serialization.py
│   ├── determinism.py
│   ├── seed.py
│   ├── protocols/
│   │   └── world_r1.py
│   └── contracts/
│       ├── model.py
│       ├── algorithm.py
│       ├── reward.py
│       ├── composition.py
│       └── runtime.py
├── models/
│   ├── interface.py
│   ├── scheduler.py
│   ├── catalog.py
│   ├── preprocessing.py
│   ├── implementations/
│   │   ├── common_diffusers.py
│   │   ├── sd3.py
│   │   └── wan.py
│   ├── lifecycle/
│   │   ├── components.py
│   │   └── prepared.py
│   ├── state/
│   │   ├── parameters.py
│   │   ├── projection.py
│   │   └── io.py
│   └── numerics/
│       ├── runtime.py
│       ├── policy.py
│       └── execution.py
├── algorithms/
│   ├── catalog.py
│   ├── modules/
│   │   ├── interface.py
│   │   ├── descriptor.py
│   │   ├── flow_grpo.py
│   │   ├── tempflow_grpo.py
│   │   └── flash_grpo.py
│   ├── rollout/
│   │   ├── interface.py
│   │   ├── config.py
│   │   ├── request.py
│   │   ├── collector.py
│   │   ├── builder.py
│   │   ├── full_trajectory.py
│   │   ├── branching.py
│   │   └── single_step.py
│   ├── dynamics/
│   │   ├── interface.py
│   │   ├── config.py
│   │   ├── replay.py
│   │   ├── session.py
│   │   ├── selection.py
│   │   ├── transition.py
│   │   ├── flow_schedule.py
│   │   ├── sd3_flow_sde.py
│   │   └── wan_flow_sde.py
│   ├── conditioning/
│   │   ├── interface.py
│   │   ├── config.py
│   │   ├── world_r1_camera.py
│   │   └── camera_math.py
│   ├── rewards/
│   │   ├── interface.py
│   │   ├── types.py
│   │   ├── planning.py
│   │   ├── execution.py
│   │   ├── resource_descriptor.py
│   │   ├── input_selection.py
│   │   └── clients/
│   │       ├── http.py
│   │       ├── mock.py
│   │       ├── image.py
│   │       └── world_r1.py
│   ├── optimization/
│   │   ├── config.py
│   │   ├── advantage.py
│   │   ├── credit.py
│   │   ├── objective.py
│   │   ├── recompute.py
│   │   ├── slots.py
│   │   ├── execution.py
│   │   └── kernel.py
│   └── trainer/
│       ├── interface.py
│       ├── config.py
│       ├── grpo.py
│       ├── stages.py
│       └── execution_plan.py
├── data/
│   ├── catalog.py
│   ├── samples/
│   │   ├── items.py
│   │   ├── trajectory.py
│   │   └── collate.py
│   ├── sources/
│   │   ├── interface.py
│   │   ├── plan.py
│   │   ├── prompt.py
│   │   └── sampler.py
│   ├── preprocess/
│   │   ├── requirements.py
│   │   ├── producer.py
│   │   ├── plan.py
│   │   ├── cache.py
│   │   └── factory.py
│   ├── phase_schedule.py
│   ├── group_placement.py
│   └── prelude.py
├── composition/
│   ├── config/
│   │   ├── source.py
│   │   ├── specs.py
│   │   ├── bootstrap.py
│   │   ├── errors.py
│   │   └── compiler.py
│   ├── recipes/
│   │   ├── schema.py
│   │   ├── builtins.py
│   │   ├── phase_compiler.py
│   │   └── manifest.py
│   ├── registry/
│   │   ├── base.py
│   │   └── catalog.py
│   ├── compatibility/
│   │   ├── selection.py
│   │   ├── resolver.py
│   │   ├── report.py
│   │   └── evidence.py
│   └── preflight/
│       ├── static.py
│       ├── environment.py
│       ├── runtime.py
│       └── artifacts.py
├── runtime/
│   ├── composition.py
│   ├── controller.py
│   ├── lifecycle.py
│   ├── component_graph.py
│   ├── component_loader.py
│   ├── algorithm_binding.py
│   ├── model_binding.py
│   ├── resources.py
│   ├── reward_resources.py
│   ├── preprocess_binding.py
│   ├── checkpoint_binding.py
│   ├── transforms.py
│   ├── probes.py
│   └── types.py
└── artifacts/
    ├── inspection.py
    ├── terminal.py
    └── checkpoint/
        ├── protocol.py
        ├── builder.py
        ├── state.py
        ├── manager.py
        ├── reader.py
        ├── transaction.py
        ├── reference.py
        └── validation.py
```

`services/world_r1_strict/` 仍是仓库级独立服务，不移入训练包。它只共享
`visual_rl.core.protocols.world_r1` 的无重依赖 wire contract。

## 6. 唯一 owner 表

| 能力 | 唯一 owner | 明确禁止 |
| --- | --- | --- |
| 模型插件接口、descriptor 与具体 SD3/Wan | `models` | algorithm 名称、rollout loop、reward/loss/update |
| 模型专属 preprocessing/conditioning 公式 | `models.preprocessing` | dataset traversal、source cursor、cache schema/state |
| Scheduler artifact blueprint | `models.scheduler` | live cursor、SDE step selection、policy log-prob |
| 粗粒度算法蓝图 | `algorithms.modules` | 具体模型 class/path、直接加载 artifact |
| Denoising trajectory 控制 | `algorithms.rollout` | 模型权重生命周期、optimizer commit |
| Transition/replay 数学 | `algorithms.dynamics` | model pipeline loader、跨 rollout 共享 mutable cursor |
| Camera latent 初始化/hook | `algorithms.conditioning` | reward routing、模型架构分支 |
| Reward contract/client/logical routing/resource descriptor | `algorithms.rewards` | 物理 handle 生命周期、advantage、loss、optimizer；旧 RolloutBatch bridge |
| Advantage/Credit/Objective/Update | `algorithms.optimization` | 数据源选择、模型加载、reward 网络生命周期 |
| 六阶段算法循环 | `algorithms.trainer` | composition/preflight/artifact discovery |
| 样本 DTO、trajectory batch、数据源、K-repeat、phase | `data` | algorithm/model 名称分支、rollout control loop |
| Preprocess requirements/plan/batching/cache identity 与 state | `data.preprocess` | 具体模型 encode 公式、prepared model/resource lifecycle |
| 配置、recipe、通用 registry/总 catalog、compatibility、preflight | `composition` | CUDA/model/reward 重资源构造、domain implementation |
| 唯一运行时装配和物理资源生命周期 | `runtime` | 第二入口、第二 controller、业务公式、reward 逻辑路由 |
| Preprocess binding | `runtime.preprocess_binding` | encode 公式、cache schema、数据源遍历；只把 plan/cache handle 与 prepared model callable 注入成 bound port |
| Checkpoint participant collection/safe-point binding | `runtime.checkpoint_binding` | snapshot schema/serialization、live model/algorithm internals；只收集 core participant snapshots 并调用 artifact transaction |
| checkpoint、终态 metrics、inspection | `artifacts` | 训练 loop、模型 forward、reward execution |
| 通用不可变类型、序列化、wire protocol | `core` | 具体模型/算法实现 |
| 跨域 ComponentDescriptor/capability/spec/snapshot DTO | `core.contracts` | Registry 状态、具体 component、live manager/device handle |

`core.contracts.composition.ComponentDescriptor` 是 domain catalog 共享的纯 DTO。各 domain 的
`catalog.py` 只返回该 DTO fragment、implementation class-path 和 import-safe declaration-provider path；
`composition.registry` 唯一拥有 Registry 数据结构、总 catalog、冲突检查与解析。它不 import/构造具体
SD3/Wan 或有重依赖的 runtime component。config 校验与 `DeclaredContract` 由 provider 完成；
`runtime.component_loader` 只在 environment/artifact gate 后 import implementation 并构造实例。

Reward 被放在 `algorithms/rewards` 而不是继续作为顶层，是为了按用户要求把 post-training 能力聚合在
一个大域中；它仍通过窄接口独立于具体 AlgorithmModule。Flow-Factory 将 rewards 作为 peer package，
其“独立能力”原则保留，但不复制其顶层布局。

`data.samples.trajectory` 唯一拥有不可变 `TrajectoryStep/Item/Batch`；`algorithms.rollout.collector/builder`
只拥有 rollout 时的控制策略和可变预分配写入过程，最终产物必须落回 data DTO。类似地，
`algorithms.rewards.resource_descriptor` 只描述逻辑 reward 资源，`runtime.reward_resources` 才拥有实际
client/worker 的 acquire、activate、borrow 和 close。

Preprocess 同样不是三个 owner：`models.preprocessing` 只实现无数据源状态的模型专属
encode/conditioning；`data.preprocess` 拥有 requirement/plan、dataset traversal/batching、cache identity 与
可恢复 state；`runtime.preprocess_binding` 只把 prepared model callable 和 cache/resource handle 注入为
`BoundPreprocessor`。算法 PREPROCESS stage 只调用该 port，不重新实现公式、遍历或 cache。

Capability contract 也不能留在 composition：Model/Algorithm/Dynamics/Rollout/Reward、task/layout/time/
likelihood 等跨域 DTO 和枚举全部归 `core.contracts`。`composition.compatibility` 只做选择、比较、报告和
evidence，不成为任何 domain 的上游依赖。

Checkpoint 使用 participant/snapshot 边界：model、algorithm、data 等 owner 实现 core 中的
`CheckpointParticipant` port，在 safe point 产出纯 immutable snapshot DTO；runtime 只收集这些 snapshot；
`artifacts.checkpoint` 只校验、序列化和原子提交，不接收 live ParameterStateManager、具体算法 state class
或 runtime controller。

## 7. 允许的依赖方向

```text
core
├── data
├── models ──> data import-safe sample/preprocess DTOs
├── algorithms ──> models.interface + data import-safe DTOs
├── composition ──> domain descriptor fragments + import-safe contracts only
└── artifacts ──> immutable snapshot DTOs only

runtime ──> composition + models + algorithms + data + artifacts
train.py ──> runtime.composition only
```

硬性规则：

- `models/**` 对 `algorithms/**` 的静态和动态 import 数必须为 0。
- `models/**` 可以 import `data.samples` 与 `data.preprocess` 的 import-safe DTO，但 `data/**` 不得反向
  import models implementation；跨域 DTO 不得携带 pipeline、scheduler 或 device resource。
- `data/**` 不得 import composition recipe 或 models contract/implementation；compiler 必须先投影 path-free、
  data-owned `SourcePlanSpec`，artifact/environment gate 再产出含 launch-only location 与 expected content
  identity 的 `SourceLocationBinding`，loader 只接收两者组成的 `SourceLoadRequest`；model preprocessing
  declaration 必须投影/声明为 data-owned `PreprocessProducerSpec`。
- `algorithms/**` 对 `models/implementations/**` 的 import 数必须为 0。
- `algorithms/**` 不得读取 `ResolvedRecipe`/`MaterializedRecipe` 或 `semantic_config`；compiler 只传
  core 中的 `AlgorithmMaterializationSpec`、`RewardPlanSpec` 与 algorithm blueprint。
- 只有 `runtime/**` 可以同时构造具体 model 与 algorithm component。
- `composition/**` 可以读取 domain catalog fragment，但不得构造具体 component，也不得 import
  torch/diffusers/transformers/peft/requests。
- `composition.registry` 只能 import descriptor 声明的无重依赖 provider；implementation class path 只能
  由 `runtime.component_loader` 在 gate 后解析。
- `artifacts/**` 不得 import runtime controller 或具体 model/algorithm implementation。
- `artifacts/checkpoint/**` 只接收 core immutable snapshot DTO，不接收 live manager/parameter owner。
- package `__init__.py` 只导出稳定接口，不触发重资源 import。
- 任何跨域类型必须在 source owner 或 `core/contracts` 中定义，不允许为消除循环复制 dataclass。

这些规则由 AST import gate 和 wheel gate 同时执行；不能靠约定。

## 8. 当前文件到目标文件的迁移表

| 当前路径 | 目标路径/处理 |
| --- | --- |
| `core/types.py` | 按 owner 拆为 `core/immutable.py`、`core/contracts/runtime.py`、`runtime/types.py`；其中旧 `RolloutBatch/RewardVector` 在 M3 删除 |
| `core/components.py` | 旧纯 descriptor 语义迁/更名为 `core/contracts/composition.py:ComponentDescriptor`；重复 `ComponentSpec` 删除，Registry 状态仍只在 composition |
| `contracts/model_algorithm.py` | capability/binding/transition/port 分别迁 `core/contracts/{model,algorithm,runtime}.py`，随后删除顶层 `contracts` |
| `algorithm_modules/base.py` | `algorithms/modules/interface.py` |
| `algorithm_modules/components.py` | import-safe blueprint/descriptor 与 runtime materializer 拆入 `algorithms/modules/{descriptor,flow_grpo,tempflow_grpo,flash_grpo}.py` |
| `algorithms/components.py`、`algorithms/credit.py` | `algorithms/optimization/credit.py`，配置跟随对应 module |
| `algorithms/objective.py` + `optimizers/{objective,clipped_surrogate}.py` | 合并为唯一 `algorithms/optimization/objective.py` |
| `advantage/processor.py` + legacy `optimizers/advantages.py` | 保留新实现并迁入 `algorithms/optimization/advantage.py`；删除旧实现 |
| legacy `optimizers/{base,grpo,tempflow_grpo,flash_grpo,update_engine}.py` | canonical 数值/transaction fixture 迁完后删除，不形成第二算法实现 |
| `rollouts/*` | `algorithms/rollout/*`；`base.py` 拆 interface/collector/builder，immutable trajectory DTO 不在此重复定义 |
| legacy `rollout/*` | 数值/fixture 迁完后物理删除，不打包 facade |
| `dynamics/*` | `algorithms/dynamics/*`；`replay.py` 拆 model blueprint 与 algorithm replay 职责 |
| `conditioners/*` | `algorithms/conditioning/*` |
| `rewards/{types,planning,input_selection,processor,stage}.py` | 迁 `algorithms/rewards/{types,planning,input_selection,execution}.py`；planning 改为只消费 compiler 产出的 core `RewardPlanSpec`，不读取 recipe |
| `rewards/pool.py`、`rewards/components.py` | logical descriptor/port 归 `algorithms/rewards`；物理 acquire/activate/close 归 `runtime/reward_resources.py` |
| `feedback/{clients,image_rewards,world_r1_rewards}.py` | 改成原生 typed client 后迁入 `algorithms/rewards/clients/*` |
| `feedback/{base,cache,executor,provider}.py` | 无生产 owner；新 reward tests 覆盖后删除 |
| `trainers/*` | `algorithms/trainer/*` |
| `training/algorithm_plan.py` | `algorithms/trainer/execution_plan.py`；只消费 core `AlgorithmMaterializationSpec`/blueprint，不读取 recipe/component graph |
| `training/{policy_recompute,update_slots,update_execution,update_kernel}.py` | `algorithms/optimization/*` |
| `training/rollout_request.py` | `algorithms/rollout/request.py` |
| `training/stages.py` | `algorithms/trainer/stages.py` |
| `training/data_prelude.py` | `data/prelude.py` |
| `training/assembly.py` | `runtime/component_graph.py` |
| `training/ports.py` | 按 owner 拆入 runtime binding 与 algorithm trainer，不保留通用杂物包 |
| `training/types.py` | `runtime/types.py` |
| `registry/interfaces.py` | 按 owner 拆入 `models/interface.py` 及各 `algorithms/*/interface.py`，不保留跨域万能接口包 |
| `registry/models.py` | 纯 descriptor fragment 迁 `models/catalog.py` |
| `registry/{algorithms,dynamics,rollouts,conditioners,rewards,credits,trainers}.py` | descriptor fragments 迁 `algorithms/catalog.py` 及相应子域；总 catalog 只在 composition 汇总 |
| `registry/base.py` | 纯 `ComponentDescriptor` 迁 core；Registry/alias/冲突/静态 provider 解析迁 composition；implementation import/类型校验/实例构造迁 `runtime/component_loader.py` |
| `models/base.py` | `models/interface.py`；`BaseAdapter` 正式更名为 `ModelAdapter`，不保留双名 |
| `models/{sd3,wan}.py` | `models/implementations/{sd3,wan}.py` |
| `models/_common.py` | artifact/runtime loader 迁 implementations common；descriptor/config helper 迁 `models/catalog.py` |
| `models/diffusers_common.py` | `models/implementations/common_diffusers.py` |
| `models/{component_manager,prepared_bundle}.py` | `models/lifecycle/*` |
| `models/{parameter_state,state_projection,state_io}.py` | `models/state/*` |
| `models/{numerics,numerics_policy,execution_policy}.py` | `models/numerics/*` |
| `models/preprocess.py` | `models/preprocessing.py` |
| legacy `model_adapters/*` | native parity fixture 迁移后全部删除 |
| `samples/{schema,topology,trajectory,collate}.py` | 按类型迁 `data/samples/{items,trajectory,collate}.py`；immutable `BranchTopology` 明确归 `data/samples/trajectory.py`，algorithm rollout 只消费它 |
| `datasets/prompt_dataset.py` | 无 production owner，旧测试归档后删除；不把旧 dataset 栈带入目标树 |
| `data/{source_loader,source_sampler}.py` | compiler 先把 source selection 投影为 path-free `SourcePlanSpec` 并纳入 resolved identity；artifact/environment gate 再产出 `SourceLocationBinding`，loader 消费 `SourceLoadRequest(plan, locations)` 并复核 expected content identity；删除当前混合语义与 Path 的 `SourceLoadPlan`；实现拆入 `data/sources/{plan,interface,prompt,sampler}.py`，descriptor fragment 迁 `data/catalog.py` |
| `data/preprocess*.py` | 迁 `data/preprocess/*`；去除 `ModelPreprocessSpec` import，统一消费 data-owned `PreprocessProducerSpec` |
| `data/phase_compiler.py` | `composition/recipes/phase_compiler.py`；`data/phase_schedule.py` 只保留纯 immutable schedule/state |
| `configs/{bootstrap,compiler,errors,loader,specs}.py` | `composition/config/*`；`loader.py` 更名 `source.py` |
| `configs/manifest.py` | immutable recipe payload 归 `composition/recipes/manifest.py`；文件写入/resume audit 归 `artifacts/terminal.py` |
| legacy `configs/{resolver,schema}.py` | 删除；不得进入 v2 compiler |
| `recipes/*` | `composition/recipes/*` |
| `compatibility/contracts.py` | Model/Algorithm/Dynamics/Rollout/Reward capability DTO 与所有共享枚举必须迁 `core/contracts/*`，不是可选下沉 |
| `compatibility/{resolver,report,evidence}.py` | `composition/compatibility/*`，只保留选择/比较/报告/evidence |
| `compatibility/matrix.py` | evidence matrix 合入 `composition/compatibility/evidence.py` |
| `compatibility/gate_runner.py` | 可复用 gate/evidence 逻辑合入 `composition/compatibility/evidence.py`；删除包内 `main`，不保留 CLI/第二入口 |
| `preflight/{static,environment,runtime,artifacts}.py` | `composition/preflight/*`；删除 v0.7 helper |
| `preflight/types.py` | `RuntimeFacts` 等跨域 DTO 迁 `core/contracts/runtime.py`，preflight-only result 留 composition |
| `execution/resource_plan.py` | 删除 re-export facade；canonical 类型归 `models/lifecycle/components.py` |
| `execution/transforms.py` | immutable `ExecutionTransformPlan` 明确归 `core/contracts/runtime.py`；实际 executor 唯一归 `runtime/transforms.py` |
| `checkpoint/{protocol,builder,state,manager,reader,reference_state}.py` | 只保留纯 snapshot validation/serialization/read，迁 `artifacts/checkpoint/{protocol,builder,state,manager,reader,reference,validation}.py` |
| `checkpoint/coordination.py` | participant capture、distributed safe-point collection 迁 `runtime/checkpoint_binding.py` |
| `checkpoint/coordinator.py` | 只接收 immutable snapshots 的 atomic/two-phase commit 迁 `artifacts/checkpoint/transaction.py` |
| legacy `artifacts/{checkpoint,manager,audit,status,preview,builder,manifest}.py` | v0.7 行为不迁移，物理删除 |
| `artifacts/serialization.py` | `core/serialization.py` |
| `inspection.py` | `artifacts/inspection.py` |
| `runtime/production.py` | algorithm materializer/PerRolloutDynamics bind 归 `runtime/algorithm_binding.py`；model numerics 归 `runtime/model_binding.py`；optimizer/LR 构造归 `algorithms/optimization/execution.py`；trainer stage 归算法域；session lifecycle/DTO 归 runtime 对应文件 |
| `runtime/checkpointing.py` | participant collection/safe-point orchestration 归 `runtime/checkpoint_binding.py`；纯 snapshot validation/transaction 归 artifacts；terminal writer/finalizer 归 `artifacts/terminal.py` |
| `runtime/binding.py` | overall component graph/evidence 归 `runtime/component_graph.py`；model/preprocess 归对应 binding；reward acquire/bind 归 `runtime/reward_resources.py`；algorithm plan/materialization 归 `runtime/algorithm_binding.py` |
| `runtime/defaults.py` | Accelerate/session lifecycle 归 `runtime/lifecycle.py`，公开 DTO 归 `runtime/types.py` |
| `runtime/resources.py` | session container 留 `runtime/resources.py`；reward descriptor/port 下沉算法域，物理 handle lifecycle 迁 `runtime/reward_resources.py` |
| `runtime/policy_port.py` | concrete port 归 `runtime/model_binding.py`，协议下沉 `core/contracts/runtime.py` |
| `runtime/preprocess_requirements.py` | requirement DTO 归 `data/preprocess/requirements.py`；bound compiler 归 `runtime/preprocess_binding.py` |
| `runtime/reward_acquisition.py`、`runtime/reward_factory.py` | logical descriptor 归 algorithms；物理 factory/acquisition/activation 合并为 `runtime/reward_resources.py` |
| `runtime/launch_audit.py` | immutable launch evidence 归 `artifacts/terminal.py`，runtime 只提供实际 facts |
| `runtime/stage_assembly.py` | rollout/reward/advantage/optimize stage wiring 分迁相应 algorithms 子域；runtime 只注入 bound ports/resources |
| `runtime/{composition,controller,probes}.py` | 保留为唯一 composition root/controller/probe owner，移除被拆走的业务公式与旧 import |
| `builtins.py` | 删除；所有 v2 descriptor 只有一个 registry owner |
| `world_r1_protocol.py` | `core/protocols/world_r1.py` |
| `errors.py`、`core/{determinism,seed}.py` | 保留 canonical owner，只更新迁移后的 import path |
| 所有 `__init__.py` | 只导出所属域稳定 interface；根 `__init__.py` 不再暴露 legacy API/namespace |

## 9. 三个关键语义重构

### 9.1 Model–Dynamics 边界

当前错误链路：

```text
SD3Adapter/WanT2VAdapter
  -> 构造具体 SD3/Wan DynamicsReplayStateFactory
  -> Wan 还主动构造 ConditionerLatentSpec
  -> runtime 再检查 factory type 是否匹配 Dynamics component
```

目标分成 load 前静态合约和 load 后 artifact 精确绑定两层：

```text
Model domain catalog (artifact load 前)
  -> ModelDescriptorContract(
       scheduler_blueprint_schema,
       dynamics_binding_family,
       schedule_coordinate,
       accepted_replay_state_schema_ids,
       task/layout/conditioning capabilities)

Algorithm descriptor/blueprint (artifact load 前)
  -> DynamicsRequirement(family, likelihood, stochastic/replay requirements)

Composition compatibility resolver
  -> 静态拒绝不兼容 Model x Algorithm x Dynamics

ModelAdapter.load_components() (artifact load 后)
  -> SchedulerArtifactBlueprint(
       schema, scheduler class, exact config hash, artifact identity)
  -> ModelScheduleContext(latent geometry, patch/temporal semantics)

Runtime
  -> 选择已解析的 Dynamics descriptor
  -> Dynamics descriptor.bind(SchedulerArtifactBlueprint, ModelScheduleContext)
  -> PerRolloutDynamicsFactory

Conditioner/runtime
  -> 从通用 ModelScheduleContext/latent geometry 创建 conditioner 输入
  -> model 不 import 或构造算法侧 conditioner 类型
```

load 前 `ModelDescriptorContract` 新增：

- `scheduler_blueprint_schema`；
- `dynamics_binding_family`；
- `schedule_coordinate`；
- `accepted_replay_state_schema_ids`（声明可由哪类 Dynamics binder 产生，不是 Python class path）。

load 后 `SchedulerArtifactBlueprint` 再携带实际 scheduler class/config digest/artifact identity。运行时仍保留
精确 type 检查作为第二道防线，但静态 ABI 不再依赖加载权重后才发现。

### 9.2 AlgorithmBlueprint 成为编译期唯一来源

Source YAML 只选择 `algorithm`、`model`、integration policy 和 execution resource policy；
trainer/dynamics/rollout/credit/objective 不再作为平行自由覆盖项。配置所有权固定为：

| 配置 | 唯一 owner | 可以表达 | 禁止表达 |
| --- | --- | --- | --- |
| `algorithm.params` | `algorithms.modules.descriptor` 的 frozen config | beta、算法步数/选择规则、branch topology、算法 Dynamics 要求、advantage/credit/objective 数值参数 | 具体模型 class/path、任意内部 component id、group/batch/resource geometry |
| `model.params` | `models.catalog` 的 frozen config | artifact ref、模型 topology/LoRA/encode/decode 固有参数 | rollout/credit/算法名称 |
| integration policy | composition recipe | likelihood 语义、conditioner、reward/data/phase route | 替换 trainer/rollout/credit/objective；用 recipe/model 名分支选择实现 |
| `execution`/`training` | composition schema | precision、group size、global prompt batch、forward/decode/storage/recompute geometry、optimizer/LR/run steps | beta、branch count、伪造 decoded media layout 或其他算法/模型事实；改写 component alias |

Flow-GRPO config 负责 full-trajectory steps、SDE noise requirement、beta 及 GRPO credit 数值；TempFlow config
负责 steps、branch count/topology/selection，compiler 必须要求 execution group size 与 branch count 相等；Flash
config 负责 steps、候选
timestep 选择与 Flash credit 数值。World-R1 recipe 仍选择 Flow-GRPO，但通过其公开 typed config 设置
50-step/credit 参数，并通过 integration policy 增加 camera conditioner、reward/data/phase；普通 Flow×Wan
不出现 `world_r1` kernel 字符串。forward/decode microbatch 与 storage device 是 execution resource policy，
不进入模型 config，也不允许借此替换 rollout implementation。

这里的 Flow/TempFlow “SDE requirement” 指 stochastic/replay/branchability 等模型中立 capability；SD3
方程特有的 `noise_level=0.7` 由 `algorithms.dynamics.FlowSDEConfig` 唯一拥有，不能泄漏进 model config，
也不能假装可直接投影到没有同义参数的 Wan 方程。TempFlow 公式中乘 transition std 的 `2.25` 则是
credit 数学，必须由 `TempFlowGRPOAlgorithmConfig.transition_noise_scale` 经 blueprint/credit provider 进入
identity，并在 M2.2 由 canonical credit runtime 消费，禁止继续作为隐藏 class constant。

```text
composition registry
  -> resolve import-safe Algorithm declaration provider

Algorithm declaration provider parses frozen config
  -> describe_requirements(config)
  -> describe_blueprint(config)
  -> trainer default
  -> rollout default
  -> credit default
  -> objective default
  -> dynamics requirement (model-bound)

compiler
  -> materialize internal descriptors from blueprint
  -> merge only schema-declared integration overrides
  -> run Model × Algorithm × Dynamics compatibility

runtime (artifact/environment gate 后)
  -> RuntimeComponentLoader import implementation_class_path
  -> construct AlgorithmModule
  -> verify component/blueprint/requirement/spec identities
  -> materialize bound algorithm
```

- `AlgorithmSlotBlueprint` 必须携带 canonical frozen slot params，而不只是 component id/family；其参数
  只能由算法 config 与 typed execution/integration projection产生。compiler 不再回读 recipe 的内部
  `components` 来补缺。
- compiler 不构造 concrete `AlgorithmModule`，也不 import torch/模型实现；它只调用 domain catalog 中的
  无重依赖 declaration provider/blueprint factory。provider 必须与 runtime implementation 分文件。
- `ResolvedComponentIdentity` 必须锁定 alias、implementation path、declaration-provider path、config-type
  path、interface version、dependencies、frozen config 和 declared contract；resolver 校验 provider 的
  `CONFIG_TYPE_PATH` 与真实 config type 一致。任一项漂移都改变 resolved/checkpoint identity。
- compiler 将 blueprint 与 model/dynamics/integration/execution 投影为唯一
  `AlgorithmMaterializationSpec`；`ExecutionPolicySpec` 是 completion group size、rollout microbatch、storage、
  precision 与 transform plan 的唯一数值 owner，materialization spec 只保存完整 `execution_policy_id`，不得
  复制 `group_size`、`training_paradigm` 或 `execution_transform_plan_id`。
  `AlgorithmExecutionPlan.from_spec(spec, execution_policy)` 必须先验证该 identity，再从 core-owned typed
  execution-policy view 投影 paradigm 与运行期 cardinality；它不再读取
  `ResolvedRecipe`、`MaterializedRecipe` 或 `semantic_config`，也不复制 core enums。
- 删除 `AlgorithmModule.bind(policy, arbitrary_callbacks)`；只保留 `materialize()`。
- `AlgorithmModule` 不得 import composition 或自行调用 compatibility resolver；runtime 传入已经验证的
  `ModelAlgorithmBinding`，module 只核验 identity/contract。
- recipe 不重复写算法内部默认 ID；出现 `components.trainer/rollout/credit/objective` override 时在静态
  compile 结构化拒绝，不做静默忽略。
- World-R1 的 data source、phase、camera conditioner、reward route 属于 integration recipe，不伪装为
  新 optimizer。

### 9.3 Slot-streaming 与真实 forward partition 对齐

必须继续保存完整 detached `new_log_probs[B,T]`，但每个 slot 的 graph 只活到当前 backward：

```text
for forward_partition in rollout_replay_plan:
    for transition_window in partition:
        recompute only this partition/window
        write detached summary cells
        compute scaled loss for active cells
        backward immediately
        release graph
one finite/nonzero/clip gate
one optimizer/scheduler/reference commit
```

约束：

- `forward_microbatch_size` 是真实模型 forward 上限；`row_microbatch_size` 不得小于/切碎一个不可再分
  的 forward partition 后又重复整 partition。
- `UpdateSlotPlan` 直接引用 `ModelForwardReplayPlan.forward_partitions`。
- 记录 `forward_calls`、`forward_row_equivalents`、`replay_amplification`、slot 数和每 slot wall time。
- Wan trajectory 使用预分配 CPU batched builder，避免 row list + final tensor 双份峰值。
- 不缓存带 graph 的 new log-prob/prediction 来换速度。

## 10. 通用真实训练数据链

```text
G0 Source config
  -> composition.config.source
G1 Algorithm blueprint + domain registries
  -> immutable ResolvedRecipe
  -> AlgorithmMaterializationSpec / RewardPlanSpec / path-free SourcePlanSpec
  -> static Model/Algorithm/Dynamics/Reward compatibility
G2 Artifact/environment preflight
  -> model/dataset/reward identities
  -> SourcePlanSpec + launch-only locations + expected content identities
  -> SourceLocationBinding -> SourceLoadRequest
  -> immutable MaterializedRecipe
G3 Runtime construction
  -> ModelAdapter + scheduler blueprint
  -> compatible Dynamics binder
  -> reward resources ACQUIRED (not ACTIVE)
  -> one PreparedModelBundle
G4 Algorithm materialization
  -> data prelude
  -> rollout request factory
  -> rollout/dynamics/conditioner
  -> reward/advantage/credit/objective/update stages
G5 Iteration
  -> phase/source selection before K-repeat
  -> raw rows -> model preprocessing payload
  -> K-repeat/group placement
  -> latent initialization
  -> rollout trajectory + media
  -> typed reward outputs
  -> normalized advantage
  -> PolicyLossInputs
  -> slot-streaming recompute/backward
  -> one optimizer commit
G6 Persistence
  -> each domain CheckpointParticipant captures immutable snapshot
  -> runtime collects at post-commit safe point
  -> model/optimizer/LR/RNG/data/phase/algorithm/dynamics state
  -> atomic checkpoint + terminal metrics + inspection
```

关键 payload：

| 边界 | 输入 | 输出 |
| --- | --- | --- |
| Data→Model | typed sample batch | model-specific conditioning payload + identity |
| Model→Rollout | latent spec, conditioning, parameter view | one-step `ModelPrediction` |
| Model→Dynamics bind | scheduler blueprint + schedule context | per-rollout replay factory |
| Rollout→Reward | row identities, prompt, image/video, optional camera | `RewardBatchView` |
| Reward→Advantage | row-aligned reward vectors + route | normalized advantage |
| Credit→Objective | old log-prob, advantage, masks, metadata | `PolicyLossInputs` |
| Recompute→Update | one slot differentiable new log-prob | loss/backward + detached summary |
| Domain→Runtime | checkpoint participant at post-commit safe point | immutable owner snapshot |
| Runtime→Checkpoint | tuple of immutable snapshots | validated committed checkpoint tree |

## 11. 五种组合在本 infra 中的真实链路模拟

### 11.1 组合总表

这里 `P=global_prompt_batch_size`、`K=group_size`、`B=P×K`。下表写的是最终应冻结的语义；普通
Flow×Wan 当前配置中误用的 `world_r1` kernel 名称必须在 M2 改正。

| 组合 | Model | Dynamics | Rollout | Reward | Credit/Objective | 关键 ABI |
| --- | --- | --- | --- | --- | --- | --- |
| Flow × SD3 | SD3, BCHW, T2I | SD3 flow-SDE | P=1,K=8,B=8；full T=28 | image quality | GRPO + clipped + beta/ref | exact env action、reference view |
| Flow × Wan | Wan, BCTHW, T2V | 命名中立 Wan flow-SDE，无 camera hook | P=1,K=4,B=4；full T=28 | video general | GRPO + clipped, beta=0 | exact env action、无 reference |
| TempFlow × SD3 | SD3, BCHW, T2I | branchable SD3 flow-SDE | P=2,K=6,B=12；physical T=28，stored T=27 | image quality | TempFlow branch credit + clipped | shared prefix/branch identity |
| Flash × Wan | Wan, BCTHW, T2V | Wan flash kernel | P=2,K=4,B=8；physical T=40，stored T=1 | video general | Flash rectified credit + clipped | rectification metadata、exact pre-hook |
| World-R1 × Wan | Wan, BCTHW, T2V | Wan world-r1 kernel, post-hook surrogate | P=2,K=4,B=8；full T=50 + camera hooks | general + 3D | GRPO + clipped, beta=0 | sampled action与conditioned next分离 |

### 11.2 Flow-GRPO × SD3.5

```text
flow_grpo_v1
 -> AlgorithmModule: flow-grpo(beta=0.004)
 -> Model: sd3
 -> Dynamics binder: flow-sde(noise_level=0.7)
 -> Rollout: full trajectory, 28 steps
 -> group K=8
 -> image reward
 -> GRPO group advantage / exact old-new log-prob ratio
 -> reference replay + KL when beta>0
 -> 28 current-policy slots + 28 no-grad reference slots
 -> detached summary covers 224 active cells at B=8,T=28
```

静态门禁要求 BCHW、flow prediction、fractional timestep、reference policy、replayable exact-action
log-prob；`branchable` 只属于 TempFlow，不是普通 Flow 的要求。SD3 model 不能包含 Flow-GRPO 分支。

### 11.3 Flow-GRPO × Wan2.1

```text
flow_grpo_v1 integration override
 -> 同一个 FlowGRPOAlgorithmModule 源码
 -> Model: wan-t2v
 -> compatible、命名中立的 Wan flow-SDE binder
 -> full trajectory, CPU trajectory staging
 -> group K=4, beta=0
 -> remote video-general reward
 -> GRPO credit/objective/update 与 SD3 路径相同
```

这是跨模型解耦证据，不存在可冒充的 upstream native reference。Wan 无 reference view，因此 beta 必须
静态为 0。当前配置的 `kernel_variant=world_r1` 必须在冻结前移除；没有 conditioner 的普通 Flow route
不能借用 World-R1 命名。模型替换不能修改 Flow algorithm 源码。

### 11.4 TempFlow-GRPO × SD3.5

```text
tempflow_grpo_v1
 -> TempFlow algorithm blueprint
 -> SD3 + branchable flow-SDE
 -> every-policy-timestep shared-prefix branching
 -> P=2, branch_count/group K=6, B=12
 -> physical T=28, stored policy transitions=27
 -> terminal image reward
 -> branch identity preserved through reward and credit
 -> RewardResult[B,27], axis=branch_timestep
 -> TempFlow credit -> shared clipped objective -> 27 streaming slots
```

recompute 必须使用 rollout 时实际 leader-expanded conditioning；RNG 顺序、branch topology、old/new
log-prob 逐 branch 对齐进入 native parity。

### 11.5 Flash-GRPO × Wan2.1

```text
flash_grpo_v1
 -> Flash algorithm blueprint
 -> Wan flash dynamics
 -> P=2,K=4,B=8; generate 40-step continuation
 -> keyed-uniform select one policy timestep in [0,10]
 -> trajectory stores exactly one policy transition
 -> remote general reward
 -> rectification coefficient metadata
 -> Flash credit -> shared clipped objective
```

SD3 缺少 Flash 所需 rectification metadata，必须在 static preflight 失败。不能靠运行时 model-name
白名单。

### 11.6 World-R1 × Wan2.1

```text
world_r1_core/release recipe
 -> Flow-GRPO algorithm(beta=0)
 -> P=2,K=4,B=8; Wan world-r1 dynamics, 50 steps
 -> camera prompt parse once
 -> camera-warped initial latent
 -> early-step decaying camera delta hook
 -> trajectory同时保存 sampled_action 与 conditioned_next_state
 -> decode 81-frame video
 -> general reward + camera-aware 3D reward
 -> weighted sum (current defaults 1 + 1)
 -> GRPO [B,50] -> 50 streaming slots -> one optimizer commit
```

- core/release-surrogate 使用 `post_hook_base_density_surrogate`；不能标成 exact environment action。
- release phase：optimizer step 0–99 选择 main source + general/3D；100–149 选择 dynamic source + general；
  然后周期重复。
- 普通 20-step 只覆盖 main phase，因此最终除 20-step 外还要做至少 101-step phase-boundary 验证或完整
  150-step cycle；两者证据不能混称。

### 11.7 六条 route 的 G0–G6 可执行差异矩阵

六条 route 共享第 10 节的 typed payload 链；下表列出每条 route 必须实际注入的差异，不能用算法摘要
替代 G0–G6 验证。

| Route | G0–G2 config / static / artifact | G3–G4 exact bind | G5 payload、reward、update | G6 safe-point state / evidence |
| --- | --- | --- | --- | --- |
| `flow_grpo_sd3.yaml` | `flow_grpo_v1`；prompt-image + SD3 + local image reward；BCHW/flow/fractional/reference 静态门 | SD3 Adapter → concrete scheduler artifact blueprint → SD3 flow-SDE binder；一个 prepared root | `T2IItem→StackedSampleBatch[B=8]→SD3Conditioning(sd3_prompt_embeddings.v1)→Trajectory[B,28]→Reward[B]→PolicyLossInputs[B,28]`；28 current + 28 no-grad reference slots，一次 commit | commit 后保存 model/optimizer/LR/RNG/next data cursor/dynamics selection/progress；20-step、resume、native parity、memory metrics |
| `flow_grpo_wan.yaml` | `flow_grpo_v1` integration override；prompt-video + Wan + remote general reward；BCTHW/flow/no-reference，beta=0 | Wan Adapter → Wan scheduler artifact blueprint → **generic** Wan flow-SDE binder；无 conditioner | `T2VItem→Batch[B=4]→WanConditioning(wan_prompt_embeddings.v1)→Trajectory[B,28]→Reward[B]→LossInputs[B,28]`；28 current slots | 同一 safe-point schema；20-step、resume 可选、无伪 upstream parity；记录 CPU trajectory 与 remote reward facts |
| `tempflow_sd3.yaml` | `tempflow_grpo_v1`；P=2,K=6；要求 branchable/deterministic-continuation/std-dev metadata | SD3 artifact blueprint → branchable flow-SDE；branch topology 与 RNG identity 一起 bind | shared-prefix leader mainline 后扩展 B=12；physical 28/stored 27；`Reward[B,27]→TempFlow credit(transition_std_dev×2.25)→27 slots` | 保存 branch topology/selection RNG/data cursor；20-step、resume 可选、native parity 和 replay-amplification |
| `flash_wan.yaml` | `flash_grpo_v1`；P=2,K=4；要求 rectification metadata、deterministic continuation | Wan artifact blueprint → flash dynamics；single-step keyed selection policy bind | B=8；40 physical steps，窗口 `[0,10]` 只存 1 transition；`Reward[B]→Flash credit→LossInputs[B,1]`；1 slot | 保存 selected timestep policy/RNG；20-step、native parity、selection histogram 与 one-slot memory evidence |
| `world_r1_core_wan.yaml` | `world_r1_core_v1` + Flow-GRPO beta=0；main camera prompts；remote general+3D artifacts | Wan artifact blueprint → World-R1 surrogate dynamics；camera conditioner 从通用 latent geometry bind，不由 model 构造 | B=8,T=50；camera FP32 `[B,F,4,4]`；warped init + decaying hooks；sampled action/conditioned next 分离；general+3D→LossInputs[B,50] | safe-point 不保存 in-flight trajectory，只保存 next iteration state；独立 20-step route evidence |
| `world_r1_release_surrogate_wan.yaml` | 与 core 同组合，但含 main/dynamic 两 source 和 100/50 optimizer-step phase schedule | 同一 model/dynamics/conditioner bind；phase route 在 K-repeat 前选择 | main: general+3D；dynamic: general；B=8,T=50；每次 50 slots/one commit | 保存 phase/cycle/data cursor；至少 20-step（可由更长 run 覆盖）、World resume、native parity，且另做 >=101、优先 150-step phase-boundary evidence |

当前 checkpoint 的恢复语义是“从下一次完整 iteration 开始”：只在 optimizer commit 后落盘，不保存
in-flight rollout trajectory 或中间 replay state。重构后继续维持这一语义，不伪装成 timestep-level resume。

## 12. 实施顺序

### M0：计划与架构门禁（完成，2026-08-03）

- [x] 全包 LOC、目录和 import graph 扫描。
- [x] 固定 Flow-Factory commit 并审阅 registry/loader/model/scheduler/reward/trainer/workflow。
- [x] production numerical leaf 脱离旧 `model_adapters`。
- [x] 本文件通过两轮独立人工复核，目标树、owner、依赖矩阵和六 route 数据链无文档级阻断。
- [x] ownership gate 覆盖整个 `visual_rl`、根入口和动态 class path。
- [x] 建立 `composition/config`、`composition/registry`、`core/contracts` 的 import-safe 目标壳层。
- [x] 将 Registry 静态 provider resolution 与 runtime implementation loading 分开；M0 先固定 port/gate，
  具体 SD3/Wan provider 在 M1 与 model implementation 同步拆分。
- [x] 先拆清跨域 DTO：允许 `models→data` 的纯 sample/preprocess DTO；跨域 Model/Dynamics/Runtime
  port 归 `core/contracts`，仅域内私有 port 归 source owner，禁止用重复 dataclass 消循环。
- [x] 在 core 固定 `ComponentDescriptor`、全部 capability 枚举、`AlgorithmMaterializationSpec`、
  `RewardPlanSpec`、`ExecutionTransformPlan` 与 checkpoint participant/snapshot contract。

退出：计划中不存在未归属的当前生产文件；所有移动都有目标和验证方式；后续不会先在旧 compiler/
registry 实现一遍再搬家。

完成证据：M0 contracts、config/registry shell、runtime loader、data-owned preprocess/source DTO、ownership
和 wheel gate 聚焦矩阵 `172 passed`；fresh-process import gate 未加载
`torch/diffusers/transformers/peft/requests/accelerate`。真实训练继续暂停。

### M1：模型域收敛与 Model–Dynamics 解耦（完成，2026-08-03）

- [x] 建立 load 前 `ModelDescriptorContract` 和 load 后
   `models.scheduler.SchedulerArtifactBlueprint` 两层 ABI。
- [x] 从 `dynamics/replay.py` 拆出纯 scheduler artifact DTO；algorithm replay request 不得回流 models。
- [x] SD3/Wan 只保存 blueprint/context，不构造具体 replay factory。
- [x] runtime + dynamics descriptor 完成 typed bind，并保留精确 runtime type 二次检查。
- [x] 同时移除 Model→Conditioner：conditioner 从通用 latent geometry/context 构造自身输入。
- [x] 模型 preprocessing 只暴露 data-owned producer spec 与 encode callable，不拥有 source/cache state。
- [x] `BaseAdapter` 正式更名为 `ModelAdapter`；模型文件物理拆入
  `implementations/lifecycle/state/numerics`，不保留旧名或 flat-path facade。
- [x] 模型侧测试、canonical native-parity harness 与 fixture 已迁到 v0.8 public ports；
  旧 native runner/test 和 `model_adapters` 已物理删除。CPU executor 的 14 项结果仅标记为
  `cpu_fake_contract`，不计作 pinned upstream native parity。
- [x] wheel/source gate 永久禁止 `visual_rl.model_adapters`；已知旧成员和未来新增成员都会被拒绝。
- [x] 完成 SD3/Wan import-safe declaration provider 与 model catalog fragment：
  `models/catalog.py` 只提供 core-owned descriptor/provider declaration，不拥有 `_MODELS`、
  `Registry` 或 `ComponentSpec`；静态解析不得 import `models.implementations`，具体实现只能由
  runtime loader 在 environment/artifact gate 后加载。迁移期 legacy registry adapter 如仍需要，
  必须留在 registry/composition 一侧并在 M4 删除，不能回流 models。

当前证据：Model–Dynamics static/load-after ABI、typed runtime binder、conditioner/preprocess ownership
已经落地；模型物理迁移、生命周期、canonical native contract、wheel 和 architecture 聚焦矩阵
`105 passed`，provider/registry/model/architecture/wheel 聚焦矩阵 `82 passed`，config-path
recipe/cache/checkpoint release-cut 聚焦矩阵 `62 passed`。精确 Phase A Ruff、models compileall、
`git diff --check` 与全量 `1431 passed, 4 skipped` 均通过。`BaseAdapter`、旧 flat model path、旧 native
runner 和 `visual_rl.model_adapters` 源码均已消失。canonical CPU executor 覆盖 14 项 comparison，但
证据范围严格为 `cpu_fake_contract`；真实 `pinned_upstream_native` 仍属于 M6–M7。

退出：models 对 algorithms import 为 0；SD3/Wan implementation 不包含
Flow-GRPO/TempFlow-GRPO/Flash-GRPO/World-R1 module 或 recipe 分支（`sd3.flow-sde.v1`、
`wan.flow-sde.v1` 这类 Model–Dynamics ABI family id 不视为算法泄漏）；静态与 load-after
Model–Dynamics ABI 正反例通过；model catalog 不拥有 Registry 状态，fresh-process declaration
resolution 不加载 heavy dependency 或 `models.implementations`；旧模型 namespace 在源码和 wheel
中均不存在。

不属于 M1 的债务：算法 namespace 合并、AlgorithmBlueprint 唯一 compiler 来源、rollout/dynamics/
conditioning/optimization/trainer 物理迁移、algorithm class-path identity 和 Wan 中立 kernel 归 M2；
composition compiler 对 legacy compatibility/recipes/registry 的消费、data/preprocess 目录拆分、
checkpoint/artifacts 合并、runtime 大文件拆分及最终 controller/package 收敛归 M4。M1 已完成 model
declaration provider/catalog fragment；model/algorithm/internal component 的统一静态解析与 identity cut 归
M2.5，composition 目录其余物理收敛归 M4；真实
native parity、resume 与 20-step 证据归 M6–M7。

### M2：算法域收敛（当前）

#### M2.0：三路审计与实施计划（完成，2026-08-03）

- [x] 审计 module/compiler/provider/identity 全链；确认旧 resolver 在静态 gate 前 import implementation。
- [x] 审计 rollout/dynamics/conditioning/trainer 控制权和实际五组合链路。
- [x] 审计 advantage/credit/objective/recompute/update 的 production importer、数值/RNG/graph lifetime 和
  checkpoint 边界；相关 CPU contract 基线 `164 passed`。
- [x] 冻结本节的目标树、配置 owner、一次 identity release cut、删除清单和每批门禁。

#### M2.1：import-safe 算法声明壳层（完成，2026-08-03；零 production identity 变更）

1. 建立 `algorithms/modules/{descriptor,interface}.py` 与
   `algorithms/{rollout,dynamics,conditioning,optimization,trainer}/config.py`；provider/config 文件不得
   import runtime implementation、torch、具体 model 或旧万能 `registry.interfaces`。
2. 扩展 AlgorithmBlueprint slot 为 `id/family/resolution + canonical frozen params`；Flow/TempFlow/Flash
   frozen config 完整描述算法语义与内部默认，不再只保存 beta/空 config。
3. 建立 `algorithms/catalog.py` 的 import-safe domain fragments；总 Registry 状态仍只在 composition。
4. 增加 fresh-process import、provider ABI、CONFIG_TYPE_PATH、catalog conflict 和 algorithms→composition/
   models.implementations 禁止依赖测试。此切片不切 production registry/class path，因此不改变现有产物 identity。
5. M2.1 catalog 可先声明 M2.2–M2.3 的最终 runtime class path，但只允许
   `source/provider target/symbol` 精确、双向 stale-checked 的阶段性债务清单；不得创建空壳实现或宽泛
   allowlist。每个最终实现物理落地时必须在同一变更删除对应债务项，M2.4 退出时清零。

完成证据：algorithm/domain fragments、完整 provider/config/implementation identity、中央
`algorithms.catalog`、Blueprint slot params、TempFlow `transition_noise_scale` 和
`component_declaration_id` 已落地；M2.1/模型组合/架构聚焦门禁 `162 passed`，Ruff、format、compileall 与
diff-check 通过。10 项未物化 runtime target 由精确双向 stale-checked debt 管理；这不是 runtime 可加载或
真实训练证据。

#### M2.2：canonical optimization 收敛（完成，2026-08-03；production identity 暂不切换）

完成状态：`algorithms/optimization/{objective,advantage,credit,recompute,slots,execution,kernel}.py`
已成为 optimization 数值、credit plan、逐 slot recompute、slot 分区、事务执行和 streaming update 的唯一
production owner。production 已删除 full-T `PolicyRecomputer.compute()`、monolithic
`PolicyUpdateKernel.step()` 和隐藏 local accelerator；旧 `training/*`、`algorithms/*` 入口仅保留无逻辑
re-export shim，完整 full-grid 对照仅存在于 `tests/support`。旧 optimizer 实现已删除，仅因 M2.5 identity cut
暂留空 package 与只导出 canonical `PolicyLossInputs` 的 `optimizers/objective.py` shim。

三个已物化 credit target 已从 exact debt 删除，债务由 10 项降为 7 项；剩余 6 项
rollout/dynamics/conditioning 属于 M2.3，最后 1 项 trainer 属于 M2.4。M2.2 聚合门禁
`254 passed`，其中 architecture/wheel/import 子集 `39 passed`；Ruff、format、compileall 和
`git diff --check` 均通过。`row_microbatch_size` 的完整 partition replay、reference CPU 暂存和
Accelerate accumulation 语义仍是 M5/M7 性能与真实链路验证项，不能据此宣称 OOM 已解决。

1. 将 `optimizers/{objective,clipped_surrogate}.py` 中唯一仍被 production 借用的
   `PolicyLossInputs`、masked/clipped surrogate 与 reference regularizer 合并到
   `algorithms/optimization/objective.py`，保持公式、dtype、reduction 与 tolerance 不变。
2. 将新 `advantage/processor.py`、`algorithms/{components,credit}.py` 迁入
   `algorithms/optimization/{advantage,credit}.py`；`PolicyStats/ReferencePolicyStats` 归 recompute，credit
   只产 detached loss plan。
   TempFlow canonical credit 必须从 `TempFlowCreditConfig.transition_noise_scale` 读取并用于
   `transition_std_dev` 权重，默认值 `2.25` 与现有数值一致；旧 `NOISE_SCALE` class constant 删除。
3. 将 `training/{policy_recompute,update_slots,update_execution,update_kernel}.py` 原样拆到
   `algorithms/optimization/{recompute,slots,execution,kernel}.py`。完整 detached
   `new_log_probs[B,T]`、逐 slot backward、一次 optimizer commit、失败原子性、execution context 和 RNG
   顺序必须不变。
4. 把 full-T `PolicyRecomputer.compute()`、monolithic `PolicyUpdateKernel.step()` 的等价用途迁到 test/native
   oracle 后从 production 删除；OptimizeStage 只能调用 streaming path。
5. 迁移有价值的 legacy 手算 tests 后，物理删除
   `optimizers/{advantages,base,grpo,tempflow_grpo,flash_grpo,update_engine}`，不把旧 PolicyAlgorithm 链改名。
   若旧 production class path 尚需维持到 M2.5，只允许一层无逻辑 re-export shim，并在同一个 M2.5 cut
   删除；canonical 定义从本切片起只有最终路径一份。
   旧 `PolicyAlgorithm` diagnostics、TempFlow/Flash 私有 telemetry 与 `UpdateEngine` metrics merge API
   没有 production consumer，明确退休；若未来需要观测，必须由独立 observer/metrics 层重新声明，禁止
   为保留旧测试把它们塞回 credit。`optimizers/objective.py` 仅因 `training/algorithm_plan.py` 的两条旧
   stage-type identity 暂留一个只导出 `PolicyLossInputs` 的 shim，M2.5 identity cut 后与空 package 一起删除。
6. 三个 optimization runtime target 落地时，同批增加 provider config→runtime `from_config()` 类型兼容、
   provider/runtime contract 等价、canonical 实例非 legacy re-export 和禁止反向 legacy import 门禁，并从
   M2.1 exact debt 删除对应三项。
7. `PolicyStats` 只归 recompute，canonical 字段限于 differentiable current log-prob 与 reference-KL
   statistics；`transition_std_dev`/`rectification_coefficient` 已由 trajectory 保存并由 credit `plan()` 消费，
   不得再经过 recompute 形成第二条 metadata 链。`CoefficientMeanReducer` 是 credit port；local reduction
   backend 由 runtime 注入，不属于 recompute。
8. 保持两个已知性能边界为显式 M5/M7 验证项，物理迁移时不得悄悄改变数值/RNG：
   `row_microbatch_size` 当前仍会按 rollout 的完整 `forward_partitions` 重放，不能据此宣称模型 forward
   峰值已被限制；Accelerate `sync_gradients=False` 当前会形成 `ACCUMULATING`，但 logical one-update API
   会将其视为未提交并中止 iteration，任何 `gradient_accumulation_steps > 1` 配置都必须单独修复和验证。

#### M2.3：rollout、dynamics 与 conditioning 物理收敛（完成，2026-08-03；production identity 暂不切换）

完成状态：`algorithms/dynamics` 已唯一拥有 interface/config/transition/replay/session/selection/schedule 与
SD3/Wan factory；`algorithms/rollout` 已拆为 interface/request/collector/builder/config 和三种只拥有控制流的
strategy；`algorithms/conditioning` 已唯一拥有 camera config/interface/math/runtime。旧顶层
`visual_rl/{dynamics,rollouts,conditioners}` Python 源码已清零并由 source/wheel/static-import gate 禁止回归。

Wan 已使用命名中立的 `standard/flash/conditioned` profile，并用 typed likelihood/replay pair 区分普通
Flow、Flash 与 World-R1；Dynamics 内不再出现 recipe-name 分支。六个 provider config 均可构造 canonical
runtime，`CONFIG_TYPE`、`describe(config)` 和 declared contract 一致；六项 declaration-target debt 已删除，
只剩 M2.4 的 trainer target 一项。汇合后的 M2.3 宽回归 `365 passed`，unique-owner/architecture gate
`23 passed`，wheel/architecture/stale-credit 聚焦门禁 `39 passed`；Ruff、format、compileall 和
`git diff --check` 通过。

边界：当前 production compiler 仍走旧 `ComponentResolver/ComponentLoader/RegistrySet`，旧 registry 仍镜像
一份 descriptor，`algorithms.optimization.credit` 仍反向依赖 legacy `registry.interfaces`，rollout request
仍消费旧 trainer 的 `IterationIdentity`。这些分别归 M2.4–M2.5；因此本切片只证明物理 owner、typed ABI 和
CPU fake 数据链闭合，不是新的 provider/loader 已成为 production，也不是 CUDA/OOM 或真实训练证据。

1. 先迁数值叶子和各域 interface/config，再迁 replay/session/selection 与 rollout collector/builder；
   `data.samples` 继续唯一拥有 immutable trajectory DTO。
2. `rollouts/base.py` 按 interface/collector/builder 拆分，三种策略只拥有控制流；model 只经
   `PolicyRuntimePort`/scheduler blueprint/latent context 被调用。
3. 普通 Flow×Wan 改为命名中立 exact-action Wan flow-SDE；Flash profile 由 Flash blueprint 要求，World-R1
   conditioned profile 由 Flow blueprint + likelihood + camera conditioner integration 推导，不能按 recipe 名
   分支。
4. 保持 Flow/TempFlow/Flash/World-R1 rollout 数值、transition selection、replay target、branch row/RNG 顺序
   和 forward partition 完全等价；不在本切片“优化” microbatch 算法。
5. 旧 registry 仍可经单层无逻辑 shim 导向最终定义，以保持 pre-cut manifest path；禁止复制第二份 runtime
   class。所有 shim 必须登记并在 M2.5 原子 cut 同时删除。
6. conditioning/dynamics/rollout 六个 canonical target 落地时，同批验证 provider config 可实际构造 runtime、
   静态 contract 与 runtime contract 一致且实现不是 legacy re-export，并删除 exact debt 对应六项。

#### M2.4：trainer/control-flow、spec-only plan 与 runtime owner 预备（完成，2026-08-03；production identity 留待 M2.5）

完成状态：旧 `visual_rl/trainers` 与 `training/{stages,assembly,ports,types,data_prelude,rollout_request}.py`
源码已删除；Trainer interface/config/control-flow/stages/execution plan 已分别归入 `algorithms/trainer`，runtime
component graph/binding/result DTO 与 data prelude 也已取得唯一 owner。`CreditComponent/CreditPlanningPort` 已归
`algorithms/optimization.interface`，`IterationIdentity` 只有 trainer interface 一个定义。

旧、canonical 两套过渡期 `AlgorithmModule` 均已删除 `bind/_bind_runtime` callback seam；旧 module 不再反向
import composition，production 由 runtime 显式传入已验证的 `policy_runtime.binding`。新增
`runtime.algorithm_materializer.CanonicalAlgorithmMaterializer`，形成不读取 recipe、registry、legacy module 或
`semantic_config` 的完整 shadow 链：canonical Flow/TempFlow/Flash module + binding +
`AlgorithmMaterializationSpec` + typed components → `AlgorithmExecutionPlan.from_spec()` → shared Trainer execution
port；selection/declaration identity、one-shot、checkpoint/restore step 与 lazy import 均有门禁。production 尚未
接入该 shadow 路径，仍完整使用旧 recipe 链，避免在 M2.5 前出现半新半旧 identity。

所有 staged missing-target debt 及其专用 allowlist 机制已清零并删除，动态 class path 现在必须直接物化。
最终 M2.4 全量 CPU/合成回归为 `1533 passed, 4 skipped`；bind + shadow materializer + production +
architecture/wheel 联合门禁 `91 passed`，新增 owner/dependency 聚焦门禁 `36 passed`；M2.4 涉及的 46 个文件
通过 scoped Ruff、format、compileall 和 `git diff --check`。这些均不是 CUDA、OOM 或真实训练证据。

1. `trainers/{base,components,grpo}.py` 迁 `algorithms/trainer/{interface,config,grpo}.py`，不迁只供旧测试的
   `GRPOIterationPrelude`；`training/stages.py` 和新的 spec-only execution plan 迁 trainer 子域。
2. `training/rollout_request.py` 迁 rollout；`training/assembly.py` 归 `runtime/component_graph.py`；
   `training/ports.py` 拆入 runtime algorithm binding 与 trainer interface；`training/types.py` 归 runtime；
   `training/data_prelude.py` 的 data-owned 部分归 `data/prelude.py`。这些移动可以随 M2 完成控制权切换，
   但若与 M4 composition/data schema 收敛同批更安全，M2 退出时允许 `training` 仅剩明确登记的 M4 文件，
   不允许再剩算法语义或算法执行实现。
3. 删除 `AlgorithmModule.bind(policy, arbitrary_callbacks)` 和 AlgorithmModule→composition 反向依赖；
   runtime 构造 binding/materializer，module 仅验证并 materialize typed graph。
4. 建立 `AlgorithmExecutionPlan.from_spec()` 和新的 component-graph/materializer 路径，但在本切片先由
   shadow/focused tests 驱动；旧 production compiler/manifest 继续工作到 M2.5，不能出现一半 recipe 走新
   spec、一半 runtime 回读旧 semantic_config 的混合启动。
5. `algorithms.trainer.grpo:RegisteredGRPOTrainer` 落地时执行同样的 provider/runtime ABI 门禁，删除最后一项
   exact debt；M2.4 退出条件包含 debt manifest 为空并移除该阶段性机制。
6. 将 `CreditComponent/CreditPlanningPort` 迁到 `algorithms.optimization.interface`，registry 只引用 canonical
   base path；`algorithms.optimization`、trainer stage 与 runtime stage assembly 对
   `registry.interfaces` 的反向依赖归零。
7. rollout request 的 `IterationIdentity` 改为消费 canonical trainer interface；在同一变更迁移
   `trainers`/`training` owner，禁止为通过 import 临时复制第二份 identity DTO。

#### M2.5：唯一 compiler 权威与一次原子 identity release cut

切换前置：先为四个 reward alias 补齐 import-safe declaration provider、canonical `RewardComponent` interface 与
`REWARD_CATALOG_FRAGMENT`，并让 provisional/materialized `RewardPlanSpec` 成为 reward 声明与 artifact identity
的唯一输入。M2.5 只切 reward 的声明、identity、spec 与 loader 权威；现有数值 adapter/client 可暂留旧物理
路径，但必须实现 canonical interface。Reward 热路径、legacy bridge 与最终 implementation class path 迁移仍归
M3，并作为独立、fail-closed reward identity cut 记录，不伪装成与 M2.5 checkpoint 兼容。

M2.5 production cut 后的 canonical shape 固定如下；实现不得为了兼容旧 consumer 再加入
`semantic_config` 或不透明 artifact mapping：

```text
ResolvedRecipe
├── definition metadata: definition_id/name/version/fidelity_target
├── algorithm: ResolvedAlgorithmDeclaration
├── algorithm_spec: AlgorithmMaterializationSpec
├── model: ResolvedSlotDeclaration
├── internal_components: trainer/dynamics/rollout/credit/(conditioner)
├── reward_components: logical slot + ResolvedComponentDeclaration
├── reward_plan: RewardPlanSpec[provisional]
├── source_plan: SourcePlanSpec
├── execution_policy: ExecutionPolicySpec
├── training: TrainingSpec
├── phase_schedule: PeriodicPhaseSchedule | None
├── dynamics_integration: DynamicsIntegrationSpec
├── dynamics_projection: ModelBoundDynamicsProjection
├── compatibility: normalized rule/result snapshot
└── resolved_fingerprint: hash(all canonical fields above)

MaterializedRecipe
├── resolved: ResolvedRecipe
├── model_artifact_identity
├── source_content_binding: SourceContentBinding
├── reward_plan: RewardPlanSpec[materialized]
├── code_artifact_identity # 默认覆盖完整 visual_rl 包，不能只哈希 composition 子树
└── recipe_id

ComponentArtifactBindingSet # G1；recipe_id 生成后再纯派生，不能回填 recipe hash
├── recipe_id
├── bindings: tuple[ComponentArtifactBinding, ...]
└── binding_set_id
```

`source_path/config_source_id/raw YAML/override_paths/evidence label` 只进入 config/source audit；absolute artifact
path、endpoint、CA bundle、output/cache/logging/resume path 与 rank/host/PID 只进入 launch audit。它们均不得参与
resolved/materialized/resume equality。`ResolvedSlotDeclaration` 只增加 logical slot，不复制 declaration 内的
provider/config/contract identity。

默认 code artifact root 必须是完整 `visual_rl/` 包；`algorithms/`、`models/`、`runtime/` 或
`composition/` 中任一 Python 源码变化都必须改变 `code_artifact_identity` 与最终 `recipe_id`。允许测试或嵌入式
部署显式注入更窄的 `code_root`，但 production default 不得隐式缩小身份范围。

原子批次必须同时替换下列 consumer，任何一组残留即视为 cut 失败：

- `preflight/{artifacts,runtime}.py`：改读 source/reward plan 与 execution policy；
- `runtime/{component_graph,production,defaults,binding,checkpointing}.py`：禁止二次 resolve 或回读 raw config；
- `composition/recipes/{phase_compiler,source_projection}.py`：删除 projector，直接消费 typed plan；
- `rewards/planning.py` 与 `training/algorithm_plan.py`：删除 legacy projector；
- `checkpoint/builder.py`、`configs/manifest.py` 与 resume validation：只读 typed materialized/G1 identity；
- 所有 production `RegistrySet`、legacy `ComponentResolver/ComponentLoader` 与旧 shim import 同批归零。

1. production `compile_recipe_v2` 改用 `Catalog + DeclarationResolver`：先解析 algorithm provider/blueprint，
   再从 blueprint 派生 trainer/rollout/credit/objective；model-bound dynamics 由 blueprint requirement、Model
   descriptor 与 integration policy 选择，conditioner 只由 integration route 引入。
   public algorithm 不得继续复用 generic `declare_component()` 后再 duck-type config 的 `describe_*`；由
   `algorithms.modules.declarations.AlgorithmDeclaration` 与专用 provider ABI 一次原子返回 frozen config、
   requirements、blueprint。普通 model/reward/internal provider 仍使用 generic ABI；provider ABI 本身进入
   declaration identity，generic resolver 遇到 specialized ABI 必须 fail closed。
   model-bound Dynamics 投影必须是可独立测试的纯函数，至少锁定
   `Flow×SD3→flow-sde`、`TempFlow×SD3→flow-sde`、`Flow×Wan→standard`、
   `Flash×Wan→flash`、`World-R1 surrogate/exact→conditioned`，并 fail closed 拒绝
   `TempFlow×Wan`、`Flash×SD3` 与 `beta>0 Flow×Wan`；测试不得手填目标 params 后冒充已验证投影。
2. recipe schema 移除 trainer/dynamics/rollout/credit/objective 的 alias/params 自由覆盖；beta、steps、
   branch/selection 等算法语义参数迁入 `algorithm.params`；completion group size 与
   forward/decode/storage 迁入 `ExecutionPolicySpec`，policy recompute geometry 继续由唯一的
   `TrainingSpec.policy_recompute` 拥有。两者都属于 typed execution/training policy，但禁止复制同一字段。
   TempFlow 必须在 compiler 阶段要求 execution group size 与 branch count 相等。旧字段在 source parse 时
   fail closed。
3. compiler 输出并把 `AlgorithmMaterializationSpec`、`RewardPlanSpec`、path-free `SourcePlanSpec` 纳入
   `ResolvedRecipe.canonical_semantic_payload()`；内部 component/provider/config/implementation identities
   全部被 hash。`AlgorithmComponentSelection.component_declaration_id` 必须直接保存
   `ResolvedComponentDeclaration.declaration_id`，不得退化为只保存 declared-contract hash。
   即使两个 Dynamics config 产生相同 capability contract（例如 Wan standard 与 conditioned-exact），
   materialization/checkpoint identity 仍必须因完整 declaration id 不同而不同。
   `DatasetSourceSpec` 必须显式保存 `format=text|jsonl` 与 `artifact_kind=file`，loader 禁止再从本地路径后缀
   推断解析语义。含 absolute `Path` 与 expected content identity 的 `SourceLocationBinding` 只能由
   artifact/environment gate 产生，不进入 resolved identity；location 只进入 launch audit，content identity
   进入 materialized identity 与 cache key。loader 必须打开一次、读取稳定 byte snapshot、由同一批 bytes
   计算 expected identity 并解析，禁止“先复核 hash 再 reopen parse”，以堵住 preflight 与 load 之间及复核
   与解析之间替换文件的 TOCTOU。
4. production 切到 `AlgorithmExecutionPlan.from_spec()` 与最终 component graph；G1 先从完整 declaration id
   与 immutable artifact/code identity 纯生成 `ComponentArtifactBinding`，所有 slot gate 全部验证后才允许
   第一次 implementation import。`RuntimeComponentLoadGate` 必须锁 recipe、slot、declaration、artifact-set、
   canonical interface，禁止只比较 capability contract 或由调用方传 `object` 弱化 expected type。轻量 adapter
   构造后才 load/prepare；G3 `RuntimeBoundContract` 必须引用同一个 G1 artifact contract/receipt，不得重新拼一份。
   production 对旧 `ComponentResolver`、旧 `ComponentLoader`、`RegistrySet` 和
   `training.assembly.load_component_graph` 的引用归零。
5. preprocess requirement/compatibility identity 与 payload cache identity 分开：requirement set 仍用于 producer
   coverage 与 checkpoint audit，但不得进入 `PreprocessPlan.plan_id`/cache key。cache 只 hash producer 实现与
   输出 schema、真正影响 preprocess bytes 的 config/transform、相关 model artifact identity 和 source content
   identity；algorithm/rollout/conditioner/reward/group/beta 及与 preprocess 无关的 LoRA/训练参数不得污染 key。
6. 同批切换所有 algorithm/internal provider/config/implementation class path，并删除 M2.2–M2.4 的临时
   shim。这是 M2 唯一一次原子 release cut：recipe semantic shape、resolved/materialized/checkpoint identity
   同时变化；旧 checkpoint/recipe 必须在 mutable state load 前 fail closed，不保留 alias 或路径重写。
   算法变更不得污染 model-only preprocess cache key。

#### M2.6：旧路径关闭与完整门禁

1. class paths、configs、tests、native harness、scripts、docs、CI 和 checkpoint fixtures 全部切 canonical path。
2. 物理删除 `visual_rl.algorithm_modules`、`advantage`、`optimizers`、`rollouts`、`dynamics`、
   `conditioners`、`trainers`；`training` 中所有 algorithm-owned 文件物理消失，整个 `training` 最迟 M4
   退出时删除。源码/wheel 不保留 facade。
3. 六份正式 config 在 fresh process 静态 compile 时不得加载 torch/diffusers/transformers/peft/
   accelerate/requests、model implementations 或 algorithm runtime implementations；pre-gate spy import 失败。
4. 六 route 派生 slot id/params 必须与 blueprint 精确相等；内部 slot override、runtime blueprint drift、旧
   class path/checkpoint 均结构化 fail closed。
5. 跑各批 focused numerical/RNG/update/checkpoint/architecture tests、六 route fake G0–G6、full pytest、
   Ruff、compileall、diff-check、source/wheel forbidden-prefix 与 isolated-install gate。M2 不运行真实训练。

退出：AlgorithmBlueprint/`AlgorithmMaterializationSpec` 是 compiler 与 runtime 的唯一内部算法事实来源；
algorithm domain 不读 recipe、不 import composition/具体模型/旧 registry；旧算法 namespace 无第二 owner；
复用现有内部能力的新算法只需新增 module/provider 和必要 integration recipe，确需新 rollout/dynamics/credit
时只增加对应 `algorithms/**` 文件，绝不修改 models。

### M3：Reward 原生化

1. 先把 `world_r1_protocol.py` 迁到 `core/protocols/world_r1.py`，同步 strict service imports。
2. clients 直接消费 `RewardBatchView` 并返回 typed point/group output。
3. 移除 `_LegacyFeedbackInputAdapter`、Torch↔NumPy 多次桥接和 camera FP64 非必要转换。
4. logical routing/descriptor/input selection/execution 留算法 reward 域；物理 pool/factory 生命周期归 runtime。
5. 删除 `feedback`；更新严格服务 protocol import。

退出：reward hot path 不出现 legacy RolloutBatch/RewardVector；同物理资源去重、逻辑 reward 不合并；
row/camera identity 保持。

完成状态（2026-08-03）：以上五项均已完成。`algorithms/rewards` 不反向依赖 `runtime`，只通过
`RewardResourceHandle` / `RewardResourcePoolView` protocol 使用资源；runtime 唯一拥有 pool、borrow、
activate/close 生命周期。source、wheel 与 fresh-process import gate 同时禁止 `visual_rl.feedback` 和
`visual_rl.rewards` 回归。M3 后 reward/runtime 聚焦回归 `116 passed`，architecture/wheel 聚焦回归
`104 passed`，全量非分布式回归 `1676 passed, 3 skipped, 2 deselected in 57.30s`；clean wheel 与
隔离安装验证通过。该 identity cut 按计划不声明与 M2.5 旧 reward checkpoint 兼容。

### M4：Data、Composition、Artifacts、Runtime 收敛

1. 合并 samples/data，删除无 production 引用的旧 `datasets`，不把旧数据栈迁入目标树。
2. 在 M0/M2 壳层上补齐 recipes/compatibility/preflight，不再另建第二 registry/compiler。
3. 删除 v0.7 resolver/schema/builtins/preflight helper。
4. 先建立 core checkpoint participant/snapshot port；各 domain capture snapshot，runtime 在 safe point
   收集，artifacts 只校验/序列化/原子提交；再合并 checkpoint/terminal 并删除 v0.7 commit-chain。
5. 拆分 `runtime/production.py`、`checkpointing.py`、`binding.py`。
6. `train.py` 仍只调用 `runtime.composition.create_default_run_controller()`。

退出：顶层仅有第 5 节七个能力域；无第二 controller/runner/API；所有 package import-safe。

### M5：内存边界与可观测性

1. UpdateSlot 与 rollout forward partition 对齐。
2. 所有正式 recipe 显式给出 forward/decode/storage/recompute 几何。
3. Wan 使用预分配 batched trajectory builder。
4. terminal metrics 增加 slot/replay/GPU/CPU/trajectory 指标。
5. 异常 traceback 不保留 trajectory/prepared graph/closure。

退出：单测证明 graph 生命周期、失败原子性和 host builder；仅允许不加载真实 model/reward artifact 的
synthetic/unit memory profile 验证 replay amplification 上限。任何真实 profile 归 M7，继续暂停。

### M6：本地完整冻结

- 全量 pytest；
- changed-file Ruff + package compile；
- import/ownership/wheel forbidden-prefix gate；
- build wheel、isolated install、source/wheel manifest；
- 六份正式配置走同一个 default controller 的 fake G0–G6；
- checkpoint/resume deterministic tests；
- fixed native fixtures 与公式容差通过。

完成 M6 后生成唯一 final source/release revision。此前 release-e 及更早版本全部标为 diagnostic。

### M7：真实 GPU bring-up 与最终实验（bring-up 已启动）

最终证据严格顺序如下；当前冻结前运行只执行相同路线的 engineering bring-up，不替代这些步骤：

1. reward 服务从 final wheel 启动并产生新 marker；
2. 六条 route 逐条 1-update stage probe；
3. 六条 route 逐条至少 20-step continuous run；World release 的更长 phase run可覆盖其 20-step 要求；
4. Flow×SD3 与 World-R1 release×Wan 做中断/resume；
5. 有官方基准的 Flow×SD3、TempFlow×SD3、Flash×Wan、World-R1 release×Wan 做 native parity；
6. World-R1 release 做 step-100 phase boundary（至少 101 step，优先完整 150 step）；
7. 最后才更新 evidence 文档和完成状态。

## 13. 真实实验必须记录的证据

每条 route 必须记录：

- source manifest、wheel SHA-256、model/dataset/reward artifact tree identity；
- 完整 resolved recipe、algorithm blueprint、model-dynamics binding identity；
- Python/PyTorch/CUDA/driver/GPU/precision/allocator 环境；
- B/K/T、frame/height/width、forward/decode/update partition；
- 每 step/slot wall time、forward calls、row-equivalents、replay amplification；
- GPU allocated/reserved/driver peak、CPU maximum RSS、trajectory bytes；
- loss、advantage、old/new log-prob、clipfrac、KL、grad norm、non-finite gate；
- checkpoint tree、RNG、data cursor、phase、algorithm/dynamics state；
- SUCCESS、exit code、OOM scan 和 resume 前后 exact continuation。

“20 steps”必须是 20 次 optimizer commit，不是 20 个 diffusion timestep，也不是只生成 20 个样本。

## 14. 验收门禁

| Gate | 要求 |
| --- | --- |
| A0 Ownership | 唯一 owner 表、import DAG、forbidden namespace 全通过 |
| A1 Compile | blueprint 派生内部组件；不兼容在 artifact load 前失败 |
| A2 Runtime bind | model scheduler blueprint 与 dynamics ABI 精确绑定 |
| A3 Numerical | rollout/recompute/native formula/RNG fixture 等价 |
| A4 Update | slot-streaming、one commit、失败原子性、无 graph 泄漏 |
| A5 Persistence | checkpoint/resume state 与 final no-op restore 等价 |
| A6 Packaging | final wheel 无 legacy namespace，isolated install 通过 |
| A7 Real 20-step | 五种组合的六条 route 均成功且没有 CUDA/CPU OOM |
| A8 Parity/phase | 四条 native parity、两条 resume、World release step-100 边界 |

任何 narrow smoke 只能证明对应 gate，不能扩大为“框架已真实可用”。

## 15. 当前进度

| 项目 | 状态 |
| --- | --- |
| 分支与目标 | 已建立 |
| 旧 API/Runner/RuntimeFactory | 已删除并有 source/wheel gate |
| slot-streaming 核心修复 | 已实现并有单测；forward partition 与真实内存 profile 仍属 M5/M7 |
| numerical leaf 脱离 model_adapters | 已完成 |
| Qwen 3D scorer direct-device loader | 代码与 mock tests 已完成；bundled service revision 已级联并通过 deterministic digest contract；M6 final freeze 后仍需再次计算 |
| 全包 ownership/重复域审计 | 已完成 |
| Flow-Factory 固定源码复核 | 已完成，固定 commit 与真实数据链见第 3 节 |
| M0 架构壳层与门禁 | 已完成，contracts/config/registry shell、runtime loader、ownership/wheel gate 已落地 |
| M1 模型物理域 | 已完成：`ModelAdapter` rename、目录拆分、旧 model_adapters 删除、wheel forbid、import-safe provider/catalog fragment 均已落地 |
| Model–Dynamics semantic decoupling | 已完成 static ABI、artifact blueprint、typed runtime bind 和 exact replay-state 二次检查；2026-08-08 审计进一步删除 `integration.py` 的 SD3/Wan `if/elif` closed switch，改为可由 compiler/preflight/production controller 注入的 immutable `DynamicsProjectionRegistry`。model 只提供 binding family，algorithm 只提供 requirements/blueprint，自定义模型 projector 不修改核心 compiler；新增双向 import gate，强制 model 不导入 algorithm、algorithm 只通过 model port/scheduler ABI 连接而不能导入具体实现；相关 projection + architecture focused gate `62 passed` |
| M2 算法域审计 | 已完成 module/compiler、control-flow、optimization 三路只读审计；确认三份事实来源、旧 resolver pre-gate import 和 legacy optimizer 删除范围；相关 CPU contract `164 passed` |
| 当前回归证据 | 2026-08-08 frozen code `56507f6e…375b`：全量非分布式主套件 `1649 passed, 1 skipped, 1 deselected in 61.98s`；clean wheel/core-only isolated install 已通过，wheel SHA-256 `6f1533ef8e1d3ed471414ddea8ee956db199bd3b154500ed66a44f51b31bca61`；183 个 wheel Python member 与 source 逐文件一致。六份 final config 和两个 frozen-wheel reward marker 已由 freeze record `a6c961fc…` 统一绑定 |
| 顶层目录最终收敛 | 七个能力域已形成；旧 configs、checkpoint、inspection、samples、preflight、compatibility 等顶层 namespace 已物理删除。runtime 已收敛到计划中的 15 个顶层文件；checkpoint/terminal 语义与剩余 legacy leaves 已归并，M4 结构迁移完成；strict wheel/import gate 留给 M6 重跑 |
| AlgorithmBlueprint compiler ownership | M2.1 声明层、M2.2 optimization、M2.3 rollout/dynamics/conditioning、M2.4 trainer/spec-only plan/materializer、M2.5 production identity cut 与 M2.6 旧 owner/shim 物理删除均已完成 |
| M2.5/M2.6 原子切换 | 已完成：consumer DAG/canonical shape/owner、canonical reward provider/catalog、`RewardPlanSpec`、path-free `SourcePlanSpec`/`SourceContentBinding`、typed `ExecutionPolicySpec` receipt、specialized algorithm ABI、typed decoded-media、preprocess cache identity、blueprint-owned beta、model-bound Dynamics 投影、canonical Resolved/Materialized schema、strict all-slot `ComponentLoadPlan`/G1/G3 均已接入 production；旧 namespace 在 source、wheel 与 isolated install 中均不可导入 |
| Reward 原生化 | M3 已完成：typed input/output、row/camera identity、logical/physical resource owner 分离、legacy bridge/namespace 物理删除与 wheel gate 均已落地 |
| final full test/wheel | 已完成并冻结：source `56507f6e…`、wheel `6f1533ef…`、general marker `648cebf…`、3D marker `fbceb863…`；freeze record SHA-256 `a6c961fc4b2670f1df947d73015f1572e35840acb8fd9ad6bf734905af5b07f9` |
| real GPU bring-up | **进行中**：`10.130.140.73`；Flow×SD3 attempt 1 因 NFS 阻塞取消，attempt 2 在 preprocess 30.8 GiB OOM；模型层 CPU-static text-encoder 修复后的 attempt 3 已完成 20/20、SUCCESS、step-20 checkpoint，driver peak 21,039 MiB。TempFlow×SD3 也已完成 20/20 并生成 SUCCESS。Flow×Wan attempt 1 成功加载模型但首个 optimizer recompute 将默认的全部四行同时前向，30.58 GiB 后再申请 560 MiB 发生 OOM；四份正式 Wan config 已将 `training.policy_recompute.row_microbatch_size/transition_window_size` 固定为 `1/1`。attempt 4 无 OOM，且确认 reward/advantage 非退化，但所有已存在 grad tensor 数值恰为零；根因是 standard Wan 从模型 scheduler 继承确定性采样，导致策略动作等于均值。现已将 stochastic sampling 提升为显式 Dynamics 投影、contract、factory/replay identity 并增加 mode-drift gate；本地相关回归合计 158 passed，GPU 3 attempt 6 正在验证首个真实 commit。Wan `/dev/shm` 副本已核对 21 个文件、28,928,888,051 字节，Flash×Wan 20-step bring-up已在 GPU 2 启动。旧 general reward 的真实 smoke 返回 `0.231689453125`，但源码审查确认其内部 CUDA autocast 违反声明的 FP32 runtime；新 revision `world-r1-8e46b1b63498` 已移除 autocast，GPU 4 的真实 HPS smoke 返回 `0.23295444250106812`，checkpoint/health/score receipt 已哈希落盘。3D reward 使用按冻结 requirements 新建的 tmpfs venv以及 tmpfs source/Qwen/DA3/LPIPS；修复 venv `PATH` 后，真实 worker 已完成 DA3、Gaussian Splatting 与 Qwen3-VL 评分，4 帧 smoke 返回有限分数 `1.9351264214596995`，健康/score receipt 已哈希落盘。所有这些仍是 `engineering_bringup_not_release` evidence |
| final stage-1/20-step/resume/parity | A7 为 4/6 route accepted：Flow×SD3、TempFlow×SD3、Flow×Wan、Flash×Wan 均为同一 freeze 下的 fresh 20/20、SUCCESS、step-20 checkpoint、正梯度和有界显存；World-R1 core/release 尚未形成正式 20-step 终态。A8 resume/native parity/step-100 phase-boundary 不由这四条 A7 收据替代，仍单独记为未完成 |
| final A7 实际运行 | 进行中：四条非 World-R1 route 已正式 accepted。World-R1 core/GPU3 已提交 step-10 checkpoint并继续计算；release/GPU6 首次 final run 收到 SIGTERM、exit 143 且无 SUCCESS，因此明确拒绝。事件驱动恢复 PID `2270592` 等待 core accepted 后保留失败证据，并在释放的 GPU3上以相同 freeze/config fresh 重跑 release；所有 route 的 stdout、trainer PID、exitcode、launch receipt 和 15 秒显存 CSV 都位于独立 tmpfs evidence root |

旧 source-e 已有 SD3 Flow/TempFlow 与 reward 服务诊断记录，但源码随后发生变化，因此不计入最终 A7/A8。

2026-08-08 补充：三个 GRPO algorithm requirements 已显式加入
`stochastic` transition feature；同一 scheduler blueprint 下不同采样模式的
factory/replay identity 必须不同，config/replay mode drift 在 runtime fail closed。
Flow×Wan stochastic 与 World-R1 core 的 1-step 已分别在 GPU 3/GPU 7 完成：
两者均为 exit 0、SUCCESS、step-1 complete/latest，梯度范数分别为
`0.0006251836894080043` 与 `0.004145184997469187`，无 OOM/零梯度/异常日志命中。
这证明 Wan stochastic Dynamics 修复能真实产生非零 policy gradient，并且 World-R1
camera + general/3D reward 链能提交一步。release-surrogate 1-step 仍在 GPU 6 运行；
后续继续按 20–30 分钟的关键节点检查，不做持续轮询。
release-surrogate 随后已形成 `committed_steps=1/update_count=1`，且无 OOM、零梯度或
非有限值日志；但在 NFS checkpoint coordinator staging 写入约 284 MB 的
`rank-0.pt` 时进入 `D` 状态，内核等待点为 `folio_wait_bit_common`，所以没有形成
SUCCESS/latest；约 40 分钟后该写入自行恢复，任务最终 exit 0、SUCCESS、step-1
complete，日志无 OOM/零梯度/非有限值/异常命中。该问题属于产物 I/O，不是模型或算法计算；六份 final server config
现已把 output_dir 固定到独立 tmpfs root，训练达到 SUCCESS 后再向 NFS 归档。
这些 final config 相对 bounded candidate 只改变 output/reward deployment path，
六份均可由 compiler 解析，远端逐文件 SHA-256 与本地记录一致。
当前 code candidate 的非分布式完整回归在删除第二 compatibility 路径并加入
更严格的双向 ABI/条件分支门禁后全绿：`1649 passed, 1 skipped, 1 deselected in
61.98s`。同一 runtime tree 构建出的 clean wheel 已在不安装 Torch/Diffusers 的
隔离环境完成安装、静态 recipe 编译、旧 namespace 缺席与模块入口 smoke；wheel
SHA-256 为 `6f1533ef8e1d3ed471414ddea8ee956db199bd3b154500ed66a44f51b31bca61`。
隔离检查器同时修正了两个自检边界：父包已删除时嵌套 `find_spec` 应视为 absent，
静态编译检查不得主动导入 concrete runtime materializer。最终 A7 仍需先冻结 reward
deployment identity，再从该 candidate 启动六条正式 20-step 路线。
同一 wheel 已上传服务器并安装到 tmpfs reward service venv，`python -I` 证明
`visual_rl` 与两个 strict reward app 均从 site-packages 导入。旧 source-tree 的
8092/8093 origins 已按 Gunicorn lifecycle 退出且无残留 worker/manager；frozen-wheel
origins 已从中立 `/tmp` CWD 在原端口启动，等待真实 health/score receipt 后生成 final
marker。无需 remote reward 的 Flow×SD3 final 20-step 已先在 GPU 3 启动，初始 driver
占用 511 MiB，stdout/PID/exitcode/15 秒显存 CSV 均写入 final tmpfs evidence root。
两个 frozen-wheel origins 现均返回 strict-v2 HTTP 200；general 的真实 HPS request
返回有限分数 `0.23296520113945007`，release marker 已按 exact schema 生成并逐文件
哈希复制到 final artifact path，manifest SHA-256 为
`648cebfbbc4eabdf6003022ee208c3d6653d207e46571e96c5f38600f8cff123`。
TempFlow×SD3 final 20-step 已在释放后的 GPU 6 启动，Flow×Wan final 20-step 已在
general marker 可用后于 GPU 7 启动；三条正式任务均由同一 launcher 绑定真实 trainer
PID 与独立 15 秒显存 CSV，不读取逐步日志。
3D frozen-wheel origin 的真实 DA3/GS/Qwen request 也返回有限分数
`1.9351264214596995`，exact-schema marker 已逐文件复核并复制到 final artifact path，
manifest SHA-256 为
`fbceb8637bd068e29cbfc111bb38565b987901ea050839e67d2dcac43da60af0`。
至此 source `56507f6e…`、wheel `6f1533ef…`、六份 final config 及两个 reward
deployment identity 均已冻结；剩余三条 Wan route 等待 GPU 释放后启动。
旧 Flash bring-up 使用 stochastic 修复前源码，已在保留 partial artifacts 的前提下
正常停止；其专用旧 8091 reward origin 的端口与 GPU manager 也已释放。final
Flash×Wan 20-step 随即在 GPU 2 从 511 MiB baseline 启动。旧 NFS service env 遗留一个
D-state resource tracker 与 reparented zombie，但二者不再监听端口或持有 GPU；该清理
缺陷只记录为 diagnostic，不进入 frozen-wheel final deployment。
World-R1 core/release 已分别挂到 GPU 3/GPU 6 的一次性 success-dependent queue：
只有前序 Flow×SD3/TempFlow×SD3 出现 exitcode 0 与 SUCCESS 后才启动，不读取中间
step 状态；任何前序失败都会令 queue fail closed。这避免持续轮询，也避免 GPU 释放后
空闲等待人工操作。
终态验收不依赖人工目测：`audit_a7_route.py` 在进程退出后调用 canonical
`audit_run`，并额外要求 exit 0、fresh 20/20、step-20 progress、正且有限的最终梯度、
frozen code/config/reward identity、无 OOM/异常签名，以及绑定 trainer PID/物理 GPU
且以 dead-target row 结束的显存 CSV；`audit_a7_matrix.py` 再要求六条 route 精确覆盖
同一 freeze/config/reward 矩阵。两者仅在终态执行，不能把 live/partial run 判为成功。

2026-08-08 架构补充审计：`composition.compatibility.graph` 是一条无生产
caller、仅被自身旧测试引用的 algorithm-optional 第二兼容路径。该路径及
`match_legacy_model_dynamics` 已物理删除，架构门禁明确禁止该模块重现；
production compiler/runtime 只能使用必须携带 `AlgorithmRequirements` 的 canonical
matcher。直接相关门禁 `57 passed`，Ruff 通过；因源码又发生变化，上述
GPU engineering jobs 仍只是 diagnostic，不能代替最终冻结版 A7。该模块也已
加入 clean-wheel 禁止清单，定向 wheel/source 门禁 `3 passed`，Ruff 通过。
双向依赖门禁同时从“允许导入 `visual_rl.models` 包根”收紧为符号级 ABI：
algorithm 只能使用 `ModelInput`/`ModelLatentSpec`/行投影 payload 与 scheduler
blueprint/context 符号；即使未来包根误重导出 `SD3Adapter`/`WanT2VAdapter`，也无法
绕过 concrete-model import gate。同一门禁也扫描配置中的动态 class-path 字符串，
禁止以 `"visual_rl.models.implementations...:Class"` 或反向 algorithm class path 绕过
Python import 检查。相关双向依赖门禁及动态引用门禁均通过，Ruff 通过。
算法运行实现中原先从宽泛 `visual_rl.models` 包根导入的四处类型引用，
已改为直接依赖 `visual_rl.models.interface`；算法域现在只跨界访问
`models.interface` 与 `models.scheduler` 两个窄端口模块。相关 rollout/recompute
语义回归 `18 passed`，最终架构门禁与 Ruff 通过。
同时新增条件分支级别的 AST 门禁：`models/**` 不得在 `if/match` 中按
Flow/TempFlow/Flash/World-R1 策略名分支，`algorithms/modules/**` 不得按 SD3/Wan
模型名分支。这防止不通过 import 却用字符串/枚举将两域重新耦合；相关
门禁 `2 passed`，Ruff 通过。
六份正式 schema-v2 配置新增 A7 envelope 门禁：每条 route 的
`max_optimizer_steps` 不得低于 20（release-surrogate 固定为 150），全部
`transition_window_size=1`，四份 Wan 配置固定 `row_microbatch_size=1`。该门禁
只防止验收几何回退，不代替真实 GPU 证据；定向测试与 Ruff 通过。

## 16. 决策记录协议

实现时遇到本文件未覆盖的问题，先追加一行并明确影响，再修改代码：

| 日期 | 问题 | 可选方案 | 决定 | 原因 | 影响的文件/Gate |
| --- | --- | --- | --- | --- | --- |
| 2026-08-03 | Reward 是否继续顶层独立 | 顶层 `rewards` / `algorithms/rewards` | `algorithms/rewards` | 按 post-training 大域聚合，同时保留窄接口和独立资源生命周期 | M3, A0 |
| 2026-08-03 | 是否保留 legacy facade | 长期兼容 / 单 slice 兼容 / 立即删除 | 只允许单 slice | 未发布 v0.8 不需要永久双栈；wheel 必须单一表面 | M1–M4, A6 |
| 2026-08-03 | World-R1 是否新算法 | 新 AlgorithmModule / Flow-GRPO integration recipe | integration recipe | 其差异是 camera/data/reward/phase，不是新的 policy objective | M2, A1 |
| 2026-08-03 | 何时开始真实实验 | 每 slice 后 / final freeze 后 | final M6 后 | 避免用旧 hash 证明已变更代码 | M7, A7 |
| 2026-08-03 | Registry 谁拥有 | 每域一套 Registry / composition 单一 Registry | domain catalog fragment + composition 单一 Registry | 既保留域内注册信息，又避免多套解析与冲突规则 | M0, A0 |
| 2026-08-03 | Catalog descriptor 类型归谁 | composition / core | `core.contracts.composition.ComponentDescriptor` | domain catalog 不应反向依赖 composition Registry | M0, A0 |
| 2026-08-03 | 静态 resolver 是否 import implementation | 继续 import / provider 与 loader 分开 | composition 只解析 import-safe provider，runtime gate 后加载 implementation | 防止静态兼容检查提前加载模型/CUDA依赖，构造权保持唯一 | M0–M2, A0–A2 |
| 2026-08-03 | Capability contracts 归谁 | composition / core | 全部跨域 capability DTO/enums 归 core | models/data/algorithms 不应依赖 composition | M0, A0–A1 |
| 2026-08-03 | Model–Dynamics ABI 时机 | load 后才检查 / load 前后两层 | static descriptor + concrete artifact blueprint | artifact load 前 fail-fast，load 后再校验实际 config/hash/type | M1, A1–A2 |
| 2026-08-03 | Models 是否可依赖 data | 完全同级 / 允许纯 DTO | 允许 `models→data` import-safe DTO | sample/preprocess 有明确 source owner，禁止 data 反向依赖 model implementation | M0–M1, A0 |
| 2026-08-03 | Data/Algorithm 是否读取 recipe | 直接读取 / compiler 投影 | compiler 输出 `SourcePlanSpec`/`AlgorithmMaterializationSpec`/`RewardPlanSpec`，artifact gate 另产出 source/reward binding | domain 只消费自己的 typed input，避免反向依赖 composition | M0–M2, A0–A1 |
| 2026-08-03 | Trajectory owner | rollout 与 data 各一套 / DTO 与 builder 分离 | data 拥有 immutable DTO，rollout 拥有 collector/builder | 避免两个同名 trajectory 模型，同时保留控制与存储边界 | M2/M4, A0 |
| 2026-08-03 | Reward resource owner | algorithm 或 runtime 全包 | algorithm 描述逻辑资源，runtime 管物理生命周期 | reward 语义可测试且设备/连接生命周期只有一个 owner | M3, A0 |
| 2026-08-03 | Checkpoint capture 边界 | artifacts 读取 live owner / participant snapshot | domain capture，runtime collect，artifacts commit | artifacts 不依赖具体 model/algorithm/live manager，safe-point 语义明确 | M4, A5 |
| 2026-08-03 | Branch topology owner | algorithm / data DTO | immutable topology 归 data trajectory DTO | rollout 可消费，data 不反向依赖 algorithm | M2/M4, A0 |
| 2026-08-03 | ExecutionTransformPlan owner | model / runtime / core | plan 归 core，executor 归 runtime | 跨 model/runtime 共享但不携带资源 | M0/M4, A0 |
| 2026-08-03 | gate_runner 包内 main | 保留 / scripts / 删除 | 删除 main，逻辑并入 compatibility evidence | 无第二 CLI/入口，wheel surface 单一 | M4, A6 |
| 2026-08-03 | Release build 是否信任已有 `build/`/egg-info | 复用缓存 / 构建前清理 | 安全清理 repo 内固定 `build/` 与 `visual_rl.egg-info` | setuptools 缓存会把已删除源码重新打进 wheel；必须拒绝 symlink 并从当前源码生成发布物 | M2.6/M6, A6 |
| 2026-08-03 | 验收计数 | 五组合 / 六配置 | 五组合、六条 route | World core/release 是同组合不同 integration，二者都需独立 20-step evidence | M6–M7, A7 |
| 2026-08-03 | 旧模型内的上游敏感参数迁到哪里 | 继续放 model / 使用通用默认 / recipe 显式锁定 | recipe 显式锁定 | TempFlow forward microbatch 和 World-R1 camera wrap 是算法/integration 语义，不属于 SD3/Wan；删除旧 adapter 时必须保持数值基线 | M1–M2, A1–A2 |
| 2026-08-03 | 冻结组件 offload 由谁控制 | model bool + `empty_cache` / runtime resource plan | `ComponentManager` + stage resource plan | residency、失败回滚和 prepared root 已由 runtime 统一管理；模型不得重复维护 GPU 状态机，`empty_cache` 不是正确性契约，显存峰值由 M5 指标和 M7 profile 验收 | M1/M5, A0/A7 |
| 2026-08-03 | 旧 SD3 native runner 如何退休 | 直接归档 / 继续作为 v0.8 证据 / 迁 canonical executor | 迁移 14 项到 canonical public-port executor，并强制区分 `cpu_fake_contract` 与 `pinned_upstream_native` | CPU fake 只证明合同完整性和 fail-closed，不能冒充 native numerical parity；真实模型/CUDA 对比只在 M6 freeze 后执行 | M1/M6–M7, A3/A6–A8 |
| 2026-08-03 | M1–M4 物理路径变化是否兼容旧 identity | 保留旧 alias / 重写旧 manifest / pre-release release cut | v0.8 pre-release release cut；新 interface/implementation/config class paths 是 canonical identity，不保留旧 alias，M6 后冻结 | class/config path 已进入 resolved recipe、component contract 和 preprocess identity；旧 alias 会重新引入双栈并破坏 A6。M6 前产物只作 diagnostic | M1–M6, A1/A5/A6 |
| 2026-08-03 | release-cut 前 checkpoint 是否允许 resume | 路径映射后恢复 / 原地升级 / fail closed | fail closed，旧 checkpoint 只读归档，不做 class-path rewrite；只在 M6 final source 上生成正式 resume fixture | recipe/component identity 改变代表实际代码与 contract 已改变；自动重写会绕过 compatibility gate 并制造虚假等价。恢复必须在 mutable state load 前拒绝 | M1/M4/M6, A5–A6 |
| 2026-08-03 | release-cut 前 preprocess cache 如何处理 | 沿用旧 key / 自动删除 / 新 identity miss-and-rebuild | 新 plan id/cache key，旧条目不得命中；显式提供旧 descriptor 时 fail closed，按新 identity 重建；不在迁移代码中自动删除用户缓存 | producer implementation/output schema、实际影响 preprocess bytes 的 config/transform、相关 model artifact 与 source content identity 属于 cache key；完整 model/algorithm/rollout/conditioner manifest 和 requirement-set id 只作 compatibility/audit，不能造成无关 cache miss | M1/M2.5/M4/M6, A3/A5 |
| 2026-08-03 | model catalog 在迁移期能否拥有 legacy Registry | models 内继续 `_MODELS` / models 只产 descriptor-provider fragment | models 只产 import-safe fragment/provider；Registry 状态始终由 composition/迁移期 registry adapter 拥有 | 否则模型域同时拥有声明、解析和全局 registry 状态，违反单一 owner；provider output 必须使用 core-owned DTO，不能形成 models→composition 反向依赖 | M1/M4, A0–A2 |
| 2026-08-03 | 算法内部 id 与 params 由谁决定 | recipe components / runtime trainer / algorithm blueprint | frozen algorithm config 生成含 canonical slot params 的 blueprint；compiler 投影 typed integration/execution policy | 只有 id 没有 params 的 blueprint 仍会迫使 recipe 成为第二事实来源；资源几何与算法数学必须分 owner | M2.1/M2.5, A0–A1 |
| 2026-08-03 | declaration provider 哪些字段进入 identity | 只 hash implementation/config / 完整 provider manifest | alias、implementation path、provider path、config-type path、interface、deps、frozen config、contract 全部进入 resolved identity | provider 替换或 config parser 漂移同样会改变语义；不锁定会让静态编译与 runtime 加载不一致却命中旧 checkpoint | M2.1/M2.5, A1/A5–A6 |
| 2026-08-03 | M2 class-path/recipe/checkpoint 如何迁移 | 每小步反复改 identity / 一次原子 release cut | shadow provider/catalog 壳层与最终实现先在旧 production manifest 后准备；compiler、recipe、runtime loader、canonical paths 于 M2.5 同批切换一次并删除 shim | 避免 production catalog/loader 提前消费未实现路径或产生多个不可比较的中间产物；shadow declaration catalog 的未物化路径只由精确 debt 管理；v0.8 未发布且已有 fail-closed release-cut 决策，不保留长期 alias | M2.1–M2.5, A1/A5–A6 |
| 2026-08-03 | M2.1 catalog 指向尚未物理迁移的最终实现如何门禁 | 创建空壳 / 宽泛跳过 / 精确阶段债务 | 只登记 10 条 `source,target,symbol` 精确且双向 stale-checked 的 missing-target debt；实现落地与债务删除必须原子发生，M2.4 退出时归零并移除机制 | 空壳会伪造可加载 runtime，宽泛 allowlist 会掩盖拼写和新缺失；精确债务既允许声明先行，又确保每次物理迁移都会迫使门禁更新 | M2.1–M2.4, A0/A1/A6 |
| 2026-08-03 | legacy optimizers 如何处理 | 整包改名 / 长期 facade / 只迁数值叶子 | 只迁 clipped/objective/reference 数值叶子与有价值 oracle，删除旧 PolicyAlgorithm/UpdateEngine 链 | production 已走新 credit + slot-streaming；整包迁移会保留第二算法实现且重新开放旧 DTO 路径 | M2.2, A0/A3–A4 |
| 2026-08-03 | full-T recompute/update 兼容入口是否保留 | production fallback / deprecated facade / test-only oracle | oracle 迁 tests/native harness，production 只保留 slot-streaming | full-T 路径会重新保留 T 个 autograd graph，是 K=8/T=28 OOM 的直接回归入口 | M2.2/M5, A4/A7 |
| 2026-08-03 | M2 是否必须删除整个 `training` | M2 全删 / M4 再全删 | M2 删除并迁走全部 algorithm-owned 文件；仅明确属于 data/runtime/composition 收敛的文件可暂留，M4 退出时整个 namespace 物理删除 | 避免为了目录纯度把 data/runtime 职责错误塞进 algorithms，同时禁止旧 training 继续拥有算法语义 | M2/M4, A0/A6 |
| 2026-08-03 | 普通 Flow×Wan 与 World-R1 Dynamics 如何区分 | 共用 `world_r1` 名称 / recipe-name 分支 / capability 投影 | exact Flow×Wan 使用中立 Wan profile；World-R1 conditioned profile 由 algorithm requirement、likelihood 与 camera conditioner capability 推导 | World-R1 是 integration recipe；普通跨模型组合不应泄漏其名称，runtime 也不得按 recipe 名分支 | M2.3/M2.5, A1–A3 |
| 2026-08-03 | Dataset source 语义与绝对路径是否使用同一 DTO/identity | 整个旧 `SourceLoadPlan` 进 resolved identity / 路径完全不审计 / semantic、binding、request 三层 | compiler 产出 path-free `SourcePlanSpec`（`source_id/selector/artifact_ref/artifact_kind/format`）并纳入 resolved identity；artifact/environment gate 产出 `SourceLocationBinding(location, expected_content_identity)`；loader 消费 `SourceLoadRequest`，只打开一次并以同一稳定 byte snapshot 计算 identity 和按显式 format 解析；删除当前混合 DTO | 绝对路径随机器变化，不能污染可移植 recipe identity；后缀目前会改变 text/jsonl 解析，说明 format 是语义；hash 后 reopen 仍有竞态，只有验证并解析同一 bytes 才关闭 TOCTOU | M2.5/M4, A1/A3/A5 |
| 2026-08-03 | beta、group size 与 rollout 字段的唯一 owner | execution 与 algorithm 双写 / 全部归 rollout config / 数学和运行几何分层 | beta、steps、branch/topology/selection 只归 frozen `algorithm.params`/blueprint；completion group size、forward/decode/storage 归 `ExecutionPolicySpec`；row/transition recompute 只归 `TrainingSpec.policy_recompute`；TempFlow 要求 group size 等于 branch count | beta/selection 改变目标或采样定义；group/batch/microbatch/storage/recompute 是运行几何但仍进入 recipe/checkpoint identity。两个 typed spec 分域可以，但同一字段不得复制 | M2.5, A0/A1/A4 |
| 2026-08-03 | Reward 在 M2.5 与 M3 如何分界 | 全留 M3 / M2.5 连数值热路径一起迁 / 先切声明再切执行 | M2.5 建 canonical interface、四个 import-safe provider/catalog 与唯一 `RewardPlanSpec` 链，runtime adapter 暂留旧路径但实现新接口；M3 再迁 client/bridge/pool/factory/implementation path | compiler 原子切换前必须拥有完整 reward declaration identity；把 Torch/NumPy/remote client 热路径塞入 M2.5 会扩大算法 identity cut。M3 路径变更是独立 fail-closed reward cut | M2.5/M3, A0/A1/A5–A6 |
| 2026-08-03 | M2.5 最小原子切换边界 | 只替换 compiler / 分层长期双读 / 全消费者同批切换 | schema/builtins/configs、Resolved/Materialized identity、preflight、runtime graph/production、binding/stage、checkpoint/manifest/resume 与旧 resolver/loader/projector/shim 同批切换 | 任何遗漏都会让 runtime 从 `semantic_config` 重建第二事实，或产生新静态 identity 配旧 mutable state；preprocess cache key 必须只含 model/preprocess producer identity，不含无关 algorithm/rollout identity | M2.5, A1/A3/A5–A6 |
| 2026-08-03 | Launch absolute path 是否决定 launch/resume 等价 | 纳入 launch id / raw manifest 全字段相等 / 仅审计位置并按内容身份判等 | absolute artifact locations 写入 launch audit 便于诊断，但不进入 recipe/materialized/launch ID，也不作为 resume 等价条件；resume 以 canonical spec、artifact content identity 与 runtime binding facts 判等 | 同一已校验 artifact 移动目录不应使可移植恢复失效；当前路径不进 launch_id 却被 raw manifest equality 拒绝是相互矛盾的双重规则，内容改变仍会由 materialized identity fail closed | M2.5/M4, A3/A5 |
| 2026-08-03 | Component loader 的 G1/G3 artifact 证据如何分层 | load 后才建 contract / capability 相等即放行 / declaration-bound G1 receipt + runtime G3 attestation | G1 用 recipe/slot/完整 declaration id/artifact-set/code identity/canonical interface 生成 `ComponentArtifactBinding` 与 gate，全部 gate 验证后才 import；G3 `RuntimeBoundContract` 引用同一 G1 contract 并补真实 load/prepare attestation | 当前 gate 只比较 `DeclaredContract`，相同 capability、不同 provider/config/implementation 可误过；若用现有 load 后 probe contract 又形成 gate→load→contract 循环 | M2.5, A1–A3/A5 |
| 2026-08-03 | Public algorithm provider 是否沿用 generic declaration ABI | 扩展所有 `ComponentDeclaration` / config duck typing / 专用 algorithm ABI | `algorithms.modules.declarations.AlgorithmDeclaration` 与专用 ABI 一次返回 component declaration + blueprint，requirements 只取 contract；descriptor 声明 provider ABI 且进入 identity，generic resolver 拒绝 specialized ABI | 普通 provider 不应携带算法专属 nullable 字段；多次调用 config/provider `describe_*` 会产生不一致的三份事实并让 runtime 再派生 | M2.5, A0–A1/A6 |
| 2026-08-03 | Preprocess compatibility receipt 是否进入 payload cache key | 完整 requirement set 全 hash / 完全忽略 compatibility / 分离 receipt 与 payload identity | requirement set/algorithm compatibility receipt 单独保存并进入 checkpoint audit；`PreprocessPlan`/cache key 只保留 producer/output/真正字节依赖、相关 artifact 与 source content identity | 当前 rollout manifest、algorithm plan、conditioner 与部分纯训练 model config 会造成相同 embedding 无谓 miss，既破坏跨算法复用，也掩盖 model/algorithm 解耦是否真实 | M2.5/M4, A1/A3/A5 |
| 2026-08-03 | Decoded media layout 由谁声明 | 用户 execution 字符串 / rollout 按 rank 猜测 / model typed output | `ModelAdapter.decode()` 返回 `DecodedMediaBatch(tensor, layout)`，layout 是 model decoder 真实输出并由 runtime 校验；`ExecutionPolicySpec` 不接受 decoded layout，若未来需要转置则作为显式 transform 产生新 typed output | 当前 `auto` 默认 BFCHW，显式 BFHWC 又不会验证/transpose，可能只改标签不改 tensor，导致 reward 静默读错轴；layout 不是显存 geometry | M1/M2.5, A0/A2–A3 |
| 2026-08-03 | model-bound Dynamics 投影如何读取算法数学与 integration 语义 | duck-type 具体 config / 按 recipe 名分支 / typed blueprint + typed integration policy | `AlgorithmBlueprint.beta` 与 slot params 是算法数学的唯一 compiler 输入；composition 只结合 model `dynamics_binding_family` 和 typed integration policy 选择 Dynamics declaration，禁止再次读取具体算法 config 或按 recipe/model 名称拼参数 | beta 原先虽在 frozen config 中，却不在 blueprint，compiler 构造 `AlgorithmMaterializationSpec` 时会被迫形成第二事实；World-R1 conditioned 语义必须是显式 integration fact，而不是内部 component override | M2.5, A0–A3/A6 |
| 2026-08-03 | `AlgorithmMaterializationSpec` 如何消费 execution geometry | 复制 `group_size`/paradigm/transform id / 信任 structural view 自报 id / 只引用完整 execution policy identity并在边界重算 | `ExecutionPolicySpec` 唯一拥有 group size、rollout microbatch/storage、precision 与 transform plan；materialization spec 只保存 `execution_policy_id`，完整 `spec_id` hash 该引用，另设不含 policy 的 `algorithm_semantics_id`；composition 用完整 canonical payload 生成 core-owned frozen `ExecutionPolicyReceipt`，algorithm 重算 policy/projection identity 后才投影 paradigm/cardinality | 复制字段会双写；任意 structural object 可伪造 id 后改变 group/transform；反之若 spec 保存 policy reference却排除于 `spec_id`，policy 漂移时 checkpoint state id 不变。receipt 保持 algorithms→composition 为零且关闭伪造边界 | M2.5, A0–A1/A4–A6 |
| 2026-08-03 | G1 对 specialized algorithm 锁哪一层 identity | 模糊 `declaration_id` 接受两种命名空间 / 直接锁 `algorithm-declaration` / 锁 runtime component declaration | `ComponentArtifactBinding.component_declaration_id` 只锁 `component-declaration.v1`；完整 `algorithm-declaration.v1`、blueprint 与 requirements 由 `ResolvedRecipe`/recipe id 锁定 | G1 验证要加载的具体 runtime component，实现门只需要统一 component declaration；public algorithm 的专用原子声明是更上层 recipe 事实，混入同一字段会模糊 ABI 与形成两种格式 | M2.5, A1–A3/A5 |
| 2026-08-03 | G1 如何证明完整 component graph 而非单 slot 自证 | 单组件裸 load / 只传 binding set / 外部 anchored exact load plan | artifact gate 从 `MaterializedRecipe` 派生 `ComponentLoadPlan(expected_recipe_id, expected_binding_set_id, exact slots/artifact names)`；`load_all` 在第一次 implementation import 前精确比较 plan、binding set、每项 recipe/declaration/binding/artifact 覆盖，单组件 `load` 也不得绕过完整图证明 | 只验证调用方给出的单 slot 会允许 stale recipe、slot 子集或 code-only reward artifact 静默通过；recipe anchor 不进入纯 launch-topology 的 environment DTO，避免职责污染 | M2.5, A1–A3/A5–A6 |
| 2026-08-03 | Dataset content 如何进入 `MaterializedRecipe` | 复用带绝对路径的 location binding / 退化为 mapping / 独立 path-free typed binding | 新增 `SourceContentBinding`，只含 `source_plan_id` 与各 artifact content identity；`SourceLocationBinding` 继续只用于 launch/load audit，稳定 loader 验证同一 byte snapshot | materialized/resume identity 必须跨机器可移植，但仍需 typed exact coverage；直接复用 location DTO 会把绝对路径带回 identity | M2.5, A1/A3/A5 |
| 2026-08-03 | Compatibility 报告哪些内容进入 recipe identity | 完整 issue/hint 文案 / 完全不 hash / 规范化规则结果与诊断展示分离 | identity 只保存 compatibility rule-set version、规范化 issue code/producer/consumer/required/provided 与 bindings；`hint` 等人类文案只进 source/launch inspection，不进 resolved fingerprint | 修改报错措辞不应使 checkpoint 失效，但省略规范化判断事实会无法审计当时为何允许或拒绝组合 | M2.5, A1–A3/A5 |
| 2026-08-08 | Wan 策略采样模式由模型 scheduler 还是算法 Dynamics 拥有 | 继承模型默认值 / trainer 临时覆盖 / typed Dynamics config + replay binding | 采样模式归 Dynamics；canonical Wan GRPO 投影显式绑定 stochastic，factory/replay/config identity 全部记录并在 runtime 检查一致 | scheduler 的 `stochastic_sampling=False` 是模型推理默认，不代表 GRPO policy transition；继承它会令 `action == mean` 并使 score-function 梯度恒为零。显式 Dynamics owner 保持 model/algorithm 分离，resume 时 mode drift 也会 fail closed | M2.3/M5/M7, A1–A5/A7 |

新增决定不得只写“为了方便”；必须说明 owner、依赖方向、identity、checkpoint 和验证后果。

## 17. 完成定义

只有同时满足以下条件，才能把目标标记完成：

1. 目标文件树和依赖矩阵落地，legacy namespace 不存在于源码和 wheel；
2. Model 不构造具体 algorithm/dynamics 实现，Algorithm 不 import 具体模型；
3. AlgorithmBlueprint 是 compiler 和 runtime 的唯一内部结构来源；
4. 五种组合的六条 route 使用同一 composition root，兼容与不兼容都在正确 gate 表现；
5. slot-streaming 不保留跨 timestep graph，forward partition 真正限制峰值 batch；
6. 全量本地测试、wheel contract、isolated install 通过；
7. final source/wheel 上六条 route 各至少 20 optimizer steps，无 GPU/CPU OOM；
8. resume、native parity 和 World phase-boundary 的精确证据齐全；
9. `Plan_8_2.md`、README、架构文档和 evidence 互相一致。
