# VisualRL 实验计划与证据账本

更新时间：2026-07-15

本文件是当前实验顺序、冻结条件、执行状态、验收门槛和结果边界的唯一事实来源。项目 Goal、架构与学习路线以 `docs/PROJECT_OVERVIEW.md` 为准；`experiments/INFRA_VALIDATION_WORKLOG.md` 是合并阶段的历史修改/验收快照，不再维护当前 backlog；每个具体实验的完整 recipe 和原始结果放在各自子目录中。

## 总规则

1. 每项实验在启动前冻结源码归档、数据、模型/reference、环境、seed 和门槛。
2. active 与 control 必须匹配除学习率以外的所有变量。
3. 失败 seed、失败颜色和失败样本不得从聚合结果中删除。
4. 前一硬门槛失败时停在当前规模定位，不用更长训练掩盖问题。
5. Tiny/Fake、真实推理、单步更新、resume、短训练、多 seed 效果是不同证据，不互相替代。
6. 不访问其他用户目录，不终止其他用户进程，不使用 root；GPU 被占用时等待或降级为单卡顺序执行。
7. 已冻结实验与 infra 修复必须分开。历史 E2b 已按该规则完成；其首次失败与 attempt 2 证据继续保留，任何后续修复都不得回写或覆盖旧 attempt。
8. Infra 验收分为两条通道：deterministic/audit 通道要求机械正确性和可复现性；performance/training 通道允许预注册浮点容差并用重复运行、多 seed 和 control 判断效果。两类结论不得互相替代。

## 当前状态总览

| ID | 实验 | 状态 | 当前结论/下一动作 |
|---|---|---|---|
| B0 | 合并后本地 acceptance | 已通过 | `2adfbfd` 完整 non-distributed：883 passed、2 skipped、5 deselected；真实双进程 Gloo：5 passed、885 deselected；Ruff、compileall 与 diff check 全部通过 |
| M0 | 当前 GitHub `main` | 已发布并通过本地与 P6 门槛 | `main` 已包含 `ee4a44f`：在 `a34a3b8` 基线上加入 Wan/SD3 gradient-checkpointing 显式 on/off、effective-state、provenance 和 resume drift 合同；927 passed、2 skipped、5 deselected，Gloo 5 passed，Ruff/compileall/diff check、独立 review 与 P6 固定 Wan 单步真实 GPU A/B 通过 |
| G0 | gradient-checkpointing 合同修复 | 已完成并发布 | `ee4a44f` 已进入 `main`。P6 证明固定 `128×128`、5 帧、4 denoising step、单 GPU、单步 Wan LoRA 下三对训练语义 exact，update peak median 下降 `3,190,562,816` bytes（`17.57%`）；不能外推到稳态吞吐、任意尺寸或质量 |
| B1 | 历史细粒度回归套件 | 已通过 | 从 Git 临时导出运行：190/190；不恢复旧测试文件 |
| B2 | 合并后真实 correctness gate | 已通过（SD3/W5/W6-general/W7b/O4） | W7b 的 480 个 gradient tensor 逐位一致；W5 六段 valid、两份 11-gate 比较 exact；`reward_general` 三方分数逐位一致且坏请求 fail closed；SD3 continuous/resume 16 个 exact gate 通过；O4 完成真实 HPS 单步和恢复后 reward 中断 fail-closed |
| A1 | 外部算法插件最小接入 | 已通过 | 独立模块零核心改动完成 config→rollout→reward→loss→update→metrics→checkpoint；checkpoint 记录外部类身份 |
| A2 | 算法互换 contract | 已通过 | TinyDiffusion 上 GRPO/full、TempFlow/branch、Flash/single-step 均通过严格 RolloutBatch、公共 metrics/manifest/checkpoint 与算法特有 credit metadata |
| A4 | scorer 插件装载与隔离 | 已通过（本地 + O4 真实 HPS） | 外部 batch scorer、方向、cache key/version、坏 shape、timeout×3、invalid/raise、健康 scorer 不污染全部通过；O4 进一步验证真实 HPS healthy request 和恢复后 pre-scorer 服务中断 fail-closed |
| A5/D-data1/D-data2 | 固定 HF 数据 provider 与统一追溯 | 已通过 | 固定 PartiPrompts revision 的 12 行离线快照与本地副本，经同一 runner 完成一步；metrics、rollout、adapter、规范化 SampleManifest 精确一致，24/24 样本可反查源行 |
| A6 | 新算法改动面审计 | 已完成，有 CLI 边界 | Python API 核心改动 0；新增外部 module/config/harness。通用 CLI 缺少从 YAML 自动发现外部 plugin 的机制，仍需薄启动器 |
| D-data7 | train/held-out 泄漏检测 | 已通过，并发现旧数据问题 | 三类合成泄漏均 fail-fast；旧 3,600/12 split 检出 2 对近重复，不能再用于正面效果 claim；T3b 90 条 fixture 审计干净 |
| D-data4 | 坏数据与生成媒体 cache | 已通过 | 空/重复/坏编码 prompt、缺/重复 manifest、损坏/半写 rollout media 均明确拒绝；skip/repeat 必须显式 |
| D-data5 | shuffle/cursor/resume | 单进程严格通过，多卡不支持 | 连续6步与3+resume 的 metrics/manifest/adapter/rollout 精确一致；完整 epoch 恰好覆盖一次；WORLD_SIZE=2 在输出前拒绝 |
| D-data6 | reward cache 并发、失效与恢复 | 已通过 | 10 进程同 key 压测无半写/临时文件；损坏条目按 hash 隔离并重算；metadata/media/scorer version 变化都会失效 |
| C3/C4 | checkpoint 原子发布、完整性与 schema 迁移 | 已通过 | 三阶段故障注入不误发 latest；adapter/state/metadata/额外文件损坏在参数恢复前拒绝；checkpoint v1/v2 必须显式迁移到 v4、旧 manifest 必须显式迁移到 v2，未知 schema fail closed |
| C5 | metrics/manifest/cache/checkpoint/latest 交叉一致性 | 已通过 | 两步8样本身份链闭合，双次审计精确相同；reward/prompt/sample ID/seed/model/config 六类篡改全部检出 |
| C6 | 默认 BF16/CUDA 容差审计 | 已完成，精确门槛失败 | 同 checkpoint 三分支逻辑一致，但原预注册 adapter exact、metrics/held-out `1e-6` 全失败；精确 resume 必须使用 deterministic runtime |
| Q4/Q5/Q6 | reward-hacking、统计与报告重建 | 已通过 | 216 个 raw pairs 完整保留，bootstrap 精确复现；执行/像素 gate 分离但失败结论不变；JSON/SVG/HTML 两次重建 hash 一致 |
| Q1 | 固定独立 PickScore | scorer 通过，训练效果失败 | 432项两遍评分精确一致、control精确0；active mean `-0.001512`、CI跨0、仅1/3 seed正且三色均负，进一步禁止T5扩容 |
| Q2 | 人工双盲面板 | 已冻结，等待真实评审 | 36 active +12 identical controls，opaque A/B 与独立 key；至少2名真实评审，禁止AI代填 |
| O1 | 五阶段真实进程中断恢复 | 已通过 | reward/cache/optimizer/checkpoint/latest 后 child exit 91；恢复到 step2 与连续 reference 的 adapter/metrics/manifest 精确相同，无重复 step/样本 |
| O3/O5 | 写入失败恢复与生命周期状态 | 已通过 | completed/failed/live/stale/missing 可区分且聚合 fail-closed；ENOSPC/半写/只读不推进 latest，恢复后 exact，stale tmp 清零 |
| O2 | 真实 CUDA OOM | 已通过 fail-safe；自动降级不支持 | GPU2 申请192 GiB后明确 failed/invalid，step0 且无 metrics/latest/checkpoint，显存释放；同环境16 px control完成一步 |
| O4 | 真实 HPS reward 服务中断 | 已通过（窄边界） | GPU2 Wan + GPU4 HPS：健康 step1 完成真实更新；从 step1 恢复后，step2 请求进入但 scorer 前终止自有服务，1 次请求/0 重试/0 optimizer/无 step2 artifact，step1 持久状态不变，进程、端口和 GPU 清理完成。下一步扩展 mid-scorer/commit 故障点与稳态 reward profiling |
| S3 | GPU/进程所有权安全 | 已通过（单机、自有进程） | 普通 UID + GPU2；错 start tick 不发信号，只有 UID/PID start/command token 全匹配才终止自建 child |
| E1 | SD3 3-seed active/control，10 steps | 已完成，效果门槛失败 | 六个 run 均 execution valid；聚合 CI 跨 0，blue 三个 seed 全为负；禁止扩大效果规模 |
| E2 | SD3 真实 resume 等价性 | 已完成，严格等价失败 | 能完整恢复并继续，但独立运行在 resume 前已出现数值漂移，最终 LoRA 不相等 |
| E2b | SD3 独立运行确定性审计 | 已完成，默认模式严格复现失败 | 原门槛和失败证据保持不变；D0 已定位到 backward，D1/E2c 已用显式确定性模式解决 |
| D0 | SD3 单步因果定位 | 已完成 | checkpoint/RNG/SDE/forward 精确一致；首次差异为 368/382 个 backward gradients |
| D1 | 确定性五步跨 GPU 复现 | 已完成，严格通过 | GPU2/GPU3 从同一 step-5 checkpoint 到 step 10：adapter、state、metrics、held-out、PNG 全部精确一致 |
| E2c | 内置确定性真实 resume | 已完成，严格通过 | 连续 2-step 与 1+resume-to-2 的 adapter、optimizer/RNG、metrics 和 36 张最终图精确一致 |
| I0 | checkpoint 指纹 v2 | 已完成，真实 SD3 严格通过 | 数据内容身份与来源路径分离；12/12 公共、190/190 历史回归和真实 moved-path resume 全部通过 |
| E3 | 原生 TempFlow 与 infra 数值对齐 | E3a v6、E3b v3 已通过 | 共享 rollout 的训练数学与显式随机输入下的完整 sampler 均对齐；效果结论仍由 E3e 单独决定 |
| E3e | TempFlow reference-compatible 效果验证 | 已完成，效果门槛失败 | 六个 run 有效且 control 精确为零；CI 跨 0、2/3 seed 为正、red 为负、两个 active 像素护栏失败；禁止扩容 |
| T2 | TempFlow 效果失败诊断 | T2c 已通过 | 全量数据严格平衡；三色各一步的 3-step active 总体和每色均为正、control 精确零、像素护栏通过；只解锁分层平衡的 3-seed 10-step 复测，长程仍锁定 |
| T3 | TempFlow policy-identity 效果复测 | 机械通过，效果失败 | 六 run/60 步零漂移零 clipping；active mean `+0.00851`、CI 跨 0、2/3 seed 正向、red 与两个像素护栏失败；进入 T2c，不扩容 |
| T3b | TempFlow 分层平衡窗口效果复测 | 机械/reward 通过，安全效果失败 | active mean `+0.10386`、CI95 下界 `+0.06757`、3/3 seed 与三色均值正；controls 精确零，但 seed307/419 像素护栏失败，禁止 20-step 扩容 |
| T4 | TempFlow policy-identity resume 等价 | 已通过 | 连续10-step 与5+resume-to-10 的 LoRA、state、metrics、manifest、step0/5/10 原始评估和 PNG 全部严格一致 |
| E4 | 独立语义评分与人工固定面板 | 独立评分已完成，人工面板待评审 | Q1/Q3 已完成固定 PickScore；Q2 面板已冻结，仍需至少两名真实评审 |
| E5 | SD3 20/50-step 多 seed | 未解锁 | 仅在 E1 跨 run 全部门槛通过后启动 |
| V1 | Flash-GRPO/Wan 真实路径 | W0-W7b 已通过 | 原生同/异构 selected sampler 的 media/embedding/latent/logprob/loss/480 梯度逐位一致；轨迹状态存储 10.5x 减少，但吞吐仍待 P2 实测 |
| V2 | World-R1 reward + Wan 闭环 | W6 已通过；`2adfbfd` general parity 已复验 | post-merge `reward_general` direct/reference/infra 均为 `[0.260009765625, 0.1943359375]`、两组差值为 0；坏请求 500、silent fallback false。真实 DA3+Qwen 3D 仍沿用既有证据 |
| W8 | Wan 真实HPS 3-seed active/control | 已完成：机械通过、效果失败 | 12/12 run有效，active更新/control精确零、像素护栏通过；World/Flash配对均值均为负且CI跨0。后续 Q3 已确认独立效果同样失败，W9继续锁定 |
| Q3 | Wan独立视频质量诊断 | 已完成：安全通过、独立效果失败 | 240视频/1,200帧双遍PickScore与step0身份精确；无塌缩且时间/清晰度护栏通过，但World/Flash独立效果CI均跨0。该失败诊断已闭环，后续资源工作由 P3/P6 承接 |
| P1 | 单卡阶段 profiler | 主要阶段已覆盖 | W8 已记录真实 Wan load/rollout/reward/recompute-backward-optimizer/cache/checkpoint/artifact；O4 冷启动 step 的 reward 占 87.91%。下一步先做 reward warm/cold 与批处理稳态 A/B，再做 native 对照 |
| P3 | 单卡 32GB 可行域 | 部分完成；旧 GC A/B 无效，P6 新 A/B 通过 | 15 个采样单元和 3 个一步训练单元通过；`480x832/5帧/4-step` 一步训练峰值 `28,336 MiB`。旧 silent-no-op 结果仍作废；P6 已用修复后的 effective-state 合同证明小尺寸单步 update 显存下降，仍缺多 prompt/seed、OOM bracket、其他尺寸与真实 reward 共驻 |
| P5 | HPSv2 prompt-group call coalescing | 微基准通过，完整训练待验证 | 同一 GPU/进程/两张同 prompt 图片下，一次 list call 与两次单图 call reward 逐位一致，调用时间均值改善 `2.084x`；远端 HPSv2 1.2.0 的 list 分支内部仍逐图 forward，因此这只证明减少重复 scorer 调用的潜力，不是 tensor/GPU batching。下一步先完善 P7 单 GPU 合同，再以固定 payload 与完整 O4 one-step A/B 验证顺序、fail-closed、显存和端到端收益 |
| P6 | Wan gradient-checkpointing 严格 A/B | 已通过（固定单步配置） | schema-v2 attempt 2 的 8 个独立进程全部 valid；三对 measured 的 14 项 rollout/reward/logprob/gradient/state/update 摘要全部 exact，on/off update peak median 为 `14,966,723,072 / 18,157,285,888` bytes，下降 `3,190,562,816` bytes（`17.5718%`）。attempt 1 因把 executor timing 纳入 reward digest 而不具比较资格，保留为失败证据；结论不覆盖稳态吞吐或质量 |
| P7 | World-R1 prompt-group call-coalescing patch | CPU manager contract 通过，真实 instance/HPS/GPU 待验证 | 冻结上游源码和 patch，13 个 manager mock tests 通过；独立审计确认它减少每个 prompt 的 `hpsv2.score` 调用次数，但不是一次 batched GPU forward。发布前先补真实 `GeneralRewardInstance`、异常后锁恢复与单可见 GPU fail-closed，并只在可信 loopback wrapper 中运行；随后做固定 payload HPS A/B 和完整 O4 one-step correctness/吞吐 A/B |
| MG1 | 真实双卡 DDP/NCCL correctness | 未完成 | CPU/Gloo 合同已通过，但真实 GPU 必须验证相同 effective batch、完整 prompt group 不拆分、无重复样本、rank 一致 update/failure/rollback、checkpoint/resume 和资源清理；若当前 sampler/cursor 合同不足，先修产品再实验 |
| MG2 | 双卡扩展效率 | 锁定 | 仅在 MG1 通过后，以相同语义的单卡 reference 比较吞吐、step time、通信占比和每卡峰值显存；两张或多张 GPU 各跑独立 run 不计 DDP 扩展 |
| W10 | World-R1 / Flash-GRPO 大型多 reward 验证 | 已规划，等待近期门槛 | 保留 W8 低分辨率、单 HPS、10-step 效果失败结论；P7、doctor/高层 API 和 10–100 step 稳态门槛完成后，再按复杂视频 prompt、共享多维视频 reward、World-R1 3D reward、独立 held-out evaluator 和 240-step × 3-seed active/control 计划推进；本阶段仍优先于 WM0-minWM |
| WM0-WM3 | minWM 交互式世界模型方向 | 已规划，未启动 | 不替换 Wan；以 Wan2.1/minWM checkpoint 为外部 backbone/runtime，先完成 action-conditioned 因果推理与统一数据合同，再按 zero-update、LoRA 单步、resume、active/control 门槛决定是否进入 RL post-training |

## B0/B1：当前代码回归基线

### 结果

- `2adfbfd` 完整 non-distributed suite：883 passed、2 skipped、5 deselected（2026-07-15）。
- `2adfbfd` 真实双进程 Gloo distributed suite：5 passed、885 deselected；覆盖跨 rank 更新、回滚、Flash 全局 coefficient oracle 与 microbatch `no_sync()` 通信次数 oracle。
- 合并前独立保留的公共 acceptance 为 27 passed、2 skipped，历史细粒度 suite 为 190 passed；它们现在已被纳入更大的合并后套件，不再作为当前主基线单独计数。
- `visual_rl/`、`tests/`、`scripts/`、`train.py` 的 Ruff 与 compileall 通过。
- 全工作区 `git diff --check` 当前通过。

### 证据边界

只证明当前 CPU/Tiny/Fake 行为与真实 CPU/Gloo 分布式合同未见回归，不证明真实 SD3/Wan GPU 数值、NCCL 行为或训练效果；后者必须由新的远端 post-merge gate 给出。

## B2：`2adfbfd` 合并后 Wan correctness gate

- 修复后的 dtype 合同是：只把送入 transformer 的副本转换为 BF16；SDE 使用的 current/next latent 保留采样轨迹中的原始 FP32，不允许先经过 BF16 round-trip。非 BF16 可精确表示数值的本地回归已固定这条边界。
- W7b 在真实 Wan 异构 selected-index 路径上严格通过：media、embedding、current/next latent、timestep、old/new log-prob、KL、loss 和 480 个 gradient tensor 均与独立 scalar reference 逐位一致，参数保持不变。
- W5 attempt 1 在训练前失败：实验 harness 先调用 `torch.cuda.*`，后构造启用 deterministic runtime 的 runner，因而被 runtime guard 正确拒绝。这是 harness 启动顺序问题，不是模型训练或 resume 数值失败；失败目录与日志保留。
- 修正 harness 后使用新 attempt 2 目录重跑。World-R1 与 Flash-GRPO 各自的 continuous-to-2、split-to-1、fresh-process resume-to-2 共六段 run 均 `valid=true`；两份比较各自的 11 个 exact gate 全部通过。
- `reward_general` post-merge attempt 1 通过：direct、World-R1 reference HTTP 与 infra client 三路分数均为 `[0.260009765625, 0.1943359375]`，两组比较差值为 0；坏图像与坏 pickle 均返回 HTTP 500，`silent_fallback_detected=false`。该探针只使用 loopback 上的 legacy 协议，不开放外部服务面。
- 完整非 checkpoint 证据索引见 [WAN_RESULTS_2adfbfd.md](postmerge_validation_20260715/WAN_RESULTS_2adfbfd.md)。

W5/W6/W7b 证据确认了 Wan sampler/recompute、deterministic resume 与 `reward_general` 协议 parity。后续 O4 已在 `d6ce72a` 上补齐真实 HPS 单步训练和恢复后 pre-scorer 服务中断门槛；SD3 post-merge continuous/resume 的 16 个 exact gate 也已通过。它们仍不改变既有 W8/Q3 效果失败结论，也不解锁质量提升、速度提升、P2 或 W9 声明。

## E1：SD3 多 seed 10-step active/control

完整冻结配置：`experiments/sd3_multiseed_10step_20260713/recipe.json`。

### 设计

- 训练 seeds：201、307、419。
- 每个 seed：active + zero-LR control。
- SD3.5 Medium、256 px、20 diffusion steps、10 optimizer steps。
- TempFlow branching，branch count 2，LoRA rank 8。
- guarded RGB reward。
- 12 条平衡 held-out prompts × 3 evaluation seeds = 每 run 36 个配对样本。
- GPU2 跑 active，GPU3 跑 matched control；按 seed 分三波顺序执行。

### 预注册跨 run 门槛

- 六个 run 全部 execution valid 且 pixel guardrail 通过。
- active training seed 数量至少 3。
- active 正向 seed 比例至少 0.8。
- active 分层 bootstrap CI95 下界大于 0。
- red/green/blue 三类聚合均值全部大于 0。
- active-control CI95 下界大于 0。
- active 均值超过 control eval-cluster RMS 的两倍。

### 当前执行记录

| Seed | Active | Control | 结果 |
|---:|---|---|---|
| 201 | 完成 | 完成 | active 参数 L2 `0.053933`，平均 delta `+0.001241`，CI95 `[-0.000628, +0.004489]`；blue `-0.000517`、green `+0.002153`、red `+0.002087`；control 参数和评估 delta 精确为 0；像素护栏通过 |
| 307 | 完成 | 完成 | active 参数 L2 `0.053120`，平均 delta `+0.001561`，CI95 `[-0.000671, +0.003507]`；blue `-0.000099`、green `+0.003542`、red `+0.001239`；control 参数和评估 delta 精确为 0；像素护栏通过；模型加载曾受共享存储 I/O 拥塞影响 |
| 419 | 完成 | 完成 | active 参数 L2 `0.054443`，平均 delta `+0.000009`，CI95 `[-0.001133, +0.001312]`；blue `-0.002447`、green `-0.001034`、red `+0.003507`；control 参数和评估 delta 精确为 0；像素护栏通过 |

### 聚合结论

- active 三 seed 聚合均值 `+0.000937`，分层 CI95 `[-0.000317, +0.002383]`。
- active-control CI95 同样跨 0。
- blue 聚合 `-0.001021`，且三个 training seed 全为负；green `+0.001554`；red `+0.002278`。
- 通过 execution、seed count、positive-seed fraction 和 control-noise 门槛。
- 未通过 active CI、active-control CI、every-color-positive 三个门槛。
- `eligible_for_effectiveness_claim=false`；E5 20/50-step 效果实验保持锁定。

### 晋级规则

- E1 完成后无论效果是否通过，都继续 E2，因为 resume 是 infra 正确性属性。
- 只有 E1 所有跨 run 门槛通过，才解锁 E5 的 20/50-step 效果实验。
- 若门槛失败，保留全部结果并分析 seed、颜色、prompt 与 reward 的失败结构，不直接延长训练。

## E2：SD3 真实 checkpoint/resume 等价性

### 设计草案

- 固定 E1 中一个 training seed 和相同模型/数据/recipe。
- A：连续运行到 10 steps。
- B：运行 5 steps，完全退出进程，从 checkpoint resume 到 10 steps。
- 比较 LoRA、optimizer、RNG、dataset cursor、metrics 5-9、manifest、held-out 和 checkpoint identity。

### 硬门槛

- base transformer hash 不变；只允许 LoRA 更新。
- LoRA/optimizer state 在运行前声明的容差内等价。
- step、prompt 顺序、rollout seed、manifest、metrics 无重复或缺失。
- 不兼容 source/config/data identity 必须拒绝 resume。

### 当前执行记录

- `split_to_5`：execution valid，5 行 metrics，checkpoint 包含 LoRA adapter、training state、adapter metadata 和 step/config identity；参数 delta L2 `0.036917`。
- step-5 adapter SHA256：`4490296c3966a5b3fc3a6806eba4bd2fab6f692dae9bb3bd5017d685b0c2ba13`。
- `resume_to_10`：独立 Python 进程成功加载 step-5 checkpoint，执行 5 个新增 step，到达绝对 step 10；execution、数值、参数、held-out 和像素护栏均通过。
- 严格等价失败：连续 adapter SHA256 `087d9a6b...`，resume adapter SHA256 `0aa20f0a...`；LoRA tensor `max_abs=0.000127196`、`L2=0.020282`。
- step 5-9 的最大 `reward_mean` 差 `0.006129`，最大 `grad_norm` 差 `0.007264`；held-out mean delta 相差 `0.000310`。
- 诊断发现新的 split run 在 resume 发生前的 steps 1-4 已与旧 continuous reference 轻微分叉，而 resume 后 prompt 顺序完全一致；因此当前设计不能把最终漂移全部归因于 checkpoint，需先做 E2b 独立运行确定性审计。

## E2b：同一 checkpoint 的独立运行确定性审计

### 首次启动结果

- GPU2、GPU3 两条分支都在加载模型后、进入第一个训练 step 前退出。
- checkpoint 训练语义指纹为 `39c5f87d...a76870a5`，启动配置指纹为 `8145f101...20669f`；resume guard 按设计拒绝加载。
- 只读重算精确确认：唯一相关差异是训练/评估文件从 E2 原绝对路径换成了 E2b 复制路径；两组文件的 SHA256 分别完全一致，但 `dataset.path` 和 `evaluation.path` 属于当前指纹语义。
- 两条分支均没有 metrics、summary、held-out 或新 checkpoint，因此不能比较 branch A/B/C，也不能据此判断 BF16/CUDA 是否确定。
- 该保护机制按当前实现正确 fail-fast，但也暴露了可迁移性问题：已经有内容 hash 时仍绑定绝对路径，会拒绝内容完全相同的数据副本。
- 已下载两份失败日志和 PID 记录，未下载 checkpoint；完整说明见 `experiments/sd3_runtime_determinism_20260713/RESULTS.md`。

### 当前门槛与下一动作

首次 attempt 的 `all_branches_execution_valid` 未通过，失败证据已保留且未覆盖。修正版 recipe 随后冻结为 `experiments/sd3_runtime_determinism_20260713/attempt_2_recipe.json`：

- source checkpoint 保持 `sd3_resume_equivalence_20260713/runs/split_to_5/checkpoint_000005`；
- 训练数据恢复为 `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_resume_equivalence_20260713/data/train.txt`；
- 评估数据恢复为 `/home/v-qiaoqifan/visual_rl_experiments/visualrl_remote_cli_smoke/sd3_resume_equivalence_20260713/data/heldout_pilot.txt`；
- 只允许改变 `paths.output_dir`、`paths.resume_from`、`train.max_steps` 和 `train.save_every`；
- GPU2/GPU3 使用新的 `attempt_2_repeat_gpu2` / `attempt_2_repeat_gpu3` 输出目录，不覆盖 attempt 1 或 E2 reference；
- 启动前只读重算的指纹必须等于 `39c5f87d...a76870a5`，否则继续 fail-fast，不进入训练。

### Attempt 2 执行与严格比较

- 启动前指纹预检严格匹配 `39c5f87d...a76870a5`；运行期间未修改 checkpoint 或 fingerprint 实现。
- A（已有 GPU2 reference）、B（新 GPU2）、C（新 GPU3）均 `valid=true`，各 5 行 metrics，正确从 base step 5 到 target step 10。
- 三条分支的 config fingerprint 相同，steps 5-9、prompt、seed、dataset index 和 branch timestep 序列完全一致。
- 最终 adapter SHA256 三者各不相同：A `0aa20f0a...`、B `a2cd2708...`、C `f9eb763a...`。
- A/B 同为 GPU2 独立运行，adapter `max_abs=0.000051165`、`L2=0.012349`；最大训练 metric 差 `0.016186`，held-out mean 差 `0.000458`。
- B/C adapter `max_abs=0.000048060`、`L2=0.010011`；最大训练 metric 差 `0.011257`，held-out mean 差 `0.001113`。
- 当前 runtime 为 PyTorch 2.11.0 + CUDA 13.0，deterministic algorithms 关闭、cuDNN deterministic 关闭、cuDNN TF32 开启。

执行闭环和顺序恢复门槛通过，但 adapter 精确相等、`1e-6` metrics 容差和 `1e-6` held-out 容差失败。因此 E2b 自身的 `eligible_for_exact_reproducibility_claim=false` 保持不变，不事后放宽门槛。后续 D0、D1、E2c 已按新实验 ID 完成定位、运行时修复和内置 resume 验证；I0 指纹修复仍与确定性实验分开实施。

## D0：SD3 单步因果定位

两次默认运行从同一 E2 step-5 checkpoint 顺序启动在 GPU2，并在一次 optimizer step 内按因果顺序记录 tensor hash。

- 382 个初始 LoRA、1146 个 optimizer-before tensor、RNG、输入、15 个 SDE/rollout tensor、reward、advantage 和 7 个 forward/loss tensor 全部精确一致。
- 首次差异为 backward gradients：368/382 个 tensor 不同；之后 736/1146 个 optimizer-after tensor 和 368 个更新后参数不同。
- 因此数据、checkpoint RNG、SDE 与 forward 不是 E2b 漂移根因；默认 CUDA/BF16 backward kernel 是首个分叉点。
- 冻结确定性 bundle 后，两次独立单步进程的全部因果类别精确一致，包括 382/382 个 gradients。

完整结果：`experiments/sd3_single_step_determinism_20260713/RESULTS.md`。

## D1：确定性五步同 checkpoint 复现

用外部 wrapper 对冻结 E2b 源码施加已验证确定性 bundle，从同一 step-5 checkpoint 分别在 GPU2/GPU3 独立恢复到 step 10。

- 两条分支均 `valid=true`、各 5 行 metrics。
- 最终 adapter SHA256 同为 `b814f813...b4c452`。
- metrics、prompt sequence、held-out、optimizer/plugin/RNG、implementation/config identity 和 72 张 PNG 内容全部精确一致。
- 原始 manifest 仅绝对 checkpoint 输出路径不同；位置归一化后语义一致。
- `all_exact_gates_passed=true`，`eligible_for_deterministic_repeat_claim=true`。

完整结果：`experiments/sd3_deterministic_resume_repeat_20260713/RESULTS.md`。

## E2c：内置确定性 checkpoint/resume

在当前源码中加入显式 `runner.deterministic_runtime` 和 CLI flag，并将实际 runtime identity 纳入 checkpoint implementation identity。真实 SD3 比较单进程 steps 0-1 与独立 step 0 + 新进程 resume step 1。

- 三条有效 run 均 `valid=true`；resume 正确加载 base step 1 到 target step 2。
- step 0/1 metrics 分别精确一致。
- 最终 adapter SHA256 同为 `699ec49f...cad6bf8`。
- optimizer、plugin、RNG、step、config/implementation identity 精确一致。
- 共同 baseline 到最终模型的 held-out 统计和 36 张最终 PNG 全部精确一致。
- 合并 manifest 排除纯位置 `checkpoint_path` 后语义一致。
- `all_exact_gates_passed=true`，`eligible_for_exact_checkpoint_resume_claim=true`。

一次误用同名 `flow_grpo` cu128 环境的恢复被 runtime identity 在训练前安全拒绝；正确环境为 `visual-rl-sd35`、PyTorch 2.11.0+cu130 / CUDA 13.0。失败 run 无 metrics/summary，证据保留。

完整结果：`experiments/sd3_deterministic_integrated_resume_20260713/RESULTS.md`。该两步实验验证 correctness，不解锁 E1 效果扩容；更长的内置 5+5 resume 可作为增强证据。

## I0：checkpoint 配置指纹 v2

### 已确认根因

旧版 v1 `config_fingerprint()` 对完整 `dataset` 和 `evaluation` section 做哈希，仅排除 output/resume 位置以及最大训练步数/保存间隔。因此，即使 `content_sha256` 完全一致，`dataset.path` 或 `evaluation.path` 改变也会拒绝 resume。

这个行为在 v1 下是正确的安全 fail-fast，但把“数据语义身份”和“数据存储位置”混在了一起，阻碍同内容数据在不同服务器/目录间迁移。

另一个安全缺口是：当 config 没有显式填写 `content_sha256` 时，实际文件内容虽然会被 `PromptDataset` 读取，但计算结果不会写回参与指纹的 config。同一路径内容被替换时，v1 指纹可能保持不变。因此 v2 不能只做“删除 path”这一项机械修改，必须先读取实际数据并形成经过验证的内容身份。

当前 aggregate fingerprint 还混入 implementation identity，其中包含 git commit/diff、runtime tree 和参数签名。I0 本身会改变代码身份，所以必须先用旧代码完成 E2b；修复后也不能承诺旧 E2 checkpoint 在新代码上无条件恢复。

### v2 目标语义

训练语义指纹继续包含模型、reward、seed、算法、LoRA、采样和全部非位置训练参数。数据相关部分改为：

```json
{
  "data_identity": {
    "train": {
      "content_sha256": "4ccbe865...",
      "split_name": "train",
      "sampling_seed": 201,
      "sampling_strategy": "deterministic_shuffle"
    },
    "evaluation": {
      "content_sha256": "aeefd1c4...",
      "split_name": "heldout"
    }
  },
  "data_source": {
    "train_path": "/server/path/train.txt",
    "evaluation_path": "/server/path/heldout_pilot.txt"
  }
}
```

`data_identity` 参与 resume 兼容判断；`data_source` 写入 checkpoint/artifacts 供追溯，但不参与 v2 指纹。`repeat_per_prompt`、evaluation seeds/max prompts 等非路径字段仍属于训练语义，不能随路径一起排除。

安全规则：

- 内容 hash 不同：拒绝；
- split、sampling seed/strategy、repeat 或其他数据语义不同：拒绝；
- 只有路径不同且实际文件内容与声明 hash 相同：允许；
- 文件缺失、无法读取或实际 hash 不符：在恢复 optimizer/RNG 前拒绝；
- 不增加可绕过检查的通用 `--force-resume`。

### 带版本的 checkpoint metadata

新 checkpoint 至少记录：

```json
{
  "config_fingerprint_version": 2,
  "config_fingerprint": "...",
  "training_semantics_fingerprint": "...",
  "data_identity_fingerprint": "...",
  "implementation_identity_fingerprint": "...",
  "data_identity": {"train": {}, "evaluation": {}},
  "data_source": {"train_path": "...", "evaluation_path": "..."}
}
```

aggregate `config_fingerprint` 可以保留用于完整一致性，但诊断和兼容判断必须能够区分训练语义、数据身份与实现身份。

没有 `config_fingerprint_version` 的旧 checkpoint 一律视为 v1，继续用现有包含绝对路径和 implementation identity 的算法校验；不得原地静默升级。在相同代码身份下，v1 checkpoint 继续按原规则加载或拒绝；代码身份因 I0 改变时仍应拒绝。若未来确需跨这次纯 checkpoint-infra 改动迁移旧 checkpoint，必须设计单独、可审计的迁移工具或 allowlist，不能偷偷用 checkpoint 中的旧 implementation identity 替代当前实现。

### 错误解释与回归门槛

恢复失败时输出结构化字段 diff，而不是只给两个 hash。至少覆盖：

- v2 相同内容、不同训练/评估路径：允许恢复；
- v2 不同内容、相同路径：拒绝；
- split、sampling seed/strategy、reward、seed、算法或 LoRA 配置改变：拒绝；
- output、resume 位置、终止步数、保存间隔改变：允许；
- 数据文件缺失或实际内容与声明 hash 不符：拒绝；
- v1 checkpoint 在相同 implementation identity 下仍按原含路径规则加载或拒绝；实现身份变化继续拒绝；
- 未知 fingerprint version：拒绝；
- `checkpoint.json` 与 `training_state.pt` 的 fingerprint version/hash 不一致：拒绝。

错误信息示例：

```text
Resume rejected: checkpoint uses config fingerprint v1.
Fingerprint mismatch: v1 binds absolute data paths and implementation identity.
Exact field diff is unavailable because this checkpoint did not store a canonical identity payload.
Reuse the original path and original implementation, or run an audited migration workflow.
```

v2 checkpoint 保存 canonical identity 后才能稳定给出字段级 diff；v1 只有 hash 时，除非旁边有经过验证的旧 resolved config，否则不能伪造精确字段差异。

I0 只修改 fingerprint/diagnostic/compatibility 层，没有顺带重构配置 API 或实验算法。

### 实现与验证结果

- 当前新 checkpoint 写入 `config_fingerprint_version=2`、aggregate/component fingerprints、canonical identity、`data_identity` 和只用于追溯的 `data_source`。
- 保存与恢复都会读取真实 train/evaluation 文件并校验内容身份；缺失文件、内容替换和声明 hash 不一致都会在恢复 optimizer/RNG 前拒绝。
- v2 mismatch 报告字段路径；无版本的旧 checkpoint 继续视为 v1，并保留含绝对路径的原规则，不自动迁移。
- 公共 acceptance 12/12、Git 历史细粒度回归 190/190、Ruff 全部通过。
- 真实 SD3 moved-path smoke 比较 continuous steps 0-1 与原路径 step 0 + 新路径 resume step 1：24 个严格门槛全部通过。
- 最终 adapter SHA256 同为 `699ec49f...cad6bf8`；optimizer/plugin/RNG、step metrics、manifest、共同 held-out 和 36 张最终 PNG 均精确一致。
- split checkpoint 记录原来源路径，resume checkpoint 记录新来源路径；两者 `data_identity` 和 aggregate config fingerprint 精确一致。
- 完整结果见 `experiments/sd3_v2_path_resume_20260714/RESULTS.md`；comparison SHA256 为 `2a9840af...906e787`，本地证据不含 checkpoint/cache/图片本体。

I0 基础设施门槛现已解除，E3 可以开始冻结和执行；E1 效果门槛仍失败，因此 E5 的 20/50/100-step 效果扩容继续锁定。

## E3：原生 TempFlow 与 VisualRL 单步数值对齐

冻结 recipe：`experiments/sd3_tempflow_numerical_alignment_20260714/recipe.json`。

### 分阶段设计

E3 不直接把两个完整训练进程混在一起比较，而是分离两个问题：

- **E3a：共享 rollout 的训练数学对齐。** 同一真实 SD3、prompt、初始 LoRA、branch latent、SDE target、reward 和 timestep；分别执行原生 `train_sd3_pr.py` 的 selected-transition 公式与 VisualRL 公式，比较 new log-prob、advantage、时间权重、loss、gradient、AdamW state 和单步 LoRA delta。
- **E3b：独立采样器对齐。** 只有 E3a 全门槛通过后，才给原生/infra 两侧注入同一 initial latent 和显式 SDE noise，独立生成并比较 main ODE prefix、六个 branch targets、old log-prob 和最终 media hash。

E3a 使用 GPU2，空闲时可退到 GPU3；单进程、deterministic runtime、SD3.5 Medium、256 px、20 diffusion steps、branch step 0、原生固定 branch count 6、BF16、LoRA rank 8。reference 源码文件 hash、当前 infra 源码 hash、环境和门槛在启动前冻结。

原生 reference 的字面语义必须保留并显式比较，不能先改到和 infra 一样：

- `PerPromptStatTracker` advantage denominator epsilon 为 `1e-4`；
- selected-transition temporal weight 为 `2.25 * std_dev_t`；
- transformer 使用 train mode；
- AdamW 后执行 `max_grad_norm=1.0`，本次要求裁剪实际不触发，否则 infra/reference 不是同一更新条件。

如果 advantage、temporal weight 或后续梯度门槛失败，结论是算法语义尚未对齐，停止 E3b 并先形成修复方案；不能用“BF16 误差”概括确定性的公式差异。

### 共享条件

固定同一 prompt、noise、seed、timestep、scheduler 和 BF16/FP32 策略，对齐：latent、next latent、old/new log-prob、advantage、loss、gradient norm 和单步 LoRA delta。

### 硬门槛

- shape、timestep 和 conditioning 完全一致。
- 全部数值 finite；更新前 ratio 接近 1、clipfrac 为 0。
- BF16 new log-prob、advantage、temporal weight、loss 和 gradient `max_abs_delta <= 1e-5`；gradient relative L2 `<=1e-3`、cosine `>=0.99999`。
- 单步更新后参数 `max_abs_delta <=1e-6`、relative L2 `<=1e-3`。
- native gradient norm 必须不超过 `1.0`，保证 reference clipping 不改变本次比较。
- probe 模式不得改变参数 hash。

### E3a 执行结果

- v6 全部门槛通过，`failed_gates=[]`，`eligible_for_e3b_sampler_alignment=true`。
- new log-prob、advantage、temporal weight、loss、382 个 gradient tensor 和更新后 382 个 LoRA tensor在 infra/native 两侧精确一致。
- sampling eval 到 training recompute 的 log-prob 最大差 `2.956390380859375e-05`，低于 source-compatible PPO clip `1e-4`。
- v3 retry2、v4、v5 的失败证据保留；分别定位 transition std、不同 batch kernel 和旧 `1e-5` sampling/recompute 门槛问题。
- 完整结果见 `experiments/sd3_tempflow_numerical_alignment_20260714/RESULTS.md`。该结果已作为 E3b 的执行前置门槛。

### E3b 执行结果

- v3 全部门槛通过，`failed_gates=[]`，`eligible_for_tempflow_sampler_parity_claim=true`。
- native/infra 精确消费相同的 9 份 SDE noise；10 个 ODE state、9×6 个 SDE target、old log-prob、KL、解码 media 和最终 RolloutBatch 字段全部逐位一致。
- sampler 前后 LoRA 参数 hash 不变。
- v1 暴露探针未解开 `torch.no_grad` wrapper；v2 暴露 expected parent 未按 adapter contract 转 dtype。两次失败均保留，v3 只修正探针。
- 完整结果见 `experiments/sd3_tempflow_sampler_alignment_20260714/RESULTS.md`。E3a/E3b correctness 通过不改变 E3e 效果失败和长步数锁定状态。

## E3e：TempFlow reference-compatible 效果验证

- 三个 active 和三个 zero-LR control 均执行有效、各 10 行 metrics；control 参数和 held-out delta 精确为零。
- active 聚合均值 `+0.0149758`，CI95 `[-0.0363200, +0.0576872]`；seed 201/307/419 分别为 `+0.0130840/+0.0487904/-0.0169470`。
- blue/green/red 聚合均值为 `+0.0262004/+0.0376971/-0.0189702`。
- seed 307、419 的像素护栏失败，active 训练中 `clipfrac_max=1.0`。
- `eligible_for_effectiveness_claim=false`；禁止 20/50/100-step 扩容。完整结果见 `experiments/sd3_tempflow_reference_effect_20260714/RESULTS.md`。

## T2：TempFlow 效果失败诊断

- T2a 已从冻结 E3e 证据完成观测定位，不重新训练：zero-LR control 参数变化精确为 0，但 30 个 step 中 63.3% 有 `clipfrac > 0`。
- 匹配 active/control 的逐 step `logprob_delta_abs_max` 相关系数为 `0.994030`；control branch 8 偏移约为 branch 0 的 `257.45` 倍，下一轮 branch 0 重置。
- seed419 的训练 target 由 reward 规则映射为 80% red；它同时有 red held-out `-0.074386`、dynamic-range ratio `0.792467` 和 spatial-std ratio `0.787614`。
- 结论边界：这些证据排除了“只有参数更新才导致 clipping”，但不能证明 clipping 或 red-heavy target 单独造成 held-out 退化。
- T2b v2 已完成固定轨迹 mode 因果消融：eval/train recompute 最大差为 0，但二者与 sampled old log-prob 最大差均为 `0.007318`、最大 `clipfrac=1.0`，branch 8/0 比 `150.837`。mode 假设被否定。
- T2d v1 全部门槛通过：single-parent forward 完全恢复 sampling old log-prob（最大差 0、`clipfrac=0`），six-branch batch 复现 `0.007318`/`clipfrac=1.0`；首次非零差在 BF16 noise prediction `0.515625`，SDE mean 差 `0.167527`，std 差 0。
- T2e v4 已通过实际 shared-prefix adapter policy-identity 复验：正式路径覆盖九个 transition，old/new 最大差 `0.0`、`clipfrac=0.0`、参数指纹不变。v1-v3 均为加载模型前的 staging 失败并完整保留。
- 冻结两种显式执行模式：`tempflow_reference_mode=true` 只支撑原生 parity 结论；`false` 支撑 policy-identity/effectiveness。T3 三 seed 10-step active/control 已完成且效果门槛失败；后续 T2c 也已完成，结论见当前总览，20/50/100-step 仍锁定。
- T3 已完成：两项 preflight 与六个正式 run 的机械门槛全部通过，60/60 步 old/new log-prob 和 clipfrac 为 0；但 active mean `+0.00851066`、CI95 `[-0.0436011,+0.0602622]`、正向 seed `2/3`、red `-0.0393180`，seed307/419 像素护栏失败。
- `eligible_for_effectiveness_claim=false`、`eligible_for_20_step_expansion=false`。这证明 policy drift 是机械 bug，但不是效果失败的充分原因；后续 T2c 已完成，长程仍锁定。
- 完整可重建结果见 `experiments/sd3_tempflow_effect_diagnosis_20260714/README.md`、`experiments/sd3_tempflow_mode_ablation_20260714/RESULTS.md`、`analysis.json` 和 `report.html`。HTML 已通过结构校验；自动视觉 QA 受本地 `file://` 浏览器策略限制，待人工检查。

## 后续实验队列

### E4：独立质量评估

训练 reward 之外至少接入一个固定版本的语义/偏好评分器，并保留固定 before/after 人工盲评面板。主要检查 reward 提升是否对应 prompt adherence 和可见质量，而不是新的评分器投机。

### V1/V2：真实视频路径

- Flash/Wan：真实 checkpoint 推理 -> single-step/log-prob parity -> LoRA 单步更新 -> resume -> bounded run。
- World-R1：Wan LoRA 补齐 -> 真实 reward server 协议/方向/延迟 -> full trajectory parity -> active/control -> bounded multi-seed。
- SD3 post-merge exact resume 与 O4 HPS-backed Wan 单步/中断门槛均已完成。下一步不再重复机械 preflight，而是先做 reward warm/cold、逐样本/批量、checkpoint 常驻和 10–100 step 稳态 profiling，再决定异步 overlap 或多 reward worker；这些结果仍不能替代独立质量实验。

### W10：World-R1 / Flash-GRPO 大型多 reward 验证

W10 是 P7、doctor/高层 API 和 10–100 step 稳态门槛之后的下一项大型质量主线，并优先于 WM0-minWM；当前立即执行的工程任务仍是 P7。W10 验证的是 VisualRL 能否在真实 Wan、复杂视频 prompt、多元 reward、长时间更新和中途恢复下保持正确，并让多个主要 reward 在独立 held-out 数据上共同改善。W10 不是把 W8 的旧 recipe 直接增加步数：W8 的 `64px/5帧/2-step`、12 条 PartiPrompts、单一 HPS、10-step 效果失败结论永久保留；W10 使用新的实验 ID、数据身份、reward 身份和预注册门槛。

#### 证据分层

W10 必须分开报告三类结论：

1. **Infra correctness**：数据顺序、rollout、reward、梯度、optimizer、checkpoint、resume、artifact 和故障恢复正确。
2. **Optimization effectiveness**：active 相对匹配 control 的训练与 held-out reward 有统计提升。
3. **Independent quality**：未参与训练的 evaluator、像素/视频护栏和人工面板不退化。

训练 reward 上升不能单独证明 Infra 正确，也不能替代独立质量。单步 reward 允许因 prompt、noise 和 timestep 波动；“reward 增长”按冻结 checkpoint 窗口和 held-out 前后差定义，不要求每一步单调上升。

#### 数据计划

- 训练集目标为 240 条固定视频 prompt，优先从 [`VidProM`](https://vidprom.github.io/) 的冻结 revision 中筛选并保存原始行号、筛选脚本、许可证、内容 SHA256 和近重复审计。
- 训练集按人物/动物动作、多对象交互、空间关系、运动方向与速度、数量/属性绑定、复杂场景/镜头运动六类分层，每类目标 40 条；不得由最终结果反向删除失败 prompt。
- held-out 使用与训练集无内容或语义近重复的 [`T2V-CompBench`](https://t2v-compbench-2025.github.io/) 固定分层子集，目标为七类各 20 条、共 140 条；该集合不得进入训练或 reward normalization。
- [`VBench`](https://github.com/Vchitect/VBench) 只用于 baseline/final 独立评估；不作为每 step reward，也不据其结果调整已运行 recipe。
- World-R1 额外冻结带显式 camera trajectory 的 prompt/trajectory 子集；共享算法比较与 World-R1 原生 3D 验证分开报告。

#### Reward 计划

两条算法共享的训练 reward 至少覆盖：

```text
text-to-video alignment
visual quality
motion quality
temporal consistency / dynamic guardrail
```

优先评估 [`VideoReward`](https://github.com/KlingAIResearch/VideoAlign) 的 text alignment、visual quality 和 motion quality 三个分量；在接入前必须完成固定视频的 direct/reference/infra parity、batch-preservation、方向、无效输出、timeout、cache identity 和服务中断门槛。World-R1 原生分支再增加：

```text
score_meta_view
score_reconstruction
score_trajectory_alignment
```

- 每个 reward 分量保存 raw score、version、checkpoint identity、输入 frame policy 和 latency；不得只保存加权总分。
- 在独立 calibration snapshot 上冻结每个分量的 mean/std、方向、clip 和权重，再写入 config/checkpoint identity；训练中不得自适应改变 normalization 来制造上升趋势。
- HPSv2、PickScore、T2V-CompBench/VBench 指标和人工盲评至少保留一组完全不参与训练的独立验收信号。
- 共享算法比较只使用两边都能计算的 reward core；World-R1 的 3D reward 作为单独 native arm，不与缺少相同 camera-trajectory 条件的 Flash 数值直接排名。

#### 算法与公平性

- World-R1 与 Flash-GRPO 使用同一 Wan2.1 base revision、初始 LoRA、prompt 顺序、seed 集合、共享 reward 版本、训练/评估 split 和 artifact schema。
- World-R1 保留 full-trajectory 与 camera-aware/3D 语义；Flash-GRPO 保留 single-step、selected timestep 和 rectification 语义，不为表面统一而改写参考算法。
- 两条路径 diffusion steps 和保留轨迹不同，因此“相同步数”不等于相同计算量。效果按各自 active/control 判断；跨算法效率另按 reward gain/GPU-hour、samples/s、reward calls、峰值显存和最终独立质量报告。
- 正式配置优先使用 P3 候选 `480x832/5帧/4-step`；必须先证明 World-R1、Flash 和多 reward 服务都能在冻结资源布局下安全运行。若共同配置不成立，选择两条路径共同通过的最高规格并保留失败证据，不允许运行时静默降级。

#### 分阶段执行

1. **W10-A，5-step 联调**：24 条分层 prompt，验证所有 scorer、raw/normalized reward、方向、batch、cache、artifact 与 GPU/进程清理。
2. **W10-B，20-step active/control**：每条算法一个 seed，确认 active 非零更新、control 精确零更新、主要 reward 梯度方向合理且无明显独立护栏退化。
3. **W10-C，60-step 单 seed**：检查 reward 趋势、clip/ratio/gradient、显存、句柄、cache、checkpoint 和 reward server 的稳态行为；未通过不得启动正式长跑。
4. **W10-D，240-step 正式实验**：完整覆盖 240 条训练 prompt；World-R1/Flash 各 3 seeds × active/control，共 12 个正式 run。固定 checkpoint/eval 点为 baseline、20、60、120、180、240。
5. **W10-R，恢复哨兵**：每条算法至少完成一次 deterministic `40 continuous` 对 `20 + process exit + resume-to-40`；大型 effectiveness run 使用 performance runtime 时只按预注册容差与统计门槛判断，不宣称 bitwise exact。

#### Infra 硬门槛

- 所有计划 step、sample、prompt 和 reward 记录无缺失、重复或错序，数值全部 finite。
- active LoRA 非零更新、base model 不变；matched zero-LR control 参数精确不变。
- commit marker、checkpoint tree digest、manifest、metrics、cache 和 status/audit 全部一致；恢复后 dataset cursor、RNG、optimizer、plugin 和 sample identity 连续。
- 任何 reward 服务失败、OOM、进程中断或写入失败不得产生部分 optimizer 更新或伪 completed artifact；已提交旧 step 保持权威。
- GPU、端口、子进程、临时目录和文件句柄在每个 run 后释放；只管理本实验拥有且身份完全匹配的进程。

#### Reward 增长门槛

对每个主要训练 reward 分量分别计算 held-out `final - baseline`；目标是每个分量均大于 0，而不是只要求加权总分上升。正式晋级还要求：

- World-R1 与 Flash 各自 3/3 active seed 的 aggregate active-control 均值为正；
- hierarchical bootstrap CI95 下界大于 0；
- train 与 held-out 的 checkpoint-window 趋势方向一致；
- 任何主要分量不得出现统计显著退化，不允许一个 reward 暴涨掩盖另一个 reward 下降；
- control 的参数变化为零，reward 波动必须落在预注册 sampling-noise 边界内。

W10-B 或 W10-C 若未表现出可信的多 reward 改善，停止并检查数据、reward 尺度、分量冲突和 credit assignment；不得用 W10-D 的更长训练掩盖失败。

#### 独立质量与停止条件

- T2V-CompBench 分层结果、VBench temporal consistency/motion smoothness/dynamic degree/overall consistency、固定 HPS/PickScore 和像素视频护栏不得显著退化。
- 保留固定 before/after 人工双盲面板；不得由训练 reward 模型或 AI 自动替代真实评审结论。
- 出现饱和、静态化、闪烁、运动消失、prompt/reward hacking，或只有训练 reward 增长而独立指标不增长时，效果验收失败。
- W10 通过只证明当前模型、数据、reward 和 240-step 范围内的 correctness/effectiveness；不自动外推为长程收敛、SOTA、吞吐提升或 minWM 可行。

#### 资源布局

P3 的 `480x832/5帧/4-step` 单步训练物理峰值为 `28,336 MiB`，因此多 reward 不与训练模型共驻同一张 32GB GPU。正式 recipe 冻结独立资源角色，例如：

```text
GPU2：Wan / LoRA training
独立 reward GPU：VideoReward、World-R1 general/3D worker
CPU/本地盘：artifact、cache、validator 与报告重建
```

启动前必须检查物理 GPU 基线、模型/reward checkpoint 身份、端口所有权和预计峰值；GPU 不满足时等待或安全停止，不动态降低分辨率、帧数、reward 或 batch 后继续记为同一实验。

### WM0-WM3：从 Wan 视频生成扩展到 minWM 交互式世界模型

这一方向不是用 minWM 替换 Wan，也不把 minWM 的完整 Trainer 合并进 VisualRL。minWM 是将双向 T2V/TI2V backbone 转换为相机或动作可控、因果、少步自回归世界模型的完整流水线；其官方实现本身包含 Wan2.1-T2V-1.3B 路径。因此本项目采用以下关系：

```text
Wan2.1 / minWM checkpoint（外部 backbone 与 reference runtime）
-> action-conditioned causal rollout adapter
-> VisualRL reward / evaluation / checkpoint / audit
-> 有界 RL 或 reward-guided post-training
```

参考资料：[`minWM` technical report](https://arxiv.org/abs/2605.30263) 与 [`shengshu-ai/minWM`](https://github.com/shengshu-ai/minWM)。

#### 架构边界

- `ExperimentRunner` 继续是 VisualRL 唯一训练协调器；不得复制 minWM 的训练循环到核心主线。
- minWM 作为外部 reference/runtime，通过新的 `WorldModelAdapter` 或等价插件边界接入。
- 复用现有 Wan checkpoint、artifact、安全恢复和评估能力；minWM 自己的多阶段 checkpoint 只通过显式 adapter 映射。
- 第一阶段只读推理，不下载或启动完整四阶段训练；未经有界门槛验证，不进入大规模 SFT、AR diffusion、Causal ODE/CD 或 DMD 复现。

#### WM0：只读可行性与身份冻结

- 冻结 minWM commit、Wan2.1 base/DMD checkpoint、环境、推理脚本、action/camera trajectory 和输出校验和。
- 使用官方 Wan Action2V checkpoint 完成至少两条不同相机动作序列的分段因果推理。
- 保存首帧、分段视频、动作序列、history/state identity、每段 seed 与 wall time；不得只保留最终拼接视频。
- 建立 external-reference 与 adapter 输出的 media、动作顺序、chunk 数量和状态传递一致性门槛。

WM0 通过条件：同一 checkpoint、初始状态、动作和 seed 的 reference/adapter 轨迹语义一致；重复运行在声明的 deterministic/tolerance 模式下满足预注册门槛；失败时停止在 adapter 层，不复制 minWM Trainer 绕过合同。

#### WM1：正式世界模型数据合同

新增或等价表达统一的 `WorldRolloutBatch`：

```text
initial_state
action_sequence
generated_chunks
history_state / history_identity
frame_timestamps
rollout_latency
termination_mask
sample_id / trajectory_id / chunk_id
```

合同必须明确 action 与生成 chunk 的一一对应、历史窗口、因果 mask、跨 chunk seed、padding/termination 语义和 image/video layout。关键训练语义不得只放在自由 metadata 中。cache、manifest、checkpoint 和 resume 必须能够验证 trajectory/chunk/history identity，拒绝跨 generation 或跨轨迹混用。

#### WM2：世界模型独立评估

至少冻结四类与训练 reward 分离的 evaluator：

- action/camera adherence：生成运动是否遵循输入动作及方向；
- temporal/spatial consistency：主体、背景、遮挡和场景几何是否跨 chunk 连续；
- long-horizon drift：随 rollout 长度增加的身份、结构和画质退化；
- latency-quality trade-off：首帧延迟、单动作响应时间、吞吐、峰值显存与质量。

必须保留固定 active/control、失败动作和失败轨迹；reward 上升不能替代独立 evaluator 与人工盲评。普通视频 PickScore/HPS 只能作为画质辅助指标，不能单独证明世界模型的可控性或因果正确性。

#### WM3：有界训练与晋级门槛

按以下顺序执行，不直接复现完整 minWM 四阶段训练：

1. zero-update/reference parity；
2. LoRA 单步更新，验证梯度、参数变化和 base-model 不变性；
3. continuous 与 split/resume 的 trajectory、optimizer、RNG、history identity 和最终 adapter 比较；
4. 一条短 action trajectory 的 active/control；
5. 多 seed bounded post-training，并同时检查 action reward、独立一致性、安全护栏和延迟。

只有 action/camera reward 与至少一个独立 world-model evaluator 同时改善，control 保持零更新或预注册噪声边界，且长程一致性与延迟护栏不退化，才允许扩大步数或接入 Causal Forcing/DMD 后训练。若必须引入第二个 Runner、无法稳定表达 history identity，或改善只存在于训练 reward，则停止主线接入并将其保留为独立研究分支。

#### 执行顺序

WM0 延后到 W10 得出明确的通过或失败结论并完成证据收口之后。执行顺序是先完成 P7、doctor/高层 API 和 10–100 step 稳态门槛，再进入 W10；W10 完成后才扩展 action-conditioned world-model 合同。WM0-WM2 届时可作为下一代研究 POC；WM3 仍不得跳过当前 artifact/checkpoint 安全边界、资源预算和 W10 暴露出的 reward/长稳问题。

### P1：效率与多卡

记录 rollout/reward/backward/checkpoint 的 wall time、峰值显存、GPU 利用率和 samples/s；再比较相同 effective batch 的单卡与真正双卡训练。GPU2/GPU3 分别跑 active/control 不算双卡训练。

## 更新日志

- 2026-07-15：P6 schema-v2 attempt 2 通过。RTX 5090 GPU7 上 2 次 warm-up 与 6 次交错 measured 均为独立进程、同一冻结身份且资源清理有效；三对 on/off 的 14 项训练摘要全部 exact。gradient checkpointing 将固定 Wan 单步 update peak median 从 `18,157,285,888` 降至 `14,966,723,072` bytes，减少 `3,190,562,816` bytes（`17.5718%`），超过 512 MiB/5% 双门槛。独立重跑 validator 与远端结果逐字节一致；只据此发布 `ee4a44f` 合同和窄边界显存结论，下一项当前任务为 P7。
- 2026-07-15：登记 W10 World-R1/Flash-GRPO 大型多 reward 验证，作为 P7、doctor/API 与稳态门槛之后、WM0 之前的后续大型质量阶段。W8 低语义 10-step 失败保持不变；W10 新建 VidProM 240 条分层训练 snapshot、T2V-CompBench held-out、VideoReward 共享分量、World-R1 3D 分量、5/20/60/240-step 分级门槛、3-seed active/control、deterministic resume 哨兵与独立 VBench/人工质量验收。多 reward 原始分量必须分别增长，不能只看加权总分或用长跑掩盖短阶段失败；全局执行顺序以当前总览为准。
- 2026-07-15：O4 attempt 3 通过。GPU2 的真实 Wan/HPS healthy step1 完成一次 optimizer 更新并发布 checkpoint；从 step1 恢复后在 step2 请求进入、scorer 前终止 GPU4 的自有 reward server，1 次请求、0 重试、0 optimizer、无 step2 artifact，step1 关键状态和 semantic tree 不变，所有进程/GPU/端口清理通过。健康 step 的 reward 时间占 87.91%，只作为下一轮稳态 profiling 假设，不作加速结论。
- 2026-07-15：高层 Wan/Flash-GRPO API commit `806c21a` 经 899 个 non-distributed、5 个真实 CPU/Gloo DDP、Ruff、compileall、diff check 和独立 review 后 fast-forward 推送到 GitHub `main`；发布树不含 `exercises`、`.codex` 或 O4 草稿。
- 2026-07-15：登记 WM0-WM3 minWM 交互式世界模型方向。明确 minWM 不替换 Wan，也不引入第二套 Runner；先以 Wan2.1/minWM checkpoint 完成只读因果推理、`WorldRolloutBatch` 合同和独立 world-model evaluator，再用 zero-update、LoRA 单步、resume 与 active/control 决定是否进入有界 RL post-training。
- 2026-07-15：`2adfbfd` 完成本地统一回归（883 passed、2 skipped、5 deselected；Gloo 5 passed、885 deselected）和真实 Wan post-merge gate。W7b 480 个梯度逐位一致；W5 attempt 1 因 harness 在 deterministic runtime 前初始化 CUDA 而于训练前失败，修正顺序后的 attempt 2 六段 run 全部 valid、World/Flash 两份 11-gate 比较均 exact；`reward_general` 三方分数逐位一致、坏请求 500、无 silent fallback。质量/速度声明继续锁定，SD3 与 HPS-backed Wan 训练 post-merge 仍待执行。
- 2026-07-13：建立账本；记录 B0/B1 通过；冻结并完成 E1。六个 run 全部执行有效，但效果门槛失败；下载非 checkpoint 证据并转入 E2。
- 2026-07-13：E2 第一进程完成 step 5；验证 checkpoint 完整后启动独立 resume-to-10 进程。
- 2026-07-13：E2 resume 机械闭环通过，但严格等价门槛失败；证据表明独立 BF16/CUDA 运行在 resume 前已存在漂移。暂停 E3，新增 E2b 确定性审计。
- 2026-07-13：E2b 首次 GPU2/GPU3 分支均被 checkpoint 配置指纹安全拒绝，零训练 step。精确定位为复制数据后的绝对路径变化；保存非 checkpoint 失败证据，未重启，E3 保持暂停。
- 2026-07-13：冻结 E2b `attempt_2_recipe.json`，要求复用 E2 原数据绝对路径且不改 infra；同时登记 I0 指纹 v2，执行顺序固定为 attempt 2 -> CUDA/BF16 判断 -> I0 实现/回归 -> E3。
- 2026-07-13：E2b attempt 2 在 GPU2/GPU3 完成。三分支执行有效、顺序一致，但最终 LoRA 和数值结果严格不等；确认当前 BF16/CUDA 执行路径不可逐位复现，E3 不解锁。
- 2026-07-13：D0 单步因果探针确认默认模式首次分叉在 backward gradients；SDE、RNG 和 forward 精确一致。确定性 bundle 使全部因果类别精确一致。
- 2026-07-13：D1 在 GPU2/GPU3 完成两个独立五步 continuation；adapter、训练状态、metrics、held-out 和 PNG 内容严格一致，确定性 bundle 通过扩展验证。
- 2026-07-13：实现内置 deterministic runtime 和 runtime-bound checkpoint identity；公共 acceptance 9/9、历史回归 190/190、Ruff 通过。
- 2026-07-13：E2c 真实 SD3 连续 2-step 与 1+resume-to-2 严格等价全部通过。一次 cu128 环境误恢复被安全拒绝；E3 现在只剩 I0 实现/回归门槛，E1 效果扩容仍锁定。
- 2026-07-14：实现 checkpoint 指纹 v2；公共 acceptance 增至 12/12，历史回归 190/190 和 Ruff 通过。真实 SD3 相同内容换路径 resume 的 24 个严格门槛全部通过，I0 完成并解锁 E3；E1/E5 效果扩容状态不变。
- 2026-07-14：冻结 E3a 共享-rollout 数值对齐 recipe，明确 deterministic/audit 与 performance/training 两条验收通道；E3a 先比较原生 `1e-4` advantage epsilon、`2.25 * std_dev_t` 时间权重与 infra 当前公式，失败即停止 E3b。
- 2026-07-14：E3a v6 通过全部共享-rollout 数值门槛；infra/native 的训练数学、382 个 gradient 和单步 LoRA 更新精确一致，解锁 E3b sampler 对齐。
- 2026-07-14：完成 TempFlow reference-compatible 三 seed active/control。六个 run 执行有效且 controls 精确为零，但效果 CI、正向 seed 比例、red 类和像素护栏门槛失败；保留证据并禁止长步数扩容。
- 2026-07-14：建立 `experiments/INFRA_VALIDATION_WORKLOG.md`，持续记录改动文件、证据边界和剩余可靠性实验。
- 2026-07-14：将全部待补实验扩展为 A/T/W/D/Q/C/O/P/S 九组验收矩阵，并启动持续 Goal；锁定项必须等待上游硬门槛通过。
- 2026-07-14：E3b v3 通过全部 sampler parity 门槛；显式相同 initial latent/SDE noise 下 native 与 infra 的完整 ODE/SDE 轨迹、log-prob、media 和 RolloutBatch 逐位一致。v1/v2 失败证据保留。
- 2026-07-14：完成 T2a 冻结证据诊断。zero-LR control 仍复现高 clipping，active/control drift 相关 `0.9940` 且随 branch 位置放大；seed419 的 80% red target 与 red/像素护栏退化同时出现。T2b/T2c 因果消融待执行，T3/长程继续锁定。
- 2026-07-14：T2b v2 固定轨迹消融有效完成并否定 eval/train mode 假设；两种重算逐位相同，但都与 sampling old log-prob 偏离并随 branch 放大。停止 T2c，新增 T2d batch/kernel/dtype 首次分叉定位。
- 2026-07-14：T2d v1 全部门槛通过。single-parent forward 精确恢复 old log-prob，six-branch BF16 forward 复现 high clipping；first divergence 定位到 noise prediction。登记 T2e 现有 shared-prefix adapter 路径复验，仍不解锁 T3。
- 2026-07-14：T2e v4 全部门槛通过。正式 adapter shared-prefix 路径覆盖九个 transition，old/new log-prob 最大差和 clipfrac 均为 0，参数指纹不变；解锁 T3 三 seed 10-step active/control，但不解锁更长训练。
- 2026-07-14：为 T3 增加显式 `tempflow_execution_mode`，默认不改变 reference-compatible 行为；policy-identity 模式强制严格初始一致性。公共 acceptance 15/15、Ruff 和 diff check 通过；冻结 T3 source、recipe、validator 与 seed201 单步 active/control preflight。
- 2026-07-14：T3 两项 preflight 与六个 10-step run 全部执行有效；60/60 步 old/new/clipfrac 精确归零，controls 精确零更新。效果 CI、正向 seed、red 和像素护栏仍失败；不扩容，转入 T2c reward/data 消融。
- 2026-07-14：O2 在 GPU2 触发真实 192 GiB CUDA OOM；failed/invalid、step0、无 metrics/latest/checkpoint，显存完全释放；同环境 16 px control 完成。fail-safe 通过，自动降级仍不支持。
- 2026-07-14：S3 单机自有进程 smoke 通过。错 start tick 不发信号，只有 UID、PID start tick 和 command token 全匹配才终止实验自建 child；不声明集群级调度。
- 2026-07-14：W6 general attempt 5 通过。HPS direct、World-R1 HTTP 与 infra client 分数逐位一致；staging reference 的静默 `0.5` 兜底改为 HTTP 500，坏图像/坏 pickle 均 fail closed；随后转入 DA3+Qwen 3D 门槛。
- 2026-07-14：登记 W7b 异构 selected-index batch parity；W7 已证明同 batch 同 index 的 native 数值正确性，但不能据此声称多样本异构时间步或吞吐能力。
- 2026-07-14：W6 3D attempt 1 正确拒绝 World-R1 worker 静默返回的 `0.0`/空 artifact；定位为服务器缺少 nvcc，非32GB显存不足。用户态 CUDA 12.8.61 为 RTX 5090 `sm_120` 编译 gsplat 后，attempt 2 的真实 DA3+Qwen reference/infra parity、component、MP4/PNG、cache、坏请求500和安全停止全通过，W6 完成。
- 2026-07-14：W7b attempt 2 通过。两个真实 Wan prompt 使用反向异构 index `[2,1]`，分组/恢复后 media、embedding、前后 latent、时间步、old/new log-prob、loss 和480梯度均与独立 scalar reference 逐位一致；W8 解锁到1-step真实 reward preflight。
- 2026-07-14：W8 真实 HPS 1-step preflight 通过。World-R1 与 Flash-GRPO 各自的 active/control 在 GPU2 上顺序执行，训练前 rollout、media、prompt/timestep 和 HPS 分数逐位相同；active LoRA delta L2 为 `0.195539/0.239434`，两个 control 参数精确不变。HPS+Wan 物理显存峰值25,813 MiB，服务安全停止后GPU2回到0 MiB；下一步进入固定 HF prompt 快照的3-seed 10-step，不直接进入W9。
- 2026-07-14：W8 3-seed 10-step recipe 与固定 PartiPrompts 快照已冻结。attempt1 因 JSON seed 解析在训练前 fail closed；修复后 attempt2 又在首个训练前遭遇外部任务占用GPU2约22GiB。严格只终止本实验自建子进程并安全停止HPS，保留两次无checkpoint证据；GPU启动基线收紧为16MiB，等待空闲后用新目录重试。
- 2026-07-14：C2/S2 checkpoint identity fault matrix 通过。模型 revision、LoRA targets、dtype、scorer version/endpoint/实现、deterministic runtime、数据内容八类变化全部结构化拒绝，且 optimizer/plugin/RNG 尚未恢复；相同内容换路径与 output/resume/target/save 参数变化仍允许。
- 2026-07-14：P4 CPU cache/checkpoint 成本实验通过。rollout cache 开/关的训练语义精确一致，reward cache hit 后端调用从1次封顶且约545×加速；checkpoint size/save/resume时间已记录。真实Wan阶段时间由W8 profiler补充，不外推Tiny数字。
- 2026-07-14：W8 attempt3 正式完成。World-R1/Flash-GRPO 共12个真实Wan 10-step run与6组matched active/control全部机械有效，active非零更新、control精确零、视频像素护栏通过，GPU2峰值23,616MiB且结束归零；但World配对均值`-0.000418`、Flash `-0.000489`，两者CI95均跨0且未满足3/3 seed正向。W9按硬门槛锁定，120个无checkpoint媒体已下载用于Q3失败诊断。
- 2026-07-14：Q3 attempt2通过评分器与视频安全门槛。240视频/1,200帧固定PickScore两遍、step0 media/score精确一致；World/Flash均无塌缩，闪烁/清晰度/运动护栏通过，但独立paired mean仅`+0.000205/+0.000047`且CI95跨0。contact sheet暴露64px/2-step语义基线过弱；W9继续锁定，先转P3采样可行域。attempt1不同文本batch的step0 score漂移保留。
