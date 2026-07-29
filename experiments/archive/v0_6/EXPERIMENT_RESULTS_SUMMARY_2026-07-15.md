# VisualRL 实验结果总结与后续修复方向（历史快照）

> Historical v0.6 evidence; not a v0.7 usage/config contract.

> 状态：本文件冻结合并阶段当时的结果与判断，其中部分“下一步”已经完成。当前 Goal、状态和执行顺序只以 `docs/PROJECT_OVERVIEW.md` 与 `experiments/EXPERIMENT_PLAN.md` 为准。

更新时间：2026-07-15

本总结基于当前实验账本、42 个实验目录、44 份结果文档、主要机器可读结果与冻结 validator。实验效果结论来自合并前的冻结证据；两份 `framecode` 合并后的 CPU/Gloo 代码结论、Wan correctness gate 与窄范围 reward_general HPS score-path parity 已于 2026-07-15 在 `2adfbfd` 上重新回归。SD3 deterministic resume、质量改善和速度收益仍没有合并后新证据。

## 合并后本地复验

- 完整 non-distributed suite：883 passed、2 skipped、5 deselected。
- 真实双进程 Gloo distributed suite：5 passed、885 deselected；包括跨 rank 更新/回滚、Flash 全局 coefficient oracle 与 microbatch `no_sync()` 通信次数 oracle。
- Ruff、compileall、`git diff --check` 全部通过。
- `2adfbfd` 修复 Wan recompute 的 latent 精度合同：transformer 仍接收 BF16 model input，但送入 SDE 的 current/next latent 保留 rollout 的原始 FP32 值；非 BF16 精确可表示输入的回归测试锁定了这条边界。
- artifact 审计已改为先验证可信根、commit marker 与 checkpoint tree digest，再进入受信训练状态读取；checkpoint v1/v2、manifest v1 只能显式迁移。
- async reward 默认串行，线程模式不再伪装 hard-timeout；DDP snapshot 有 bytes/time 指标与默认 1 GiB fail-fast 预算；formal `RolloutBatch` 的重复 prompt 必须给出无歧义 occurrence `group_id`。
- Runner 会在模型、process group 和 artifact 副作用前重验 direct/mutated config；DDP 非末尾有效 microbatch 使用 `no_sync()`，两 rank oracle 显示通信 hook 从每步 2 次降为 1 次且梯度/参数一致。
- 安装后 CLI 可直接列出并运行 `preset:NAME`，并提供安全的 `status`/`audit`；Python API 可显式 trusted validate 和 detached resume，不再要求研究者退回内部 Runner 或手写恢复顺序。

这些结果把“合并后本地代码没有已知机械回归”的门槛补齐，但不替代 SD3 的远端真实模型复验，也不能证明 HPS reward 能改善训练质量或带来速度收益。

## 合并后 Wan 远端复验（`2adfbfd`）

- W7b 在 RTX 5090 上通过严格异构 Flash parity：sample/tensor、selected index、grouped seed、scheduler metadata、保留 transition、recomputed log-prob、loss、480 个 gradient 和参数不更新门槛全部精确通过；SDE current/next latent 在 transformer 使用 BF16 时仍为原始 FP32。
- W5 attempt_1 的 World-R1/Flash 都在训练前被拒绝，错误为 `deterministic runtime must be configured before CUDA is initialized`。根因是实验 harness 在构造 Runner 前调用 `torch.cuda.*`；失败记录已保留，不能当作模型训练失败或覆盖删除。
- 修正 harness 初始化顺序后的 W5 attempt_2 中，World-R1 与 Flash 均通过 continuous 2-step 对 split 1-step + resume-to-2 的 exact comparison：最终 adapter、训练状态、step 0/1 语义指标、manifest、checkpoint metadata，以及 marker/tree-digest/status 审计门槛全部通过。
- reward_general attempt_1 严格通过：HPS direct reference、loopback HTTP 与 infra 三路分数逐项相同（两项对照的最大绝对误差均为 `0.0`）；无效 payload 与服务内部故障都返回 500，且没有 silent fallback。该兼容协议使用 `legacy_pickle`，只允许 loopback 可信主机，不能作为公网或不可信输入协议使用。

完整机器可读证据索引、失败边界和校验和见 [`WAN_RESULTS_2adfbfd.md`](postmerge_validation_20260715/WAN_RESULTS_2adfbfd.md)。这组结果只关闭 Wan post-merge correctness 与受控 legacy reward score-path parity，不覆盖 SD3、HPS 训练效果、长期质量、吞吐或效率结论。

## 总结结论

| 衡量层级 | 当前结论 | 证据边界 |
|---|---|---|
| Infra 能否跑通 | 基本通过 | CPU/Tiny、真实 SD3、真实 Wan、真实 HPS/3D reward、checkpoint/resume 与多类故障路径已有证据；合并后 Wan 与受控 HPS score path 已重跑，SD3 仍待重跑 |
| Infra 能否正确训练 | 机械正确性通过，训练有效性未通过 | 数学、梯度、更新、zero-LR control 和 exact resume 已验证；独立质量改善没有通过 |
| 是否减少实验代码并易于使用 | 部分通过 | 外部算法可零核心改动接入，模型/算法/scorer/data contract 已统一；通用 CLI 插件发现、外部媒体 provider、DDP 与干净环境一键复建仍缺失 |
| 是否提升速度、效率和质量 | 尚未证明 | Flash 只证明 retained state bytes 降低 10.5 倍；没有 native/infra 性能对照；HPS/PickScore 效果均未通过 |

## 已得到的可靠结果

1. SD3/TempFlow correctness：训练数学、完整 sampler、382 个 gradient 和单步 LoRA 更新与 reference 对齐；deterministic runtime 下跨 GPU repeat 和 checkpoint/resume 可严格一致。
2. Wan correctness：真实 Wan load、PEFT-only save/load、World-R1/Flash 一步闭环、reward server parity，以及 Flash 同构/异构 selected-index parity 已通过；在 `2adfbfd` 上，W7b 严格 parity、World-R1/Flash W5 exact resume 与受控 loopback reward_general 三路 HPS score parity 已重新通过，并验证 SDE FP32 latent 合同。
3. 数据与 artifact：固定 HF revision、内容身份、坏输入拒绝、cache 并发/损坏隔离、checkpoint 原子发布、交叉一致性和多阶段进程中断恢复已有窄而明确的证据。
4. 真实 bounded training：W8 的 12/12 Wan run 机械有效，active 参数更新、control 精确不更新，短程视频安全护栏通过。
5. P3 32GB 可行域：15 个采样单元全部完成；`480x832`、5 帧、4/8/20/40 diffusion steps 均能推理；`480x832`、5 帧、4-step 一步训练峰值为 `28,336 MiB`，相对旧 64px/2-step baseline 的固定 PickScore 最大增益为 `+0.023452`。

## 实验暴露出的主要问题

### 1. 训练 reward 上升不等于质量提升

TempFlow T3b 的 RGB reward 明显上升，但两个 seed 的像素护栏失败；独立 PickScore 均值为负且置信区间跨零。Wan W8 的 World-R1/Flash HPS 配对均值分别为 `-0.000418/-0.000489`，Q3 独立 PickScore 的置信区间也都跨零。

修改方向：旧低语义 recipe 继续锁定；正式训练使用 P3 已验证的 `480x832/5帧/4-step` 候选、更多样的固定 prompt snapshot、语义/偏好 reward 与独立像素安全约束，并继续保留 active/control、多 seed、独立 scorer 和人工盲评。

### 2. 默认 BF16/CUDA 运行不能支持逐位复现声明

默认 performance runtime 的首次分叉发生在 backward gradient；只有显式 deterministic runtime 能支持 exact repeat/resume。

修改方向：配置、CLI、checkpoint identity 和报告必须明确区分 `exact deterministic` 与 `performance best-effort`，禁止静默切换或混用结论。

### 3. 效率结论缺少 native 对照

W8 的每 run 均值为 model load `11.91s`、rollout `15.36s`、reward `18.37s`、recompute/backward/optimizer `8.76s`；reward 是当前最大可优化阶段之一。现有 P4 约 545 倍 cache 加速来自故意延迟的 CPU workload，不能外推到 Wan。

修改方向：先完成 P2，同模型、prompt、seed、batch 和 timestep 下做 native/infra 多次 warm/cold 对照，再决定 scorer batching、进程复用、异步 overlap 或缓存优化。

### 4. 产品化与易用性仍有缺口

外部算法仍需要薄 Python 启动器；真实图像/视频 dataset provider、配置 dry-run、插件发现、DDP 数据 shard/cursor 和干净环境复建尚未闭合。

修改方向：加入 YAML `plugin_modules` 或 entry-point 发现、`validate-config`/dry-run、统一 provider/scorer 模板、一条命令 smoke，并把插件源码身份纳入 checkpoint。

### 5. 故障边界还不完整

真实 reward 服务中途退出的 O4 尚未完成；多卡 hard exit/collective hang、真实断电仍没有完整实验。合并审计发现的旧 artifact 不安全加载、marker tree digest 未重验、状态权威性与目录 fsync 问题已在本地修复并回归，但仍需远端故障注入验证真实进程边界。

修改方向：远端验证修复后的 marker/tree-digest 与 recovery 顺序；reward 请求继续使用幂等 step/token；服务失败不得提交 optimizer、metrics、latest 或 checkpoint；需要真正 hard timeout 的不可信 scorer 使用进程隔离，线程 provider 只提供明确的协作取消能力。

## 合并后远端验证顺序

1. Post-merge correctness：Wan W5 exact resume、W7b 异构 selected-index parity 与 reward_general HPS direct/HTTP/infra parity 已完成；下一步完成 SD3 deterministic resume，HPS 后续只进入训练效果与性能验证。
2. P3/O4/D3 复合预检：`480x832/5帧/4-step` batch 2 推理边界、batch 1 real-HPS active/control、reward 服务中断零部分提交、同内容换路径 resume。
3. 新高语义 bounded effectiveness：优先 World-R1，3 seeds × active/control × 10 steps，同时预注册 HPS、PickScore、视频护栏和人工面板。
4. P2 native/infra overhead：warmup 后至少 5 次重复，报告 p50/p95、samples/s、显存、GPU 利用率、I/O，并分开 deterministic/performance runtime。

## 当前不得宣称

- 不得宣称 VisualRL 已稳定提升生成质量。
- 不得宣称 Flash 已提升训练吞吐；当前只证明状态存储减少。
- 不得把 zero-LR/control、短 smoke 或一步更新扩展成中长程稳定性结论。
- 不得把 P3 候选筛选写成多 seed 训练效果通过。
- 不得把已完成的 Wan/reward score-path post-merge correctness 外推为 SD3 已复验、HPS 能改善质量或 reward 服务有性能收益。
