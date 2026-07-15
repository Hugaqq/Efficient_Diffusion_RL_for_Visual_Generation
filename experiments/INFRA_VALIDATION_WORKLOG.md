# VisualRL 修改与验收台账（历史快照）

更新时间：2026-07-15

> 状态：本文件冻结合并阶段的修改与验收映射，不再维护当前 backlog。当前状态和顺序只以 `docs/PROJECT_OVERVIEW.md` 与 `experiments/EXPERIMENT_PLAN.md` 为准。本文历史 `P5/P6` 分别表示双卡 correctness/扩展效率；当前计划已将它们迁移为 `MG1/MG2`，避免与后来建立的 P5–P7 单卡优化实验冲突。

这份文档记录合并阶段 infra 验证中修改了什么、为什么修改、证据到哪一步，以及当时仍缺哪些实验。

## 当前结论

| 工作流 | 当前状态 | 现在能说明什么 | 还不能说明什么 |
|---|---|---|---|
| TempFlow 原文公式与 sampler 对齐 | E3a v6、E3b v3 均已通过 | infra 已精确复现 reference 的训练数学、382 个 gradient、单步 LoRA 更新，以及显式相同随机输入下的完整 ODE/SDE sampler | 训练有效性仍未通过，不能扩容 |
| TempFlow 训练效果 | T3b 3 active + 3 zero-LR control 已完成；安全效果门槛仍失败 | 六 run/60 步机械正确，control 精确零；active mean `+0.10386`、CI95 下界 `+0.06757`，3/3 seed 和三色均值均正，证明当前 reward 目标可被推动 | seed307/419 亮度比升至 `1.54×/1.92×`，seed419 空间方差也超限；疑似 reward hacking，不能扩容或声称图像质量有效 |
| TempFlow 失败诊断 | T2c、T3b 已完成 | 全量 target、分层窗口与数据泄漏已排查；平衡多样 prompt 后 reward 统计显著转正 | 像素护栏失败成为当前主阻塞；先做独立 scorer、盲评和 reward-hacking 分析，长程继续锁定 |
| 数据 provider 与追溯 | A5、D1、D2、D4、D6、D7 已通过；D5 单进程通过 | 固定 HF/local 可追溯；坏输入/cache 与泄漏 fail-fast；deterministic shuffle 的连续6步与3+resume完全一致；reward cache 并发写、损坏隔离与内容失效门槛通过 | D5 多卡 shard 因核心只支持单进程而未满足；外部图像/视频 dataset provider 不支持 |
| 效果分析与报告 | Q1、Q3-Q6 已完成；Q2 面板已冻结 | SD3/Wan 独立 PickScore 均可重复；Q3 的240视频/120配对、时间质量指标、bootstrap和contact sheet可重建 | SD3与Wan独立效果均失败；Q2仍需至少2名真实人工评审 |
| Wan / World-R1 / Flash-GRPO | W0-W7b 已通过；`2adfbfd` 已重跑 W5/W6-general/W7b；W8机械通过/效果失败；Q3安全通过/独立效果失败 | 当前提交再次证明真实 Wan 异构 Flash parity、World/Flash deterministic resume 和 `reward_general` 协议 parity；历史视频安全证据仍保留 | 当前提交尚未重跑 HPS-backed Wan 训练；HPS与独立PickScore均未证明质量提升，尚未证明Flash吞吐收益，W9锁定 |
| `2adfbfd` post-merge correctness | 本地与 Wan gate 已通过 | non-distributed 883 passed、2 skipped、5 deselected；Gloo 5 passed、885 deselected；W7b 480 梯度逐位一致；W5 六段 valid、两份 11-gate 比较 exact；general reward 三方 exact 且坏请求 500 | SD3 与 HPS-backed Wan 训练 post-merge 尚待执行；本轮不提供质量提升或速度提升证据 |
| Wan 权重获取 | 已通过 W0 | 本地官方 CLI 完成固定 revision `0fad780a...`；服务器重新读取 28,928,887,859 bytes，19/19 文件 size/SHA256 与本地 manifest 完全一致 | 只证明权重身份；真实 load、显存与训练仍由 W1-W4 决定 |

## 本轮 TempFlow 相关修改

| 文件 | 修改内容 | 对应验收目的 |
|---|---|---|
| `visual_rl/configs/presets/sd3_tempflow_adapter.yaml` | 对齐 reference 配置：`advantage_epsilon=1e-4`、`clip_range=1e-4`、`2.25 * transition_std`、六分支、AdamW、`max_grad_norm=1.0` 等 | 避免用 infra 默认值冒充原文算法 |
| `visual_rl/configs/schema.py` | 增加 advantage epsilon、advantage dtype、最大梯度范数等显式配置 | 让关键算法语义进入配置、checkpoint 身份和审计记录 |
| `visual_rl/optimizers/advantages.py` | advantage 归一化 epsilon 可配置，并支持保留 float64 reference 计算 | 逐项对齐原实现 advantage 数值 |
| `visual_rl/optimizers/tempflow_grpo.py` | 增加 reference `std_dev_t` 权重模式，并在需要时保留 advantage dtype | 对齐 temporal weighting 和 loss dtype |
| `visual_rl/optimizers/factory.py` | 把新配置传给 advantage 和 optimizer plugin | 确保 YAML 配置实际生效 |
| `visual_rl/optimizers/algorithm_plugin.py` | 训练前切换 adapter 模式，并执行可配置 gradient clipping | 对齐原实现的 train mode 和更新顺序 |
| `visual_rl/model_adapters/base.py` | 增加 sampling/training 生命周期钩子 | 让不同模型明确控制 eval/train 状态 |
| `visual_rl/model_adapters/sd3.py` | 增加 TempFlow reference mode、完整六分支训练 batch、正确的 transition std、sampling/train 状态切换 | 对齐 SD3 rollout 与 log-prob 重算语义 |
| `visual_rl/runner.py` | rollout 前调用 sampling 钩子 | 保证采样和训练阶段模式可审计地分开 |
| `scripts/legacy_cli.py` | 增加显式 `--allow-initial-clipping`；默认仍保留严格零 clipping 门槛 | 只在复现 upstream eval/train 语义的实验中允许初始 ratio 进入 clip 区间 |
| `scripts/legacy_cli.py`、`scripts/remote_smoke.py` | 增加显式 `--tempflow-execution-mode`，默认保持 `reference-compatible`；`policy-identity` 强制拒绝初始 clipping 放宽 | 把原生 parity 与效果训练语义分开记录，避免结论串用 |
| `tests/test_visual_rl.py` | 增加 reference preset、advantage dtype、temporal weight、deterministic/checkpoint 等回归 | 防止后续重构悄悄改回默认公式 |
| `experiments/sd3_tempflow_numerical_alignment_20260714/` | 冻结 E3a recipe、探针、源码归档和说明 | 保存 correctness 实验的可复核输入 |
| `experiments/sd3_tempflow_reference_effect_20260714/` | 冻结三 seed active/control 效果实验 | 将“公式正确”与“训练有效”分开验证 |
| `experiments/sd3_tempflow_effect_diagnosis_20260714/` | 从冻结证据重建 T2a 诊断、原生图表和 HTML 报告 | 把 high clip、branch 位置、seed419/red slice 与图像护栏分层解释；不重新训练、不覆盖失败结果 |
| `experiments/sd3_tempflow_mode_ablation_20260714/` | 固定轨迹比较 eval/train recompute | 排除 transformer mode 是 drift 原因 |
| `experiments/sd3_tempflow_batch_kernel_ablation_20260714/` | 比较 single-parent 与 six-branch BF16 forward | 将 first divergence 定位到 transformer batch shape，而非 SDE std |
| `experiments/sd3_tempflow_shared_prefix_identity_20260714/` | 调用正式 adapter shared-prefix 路径重算九个 transition | 证明 policy-identity 模式能使 old/new log-prob 与 clipfrac 精确归零，并保留全部 staging 失败记录 |
| `experiments/sd3_tempflow_policy_identity_effect_20260714/` | 冻结 T3 source、recipe、单步 preflight、六 run 顺序 runner 和非 checkpoint validator | 只改变 execution mode，先做严格 1-step active/control 门槛，再运行匹配 3-seed 10-step 效果复测 |
| `experiments/sd3_tempflow_reward_data_ablation_20260714/` | 增加全量 target 审计、三条无歧义平衡数据、3-step runner/validator 和冻结 recipe | 区分 dataset 总体平衡、短窗口抽样方差与 reward/update 方向；失败不允许用更长训练掩盖 |
| `experiments/sd3_tempflow_stratified_effect_20260714/` | 从原 train 构造 90 条多样、三色各30且三 seed 窗口均为 4/3/3 的 fixture；冻结六 run runner、双层 validator 和 recipe | 复测 T3 失败是否由 10-step 窗口高方差驱动，同时避免循环三条 prompt 的过拟合结论 |
| `experiments/sd3_tempflow_resume_equivalence_20260714/` | 复用 T3 连续 10-step reference，新增 5-step split、resume-to-10 runner 和严格 state/artifact comparator | 验证 TempFlow 原配置在 deterministic policy-identity 路径上的真实 checkpoint/resume 等价性 |

说明：`allow_initial_clipping` 不是全局放宽正确性标准。它只用于审计 upstream/reference-compatible six-branch batched recompute 行为；T2b 已证明本次偏移并非 eval/train mode 引起。后续 effectiveness 走显式 policy-identity shared-prefix 模式，仍要求初始 old/new log-prob 一致和 `clipfrac=0`。

## 本轮 Wan 相关修改

| 文件 | 修改内容 | 对应验收目的 |
|---|---|---|
| `visual_rl/model_adapters/wan.py` | 接入 Wan upstream LoRA targets；冻结非训练模块；增加 gradient checkpointing、sampling/train 模式、PEFT-only save/load、运行时 metadata 和 World-R1 不同函数签名兼容；recompute 仅将 transformer 输入副本转为 BF16，SDE current/next latent 保持原始 FP32 | 让 32GB 单卡只训练 LoRA，避免 checkpoint 写出完整 Wan transformer，并同时保持 upstream BF16 transformer 输入与 SDE 轨迹精度 |
| `tests/test_wan_latent_precision.py` | 用非 BF16 可精确表示的 FP32 current/next latent 覆盖 Flash 六返回值与 coefficient 路径；断言 transformer 实际收到 BF16，而 SDE 逐位收到原始 FP32 | 防止 recompute 把轨迹 latent 先 BF16 舍入后再转回 FP32，固定 `2adfbfd` 的 dtype 合同 |
| `experiments/wan_world_r1_flash_smoke_20260714/world_r1_real_wan.yaml` | 真实 Wan + World-R1 一步 bounded smoke 配置 | 验证 rollout、video reward、GRPO、backward、更新和 checkpoint 闭环 |
| `experiments/wan_world_r1_flash_smoke_20260714/flash_grpo_real_wan.yaml` | 真实 Wan + selected single-step/rectification 一步 smoke 配置 | 验证当前 Flash-style loss 路径可在真实 Wan 上运行 |
| `experiments/wan_world_r1_flash_smoke_20260714/feasibility.json` | 记录单卡可行范围和论文规模不可行项 | 防止把 bounded smoke 写成 paper-scale 复现 |
| `experiments/wan_world_r1_flash_smoke_20260714/download_wan_resumable.sh` | 固定模型 revision，逐文件断点下载与大小校验；改成 macOS/Linux、本地/服务器两用 | 解决服务器直连不稳定，并保证上传前快照完整 |
| `experiments/wan_world_r1_flash_smoke_20260714/download_wan_hf_official.sh` | 镜像 403 后改用官方 Hugging Face CLI，单 worker 复用已有文件，并生成 19 文件 SHA256/size manifest | 保留已下载的 18GB，避免过期镜像链接无限重试，并为本地到服务器身份校验提供内容指纹 |
| `experiments/wan_world_r1_flash_smoke_20260714/README.md` | 固定两个真实单步 run 的硬门槛和结论边界 | 明确 smoke 只证明集成，不证明训练效果 |
| `experiments/wan_world_r1_flash_smoke_20260714/remote_wan_load_sample.py` | 独立进程真实 load、最小视频 rollout、显存记录、finite gate 与 PEFT-only 保存 | 覆盖 W1，并为 W2 冻结参数与 rollout 基准 |
| `experiments/wan_world_r1_flash_smoke_20260714/remote_wan_roundtrip.py` | 新进程加载 PEFT，验证参数对象不替换、trainable hash 与同 seed rollout 精确复现 | 覆盖 W2，防止“文件存在”被误当作可恢复 checkpoint |
| `experiments/wan_world_r1_flash_smoke_20260714/verify_wan_snapshot.py`、`remote_wan_train.py` | 对 19 文件逐个做 size/SHA256/revision 校验；训练 wrapper 记录显存、耗时、参数差、梯度、状态和 checkpoint 完整性 | 让 W0/W3/W4 的通过与失败都由机器可读硬门槛决定 |
| `experiments/wan_world_r1_flash_smoke_20260714/environment_update.json` | 记录 `flow_grpo` 增补 `ftfy==6.3.1`/`wcwidth==0.8.2`，无 root | 修复 Diffusers Wan prompt 清洗的明确缺依赖；不把环境变化混入算法修改 |
| `experiments/wan_world_r1_flash_smoke_20260714/remote_wan_resume_run.py`、`compare_wan_resume.py` | deterministic continuous2 与 1+resume-to-2 runner；runner 先配置 deterministic runtime、再允许任何 `torch.cuda.*` 调用；递归比较 metrics、manifest、PEFT、optimizer/plugin/RNG 与 identity | 覆盖 W5，并保留 attempt 1 的 harness 启动顺序失败；attempt 2 继续使用 exact 门槛而不事后放宽 |
| `experiments/wan_world_r1_flash_smoke_20260714/remote_wan_train.py` | 增加显式 active/zero 更新期望；active 要求 LoRA 非零变化，zero-LR control 要求参数 hash 和差值精确为零；默认仍为 active | 让同一真实训练 wrapper 能执行匹配 control，而不会把“没有更新”误判为训练失败 |
| `experiments/wan_w8_real_reward_preflight_20260714/` | 冻结 World/Flash 四份配置、同卡 HPS 服务编排、进程所有权、显存监控、active/control rollout/reward/参数硬门槛和无 checkpoint 证据 | 覆盖 W8 1-step preflight，并在进入 3-seed 10-step 前证明真实评分服务与 Wan 训练能在单张 32GB GPU 共存 |
| `experiments/wan_w8_real_reward_effect_20260714/` | 冻结 3-seed × active/control 的 12 个 10-step run、固定 HF prompt/HPS 身份、阶段 profiler、bootstrap 与视频像素护栏；保存两次训练前失败和正式无 checkpoint 证据 | 覆盖 W8 多步机械正确性与训练效果硬门槛；结果为机械通过、World/Flash 效果均失败，因此阻止 W9 扩容 |
| `experiments/wan_q3_video_quality_20260714/` | 对W8的240个视频做1,200帧双遍固定PickScore、时间加速度、局部闪烁、清晰度、运动/塌缩护栏和6张contact sheet；保留跨batch身份失败attempt1 | 覆盖Q3独立视频诊断；视频安全通过，但World/Flash独立效果CI均跨0，确认W9继续锁定并暴露64px/2-step语义基线过弱 |
| `visual_rl/model_adapters/wan.py`、`visual_rl/rollout/single_step.py`、`visual_rl/rollout/rectification.py` | 接入 Flash 原生单随机步 sampler/recompute kernel；异构 selected index 按标量原生调用分组后恢复原样本顺序；按实际 scheduler timestep 使用 reference 十项 rectification 表 | 让 W7/W7b 验证真正的 Flash 轨迹语义和多样本顺序，不再用 World-R1 全 SDE trajectory 截断冒充 native Flash |
| `visual_rl/third_party/legacy.py` | 将仓库实际使用的相邻 `code_base/<repo>` 加入默认 reference 搜索根 | 让 World-R1/Flash 默认配置在当前 checkout 可解析，不再必须手写绝对路径 |
| `visual_rl/feedback/clients.py`、`visual_rl/feedback/world_r1_rewards.py` | 增加真实 World-R1 JPEG/pickle wire format、camera trajectory、component metadata 和响应 count/finite/schema 校验 | 修复原客户端把 raw tensor 直接发送给 reference server 的协议错误；异常响应 fail closed |
| `experiments/wan_world_r1_flash_smoke_20260714/flash_grpo_diffusers_033_compat.patch`、`flash_grpo_generator_plumbing.patch` | 仅对 staging reference 补 Diffusers 0.33 API、无 cache context、homogeneous index 和 selected-SDE generator 传递 | 保留原算法数学，同时消除 release API 不兼容和“同 seed 随机步仍漂移”的上游 RNG 缺陷 |
| `experiments/wan_world_r1_flash_smoke_20260714/world_r1_general_fail_closed.patch` | 仅对 staging reference 移除内部异常返回固定 `0.5` 的静默兜底，让错误在 HTTP 边界成为 500 | 防止 reward 服务“看似成功但给出伪分数”；attempt 1 的原始 fail-open 证据仍保留 |
| `experiments/wan_world_r1_flash_smoke_20260714/remote_wan_flash_native_parity.py`、`remote_wan_flash_heterogeneous_parity.py`、`remote_world_r1_general_reward_probe.py`、`remote_world_r1_3d_reward_probe.py` | 冻结真实 Wan native parity、异构 batch、HPS 三方 parity 与 DA3/Qwen live-server 门槛 | 覆盖 W6/W7/W7b，保留 attempt1 失败，不用修复后结果覆盖原因链 |
| `experiments/wan_world_r1_flash_smoke_20260714/reward_environment_update.json` | 冻结专用 reward venv 与用户态 CUDA 12.8.61/gsplat sm_120 编译链；MoviePy 固定到 1.0.3 | 解决 `moviepy.editor` 与服务器无系统 nvcc 导致 3D renderer 被禁用的问题；不修改 `flow_grpo` 或系统环境 |

当前 Flash 已接入原生 single-selected-step 存储语义，并在 W7 及 `2adfbfd` post-merge W7b 逐位对齐；但仍执行全部 denoising forward，因此只证明 21 个 state 降为 2 个 state 的轨迹存储下降，不证明吞吐提升。异构 selected index 的分组/顺序恢复、原始 FP32 SDE latent 和 480 个梯度 tensor 已通过真实 Wan 门槛；速度结论仍必须等待 P2 matched native/infra 实测。

## 本轮数据 provider 与追溯实验

| 文件 | 修改内容 | 对应验收目的 |
|---|---|---|
| `experiments/a5_d1_hf_dataset_provider_20260714/source/` | 用官方 Hugging Face CLI 下载固定 revision 的 PartiPrompts TSV 与 Apache-2.0 数据卡 | 冻结真实外部数据的版本、许可和源文件身份 |
| `experiments/a5_d1_hf_dataset_provider_20260714/build_snapshot.py` | 校验源文件 hash，等距选取 12 行，生成带 repo/revision/row/prompt ID/预处理版本的离线快照 | 证明无需网络即可重载同一有序样本集合 |
| `experiments/a5_d1_hf_dataset_provider_20260714/hf_snapshot.yaml`、`local_copy.yaml` | 同一 TinyDiffusion/GRPO 一步 recipe，仅交换 HF 物化文本与字节相同的本地文本路径 | 验证 dataset provider 不要求 runner 专用分支 |
| `experiments/a5_d1_hf_dataset_provider_20260714/validate_results.py` | 比较内容指纹、metrics、rollout、media、adapter、规范化 SampleManifest，并执行 sample-to-source join | 验证统一 manifest 和原始样本追溯 |
| `experiments/a5_d1_hf_dataset_provider_20260714/RESULTS.md` | 固定所有通过门槛和“不证明训练有效”的结论边界 | 防止把 provider smoke 扩大成效果声明 |
| `visual_rl/datasets/prompt_dataset.py` | 增加 NFKC/casefold/标点空白规范化，以及阈值 `0.92` 的近重复 split 审计；发现冲突时 fail-fast | 防止训练集轻微改写泄漏到 held-out 后制造虚假效果 |
| `tests/test_visual_rl.py` | 增加规范化重复、近重复拒绝与干净 split 正例 | 固定 D7 公共行为 |
| `experiments/d7_prompt_leakage_20260714/` | 保留修复前漏检，审计旧 3,600/12 数据与当前 T3b fixture | 区分 infra 检测能力和某个既有数据集是否干净 |
| `visual_rl/evaluation/cross_run.py` | 将 mechanical execution 与 pixel guardrail 拆成两个 gate | 避免把“run 跑完但质量护栏失败”误写成执行失败，同时保持最终不合格 |
| `tests/test_visual_rl.py` | 增加 cross-run gate 语义回归 | 固定字段含义和失败结论 |
| `experiments/t3b_analysis_acceptance_20260714/` | 从冻结 T3b raw records 重算统计、slice/correlation、SVG 与 HTML，并验证双次重建 hash | 覆盖 Q4 reward-hacking 分析、Q5 统计复核、Q6 报告可重建 |
| `visual_rl/configs/schema.py`、`visual_rl/datasets/prompt_dataset.py` | 增加显式空 prompt 策略，默认 error、可显式 skip | 防止数据读取静默改变样本数 |
| `visual_rl/artifacts/checkpoint.py` | 默认 error 对干净旧数据不改变 fingerprint v2；显式 skip 进入身份 | 在加强数据规则时避免无关破坏旧 checkpoint resume |
| `visual_rl/artifacts/manifest.py` | 缺失/未知 record 字段改为带 index 的结构化 ValueError | 提高坏 artifact 可定位性 |
| `visual_rl/rollout/cache.py` | 增加 cache triplet 完整性、解码、字段、batch 和 SHA256 验证 | 检出损坏媒体与半写 cache |
| `experiments/d4_bad_data_handling_20260714/` | 系统注入空行、重复、坏 UTF-8、缺字段、坏 media 和半写文件 | 覆盖 D4 正反门槛 |
| `experiments/d5_shuffle_cursor_resume_20260714/` | 7 prompt、batch3 的连续6步与3+resume-to-6；验证 epoch/index cursor 与 WORLD_SIZE=2 拒绝 | 覆盖单进程 D5，并如实冻结多卡 sharding 缺口 |
| `visual_rl/feedback/cache.py` | 增加每 key 跨进程锁、唯一临时文件、`fsync`、原子替换，以及损坏内容 hash 隔离 | 防止并发 reward cache 出现半写文件，并让损坏可检测、可重建 |
| `experiments/d6_cache_concurrency_20260714/` | 10 进程同 key 压测、真实 router cache 截断恢复，以及 metadata/media/scorer version identity 失效 | 覆盖 D6 并冻结机器可读证据；不扩大为 DDP rollout cache 声明 |
| `visual_rl/artifacts/checkpoint.py` | checkpoint format v2 增加 adapter/training-state 逐文件 size+SHA256；resume 先只读验证再加载；保留 v1 reader，并增加显式非原地 v1→v2 migration | 防止损坏权重先污染模型，保证失败写入不被当成可恢复 checkpoint，并提供可审计迁移路径 |
| `visual_rl/artifacts/manifest.py` | 缺失/未知 schema 改为 fail closed；旧无版本 manifest 只能显式迁移到 v1 | 避免 reader 静默猜测 artifact schema |
| `experiments/c3_c4_checkpoint_integrity_20260714/` | 在 adapter/state/artifact 阶段注入失败，篡改四类 checkpoint 内容，并验证 v1/v2 与 manifest schema 迁移 | 覆盖 C3/C4 的原子发布、加载前完整性和前后兼容门槛 |
| `visual_rl/rollout/cache.py` | rollout metadata 增加实际 batch seed | 让 seed 能在 cache 与 SampleManifest 之间交叉对账 |
| `visual_rl/artifacts/audit.py` | 新增 run 级 fail-closed 审计器，连接 metrics、manifest、rollout cache、resolved config、checkpoint 与 latest | 把单文件“可读”提升为跨 artifact provenance 一致性检查 |
| `experiments/c5_cross_artifact_consistency_20260714/` | 两步 run 双次确定性审计，并逐一篡改 reward mean、prompt、sample ID、seed、model name、config fingerprint | 覆盖 C5，确保六类不一致全部被检出 |
| `experiments/c6_default_runtime_tolerance_20260714/` | 从冻结 E2b 同 checkpoint 三分支证据重算默认 BF16/CUDA 漂移分布，保持原 `1e-6` 门槛不变 | 覆盖 C6，并明确默认模式不能用于精确 resume；不把观察到的任意误差事后称为可接受 |
| `experiments/q1_q2_independent_evaluation_20260714/run_pickscore_remote.py` | 固定 PickScore/processor 权重 hash，对432项图文两遍评分并按 active/control、seed、颜色聚合 | 覆盖 Q1；将 scorer 可复现性与训练效果门槛分开 |
| `experiments/q1_q2_independent_evaluation_20260714/blind_panel/` | 冻结48对 opaque A/B 面板、12个 identical controls、response template、协议和独立 blinding key | 准备 Q2，同时防止 URL/文件名泄露 condition/phase；人工标签仍待真实 rater |
| `experiments/o1_process_interruption_20260714/` | 五个 child process 在 reward/cache/optimizer/checkpoint/latest 后以91退出，再从持久化 latest 恢复并对比两步 reference | 覆盖 O1 的 exactly-once 更新与样本记账；首次缺 PYTHONHASHSEED staging 失败单独保留 |
| `visual_rl/artifacts/status.py`、`visual_rl/runner.py` | 原子记录 running/failed/completed、PID、target/global/latest step 和错误；只读识别 dead PID 为 stale-running；聚合只接受 completed valid | 让自动化不再用日志猜运行状态，禁止缺失/失败 run 进入聚合 |
| `visual_rl/artifacts/manager.py` | 新进程启动时清理六类已知单 writer stale `.tmp` | 让 metrics/report 半写可在安全恢复时自动清理 |
| `experiments/o3_o5_failure_status_20260714/` | 注入 checkpoint ENOSPC、metrics/report 半写、只读目录，并验证 complete/failed/live/stale/missing 状态 | 覆盖 O3/O5 与精确恢复门槛 |
| `experiments/o2_cuda_oom_20260714/remote_worker.py` | 补齐可执行入口，并在 GPU2 上以超大真实 CUDA tensor 与小尺寸 control 成对运行 | 覆盖 O2 的真实 OOM fail-closed；自动降级能力仍明确不支持 |
| `experiments/s3_gpu_process_safety_20260714/` | 用 UID、PID start tick 和 command token 绑定自建进程；错身份拒绝，精确身份才允许 SIGTERM | 覆盖 S3 的单机所有权门槛；不触碰其他用户进程 |

## 同一工作区中较早完成、但仍未提交的 infra 修改

下列文件属于此前 E2/E2b/I0 验证，不是 Wan 下载临时产生的改动；保留在同一工作区，需要在最终合并前一起审查：

- `visual_rl/core/determinism.py`、`visual_rl/core/seed.py`、`docs/DETERMINISTIC_RUNTIME.md`：显式 deterministic runtime。
- `visual_rl/artifacts/checkpoint.py`：checkpoint 指纹 v2、数据内容身份与路径来源分离、结构化 mismatch 诊断。
- `docs/PROJECT_OVERVIEW.md`、`experiments/EXPERIMENT_PLAN.md`、`experiments/sd3_runtime_determinism_20260713/RESULTS.md`：架构说明、实验账本和结果更新。
- `experiments/sd3_single_step_determinism_20260713/`、`experiments/sd3_deterministic_resume_repeat_20260713/`、`experiments/sd3_deterministic_integrated_resume_20260713/`、`experiments/sd3_v2_path_resume_20260714/`：已完成的确定性、resume 和路径迁移证据。

这些既有修改已经分别通过真实 SD3 严格复现或 resume 实验，但最终提交前仍要重新运行统一测试并审查完整 diff。

## 全量待补实验矩阵（唯一权威清单）

下面的清单是当前 VisualRL infra 范围内的完整验收 backlog。新增实验必须先登记 ID、依赖、门槛和结论边界；没有列在这里的临时尝试不能改变项目验收结论。

状态含义：`进行中` 表示已有受监控任务；`待执行` 表示依赖已满足；`锁定` 表示上游硬门槛未通过；`部分通过` 表示已有窄证据但覆盖不完整。

### A. 抽象能力与新组件接入

| ID | 实验 | 状态 | 通过标准 |
|---|---|---|---|
| A1 | 外部算法插件最小接入 | 已通过 | 独立 `experiments/` module 在核心文件哈希不变时完成 config -> rollout -> loss -> update -> metrics -> checkpoint；实现身份记录外部类 |
| A2 | 算法互换 contract | 已通过（TinyDiffusion） | 同一 TinyDiffusion/prompt/reward 上 GRPO full、TempFlow branch、Flash selected-step 均通过严格 batch shape/dtype、credit metadata、公共 metrics/manifest/checkpoint；真实 native parity 仍由 T1/W7 负责 |
| A3 | 模型 adapter conformance | 已通过（当前三类） | Fake、真实 SD3、真实 Wan 均覆盖 load/sample/recompute/save/load/runtime metadata；Wan W1-W5 进一步通过 PEFT-only、finite backward 和 exact resume |
| A4 | scorer 插件装载与隔离 | 已通过（本地） | 外部 luminance batch scorer 与内置 scorer 通过方向、cache identity/version、坏 shape、timeout/retry、invalid/raise 和健康 router 隔离；真实 World-R1 服务仍由 W6/O4 覆盖 |
| A5 | dataset provider 可替换性 | 已通过 | 固定 revision PartiPrompts 物化快照与本地字节等价文本经同一未分支 runner 各完成一步；metrics、rollout、adapter、规范化 SampleManifest 精确一致 |
| A6 | “加入新算法”改动面审计 | 已完成，有边界 | A1 核心改动 0，只新增外部 module/config/harness；但通用 CLI 没有 YAML `plugin_modules`/entry-point 自动发现，第三方插件仍需薄启动器，记为产品化缺口 |

### T. TempFlow 正确性与效果

| ID | 实验 | 状态 | 通过标准 |
|---|---|---|---|
| T1 | E3b 独立 sampler 数值对齐 | 已通过 | v3 中 10 个 ODE state、9×6 SDE targets、old log-prob、media 和 RolloutBatch 字段全部逐位一致；参数未变化 |
| T2a | 高 clipfrac 与失败 slice 观测定位 | 已通过 | 冻结证据证明 zero-LR control 也高 clipping，量化 branch-position 放大、active/control 相关、seed419 target mix、red slice 和像素护栏；报告可重建 |
| T2b | eval/train mode 固定轨迹因果消融 | 已完成，假设门槛失败 | eval/train recompute 逐位相同；eval/eval 仍最大偏移 `0.007318`、`clipfrac=1.0`，排除 transformer mode；参数不变、noise 消费与重复性门槛通过 |
| T2c | reward target 与训练数据平衡消融 | 已通过 | 全量 train 为 1,200/1,200/1,200、held-out 为 4/4/4，运行 target 36/36 一致；3-step balanced active 总体 `+0.01985` 且三色均正，control 精确零、像素护栏通过 |
| T2d | sampling/recompute batch/kernel/dtype 定位 | 已通过 | 单 parent old/new log-prob 精确一致；六分支 batch 复现 drift；first divergence 为 BF16 noise prediction，SDE mean 跟随、std 精确一致 |
| T2e | shared-prefix 实际 adapter policy-identity 复验 | 已通过 | v4 中正式 shared-prefix 路径覆盖 transition 0–8；old/new 最大差 `0.0`、`clipfrac=0.0`、参数指纹不变；v1-v3 staging 失败均保留 |
| T3 | 修正版 10-step active/control 复验 | 机械通过，效果失败 | 六 run 均 valid；60/60 步 log-prob/clipfrac 为 0，controls 精确零更新；active mean `+0.00851`、CI95 `[-0.04360,+0.06026]`、red `-0.03932`；D7 另发现旧 full train/held-out 有 2 对近重复，进一步禁止正面效果 claim |
| T3b | 分层平衡 3-seed 10-step 复测 | 机械/reward 通过，安全效果失败 | 六 run/60 步机械正确；active `+0.10386`、CI95 `[+0.06757,+0.13566]`、3/3 seed 与三色均值正；但 seed307/419 像素护栏失败，禁止扩容 |
| T4 | TempFlow reference 配置 resume | 已通过 | 连续10-step 与5+resume-to-10 的最终 LoRA、完整训练 state、逐步 metrics、manifest、step0/5/10 原始评估与 PNG 全部严格一致 |
| T5 | TempFlow 20/50/100-step 效果扩容 | 锁定 | T3b 的 CI/seed/颜色已通过但像素护栏失败；必须先由独立质量证据排除 reward hacking，不能仅凭 reward 上升解锁 |

### W. 真实 Wan / World-R1 / Flash-GRPO

| ID | 实验 | 状态 | 通过标准 |
|---|---|---|---|
| W0 | 固定 Wan snapshot 获取与身份校验 | 已通过 | revision `0fad780a...`，19/19 必需文件、总计 28,928,887,859 bytes 的 size/SHA256 在服务器重新读取后与 laptop manifest 完全一致；历史 README/assets/`.part` 不参与 loader 身份 |
| W1 | 真实 Wan load 与显存门槛 | 已通过 | GPU2 真实 `WanPipeline` + rank16 LoRA 完成 5帧64px/2-step；所有 tensor finite；峰值 allocated 13.55 GiB、reserved 13.81 GiB。attempt1 缺 `ftfy` 失败保留 |
| W2 | Wan PEFT save/load round-trip | 已通过 | PEFT 23,659,220 bytes、无完整 transformer；新进程 trainable hash 精确相同、Parameter identity 不变；media/trajectory/logprob/KL 全部逐位一致 |
| W3 | World-R1 真实 Wan 一步闭环 | 已通过 | 两条真实视频、local video reward、GRPO、2 timestep recompute、480/480 finite gradient tensor、5894217 nonzero element、LoRA delta L2 `0.188665`、PEFT v2 checkpoint 全通过 |
| W4 | Flash-style 真实 Wan 一步闭环 | 已通过（兼容路径） | selected step0、rectification 1.0、480/480 finite gradient tensor、5898240 nonzero element、LoRA delta L2 `0.171065`、PEFT v2 checkpoint 全通过；不声称 native Flash 效率 |
| W5 | World-R1 / Flash Wan resume 等价 | 已通过；`2adfbfd` post-merge 复验通过 | attempt 1 因 harness 在 deterministic runtime 前初始化 CUDA 而于训练前失败并保留；attempt 2 两算法各 continuous2、split1、fresh-process resume-to-2 共六段 valid，两份比较各 11 个 exact gate 全通过 |
| W6 | World-R1 reference reward server parity | 已通过；`2adfbfd` general attempt 1 复验通过 | post-merge direct/reference/infra 均为 `[0.260009765625, 0.1943359375]`、两组差值 0；坏图像/坏 pickle 为 HTTP 500，silent fallback false，仅暴露 loopback legacy 协议。3D parity 仍沿用既有证据 |
| W7 | Flash native selected-step sampler parity | 已通过 | 真实 Wan 20-step/index3 中 media、前后 latent、timestep、old/new log-prob、loss、480 梯度逐位一致；参数不变；保留 attempt1 RNG 漂移及 generator 修复证据 |
| W7b | Flash 异构 selected-index batch parity | 已通过；`2adfbfd` post-merge 复验通过 | transformer 输入为 BF16，SDE current/next 保持原始 FP32；真实 Wan 异构 batch 对独立 scalar reference 的 media/embedding/latent/timestep/log-prob/KL/loss/480 gradient 全部逐位一致，参数不变 |
| W8 | 真实 Wan 3-seed bounded active/control | 已完成：机械通过，效果失败 | 12/12 run 和 6/6 matched pair 机械有效，active 非零更新、control 精确零、视频像素护栏通过；World mean `-0.000418`、CI95 `[-0.002747,+0.002014]`，Flash mean `-0.000489`、CI95 `[-0.002356,+0.001249]`，均未稳定提升；GPU峰值23,616MiB并完全释放，W9锁定 |
| W9 | Wan 中长程稳定性/效果扩容 | 锁定 | W8 效果门槛通过后逐级增加 steps；监控显存漂移、NaN、视频崩坏、checkpoint 时间和 reward hacking |

### D. 训练数据与 SampleManifest 可靠性

| ID | 实验 | 状态 | 通过标准 |
|---|---|---|---|
| D1 | 固定 revision Hugging Face 数据集导入 | 已通过 | 固定 `nateraw/parti-prompts@944b...`，记录 repo/revision/config/split、源 hash、12 个 row/prompt ID、Apache-2.0 和预处理版本；离线重载精确一致 |
| D2 | 本地文件与 HF 数据统一 manifest | 已通过 | 两种来源生成规范化后精确相同的 24 条 SampleManifest；每个 sample 可经 prompt ID 反查固定源行、revision 和变换版本 |
| D3 | 内容身份与路径迁移 | 部分通过 | I0 已证明同内容换路径可 resume、内容变化会拒绝；本次真实 HF snapshot 与本地副本内容指纹、训练输出精确相同；还需 Wan run 复验 |
| D4 | 坏样本/重复样本/空数据处理 | 已通过 | 空/全空/重复/坏 UTF-8 prompt、缺/重复 manifest、损坏/缺字段/半写 rollout media 均明确 fail-fast；skip/repeat 必须显式，不静默改变样本数 |
| D5 | shuffle/cursor/resume 完整性 | 部分通过：单进程严格通过，多卡不支持 | 连续6步与3+resume 的 metrics/manifest/adapter/rollout 精确一致，完整 epoch source index 恰好一次；`WORLD_SIZE=2` 在输出前拒绝，故多卡 shard 门槛未满足 |
| D6 | 数据缓存并发与失效 | 已通过 | 6 writer×20 与4 reader×100 全部成功，400/400 读取均完整、无临时文件；坏 cache 按内容 hash 隔离并重算；metadata/预处理、media、scorer version 变化均失效 |
| D7 | 训练/held-out 泄漏检测 | 已通过，并发现旧数据问题 | exact/normalized/near-duplicate 合成门槛全部 fail-fast；旧 full train/held-out 检出 2 对近重复并阻止正面 claim；当前 T3b fixture 三类泄漏均为零 |

### Q. 评分、数据分析与效果解释

| ID | 实验 | 状态 | 通过标准 |
|---|---|---|---|
| Q1 | 独立固定版本评分器 | 已完成，scorer通过/训练效果失败 | PickScore v1 权重与 CLIP-H/14 processor hash 固定；432项两遍最大差0，control精确0；active mean `-0.001512`、CI跨0、仅1/3 seed正、三色均负，不能支持效果 claim |
| Q2 | 固定 before/after 人工盲评 | 面板已冻结，等待真实评审 | 36 active +12 identical control，opaque A/B 文件名与固定随机盲化；至少2名独立评审，禁止用模型替代；未收标签前不能判通过 |
| Q3 | 视频专项质量分析 | 已完成：安全通过，独立效果失败 | 240视频/1,200帧、120配对；两遍评分与step0身份精确，World/Flash均无塌缩且闪烁/清晰度/运动护栏通过；PickScore mean `+0.000205/+0.000047`但CI均跨0，不能支持质量提升 |
| Q4 | reward hacking / slice 分析 | 已通过 | 216 个 paired records 按 seed/颜色/长度与像素指标分析；reward 与空间方差/动态范围变化中等相关，两个 seed 像素护栏失败，因此明确阻止效果 claim |
| Q5 | 统计方法复核 | 已通过 | raw-record 独立 hierarchical bootstrap 在 `1e-15` 内复现 mean/CI/颜色/control；缺失 seed 阻止、重复 seed 拒绝、失败 slice 全保留 |
| Q6 | 报告可重建性 | 已通过 | 仅用无 checkpoint 冻结 evidence 重建 JSON/paired rows/两张 SVG/HTML；两次 manifest hash 精确一致，报告数字与 JSON 一致 |

### C. Checkpoint、确定性与 artifact 完整性

| ID | 实验 | 状态 | 通过标准 |
|---|---|---|---|
| C1 | SD3 deterministic/repeat/resume | 已通过 | D0/D1/E2c 已提供 backward 定位、跨 GPU repeat 和真实 resume 严格等价证据 |
| C2 | checkpoint 指纹 v2 迁移与拒绝 | 已通过 | 同内容换路径及 output/resume/target/save 变化允许；模型 revision、LoRA targets、dtype、scorer version/endpoint/实现、runtime、数据内容八类变化均用准确字段拒绝，且在 optimizer/plugin/RNG 恢复前发生 |
| C3 | checkpoint 原子性与损坏恢复 | 已通过 | adapter/state 失败均清理 staging 且不生成 checkpoint/latest；artifact record 失败不发布 latest；旧 latest 在下一保存失败后逐字节不变；adapter/state/metadata/额外文件损坏均在恢复参数前拒绝 |
| C4 | artifact schema 前后兼容 | 已通过 | checkpoint v1 保持可读且明确无 hash 保证；显式非原地迁移到 hash-bound v2 可验证；缺失/未知 SampleManifest schema 拒绝，旧无版本格式只能显式迁移 |
| C5 | metrics/manifest/checkpoint 交叉一致性 | 已通过 | 两步/8 样本的 step、sample ID、prompt、seed、reward、model metadata、data/config identity 与 latest 闭合一致；双次审计精确相同；六类单点篡改全部检出 |
| C6 | 默认性能模式容差审计 | 已完成，默认模式精确门槛失败 | 同 checkpoint 三分支/三对比较给出漂移分布；原预注册 adapter exact、metrics/held-out `1e-6` 三门槛全部失败；默认模式不能作精确 resume 声明，deterministic runtime 才可 |

### O. 故障恢复、可观测性与长稳

| ID | 实验 | 状态 | 通过标准 |
|---|---|---|---|
| O1 | 进程中断恢复 | 已通过（单进程 deterministic） | reward、rollout cache、optimizer update、checkpoint 提升、latest 发布后五次真实 child 终止；恢复后 adapter/metrics/manifest 与连续两步精确一致，steps `[0,1]` 无重复记账 |
| O2 | CUDA OOM 与自动降级 | 已通过 fail-safe；自动降级不支持 | GPU2 真实申请 192 GiB 触发 `torch.OutOfMemoryError`；failed/invalid、step0、无 metrics/latest/checkpoint，退出后显存归零；16 px matched control 完成。当前明确不支持自动降级，未伪装成已覆盖 |
| O3 | 磁盘不足/只读/半写入 | 已通过（本地 fault injection） | checkpoint ENOSPC、metrics/report 半写与只读目录均不推进 target-step latest/valid；旧 checkpoint不变，stale tmp恢复清理，最终 adapter/metrics/manifest exact |
| O4 | reward/scorer 服务超时与重试 | 部分通过，待真实服务中断门槛 | A4 已验证本地 timeout/retry/fail closed；真实 HPS/3D 坏 payload、静默 fallback 检出、cache 和正常停止已通过；仍需服务中途退出时不重复更新/记账 |
| O5 | 日志与进度完整性 | 已通过（单机 PID） | 原子 run_status 区分 completed/failed/live running/dead-PID stale running；聚合只接收 completed valid，failed/stale/missing 全拒绝；跨机器 heartbeat 仍不声明 |
| O6 | 中长程内存与句柄泄漏 | 锁定 | correctness 通过后运行不宣称效果的稳定性 workload；显存/主存/文件句柄无持续无界增长 |

### P. 训练效率、资源与扩展

| ID | 实验 | 状态 | 通过标准 |
|---|---|---|---|
| P1 | 单步阶段 profiler | 主要阶段已通过，细分仍待P2 | W8 12个真实Wan run已记录 load、rollout、reward、recompute/backward/optimizer、cache、checkpoint、artifact 与物理显存；反向/optimizer仍合并且无独立held-out阶段，留给P2细分 |
| P2 | infra overhead 基线 | 待执行 | 同模型/配置下 native 与 infra 比较吞吐、峰值显存和 I/O；数值/样本语义先对齐再谈性能 |
| P3 | 单卡 32GB 可行域 | 部分完成 | 15 个采样单元与 3 个一步训练单元通过；`480x832/5帧/4-step` 一步训练峰值 `28,336 MiB`。仍需 matched 多 prompt/seed、可重复 OOM 边界和真实 reward 资源布局 |
| P4 | 缓存与 checkpoint 成本 | 已通过（CPU语义 + 真实Wan观测） | cache on/off 的 metrics/reward/manifest/参数精确一致；reward backend 只调用1次且 hit 加速约545×；W8真实Wan每10-step checkpoint save均值`0.308s`、rollout cache均值`0.048s`。仍不把Tiny加速比外推为Wan吞吐 |
| P5 | 真正双卡训练正确性 | 锁定 | 使用 DDP/accelerate 等一个训练任务的并行；与单卡相同 effective batch 在容差内等价，无重复样本 |
| P6 | 双卡扩展效率 | 锁定 | P5 通过后报告吞吐、step time、通信占比和峰值显存；GPU2/3 各跑独立 run 不计扩展 |

### S. 环境、安全与最终交付

| ID | 实验 | 状态 | 通过标准 |
|---|---|---|---|
| S1 | 环境可重建性 | 部分通过 | 冻结 Python/PyTorch/CUDA/diffusers/PEFT 与 source hash；还需从干净环境执行 CPU acceptance 和一个真实 adapter load |
| S2 | runtime 不兼容拒绝 | 已通过（单进程合同） | 既有 cu128/cuda13 mismatch 与本轮 dtype/deterministic bundle/模型 revision 结构化拒绝均通过；全部发生在 optimizer/plugin/RNG 恢复前。跨机器调度不在此门槛内 |
| S3 | GPU/进程所有权安全 smoke | 已通过（单机、自有进程） | 普通 UID、`CUDA_VISIBLE_DEVICES=2`；PID 必须同时匹配 UID/start tick/command token，伪造 start tick 不发信号，精确记录只终止自建 child；不声明集群级调度 |
| S4 | 从冻结材料重跑 end-to-end | 锁定 | 用文档、source archive、data/model revision 从新输出目录重跑一个 SD3 和一个 Wan 闭环，结论与证据一致 |
| S5 | 最终验收报告 | 锁定 | 汇总通过、失败、豁免、资源边界和不支持能力；每项 claim 都链接到 recipe、结果和硬门槛 |

## 当前关键路径

```text
W0 -> W1 -> W2 -> W3/W4 -> W5 -> W7(pass) -> W7b(pass)
                                      + W6(pass) -> W8(mechanical pass/effect fail) -> Q3(safety pass/effect fail); P3 next, W9 locked
E3a(pass) -> T1(pass) -> T2a(pass) -> T2d(pass) -> T2e(pass) -> T3(effect fail) -> T2c(pass) -> T3b(reward pass/pixel guard fail); T4(pass); T5 locked
A1/A3/A4/A5(pass) -> abstraction claim, pending real Wan for A3
D1-D2/D4/D6/D7(pass) + C3-C5(pass) + D3/D5/C2(partial) -> data/checkpoint reliability claim
Q1/Q3-Q6(completed; independent effects fail) + Q2(await human) -> effectiveness interpretation claim
O1-O6 + P1-P6 + S1-S4 -> operational/efficiency claim
all required gates -> S5 final acceptance
```

`2adfbfd` 的本地统一回归、Wan W5/W7b post-merge correctness 与 `reward_general` 协议 parity 已完成，完整索引见 [WAN_RESULTS_2adfbfd.md](postmerge_validation_20260715/WAN_RESULTS_2adfbfd.md)。下一步补 SD3 真实路径与 HPS-backed Wan 训练 preflight；general reward parity 不能替代 active/control 训练。`W8` 与 `Q3` 的既有效果失败结论不变，`W9`保持锁定；P2 速度 claim 同样保持锁定。P3、`O4`、`S1/S4` 可继续推进，Q2等待至少2名真实评审，D5多卡与P5共用单进程阻塞。

## 尚未完成项目汇总（按依赖排序）

这份汇总只列“仍需动作”的项目；具体通过标准以上方唯一权威矩阵为准。

1. **当前关键门槛**：`2adfbfd` 的 Wan W5/W7b correctness 与 `reward_general` 协议 parity 已重验；仍需完成 SD3 与 HPS-backed Wan 训练 post-merge gate。W8/Q3 的机械与视频安全通过但HPS/PickScore效果失败，W9不能启动；TempFlow T5同样因独立评分和像素护栏失败继续锁定。
2. **真实训练与效果**：P3 已找到 `480x832/5帧/4-step` 的单卡候选；仍需补 matched 多 prompt/seed、OOM bracket 与真实 reward 资源布局，之后才能重新预注册reward/超参数/数据方案。不能在当前失败recipe上直接增加optimizer steps。
3. **数据与 checkpoint**：D3 在真实 Wan 上验证同内容换路径；D5/P5 需先实现多卡训练后再验证 shard/cursor；C2/S2 的八类身份不兼容拒绝已经完成。
4. **人工与独立质量**：Q2 等待至少2名真实评审；任何训练效果 claim 必须同时参考独立评分、人工面板和 reward-hacking 护栏。
5. **故障与长稳**：O4 完成真实 reward 服务中途退出、不重复更新/记账门槛；O6 需定义不宣称效果的稳定性 workload 后检查显存、主存和句柄泄漏。
6. **性能与资源**：P1 主要阶段已有真实Wan数据；仍需P2 native/infra overhead、P3 单卡32GB可行域。P5 真双卡正确性通过后才做P6扩展效率。
7. **环境与交付**：S1 干净环境重建、S4 冻结材料端到端复跑，最后完成S5验收报告；S2 runtime mismatch拒绝已完成。
8. **已确认但暂不补的产品边界**：A6 的通用插件自动发现、O2 自动降级、D5/P5 的 DDP 能力均未实现；如纳入“成熟产品”范围，必须先实现再执行相应实验，不能以文档豁免冒充通过。

## 重点实验详细门槛

### R1：独立 sampler 数值对齐（最高优先级）

在 E3a 共享 rollout 数学对齐之后，让 native TempFlow 和 infra 各自执行采样，但注入同一个 initial latent 和每一步显式 SDE noise。比较 ODE prefix、六个 branch target、timestep、old log-prob 和 media hash。

通过标准：所有离散结构完全一致；浮点 tensor 在预注册容差内；探针本身不改变模型参数。这个实验回答“公式一样以后，数据是怎样产生的也一样吗”。

### R2：Wan PEFT checkpoint/resume 等价

World-R1 和 Flash 一步 smoke 通过后，分别执行连续两步与一步保存 + 新进程 resume 到第二步。检查 LoRA、optimizer、RNG、prompt/seed 顺序、metrics、manifest 和 PEFT-only artifact。

通过标准：deterministic 模式要求严格一致；默认 performance 模式使用预注册容差和重复运行。还要验证错误 base model/revision、错误 LoRA target 和损坏 adapter 会在训练前拒绝。

### R3：真实 reward 协议与失败处理

W6 已完成真实 `reward_3d` / `reward_general` reference parity；`2adfbfd` 又重跑了 loopback legacy `reward_general`，三方分数逐位一致、坏请求 HTTP 500 且无 silent fallback。尚未完成的是 O4 的服务中途退出/重试 exactly-once 门槛，以及当前提交上的 HPS-backed Wan 训练 preflight。

剩余通过标准：服务在训练中途退出或超时时 fail closed；重试不重复记账或重复更新；HPS-backed bounded run 的 scorer identity、media/reward、训练状态和 artifact 链闭合。协议 parity 本身不构成训练效果声明。

### R4：Flash-GRPO 原生 sampler 对齐和效率

将当前“完整 trajectory 后筛 timestep”的兼容路径，与 Flash-GRPO native selected-step sampler 在同一 Wan、latent、noise、timestep 下比较 log-prob、loss 和 gradient；随后测峰值显存与每 step 时间。

通过标准：数值在预注册容差内，且只有在显存/速度实测改善后才能声称 Flash 效率能力。

### R5：有对照的真实训练有效性

仅在一步 correctness、resume 和 reward parity 全部通过后，分别为 TempFlow、World-R1、Flash 设计至少 3 个 seed 的 active/zero-LR control。先跑 10-step pilot；只有 active-control 的预注册 CI、分组指标和安全护栏全部通过，才扩到 20/50/100 steps。

通过标准：多数 seed 同方向、active-control CI 下界大于 0、关键 prompt 类别不退化、参数确实更新、control 精确不更新。否则结论是“闭环可运行但效果证据不足”，不能靠延长训练掩盖。

### R6：独立质量评分与人工盲评

训练 reward 之外固定一个独立语义/质量评分器，并保存固定 before/after 样本面板做盲评。视频还要检查运动、时间一致性、闪烁和崩坏，而不只看 reward。

通过标准：训练 reward、独立评分和人工判断方向一致；若背离，按 reward hacking 风险处理。

### R7：数据与 artifact 可靠性

用真实 Hugging Face 数据子集验证内容 hash、split、prompt 预处理版本、去重、坏样本隔离、数据移动后 resume、数据内容被替换时拒绝，以及 metrics/manifest/checkpoint 的交叉一致性。

通过标准：同内容换路径允许，内容或语义变化拒绝；中断后没有重复/漏样本；每个结果能追溯到固定数据 revision 和样本 ID。

### R8：资源、故障恢复与扩展

记录单卡的 rollout/reward/backward/checkpoint wall time、峰值显存、GPU 利用率和 samples/s；注入 OOM、进程中断、磁盘不足和 reward 服务失败。真正的双卡实验要使用 DDP/accelerate 等同一训练任务的并行机制，GPU2/GPU3 各跑一个独立 run 不算双卡扩展。

通过标准：资源指标可重复，失败不会留下被误判为 valid 的 summary/checkpoint，resume 后不重复更新；双卡结果在容差内保持训练语义并展示可量化收益。

## 验收顺序

1. `2adfbfd` 的本地统一回归、Wan W5/W7b correctness 与 `reward_general` 协议 parity 已完成；失败 attempt 保留，不再重复旧的 snapshot/load/smoke 链。
2. 先补当前提交的真实 SD3 deterministic/resume gate；失败则停在 SD3 层定位，不进入扩容。
3. 再做 HPS-backed Wan bounded post-merge preflight，核对 scorer/media/reward/训练/artifact 身份；只作为机械门槛，不从单次 bounded run 推导质量。
4. 并行推进 O4 服务中断 exactly-once、P2 matched native/infra overhead 与 P3 单卡可行域；没有 P2 实测不得声称速度提升。
5. W8/Q3 与 TempFlow 既有效果门槛仍失败；新质量实验必须重新预注册多 seed、control、独立 scorer 与人工面板，不能直接延长旧 recipe。
6. 最后完成 S1/S4 的干净环境和冻结材料端到端复跑，再形成 S5 最终验收报告。

## 文档维护规则

- 每次修改源码、配置、实验门槛或结论边界时，在对应表格追加或更新一行。
- 每个实验完成后记录 recipe/source/data/model hash、运行目录、硬门槛、失败样本和结论边界。
- 失败 run 不覆盖、不删除；后续修复使用新 attempt 目录。
- checkpoint 等大文件不复制进 Git 证据目录，只保存必要的 hash、metadata、metrics、summary 和日志。
- “能跑”“数学正确”“可恢复”“训练有效”“更高效”是五种不同结论，必须由不同实验分别支持。

## 最近一次统一校验

- `2adfbfd` 完整 non-distributed suite：883 passed、2 skipped、5 deselected（2026-07-15）。
- `2adfbfd` 真实双进程 Gloo distributed suite：5 passed、885 deselected；覆盖跨 rank 更新/回滚、Flash 全局 coefficient oracle 和 microbatch `no_sync()` 通信次数 oracle。
- `ruff check visual_rl tests scripts train.py`、compileall 与 `git diff --check`：通过。
- 合并审计的 artifact trust/order、marker tree digest、transaction recovery、async reward cancellation、DDP snapshot budget、Flash coefficient、formal occurrence grouping、direct config 重校验与 DDP microbatch 通信已完成本地修复和回归；真实 GPU Wan W5/W7b 与 `reward_general` 协议已复验，NCCL、SD3 与 HPS-backed Wan 训练 post-merge 仍待复验。
- `2adfbfd` W7b 严格通过：transformer BF16 输入与 SDE 原始 FP32 current/next 合同成立，media/embedding/latent/log-prob/KL/loss 和 480 个 gradient tensor 与独立 scalar reference 逐位一致。
- `2adfbfd` W5 attempt 1 在训练前被 deterministic runtime guard 拒绝，根因是 harness 提前调用 `torch.cuda.*`；失败证据保留。修正启动顺序后的 attempt 2 有六段 `valid=true`，World/Flash 两份比较各 11 个 exact gate 全通过。完整索引见 [WAN_RESULTS_2adfbfd.md](postmerge_validation_20260715/WAN_RESULTS_2adfbfd.md)。
- `2adfbfd` `reward_general` attempt 1 通过：direct、reference HTTP、infra client 三路均为 `[0.260009765625, 0.1943359375]`，两组差值为 0；坏图像/坏 pickle 均返回 HTTP 500，`silent_fallback_detected=false`，服务仅绑定 loopback legacy 协议。
- 上述 post-merge 结果只支持 correctness/resume/reward 协议 parity；未运行 HPS-backed Wan active/control、未比较 native/infra 吞吐，也未产生新的质量证据，因此质量与速度 claim 继续锁定。
- TempFlow E3a 与效果实验的非 checkpoint 证据已经下载到各自实验目录。
- TempFlow E3b v1/v2/v3 非 checkpoint 证据已下载；v3 全部门槛通过。
- TempFlow T2a 诊断已由冻结证据重建；报告 validation/package 和结构校验通过。自动浏览器禁止本地 `file://`，视觉 QA 待人工完成。
- TempFlow T2b v2 有效完成并否定 eval/train mode 假设；v1 BF16 哈希序列化失败和 v2 结果均已下载，不含模型/checkpoint。
- TempFlow T2d v1 全部门槛通过：first divergence 已定位到 single-parent 与 six-branch BF16 transformer forward 的 batch shape；非 checkpoint 证据已下载。
- TempFlow T2e v4 全部门槛通过：正式 shared-prefix 路径九次被调用，old/new log-prob 与 clipfrac 精确归零、参数不变；v1-v3 staging 失败与 v4 非 checkpoint 证据均已下载。
- TempFlow T3 六个 10-step run 机械门槛全部通过，但效果门槛全部失败；不含 checkpoint 的 logs/config/metrics/manifest/previews/aggregate/validation 已下载，20/50/100-step 保持锁定。
- W6 general attempt 5 已通过：direct/reference/infra 均为 `[0.260009765625, 0.1943359375]` 且差值为0；坏图像与坏 pickle 均返回500，不再静默返回 `0.5`；非模型证据已下载。
- W6 3D attempt 2 已通过：DA3+Qwen reference/infra 差 `1.99e-8`，三个 component、非空 MP4/PNG、cache、坏 payload 500、安全进程组停止全通过；attempt1 无 nvcc 导致的静默0分及修复链均保留。
- W7b attempt 2 已通过：真实 Wan `[2,1]` 异构时间步分组后恢复原顺序，media/embedding/latent/log-prob/loss 与480梯度逐位一致；attempt1 的 optional embedding harness失败保留。
- W8 1-step preflight 已通过：World/Flash 各一组 active/control 均从相同 LoRA、相同 rollout/media/HPS 分数开始；active delta L2 分别 `0.195539/0.239434`，control 精确为0；同卡 HPS+Wan 峰值25,813 MiB，服务安全停止且GPU2回到0 MiB。无 checkpoint 证据已下载。
- W8 3-seed 10-step attempt3 已完成：12/12 run与6/6配对机械有效，World/Flash像素护栏通过；World配对均值`-0.000418`、CI95 `[-0.002747,+0.002014]`，Flash均值`-0.000489`、CI95 `[-0.002356,+0.001249]`，两者效果门槛失败。GPU2峰值23,616MiB并归零；120个媒体及全部非checkpoint证据已下载，W9锁定、Q3转为失败诊断。
- Q3 attempt2 已完成：240视频/1,200帧的固定PickScore两遍与step0配对精确一致；World/Flash视频安全门槛通过、无塌缩，但独立PickScore paired mean仅`+0.000205/+0.000047`且CI95均跨0。6张contact sheet显示64px/2-step复杂语义通常不可辨；W9继续锁定，P3成为下一关键项。attempt1跨文本batch产生`2.67e-5`身份漂移的失败证据保留。
