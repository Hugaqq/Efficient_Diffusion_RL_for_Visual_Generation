# VisualRL v0.7 正式实验计划

更新日期：2026-07-29

本文是当前实验执行顺序、输入和晋级规则的正式入口。W06 已准备配置与
controller，但没有执行真实训练。本计划中的 C20、Q100、Flow native、
MG1/NCCL、远端执行与上传当前全部为 `not_run`。

用户用法见 [v0.7 用户指南](../docs/V0_7_USER_GUIDE.md)，严格状态矩阵见
[v0.7 验收文档](../docs/V0_7_ACCEPTANCE.md)，固定 suite 的文件级说明见
[experiments/v0_7/README.md](v0_7/README.md)。

## 唯一执行合同

每个 role 都是一份完整 YAML，训练只通过 public Python API：

```python
from pathlib import Path

import visual_rl as vr

config_path = Path("/absolute/path/to/fixed-role.yaml")
experiment = vr.load(config_path)
experiment.resolve()
report = experiment.validate()
if not report.ok:
    raise RuntimeError(report)

result = experiment.run()
status = vr.inspect_run(result.output_dir)
audit = vr.audit_run(result.output_dir)
if not status.ok or not audit.ok:
    raise RuntimeError("run evidence is not authoritative")
```

单进程 role 使用用户脚本：

```bash
python run_experiment.py
```

DDP role 使用同一脚本；完整 YAML 决定 `ddp` 与设备：

```bash
torchrun --standalone --nproc-per-node=2 run_experiment.py
```

脚本不接收训练语义参数。中断与恢复使用两份预先冻结的完整 YAML；恢复配置通过
`resume.from` 指向同一个 run directory。

## 固定输入

W06 唯一 role 表和依赖关系由
[interrupt_resume.py](v0_7/interrupt_resume.py) 维护。完整配置位于
[v0_7/configs](v0_7/configs/)，包含：

- Tiny S100：continuous、interrupted、resume；
- Flow-GRPO/SD3：C20 三角色与 Q100 seeds 17/29/43；
- TempFlow-GRPO/SD3：C20 三角色与 Q100 seeds 17/29/43；
- Flash-GRPO/Wan：C20 三角色与 Q100 seeds 17/29/43；
- World-R1/Wan：C20 三角色与 Q100 seeds 17/29/43；
- MG1 Tiny DDP：continuous、interrupted、resume。

Q100 的固定 prompt/evidence 输入见
[q100_inputs.json](v0_7/evidence/q100_inputs.json)。实验控制器不得动态生成
新的算法参数、改变 seed、缩短 gate 或绕过 public status/audit。

## 阶段 A：环境与 source readiness

在任何真实训练前记录：

- candidate commit 和 clean worktree；
- Python、Torch、CUDA、Diffusers、Transformers、PEFT 版本；
- GPU 型号、数量、显存和 NCCL availability；
- SD3/Wan checkpoint 与 reference repo 的精确路径；
- World-R1 strict service health/revision；
- 输出目录可写且每个 role 使用新的目录。

缺任一依赖时 role 保持 `not_run`。不得自动下载未知 revision、切换模型、使用
fake service 或把环境失败计为训练失败。

## 阶段 B：C20 correctness

每个算法独立执行：

```text
continuous 20 step
+ interrupted 10 step
+ fresh resume to 20
```

验收：

- 所有进程 exit code 为 0；
- public status 和 audit 通过；
- committed step 精确为 20；
- continuous 与 fresh-resume 的 adapter、optimizer、RNG、global step、
  run-id normalized manifest 和全部 core metrics 等价；
- failed/interrupted 未提交 step 不出现在 marker chain；
- reward、update 或 artifact 失败时 head 不错误前进。

Flow-GRPO C20 通过后还必须执行 W04 的 14-item CUDA native parity。任何一个
item 失败都阻断 Flow Q100。

## 阶段 C：Q100 reward evidence

只有对应算法的 C20（以及 Flow native）通过后，才运行 seeds 17、29、43 的
100 committed steps。Q100 每个 source 必须：

- exactly 100 audited committed steps；
- prompt、seed、algorithm 和 model family 与冻结 role 一致；
- manifest 中每个 sample 都有 finite `weighted_total`；
- 无缺步、重复 sample identity 或跨 source 污染；
- candidate commit 与 environment report 一致。

只在 evidence completeness 通过后计算 reward verdict：

- early window：artifact steps `0..35`；
- late window：artifact steps `64..99`；
- 先在 prompt 内平均，再等权 seed 与 prompt；
- `pooled_delta > max(0, 0.1 * pooled_early_std)`；
- 三个 seed 至少两个 delta 为正，且 median delta 为正；
- 100 点 Theil-Sen slope 为正，pair count 精确为 4,950。

`evidence_complete` 与 `reward_pass` 分开报告。reward 失败不能抹去已经通过的
C20 correctness，也不能改写为质量提升。

## 阶段 D：MG1/NCCL

MG1 只在真实双 GPU、单机 NCCL 环境执行：

1. 三个冻结的内部 NCCL node；
2. Tiny C20 continuous；
3. Tiny interrupted + fresh resume；
4. rank failure、rank-zero artifact 和 continuous/resume parity。

CPU/Gloo 测试不能替代 MG1。缺双 GPU 或 NCCL 时保持 `not_run`。

## 阶段 E：wheel 与远端执行

最终 candidate wheel 的 build、RECORD/content checker、干净环境安装、
`pip check` 和非仓库目录 public API smoke 属于 W07。W06 source readiness
不等于 wheel 验收。

远端执行、模型下载和 evidence 上传必须由用户明确授权。上传前只收集
curated machine-readable evidence，不上传 checkpoint、credential、个人路径或
未审查日志。

## 当前状态

| Gate | 状态 |
|---|---|
| W06 source/config/controller | `verified locally` |
| Tiny real S100 | `not_run` |
| Flow-GRPO C20 | `not_run` |
| Flow native CUDA parity | `not_run` |
| TempFlow-GRPO C20 | `not_run` |
| Flash-GRPO C20 | `not_run` |
| World-R1 C20 | `not_run` |
| 四算法 Q100 | `not_run` |
| MG1/NCCL | `not_run` |
| final wheel build/base install/outside import | `verified locally` |
| remote execution/upload | `not_run` |

当前状态不得引用 v0.6 历史报告晋级。机器可读最终验证由
[verify_evidence.py](v0_7/verify_evidence.py) 执行；缺 evidence 时必须失败，
不能生成占位通过结果。
