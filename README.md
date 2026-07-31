# VisualRL v0.7

VisualRL 是面向图像与视频 diffusion reinforcement learning 的研究基础设施。
v0.7 只保留一条公开训练路径：

```text
一份完整 YAML
→ visual_rl.load()
→ resolve()
→ validate()
→ run()
→ inspect_run() / audit_run()
```

模型、rollout、reward、算法、训练步数、分布式模式、artifact 和恢复来源都由
这份 YAML 决定。包不提供训练 CLI、preset 合并、外部插件注册或第二个
Runner。

## 当前证据状态

源码、固定实验配置、离线验证工具、全仓自动化、Tiny single/Gloo API smoke
和基础 wheel 构建/隔离安装已经在本地验证。远端 32 GB RTX 5090 上的 BF16
operational C20 已完成 Flow-GRPO、TempFlow-GRPO 和 Flash-GRPO 的
continuous/interrupted-resume、public audit 与 semantic parity；World-R1
C20 正在运行。精确 digest、显存峰值、artifact 位置和证据边界见
[operational C20 evidence](docs/V0_7_OPERATIONAL_EVIDENCE.md)。

这些运行来自 dirty engineering candidate，不替代仍为 `not_run` 的 Flow
FP32 native parity、Q100、多 seed reward-improvement、MG1/NCCL 和最终
clean-commit release envelope。`not_run` 不是通过、跳过即成功或质量提升
声明。

当前文档：

- [v0.7 用户指南](docs/V0_7_USER_GUIDE.md)
- [v0.7 验收矩阵](docs/V0_7_ACCEPTANCE.md)
- [v0.7 operational C20 evidence](docs/V0_7_OPERATIONAL_EVIDENCE.md)
- [项目边界](docs/V0_7_SCOPE.md)
- [项目概览](docs/PROJECT_OVERVIEW.md)
- [固定实验计划](experiments/EXPERIMENT_PLAN.md)
- [W06 固定实验套件](experiments/v0_7/README.md)
- [更新记录](CHANGELOG.md)

## 完整 Tiny YAML

下面是一份可独立解析的完整单进程 CPU 配置。仓库中的权威测试副本是
[tests/fixtures/configs/tiny_grpo.yaml](tests/fixtures/configs/tiny_grpo.yaml)。

```yaml
schema_version: 1

run:
  seed: 42

model:
  name: tiny_diffusion
  adapter_checkpoint: null
  params:
    image_size: 16

dataset:
  path: null
  prompts: ["a red cube", "a blue cube"]
  split: train
  repeat_per_prompt: 1
  require_unique: true
  sampling_strategy: sequential
  sampling_seed: 42
  empty_prompt_policy: error

rollout:
  name: full_trajectory
  params:
    samples_per_prompt: 2
    num_steps: 2

reward:
  components:
    - name: mock
      weight: 1.0
      params:
        mode: prompt_media
  execution:
    microbatch_size: null
    max_retries: 0
  cache_dir: null

algorithm:
  name: grpo
  params:
    clip_range: 0.001
    adv_clip_max: 5.0
    beta: 0.0
  advantage:
    epsilon: 1.0e-6

optimizer:
  learning_rate: 1.0e-4
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_weight_decay: 1.0e-4
  adam_epsilon: 1.0e-8
  max_grad_norm: null
  max_initial_logprob_delta: null
  require_initial_clipfrac_zero: false
  require_finite_gradients: true
  require_nonzero_gradients: false

runtime:
  max_steps: 1
  batch_size: 2
  precision: fp32
  update_microbatch_size: 2
  deterministic: true
  progress: false
  distributed:
    mode: single
    device: cpu
    timeout_s: 30.0
    max_snapshot_tensor_bytes: null

artifacts:
  output_dir: runs/tiny-grpo
  checkpoint_every: 1
  checkpoint_keep_last: 2
  preview_samples_per_event: 0

resume:
  from: null
```

真实 SD3/Wan 的完整基线位于 [configs](configs/)；固定 C20/Q100/MG1 配置位于
[experiments/v0_7/configs](experiments/v0_7/configs/)。

## 唯一 Python 入口

用户创建自己的 `run_experiment.py`。脚本固定一个完整 YAML 路径，不接收命令行
覆盖，也不在 Python 中重新拼装训练语义：

```python
from pathlib import Path

import visual_rl as vr

config_path = Path("/absolute/path/to/complete-config.yaml")
experiment = vr.load(config_path)
experiment.resolve()
report = experiment.validate()
if not report.ok:
    raise RuntimeError(report)

result = experiment.run()
status = vr.inspect_run(result.output_dir)
audit = vr.audit_run(result.output_dir)
if not status.ok or not audit.ok:
    raise RuntimeError("authoritative artifact validation failed")
```

`run(callbacks=[...])` additionally accepts constructed, read-only Callback
observers for lifecycle metrics and authoritative commit paths. They do not
enter YAML, checkpoint identity, or the training data path; see the
[v0.7 user guide](docs/V0_7_USER_GUIDE.md#read-only-callbacks).

单进程启动：

```bash
python run_experiment.py
```

双 rank DDP 使用同一个脚本和一份将
`runtime.distributed.mode/device` 配置为 `ddp/cpu` 或 `ddp/cuda` 的完整
YAML：

```bash
torchrun --standalone --nproc-per-node=2 run_experiment.py
```

恢复训练不增加第二入口：创建另一份完整 YAML，使 `resume.from` 与
`artifacts.output_dir` 指向同一个已有 run directory，再运行同一用户脚本。

## 代码主线

```text
VisualRLConfig
→ PromptDataset
→ ModelAdapter
→ RolloutEngine / RolloutBatch
→ RewardExecutor / RewardBatch
→ AdvantageResult / PolicyLossInputs
→ PolicyObjective / UpdateEngine
→ ExperimentRunner._execute_step()
→ CommitCoordinator / authoritative commit marker
```

单卡与 DDP 共用这条 step lifecycle。算法只准备 typed loss inputs；GRPO、
Flash-GRPO 和 TempFlow-GRPO 的 ratio、clip、policy loss、approximate KL 与
clip fraction 只在一个公共 objective 内计算。

## 本地源码验证

本地验证只证明对应测试合同，不替代真实 GPU、NCCL 或质量实验：

```bash
conda run -n visual-rl python -m pytest -q tests/test_experiment_api.py
conda run -n visual-rl python -m pytest -q tests/test_documentation_contract.py
conda run -n visual-rl python -m ruff check \
  --select E4,E7,E9,F visual_rl tests
```

确定性边界见
[docs/DETERMINISTIC_RUNTIME.md](docs/DETERMINISTIC_RUNTIME.md)。World-R1
companion service 与训练包使用同一个 wheel；独立服务环境的部署步骤见
[services/world_r1_strict/README.md](services/world_r1_strict/README.md)。
