# Deterministic runtime

VisualRL 的 deterministic runtime 用于缩小相同输入、checkpoint 和运行环境下的
随机性来源。它不保证不同 GPU、Torch/CUDA/PEFT 组合之间逐元素一致，也不等同于
模型质量或性能结论。

## 配置

确定性开关属于唯一完整 YAML 的 `runtime.deterministic` 字段。可运行的完整示例
见 [README](../README.md#完整-tiny-yaml) 和
[Tiny fixture](../tests/fixtures/configs/tiny_grpo.yaml)。不要使用局部 YAML
overlay、命令行覆盖或第二份 runtime 配置。

确定性配置会：

- 启用项目验证过的 PyTorch deterministic algorithms；
- 固定 cuBLAS workspace、TF32 和 cuDNN 确定性选择；
- 从 YAML 的 `run.seed` 和 logical step/rank 推导训练 seed；
- 将必要的 optimizer、GradScaler 和各 rank RNG 放入 format-v5 checkpoint；
- 在恢复前验证运行拓扑和机械训练合同。

Python hash randomization发生在解释器启动前；需要这一层时，由用户在固定环境中
设置与实验记录一致的 `PYTHONHASHSEED`，但训练 seed 与训练语义仍只来自完整
YAML。用户脚本仍是公开 Python API：

```python
from pathlib import Path

import visual_rl as vr

experiment = vr.load(Path("/absolute/path/to/complete-config.yaml"))
report = experiment.validate()
if not report.ok:
    raise RuntimeError(report)
experiment.run()
```

单进程与 DDP 分别通过同一个 `run_experiment.py` 启动：

```bash
python run_experiment.py
torchrun --standalone --nproc-per-node=2 run_experiment.py
```

## Resume 与证据边界

恢复必须使用另一份完整 YAML，并让 `resume.from` 与
`artifacts.output_dir` 指向已有 run directory。恢复只信任 authoritative
commit marker 和 format-v5 checkpoint；projection 文件不是恢复身份来源。

W04 已对 Tiny/SD3 reference statistics 和数值合同完成本地测试。真实
Flow native CUDA、C20/Q100 与 MG1/NCCL 仍是 `not_run`，因此当前不能把源码级
deterministic tests 外推为真实设备上的逐元素 parity。

实验与发布状态见 [v0.7 acceptance](V0_7_ACCEPTANCE.md)。
