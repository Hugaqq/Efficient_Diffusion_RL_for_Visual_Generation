# VisualRL v0.8

VisualRL 是面向图像与视频 Diffusion RL 的模块化训练基础设施。v0.8 Phase A 对外只暴露
`ModelAdapter` 与粗粒度 `AlgorithmModule` 两个可替换轴；trainer、dynamics、rollout、
reward、conditioner 和 credit 继续作为算法模块的内部装配资产。仓库只保留一条训练执行路径：

```text
schema-v2 YAML
→ recipe compile / compatibility preflight
→ default RunController composition root
→ bind ModelCapabilities × AlgorithmRequirements
→ PolicyRuntimePort → BoundAlgorithm
→ internal six-stage iteration
→ atomic checkpoint and terminal artifacts
```

模型代码不读取 algorithm/recipe id；算法模块不导入 SD3、Wan 或 Diffusers pipeline class，
并且只通过 `models.interface` 与 `models.scheduler` 两个窄 ABI 访问模型域。
composition root 在加载权重、初始化 CUDA 或构造 reward worker 前独立解析两个 registry 并完成
capability 求交。运行时 rollout 与 policy recompute 只持有 `PolicyRuntimePort`，不会拿到 raw
Adapter。这里的“可组合”表示合同兼容，不表示任意模型和任意算法已经拥有真实训练证据。
新增模型、算法或组合时的职责表、连接链和永久门禁见
[Model–Algorithm boundary](docs/MODEL_ALGORITHM_BOUNDARY.md)。

唯一训练入口是 `python -m visual_rl.train CONFIG`。包不提供 console script、
`load().run()` Python API、旧 Runner 或第二套 runtime factory。

## 安装

从仓库根目录安装训练依赖：

```bash
python -m pip install -e '.[train]'
```

模型权重、数据集和 reward artifact 不包含在 wheel 中，运行前必须按配置中的
`launch.artifacts` 提供本地文件。Wan 视频配置使用的远端 reward origin 还需要按
[companion service guide](services/world_r1_strict/README.md) 单独启动。
当前 v0.8 配置严格锁定 `world-r1-8e46b1b63498`。仓库中对应的 general/3D
reward artifact 来自 frozen-wheel 服务的 checkpoint、strict health attestation 和真实有限值
score receipt，不是手写占位。最终 general/3D marker SHA-256 分别为 `648cebf…` 和
`fbceb863…`，并与 code `56507f6e…`、wheel `6f1533ef…` 一起进入同一 freeze record
`a6c961fc…`。这些 identity 只证明本次冻结的 artifact/service 绑定；空目录、手写占位或
另一次部署仍不构成可用 artifact，也不能继承本次 A7 结果。

## schema-v2 配置

| 配置 | 组合 | 定位 |
| --- | --- | --- |
| [flow_grpo_sd3.yaml](configs/v2/flow_grpo_sd3.yaml) | Flow-GRPO + SD3.5 | full-trajectory 图像路径 |
| [flow_grpo_wan.yaml](configs/v2/flow_grpo_wan.yaml) | Flow-GRPO + Wan2.1 T2V | 复用相同算法模块和 Wan adapter 的 full-trajectory 视频路径；`beta=0` |
| [tempflow_sd3.yaml](configs/v2/tempflow_sd3.yaml) | TempFlow-GRPO + SD3.5 | branching rollout 图像路径 |
| [flash_wan.yaml](configs/v2/flash_wan.yaml) | Flash-GRPO + Wan2.1 T2V | single-step 视频路径 |
| [world_r1_core_wan.yaml](configs/v2/world_r1_core_wan.yaml) | World-R1 core + Wan2.1 T2V | camera conditioner + general/3D reward |
| [world_r1_release_surrogate_wan.yaml](configs/v2/world_r1_release_surrogate_wan.yaml) | World-R1 release-surrogate + Wan2.1 T2V | main/dynamic phase schedule；不代表 exact-environment likelihood |

生产配置的顶层结构固定为：

```yaml
schema_version: 2
recipe: flow_grpo_v1
overrides:
  training:
    max_optimizer_steps: 20
    gradient_accumulation_steps: 1
  execution:
    distribution_mode: single
launch:
  output_dir: ../../runs/v2/flow-grpo-sd3
  resume_from: null
  checkpoint_every_optimizer_steps: 10
  artifacts:
    model: ../../checkpoints/stable-diffusion-3.5-medium
    datasets:
      main: ../../data/prompts/geneval_rgb_train_36.txt
    rewards:
      reward_quality: ../../reward_artifacts/prompt-color-guarded-v1
```

路径相对于 YAML 所在目录解析。建议复制一份配置，再修改 artifact、输出目录和
允许覆盖的 training/execution 字段；不要把旧 schema-v1 字段混入 v0.8 recipe。

显存受限时只调整 policy replay 的执行几何，不要降低算法的 K/T：

```text
overrides:
  training:
    policy_recompute:
      row_microbatch_size: null  # null 表示保持 rollout 的完整 forward batch geometry
      transition_window_size: 1  # 每次只保留一个 timestep 的 autograd graph
```

不要把 `row_microbatch_size` 直接改成 4/2/1 当作通用 OOM 降级方案。BF16/FP16 下，
改变 forward batch geometry 可能使 rollout 与 policy replay 的 log-prob 出现数值漂移；
只有具体 model/algorithm recipe 通过首轮 parity gate 后，才能显式启用更小的 row microbatch。

完整 `[B,T]` `new_log_probs` 会作为 detached diagnostics 保存；带梯度的值只在当前
slot 内完成 loss/backward，不能跨 timestep 缓存，否则会重新引入 K×T 计算图 OOM。

## 运行与恢复

完成 artifact 和 reward service 准备后，直接运行对应配置：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m visual_rl.train configs/v2/flow_grpo_sd3.yaml
python -m visual_rl.train configs/v2/flow_grpo_wan.yaml
python -m visual_rl.train configs/v2/tempflow_sd3.yaml
python -m visual_rl.train configs/v2/flash_wan.yaml
python -m visual_rl.train configs/v2/world_r1_core_wan.yaml
python -m visual_rl.train configs/v2/world_r1_release_surrogate_wan.yaml
```

每次只运行所需的一条 recipe。32 GiB CUDA 验证使用
`expandable_segments:True`，避免 rollout 与逐 slot replay 的不同 allocation shape 留下大块
不可复用缓存；该环境值应与 GPU、Torch/CUDA 版本一起写入实验记录。它不能修复仍然存活的
autograd graph，因此不能替代 `transition_window_size: 1`。

成功时 stdout 是包含 `run_id`、`output_dir`、`committed_steps` 和
`authoritative_checkpoint` 的单行 JSON。配置错误使用退出码 2，运行失败使用退出码
1，用户中断使用退出码 130。

恢复训练需要新建一份语义相同的 schema-v2 YAML，使 `launch.output_dir` 保持不变，
并把 `launch.resume_from` 指向一个完整 safe-point checkpoint，例如
`/absolute/run/checkpoints/step-10`。恢复来源是 checkpoint 目录，不是 run 根目录。

## 只读检查

终止成功的 v0.8 run 使用独立 inspection 模块检查：

```python
from visual_rl.artifacts.inspection import audit_run, inspect_run

status = inspect_run("/absolute/run")
audit = audit_run("/absolute/run")
if not status.ok or not audit.ok:
    raise RuntimeError((status.errors, audit.errors))
```

`inspect_run()` 快速核对 `SUCCESS`、`latest.json` 和 authoritative checkpoint；
`audit_run()` 进一步验证 checkpoint state tree、resolved recipe、run manifest 和 metrics
摘要。它们只理解 v0.8 terminal layout，不把未完成 safe point 或旧 commit-chain run
误判为成功。

一个终止 run 的主要产物为：

```text
recipe.resolved.json
launch.resolved.json
SUCCESS
resolved_recipe.json
run_manifest.json
metrics.jsonl
checkpoints/latest.json
checkpoints/step-N/
```

前两个文件在重模型构造前后分别写入：`recipe.resolved.json` 固化 materialized recipe，
`launch.resolved.json` 固化 RuntimeBind facts、artifact locations 和脱敏 reward runtime audit。
`resolved_recipe.json` 则是 terminal checkpoint/finalizer 保留的历史终态快照；两者不是同一文件的
重复别名。fresh retry 必须逐字节一致，resume 只允许 source locator/`resume_from` 的声明性差异。

## 当前支持边界

当前实现只承诺 single-process、单设备、`gradient_accumulation_steps: 1` 的装配路径。
它不承诺 DDP、多节点、FSDP、DeepSpeed、异步 reward 或任意模型与任意算法自由组合。

自动化测试集中覆盖 strict schema-v2 compile、六配置默认 composition、fake one-update、
safe-point resume，以及 TempFlow/Flash/World-R1 的关键合同。
其中 fake SD3 leaf 只用于验证默认装配和更新生命周期。
它不等价于真实 SD3.5/Wan 权重的 one-update/native parity，也不构成训练质量、吞吐或多 GPU 支持声明。
六份配置是正式接口示例，不证明你本地的 artifact、GPU 或 reward service 已就绪。

## 真实 GPU 验证状态

冻结候选正在 `10.130.140.73` 上执行六条独立的 20-optimizer-commit route。当前已有四条
正式通过同一 freeze 下的 exit-zero、`SUCCESS`、step-20 checkpoint、正梯度、失败日志扫描和
15 秒显存序列验收：

| Route | 状态 | Driver memory peak |
| --- | --- | ---: |
| Flow-GRPO × SD3.5 | 20/20 accepted | 24,892 MiB |
| TempFlow-GRPO × SD3.5 | 20/20 accepted | 25,148 MiB |
| Flow-GRPO × Wan2.1 | 20/20 accepted | 17,568 MiB |
| Flash-GRPO × Wan2.1 | 20/20 accepted | 22,555 MiB |
| World-R1 core × Wan2.1 | running；step-10 checkpoint 已提交 | pending |
| World-R1 release-surrogate × Wan2.1 | 首次 run 收到 SIGTERM；同 freeze fresh retry 已排队 | pending |

因此当前只能声明前四条 route 的真实单卡 20-step 可运行性，不能提前宣称六路 A7 已完成。
World-R1 的失败 run 会作为拒绝证据保留，不由后续 retry 覆盖。逐路 acceptance、恢复队列和
资源检查见 [v0.8 real-GPU evidence](experiments/v08_modular_gpu_20260808/README.md)。

第三方来源与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
