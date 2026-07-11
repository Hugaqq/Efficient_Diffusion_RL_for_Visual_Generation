# VisualRL 实验计划

本文档用于规划从 smoke test 走向小规模真实实验的路线。目标不是追求大规模 SOTA，而是验证当前 `visual_rl` infra 是否能在同一套训练主线下支持不同 rollout、reward、algorithm 和 artifact。

## 实验目标

当前主线是 `ExperimentRunner`：

```text
prompts -> rollout -> rewards -> advantages -> logprobs -> loss -> optimizer
```

第一阶段实验应证明：

- 同一套 trainer 可以切换不同 rollout 方式。
- 同一套 trainer 可以切换不同 GRPO 变体。
- reward、advantage、logprob recomputation、loss、metrics 和 checkpoint 能形成闭环。
- 实验输入和输出可以复现、比较和复查。

## 实验分层

建议按三层推进：

```text
Experiment A: Controlled Tiny RL
Experiment B: Real Image RL Bridge
Experiment C: Video / World-R1 Feasibility
```

三层分别回答：

- A：当前 infra 是否能稳定完成真实训练闭环。
- B：当前 infra 是否能接真实 image diffusion model。
- C：当前 infra 是否能走向 video rollout 和 World-R1 reward。

## Experiment A: Controlled Tiny RL

### 目标

用轻量可控任务证明训练主线可以稳定学习，并比较不同 rollout / algorithm。

```text
tiny_diffusion + prompt_color reward
```

比较对象：

- `grpo + full_trajectory`
- `flash_grpo + single_step`
- `tempflow_grpo + branching`

### 数据集

创建 prompt 文件：

```text
data/prompts/color_control_60.txt
```

格式为每行一个 prompt。当前 `PromptDataset` 支持 inline prompts 或 txt 文件，不需要复杂 dataset loader。

示例：

```text
a red square
a blue square
a green square
a red object on a plain background
a blue object on a plain background
a green object on a plain background
```

注意：`prompt_color` reward 只识别 `red`、`green`、`blue`，因此这个实验只测试 infra 和 RL 链路，不测试复杂语义理解。

### 配置建议

公共配置方向：

```yaml
model:
  name: tiny_diffusion
  model_family: image
  extra:
    image_size: 16

dataset:
  path: data/prompts/color_control_60.txt
  repeat_per_prompt: 1

sample:
  batch_size: 4
  num_steps: 4
  samples_per_prompt: 4

rewards:
  weights:
    prompt_color: 1.0
  clients:
    prompt_color:
      name: prompt_color
  fail_policy: raise

train:
  learning_rate: 0.05
  max_steps: 100
  save_every: 25
```

算法配置：

```yaml
# A1
sample:
  name: full_trajectory
algorithm:
  name: grpo
```

```yaml
# A2
sample:
  name: single_step
rollout:
  selected_step_strategy: iso_temporal
  timestep_range: [0, 3]
algorithm:
  name: flash_grpo
```

```yaml
# A3
sample:
  name: branching
rollout:
  branch_count: 3
  exploration_k: 3
  include_main: false
algorithm:
  name: tempflow_grpo
```

### 实验矩阵

```text
3 algorithms x 3 seeds
```

建议 seeds：

```text
7, 11, 23
```

### 观察指标

必须记录：

- `reward_mean`
- `reward_std`
- `loss`
- `approx_kl`
- `clipfrac`
- `old_logprob_mean`
- `new_logprob_mean`
- `logprob_delta_mean`
- `logprob_delta_abs_max`

成功标准：

- `reward_mean` 整体上升。
- `loss` 无 NaN。
- `approx_kl` 和 `clipfrac` 不异常爆炸。
- `old/new logprob` 存在非零变化。
- `metrics.jsonl`、rollout cache、checkpoint 正常写出。

### 资源和时间

```text
设备：CPU 即可，GPU 更快
时间：约 10-30 分钟完成 3 algorithms x 3 seeds
磁盘：< 1GB
```

## Experiment B: Real Image RL Bridge

### 目标

验证真实 image diffusion model 下，当前 infra 可以完成：

```text
SD3 adapter -> rollout -> reward -> TempFlow-GRPO loss -> LoRA checkpoint
```

当前推荐使用 SD3/TempFlow adapter，不建议同时恢复 FLUX、QwenImage、SD1.5。

### 数据集

建议准备两个 prompt subset：

```text
data/prompts/parti_100.txt
data/prompts/geneval_color_80.txt
```

来源：

- PartiPrompts：适合做通用 text-to-image prompt subset。
- GenEval：适合抽取 color、object、count、position 等结构化 prompts。

当前 `PromptDataset` 最终只需要 txt 文件，因此数据获取流程应收敛为：

```text
download or clone source
-> extract prompt text
-> save one prompt per line
```

### 实验设置

建议分两版：

```text
B1: SD3 + GenEval color subset + prompt_color
B2: SD3 + PartiPrompts subset + remote reward
```

B1 目标是验证真实模型训练链路。B2 目标是验证 reward router 对真实 reward provider 的扩展能力。

### 配置建议

基于：

```text
visual_rl/configs/presets/sd3_tempflow_adapter.yaml
```

建议起步参数：

```yaml
model:
  name: sd3_tempflow
  model_family: sd3
  model_path: <local sd3 checkpoint path>
  extra:
    resolution: 256
    dtype: bfloat16
    lora_rank: 32
    lora_alpha: 64

dataset:
  path: data/prompts/geneval_color_80.txt

sample:
  name: branching
  batch_size: 1
  num_steps: 3

algorithm:
  name: tempflow_grpo

train:
  learning_rate: 0.00001
  max_steps: 20
  save_every: 10
```

### 实验节奏

先跑 pilot：

```text
5 steps, batch_size=1, resolution=256
```

只有当 pilot 写出合法 metrics、rollout cache 和 checkpoint 后，再扩大到：

```text
20-50 steps
50-100 prompts
3 seeds
```

### 资源和时间

```text
最低建议：1 张 24GB GPU，LoRA，256 resolution，batch_size=1
更稳配置：A100 40GB / 80GB
5-step pilot：15-40 分钟
20-50 step 小实验：2-8 GPU 小时
3 seeds x 2 configs：12-48 GPU 小时
磁盘：5-30GB，取决于 rollout cache 和 checkpoint 保存策略
```

## Experiment C: Video / World-R1 Feasibility

### 目标

验证当前数据契约能否从 image 扩展到 video：

```text
video media
+ video rollout metadata
+ reward_general / reward_3d
+ RewardRouter
+ metrics / rollout cache
```

这一步不是完整 video RL 训练结论，而是 bounded feasibility。

### 数据集

准备小型 video/world prompt set：

```text
data/prompts/vbench_world_32.txt
```

示例：

```text
orbit left around a ceramic vase on a table
push in toward a red toy car
pan right across a small kitchen
a camera moves around a wooden chair
a toy car rotating on a table
```

可参考 VBench prompt suite，但第一版应控制到 32 条左右。

### 实验设置

分两步：

```text
C1: mock_wan + mock reward
C2: bounded Wan + reward_general / reward_3d
```

C1 基于：

```text
visual_rl/configs/presets/world_r1_wan_v02_mock.yaml
```

C2 基于：

```text
visual_rl/configs/presets/world_r1_wan_bounded.yaml
```

### C2 前置条件

只有满足以下条件，才能声称 World-R1/Wan bounded path 有效：

- 真实 Wan checkpoint 可加载。
- World-R1 reference path 可解析。
- `reward_general` server 可访问。
- `reward_3d` server 可访问。
- `sample()` 输出合法 `RolloutBatch`。
- `recompute_log_probs()` shape 等于 `old_log_probs` shape。
- `ExperimentRunner` 写出 metrics、rollouts、checkpoint。

### 资源和时间

```text
C1 mock video:
  CPU/GPU 都可
  < 30 分钟

C2 bounded real Wan + World-R1:
  建议 A100 80GB 或同级高显存 GPU
  reward server 可能需要额外 GPU 或额外显存预算
  max_steps=1, frames=3-8, low resolution:
    30-90 分钟用于 probe + bounded run
  8-16 prompts offline scoring:
    2-12 GPU 小时，取决于视频分辨率、帧数和 reward server 速度
```

## 最终实验表

```text
Exp A1: tiny + grpo + full_trajectory + prompt_color
Exp A2: tiny + flash_grpo + single_step + prompt_color
Exp A3: tiny + tempflow_grpo + branching + prompt_color

Exp B1: sd3 + tempflow_grpo + branching + GenEval color subset + prompt_color
Exp B2: sd3 + tempflow_grpo + branching + Parti subset + remote reward

Exp C1: mock_wan + full_trajectory + mock reward
Exp C2: bounded Wan + reward_general/reward_3d + max_steps=1
```

## 交付物

每个实验至少保存：

- config yaml
- prompt txt
- `metrics.jsonl`
- rollout cache
- checkpoint
- reward curve
- KL / clipfrac curve
- 样本可视化对比
- 一页实验报告

## 建议时间线

```text
第 1 天：
  准备 color_control_60.txt
  跑 Experiment A 的 3 algorithms x 3 seeds
  汇总 reward / KL / clipfrac 曲线

第 2 天：
  整理 PartiPrompts / GenEval prompt subset
  写 B1/B2 configs
  跑 SD3 5-step pilot

第 3-4 天：
  跑 SD3 小规模实验
  整理 reward、KL、clipfrac 和样本图

第 5 天：
  跑 mock video C1
  准备 World-R1 reward server probe

第 6-7 天：
  如果 GPU、checkpoint、reward server 可用，跑 C2 bounded Wan run
  如果不可用，只报告 C1 和 reward-server contract readiness
```

## 结论边界

第一版实验的合理结论是：

```text
VisualRL infra 可以在统一 trainer 下支持不同 rollout、reward 和 algorithm 的小规模可复现实验。
```

不要把第一版实验表述为：

```text
已经完成完整 World-R1 video RL training
已经证明大规模 Wan training 效果
已经超越原论文结果
```

更准确的表述是：

```text
当前实验验证 infra feasibility，后续需要在真实 Wan checkpoint 和 World-R1 reward server 上完成 bounded video RL run。
```
