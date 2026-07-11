# VisualRL 每日学习笔记

这份文档根据开发 VisualRL 时实际提出的问题整理。它不是聊天记录的逐句复制，
而是把问题归纳成以后可以复习和检索的知识点。

维护约定：

- 按日期分节，最新日期放在前面。
- 每天先列主题索引，再记录核心概念、项目例子和常见错误。
- 只记录已经讨论或实际使用过的内容，不提前堆积尚未使用的知识。
- 同一天再次询问相同主题时，补充原有条目，不重复创建标题。

## 7.11 · 2026

### 当天主题

- Dataclass
- Dict / List
- Classmethod
- Python file process
- AI Infra 知识地图与学习优先级
- Plugin 注册与职责边界
- Checkpoint / Artifact 提交顺序
- Rollout group id 与 timestep 语义

### 本轮容易忽略的工程问题

#### Plugin 不只是“包一层函数”

`OptimizerPlugin` 现在同时负责创建 optimizer 和完成一次参数更新。Runner 只把
adapter 参数、batch、reward 和 step context 交给 plugin。这样新算法需要特殊 optimizer
时，不必修改公共训练循环。

公开注册函数位于 `visual_rl/plugins.py`。新增组件应通过注册函数接入，不直接修改
Runner，也不访问 registry 的内部 `_items`。

#### 一步训练的提交边界

可靠顺序是：

```text
optimizer update
-> 完整写入临时 checkpoint 目录
-> 写 manifest / reward / metrics / report
-> 最后更新 latest.json
```

`latest.json` 是“这一步已经完整提交”的指针。若程序在 artifact 写盘前失败，虽然可能
留下一个完整但未提交的 checkpoint 目录，恢复入口仍不会误用它。

从较旧 checkpoint 恢复时，必须删除 `step >= start_step` 的 manifest、metrics 和
rollout cache 记录，否则重跑同一步会得到重复 `sample_id`。

#### Prompt 文本不是可靠 group id

两个父样本可以拥有完全相同的 prompt 文本，但它们仍应是两个独立 GRPO group。
因此 advantage 优先按 `parent_prompt_index` 分组，只在没有显式 id 时才用 prompt
兜底。

#### Step index 与 timestep value

`branch_step_index` 表示第几个 transition；`branch_timestep_value` 是 scheduler 的真实
时间值。SD3 的真实 transition 数可能是 `num_steps - 1`，并且 timestep 可能是浮点数，
不能用 `range(num_steps)` 和强制 `int64` 偷换这两个概念。

### Dataclass

#### 作用

`dataclass` 适合表示“主要用于保存数据”的 class。它可以根据字段定义自动生成
`__init__`、对象显示和比较等基础方法，减少重复代码。

当前项目中的对应关系：

```text
SampleRecord   = 一条样本记录
SampleManifest = 同一次实验中的一组样本记录
```

例如，定义字段：

```python
@dataclass
class SampleRecord:
    run_id: str
    sample_id: str
    step: int
```

之后 Python 会自动提供类似下面的创建方式：

```python
record = SampleRecord(run_id="demo", sample_id="sample-0", step=0)
```

#### 必填字段和可选字段

没有默认值的字段是必填字段：

```python
prompt: str
```

有默认值的字段可以不传：

```python
seed: int | None = None
```

`int | None` 表示这个字段既可以保存整数，也可以暂时没有值。

#### `field(default_factory=...)`

List 和 Dict 是可变对象，不应直接把 `{}` 或 `[]` 作为 dataclass 的共享默认值。

```python
records: list[SampleRecord] = field(default_factory=list)
metadata: dict[str, Any] = field(default_factory=dict)
```

`default_factory` 会在每次创建对象时生成一个新的 List 或 Dict，避免多个对象意外
共用同一个容器。

#### `asdict()`

`asdict()` 会把 dataclass 递归转换成普通 Dict：

```text
SampleManifest 对象
-> asdict(manifest)
-> 可以交给 json.dump() 的普通字典
```

这就是 `SampleManifest.to_dict()` 的主要作用。

### Dict / List

#### List

List 是有顺序、可以追加内容的容器：

```python
records: list[SampleRecord] = []
records.append(record)
```

`append()` 会把一个元素添加到 List 末尾。它直接修改原 List，通常返回 `None`。

Manifest 中的数据关系是：

```text
SampleManifest.records
    ├── SampleRecord 0
    ├── SampleRecord 1
    └── SampleRecord 2
```

#### List comprehension

`from_dict()` 中使用了列表推导式：

```python
records = [
    SampleRecord(**record_data)
    for record_data in data.get("records", [])
]
```

它的含义是：依次读取 `records` 中的每一个字典，把每个字典恢复成一个
`SampleRecord`，最后组成新的 List。

#### Dict

Dict 使用 key 查找 value：

```python
metadata = {"width": 512, "height": 512}
width = metadata["width"]
```

类型：

```python
dict[str, Any]
```

表示 key 必须是字符串，value 可以是不同类型的数据。

#### `**` 字典解包

创建对象时：

```python
SampleRecord(**record_data)
```

如果：

```python
record_data = {"run_id": "demo", "sample_id": "sample-0", "step": 0}
```

它相当于：

```python
SampleRecord(run_id="demo", sample_id="sample-0", step=0)
```

合并字典时：

```python
metrics = {
    "step": step,
    **plugin_metrics,
}
```

表示先加入 `step`，再把 `plugin_metrics` 中的所有键值对展开到新字典中。这只是
Python 语法，不是二级指针。

### Classmethod

#### `self`、`cls` 和静态方法

| 写法 | 第一个参数 | 需要已有对象吗 | 典型用途 |
|---|---|---:|---|
| 普通实例方法 | `self` | 是 | 操作当前对象 |
| `@classmethod` | `cls` | 否 | 创建或恢复对象 |
| `@staticmethod` | 无自动参数 | 否 | 与类相关的独立工具函数 |

`self` 代表某一个已经创建的对象：

```python
manifest.save(path)
```

这里 `save()` 保存的是 `manifest` 自己，因此使用 `self`。

`cls` 代表当前 class：

```python
manifest = SampleManifest.load(path)
```

读取文件之前还没有 Manifest 对象，所以 `load()` 使用 `@classmethod`。它通过
`cls.from_dict(data)` 或 `cls(...)` 创建并返回新对象。

#### 当前项目中的两个 classmethod

```text
SampleManifest.from_dict(data)
    Dict -> SampleManifest

SampleManifest.load(path)
    JSON 文件 -> Dict -> SampleManifest
```

`cls` 不是“传入其他 class”。调用 `SampleManifest.load(...)` 时，`cls` 就是
`SampleManifest`；如果未来由子类调用，它也可以代表那个子类。

#### 返回类型标注

```python
def load(...) -> "SampleManifest":
```

这里的返回类型标注主要用于阅读、编辑器提示和静态检查。真正保证返回 Manifest
的是函数内部执行了 `return cls.from_dict(data)`，而不是标注本身。

### Python file process

#### 路径、文件对象和数据对象

文件操作需要区分三个概念：

```text
数据对象：Dict、List、SampleManifest
文件路径：Path，说明文件在哪里
文件对象：handle，表示已经打开的文件
```

`json.dump()` 需要文件对象，不能直接把路径当成文件对象传入。

#### `Path`

```python
from pathlib import Path

path = Path("runs/demo/sample_manifest.json")
```

创建 `Path` 只是在内存中表示路径，不会立即创建文件。

常用操作：

```python
path.parent
path.name
path.exists()
path.parent.mkdir(parents=True, exist_ok=True)
```

`parent` 是属性，不是函数：

```text
path.parent    正确
path.parent()  错误
```

类型标注：

```python
path: str | Path
```

表示调用者可以传字符串，也可以传 `Path`。函数内部用 `Path(path)` 将两种输入
统一成 Path 对象。

#### 获得文件对象

```python
with path.open("w", encoding="utf-8") as handle:
    ...
```

`handle` 是文件对象。`with` 代码块结束后，Python 会自动关闭文件，即使中途
发生异常也会关闭。

常见模式：

| 模式 | 含义 |
|---|---|
| `"r"` | 读取文本文件 |
| `"w"` | 写入并覆盖文本文件 |
| `"a"` | 追加到文本文件末尾 |
| `"rb"` | 读取二进制文件 |
| `"wb"` | 写入二进制文件 |

Manifest 是 JSON 文本，因此保存使用 `"w"`，读取使用 `"r"`。

#### JSON 的四个函数

| 函数 | 输入或输出位置 | 作用 |
|---|---|---|
| `json.dump()` | 文件对象 | Python 对象写入文件 |
| `json.load()` | 文件对象 | 从文件读取 Python 对象 |
| `json.dumps()` | 字符串 | Python 对象转换成 JSON 字符串 |
| `json.loads()` | 字符串 | JSON 字符串转换成 Python 对象 |

字母 `s` 可以记成 string。

#### `save()` 数据流

```text
接收 str 或 Path
-> Path(path)
-> validate()
-> 创建 path.parent
-> to_dict()
-> path.open("w")
-> json.dump(..., handle)
-> 自动关闭文件
```

`save()` 使用 `self`，因为它保存当前已经存在的 Manifest。它只产生文件副作用，
通常返回 `None`。

#### `load()` 数据流

```text
接收 str 或 Path
-> Path(path)
-> path.open("r")
-> json.load(handle)
-> 得到 Dict
-> cls.from_dict(data)
-> validate()
-> 返回 SampleManifest
```

`load()` 使用 `@classmethod`，因为读取前没有已有对象。

#### 常见错误

```text
错误：path.parent()
原因：parent 是属性，不是方法。

错误：json.dump(data, path)
原因：第二个参数需要已经打开的文件对象。

错误：只获得 path.parent，但没有 mkdir()
原因：Path 不会自动创建目录。

错误：save() 和 load() 都写成实例方法
原因：load 前没有可供调用的 Manifest 对象。
```

### AI Infra 知识地图与学习优先级

#### 当前判断

目前最明显的短板不是某个高级分布式框架，而是 Python、PyTorch 和软件工程基础
还没有形成稳定的知识体系。依据包括：开发过程中需要反复确认 `self`、
`classmethod`、Dict/List 解包、`detach()`、`backward()`、`Path`、文件对象、抽象
接口和 factory 的含义。

这并不代表不适合做 AI Infra。当前已经表现出的优势包括：会质疑不必要的抽象、
关注统一数据流、主动区分训练热路径与 artifact 旁路、重视测试与可复现性，并且
愿意把问题追问到真正理解为止。

#### 按重要性排序的知识清单

| 排名 | 知识领域 | 需要掌握的内容 | 当前目标 |
|---:|---|---|---|
| 1 | Python 基础与标准库 | 变量、函数、class、`self/cls`、容器、typing、异常、context manager、Path、JSON、import/package | 不依赖提示写出 Manifest 一类的数据模块 |
| 2 | PyTorch Tensor 与 Autograd | shape、dtype、device、计算图、leaf tensor、`requires_grad`、`detach`、`no_grad`、`backward`、梯度累积 | 能逐行解释并独立写出一次 update |
| 3 | 软件工程基础 | 模块边界、接口、依赖方向、ABC、composition、factory/registry、公开 API、包结构 | 能判断一个抽象应保留、合并还是删除 |
| 4 | 完整训练生命周期 | dataset、forward、loss、backward、optimizer、scheduler、metrics、checkpoint、resume、EMA | 能实现真正等价的保存和续训 |
| 5 | 测试、调试与可复现性 | unit/contract/integration/smoke、mock、断点、日志、seed、Git、配置快照、最小复现 | 修改主线后能说明“证明了什么” |
| 6 | 数学与机器学习基础 | 线性代数、概率、期望方差、梯度、优化、最大似然、归一化、KL divergence | 能从公式判断代码数值是否合理 |
| 7 | Linux 与进程模型 | shell、环境变量、文件权限、进程/线程、stdin/stdout、signal、磁盘和内存、SSH | 能独立定位训练启动和环境问题 |
| 8 | GPU 与 CUDA 基础 | GPU 执行模型、显存构成、activation/gradient/optimizer state、mixed precision、CPU-GPU copy、OOM | 能估算显存并解释性能瓶颈 |
| 9 | 分布式训练 | rank/world size、process group、collective、all-reduce、DDP、FSDP、ZeRO、NCCL、梯度同步 | 能把单卡训练扩展到双卡并检查等价性 |
| 10 | 数据与存储 Infra | Dataset/DataLoader、sampler、shuffle、shard、streaming、cache、manifest、schema/version、对象存储 | 能保证样本、reward 和 checkpoint 可追踪 |
| 11 | 网络与服务化 | HTTP/RPC、client/server、序列化、batching、timeout、retry、并发、异步、幂等、健康检查 | 能可靠接入 World-R1 reward server |
| 12 | Diffusion / Flow 基础 | latent、noise schedule、timestep、denoising、ODE/SDE、flow matching、trajectory、transition log-prob | 能解释 image/video rollout 里的每个 tensor |
| 13 | RL / PPO / GRPO 基础 | policy gradient、importance ratio、clip、KL、reward、advantage、group normalization、credit assignment | 能解释三个 GRPO 变体为何不同 |
| 14 | 实验管理与可观测性 | config、metric、log、trace、artifact、可视化报告、实验对比、告警 | 能定位一次实验从哪一步开始异常 |
| 15 | 容器与任务调度 | Docker image、volume、network、GPU runtime、SLURM、Kubernetes job、资源申请 | 能稳定提交和复现实验任务 |
| 16 | 可靠性与容错 | checkpoint 原子性、断点续训、重试边界、数据校验、故障恢复、分布式超时 | 训练中断后不丢失或污染实验状态 |
| 17 | 性能与成本工程 | throughput、latency、utilization、profiling、benchmark、capacity planning、GPU-hour | 能用数据决定优化位置而不是凭感觉 |
| 18 | 安全与依赖治理 | secret、token、权限、远程代码、依赖锁定、模型来源和供应链风险 | 不把凭证和不可信代码带进训练环境 |
| 19 | 高级 Kernel 与编译 | C++、CUDA kernel、Triton、算子融合、`torch.compile`、通信计算重叠 | 只在 profiler 证明必要后学习 |

#### 推荐学习顺序

```text
第一层：Python -> PyTorch -> 软件工程 -> 单卡训练循环 -> 测试
第二层：Linux -> GPU -> 双卡 DDP -> checkpoint/resume
第三层：数据管线 -> reward server -> diffusion/flow -> GRPO
第四层：observability -> Docker/调度 -> 容错 -> 性能成本
第五层：CUDA/Triton/编译优化
```

#### 当前项目中的练习路径

1. 独立完成 `SampleManifest`、测试和 JSON round-trip，补齐 Python 文件操作。
2. 独立解释 `VisualRLTrainer` 的每一步，画出 tensor、梯度和文件数据流。
3. 修复 checkpoint，使 optimizer 和统计状态可以真正 resume。
4. 用 tiny model 做单卡训练，记录显存、step time 和吞吐量。
5. 将同一 tiny 训练扩展到双卡 DDP，比较 loss 和有效 batch size。
6. 把 World-R1 mock provider 换成受控的本地服务，加入 timeout、retry 和 cache。
7. 完成一个小规模 image/video 实验，并自动保存 config、manifest、metrics 和报告。

#### 暂时不要优先投入

- 不要先学习复杂 Kubernetes 集群管理。
- 不要先手写 CUDA/Triton kernel。
- 不要同时接入更多模型和算法。
- 不要只背 FSDP、ZeRO、NCCL 名词而没有单卡和 DDP 实验。
- 不要把 tiny smoke 的通过描述成真实训练已经验证。

#### 官方学习入口

- [Python Classes](https://docs.python.org/3/tutorial/classes.html)
- [PyTorch Autograd](https://docs.pytorch.org/docs/stable/autograd.html)
- [PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)
- [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### 当天总结

```text
Dataclass 负责定义结构化数据
Dict / List 负责保存和组织数据
Classmethod 负责从外部数据创建对象
文件操作负责让内存数据持久化
AI Infra 学习应从 Python/PyTorch 的可执行能力逐步扩展到 GPU 和分布式系统
```

它们在当前 Manifest 中连成一条完整链路：

```text
JSON 文件
<-> Dict / List
<-> SampleRecord / SampleManifest dataclass
```

## 5.29 · 2026

### 当天主题

- VisualRL 的理想抽象
- RolloutEngine
- 算法层
- 代码阅读顺序
- 项目报告与简历表达

### VisualRL 的理想抽象

项目目标不是简单拼接三个研究仓库，而是建立一条统一的 image/video Diffusion RL
数据流：

```text
Config
-> PromptDataset
-> RolloutEngine
-> FeedbackProvider
-> OptimizerPlugin
-> artifacts / report
```

当前主线只服务 Flash-GRPO、TempFlow-GRPO 和 World-R1/Wan；其他模型和旧实现
不进入核心设计。

### 三项工作的角色

| 来源 | 在 VisualRL 中的角色 |
|---|---|
| TempFlow-GRPO | timestep-aware optimizer 和 image/flow RL 思路 |
| Flash-GRPO | video rollout 与 Wan 训练能力 |
| World-R1 | 3D/world-aware reward provider |

### RolloutEngine

RolloutEngine 管理“如何采样”，不等于模型本身：

```text
prompt + metadata
-> ModelAdapter
-> image/video trajectory
-> RolloutBatch
```

三种 rollout 行为：

- `FullTrajectoryRollout`：普通 GRPO 完整轨迹。
- `SingleStepRollout`：Flash-GRPO 单步训练样本。
- `BranchingRollout`：TempFlow-GRPO 分支样本。

### 算法层

算法文件主要是 policy loss kernel，不是完整 Trainer。完整 update 还包括 advantage、
log-prob 重算、backward、optimizer step 和 metrics，这些行为应由
`OptimizerPlugin` 统一封装。

### 推荐阅读顺序

```text
测试
-> core types
-> runner / trainer 主循环
-> rollout
-> feedback
-> optimizers
-> algorithms
-> configs
-> legacy/reference code
```

先理解数据如何流动，再阅读具体公式和模型实现。

### 项目表达

对外应描述为：

> 构建面向 image/video generation 的统一 Diffusion RL infra，统一 rollout、
> feedback、optimizer 和实验产物，降低新 reward、新算法和新模型的接入成本。

不应只描述为“整合了三个项目”。

## 7.11 · 2026

### 当天主题

- 从两套框架收敛为唯一 `ExperimentRunner`
- Plugin 不只是抽象类，还必须有统一构造和注册路径
- Reward 与 advantage normalization 的职责边界
- 真正的 TempFlow shared-prefix branching
- 完整 checkpoint 与 resume 等价性
- 通过程序填空学习已完成的工程实现

### 关键结论

```text
FeedbackProvider 只给出 raw / weighted reward
AdvantageComputer 只做训练归一化
OptimizerPlugin 完成一次完整 update
ExperimentRunner 只协调，不包含算法公式
ArtifactManager 只落盘，不进入计算图
```

GRPO 的同一个 prompt 至少需要两个 sample；否则 reward 减去组均值后恒为 0。

TempFlow 必须区分：

```text
branch_step_index       轨迹中的位置
branch_timestep_value   scheduler 的真实时间值
```

断点续训不只是加载模型，还要恢复 Adam moments、plugin state、step 和 RNG；本轮用“连续 2 步”和“1+1 resume”参数完全一致的测试验证。

### 容易疏忽

- 抽象类的价值是统一约定，不一定减少代码行数。
- 只有接口没有 factory/registry，外部实现仍无法由 config 选择。
- 测试能运行不等于训练语义正确；需要数值不变量测试。
- Tiny/Fake runtime 只能证明 infra contract，不能代表真实 SD3/Wan 性能。

### 下一步学习

按照 `docs/FILL_IN_LEARNING_PLAN.md` 完成七个 20-40 分钟程序填空；每题先独立完成，再对照正式实现复盘数据流和错误边界。
