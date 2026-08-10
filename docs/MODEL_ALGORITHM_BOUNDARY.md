# Model–Algorithm boundary

VisualRL v0.8 只有两个公开可替换轴：`ModelAdapter × AlgorithmModule`。二者不直接
构造或识别对方；不可避免的 scheduler/transition 耦合由 composition 层的 model-bound
Dynamics 投影显式承接。

## 职责与依赖

| Owner | 负责 | 不负责 |
| --- | --- | --- |
| `visual_rl.models` | artifact/component 生命周期、conditioning、latent geometry、单步 `predict()`、reference view、decode、不可变 scheduler blueprint | rollout loop、transition/log-prob、reward、advantage、loss、算法/recipe 分支 |
| `visual_rl.algorithms` | requirements/blueprint、Dynamics、rollout、reward/advantage/credit、recompute/objective/update、trainer stage | 具体 SD3/Wan class、Diffusers pipeline 构造、模型名分支 |
| `visual_rl.composition` | registry/declaration 解析、capability matching、model-bound Dynamics projection、静态拒绝原因 | CUDA/model/reward 实例构造、训练公式 |
| `visual_rl.runtime` | 唯一 composition root；加载并准备具体组件、创建 `PolicyRuntimePort`、物化 `BoundAlgorithm` | 按 recipe 名实现另一套算法或 rollout |

允许的核心依赖方向是：

```text
models.interface + models.scheduler ──┐
                                      ├── composition declarations/contracts
algorithms declarations/contracts ────┘
                                                   │
                                                   ▼
runtime: prepared ModelAdapter → PolicyRuntimePort → BoundAlgorithm
```

`models/**` 不得 import `algorithms/composition/runtime/training`；`algorithms/**` 可以使用
import-safe model port/scheduler ABI，但不得 import `models.implementations`。

## 编译期连接

1. model provider 返回 `ModelDescriptorContract`，声明 task/media/layout/prediction/time、
   scheduler blueprint schema、Dynamics binding family 和 replay schema。
2. algorithm provider 原子返回 frozen config、`AlgorithmRequirements` 和
   `AlgorithmBlueprint`。
3. blueprint 固定描述 trainer、model-bound Dynamics、rollout、credit。算法只声明
   Dynamics `implementation_family`，不能指定某个模型的 concrete Dynamics class。
4. `DynamicsProjectionRegistry` 使用
   `(model_binding_family, algorithm_implementation_family)` 选择 concrete Dynamics
   declaration和 frozen params。
5. compatibility matcher 对 model、algorithm、Dynamics 三方能力求交；不兼容组合在加载
   模型权重、初始化 CUDA 或获取 reward resource 前失败。

当前 built-in projection 是：

```text
sd3.flow-sde.v1 × flow-sde → flow-sde
wan.flow-sde.v1 × flow-sde → wan-flow-sde(profile=standard|flash|conditioned)
```

这不是模型/算法白名单。新模型 family 通过注册新的 projector 扩展，不修改算法；新 Dynamics
family 通过新的 algorithm requirement/projector 扩展，不修改模型。

## 运行时连接

runtime 准备模型后创建 `DefaultPolicyRuntimePort`。算法只看到这个 port 和已经验证的
`ModelAlgorithmBinding`：

```text
rollout
  → policy.prepare_latents(...)
  → policy.predict(ModelInput)          # exactly one model forward
  → policy.transition(sample request)   # DynamicsSession samples action/log-prob
  → policy.decode(...)

optimize
  → policy.predict(current/reference)
  → policy.transition(evaluate request) # replay stored action/log-prob
  → credit/objective/backward/commit
```

模型 prediction 不等于 diffusion transition。Dynamics 才拥有 mean/std、随机采样、
arbitrary-action likelihood 和 replay snapshot；rollout 才拥有 full/branching/single-step 控制流。

## 当前兼容组合

| Algorithm/recipe | Model | 连接结果 |
| --- | --- | --- |
| Flow-GRPO | SD3.5 | full trajectory + SD3 flow-SDE |
| Flow-GRPO | Wan2.1 T2V | full trajectory + Wan standard stochastic flow-SDE |
| TempFlow-GRPO | SD3.5 | branching + branchable SD3 flow-SDE |
| Flash-GRPO | Wan2.1 T2V | single-step + Wan flash profile/rectification metadata |
| World-R1 core/release | Wan2.1 T2V | Flow-GRPO + conditioned Wan Dynamics + camera conditioner + general/3D rewards |

TempFlow × Wan 当前因 Wan Dynamics 不提供 branching 被拒绝；Flash × SD3 因 SD3 binding 不提供
single-step rectification metadata 被拒绝。解耦意味着兼容组合无需修改另一轴，不意味着任意
笛卡尔积都合法。

## 扩展规则

新增模型：

1. 在 `models/catalog.py` 增加 import-safe config/declaration provider。
2. 在 `models/implementations/` 实现 `ModelAdapter` 的 component、conditioning、latent、单步
   prediction 和 decode ports。
3. 声明 scheduler artifact/binding family；不要实现 rollout 或 algorithm switch。
4. 若现有 Dynamics family 可承接它，在 composition 注册 projector；否则先新增独立 Dynamics。

新增算法：

1. 在 `algorithms/modules/config.py` 定义 frozen config、requirements 和完整 blueprint。
2. 在 `algorithms/catalog.py` 注册 import-safe provider/runtime target。
3. 复用或新增 rollout、Dynamics family、credit 和 trainer component；不要 import concrete model。
4. 为所需 model family 注册 projector，并加入兼容/不兼容正反例。

新增组合只允许修改 composition projector/config integration；如果必须同时修改 model 和 algorithm
实现，说明公共 port 或 capability contract 还缺少真正的语义，应先补 contract，而不是加入名字判断。

## 永久门禁

下列检查必须保持通过：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_phase_a_public_axes.py::test_model_package_never_imports_algorithm_or_orchestration_packages \
  tests/test_phase_a_public_axes.py::test_canonical_algorithms_never_import_concrete_model_implementations \
  tests/test_phase_a_public_axes.py::test_public_composition_never_branches_on_model_or_algorithm_names \
  tests/test_phase_a_public_axes.py::test_incompatible_pair_fails_before_runtime_or_model_construction
```

这些门禁证明结构，不证明真实训练。每条正式 route 仍需独立的 frozen config、20 次 optimizer
commit、exit zero、`SUCCESS`、step-20 checkpoint、正梯度、失败日志扫描和显存序列。
