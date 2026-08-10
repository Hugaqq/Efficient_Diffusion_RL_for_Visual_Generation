# VisualRL v0.8 项目概览

更新日期：2026-08-09

VisualRL 是面向图像和视频 Diffusion RL 的模块化训练基础设施。v0.8 的核心目标不是让
任意模型和算法无条件组合，而是把两条独立变化轴做成可声明、可匹配、可拒绝和可验证的组件：

```text
ModelAdapter × AlgorithmModule
```

模型、算法和 reward artifact 的可用性最终由真实训练证明，不从 fake model、相邻配置或历史
版本继承。

## 唯一训练主线

```text
schema-v2 YAML
→ import-safe model/algorithm declarations
→ compile + static compatibility
→ artifact/environment preflight
→ runtime component graph
→ prepared ModelAdapter
→ PolicyRuntimePort + model-bound Dynamics
→ BoundAlgorithm
→ rollout/reward/advantage/credit/recompute/update
→ atomic checkpoint + terminal inspection
```

唯一训练入口是：

```bash
python -m visual_rl.train CONFIG
```

项目不再提供 v0.7 的 `visual_rl.api`、`vr.load().run()`、public Runner、
`runtime_factory.py`、`builtins.py` 或第二套训练循环。历史 v0.7 文档只用于解释旧产物，不能作为
v0.8 使用指南或源码路径索引。

## 模型与算法边界

- `ModelAdapter` 拥有模型 artifact/component、conditioning、latent geometry、单步
  current/reference prediction 和 decode；不拥有 rollout、transition 或算法判断。
- `AlgorithmModule` 声明 requirements 和完整 blueprint，并组合 trainer、Dynamics、rollout、
  reward、advantage、credit 和 update；不导入具体 SD3/Wan implementation。
- composition 在重资源加载前匹配 model/algorithm/Dynamics contract，并通过
  `DynamicsProjectionRegistry` 选择 model-bound Dynamics。
- runtime 是唯一同时构造具体 model 和 algorithm component 的 composition root；算法运行时只拿
  `PolicyRuntimePort` 和已验证的 `ModelAlgorithmBinding`。

详细职责、扩展步骤和永久门禁见
[Model–Algorithm boundary](MODEL_ALGORITHM_BOUNDARY.md)。

## 当前六条 route

| 配置 | Model × Algorithm/integration | 当前 frozen A7 状态 |
| --- | --- | --- |
| `flow_grpo_sd3.yaml` | SD3.5 × Flow-GRPO | 20/20 accepted；峰值 24,892 MiB |
| `flow_grpo_wan.yaml` | Wan2.1 T2V × Flow-GRPO | 20/20 accepted；峰值 17,568 MiB |
| `tempflow_sd3.yaml` | SD3.5 × TempFlow-GRPO | 20/20 accepted；峰值 25,148 MiB |
| `flash_wan.yaml` | Wan2.1 T2V × Flash-GRPO | 20/20 accepted；峰值 22,555 MiB |
| `world_r1_core_wan.yaml` | Wan2.1 T2V × Flow-GRPO + camera/general/3D | running；step-10 已提交 |
| `world_r1_release_surrogate_wan.yaml` | 同组合 + main/dynamic phase integration | 首次 run exit 143；同 freeze fresh retry 已排队 |

“20/20 accepted”要求同一 frozen code/wheel/config 下同时满足：20 次 optimizer commit、exit zero、
`SUCCESS`、完整 step-20 checkpoint、正梯度、日志无失败签名，以及绑定 trainer PID/物理 GPU 的
显存时间序列。World-R1 两条尚未满足，因此当前不能声明六路 A7 完成。

## 代码阅读顺序

1. `visual_rl/models/interface.py`：模型单步 port 和生命周期边界。
2. `visual_rl/algorithms/modules/{config,descriptor,interface}.py`：算法 requirements/blueprint/runtime facade。
3. `visual_rl/composition/config/{compiler,integration}.py`：recipe 编译和 model-bound Dynamics projection。
4. `visual_rl/composition/compatibility/`：model/algorithm/Dynamics 匹配与拒绝原因。
5. `visual_rl/runtime/{lifecycle,component_graph,model_binding,algorithm_binding}.py`：唯一具体装配路径。
6. `visual_rl/algorithms/rollout/` 与 `optimization/`：轨迹控制、recompute、objective 和一次 commit。
7. `visual_rl/artifacts/checkpoint/` 与 `artifacts/inspection.py`：safe point、原子持久化和只读审计。

## 当前支持边界

- single process、单 GPU；
- `gradient_accumulation_steps=1`；
- SD3.5/Wan2.1 T2V 与当前列出的 GRPO family/integration；
- remote reward service 作为独立进程和 artifact identity；
- checkpoint 只在 optimizer commit 后保存，不恢复 in-flight trajectory。

当前不承诺 DDP、多节点、FSDP、DeepSpeed、异步 reward、新的非 GRPO trainer family，或未列出的
模型—算法组合。结构测试通过不等于真实 CUDA 支持；真实 20-step 通过也不等于 native parity、
resume、训练质量或 phase-boundary 已验证。

## 证据入口

- 总计划与 gate：[Plan_8_2.md](../Plan_8_2.md)
- frozen A7 路线、收据和恢复策略：
  [v0.8 real-GPU evidence](../experiments/v08_modular_gpu_20260808/README.md)
- v0.7 历史 operational evidence：[V0_7_OPERATIONAL_EVIDENCE.md](V0_7_OPERATIONAL_EVIDENCE.md)

版本变化见 [CHANGELOG](../CHANGELOG.md)。
