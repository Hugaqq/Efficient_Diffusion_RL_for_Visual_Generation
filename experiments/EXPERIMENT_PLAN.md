# VisualRL v0.8 实验证据计划

更新日期：2026-08-02

本文只定义 v0.8 后续实验的证据边界。v0.7 的计划、schema-v1 配置、Python
驱动和结果已经冻结在 [v0.7 archive](v0_7/archive/README.md)；这些记录用于审计
历史结论，不能作为 v0.8 controller、checkpoint format 或模型适配已经通过的证据。

## 当前执行表面

v0.8 训练只接受 schema-v2 配置，并由唯一 module entry 启动：

```bash
python -m visual_rl.train configs/v2/flow_grpo_sd3.yaml
```

该命令只说明公开接口形状。仓库当前没有随附模型权重、reward artifact、完整
World-R1 dynamic prompt 或四条真实 GPU 运行记录，因此不能据此宣称任一正式配置
已经完成真实 one-update。

成功终止的 v0.8 run 使用独立只读模块检查：

```python
from visual_rl.artifacts.inspection import audit_run, inspect_run

status = inspect_run("/absolute/run")
audit = audit_run("/absolute/run")
if not status.ok or not audit.ok:
    raise RuntimeError((status.errors, audit.errors))
```

inspection 只验证 v0.8 terminal layout 和 authoritative checkpoint。它不能把旧
commit-chain、未完成 run 或 v0.7 evidence 转换成 v0.8 成功记录。

## 当前证据状态

当前目标环境审计记录为 Python 3.10.20、Torch 2.12.0、Diffusers 0.39.0、Transformers 5.14.1、
`CUDA available=False`，且 accelerate/peft 缺失；因此它不能执行真实 GPU one-update。TempFlow、
Flash 和 World-R1 下述“结构已落地”均只描述 typed/fake implementation，不能替代 native GPU
parity record。

| Evidence gate | 当前状态 | 可宣称范围 |
| --- | --- | --- |
| strict schema-v2 compile | `passed locally` | 五份正式配置可静态解析 |
| Flow + fake SD3 default composition | `passed locally` | single-process fake two-step continuous vs step-1 resume exact G7a |
| 五份 Demo fake G0-G6 + terminal/no-op lifecycle | `passed locally on current tree` | 五条 default-controller 路径均完成一次 update、终态 checkpoint 与 completed/no-op restore；仅 Flow 有 exact continuation 的独立 evidence，其他路径的 terminal restore 不构成 G7a |
| SD3 resolution/patch-aware dynamic-shift schedule fixture | `not_run` | 当前不能声明 SD3 native timestep/sigma parity |
| TempFlow per-timestep × K branch topology fixture | `local CPU structural tests passed` | paper path 已是 nonterminal N-1 cells：schedule=N、physical=`(N-1)(N+4)/2`；显式 ODE、同-step SDE/ODE prediction 复用、selection/snapshot identity、base-B0 initial draw + K expansion 与逐 timestep reward/credit 轴已通过。上游 branch RNG 顺序、B×K mainline 计算差异和真实 GPU parity 仍未完成 |
| Flash first-10 prompt-shared + Wan 512 encoding fixture | `single-process fake only` | first-10、同 prompt 共享 mapping、checkpoint identity 与 512 positive/negative fake encoding 已有结构证据；真实 Wan tokenizer/model fixture 未跑，rank broadcast 未实现并 fail closed |
| World-R1 keyed-random batch-shared reward-frame fixture | `structural only` | recipe-owned keyed-uniform all-frame/batch-shared selection 与 provenance 已接线，fixed-middle 保留为 extension；尚无真实 Wan rollout、general/3D service 或 resume 纵向证据 |
| G3 reference-policy state evidence | `passed locally` | content-addressed mode/owner/restorable-state/projection/numerics evidence 已进入 graph-level binding；Flow active、TempFlow/SD3 capability-only、Wan none。它不证明真实 reference forward 数值 parity |
| early recipe/launch manifests | `passed locally` | 模型加载前 recipe manifest、RuntimeBind 后 launch manifest、fresh/resume exact-byte、drift/symlink fail-closed 与 session cleanup 已覆盖 |
| core wheel release contract | `local Python 3.10 build/install passed` | 最终 wheel SHA-256=`507ac83c2b7186e76c74830956c62c46b4061c1b70e6ec396c3234c3193ae3bd`；archive verifier、core-only isolated install、退役模块/入口缺失与 recipes/registries/help smoke 已通过；checked-in GitHub Actions 尚未在远端运行 |
| Flow-GRPO + SD3 real one-update | `not_run` | 无真实模型结论 |
| TempFlow-GRPO + SD3 real one-update | `not_run` | 无真实模型结论 |
| Flash-GRPO + Wan real one-update | `not_run` | 无真实模型结论 |
| World-R1 + Wan real one-update | `not_run` | 无真实 reward/camera 纵向结论 |
| native parity records | `not_run` | 无 v0.8 数值等价声明 |
| DDP/FSDP/DeepSpeed | `out_of_scope` | 当前 Demo 只承诺 single-process |

仓库保留的单点分支策略不属于论文 fidelity path：当前 single-point branching 只算 ablation，
不能替代上表的 per-timestep × K TempFlow 证据。

当前最终本地回归为 `1349 passed, 4 skipped`；4 个 skip 均为 CUDA unavailable gate。这个数字只
证明 CPU/结构/fake/release-contract 测试，不授予上表四条真实 one-update 或 native parity。

## 晋级顺序

1. 保持当前定向/full suite、五配置 fake G0-G6/terminal-no-op、manifest/checkpoint 和 wheel
   contract 为每次 source delta 的本地回归门禁；仅 Flow two-step 路径承担 G7a exact continuation。
2. 在远端 CI 运行 checked-in release workflow，保存 wheel、digest 与 isolated-install log；本地
   release pass 不能冒充远端 artifact。
3. 固定 Python、Torch、CUDA、Diffusers、Transformers、PEFT、GPU、模型、数据和
   reward service revision；缺失项必须记录为 `not_run`。
4. 依次完成 Flow/SD3、TempFlow/SD3、Flash/Wan、World-R1/Wan 的真实单卡
   one-update、failure teardown 和 safe-point resume；进入各路径前先通过上表对应的 schedule、
   topology、selection、encoding/reward-input fixture。
5. 每条路径生成绑定精确 recipe、artifact、environment、容差和代码 revision 的
   evidence record；surrogate 与 exact-environment likelihood 必须分开标注。
6. 最终 release 将远端 CI artifact、全量回归结果与四条真实 EvidenceRecord 一起发布。

## 证据纪律

- `not_run`、setup failure、缺 artifact 和缺 GPU 不能写成算法失败或通过；
- fake leaf、analytical dynamics 和 contract tests 不能升级为真实权重 parity；
- 历史 v0.7 结果不得重命名或复制成 v0.8 record；
- 不自动下载未固定 revision，不上传 checkpoint、credential、个人路径或未审查日志；
- 只有完整 evidence record 才能更新当前状态表。
