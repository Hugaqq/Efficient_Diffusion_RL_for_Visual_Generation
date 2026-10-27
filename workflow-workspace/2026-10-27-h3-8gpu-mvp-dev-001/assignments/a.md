# Assignment A — W1

- active: true
- task_id: H3-M4a1-anchor
- pair: coding-a → simplifier-a
- execution: parallel W1；见decisions/wave-01.md
- integration_parent: `b7f938506f16034ac5d2896ddfa44c16eb4af83d`
- source_dependencies: M1a `7b29f4bf8d990794d4b9bc826a82bd390861d530`、M1b `725f48f4d67c67d90c60cc3536f4bc40f12802f2`，均approved；无待返工issue。

## 唯一目标与文件

实现**rank-local完整collection的一次性冻结replay旧锚**，真实调用现CPS evaluate，不是保存rollout_score别名。

仅允许新 `visual_rl/algorithms/optimization/h3_anchor.py`、新 `tests/test_h3_mvp_cpu_anchor.py`、`workflow-workspace/2026-10-27-h3-8gpu-mvp-dev-001/coding-a/H3-M4a1-anchor.md`。现CPS/joint/schedule、objective、旧tests、trainer/runtime均只读；不import或修改B/C本波新模块。

## 冻结合同

- 输入是明确有序 `(sample_id, H3CPSRecord)` 单格序列，B1，所有样本完整覆盖同一N步（含q0），同layout/schedule/eta/score/quantization、同collection_id/behavior_id。先在**任何模型调用前**验证唯一sample/index覆盖、无空/缺格/重复、显式最大格数/输入轨迹bytes上限；不同sample不按prompt文本合并。IDs由未来pool给，不自己造data sampler。
- 最小一次性owner绑定collection/behavior，`capture`只允许成功一次；失败不公布部分scores且该owner不能悄悄重新capture。没有re-anchor/update后刷新API。可提供最小冻结结果类型；不另造pool/runner/manifest。
- `capture`接收 `predict(record_on_device) -> joint_noise_ward_prediction` callback、明确device及输入records；逐格搬运detached x/action/诊断值到指定设备，通过原CPS record构造和`evaluate`重算。一次最多一格临时设备record；不要改原h3_cps加to方法，不缓存GPU轨迹或图。输入CPU记录的tensor所有权保持；不读取内部下划线buffer。
- capture在no_grad内按输入固定顺序调用callback一次/格，将真正replay score冻结为CPU FP32。有意篡改rollout_score的测试仍必须捕获正确evaluate值。只能返回有限完整结果，不做epsilon/补值/丢样；不删末步。
- 模型EVAL/behavior参数固定/credit已完成是上层阶段前置；此owner不持有模型、optimizer或旧权重。可用明确version验证callback在capture前后读取behavior标识并拒变更；不要hash巨大基座或复制权重。不得声称本项已经实现全rank冻结门。
- 读取scores必须核对collection/behavior和sample/index，返回detached副本、固定映射；外部mutation不改owner。存储只保小scores/合同/键，不保predict callback或模型引用。current预测变化后，old值不变。
- 下游使用既有 `ClippedSurrogateObjective` / `PolicyLossInputs`（beta0），不得重写PPO。此项不改该objective，仅测试真实old锚如何进入它。

## 验收

真实CPS+tiny trainable预测叶子：capture恰N*rows次、no_grad、全覆盖包括末步；读取old与逐格参考evaluate一致；rollout_score不能冒充锚；模型更新后old保持不变而current改变，实际既有objective/backward使用冻结old可得到有限梯度。测试输入缺格/重复/合同冲突在callback前拒绝、非有限callback中途失败无部分结果、二次capture拒绝、版本错误、输入/读取mutation保护及上限提前拒绝。明确局部算法证据不是controller/world门已实现。

测试只写新feature文件，核验当前worktree导入路径；无需移动原22项。真实运行plan最新版Flash固定七项与 `tests/test_h3_mvp_cpu*.py` 通配符命令，禁止skip/xfail/全量测试，最后源码编辑后重测，记录命令/exit/数量/时间、实际branch/HEAD和diff/check。分支由引擎提供，不手工Git提交/合并。

## 非目标

不做rollout/reward/advantage/pool消费、world计划、FSDP/PG、checkpoint、controller接线。本项代码必须可被后续OptimizeStage调用但不提前改stage。若现合同不足报告原owner接口问题，不越界修别组。简化器A重点看真正replay、固定action/旧锚、顺序/上限/失败原子性及非重复PPO；changes回A重测重审。GPU_PENDING。
