# Assignment A

- active: true
- status: assigned
- task_id: H3-M1b-cps
- pair: coding-a → simplifier-a
- execution: sequential; B/C idle
- source_dependency_commit: `7b29f4bf8d990794d4b9bc826a82bd390861d530`（M1a coding snapshot，配对 approved）
- integration_parent: `8a6763cf3defcb07415dfeda501204e630b42f5a`（含 M1a 审查记录）；接收后记录引擎实际 branch/HEAD，不自行切分支或提交。
- previous_task: H3-M1a-joint-schedule approved，无 changes、无失败测试待返工。旧交付/审查记录保留，不覆盖。

## 唯一目标

一个原子功能：基于已实现 `H3JointLayout` / `H3ScheduleSnapshot` 的 **joint CPS 采样与固定 action replay 评分**。这是后续 rollout/replay 使用的真实数值内核，不是训练 loop；不改旧 Gaussian ABI。

## 允许写入

1. 新 `visual_rl/algorithms/dynamics/h3_cps.py`
2. `tests/test_h3_mvp_cpu.py`：保留已有六项及断言，新增 CPS 目标测试和导入身份核验。
3. `workflow-workspace/2026-10-27-h3-8gpu-mvp-dev-001/coding-a/H3-M1b-cps.md`

已批准 `h3_joint.py` / `h3_schedule.py` 此项只读；如真实接口问题需要改动，先报告最小扩大范围。不改原七项、旧 Dynamics、runtime、registry、recipe、task/reference/AGENTS/.gitignore/workflow。引擎负责原子 snapshot。

## 数学锁（选定 UniRL 语义，非 Gaussian）

对两个模态分别使用 M1a snapshot 的 `(s,u)`，输入预测必须已经是 **noise-ward**，此内核不再翻符号。FP32 中：

```
q = u * sin(eta*pi/2)
x0 = x - s*v
e_hat = x + (1-s)*v
mu = (1-u)*x0 + sqrt(u*u-q*q)*e_hat
action = FP16(mu + q*z)
score_m = -mean((stopgrad(action).float() - mu.float())**2)
score = (Dv*score_video + Da*score_audio)/(Dv+Da)
```

- `Dv`/`Da` 来自 layout 的 generated 元素数，不是 0.5/0.5、不平均两模态 q、不加 Gaussian `1/(2q²)` 或 log(q)。结果明确命名/标记 CPS proxy score，不宣称 exact likelihood/KL。
- eta 默认0.7，首版限定有限 `0 < eta <= 1`，拒绝 bool/NaN/inf/0/负值/越界；eta0 ODE 不在此 policy 接口。方差/q 不加 epsilon，不用 clamp 吞掉异常。仅 snapshot 最后 transition 可 q=0；全部 N 步可评分，末步不删、不改计数。非末步 float32 下溢导致 q=0 应拒绝。
- 两流 mean/q/score 都按 FP32；存储 x/action 明确 FP16。输入预测允许已说明的浮点精度，先 FP32 运算；形状/设备与 packed state 匹配，不能静默广播错行。不要改 layout 的通用 dtype 行为，有限浮点/存储约束由 CPS 边界负责。

## 最小调用合同

实现一个窄 `H3CPSDynamics`（或同等简单显式函数组），绑定 layout、精确双 schedule 与 eta，提供 sampling 与 evaluate。不要继承 Gaussian `Dynamics` 并放松其校验。可定义最少所需的 stats/record 类型，命名和签名在交付中固定供下游使用：

- stats 由唯一数学函数产生，sample 与 evaluate 共用；暴露足够 mean/q/双 dt 信息供诊断测试，不复制两份公式。
- sample 接收 detached FP16 joint x、noise-ward joint prediction、显式 transition index、显式 video/audio `torch.Generator`（与目标 device 兼容）；不依赖全局 RNG。两模态各画一份 FP32 noise，固定 video 后 audio 的调用顺序；**末步 q0 也消费两份 noise**。允许 CPU 测试 tiny B>1；生产仍 B1。
- sample 返回 detached 的紧凑转移/record，至少拥有 layout/schedule/index/eta/score-family/quantization 合同及 x/action（不保模型/Generator/计算图）。action 就是本 T2VA 无 conditioner 情况下的 next state。可带 detached rollout score 诊断，但不得把它冒称已冻结 replay old anchor。
- evaluate 用 record 的精确合同与存储 action，加当前模型预测重新计算 FP32 score `[B]`；梯度只经过当前 prediction→mean→score，固定 action 与 x 不留图。不得重新抽 noise 或从 current mean 重新生成 action；若 dynamics 与 record 配置冲突须拒绝，不能悄悄换 schedule/layout/eta。
- sample/record 不得因调用方后续修改原 x/action 缓冲而漂移；简单拥有副本和受控读取即可，不造哈希/池/序列化框架。CPS score 的 finite 检查和 FP16 quantization overflow 检查要保留。
- 本项**不实现 old-policy anchor**；更新前冻结 replay anchor 依旧是后续真实阶段任务。固定 action 的所有权不等于旧策略锚已完成。

## 必须的目标 CPU 测试

实际消费 M1a 两合同和新内核，禁止 mock 所测数学：

1. 不等 AV 元素数、非相同双 schedule，手算 FP32 `mu/q/score`；证明 joint 元素权重，不是模态等权或 Gaussian。测试不只用零输入/零预测。
2. 显式 generator 可复现 sample、FP16 action roundtrip；更换 video/audio RNG 分别影响相应模态；不污染全局 RNG；末步 generator state 证明仍消费两个 noise draw。
3. sample 固定 action 后 evaluate 初始分数一致，修改当前预测产生正确非零梯度/分数变化；action/x 不带图，输入和读取 mutation 不改变 record 的固定内容。evaluate 不消费 RNG。
4. 末步双 q0 仍得到合法 finite CPS score、FP16 量化残差下可有非零梯度；选非精确 FP16 数值并对照解析梯度，不能用全零例掩盖。所有 N index 可评分。
5. 在本 H3 目标测试中直接使用当前真实 `TransitionMeanStd.validate_against`（或真实旧 Dynamics 公共端口）证明 Gaussian q0 仍被拒绝；正 q 的 Gaussian 与 CPS 数值不同。不得修改旧代码/断言使它通过。
6. shape/dtype/device、nonfinite prediction/state/action、非法 eta/index、合同不匹配、FP16 overflow 等负例；不得靠 epsilon/删 finite 检查修复。

原样执行 plan.md 两条 CPU 命令，Flash仍7/7、H3目标集无 skip/xfail；不跑全量。记录实际命令/exit/时间/数量，最后源码编辑后重测。记录当前 `git diff`（本轮既有测试改动）及新增文件、`git diff --check` 和 branch/HEAD；未获引擎 snapshot SHA 写 pending。

## 来源、许可及非目标

先读当前真实两合同与旧 Gaussian 代码；重读 reference/UniRL/LICENSE 和固定 `8cfbac17765004afb68fef08a4e011a4c8daf555` 的 `unirl/sde/kernels.py`（CPSSDEStrategy及量化 finalize）、`unirl/models/minimax_h3/diffusion.py` 的双流路径。独立实现给定数学并保留语义来源说明，不 import reference、不复制上游通用 registry/训练框架。若需实质复制及额外许可文件，先请求明确范围。

不实现 adapter/权重/LoRA/FSDP2/生产装配/评分服务/advantage/旧锚/optimizer/checkpoint。Simplifier A 重点审 action detach、精度/双 schedule、q0 与 RNG、AV 权重、Gaussian 严格边界、重复公式/过度 artifact 及未来消费者接口；changes 回 A 修复并两集重测。本项通过只证明数值合同，不能称 MVP、GPU 或质量通过。
