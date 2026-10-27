# Assignment A

- active: true
- status: assigned
- task_id: H3-M1a-joint-schedule
- pair: coding-a → simplifier-a
- execution: sequential; 本轮 B/C idle
- source_dependency_commit: `076635a61263ffd5e7c5be4cc64d77ef99242c51`
- coordination_parent: 引擎将包含本轮 plan/assignment 的 snapshot 交给你；先核验实际 branch/HEAD 并记录，不自行切分支。无前置已批准功能提交。

## 唯一目标

实现可被下一项 CPS 和真实 H3 adapter 消费的 **generated joint state packing + 精确双 schedule** 最小合同。只做这一项，不实现整个 H3、更不顺手改旧 Gaussian 或 runtime。

## 允许写入

1. 新 `visual_rl/models/h3_joint.py`
2. 新 `visual_rl/algorithms/dynamics/h3_schedule.py`
3. 新 `tests/test_h3_mvp_cpu.py`
4. `workflow-workspace/2026-10-27-h3-8gpu-mvp-dev-001/coding-a/H3-M1a-joint-schedule.md`

除此不改；尤其不改原七项、task/reference/AGENTS/.gitignore/workflow、旧配置。需要扩大范围先报告，不在本项顺手改。引擎负责源码 snapshot，不手工 git commit。

## 输入/输出接口锁

- `models/h3_joint.py` 定义窄 `H3JointLayout`（命名若有更小明确方案在交付说明）。它描述正整数 video rows/channels 与 audio rows/channels；video/audio 两者都必须存在。输入分别 `[B,V,Cv]`、`[B,A,Ca]`，输出 `[B,V*Cv+A*Ca]`，**video first, audio second**；unpack 返回原两流形状，不改变 dtype/device、数值或梯度。真实 H3 模型的 text/audio/video attention token 排列与这个轨迹 storage packing 不是同一概念，禁止混同。
- 范围是 T2VA generated rows，无 condition/padding 冒充生成元素。不要为未来 FL2VA/SP 造 mask 框架；不支持的几何/行数必须明确拒绝。B1 是生产微批，CPU 可用 B>1 检查 batch 维不混样。不在 layout 中持有模型或状态张量，不加入模型→算法 import。
- `algorithms/dynamics/h3_schedule.py` 定义窄 `H3ScheduleSnapshot`：两套精确 FP32 sigma 数组、共同 transition count、按显式 index 取 `(video_s, video_u, audio_s, audio_u)`，无可变 cursor。N 次 model evaluation 对应 N+1 grid points，双 grid 同长，有限、[0,1] 内、严格递减、起1终0，不能平均成一条 dt。
- 提供 build 默认 video shift12/audio shift3 的 N-step grids：CPU float32 linspace(1,0,N+1)，`shift*s/(1+(shift-1)*s)`；有限正 shift。若 float32 碰撞导致非严格递减，拒绝，不能偷偷改变 N。提供精确 snapshot 输入构造/读取即可，不造通用 scheduler registry。
- 拥有输入数组的快照，外部修改原数组或读取结果不得静默改 replay schedule；可以用简单不可变表示/防御复制。不要为这一步新造多层 hash/manifest 框架。index 越界/非法类型拒绝。不得把 sigma、模型 t=1-sigma 与 index 混用。
- 本项不新增 CPS score 或把 snapshot 塞进旧 Gaussian ABI；下一项 H3-M1b 将明确消费此合同，生产接线尚未完成必须如实标注。

## 必读来源/许可

读当前实际相关源码与本目录 task/plan。参考 UniRL `8cfbac17765004afb68fef08a4e011a4c8daf555`：`unirl/models/minimax_h3/diffusion.py::pack_dual_streams/unpack_dual_streams`、`packing.py`、`vendor/scheduling_minimax_h3.py`。先读 `reference/UniRL/LICENSE` 及文件头（Apache-2.0）。本项自行实现窄合同与数学，不整体复制 vendor、不 import reference。源码 docstring 标明所选语义来源；如确需移植实质源码及新增许可文件，先请求扩大路径并明确来源，不遗漏 notices。

## 验收

- 目标测试用真实新合同实现，不 mock packing/schedule；含不等 V/Cv/A/Ca 的逐元素 pack/unpack，batch 不串行、两模态均有梯度、无意 dtype/device 转换负例。
- 对默认10步检查11点、独立 shift12/3 的手算参考/中间点不同、按 index 末步双 u=0、准确 FP32 保持；非法 shape/宽度/空音频/nonfinite/不递减/不等长度/错误端点/越界拒绝；输入和返回值 mutation 不污染 snapshot。
- 新测试核验导入的 `visual_rl` 和新模块路径在当前 worktree，不把主 checkout 当被测代码。只用 tiny CPU tensors，无 GPU/网络/模型下载。
- 原样运行 plan.md 两条 CPU 命令；原七项仍7/7，H3目标集全部实跑无 skip/xfail；`git diff --check`。报告命令/exit/真实结果、允许文件 diff、当前 branch/HEAD。引擎结束后 snapshot SHA 未知就记 pending，不能伪造。

## 非目标与审查重点

无 model loading/LoRA/FSDP2/registry/recipe/reward/anchor/checkpoint/训练 loop；不因本项通过宣称 MVP 完成。Simplifier A 检查这是最小可用合同而非通用多模态框架，存储顺序/双 schedule/所有权与数值边界正确，原七项未改。返回 approved 或具体 changes；changes 必须回 Coding A 修复后两集重测。
