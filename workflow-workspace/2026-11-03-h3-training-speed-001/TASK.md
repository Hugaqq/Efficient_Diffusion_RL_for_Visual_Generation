# H3同协议训练提速：Planner → Coder → Simplifier 循环

## 用户授权
用户要求在VPS部署真实三角色持久工作流：Planner结合硬件/代码查找并行或节时机会并给计划，Coder实现，Simplifier在保持功能的前提下简化，然后返回Planner。Planner认为没有明确可实施的显著节时修改才正常结束。**原子commit**。最新用户提供的成本分析是规划依据，不是实测提速证据。

原Euler推理B继续paused，禁止恢复；旧两步任务complete，禁止重跑旧训练。当前授权为代码审计、修改、CPU验证、Git提交；不从性能目标文字推导额外训练更新授权。本阶段GPU仅允许读取已存证据，无训练、推理、HPS请求、占卡benchmark或服务变更。最终要声称端到端提速，必须另按实验卡对最终候选进行同协议真实验证；提交验收方案并保留“性能待实测”，不能以CPU结果冒充GPU收益。

## 工作区/源码
- 原始clean生产SHA：3db2a05e1c464bd8cbd97c0290a094d869a9d1fd。
- 旧源码 `/home/user/projects/h3-visual-rl/.scratch/h3-direct-20261028-001` 是linked worktree，不满足workflow delivery入口。保留不改，在VPS建立独立干净主checkout `/home/user/projects/h3-visual-rl/.scratch/h3-speed-main-20261103-001`，分支 `perf/h3-training-speed-20261103-001`，从同SHA开始。
- 正式工具工作流从该主checkout根启动，角色在工具管理worktree顺序执行；不要手工移动/改写`.pi/workflows`，不要stash/reset他人更改，不改旧framecode分支或历史。
- 每轮报告写 `workflow-workspace/2026-11-03-h3-training-speed-001/{planner,coder,simplifier}/`。只三种角色，不另启动隐式后台coding agent。

## 固定科学协议，不得用改变问题制造提速
480×320，124帧24fps，10步，前5选1随机转移其余Euler，4prompt×8=32，lr1e-4，LoRA32/64；原始模型/编码器/VAE及精度、HPSv3逐帧124/top37、原数据与全局优势/Flash修正/梯度缩放、optimizer次数与保存恢复语义不变。保持每样本单转移重算/反向释放，允许累积梯度不留整链图。
不得减少评分帧、减少样本/步数、改变分辨率、蒸馏/量化、换reward、跳过核验或吞掉异常来包装加速。流水线/批处理需保证样本ID绑定、原始张量输入、评分数值与排序容差明确、同一冻结行为策略、参数更新边界和确定性随机流。异步必须有界排队、背压、错误传播、取消、排空与资源释放；不得保留32条FP32全视频在RAM。

## 硬件与已测基线（inputs供核查）
- 10张CMP170HX 64GiB，获准训练8张（物理0/1/2/3/6/7/8/9），独立HPS物理4；GPU5不能假定获准独占。PCIe Gen2×4，无NVLink，PIX/PHB/SYS跨NUMA。
- 旧同协议第二轮：采样约42.6min、解码评分约50.7min、优化阶段约13min，外加初始化/审计/保存。前两项约93min不是完整训练轮；整个两步监控约294.31min，不能将所有开销平均为稳态轮成本。
- allocated全程峰19.71GiB（VAE），reserved约62.87GiB，整卡采样最大63.39GiB。reserved不等于活跃张量，也不能据allocated低保证任意驻留/并发安全。
- 同配置标准全GPU DDP最低常驻64.21GiB/卡（不含激活），不能直接切。D2四副本混合分片静态32.91GiB只是候选估算；框架1D DTensor、梯度归约和checkpoint兼容需先审核，不能一行改mesh便称已验证。

## 用户成本估算与优先顺序
1. 复用VAE，核对每样本约8s加载/37s解码的实际测量边界；仅复用估算约省4min/轮，不是倍数提升。
2. 解码与HPS流水：评分仍约50s/样本时，辅助段理论约51→27min，但需真实确认设备资源、传输/排队/序列顺序；前两大阶段约93→70min（1.3×），加上优化/保存后完整收益更低。
3. 分解42.6min采样：算子计算、FSDP聚参、H2D/D2H、CPU等待、同步和重复处理。不能把Gen2理论带宽当实测；先从代码与现有事件找证据，必要时实现可关闭的短profile入口供后续获准测试。
4. 分解HPS的50s：预处理、模型搬运、排队、前向等。审查健康服务是否每请求重载/卸载；独占GPU上的常驻与批处理必须保留准入、offload回退及原评分数学，不直接动服务。
5. 其他无语义变化的显著冗余：重复metadata/embedding/模型身份校验、重复解码、IO与通信同步。不得删掉安全与完整性门，缓存必须定义失效与身份检查。

目标先争取约1.3–1.5×，约2×端到端是待验证目标，2–3×仅条件假设；不承诺5×/10×。采用Amdahl口径逐项报告：实测基线、计算假设、代码已完成、CPU已验证、GPU未验证。高收益前置的小改动可以先做，但不要为凑循环做纯微优化。

## 原子commit与测试
- Planner每轮只选一个独立逻辑优化单元，明确范围、预期省时、协议不变量、故障场景和验收项。维护候选清单、完成项、暂缓项及理由。
- Coder将该优化及对应回归覆盖做成一个可独立理解/回退的commit；功能和必要测试一起提交。不同无关优化不得塞进同一commit。不能提交已知失败或未完成代码。
- Simplifier先审查，再仅做有明确维护收益且功能等价的简化；有改动则单独原子commit，无改动不得造空commit。不把正确性/取消/资源生命周期保护“简化”掉。
- 每次代码提交前跑**原七项** `tests/test_flash_collection_runtime.py`，7个原函数保留、不skip/替换/新增函数来改变数量，可在原函数中扩展检查。CPU假计数只验证逻辑，不证明真GPU性能。
- 固定CPU gate：`PATH=/home/user/projects/h3-visual-rl/.scratch/h3-media-tools:/usr/local/bin:/usr/bin:/bin VISUALRL_FLASH_AUTHOR_DATA=/home/user/projects/h3-visual-rl/.scratch/flash-grpo/source/dataset/video PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/user/projects/h3-visual-rl/.scratch/h3-cpu-venv/bin/python -m pytest -q tests/test_flash_collection_runtime.py`。另做`git diff --check`，确认导入当前worktree而非旧editable checkout。
- workflow工具会保存必要快照；Coder/Simplifier在通过门禁后主动做描述清晰的commit。不要amend/rebase/squash，不force push，不改变工具管理分支。报告每次commit SHA、测试实际输出与未验证项。
- 最终integration重新执行原七项与diff检查并保持树clean，安全发布到本次perf分支；不自动部署GPU。通过后才可将该perf分支push到现有origin供审核，禁止改旧分支/主线或推未通过验证的代码。

## 三角色闭环与结束
Planner首次审代码与证据，写PLAN与候选表后返回implement；Coder实现并验证后到Simplifier；Simplifier实现等价简化/反馈后回Planner。Planner必须读取真实diff/commit/test证据，不能只信上一角色总结。
- 有未解决正确性问题：下一轮优先修复，不判完成。
- 仍有明确可实施的显著节时项：继续下一轮。
- 无进一步有明确依据的代码修改：可结束**代码交付阶段**，给出最终优化清单、原子commit映射、CPU证据、GPU验收计划及所有未测收益。不得称硬件最优/已达2倍/全部可能性耗尽。
- 如果下一项必须先有真实GPU测量才能合理决定，Planner可waiting_external，列出最小测量卡；普通代码/测试错误不伪装外部等待。
- 预算保护：最多40个角色步骤、8小时活动时间；触顶是暂停/待续，不是优化完成，不自动扩预算。

代码交付结束后等待主会话核查。原推理B保持暂停，不将它当性能benchmark恢复。
