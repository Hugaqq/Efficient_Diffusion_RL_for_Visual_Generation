# Round 004 — F0：默认关闭的单rollout-cell CPU/CUDA profile入口

## 决策与上轮复审

本轮唯一单元是采样/FSDP优化的**必要测量前置F0**，不是mesh修改，也不宣称直接节时。返回implement。

P1 `9e183bb`已独立审核真实累计diff、当前全部helper/memory/dual reward/handoff/backend/launcher及相关回归。Simplifier无必要修改/无commit。当前HEAD `f4ef71e4247f701338f7359f3c6968213921a1a3`；Planner固定原七项 **7 passed, 3 warnings in 51.64s**，无发现的P1正确性缺口，流水仍默认off。

## 收益依据与为何现在测量

存证第二轮rollout42.600min、40个rank-local root前向，约63.9s/前向（仅粗平均，含gate/转移/日志等）；anchor254.950s/4、current256.249s/4、backward265.359s/4，同量级。42.6min占127.73min保存轮约33.4%（分母**含优化/校验/保存、不含初始化等**），占前两段93.918min约45.4%（**不含优化/保存**）。没有算子/NCCL/搬运/host时间拆分。

已存每次遍历allgather接收约56.49GB只是字节估算，Gen2×4不是实测有效带宽，不能据此声称通信占比；标准DDP静态64.21GiB已超卡容量，D2需要1D norm/归约/checkpoint架构审查，不盲改mesh。

VAE优先项仍开放，但P1后不能直接叠加原串行省192s的估计：在D45/H50的纯假设下，owner内复用仅加快第2..4次decode、不改变首个decode，`Dfirst+4H`临界路径甚至可能不变。实际D/H/CPU竞争均未测，不能将此假设当作VAE永久无收益的证据。先交付F0以便获准后合理定位主采样成本；P1、VAE和HPS已有细分日志也应同时收集。

F0本身没有预计speedup，开启会增加采集/同步/导出开销；profile窗口耗时不是端到端性能结果。当前不启动任何GPU或额外更新。

## 范围（仅这一个观测单元）

允许：
1. 新 `visual_rl/runtime/h3_rollout_profile.py`：窄的one-shot recorder与受控产物发布。
2. `visual_rl/algorithms/trainer/h3_dual_rollout.py`：可选cell scope hook，默认None；只围住原单cell代码，不改变任何运算/循环/异常门。
3. `visual_rl/algorithms/trainer/h3_dual_stages.py`：默认None的profile能力传递到真实collect；与anchor/train/reward解耦。
4. `scripts/launch_h3_resolution_memory.py`：唯一新开关`--profile-rollout`，默认False；安装hook、配置一致/能力准入与产物收据。保留P1/原两步所有门禁。
5. `tests/test_flash_collection_runtime.py`：仅原七函数内部必要回归，不增删/skip/替换/参数化函数。
6. Coder自己round-004/commit材料。

不改FSDP/mesh、binding安全校验、VAE/HPS/依赖、科研profile/schema/精度、memory journal reset逻辑、checkpoint/optimizer或任务授权。不要建立新runner/调度器，不运行torchrun/训练/profile命令。

## 固定one-shot窗口与默认无影响

- CLI开关strict bool；off时不实例化/启动profiler，不创建profile目录，不要求CUPTI，不改变模型forward次数。不得用“CPU fallback”伪装GPU profile可用。
- on只用于既有validate_fresh约束的两更新入口：**optimizer_step=1（第二次更新，0-based）、local row=1（第二条样本）、transition index=0，一次cell/rank，共8个窗口**。固定版本/目标，无任意长schedule/重复采样或新增更新。第一轮40前向及第二轮第一条样本用于自然warmup，不新增warmup调用。
- hook接口接收小元数据（optimizer_step,row,index,sample_id,selected_index和原gate能力），不接收/保存latent、prediction、model/optimizer。非目标cell仅透明scope；命中一次后不能重入/重放；每rank只采一个cell。
- scope覆盖原forward前gate、prepared预测、原transition及结果gate，保留既有memory span/同步。结束前`prediction`按原逻辑释放。原row起始latent H2D在该cell scope外，**必须在metadata注明未覆盖**，不能把这个窗口声称为整个rollout。
- 采用public torch.profiler CPU+CUDA活动，关闭record_shapes/profile_memory/with_stack/with_flops等保留额外对象/昂贵采集选项；无新autocast、RNG、参数cast、empty_cache或额外forward/backward。
- 导出原始Chrome trace以观察真实CPU op、CUDA kernel、memcpy与NCCL重叠。不把CPU enqueue时长当GPU执行时间，不把所有NCCL归FSDP（现有gate本身也用NCCL），不把CPU与CUDA重叠时长简单求和成墙钟。

## rank一致、失败门及线程边界

- 在首个model调用前，TwoUpdateBackend所有rank协商规范化的profile开关/固定窗口/预算版本；任何差异在开始训练前失败。可直接使用原_exchange，**不要在要求local-only的_phase action内偷偷调用未约定的PG collectives**；局部文件初始化仍走原错误交换习惯。
- 活动支持/初始化/启动失败应在目标forward前通过相同位置原gate传播，不能让一rank退出、其它直接进FSDP。无CUDA activity/无所需public API时fail closed，不自动降CPU或忽略warning当成功。
- 只在main训练线程启动/停止CUDA profiler；不从P1 CPU消费者或文件worker调用CUDA。成功body后的stop失败经相同位置gate传播；模型/collective本身已抛错时按原整attempt teardown，不加可能死锁的救援collective。
- 原model/dynamics异常必须保留为主错误；profiler __exit__/stop/文件清理失败可附note，不能覆盖、吞掉、重试原model。body成功但采集/发布失败要显式抛错，不能生成虚假成功receipt或继续额外更新。
- profile关闭后的trace序列化/流式hash/文件检查可用已有file_phase主线程heartbeat/CPU文件worker，再通过原错误gate交换状态。只有**已停止且确认是CPU序列化的export**可交文件worker；stop/finalize仍在main。不得将CUDA/PG action放进file_phase worker。保留3600s整attempt deadline，无外部等待伪装。

## 产物/预算/身份

- 输出只能是本次新run输出下固定`rollout-profile/rank-N/`，不接受任意外部目录。不开启不落盘。拒绝symlink/已有目标，不覆盖历史trace；每rank目标包含固定update/row/index。
- 每rank成功发布最多1份trace（建议发布上限128MiB）及1份小型JSON receipt（建议上限64KiB）；流式SHA256，不read_bytes大trace。使用临时文件、校验、fsync/原子发布，无成功标志先于trace完整写完。
- PyTorch export_chrome_trace只能先写路径再检查size；**发布上限不是采集内存/临时磁盘峰的硬cap**。单cell、无shape/stack/tensor retention限制范围；实验卡需预留临时空间，导出过大则拒绝发布并清理自己已确认owned临时文件、整attempt报错，不静默截断。不能删除未知文件/改.gitignore掩盖输出。
- receipt含schema/version、final SHA（沿用launcher实际clean/expected guard）、rank/已admitted GPU UUID、固定窗口与实际sample_id/selected index、采集设置、单窗口instrumented wall时长、trace bytes/SHA、coverage与限制。不得保存prompt全文、媒体、weights、activation或完整tensor对象。
- 开启后的正式receipt应能区分CPU与CUDA活动确实被采集；支持API和实际事件检查不能被CPU-only结果冒充。CPU测试只替换不可用CUDA/CUPTI观察叶子并诚实标识，不把测试生成的CPU trace当生产CUDA证据。
- 到两步入口正常结束时必须检查每rank目标窗口恰好命中且receipt存在，开启但未命中应失败而非silent success；不开启不额外要求收据。此检查不新增训练或保存次数。

## 必要CPU测试（原七项内部）

1. real collect hook：用现有小CPU dual dynamics/predict/generators，profile off与on结果（records/old score/actions/terminals/choices）、每sample RNG generator状态和全局Python/NumPy/Torch CPU RNG相同，forward/gate/单转移选择次数不变。真实循环两轮或直接覆盖目标row/index，只有第二轮固定cell创建一次profiler，非目标不创建、不导出；anchor/train不被hook捕获。
2. 默认off：factory设为forbidden仍能执行真实默认collect与两步backend接线；不创建profile目录/CUDA探针。CLI --help含开关但在Git/GPU副作用前退出；strict-bool/unsupported能力/重复窗口/未命中均拒绝。
3. start/stop能力与rank配置：fake profiler叶子记录只在主线程/一次进入退出，真实gate/receipt或已有CPU transport leaf拒绝peer差异；启动失败不得进入predict。成功body后stop失败不得进入后续transition/update保存（按实际scope范围验证）；body抛原异常且stop也抛时保留原错误+note、只执行一次body、不能尝试重放。
4. 产物：CPU fake exporter或小型真实CPU profiler叶子执行实际文件发布/stream hash/receipt检查，校验rank/sample/window/SHA/设置；覆盖已有目标/symlink/超限/导出IO失败/receipt失败，保证无伪成功、不覆盖未知文件。不使用生产CPU fallback开关替代测试叶子。
5. 实际TwoUpdateBackend.bind安装正确hook、所有rank配置先协商，原PredictAudit/monitor/graph/frozen/Adam/媒体审计和reward_pipeline选项保持。原测试wrapper仍应通过；仅在自身CPU局部fixture补齐新可选配置字段，不放松生产门。
6. 原P1/G0/I1及七函数完整保留；不另跑新测试文件或全套，不用睡眠速度比较证明提速。

## 原子提交、验收及后续

提交前跑原七项（短入口内容已核对就是原固定环境/命令）：

```text
bash workflow-workspace/2026-11-03-h3-training-speed-001/inputs/run-cpu-gate.sh
```

长固定命令亦可，必须7/7；当前worktree导入断言保留、git diff --check通过、真实diff仅本单元。功能+必要回归同原子commit，建议 `feat(h3): add opt-in one-cell rollout profiling for communication analysis`。先测后commit，不amend/rebase/squash；Simplifier必要等价简化独立commit、同样先7/7，不造空commit。保留原未跟踪.pi/不访问/清理/stage，不写授权文件。

F0完成仅表示具备后续测量入口，不能声称NCCL瓶颈/优化收益已证实。新实验卡仍需单独授权最终SHA、独立HPS实际绑定、原8训练卡/物理4、同科研协议与新输出、恰好两更新/保存恢复；可在该授权两步中采本one-shot窗口，禁止额外warmup/旧run重跑。采样窗口instrumented耗时与完整含初始化/优化/校验/保存墙钟分别报告。CPU通过后Planner再决定下一代码项或最小测量卡，当前不写FINAL、不判全部优化耗尽，Euler-B保持暂停。
