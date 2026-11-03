# Coder round 003 — P1 owner内有界解码/评分双缓冲

## 范围与恢复核对

按当前 planner/PLAN.md，仅实现一个单元：默认关闭、仅 dual_flash/paired_4x8 的单owner四样本流水。不跨owner/rank排队，不缓存VAE，不改FSDP/HPS服务/客户端或科研profile/transaction/保存恢复数学。

初始起点315d71a。此前长命令误将数据环境变量写成解释器路径，被白名单在执行前拒绝；没有执行该命令。按主会话保存的 RECOVERY-003.md，正确门禁当时为5 passed/2 failed（43.26s），不是成功。源码已封存 recovery-003.patch 并恢复HEAD。恢复本步骤时HEAD b704adc，实际 status仅原未跟踪.pi/、diff为空；成功做一次 git apply --check 后仅一次 git apply，未重复恢复。修复两处测试夹具：原TwoUpdateBackend CPU wrapper显式reward_pipeline=False；新reward回归使用真实IterationIdentity。恢复记录已标resolved，保留历史失败原文/补丁，不amend或混淆快照为功能提交。

## 实现

- `h3_decode.py`：单次handoff把完成blocking D2H的detached CPU FP32 video/audio原storage交给H3OwnedAVMedia；不clone、重编码、resize或改dtype/metadata。原media立即失效；旧lease仍正常退出并释放decoder锁；owned close幂等、关闭后访问拒绝，不允许二次移交。不能撤销消费者刻意保存的tensor alias，未作此承诺。
- 新 `h3_reward_pipeline.py`：单producer（原aux owner工作线程）/单CPU consumer。Queue固定1，Semaphore固定2且decode前获取，直到score/观看副本结束及close后释放；包括正在decode、排队、正在消费，总在途不超过2。只保留标量rows，不建整轮媒体/future列表。
- FIFO packet冻结ordinal/sample_id/group/member/prompt/selected_index；producer原顺序decode，consumer原score_one一次/样本。完结/失败信号与队列解耦，可取消轮询等待，无满队列sentinel死锁。任一失败取消新工作，join已进行的消费后排空关闭；不返回partial rows、不重试POST。首个业务异常先于cleanup记录，后续cleanup附note。
- owner工作启动consumer前，用原 `_admit_host_memory(decode_bytes + media_bytes)` 保留24GiB reserve做保守host准入；单payload仍受media预算、decoder原workspace和40GiB/16GiB GPU安全门原样保留。
- `h3_dual_stages.py`：默认仍逐条串行decode→score，流水只在原aux_turns内部开启；同位置真实coordinator gate协商bool模式、固定容量及media/decode预算；mode/layout本地错误也经原_checked门。原8个owner顺序/更新屏障不变，前后本地behavior只读复核，不从CPU消费者调用PG/_version。原score_one/observed_score/124帧/top37/导出调用保留。
- `h3_memory.py`：只给新consumer捕获reward父上下文，thread-local独立stack；reward_score/video_save记录cpu_span_begin/end、sample/thread/parent、壁钟/异常、CPU RSS。CPU事件标记observation=cpu-consumer、counters=None，不调用CUDA snapshot/reset/synchronize、不混入共享GPU stack。写入仍使用原锁、seq、budget及flush/fsync，日志错误失败关闭。默认GPU日志/重置前持久化/high-water逻辑不改，不持锁跨HTTP。
- `h3_lifecycle.py`：strict-bool默认False，compile/bind拒绝非paired Flash，传原预算到实际stages。`launch_h3_resolution_memory.py`新增默认off的--reward-pipeline，保留原fresh两步/cadence5/geometry/graph/weight/monitor/media所有门禁。该flag不是GPU执行授权，也不是已观测独立HPS设备的证明。

## 实际CPU回归

保持七个原函数/所有原断言，不skip/参数化/替换套件。P1主要扩在原manifest函数内；原two-update wrapper仅补显式默认flag。

1. 真实tiny CPU decoder：原单lease锁仍拒绝第二lease；handoff指针、内容通路、metadata不变，原lease失效/重复handoff拒绝、前owned活跃时可下一decode、consumer异常/幂等close后weakref释放，terminal不变。
2. Event/许可/Queue spy（无sleep速度对比）：第一score阻塞时第二decode/入队真实完成；第三许可已尝试但第三decode尚未开始。验证Queue1/total2、FIFO四个有限float、每样本一次，完成无消费者残留及合法owner tensor引用。
3. decode2/transfer2、score1/nonfinite/saver/cleanup、decode+cleanup多重失败；队列满且producer等许可时取消，已在进行score可释放后join，失败原样传播、无partial rows/新POST，owned全部关闭。额外验证真实pipeline日志字节预算失败、payload超限、host不足在启动前拒绝；没有遗留consumer。
4. 实际 H3DualIterationStages.reward→helper→原H3HPSReward/JPEG/score验证→observed_score/saver，仅HTTP/重模型/协调为CPU叶子。相同四条原始tensor串行/流水完整请求字节（含全部124 JPEG、sample/prompt）相等，top37契约与有限标量相等、导出看到原storage、RNG（Python/NumPy/Torch CPU）与terminal不变。CPUfake transport不是远端模型数值验证。
5. 真实H3Coordination.gate及receipt解码，在peer7分别伪造mode/预算不一致，gate poisoned且aux未进入。strict-bool/模式错误、实际backend.bind传参、CLI --help在任何Git/GPU入口前退出。实际GRPOTrainer驱动原dual reward时真实observed saver失败，abort发生，next_optimizer_step仍0，不到credit/update或下一owner，consumer已join。
6. 双线程交错：阻塞CPU score期间GPU vae span可完整进入/退出，CPU再嵌套video_save，无stack/sample污染；消费者不调用任何CUDA counters/reset/sync。原GPU durable-before-reset/high-water/异常/字节预算测试完整保留并通过。

恢复后的固定原七项短入口 `bash workflow-workspace/2026-11-03-h3-training-speed-001/inputs/run-cpu-gate.sh` 实跑依次7 passed（55.22s、50.52s、52.32s）；最终代码（含真实trainer失败回归）再次 **7 passed, 3 warnings in 51.68s**。三条仍为原DCP单进程保存/加载及TypedStorage警告。当前worktree导入断言、真实Git/config证明、G0/I1原回归全部保留并通过。

`git diff --check` 空输出通过。实际最新源码与diff stat检查范围仅计划中的七个代码/测试文件（包括新helper），另有本角色恢复标记/报告。原未跟踪.pi/保留，不读内部/不清理/不stage；不宣称整树clean。原子提交后SHA单独记入round-003-commit.md。

## 未测收益与硬边界

已读TASK与保存MEMORY-REPORT：独立HPS物理4为历史事实，不是本轮实时观测；训练获准8卡、Gen2x4/无NVLink、reserved不等于active，不依据GPU型号记忆假设互联或驻留余量。

沿用Planner条件估算：假设D45s/H50s（非分别实测），8个四样本owner各自fill/drain辅助主体3040→1960s，可能省18min/轮。旧前两段93.918min分母**不含优化/保存**时约1.24×；旧127.73min分母**含采样/评分/优化/校验/保存、不含初始化**时约1.16×。不叠加I1未测收益，不声称51→27min/2倍已达成。CPU仅证明逻辑重叠/边界，未测GPU/IO/CPU竞争/服务/导出端到端收益。

480×320原AV两payload约459763200bytes（438.5MiB）仅为媒体storage上界；另有decoder工作集、VAE权重/JPEG/JSON/ffmpeg/internal workspace，不是整个RSS上界。consumer只做CPU与原远端调用，未改变原输入/评分/精度/随机调用。

线程不可强杀：取消不关闭仍使用中的client/media，不谎称撤销在途HTTP/ffmpeg；正常可控失败等待原操作超时/结束再join。不可终止外部/kernel挂死仍由既有aux3600s deadline poison并整attempt退出，不能正常续更/当作排空。没有新增finally collective。

无GPU/SSH/实时health或score请求/服务变更/部署/真实训练，Euler B继续暂停。开启须另批准实验卡核查最终SHA、独立HPS实际GPU/资源/身份、原协议下两更新/保存恢复和逐帧/top37数值及完整含优化保存墙钟。VAE复用、跨owner流水、FSDP仍留后续Planner选择。本轮交Simplifier独立复审，不宣称工作流结束。
