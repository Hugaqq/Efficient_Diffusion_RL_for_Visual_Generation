# Simplifier round 003 — P1 独立审查

## 结论/提交

P1 当前实现符合本轮窄范围合同，未发现必须修复的正确性缺口，亦无明确值得新增提交的等价简化。**不改源码/测试，不创建 Simplifier commit**。容量许可、显式 handoff、错误记录和后置 join/drain 均是必要安全结构，不为减少行数删除。交 Planner 再审核，非工作流结束声明。

实际 git log 确认功能提交 **9e183bb**（父 b704adc），当前 HEAD **7f57c3094999862e5c75452bb0bc77ca6c7529f2** 为工作流快照。已执行相对生产基线的授权源码累计 diff；输出超过工具50KB界限而截断，未访问其外部临时日志。通过 read 补读当前全部新 helper、memory 模块、dual reward、handoff/lease、backend/launcher接线及测试回归，而非只采信 Coder 报告。G0/I1 仍在累计 diff 中，不将其误算为本轮新增。

## 输入与审查依据

重读 TASK.md、planner/PLAN.md、baseline-MEMORY-REPORT.md 阶段与测量边界、Coder round-003/commit 收据，以及 inputs/run-cpu-gate.sh。存证说明 HPS 历史为独立物理4，非实时设备证明；不按 GPU 型号猜互联或由 reserved 推断可并发驻留余量，标准 DDP 静态超64GiB。无硬件探测/外部请求。

实际源码额外查阅 H3Coordination.aux_turns、原 observed_score/save_media、H3HPSReward.score_one、decoder blocking D2H/GPU门，确认新线程接入的真实上下游。一次按错误目录读取 h3_hps.py 返回 ENOENT，随后用 find 定位 algorithms/rewards/h3_hps.py 并读取，未修改路径/环境或依赖。

## 协议与资源检查

- 默认False、strict bool；开启只允许 dual_flash/paired_4x8，构造/bind/reward 层有准入。原同位置 gate 对模式及两个预算协商，跨rank不一致在 aux 前失败。CLI 默认off，validate_fresh/图/权重/monitor/两步审计保持；flag不是GPU执行授权。
- producer 仍为原 aux owner 工作线程，consumer 不调用 PG/模型；aux 仍按8个owner顺序，work 返回须等待消费结束，之后才进入下一owner、credit/update。behavior 在流水前后只读复核，未调用带collective的 _version。
- Queue(maxsize=1) 加 Semaphore(2)，**decode前**获取许可；直到score/导出完成并关闭后释放，包含在decode、队列、消费中的媒体总数最多2。启动consumer前 host admission 使用 decode_bytes+media_bytes，保留原24GiB reserve；decoder原payload/workspace、40GiB准入、16GiB余量及monitor未变。两payload不是完整RSS上界，另有权重、工作区、JPEG/JSON/编码scratch。
- handoff 验证detached CPU FP32，移动原video/audio storage、metadata不重编码/clone；原media失效，原lease退出释放decoder锁；owned不能再次handoff，close幂等。Python故意保留的外部tensor alias不具备可撤销保证，代码/doc未作此承诺。blocking D2H仍由原GPU decoder完成。
- frozen RewardSample 绑定ordinal/sample/group/member/prompt/selected index；FIFO消费只调用原score_one，保留完整124帧/JPEG/top37/manifest/identity/有限float检查，原observed saver在owned活跃期运行。terminal clone及生产者顺序仍保留，不新增随机调用；原模型/dtype/精度/更新数学无改动。CPU请求字节等价不是远端模型等价证明。

## 错误、取消和日志检查

- 完成/取消Event与队列分离；许可和入/取队用可取消轮询，避免满队列sentinel阻塞。producer错误在close前记录，consumer错误记录后关闭当前owner并释放许可；主路径join后关闭未消费媒体，不返回partial rows、不重试POST。
- 不能撤销已经开始的HTTP/ffmpeg；取消等待该原操作结束/超时，不并发close仍使用的client/media。既有aux 3600s deadline仍是整attempt失败边界，线程不可强杀；不将永久挂起称为正常排空，也不能带着旧线程继续下一更新。
- 局部CPU日志scope捕获reward父上下文，独立thread-local stack；score/video_save事件标记cpu-consumer、counters=None，保留sample/thread/父context/壁钟/异常/RSS，不调用CUDA snapshot/reset/synchronize。默认GPU日志逻辑未改；所有写仍走journal锁/seq/byte bound/flush/fsync，锁不跨整个score操作。CPU日志失败经pipeline传播，原业务异常优先于后续cleanup。
- 没有发现适合安全删除的重复防护；helper与stage两层有限float门分别守住局部能力和原回调合同，保留。

## 独立真实CPU门禁

先读取短入口确认其内容就是原固定环境和原七项pytest命令，再实际执行：

`bash workflow-workspace/2026-11-03-h3-training-speed-001/inputs/run-cpu-gate.sh`

结果：**7 passed, 3 warnings in 51.79s**。三条仍为DCP单进程保存/加载及TypedStorage。G0/I1与当前worktree导入断言、真实Git/config证明均包含于原套件通过结果。`git diff --check` 实际空输出通过。

复核原函数内新增回归：真实tiny decoder锁/handoff/storage/weakref；Event控制真实重叠、Queue1/总许可2、第三decode等待；decode/transfer/score/nonfinite/saver/cleanup多故障、取消等待在途操作、join及关闭；真实dual reward→原HPS/JPEG/validator/observed saver的完整请求字节/标量/RNG/terminal一致；真实gate拒绝peer配置差异；实际trainer失败不进credit/update/下一owner；双线程日志跨序退出及字节预算失败；backend/CLI离线接线与host不足拒绝。未降低函数数量、skip或删断言。

本轮未创建代码commit，无需写COMMIT_MESSAGE.txt。初始status仅原未跟踪.pi/；仅新增本报告，未stage任何文件，未读取/清理/提交.pi/，不宣称整树clean。最终status再用命令核实。

## 交Planner的剩余事项

1. 独立复审9e183bb及本轮门禁，继续选择下一单一优化；VAE复用、跨owner流水、FSDP候选不由本角色宣布耗尽。
2. 默认不启用P1。真实独立HPS设备、资源竞争、GPU逐帧数值/top37排序容差、两更新保存恢复和完整端到端耗时需新批准实验卡。当前CPU只证明逻辑与本地字节合同，没有实测speedup，本报告不新增耗时估算。
3. 最终integration/perf发布仍未执行；本角色无GPU/SSH/服务/真实HPS请求/训练/推理/部署，原Euler B保持暂停。
