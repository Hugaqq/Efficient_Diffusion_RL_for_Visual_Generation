# Planner round-003 — I1独立验收与P1规划

## I1实际diff/commit审查

当前HEAD `266a25c43d5b7c76c0caed63da0ca737b35484c1`。实际`git log -12 --oneline`确认Coder原子commit `a211cbf perf(h3): share fresh live validation at checkpoint boundaries`，父`47453a2`，后续`7b1a49f/266a25c`是工作流快照。`git show --stat --oneline HEAD`仅36行Simplifier报告，无Simplifier代码提交。

实读Coder round-002/commit收据、Simplifier round-002；实际执行相对生产3db2a05的累计授权源码diff，并直接读取最新binding/lifecycle：
- G0保持原测试夹具修复。
- I1仅binding/lifecycle和原test_approved140_manifest_counts内必要回归。无transaction/schema/配置/依赖diff。
- checkpoint_snapshot用独立deepcopy保存tail/source，执行本次完整validate_live后构建原progress/proof/data/RNG；后置所有权、tail/source与next_batch复核。旧boundary委托仍fresh完整自校验。无缓存或verified绕过。
- lifecycle在原保存分支/phase/span中消费该次snapshot，未改变cadence、pause、first/terminal或restore/final文件门。
- 回归实际驱动真实CPU tiny binding两轮、legacy/同边界重验、状态/RNG只读、非法边界/grad/counter、实际VAE文件漂移、校验后source推进/真实reservation及真实lifecycle两次保存/失败不保存。局部CUDA/host/HTTP叶子替换不是生产检查松绑。

结论：I1符合计划，未发现需要修复的正确性缺口。接受Simplifier不改代码/不造commit的决定。提交前测试时序及5文件暂存统计来自Coder已存收据，Planner只运行了允许的log/累计diff/HEAD stat及自身门禁，不擅自扩大Git命令。

## Planner真实CPU检查

固定原七项实际：**7 passed, 3 warnings in 48.32s**。原DCP单进程save/load及TypedStorage三警告，无skip。当前worktree导入断言与真实Git/config证明随整套通过。Coder最终提交前53.30s、Simplifier独立52.76s的7/7已读，时间差不是GPU或训练提速。

起始status仅`?? .pi/`；不访问/清理/提交。Planner仅维护自己的计划报告。写完后`git diff --check`空输出通过；最终status仅Planner的PLAN/CANDIDATES修改、round-003新报告和原未跟踪.pi/，无源码修改，不宣称clean。

## 下一项真实源码/硬件审计

重新读取TASK、baseline-MEMORY-REPORT、最新候选表；GPU事实仍仅存证，没有实测设备/拓扑/服务查询。已存训练8卡与独立HPS物理4支持规划但不能冒充实时绑定证明；GPU5未获独占授权。

重新读取：
- h3.py辅助decoder与标准decode lease；h3_decode.py完整实现。单lease锁跨消费，模型逐视频/audio加载并释放、返回CPU FP32，40GiB准入/16GiB margin。
- h3_dual_stages.reward与h3_coordination.aux_turns。现有owner四样本serial decode→score，8个owner顺序、主线程heartbeat、单CUDA工作stream。
- h3_memory.py完整实现。发现Journal.stack进程共享，未来跨线程reward_score/video_save与vae_decode交错会发生LIFO错误/峰重置归因问题，必须作为P1必要部分设计，不可无脑套executor。
- 原H3HPSReward和Hpsv3VideoClient：manifest/JPEG/124帧/top37与响应身份校验保留，HTTP同步、显式timeout、无POST重试；observed_score导出仍须在媒体活跃期间。
- 两步launcher backend/原exporter.span及lifecycle选项接线、现有dual stage CPU fixture。未运行这些GPU入口。
- find定位并只读本worktree保存的参考UniRL两份VAE源码（workflow-workspace/2026-10-27-h3-8gpu-mvp-dev-001/reference/.../vendor/autoencoder_kl_minimax_h3{,_audio}.py）。参考decode无采样、video默认tiling、audio weight_norm/filter；未导入/复制vendor代码，不把参考loader当作实际安装版无RNG证明，不执行历史任务命令。

## 003决策：P1单owner有界双缓冲

详见PLAN.md：默认关闭的opt-in；固定一个producer在原aux worker/stream解码，一个CPU consumer按FIFO调用原score_one/exporter；显式媒体storage所有权移交，Queue1加decode前总许可2，所有错误取消/排空/join后交原owner-end错误门。8个owner串行边界保留，credit/update前必须全部完成。原GPU安全门、VAE加载顺序、reward/精度/RNG/数学不变。CPU消费者必须隔离日志stack且不reset/synchronize GPU峰，保留lossless原门。

不先缓存VAE：8s加载未拆磁盘/构造/H2D，pair缓存需新驻留与构造RNG/身份失效合同；本轮流水直接复用现有decode调用，不作模型常驻假设。V1仍开放，非永久排除。

收益口径修正：本轮每四样本fill/drain，而非全32连续流水。D45/H50条件假设：8*(D+H+3max)=1960s vs3040s，理想省18min/轮；用旧保存轮完整含优化/校验/保存127.73min分母近似1.16×；前两段不含优化/保存93.918min分母近似1.24×。这些不是实际性能，不叠I1估算。全32连续流水约27min是P2后续假设，本轮不声称已实现。

## 待办与权限

返回implement，Coder按窄文件范围做一个功能+必要回归原子commit，先原七项7/7及diff检查，Simplifier独立复审。仍未执行GPU/SSH/HTTP/训练/推理/部署/服务变更，Euler-B保持暂停。

最终integration/perf分支发布、另行批准最终两更新验证尚未发生；V1/P2/FSDP等仍开放，本轮不写FINAL、不宣称2倍或优化耗尽。
