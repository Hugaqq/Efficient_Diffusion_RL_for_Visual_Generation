# Planner round-006 — M0恢复、成本归因、H1实施计划

## 恢复与已有操作核对

用户/主会话明确恢复，并在本步骤直接提供M0完整32行、SUMMARY、四rank原始摘录和源文件SHA；新增GPU调用0。已接收分析，不要求再提供同一M0，也不因为VAE headers/拓扑/CUDA trace仍NA就停回原请求。

实际log/HEAD stat：当前`e5cfeb446bb20d1222a1e8df09516f4682572eee`的父为上轮已审`0eb2a5b`，新增仅Planner的PLAN/CANDIDATES/MEASUREMENT-CARD/round-005四文件快照；功能源码仍到`9b0a6b3`，初始status仅原未跟踪.pi/。因此沿用本会话上轮已实际执行的原七项 **7 passed, 3 warnings in 44.91s**及代码复审，不重复该门禁/巨大累计diff/旧恢复apply。新Coder代码commit之前仍须新实跑全部七项。

本轮重新实读TASK、当前完整serve_hpsv3_video.py、原wire health/score/aggregate验证、H3 HPS CPU fixtures与原测试入口，核对可实施路径，不仅依赖上一轮报告。没有读取用户给出的工作区外来源路径，其仅作为M0来源索引。

## 本轮实质分析

选用数据原样保存M0-ROWS.csv；完整来源指纹、提供的四rank/同sample HPS摘录对应行号和成本/反事实公式见M0-ANALYSIS.md。来源SHA来自交接，不伪称独立重hash480条原文。

独立归因：
- decode_full32条1459.611s；score_full1612.952s，合计3072.563s=51.209min，与旧宏段51.318min仅约6.52s边界差。父子不能再相加。
- VAE两个load共270.054s，均值8.439s；含CPU构造/读盘/H2D，不是纯磁盘。仅owner内删24次reload旧serial上界约197.290s=3.288min，驻留/offload/hash成本还未扣。
- HPS父评分1571.599s，其中onload386.598s＋offload404.194s=790.792s=13.180min，**约50.3%服务段在重复搬运**。每样本24.712s搬运，剩余24.400s是原处理/模型/其它开销残差，不是纯kernel；score_full与服务父段差1.292s也不能全称JPEG。
- 源码精确对应每POST `.to('cuda')` /finally `.to('cpu')`；模型其实CPU已长驻，无需重新加载/换模型。这个重复跨Gen2链路搬运已有直接证据，缺完整拓扑或VAE headers不阻碍写opt-in H1。
- M0 HPS UUID与独立物理4历史绑定、上卡后allocated约15.50GiB及报告服务峰15.73GiB支持量级评估，不是实时独占/free证明；原40GiB/16GiB门与完整实际UUID必须运行时保留，不借训练allocated峰猜驻留安全。

## 与已实施P1的收益相互作用

以旧serial D/H固定且假设无重叠争用的反事实：P1各owner `Dfirst+sumH`，32主体约1984.139s=33.069min；H1若warm移除搬运后H'低于D，各owner `sumD+H'last`，主体约1663.776s=27.730min。**H1增量约5.339min/轮，而不是把13.180min全加到P1收益**。

旧完整保存轮约127.73min（含优化/完整校验/保存，不含初始化等）；上述模型约P1后109.59→加H1后104.25min、增量约1.05×。前两段不含优化/保存93.918min条件约70.44min。未实测新代码，不叠I1估算，不报2倍已达成。

H1后decode预计成为瓶颈，VAE复用重新是高优先约3.29min上界候选；P2跨owner理论额外可隐藏约2.99min尾部。均后续单元，不夹带。

## 唯一选项H1与边界

PLAN.md已改为H1 implement：默认off的专卡单UUID单模型GPU复用，保留原冷/暖实际free准入和16GiB监测、低余量真实offload再cold回退、正常idle60s卸载、main shutdown finally/幂等close、失败/monitor释放和不可确认状态不ready。健康manifest策略/代码生成新identity，不能伪装旧服务/删核验。CPU测试只替重模型/CUDA/socket叶子，真实评分/handler/原124帧/top37验证与异常清理必须执行。

仅serve_hpsv3_video.py和原七项测试内部必要回归＋Coder报告，预期一个独立原子commit；Simplifier再独立审核。原服务原样不动，**不运行服务、不请求HPS、不部署**；新private instance/端口/manifest/资源必须未来另批，原M1无服务变更范围不能套用来启H1。

MEASUREMENT-CARD已标M0恢复条件满足而非仍等待。候选表更新H1本轮选择及其余缺项，未改TASK/输入授权/其它角色报告/依赖/ignore。当前没有任何GPU新调用，原Euler-B继续暂停，最终integration/perf发布/最终两更新速度和GPU严格等价验收仍未执行。

## 门禁/报告状态

本步无源码编辑、无代码commit，不把保存的44.91s冒充本步新跑结果。写完后实际git diff --check空输出通过；diff stat仅Planner的PLAN/CANDIDATES/MEASUREMENT-CARD，status另有新M0-ROWS/M0-ANALYSIS/round-006及原.pi/，无训练源码修改。保留原.pi/不读取内部/清理/stage，不宣称整树clean。

返回implement；没有理由再等待同一M0、没有FINAL或耗尽声明。
