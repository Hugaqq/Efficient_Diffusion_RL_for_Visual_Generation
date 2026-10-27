# Assignment B — W7 WORLD8会话生命周期

- active: true
- task_id: H3-M4c2a-runtime-session
- pair: coding-b → simplifier-b
- execution: parallel W7；decisions/wave-07.md
- integration_parent: `d7c75b75210782a63a4fffcae117f2ace19fa760`
- dependencies: 已approved H3Profile/H3LocalFacts、prepare_h3_fsdp与WORLD8合同；不依赖A-BIND-01修复或C本波测试。

## 单一目标/允许文件

为现唯一RunController的create_runtime阶段提供**一个真实单节点WORLD8 NCCL session owner**，不是新的controller/runner。本项只新增 `visual_rl/runtime/h3_session.py`、`tests/test_h3_mvp_cpu_session.py` 及本task_id交付报告。现composition/controller/lifecycle/train、profile/binding/prepare等只读。下一波由本配对将该owner直接接默认生命周期，不在本波实现第二套训练循环或提前接未批准binding。

冻结供下一装配使用的最小接口：`H3RuntimeSession.create(profile: H3Profile, facts: H3LocalFacts)` 返回owner，公开只读 `process_group, rank, local_rank, local_world_size, device, gpu_uuids` 与幂等 `close()`；device是显式cuda:local_rank，gpu_uuids为实际核对后的有序tuple。无训练/optimizer/模型/评分/checkpoint责任，无可传verified=True的旁路。

## 实际准入/所有权

- lazy使用公开torch CUDA/distributed API。创建前检查profile/facts属于同一次binding、torchrun环境WORLD_SIZE=LOCAL_WORLD_SIZE=8、rank/local_rank严格0..7且相等、单节点配置；缺失/非法/多节点拒绝，不默认0..7。
- 要求CUDA_VISIBLE_DEVICES显式八个完整UUID，顺序与profile.launch.gpu_uuids严格相同；不接数字索引、空列表、默认可见十卡或MIG替代。通过实际CUDA设备属性核对可见设备数及每个逻辑device实际UUID映射；没有可用UUID观察API时拒绝并报明确缺能力，不安装依赖/调用服务器工具。profile列表仅是用户将来填写的请求集合，格式/相等不证明用户授权或独占；报告清楚实际窗口仍需外部确认。
- 真实cuda availability、NCCL可用、选定当前device、目标BF16能力检查先于模型物化。不要只凭名字/总显存/UMD版本推断能力；不进行模型加载或性能benchmark。禁止硬编码历史空闲量或提供CPU生产fallback。显存/主机峰值与可用窗口仍由后续实际资源准入确认，不把本session成功叫全资源ready。
- owner只创建一次 `init_process_group(backend='nccl', init_method='env://', timeout=有限明确值)`；拒绝已经初始化的外来PG（不得销毁/重配）。保存自己成功创建的WORLD，后续prepare复用它，不创建额外mesh/子组或Accelerator。
- 初始化后真实核对backend/world/rank和同一host、有序UUID、profile/facts身份的有界全rank一致性。复用现有有界metadata receipt合同为佳，不建新RPC框架，不gather模型/媒体。可协作本地错误统一拒绝；raw transport失败不finally补collective或barrier，timeout非自动重试。
- create部分失败释放自己已经成功获得的本地资源；外来PG不碰。close只销毁自己仍拥有的PG，幂等、无额外barrier/collective；不关闭别人model/scorer、不杀进程、不删除输出。若WORLD引用被外部替换，不能误销毁新owner的PG。
- 不改数值/RNG流、设备环境变量/线程数/NUMA亲和、训练配置或旧single默认；不通过环境猜测CPU测试模式。

## CPU验收

新feature只替CUDA属性/设备与dist通信等不可用叶子，实际session验证、创建/失败/清理控制执行。覆盖八rank合法观察、UUID顺序/实际映射错配、env缺失/uneven/多节点/重复UUID、BF16/NCCL缺失、外来PG拒绝、metadata分歧、创建后故障和close幂等/不碰外来PG。断言重模型/HTTP/optimizer从未构造；无真实init_process_group/网络/GPU调用。必须测试生产公开API调用参数、有限timeout与失败不重试，不只mock整个create。

最后编辑后plan Flash7/H3通配符真实门禁，无skip/xfail/全量/过滤warning；交付branch/HEAD、接口、替身边界、命令/exit/数量/时间/diff/check。若实际安装API不能支持UUID观察，给出具体可实现且合法的最小方案或blocked，禁止虚构。Simplifier B审所有权与真实观察；changes回B。此项不是默认入口闭环或GPU实测，GPU_PENDING。
