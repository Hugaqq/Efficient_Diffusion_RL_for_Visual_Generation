# W8 — 已稳定组件直接接唯一默认生命周期

## W7接收

当前HEAD a5fef751a88cad751a36240ffe1f5731986e1b1e，源码汇合6a7616aa26a63215cdae2b1cdd091dc33204f9e4。已读三个coding及三个review原文、最新scheduler观察/session全文与真实bound-checkpoint测试。

- A runtime-binding-r1 approved：scheduler/optimizer引用、实例、constant可观察状态统一只读核验，真实两轮计数与负例；关闭A-BIND-01，接收runtime-binding原子。
- B runtime-session approved：实际ordered UUID/native BF16/WORLD8准入及180s NCCL owner，失败只清自己PG。不是授权/空闲/资源ready或GPU证据。
- C bound-checkpoint-integration approved：真实binding/source/tail身份→磁盘DCP/原manager→fresh空Adam两门恢复→真实下一轮，含两门漂移与RNG-last。组合缺口关闭，但未执行default RunController。

本轮orchestrator亲跑Flash **7 passed in 13.64s**；H3通配符 **146 passed, 22 warnings in 22.58s**，exit0；原Flash完整历史diff相对d818b8a为空。没有遗留changes或pytest失败。C曾有主线程误用peer rank的fixture失败已修且最终/各review/本轮门禁全部通过，未删断言或skip。

## 并行优先重评与本轮选择

选择 **B单一active，outcome=b**。不是恢复顺序默认或共享test借口，而是当前剩余一个必要装配owner：

1. A候选数值/数据/pool/锚功能均已实现且批准；没有新的独立必需算法。入口事件/rollback测试必须消费B尚不存在的backend/owner实现；凭空mock它不能证明闭环。A idle，后续如所属binding/source问题回A。
2. B现在具备profile/adapter/session与已批准binding/publisher所有依赖，直接做H3-M4c2-controller，不再堆session之后的helper链。
3. C候选prepare/world/publisher与真实对象恢复组合已经完成；再写恢复服务helper会与B两门装配拥有权重复。入口恢复故障/终态补验需要B的真实output/default lifecycle路径；C idle，所属codec/transaction缺陷仍回C。
4. 运行说明必须依赖真实最终参数/输出/恢复策略，不能让C先写假命令冒充可并行实现。新test模块完全允许，此处阻塞是实际生产入口尚不存在而非测试文件共享。

B独占新runtime/h3_lifecycle.py、新test_h3_mvp_cpu_controller.py及composition.py的精确dispatch变更；train.py/controller.py/旧lifecycle/types/旧测试及已批准H3 owners只读。一次原子glue把现组件连起来，不重写算法/模型/DCP，不另造Runner或第二controller。

## 接口/恢复顺序决策

- 入口固定python -m visual_rl.train CONFIG、现RunController方法顺序不变。默认composition在compile精确识别h3_t2va_mvp_v1，其余沿原backend；invalid H3无fallback。service-injected旧factory保持。
- H3 prepare返回时允许一次建立真实binding：最终prepared参数、未reserve的source、已acquire HPS与已核验model/code均就绪。第一恢复门使用真实合同；transform只检查已完成AC/FSDP身份，不重新wrap；bind复用同binding并接原stages/trainer。不是在RESTORE前填假G3，也不提前恢复source/RNG。
- 第二门用公开source状态构造真实envelope，不调用fresh trainer的checkpoint_boundary；RNG最后，随后公开PrepareRunContext恢复next-step。C批准测试提供这一路径的真实证据。
- 首版最小resume策略固定同output run根下的最新完整非terminal checkpoint、同原max目标及全部语义；未知输出不覆盖。已terminal resume可明确早拒，不凭无tail的fresh对象重存。新RunResult必须指向真实完整checkpoint与真实run文件，不造零指标/空成功marker。
- 仅完整trainer返回后按cadence/terminal调用既有transaction。初始化/图/HPS/文件可协作错误在模型门前统一拒绝，raw模型/collective错误不救援barrier；session在逆序清理最后关闭。

## 后续验收

B必须用默认factory/实际controller与CLI路径的tiny CPU叶子跑真实更新/发布及fresh同target继续，不以此前146项替代新增入口证据。原七项和完整H3通配符最后重跑、同simplifier-b审核；如越界owner缺口据实回orchestrator再按配对修复。B稳定后可再次并行安排独立入口负例与最终交付补验；未有默认调用链/所有批准不能complete。

实际8UUID授权/窗口、H3/HPS124资源、host/GPU峰值、UUID/native BF16 API及CUDA/FSDP2/NCCL/DCP均未实测。本轮只有本地协调与CPU，GPU_PENDING。
