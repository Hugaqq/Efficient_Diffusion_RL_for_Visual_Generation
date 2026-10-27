# Assignment B — W8 默认唯一controller闭环装配

- active: true
- task_id: H3-M4c2-controller
- pair: coding-b → simplifier-b
- execution: B单一active；decisions/wave-08.md说明并行重评及A/C真实依赖
- integration_parent: `a5fef751a88cad751a36240ffe1f5731986e1b1e`
- dependencies: W7源码汇合6a7616aa26a63215cdae2b1cdd091dc33204f9e4；A runtime-binding-r1、B session、C bound-checkpoint组合均approved。A-BIND-01已关闭。

## 唯一原子目标与边界

把**已批准真实组件**接进现 `python -m visual_rl.train CONFIG` → `create_default_run_controller()` → 唯一 `RunController` 的全生命周期，包含实际完整tail checkpoint与同拓扑fresh恢复后的下一轮。此项是装配，不重写模型/算法/manager或新建Runner。

允许新增 `visual_rl/runtime/h3_lifecycle.py`、`tests/test_h3_mvp_cpu_controller.py`；仅最小修改 `visual_rl/runtime/composition.py` 做明确profile选择及backend构造。新增本task_id交付报告。`train.py`、`controller.py`、旧`lifecycle.py`、types、现H3 binding/session/source/stages/prepare/codec/transaction、现全部tests及Flash配置只读。若实际组合发现这些owner缺口，报告具体最小请求，不能私自扩散修改。不得新增独立train命令、把tests fixture导入生产或以NotImplemented/noop占位交付。

## 默认入口与编译

- 保留原public factory参数和显式service-injected `create_run_controller`旧语义；默认factory返回现RunController，不在路由中再套一个controller。内部窄backend选择在compile阶段按严格源配置recipe精确 `h3_t2va_mvp_v1` 分派，其他配置走原ProductionRuntimeLifecycleBackend。H3错误不得fallback旧Gaussian，也不根据CUDA/pytest环境隐式选mock。
- 复用原load_source_recipe与compile_h3_profile/preflight_h3_profile；未知字段/重复YAML/待填样例失败在PG/模型/HTTP前结束。CLI已有main错误JSON/exit码通过现VisualRLError体系报告，不放松旧异常处理或旧single guard。
- preflight实际全内容核验model/VAE及code，复用verify_h3_content和现filesystem artifact hash。默认code_root覆盖实际导入的visual_rl完整包（现FilesystemArtifactIdentityResolver默认即此）；取得实际预检digest并在binding重核，不使用版本号/全0/任意fixture字符串。保留declared与observed界线。运行时不得悄悄安装依赖或下载。

## 生命周期与资源归属（冻结决策）

1. create_runtime实际调用已批准H3RuntimeSession.create(profile,facts)，不得mock该owner作为生产默认。唯一WORLD在外层ledger最后close；CUDA/NCCL/UUID身份来自session，后续只复用。
2. construct_graph用真实已解析H3声明/config构造H3Adapter、ComponentManager.load的meta路径，建立真实H3PromptSource与明确HPS124资源请求/owner。请求descriptor/artifact/runtime_binding按当前真实类型构造，来自profile/manifest/hash，不用测试_request。HPS acquire必须真实原factory/client准入；构造失败关闭部分已取得资源。
3. prepare实际调用prepare_h3_fsdp，constant_scheduler=True，最终optimizer之后再建立一次H3RuntimeBinding。此时source未reserve、HPS已acquire、代码/模型内容已核验。**允许在prepare返回对象内建立真实binding以供第一恢复门使用**；它不是伪造BIND成功receipt。绑定/handle借用关系与各owner close明确，内部部分失败须本地清理。
4. restore_prepared仅有resume时调用实际H3CheckpointTransaction.restore_prepared，使用上述真实contract/state_identity/validate_live；fresh必须空Adam，无dummy step。不提前恢复source/RNG。
5. transform对prepare已经真实完成的AC/FSDP精度/最终对象作观察核验并复用已计算变换身份，不再cast/wrap/替换对象；没有要求追加变换时不编造已执行新的transform。bind复用同一个binding，构造实际H3IterationStages与其原GRPOTrainer；不得二次绑定/构建新模型。
6. restore_bound调用实际transaction，validate_live真实刷新，restore_data=source.load_state_dict；read_data从公开source state/属性构造既有精确envelope（参照已批准C组合），不调用fresh trainer.checkpoint_boundary伪造tail。RNG最后恢复后不再做消耗随机数的初始化；prepare_run只用公开PrepareRunContext设置返回next-step。
7. run只循环现trainer.run_iteration，不重写任何stage数学。每次成功返回后才可调用binding.checkpoint_boundary，按明确cadence发布；terminal阶段也须真实已完成tail，最终一次checkpoint不能重复发布同step。删掉backend-local trajectory/result长期引用，媒体/aux按既有lease与rank轮转。
8. 所有rank参加既有transaction/codec，rank0仅文件发布；不可把DCP放在writer回调。raw模型/通信异常不追加finally collective或原地retry；局部可协作图构造/HPS/绑定/文件错误用现有有界健康receipt共同拒绝，不让其他rank继续下一模型collective。资源清理无新barrier，session最后关闭。

## 最小输出/恢复策略

- 复用既有AtomicCheckpointManager实现及实际RunResult，不造另一个checkpoint manager。必要少量run元数据/metrics/最终marker可以在本模块生成，真实字段和文件路径；rank0写其他rank验证/共同获得结果。结果中的checkpoint必须inspect_complete成功，SUCCESS不得先于最终完整checkpoint。
- Fresh output要求新目录（拒已有未知/非空目录及输入树重叠，不删除/覆盖）。只接受本任务完整checkpoint，严格同world/ordered UUID/代码/模型/cache/score/数学/源内容；不mid-collection恢复，不拉旧Flash状态兼容。
- 为最小一manager闭环，首版resume限定**同一个run输出根**：resume_from是该output/checkpoints下明确完整最新checkpoint；未知run元数据/语义不匹配/已有更新后的checkpoint拒绝，不擅自覆盖。只恢复非terminal `0 < next_step < max_optimizer_steps` 继续到相同原目标；已terminal resume可明确早拒（不dummy更新或伪造fresh tail），该限制交付中写明。max目标参与现semantic，测试不能改target来制造resume相容。
- 保留已发布合法checkpoint，即使后续metric/marker IO失败；再次resume必须核验现有树/元数据，不把unknown dirty输出当新run。元数据仅实记运行，不宣称UUID授权/空闲/总资源ready或GPU质量。
- 这里不新增资源监控/守护服务/性能预算框架。既有显式host weight/trajectory/decode预算必须真正传给消费者；其余峰值/空闲窗口仍GPU验收事项，不填假的resource receipt。

## 真实CPU验收

新feature至少覆盖：

- 默认factory返回的真实RunController跑有效tiny H3配置完成≥2次真实更新、cadence与terminal checkpoint/真实RunResult；实际CLI `train.main([path])` 到同默认factory亦执行一次成功或同真实运行并核验输出，不只mock _create_controller返回结果。
- 八rank控制/通信可采用现明确CPU叶子方案；实际profile/session owner/binding/source/stages/HPS/transaction控制均执行，不高层mock backend、session.create、prepare、binding或checkpoint。重模型/VAE、CUDA/dist/FSDP/HTTP/DCP通信等叶子替换须列明。可以复用只读现feature helper，但默认生产无test路径。
- 用同target配置在成功checkpoint之后定点中断，记录continuous参考；fresh controller/adapter/prepared/真正空Adam重新加载该checkpoint并实际下一轮至相同目标，比较请求/参数/moments/scheduler/source/RNG/progress，step spy禁止恢复期dummy更新；不改私有计数。未恢复source/RNG的第一门、第二门RNG-last及controller实际stage顺序有证据。
- 全生命周期reverse close与部分失败清理，外来PG/输入变化/错误resume/未完整tail不发布SUCCESS。错误传播不增加raw失败后collective。原有待填样例在无GPU环境早拒，无模型或HTTP副作用。
- 默认路由旧SD3/Wan/Flash仍到原backend，旧配置语义/七项保持；不靠更改原测试或缩减断言修绿。

最后编辑后plan固定Flash七项与完整H3通配符真实执行，无skip/xfail/全量/警告过滤；交付实际branch/HEAD、源码范围、调用链/owner表、命令/exit/数量/时间、真实输出/恢复证据、未验证边界。若接口或范围不足，给出最小阻塞而不是以stub完成。

Simplifier B审最小装配与唯一controller、数据/数学不重复、两恢复门和真实CPU调用证据；changes回B。所有现原子approved不代表本项已完成；GPU_PENDING。
