# W5 — 实际data / 固定profile / 现manager发布

基线6f5e46fce212e00d547a3ca07fa5524018bef816；W4代码汇合fb085fb35fb239cdd4c151d39decbd7fae268c9d，三配对approved。当前Flash7/13.79s、H3通配符108/8.54s均exit0（保留9条no-dist/损坏fixture warnings）。A组合测试真实执行stages+HPSReward+H3Coordination，正式关闭W3三对象CPU组合缺口；不等于RunController/实际GPU通过。

| 配对 | task | 源码/测试拥有 |
|---|---|---|
| A | H3-M4a5-data-source | data/h3_source.py；test_h3_mvp_cpu_source.py |
| B | H3-M2b2-profile | composition/config/h3_profile.py；configs/v2/h3_t2va_mvp.yaml；test_h3_mvp_cpu_profile.py |
| C | H3-M4b2-checkpoint-publish | artifacts/checkpoint/manager.py的窄shared-staging入口；artifacts/checkpoint/h3_transaction.py；test_h3_mvp_cpu_checkpoint_publish.py |

源码均visual_rl/前缀；测试tests/。现stages/adapter/codec只读，同波不互import。三项不是第二controller/manager/data平台；只为下一轮现RunController装配补真实依赖。

## 冻结交界

A数据对象消费现adapter/dynamics，遵循data.reserve(step)->H3IterationRequest、commit(request)、abort(request)，并提供JSON-safe state_dict/load_state_dict和共同data identity/next_global_batch_id。下一输入由source hash+global occurrence+member+明确stream版本派生，不靠可变隐式RNG；checkpoint仅完整collection边界。data状态真实读取由未来runtime交C，不让Cimport同波A。

B只编译一个显式profile，不启动runtime。固定schema2/recipe=h3_t2va_mvp_v1，模型配置和launch本地model/cache/train/eval/HPS refs分离；定义真实可用的typed冻结profile、semantic ID与局部资源预检。输出供下一轮现ProductionRuntimeLifecycleBackend编译/绑定分支使用，本波不写假restore/checkpoint方法凑齐控制器。

C消费**现有**CheckpointContract/Progress/SafePoint/H3StateIdentity、实际prepared和调用者JSON-safe rank data state/behavior及RankRNGSnapshot。manager主writer创建共同staging，所有rank在writer外执行codec，主writer只发布已有目录/manifest/tree。global progress.rng_state_id绑定有序8rank RNG表摘要而非某rank私有值；本地RNG仍各rank精确校验/最后恢复。不要求A/B新模块来写生产codec/publisher。

## 下一轮装配优先级

W5结果批准后必须将固定profile+真实data+adapter+真实HPS/coordination+完整checkpoint service接入现唯一python -m visual_rl.train/RunController，不能继续以新helper数量替代。允许H3封闭生命周期分支与旧single路径并存，但不能另写Runner或手工run循环绕过RunController的两次restore门。若需泛化大量旧接口，应重新最小化而不是伪造Gaussian/numerics receipt。

CPU各分支Flash7/H3通配符→配对review→汇合再测；unknown dirty不覆盖。真实8卡UUID/模型/HPS124/依赖与DCP多rank仍待授权，GPU_PENDING。
