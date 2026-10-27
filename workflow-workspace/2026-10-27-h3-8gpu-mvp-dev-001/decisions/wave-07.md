# W7 — A原配对返工与默认生命周期前置并行

## 接收事实

当前暂存HEAD `d7c75b75210782a63a4fffcae117f2ace19fa760`，代码汇合 `c8422b3a780d933564e94dd58c52a5c8efd8f9dc`。本轮读取三份coding、三份实际review和源码/测试：

- B `H3-M2b2-profile-align` approved，唯一source解析/16MiB硬cap/版本及真实reserve/commit交叉验证；关闭B-PROFILE-01。
- C `H3-M4b2-checkpoint-publish-r1` approved，附件写完/全rank校验后rank0一次目录fsync、发布前同步及OSError共同poison；关闭C-PUBLISH-01，接受publisher原子项。不是断电实验或多卡DCP证据。
- A `H3-M4c1-runtime-binding` changes：A-BIND-01。实际_observe_model仅检查LambdaLR类型，未绑定scheduler.optimizer或实例；codec消费者却明确要求同optimizer。现135项不含该负例，不能放行。

Orchestrator亲跑Flash **7 passed in 14.61s**；H3通配符 **135 passed, 18 warnings in 14.28s**，均exit0。警告未过滤，原Flash完整历史diff相对d818b8a为空。尚无default entry或整run fresh恢复完成证据。

## 并行与接口冻结

| 配对 | 原子项 | 拥有路径 | 同波依赖 |
|---|---|---|---|
| A | H3-M4c1-runtime-binding-r1 / A-BIND-01 | h3_binding.py + test_h3_mvp_cpu_binding.py | 仅内部scheduler观察；所有公开签名/返回冻结 |
| B | H3-M4c2a-runtime-session | 新runtime/h3_session.py + test_h3_mvp_cpu_session.py | 仅稳定profile/facts/torchrun/WORLD合同，不import A/C新内容 |
| C | H3-M4b3-bound-checkpoint-integration | 仅新test_h3_mvp_cpu_bound_checkpoint.py | 当前binding合法constant输入与已批准source/publisher；不用A新增符号，不修改兄弟tests |

A只修原review，B补实际唯一runtime session owner，C验证真实对象恢复组合。这是三项独立文件工作，不以共享test为顺序理由。C正常输入在A修复前后都必须兼容，故没有逻辑等待；其成功不能替代A对应负例审查。若C发现生产缺陷，回真实owner，不用测试mock掩盖。

B冻结最小接口：`H3RuntimeSession.create(profile, facts)`→只读process_group/rank/local_rank/local_world_size/device/gpu_uuids与幂等close。只创建自己WORLD，拒外来PG；显式ordered UUID请求与实际device mapping匹配，有限timeout与同host/配置一致性。没有模型/评分/训练循环，也不改已批准数值或RNG策略。当前仅实现CPU可审代码/叶子测试，禁止实际GPU/PG/服务命令。UUID字符串不能证明用户授权/空闲；真实资源与窗口仍外部待确认。

## 为什么本波不强行接完整controller

- 唯一RunController真实顺序已读：prepare→restore_prepared→transform→bind→restore_bound；当前默认runtime仅支持single且使用Accelerator，不能删除guard让WORLD8伪通过。
- session缺真正PG/device owner，是默认create_runtime必要功能，不应临时stub。A-BIND-01未审通过不能把binding当最终accepted。
- transaction第一恢复门需要完整contract/live，而binding来源必须是实际prepared/source/scorer；第二门需要source真实回读但不该虚构新trainer已执行tail。C本波实证该组合与合法顺序，降低下一默认装配绕过门的风险。
- 全部稳定后**下一波优先由B接现composition/default backend**，消费本session、A binding、现stages及C publisher，保留唯一RunController与`python -m visual_rl.train CONFIG`，无第二Runner。完成真实CLI/controller的tiny连续/恢复验收以前不得complete。不得继续以独立helper数代替闭环。

## 门禁/批准

各组最后编辑后Flash原七项与完整H3通配符，原断言不改、无skip/xfail/全量/警告过滤。原配对审查，changes回同slot。汇合后orchestrator再复测两集/查看每个verdict；引擎暂存合并不是发布。硬件历史单节点8×64GiB、无NVLink、双NUMA不证明实际空闲、兼容或性能。GPU_PENDING。
