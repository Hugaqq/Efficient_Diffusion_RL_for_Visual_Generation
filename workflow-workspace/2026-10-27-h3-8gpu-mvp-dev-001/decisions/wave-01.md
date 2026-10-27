# Parallel wave W1 — 已确认独立边界

用户最新2026-10-27调度要求覆盖旧task/plan的顺序默认和H3测试单文件要求；不修改task/reference/workflow本体。基线 `b7f938506f16034ac5d2896ddfa44c16eb4af83d`。M2a2 coding snapshot `919b6abc7920f2dbcea764907152523882a09a59` 已配对approved并由orchestrator读取代码/独立CPU复核。此轮返回parallel，三个配对均active，各一小项。

| slot/task | 唯一源码写入 | 唯一测试写入 | 只读依赖 | 输出合同/后续消费者 |
|---|---|---|---|---|
| A/H3-M4a1-anchor | algorithms/optimization/h3_anchor.py | tests/test_h3_mvp_cpu_anchor.py | h3_cps.py、h3_schedule.py、h3_joint.py、现objective | rank-local一次性replay旧分数；未来OptimizeStage在credit后current前调用 |
| B/H3-M2a3a-text-cache | models/implementations/h3_text_cache.py | tests/test_h3_mvp_cpu_text_cache.py | h3_forward.py所需trimmed embedding合同、上游layer50证据 | CPU `[1,L,5120]`与cache身份；未来H3Adapter预处理/绑定 |
| C/H3-M3a-prepare | models/lifecycle/h3_fsdp.py | tests/test_h3_mvp_cpu_prepare.py | h3_loading.py、现PreparedModelBundle/ComponentHandle/组件协议 | WORLD8 block-FSDP2物化及最终参数optimizer；未来runtime生命周期绑定 |

所有源码路径前缀visual_rl/。每组只写对应coding报告，simplifier只写本组review。不允许修改原H3单文件测试、原七项或兄弟新增文件；不需要先搬已有测试。三项不相互import，也不依赖同波未完成结果。现有已批准模块只读，接口缺陷退回原owner，不在并行分支偷改。

## 接口冻结

- A调用 `H3CPSDynamics.evaluate(record,prediction)`，输入全是既有CPS record；预测callback由未来阶段接prepared forward，A不import C，不拥有optimizer、PG或模型权重。每次仅一格临时设备record，返回冻结CPU score。
- B只交CPU detached trimmed embedding，不调用encoder、模型forward或PG。5120维/FP32或BF16等实际dtype显式锁，cache miss必须拒绝；未来设备搬运属adapter，B不需要C。
- C消费现 `H3MetaModel` 的model/block_names/state_specs/base_to_model/lora_names/nonpersistent_buffers及 `load_h3_cpu_state`。C不改loader、不import A/B新模块；只需模型kwargs，geometry/forward已稳定。只创建FSDP prepare路径，不复制六阶段trainer。
- 后续装配将把B embedding给既有h3_t2va_forward，用C受控callable生成prediction；A只通过上层闭包消费这个预测。三者现在无需共享新公共接口。

## CPU与真实证据

每个隔离worktree原样执行plan的Flash七项和H3通配符命令。并行各自通过只证明各分支已有目标；run内暂存汇合后orchestrator再跑完整通配符集，并核对三份对应simplifier。任何changes仅回同coding，重新两集再审；暂存汇合不是批准。CPU fake叶子只替不可执行的重模型/CUDA collective等叶子，不得替整个prepare/anchor/cache控制路径。GPU/FSDP2/NCCL证据仍pending。

## 仍需串行的真实依赖

controller/recipe和完整OptimizeStage接线需要此波输出合同经review稳定；world phase/归一化需要prepare的真实reducer及公共计划接口；完整tail/checkpoint依赖实际optimizer/data进度。VAE decode是下一独立小项，不与text reader揉成整套aux。以上不是因共享测试文件串行。
