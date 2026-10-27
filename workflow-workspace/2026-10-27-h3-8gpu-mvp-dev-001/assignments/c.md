# Assignment C — W1

- active: true
- task_id: H3-M3a-prepare
- pair: coding-c → simplifier-c
- execution: parallel W1；见decisions/wave-01.md
- integration_parent: `b7f938506f16034ac5d2896ddfa44c16eb4af83d`
- source_dependencies: loader snapshot `919b6abc7920f2dbcea764907152523882a09a59`、forward snapshot `edd5c874f10ffa917a71bd9f2fa9085247075a6e` 已approved；无待返工。

## 唯一目标/文件

实现**消费已批准loader的单节点WORLD8 FSDP2 prepare→全rank物化→最终参数optimizer** 窄生产入口，非新runner。仅写新 `visual_rl/models/lifecycle/h3_fsdp.py`、新 `tests/test_h3_mvp_cpu_prepare.py`、本目录 `coding-c/H3-M3a-prepare.md`。现loader/组件/PreparedModelBundle/ParameterStateManager/runtime/旧tests只读；不import A/B本波新模块。不直接接controller或改旧single路径。

## 冻结输入/输出

- 消费 `H3MetaModel`：model全参数meta、block_names、target_names、state_specs、base_to_model、lora_names、nonpersistent_buffers；调用真实 `load_h3_cpu_state` 生成rank0完整CPU输入。只在rank==0调用，其他rank空state参与同一setter，不相信外部is_init_owner标记代替真实rank检查。
- 接收**调用者已初始化**的WORLD process group、当前CUDA device/逻辑rank及明确host预算/init seed/optimizer超参；此模块不init/destroy PG、不创建Accelerator。生产拒非WORLD8/非单节点rank布局/非CUDA/SP或TP不为1，以及已materialized/已prepare模型。不默认具体物理UUID；资源授权由后续launcher准入。
- 真实lazy PyTorch FSDP2 API：单WORLD8 device mesh，按meta.block_names对主blocks fully_shard，reshard_after_forward=True、reduce FP32、forward parameter BF16、cast_forward_inputs=False。root不fully_shard，root/refiner保持冻结复制和既有FP32岛；全部可训练必须属于分片主blocks。若目标FSDP API不能支持当前BF16 frozen/FP32 master组合，报告证据与最小修正请求，不无声整体cast或用假DTensor冒充。
- 一条外部nonreentrant block activation checkpoint路径，内部gradient_checkpointing必须关闭。包裹顺序明确，保持或显式映射state FQN，不因wrapper改名放宽strict。使用真实PyTorch checkpoint_wrapper/fully_shard，不搬UniRL完整backend；先读参考LICENSE与wrap/state并核对实际可用API。
- **先wrap/meta分片再物化**；禁止任一GPU先装完整模型。使用真实DCP `set_model_state_dict` 等全rank API，full_state_dict/broadcast_from_rank0/strict选项准确，rank0输入完整base+LoRA，其他rank也调用。不能把load_state_dict(assign=True)当真实FSDP setter；那只能是CPU测试的明确leaf替身。未支持API必须早拒，不能NotImplemented占位。
- 持久buffer随完整state；to_empty若破坏构造buffer，显式恢复loader的`rope.inv_freq`并核验dtype/shape/值，所有rank相同；不得空buffer。模型/LoRA初态统一，元数据（config/shape/dtype/LoRA初始seed/blocks）在潜在重collective前一致性准入；可用有界小表，不allgather全权重。此项只负责初始化一致性，整个训练phase计划以后单独实现。
- 所有参数物化、精度/LoRA范围核验后才构造真实AdamW及可选constant scheduler，optimizer参数恰是最终trainable参数对象，没有旧metaParameter引用或重复；默认lr1e-4/betas(.9,.999)/eps1e-12/wd0，参数可显式锁定。不要先构造旧ParameterStateManager后删除其身份保护来兼容变更。
- 复用现 `PreparedModelBundle` forward guard/单入口（可以在物化后构造bundle，优化器对象校验用现validate_optimizer_parameter_subset）；返回最小prepared结果，暴露一个模型callable和optimizer/最终参数信息，确保既有h3_t2va_forward可消费。优先复用现 `PreparedComponentHandle` 的可行部分，不强行调用accelerator.prepare第二次、不另实现复制guard框架。若旧handle要求accumulate适配，明确窄backend能力，不fake全Accelerator。
- 模型eval但允许grad；close只释放本地guard/引用，不暗加collective或关调用者PG。失败不写checkpoint、不barrier补救，不尝试回滚已执行部分collective；上层负责整attempt失败。

## CPU验收与证据边界

不运行真实CUDA/NCCL/多进程网络。目标测试只替cuda admission、mesh/FSDP collective/state setter等不可本机执行的**叶子API**；生产默认真实API存在并lazy调用，不能有CPU伪分片作为生产fallback，不mock整个prepare函数。

使用真实tiny loader/PEFT产物（可复用原目标模块只读helper或独立小fixture，不修改原test文件），验证准确事件顺序：rank0读取一次、非owner不读→AC/shard→物化/setter/真实buffer恢复→最终参数校验→optimizer构造。CPU leaf setter可显式用strict assign=True来模拟最终参数替换，证明Adam持有新的参数对象。真实torch优化器step、prepared guard和bridge forward/backward执行，拒raw旁路/二次prepare/非法topology/stray trainable/缺buffer或不完整state；失败时没有optimizer或隐藏collective cleanup。记录哪些leaf为fake，不把记录的8rank参数当真实多rank证据。

测试独立写 `test_h3_mvp_cpu_prepare.py`，核验导入当前worktree。最后编辑后跑plan最新版固定Flash七项与H3通配符集，无skip/xfail/全量，记录exit/数量/时间/branch/HEAD及diff/check。若权限/API确实不足就报告blocked，不能为门禁通过关闭保护。

## 非目标

本项不做world advantage/loss归约/global grad norm、rollout/评分/锚、data/tail、checkpoint或torchrun/controller接线。尤其不声称GPU/FSDP2实跑通过。后续在稳定prepared结果上接公共phase和world数学，C仍为owner。Simplifier C审真实生产API顺序、最终参数/buffer/精度/单owner、未重写runner以及CPU证据边界；changes回C修复两集重测。GPU_PENDING。
