# H3-M4c1-runtime-binding — Coding A / W6

## 判定、续跑与身份

- `implementation_ready: true`：本原子项实现和最终CPU门禁完成，交simplifier-a；不是默认RunController或checkpoint publisher批准。
- worktree：`/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.pi/workflows/worktrees/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-70fff509-7bc8-423f-813c-5d88c3eb6454`。
- branch：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-70fff509-7bc8-423f-813c-5d88c3eb6454`。
- attempt起点：`75b80699e0380a69c4b7068d19cdba62d420eaab`；恢复源码snapshot `23691bc744cd247eb9e1f90f07bafefd0c851ed6`，随后测试保存snapshot `7f4df76687fbebda09629a925c34ab722c9e2573`。git log实际确认相邻顺序；当前HEAD仍7f4df76，最后修正待引擎snapshot。
- 两次续跑均先读实际保存源码/status及已返回工具结果，无未返回命令；未重放先前write。上次129项成功已记录，但随后新增检查有源码变化，故最终两集必须重测。
- 已读当前assignment/wave-06/plan、总task及前置本配对审查，实际数值/参数/projection/checkpoint/profile/source/模型依赖。无本项已存在changes；C publisher C-PUBLISH-01归原C，本角色未重做/批准其任务。

## 实际范围

1. 新 `visual_rl/runtime/h3_binding.py`。
2. 新 `tests/test_h3_mvp_cpu_binding.py`。
3. `visual_rl/models/implementations/h3.py::_prepared_forward` 一行委托改为现基类 `_forward_prepared`。未绑定轻量测试仍沿基类既有fallback；已绑定必须有正确scope。
4. `visual_rl/algorithms/trainer/h3_stages.py` 仅新增只读 `checkpoint_boundary(trainer)`，不改六阶段/提交控制。
5. 本报告。

没有修改通用controller/composition/runtime/manager/codec/profile/data、原七项、其他测试或兄弟W6文件；不引用B/C同波新符号。不写第二runner，不调用publisher或发布checkpoint。

## 精确内容身份与实际对象绑定

`verify_h3_content(profile, *, code_root, expected_code_sha256)`每次调用现`composition.preflight.artifacts._hash_artifact`，不重写hash框架：

- model hash定义为`filesystem-artifact.v1`、`all-files.v1`对整个model目录的内容身份；包含transformer/base、video VAE、audio VAE全部文件，不是header摘要或字节拼接SHA。返回existing resolver的content_sha256/file_count/byte_count等实际记录。
- code hash为同codec的`python-code.v1`，覆盖caller给定代码根下实际Python内容；未来runtime必须提供完整实际import代码根与其预期digest，不能拿版本号替代。
- model内容摘要必须同时等于profile.expected.model_sha256与cache声明的model_sha256；code必须等于expected。没有verified=True参数，也没有把facts.model_weights_verified=False改成True。header facts仍保留“仅header检查”的原语义。
- 测试把真实tiny完整权重文件写入后，用同定义计算内容hash，再写入cache声明及safetensors metadata；不是用全0或任意mock字符串放行。

`H3RuntimeBinding`接收真实profile/facts、loaded H3Adapter、最终H3PreparedModel、H3PromptSource、已acquire的H3HPSReward、WORLD/rank/device、ordered GPU UUID tuple、code根及预期hash。实际核验：

- WORLD8/rank、显式rank-local CUDA索引、与profile相同有序UUID映射；CUDA当前device观察由现helper完成。UUID授权/空闲/硬件窗口仍是上层session前置，不因字符串一致宣称可用。
- 重新执行只读profile preflight并逐值比较facts；adapter config/transformer路径/config SHA实际一致，source借用的adapter/dynamics是对应对象且K/N/grid/eta/seed/rank/device/cache/数据hash/预算/stream语义一致。
- HPS此前已acquire且仍open，重新读manifest；profile/service/124帧契约/weight1/endpoint/timeout/response上限一致。本项不创建第二client、不隐式HTTP acquire。
- adapter.meta_model.model正是prepared transformer；prepared optimizer恰覆盖最终trainables、single-group真实AdamW、固定lr/betas/eps/weight_decay，拒maximize/AMSGrad/capturable/differentiable/fused；完整group超参数（除params）进入hash并持续拒漂移。
- 对照loader state_specs检查最终FQN/global shape/dtype/device/materialized、精确主blocks LoRA范围；persistent buffer精度/形状及完整覆盖，nonpersistent/rope原值与dtype；记录最终parameter/buffer对象ID并拒替换/precision漂移。
- 每个实际FSDP2 block的全部`_fsdp_param_groups`核对BF16 compute/FP32 reduction/cast_forward_inputs=False/divisor8，不仅检查第一组。普通未shard模块拒绝；测试仅替该观察叶子，另用真实FSDPState/真实MixedPrecisionPolicy元数据执行观察函数正负例。

## 当前视图/G3数值与受控调用

完成prepare后才构造真实ParameterStateManager及ModelStateProjection，未在meta上冻结对象身份。用这些实际owner/FQN/shape/dtype生成：

- ParameterDTypePolicy(trainable=float32, frozen/buffer preserve_loaded)，仅声明已观察到的storage规则，不再次cast模型。
- ParameterViewEvidence只含CURRENT，owner为transformer，restorable names为实际projection的非空trainable集合；无reference/EMA。
- ModelExecutionNumericsEvidence包含ROLLOUT/TRAIN的真实StageExecutionPolicy与ForwardAutocastPolicy。root autocast disabled、root scope compute_dtype=float32；子block BF16由上述观察独立进入执行身份，不把BF16应用整个root。
- adapter绑定证据与provider；provider每次前向检查scope、view、grad、EVAL、外层autocast、最终参数/buffer/optimizer身份。缺scope、错view、精度或对象漂移在prepared forward前拒绝。
- `predict`从source.conditioning_for(sample_id)取得真正H3Conditioning并核对传入embedding；从精确双schedule显式index取video/audio sigma，构造H3ModelInput，实际adapter.predict→基类_forward_prepared→既有prepared guard。没有raw unwrap/第二次符号转换/Gaussian fallback。
- `decode`通过adapter的CPU latent spec与AV lease；空prompt只用于无条件的shape/spec查询，不encode或造假conditioning。`score_one`委托真实已acquire HPS对象；`coordination_factory(plan)`构造真实H3Coordination。
- binding.close仅标记本binding关闭，禁止继续使用其scope/callback；不关闭借用adapter/prepared/scorer，不销毁PG。外层lifecycle继续是这些资源唯一owner。

## CheckpointContract字段来源（封闭H3 payload，不冒称旧Gaussian materializer）

所有namespaced IDs通过固定namespace+canonical实际payload计算；普通SHA也均有内容来源。仅model component ref使用已有真实registry declaration，没有编造不存在的algorithm G1 receipt。

| 字段 | 来源 |
|---|---|
| recipe_id | materialized-recipe.v2：H3语义与实际资源hash |
| resolved_fingerprint | resolved-recipe.v2：实际profile facts.semantic_payload |
| algorithm_materialization_spec_id | algorithm-materialization-spec.v1：现H3IterationStages类路径、固定training数学、实际双schedule/CPS摘要、代码hash |
| execution_policy_id | execution-policy.v1：实际两stage current policies与固定ordered拓扑 |
| reward_plan_id | reward-plan-spec.v1：实际已acquire H3HPSIdentity |
| source_content_binding_id | source-content-binding.v1：source真实JSON binding（文件/parse/stream/geometry/cache/seed/K/N等） |
| component_artifact_binding_set_id | component-artifact-binding-set.v1：实际model declaration ID、全model内容hash、cache index hash |
| runtime_bound_contract_id | live bindings、实际projection与typed numerics的SHA |
| immutable_model_revision / code_identity | 上述all-files模型内容hash / python-code内容hash |
| components | 一个真实model ref：profile.model.declaration_id、实际artifact binding摘要及runtime ID |
| model_state_projection / id | 最终ParameterStateManager.state_projection及其真实ID |
| model_execution_numerics / id | 上述实际current-only typed evidence及其ID |
| trainable_parameters | 最终实际owner-prefixed names/global shape/dtype |
| optimizer_groups | 最终参数集合及实际完整Adam group超参数hash |
| scaler_schema / lr_scheduler_schema / precision | 无scaler、已观测LambdaLR constant路径、BF16 blocks+FP32 islands/LoRA |
| preprocess_identity | 实际cache index hash与source.geometry_id |
| preprocess_requirement_set_id | 已验证layer50/raw5120/trimmed/guidance1窄合同 |
| group_size / global_batch_size / gradient_accumulation_steps | profile K / 8*K / 1；全collection一次update |
| data_sharding_version / sampler_state_schema / rng_policy | 已批准source采样版本 / h3.prompt-source.v1 / 实际STREAM_VERSION |
| dynamics_state_schema | h3.cps-dual-all-n.v1，完整双grid/eta身份在live semantics中，无随机选步状态 |
| progress_state_schema | 直接引用现CHECKPOINT_PROGRESS_SCHEMA_VERSION（当前v2），不是猜测v1 |
| ema/reference_state_schema | none，能力与证据中也没有这两视图 |
| execution_transform_plan_id / chain | 真实主blocks名称、非重入AC/FSDP2路径及已观察block precision/divisor payload摘要 |
| world_size / world_size_policy | 8 / strict |
| state_schema_versions | model2（现projection约束）、optimizer/data/rng各v1对应当前payload所有者 |

H3StateIdentity的model/topology/semantics来自以上live表；group_geometry_id只对已验证`h3-geometry:<sha>`作精确namespace提取。live表严格包含model/topology/semantics/data/cache/score/dynamics/resources/code九项。

`validate_live()`不是返回缓存dict：每次重新观察WORLD/rank/device/topology、完整文件内容、profile/HPS manifest/cache/source、最终模型/optimizer/projection/numerics，再生成实际绑定。未知scope或关闭状态拒绝。

## 完整tail边界与下游接口

```python
binding = H3RuntimeBinding(
    profile=profile, facts=facts, adapter=adapter, prepared=prepared,
    source=source, scorer=already_acquired_hps, process_group=WORLD,
    rank=rank, device=f"cuda:{rank}", gpu_uuids=admitted_ordered_uuids,
    code_root=actual_code_root, expected_code_sha256=expected_code_digest,
)
stages = H3IterationStages(
    data=source, prepared=prepared, predict=binding.predict,
    decode=binding.decode, score_one=binding.score_one,
    coordination_factory=binding.coordination_factory,
    process_group=WORLD, rank=rank, device=device,
    read_behavior_id=source.read_behavior_id,
)
# 使用原trainer/未来唯一RunController，非此binding新造循环：
result = trainer.run_iteration(step)
progress, proof, rank_data, rank_rng = binding.checkpoint_boundary(stages, trainer)
# C publisher获批且默认runtime接线后消费：
# binding.contract, binding.state_identity, binding.validate_live
```

- stages新增只读边界检查要求exact trainer/prelude、active=None、非poison、有实际成功tail，trainer.next_optimizer_step>0；source无pending/poison且step/cursor/behavior一致、所有梯度None。
- binding再核Adam真实step与scheduler.last_epoch相等，重查live资源。optimize已返回但原BaseTrainer尚未commit/推进next-step时明确拒绝。
- SafePoint为这些实际同步六阶段/tail检查的结果；同步调用无reward futures/dynamics sessions，均已释放后才标0。Progress使用实际已提交step、global cursor与source/source batch身份；selection metadata明确h3.no-selection-rng/all-N含q0（base_seed0仅表示不用选择RNG），实际采样stream版本单独进入语义。
- rank_data精确匹配C现schema：schema_version1、rank/world8、plain SHA content_id/next_global_batch_id、source_id、next-step/cursor/behavior/open_reservations0和source实际回读payload。只接受h3-data/h3-batch确切namespace，不泛化split丢类型。
- progress.rng_state_id来自真实RankRNGSnapshot.capture_current(rank)，此处只capture不restore。publisher随后替换有序8rank全局摘要，仍须最后恢复本rank RNG。
- 本项不调用尚未获批publisher，不把生成contract/proof/envelope称为已发布complete checkpoint。

## 四项CPU目标 / 失败与修复

真实profile正常fixture、全模型tiny文件hash、真实cache/source、真实loader/PEFT/非重入checkpoint wrapper/prepare/Adam/guard、真实HPS wrapper与HTTP transport叶子执行。CUDA/world/shard/DTensor观察仅明确叶子替身；没有mock binding或dtype/projection/hash逻辑。

- 实际base数值scope消费、缺scope/错view/外层autocast拒绝、typed条件绑定、双流梯度及no_grad同值、AV lease与all124 HPS POST、close不重复关闭借用对象。
- code/UUID/facts伪verified标志、source seed/eta/预算不一致、persistent buffer初始精度错误、参数替换/precision/rope/optimizer mode和超参数漂移，以及model/VAE/code/cache/HPS/data文件内容漂移全部拒绝。
- 真实H3IterationStages/BaseTrainer完成一轮，实际binding callbacks（含真实H3Coordination，通信/DTensor仅CPU观察）被消费。所得typed CheckpointContract通过自身校验，progress/Adam/scheduler/data step1/cursor8/behavior一致；cold、active reserve、optimize后tail前、poison和未清梯度均拒绝。capture RNG不改变其状态。
- block observer测试初次出现`1 failed,129 passed`：测试错误地给FSDPState只读`_fsdp_param_group`属性赋值。依据实际现API和已有真实divisor setter测试，改为`_fsdp_param_groups`列表；生产观察同步改为遍历全部组，真实policy对象的第二组错误也拒绝。未删断言/skip；最终全通过。这不是CUDA故障复现。

## 最终门禁（最后源码编辑后）

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
```

- H3 exit0：**130 passed, 16 warnings in 12.54s**（基线126+本项4，不含W6兄弟未汇合改动）。
- Flash exit0：**7 passed in 12.63s**。
- 16条既有DCP no-dist/损坏fixture警告原样保留；无skip/xfail或全量。新binding测试未调用publisher；通配符保留的既有publisher测试通过不等于批准C-PUBLISH-01。
- git diff --check exit0；最后相对7f4df76的diff为binding/test两文件92 insertions/11 deletions，前面的新增实现/adapter一行与stage只读方法已由恢复snapshot保存，不以当前diff冒充完整attempt范围。
- 原Flash整个文件相对授权d818b8a历史diff为空，exit0；旧七项断言未改。

## 交付边界

待simplifier-a审查实际证据/控制调用/完整tail；changes回A重测两集。C目录fsync必须等其原配对复审，B解析对齐须汇合后重跑；本报告不批准兄弟项。

下一默认生命周期必须在真实artifact/code与GPU/HPS准入、load/prepare之后使用此binding，并在完整trainer返回后取边界；消费本closed H3合同，而不是用旧Gaussian materializer假成功receipt。随后仍须唯一RunController默认dispatch、两阶段整run fresh恢复及下一步对照。测试code根是实际tiny Python文件fixture，模型/HPS/CUDA观察也明确为CPU叶子证据，不等于实际生产代码根/权重/GPU/HPS资源已准入。

无服务器/GPU/真实HTTP/PG创建/下载/环境升级/push或手工Git提交。**GPU_PENDING，MVP未完成。**
