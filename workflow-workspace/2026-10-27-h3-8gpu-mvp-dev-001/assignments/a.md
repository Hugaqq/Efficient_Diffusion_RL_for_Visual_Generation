# Assignment A — W6 runtime绑定原子

- active: true
- task_id: H3-M4c1-runtime-binding
- pair: coding-a → simplifier-a
- execution: parallel W6；decisions/wave-06.md
- integration_parent: `a54d296a7635fe3d552b0fe9beb03717e34249a2`
- dependencies: 已批准profile/source/adapter/stages/HPS/coordination/state codec；C publisher当前changes，只消费其已固定data envelope规范，不调用未批准publish。

## 唯一目标/允许范围

实现**实际H3对象→受控模型调用、G3/current-only数值证据、真实checkpoint合同/进度/data envelope的生产绑定**。这是下一步唯一RunController装配所缺的接口，不新增runner或又一套数学。

仅新 `visual_rl/runtime/h3_binding.py`、新 `tests/test_h3_mvp_cpu_binding.py`；允许在 `visual_rl/models/implementations/h3.py::_prepared_forward` 作最小委托改动，消费现基类 `_forward_prepared` 的已绑定数值scope（保持未绑定轻量测试现行为）；必要时仅给你拥有的 `algorithms/trainer/h3_stages.py` 添加完整tail边界只读检查方法。W/coding-a/H3-M4c1-runtime-binding.md。旧通用runtime/controller/composition/manager/codec/profile/data/七项及其他tests只读；不引用B/C本波新增符号。

## 生产接口与实际证据

- 接收真实H3Profile/H3LocalFacts、已加载H3Adapter、现prepare_h3_fsdp返回handle、真实H3PromptSource、已acquire的H3HPSReward、实际WORLD/rank/device及真实artifact/code/topology绑定。严格验证对象/彼此指向：adapter.meta_model.model就是prepared的transformer，最终optimizer参数集合、LoRA范围/dtypes、FP32 islands/rope、profile几何/K/N、精确双schedule/eta、cache/model/HPS124 contract与source身份一致。
- 必须区分profile.declared_model_sha256与**实际全内容核验**：复用现FilesystemArtifactIdentityResolver或其已有内容hash帮助完成本地model/base/VAE权重身份和code身份（不要重写大hash框架）。给出一致可复现的具体content-hash定义，与expected/cache声明核对；header预检facts.model_weights_verified=False不能当G3。可封装独立真正验证函数供未来preflight调用；未验证时bind拒绝，不提供verified=True布尔绕过。测试用真实tiny文件hash，不把mock artifact字符串当事实。
- 元数据/资源核验只读、不会联网/建PG；HPS对象此前显式acquire，其已验证identity须匹配profile，本项不隐式acquire第二client。真实硬件UUID授权/空闲准入留唯一runtime session，不能用格式正确当实际可用。绑定expected topology是严格固定WORLD8，同rank设备/UUID映射由已准入runtime传入并进入checkpoint身份。
- 最终prepare之后构造/观察ParameterStateManager/ModelStateProjection或现等价结构，不能在meta上冻结Parameter身份；核验后用真实ParameterDTypePolicy(preserve frozen/buffer, FP32 trainable)、ParameterViewEvidence(current only)和ModelExecutionNumericsEvidence形成现typed证据。数据来自实际最终FQN/shape/dtype/owner与真实参数视图；不是全0摘要、空owner、fixture receipt，也不为方便整体cast模型。
- forward scope使用现ForwardAutocastPolicy/StageExecutionPolicy等：H3无root autocast，实际prepared子blockBF16与root FP32岛已核验；当前view唯一、无EMA/reference。为现H3Adapter绑定证据与阶段scope provider，并通过_base _forward_prepared_消费；scope缺失/错view/dtype或身份漂移早拒。不要静默绕过基类证据，也不要把“compute BF16”误施加整个root。
- 返回可交H3IterationStages的真实prediction/decode闭包：predict从source.conditioning_for(sample_id)拿实际typed条件，核对传入condition一致，按精确dynamics.transition(index)的两个sigma构造H3ModelInput，调用adapter.predict；不raw unwrap，不平均sigma/index冒充t。decode通过adapter的CPU terminal spec/AV lease；score_one直接真实HPS对象；coordination_factory返回真实H3Coordination(plan,PG,device)。没有noop/常量/旧Gaussian fallback。
- 构造现CheckpointContract与H3StateIdentity/live bindings：所有recipe/算法/执行/reward/source/artifact/code/preprocess/numerics/model projection标识来自实际H3冻结payload和已核验内容，固定canonical namespace，逐字段列明来源，不声称已经通过旧Gaussian materializer。若现类型确实无法表达该H3事实则报告最小接口请求，不伪造成功receipt。
- 提供完整tail后的checkpoint边界转换：实际trainer.next_optimizer_step>0、source.state_dict无pending/poison、stages无active/poison、optimizer gradients None，step/behavior/cursor一致。用实际已提交next-step构造既有SafePoint/Progress，不能在optimize结束而BaseTrainer尚未next-step时给safe proof。
- data envelope必须满足现H3CheckpointTransaction：plain SHA content_id/next_global_batch_id可从source**已验证指定namespace**精确提取digest，不能泛化str.split丢类型。source_id、rank/world、cursor、behavior、source payload真实回读。progress.rng_state_id先用真实本rank RankRNGSnapshot身份（publisher随后替换有序global摘要），无零占位。all-N selection metadata明确无随机选步，不伪造不存在的selection RNG cursor；source的实际stream版本另进语义。
- 返回的live-validation能力每次重新核验当前文件/模型/score/cache/data绑定，不只是返回缓存dict；future restore_bound调用它。生命周期归属清楚，binding不重复关借用adapter/prepared/scorer或销毁PG。

## CPU验收

真实profile正常fixture（不依赖B新对齐符号）、真实文件hash/cache/source、loader/PEFT/prepared guard、HPS wrapper/transport叶子和typed证据；仅重模型/VAE/CUDA/collective等叶子替换。验证prediction实际经adapter typed端口与数值scope、两流backward、decode lease和真实source条件，未绑定/错view/参数身份或precision漂移/资源内容漂移拒绝。

实际构造可被现CheckpointContract校验的完整真实对象合同，修改model/cache/score/schedule/data/parameter事实会改变身份或拒绝；不能测试只assert字符串长度。让真实H3IterationStages/BaseTrainer完成一轮后边界转换，证明step/cursor/behavior与源一致；在reserve中、optimize后tail前、poison和未清grad时拒绝。不得调用未获批准的publisher来宣称完整checkpoint；本项不实现default controller。

最后编辑后plan固定Flash7/H3通配符，无skip/xfail/全量/过滤warnings；报告branch/HEAD、命令/exit/数量/时间、diff/check与下一默认生命周期必须使用的精确接口。若范围大到需额外公共文件，报告具体最小拆分，而非填假G3或另写完整Runner。Simplifier A重点审证据真实性/模型受控调用/完整tail与无重复owner，changes回A。GPU_PENDING。
