# Assignment A — W2

- active: true
- task_id: H3-M4a2-rollout-pool
- pair: coding-a → simplifier-a
- execution: parallel W2；decisions/wave-02.md
- integration_parent: `47546f94e7490dafcc3fdbc2eb57279f0839899d`
- dependencies: M1a/M1b、W1 anchor approved；W1源码汇合 `467c8fd929804aa291bf01a5fa7cda372cc507db`。无changes；实际新分支/HEAD由引擎提供。

## 唯一目标/文件

实现**rank-local真实CPS rollout→有界CPU latent-only池**，为后续现RolloutStage/RewardStage提供实际轨迹，不另造训练runner。

只写新 `visual_rl/algorithms/trainer/h3_pool.py`、新 `tests/test_h3_mvp_cpu_pool.py`、W/coding-a/H3-M4a2-rollout-pool.md。已有anchor/CPS/forward/tests只读，不import或修改B/C本波文件。

## 最小合同

- 单collection、一个固定behavior；显式给定有序样本三元组 `(sample_id, group_id, member_index)`。唯一sample、组内唯一非负member、完整固定K成员（K>=2）、局部完整group，ID不从prompt文本生成。绑定精确CPS合同、明确max_samples/max_cells/max_trajectory_bytes，预估完整N步x/action/score+terminal存储容量，任何model调用前拒超预算。
- rollout输入每个sample的detached CPU FP16初态、显式独立video/audio generators，以及 `predict(sample_id, x_on_device, transition_index) -> noise_ward_prediction` callback、target device、read_behavior_id。初始噪声由上层显式提供，不能全零代替真实采样或重新seed补失败样本。预检全部样本shape/finite/身份/数量/合同，在进入循环前失败；输入会被拥有或受控复制，不能让调用方mutation改已收集轨迹。
- 单次collect，在no_grad/EVAL由caller保证固定behavior，按sample-major再index0..N-1固定顺序：同模型callback一次，现 `H3CPSDynamics.sample` 一次，使用action作为下一x；两流完整演化，包含q0末格，沿用CPS显式RNG消费。每步仅当前设备状态/record，立即保存拥有的CPU FP16 x/action及FP32诊断record，不累计设备轨迹、图、媒体或模型引用。不要复制CPS数学、不要强制decode/构造假media。
- 输出完整有序 `tuple[(sample_id,H3CPSRecord),...]` 或等效受控视图，以及按sample_id查CPU FP16终态的拥有副本、原sample元数据。record由现CPS public接口构造，不用内部buffer，不另造record格式。池是可信sample/collection/behavior来源，reader核对这些身份，终态必须恰等于最后stored action；记录旧rollout diagnostic并不等于replay anchor。
- 失败不可公开部分可训练pool、不可局部重试/换seed补样；捕获前后行为版本漂移拒绝。状态最小new→ready/failed→released即可，不建通用事务框架。`release`幂等释放引用；真实调用必须在完整trainer tail后由未来owner执行，本项不假装optimizer/data tail已经提交。
- 不做reward/credit/anchor捕获或更新loop。目标测试可把池输出传现H3ReplayAnchor，`predict(record)`所需sample条件通过**明确有序闭包**映射（先核对key/index），无需改anchor公共签名；未来stage如需新接口单独协调。

## 验收

真实tiny trainable模型叶子+已有H3 forward/CPS完成多sample多步采样；准确call顺序、N步全保存、双模态链连续、所有stored tensors CPU detached、terminal等最后action、同seed一致/不同seed有变化、无全局RNG污染。有限显式容量在第一个predict前拒绝；重复/不完整组/非法初态/版本漂移/中途非有限无部分ready池，失败不能重试。外部输入/读结果mutation不污染池、release后读拒绝且不保callback/model。

实际池→现anchor capture→既有objective replay/backward示例测试（tiny叶子），证明不是无消费者结构；精确记录这不是controller或全rank闭环。不得缩减已有断言。

最后编辑后原样跑plan两命令：Flash七项与tests/test_h3_mvp_cpu*.py；无skip/xfail/全量。交付命令/exit/数量/时间、实际branch/HEAD、diff/check与接口；不手工commit/merge。Simplifier A审有界性/状态链/身份/失败原子性/无重复数学或训练runner，changes回A。GPU_PENDING。
