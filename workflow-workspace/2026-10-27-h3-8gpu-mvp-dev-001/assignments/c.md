# Assignment C — W2

- active: true
- task_id: H3-M3b1-world-credit
- pair: coding-c → simplifier-c
- execution: parallel W2；decisions/wave-02.md
- integration_parent: `47546f94e7490dafcc3fdbc2eb57279f0839899d`
- dependencies: W1 prepare配对approved，源码汇合467c8fd929804aa291bf01a5fa7cda372cc507db；现objective/advantage稳定。无changes待修。

## 唯一功能/允许文件

实现**固定等量rank工作量准入 + 完整world reward统计/credit与明确loss尺度**。不做整套phase scheduler或optimizer执行器。

允许新 `visual_rl/algorithms/optimization/h3_world.py`、新 `tests/test_h3_mvp_cpu_world.py`；仅为固定/暴露真实FSDP reducer divisor可精确修改 `visual_rl/models/lifecycle/h3_fsdp.py` 与 `tests/test_h3_mvp_cpu_prepare.py` 相应断言（不删旧断言）。W/coding-c/H3-M3b1-world-credit.md。其余已批准模块/旧tests只读，不import A/B本波新模块。

## 数据/通信合同

- 生产单节点WORLD8/SP1/TP1，使用调用者已初始化PG，不init/destroy。每rank相同正样本数、固定group_size K>=2、N policy步、rollout/replay B1、chunk1、guidance1/beta0与共同geometry/score profile；一次完整collection一次update。各rank sample metadata为有序 `(sample_id,group_id,member_index)`，group_id为prompt occurrence，local_complete组成员0..K-1恰齐。全world sample唯一，group不可跨rank拆散；不按重复prompt文本合组。
- 公开小型plan/admission函数在任何模型调用前收集并验证全rank有界metadata：collection_id/behavior_id、rank/work量、N/geometry/score标识和样本键。空/uneven/重复/缺member/不一致schedule/profile先拒；不是仅检查sum除得尽。返回固定global sample顺序/索引或最小不可变plan，后续phase实际按此序执行；不传媒体、latent、tensor权重。
- 本rank可协作预检失败要先编码为有限小错误receipt并参与同次健康汇合，让所有存活rank统一拒绝；不能某rank本地raise而其他开始model collective。死亡/collective内部失败仍由上层timeout退出，本模块不承诺救活。
- reward输入为每个样本三元组+finite float reward，已由真实scorer按固定权重合成；汇合仅小表。绑定plan/collection/behavior，完整每条sample恰一次、identity吻合、valid全真；NaN/缺样/重复/换组/成员或非法rank拒绝，不能丢样重归一化或换reward。设置明确表大小/样本上限。

## 数学锁

- 组内mean，**完整唯一world collection** std（ddof0），CPU float64计算 `(reward-group_mean)/(global_std+1e-4)`，输出detached CPU FP32按sample_id对齐。无额外advantage clip，不继承旧±5。全相同reward输出精确零信用，不skip optimizer（本项不执行optimizer）。
- 先查现 `GroupZScoreAdvantageProcessor` 数学：若可用真实身份/RewardResult窄适配复用则复用；不要为复用而制造假Trajectory/media/condition。若直接实现H3小表数学，保持极短单一实现并与现processor/独立手算交叉测试，不造第二通用normalizer框架或改变旧路径。
- 每slot当前objective loss为local格mean，系数 `d * n_slot / N_world_cells`，`N_world_cells = world_unique_samples*N`，包括末步q0；不能只local分母、不能又除GA/world/J。正常一格时d8，K8,N10→1/80（经FSDP平均后每全局格1/640）。接口明确n_slot/global_count/reducer_divisor，严格positive/count边界。
- d不能仅因为world=8猜测。核对实际公开FSDP2归约API，优先在现 `_shard_block` 显式设置支持的reduce-scatter divide factor8并检查能力，在prepared结果暴露锁定divisor给上层；不私自改变参数dtype/wrapping/AC。若仅依赖默认平均，需给实际API/实现证据及fail-closed版本/能力锁，不能凭CPU模拟声称真实GPU归约已验。接口不可实现时据实报告，不写恒定伪后端。

## 真实CPU验收

不创建真实PG/网络/GPU。只替collective/设备叶子，实际小表admission、排序/覆盖、归一化和系数实现不mock。8rank tiny表（可K2,N3）证明共享组身份、重复prompt不合并、global std与每rank std不同；local列顺序变换不改最终sample credit。模拟所有rank一次receipt汇合并对非法局部输入统一拒绝，model callback零次。

用真实现 `ClippedSurrogateObjective` / `PolicyLossInputs` beta0、多个trainable tiny rank副本，所有loss格累积并按实际约定d做梯度平均，与单进程完整global cell mean的loss/grad/update容差对照；不复制PPO公式替代所测objective。对uneven/empty、missing/duplicate/invalid reward/合同错配、错误slot计数拒绝；全零优势仍产生图连接有限零梯度。真实reducer setter/选项在独立叶子测试核验，明确不是CUDA collective证据。

最后源码编辑后跑plan固定Flash7项与完整H3通配符，原断言保留，无skip/xfail/全量。记录实际branch/HEAD、命令/exit/数量/时间、公共接口/后端divisor证据与diff/check；不手工commit/merge/push。

## 非目标

不做实际RolloutStage/OptimizeStage/world阶段切换、全局grad norm/clip、optimizer/tail/checkpoint、媒体decode/reward服务或controller。下一项在本plan上接实际同序模型事件/健康门和global norm，不把此表准入当所有故障安全。Simplifier C重点审world归一化/d*n/N、完整group/身份、collective拒绝路径和后端divisor证据；changes回C。GPU_PENDING。
