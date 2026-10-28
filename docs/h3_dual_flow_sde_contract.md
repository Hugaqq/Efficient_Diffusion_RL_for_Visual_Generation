# H3双流Flow-SDE／Flash扩展：实验数学合同（未接线）

该合同与现有H3 CPS相互独立，不覆盖原profile或checkpoint，不声称官方Flash数学等价或质量改善。

- 两模态共用transition index，但分别使用H3 video/audio sigma网格；网络输出须是noise-ward velocity，与现有H3绑定一致。
- 随机转移：每模态独立使用Flash参考Flow-SDE均值及非零方差，sigma_max取各自网格第二个点、sigma_min=0；video/audio各用独立generator。以实际存储后（可能量化）的联合action进行replay评分。末个transition可随机，因当前sigma>0、dt<0、std>0；索引N或num_steps<2拒绝。
- 确定性转移明确为每模态Euler x+(u-s)v，不消费随机数，不产生Gaussian log概率，不可作为policy transition参与PPO；指定策略步之外使用此规则的rollout方案仍须接入并验证。
- 对同一被选transition，分别计算两模态每元素Gaussian log概率均值，joint值为按照各自元素数量加权之和。这个联合值用于PPO旧/新比值，未作跨模态系数平均。
- 时间校正对每模态从其自身SDE均值/方差导数获取系数；分别在全局参与样本集合上求该模态的均值，再将对应模态的score梯度乘以本模态系数/本模态世界均值。使用`component-component.detach()`保持joint score和PPO比值前向值不变，仅修改每模态梯度。这是明确的双流实验设计，尚未证明与官方单流Flash等价。不能把两个模态系数求平均或直接把某一模态系数用于另一模态。
- Flow对照须使用相同双流SDE、联合评分、采样步数、数据、奖励及资源，仅按事先冻结的策略步/校正规则区分；不得混用CPS结果冒充同底层对照。Flow/Flash各自完整5 optimizer step、真实reward/checkpoint和退出证据为先决条件。

当前只实现独立数值组件和CPU测试；未接入trainer、world mean归一或生产配置，未GPU验证，不能训练或部署宣称完成。
