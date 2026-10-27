# W2 — rollout pool / AV decode / world credit

共同基线 `47546f94e7490dafcc3fdbc2eb57279f0839899d`，W1三份配对报告均approved，orchestrator汇合门禁Flash7/11.70s、H3通配符44/3.09s均exit0。仅实现/CPU接受，非GPU或完整MVP。

## 独立写入与锁定接口

| 配对 | 任务 | 写入源码 | 独立测试 |
|---|---|---|---|
| A | H3-M4a2-rollout-pool | algorithms/trainer/h3_pool.py | test_h3_mvp_cpu_pool.py |
| B | H3-M2a3b-decode | models/implementations/h3_decode.py | test_h3_mvp_cpu_decode.py |
| C | H3-M3b1-world-credit | algorithms/optimization/h3_world.py；必要时限定h3_fsdp.py固定reducer divisor | test_h3_mvp_cpu_world.py；仅reducer相关prepare测试扩展 |

源码前缀visual_rl/、测试前缀tests/。A/B不改现anchor/forward/loader/prepare；C不改A/B模块。不能import同波尚未生成文件。现基础测试及原七项不动。

共同最小数据交换不需要新共享类型：样本元数据 `(sample_id:str, group_id:str, member_index:int)`；group_id是prompt occurrence，不是prompt文本。collection_id与behavior_id由调用者显式提供，condition cache/模型身份由后续runtime绑定。

- A池输出有序CPU B1 CPS记录 `(sample_id,record)`，CPU FP16终态 `[1,D]` 与原样sample元数据；不调用B decoder/C world。未来阶段逐样本取terminal送B、reward结果附回sample元数据送C。
- B只输入既有geometry和CPU FP16 terminal，输出CPU FP32 video `[T,3,H,W]`、audio `[L,2]`、fps/sample_rate与实际长度；不import A池、不执行reward。
- C只传有界小表，reward行包含上述三元组和有限浮点reward；任何媒体/latent/模型对象禁止collective。固定同world/rank工作量在模型phase前准入；返回按sample_id可查的FP32 credit及显式loss缩放。

每组实现自己的函数真实消费者测试：A实际采样并产出可被既有anchor消费的轨迹；B真实解码控制仅替VAE重量级叶子；C真实表校验/归一化与既有PPO目标梯度对照。彼此无需未稳定接口，故三个active parallel。

## 不在本波的实质待办

world全阶段健康门/节点aux轮转、global grad norm、真实reward binding、registry/recipe/ModelAdapter与现six-stage/controller接线、完整tail checkpoint/fresh恢复仍须开发，不能视为性能优化。W2结束后优先阶段接线，避免不断增加无消费者前置模块。阶段门负责匹配actual forward/backward；本波的等量计划检查不是所有collective故障可恢复保证。

审查/门禁：每配对实际分支Flash固定七项+H3通配符；三份审核均审各自项，changes返原coding。汇合不代表批准，orchestrator再跑完整目标。无服务器/GPU/下载/升级/push，GPU_PENDING。
