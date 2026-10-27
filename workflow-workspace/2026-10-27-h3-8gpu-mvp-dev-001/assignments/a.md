# Assignment A — W4

- active: true
- task_id: H3-M4a4-object-integration
- pair: coding-a → simplifier-a
- execution: parallel W4；decisions/wave-04.md
- integration_parent: `7cf28dced4c8193ccf9321d62b5b2e479bd904f7`
- dependencies: W3 stages/HPS/coordination分别approved，代码汇合5397c46662db30ef535afc0fb4b192df019c4d62；真实对象组合仍待验。

## 唯一目标/写边界

**补齐W3三真实对象同链CPU验收**。只新增 `tests/test_h3_mvp_cpu_integration.py` 与W/coding-a/H3-M4a4-object-integration.md；现所有生产源码/测试只读。发现真实缺陷先报告具体文件、复现与最小修正，回原owner再测，不在本项更高层mock掉问题、不改兄弟源码。不得修改现93项或原七项。

## 必须实际执行

- 原GRPOTrainer/BaseTrainer与H3IterationStages，真实H3RolloutPool/H3ReplayAnchor/H3WorldPlan、真实world credit、ClippedSurrogateObjective、真实Adam及data tail。
- **真实 H3HPSReward** acquire/score_one/close，沿现factory/client/JPEG/health+score validator；只HTTP transport leaf替身，根据实际提交JPEG内容返回有限可变frame scores，严格all124/top37。不能用lambda reward替wrapper。
- **真实 H3Coordination** init/gate/aux_turns/clip_grad_norm。不得继续用_CoordinationLeaf、本地clip_grad_norm_替换整个协调器。可复用只读feature helper的MemoryWorld或独立有界8线程内存通信leaf；禁止真实PG/网络/GPU。
- 重H3/两个VAE叶子tiny；CUDA admission、FSDP通信/DTensor local-shard观测叶子可明确替换，但stage/coordination/统计/norm控制本身实际执行。prepared用现真实guarded H3PreparedModel、最终Adam参数检查。若使用CPU local shard仿真，准确说明它不是实际DTensor或CUDA归约，不凭它声称真正FSDP数值等价；已有world/norm独立数学对照不能丢。
- 每个模拟rank完整K>=2、N>=2步，至少两次完整iteration；真实计划、不同rank样本、版本/condition绑定，确保不是把rank0回执复制8份然后跳过其他work。线程通信要有有限超时且失配明确失败，不能死等或吞异常。避免在每rank模型构造中改全局线程数/共享环境。

## 验收断言

1. 8rank相同event/sequence推进；完整rollout后逐rank aux；实际work最大并发1，媒体无跨rankgather，HPS每sample一次真实protocol POST，all124帧；端口构造/acquire/close所有权明确。
2. 每rank旧锚完整冻结后才current，N含末格；正常同步语义的每格backward/唯一norm、一次Adam/scheduler、data/behavior/trainer next-step连续两次；pool在完整tail后release。保存足够事件日志证明顺序，而不是只assert次数。
3. 不同媒体fake transport分数驱动非零信用/参数变化；零信用另例仍真实backward与普通commit。校验阶段输出与独立数学参考的界限，不用tiny参数变化冒充完整同模型8卡证据。
4. 至少一个单rank aux/评分协议错误与一个post-forward finite错误让所有存活rank共同停止、无optimizer/data tail、poison后不可重试；没有失败finally补救collective。raw forward/backward不可到门失败只可说明需launcher退出，测试不制造无限挂起。
5. 当前worktree导入身份、无真实HTTP/PG/GPU；禁止skip/xfail。目标若失败不可删断言或简化为只核静态签名；交具体接口/事件缺陷给orchestrator。

最后编辑后原样跑plan Flash七项和完整H3通配符，记录真实branch/HEAD、命令/exit/数量/时间、所有替身叶子与diff/check。新增测试不计作生产controller装配。Simplifier A重点审三真实对象是否确实组合、没有高层mock遮蔽和collective失败语义；通过后才关闭W3组合待办。MVP仍缺注册runtime/controller与checkpoint，GPU_PENDING。
