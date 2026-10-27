# H3-EXP-002 — 后续必需的真实Flash single-step路径

状态：queued，先修H3-EXP-001模型准入。依据当前d5c10c9真实源码检查，非GPU训练失败栈。

`visual_rl/composition/config/h3_profile.py::H3Training`仅允许group_size/num_steps，payload固定`policy_steps="all_including_terminal_q0"`；H3 lifecycle只有`h3_t2va_mvp_v1`入口，stages使用H3RolloutPool及H3ReplayAnchor完整schedule路径。现有其他模型Flash代码不能证明H3 single-step已接线。

总需求要求Flow与Flash分别真实5更新，Flash必须真实独立single-step policy/credit/replay，保留joint AV和正确old anchor。需要Coder在模型准入后补齐最小生产接线，记录与Flash论文rectification/策略定义的差异，不把全步CPS改名或仅修改resolved_recipe标签。生产recipe应可追踪实际选定policy index/credit/replay与采样机制；Experimenter将独立真实GPU验证。

正式global_prompts=4/samples_per_prompt=8/global_batch=32，而当前WORLD8每rank完整group产生8个prompt groups；正式启动前须实现/核验真实分组与world数学，不能改成8×8或缩小用户batch。两个smoke可先明示较小规格，此点不阻止环境与Flow5。正式步数仍null，待两smoke成功后询问用户。

本轮未修改算法、未运行CPU回归，Flow/Flash实际更新均0。
