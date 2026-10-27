# H3-M2b2-profile — Simplifier B / W5

verdict=approved

仅批准固定profile编译/只读本地预检原子项；不批准默认入口、资源准入或整run恢复。outcome done仅表示review结束。

## 身份与范围

- HEAD：`230696472b735966d3ac17102120a444f4ff8697`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-f4f41edd-0c54-414a-85ec-780013256fd6`；接收工作区干净。
- 已读当前task/plan/assignment B/wave-05、Coding B交付、完整`h3_profile.py`、样例YAML、六项新测试与实际严格source loader；核对已有H3Config/cache/header/decoder/HPS合同，并查看同波data的解析/stream边界以识别后续装配要求。
- 未修改源码、测试或其他配对报告。

## 实现与简化核验

1. `visual_rl/composition/config/h3_profile.py:159–227`：复用现`load_source_recipe`与实际Registry/DeclarationResolver/H3Config；严格根/分组/字段集合，无动态class或任意dict透传。WORLD8/SP1/TP1、124帧、K/N范围、budget/seed/cadence/resume策略、完整身份和八个不同UUID格式明确校验，bool/非有限值拒绝。
2. 同文件`:81–105`：固定joint CPS/元素均值、FP16 action/FP32 score、独立shift12/3、eta.7、全N含q0、frozen replay old、B1/chunk1、beta0/current-only及Adam/PPO数学进入显式payload；不继承Flash配置。video-only HPS限制保持，不删除音频状态。
3. 同文件`:230–326`：只读真实prompt bytes/hash、精确覆盖和逐条cache finite检查；复用decoder config与真实safetensors header准入，不加载模型或get全权重tensor。预算依据固定K×N完整轨迹、最大单条condition及实际media/decoder估算拒绝不足，不缩任务。HPS仅本地manifest校验，未healthcheck。
4. `H3LocalFacts`及semantic payload清楚区分已读内容与`declared_model_sha256`；`model_weights_verified=False`。完整模型内容hash/API能力、checkpoint complete/进度/拓扑、GPU授权/空闲均未获证明。不能将此semantic ID替代materialized recipe、G3或真实运行准入。
5. `H3Launch.binding_id`与内容semantic ID分离：路径、output/resume、UUID顺序、预算、endpoint/cadence为执行绑定；数据/cache内容、双grid、训练数学/seed等进入语义。目录迁移同内容稳定及内容漂移在测试中实际验证。
6. `configs/v2/h3_t2va_mvp.yaml`明确标注待填模板；null路径/身份/host预算及空UUID列表被测试要求拒绝，不生成假资源ready。没有创建输出目录或第二训练入口。
7. `tests/test_h3_mvp_cpu_profile.py`六项执行真实YAML/declaration/cache/safetensors/protocol/preflight；失败spy禁止模型/VAE/HPS/HTTP/PG构造。旧SD3/Wan/Flash代表性compile_recipe_v2实际仍成功且同输入fingerprint稳定；这不是额外全量测试。

无本原子项必须返工内容，返工清单为空。冻结profile类型和两个入口是明确分工；未扩展成通用配置框架或Runner。少量compile期cache身份字段检查用于无I/O早拒，实际内容仍交原reader，未复制读取/评分算法。

## 独立门禁

本轮汇合树真实执行，均exit0，无skip/xfail、未跑全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 126 passed, 16 warnings in 16.95s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.30s
```

126为此前108+W5三组各6。16条warning为checkpoint单进程DCP及损坏fixture覆盖提示，真实保留未过滤；不作为多rank DCP证据。

获准原Flash文件相对`d818b8a231b52a9a024fa2f522057acdbb7bf099`完整历史diff无输出、exit0；全部原断言与该基线一致。写报告前`git diff`无输出、`git diff --check` exit0。当前clean不替代逐branch原子snapshot归属核验。

## 必须保留的后续装配边界

- profile尚未接原默认compiler/controller；下一项必须经唯一RunController消费，不把样例或typed结果冒称可运行闭环。
- 同波A数据源与本profile没有互相import，不能假设两个局部准入完全相同：当前A明确拒开头UTF8 BOM、文件上限固定16MiB；profile解析没有单独BOM拒绝且文件预算可显式更大。A还使用明确的PARSE/SAMPLING/STREAM版本常量，profile目前记录描述性策略。下一runtime绑定应核对并收敛这些具体边界/版本；本报告不声称真实data/profile组合已验证。此处未越界修改A/B接口。
- 模型权重完整身份需由真实artifact materializer核验；UUID格式不等于授权；HPS124仍须显式live准入，旧81帧服务不可借用ready声明。
- 后续绑定真实G3/current-only数值/preprocess、数据对象、stages、prepared与完整manager两阶段fresh恢复；header/目录存在不等于模型或checkpoint可用。

本轮没有GPU/服务器/HTTP/PG初始化/下载/环境升级/push。只批准B原子项，不替代A/C源码审查。**MVP未完成，GPU_PENDING。**
