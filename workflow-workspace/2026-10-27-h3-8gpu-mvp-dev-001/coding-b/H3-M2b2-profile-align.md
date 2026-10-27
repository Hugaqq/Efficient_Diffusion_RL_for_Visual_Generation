# H3-M2b2-profile-align — W6 / B-PROFILE-01

## 状态与恢复核对

- `implementation_ready: true`：本项实现与CPU门禁已完成，待simplifier-b复审；不自行批准或关闭orchestrator组合issue。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-0dd03e52-f2a5-447d-894b-6c051e7d0af7`。
- attempt起点：`75b80699e0380a69c4b7068d19cdba62d420eaab`。
- 恢复后实际HEAD：`54c43b1d3fcb1640fb216c7e76eb4071743fd979`；git log显示该引擎snapshot紧接起点，续跑status干净。
- 中断前所有工具调用已有返回：最终H3通配符129 passed/16 warnings/11.96s、Flash7 passed/11.86s、diff check成功。续跑读取当前profile改动及完整三项新增测试，确认与已保存实现一致；**未重放编辑或重跑测试**，仅补此交付报告。

## 最小源码改动

仅改`visual_rl/composition/config/h3_profile.py`：

1. 删除profile自己的文本解析逻辑，直接调用已批准`visual_rl.data.h3_source._prompts(raw)`。BOM、空记录、孤立CR与LF/CRLF精确文本规则只有source一个实现。
2. 读取前使用`min(prompt_file_bytes, h3_source._FILE_LIMIT)`。source硬上限16MiB为唯一常量；用户声明更大预算不会扩大准入，更小预算仍按小值限制读取与拒绝。
3. SHA仍在解析前针对原始bytes计算；不trim空格、不规范化Unicode、不把CRLF转换后的内容当原文件身份。
4. 原semantic字段保持，但值现在直接引用真实source版本：
   - `data.parse = h3_source.PARSE_VERSION`（当前`h3.utf8-lf-crlf.v1`）
   - `math.sampling = h3_source.SAMPLING_VERSION`（当前`h3.world8-ordered-cycle.v1`）
   - `math.random_streams = h3_source.STREAM_VERSION`（当前`h3.sha256-seeds.v1`）
   - `math.initial_state_version`继续使用既有adapter初态版本，并与真实source state交叉断言。

未改变compile/preflight签名、H3Profile/H3LocalFacts字段或公开配置schema。真实版本值替换旧描述性字符串会改变对应semantic digest，这是有意的正确身份绑定；没有声称与旧描述性ID可无检查互换。路径/UUID/预算/endpoint仍不进入内容semantic，仍有独立binding。

未改A的source或source测试、C publisher、runtime/model、原126项断言及Flash七项；无通用文本框架、新入口或环境探查。

## 新增三项真实交叉测试

只追加到`tests/test_h3_mvp_cpu_profile.py`，已有六项及断言保留：

- `test_h3_profile_source_exact_text_versions_and_real_reserve_commit`：同一真实UTF8文件/cache经过profile预检和H3PromptSource；真实H3Adapter、meta/PEFT/cache运行，仅重transformer类为tiny。LF/CRLF、有/无最终终止符，首尾空格/tab/中文、预组合与分解Unicode均保持精确字符串。把compiled字段映射到source构造，实际reserve→adapter encode/初态→commit，验证step1/cursor8/behavior推进及实际四个版本一致。
- `test_h3_profile_source_reject_same_bom_cr_empty_and_encoding_files`：train/eval BOM、孤立CR、空记录/空文件、坏UTF8分别在两真实入口拒绝。fixture特意缓存BOM前缀prompt，保证BOM负例不是由cache miss偶然遮住旧解析差异。
- `test_h3_profile_source_hard_file_cap_and_smaller_explicit_budget`：实际写入16MiB+4字节文件及其真实SHA；用户预算32MiB时两入口仍拒绝train/eval超限，不修改全局cap。合法文件在不同充足预算下semantic不变/binding变化；更小用户预算额外拒绝，source固定cap仍可接受该小合法文件，这是明确的更严格profile限制。

三项交叉测试在显式tiny adapter加载之后启用失败spy，禁止预检/source重新加载模型、VAE、HTTP session/health或PG。未mock整个compiler、reader或source.reserve/commit。原有目录移动同内容ID稳定、旧SD3/Wan/Flash代表性编译测试亦原样在通配符门禁重跑。

## 已执行CPU门禁（中断前最后编辑之后）

以下结果均已真实返回exit0；续跑未重发命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 11.86s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 129 passed, 16 warnings in 11.96s（基线126+本项3，不含本波A/C未汇合变更）
```

16条既有DCP no-dist/损坏fixture提示未过滤，无skip/xfail、无全量。中断前`git diff --check` exit0；续跑获准原Flash完整历史diff相对d818b8a再次无输出。源码修改已在引擎恢复snapshot中，当前clean不用于冒称逐历史patch审计；补报告后仅本记录新增。

## 后续/边界

交原simplifier-b审B-PROFILE-01；changes回B修复两集重测。Orchestrator据复审和汇合门禁确认组合边界关闭；本报告不批准C publisher，也不重做C任务。

完整模型身份/G3/唯一RunController/整run恢复仍由后续实际绑定完成。本项没有GPU/服务器/HTTP/PG初始化/下载/环境升级/部署/push或手工Git提交。**MVP未完成，GPU_PENDING。**
