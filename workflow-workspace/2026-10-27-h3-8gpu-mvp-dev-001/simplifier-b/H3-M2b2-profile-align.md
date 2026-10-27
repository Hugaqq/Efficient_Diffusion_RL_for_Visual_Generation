# H3-M2b2-profile-align — Simplifier B / W6

verdict=approved

B-PROFILE-01的解析、硬上限与版本对齐已核验，可交orchestrator结合汇合门禁关闭。仅本配对批准，不批准C publisher或A runtime binding，不代表默认入口/MVP完成。

## 身份与材料

- 实际HEAD：`c8422b3a780d933564e94dd58c52a5c8efd8f9dc`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-dce489eb-4fbf-4a19-ae37-b716bbf163c3`；开始工作区干净。
- 已读最新task/plan/assignment B/wave-06、对应Coding B交付、完整当前profile、source解析/cap/版本及新增三项交叉测试。核对前次B报告中明确的组合边界。
- Coding报告记录了中断前已返回的局部门禁及恢复snapshot；本次没有依赖其历史结果替代测试，而是在当前汇合树独立重跑。
- 未改源码、测试或他人报告。

## 关闭证据

1. `visual_rl/composition/config/h3_profile.py:243–253`：删除独立文本解析，直接调用`h3_source._prompts(raw)`；实际source拒BOM/空记录/孤立CR，保留精确Unicode与空白并仅剥LF/CRLF终止符。原始bytes SHA在解析前计算，没有normalization后hash。
2. 同处读取上限为`min(limit, h3_source._FILE_LIMIT)`；唯一16MiB硬cap来自source。更大profile预算不放宽source硬限，更小预算仍是明确附加限制。
3. 同文件`:93–106,332`：semantic现有字段直接引用真实PARSE_VERSION/SAMPLING_VERSION/STREAM_VERSION，保留adapter初态版本。公开compile/preflight签名及profile/facts字段结构未扩张；语义版本值改变导致digest改变是正确行为，不承诺与旧描述性ID无校验兼容。
4. `tests/test_h3_mvp_cpu_profile.py:351`：真实相同文件/cache经过preflight与H3PromptSource；使用真实adapter/meta/PEFT/cache，映射compiled参数实际reserve/commit，核对prompt、四项版本、next step/cursor/behavior。覆盖LF/CRLF及有无最终终止符、空白/tab/中文、不同Unicode编码形式。
5. 同文件`:385`：两真实入口共同拒绝BOM、孤立CR、空记录/空文件和坏UTF8。fixture包含对应BOM prompt缓存，避免以cache miss掩盖解析问题。
6. 同文件`:414`：真实16MiB+4字节train/eval文件、真实SHA、32MiB用户预算仍被两入口拒绝；没有调低全局cap或mock读取。充足预算变化只改变binding，不改变内容semantic；小预算额外拒绝符合assignment。

## 简化结论

无必须返工项，返工清单为空。复用现source内部函数/常量是本轮明确的最小边界，不引入共享解析框架或重复实现。import在使用处发生，无新增模型、HTTP或PG构造；交叉测试在显式tiny adapter加载后用失败spy确认预检/source不会再次加载。

原六项profile测试仍存在并随完整目标集通过，包括目录迁移、固定数学、非法schema/资源和旧代表性recipe编译回归。本项只关闭指定解析/cap/版本差异，不把它扩大为完整profile/data/publisher/controller端到端批准。

## 本轮独立门禁

均实际返回exit0，未跑全量，无skip/xfail或warning过滤：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 135 passed, 18 warnings in 14.07s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 15.59s
```

135为此前126+W6 A4/B3/C2。18条checkpoint no-dist及损坏fixture重写警告原样显示，不作为真实多rank DCP证明。

原Flash完整文件相对`d818b8a231b52a9a024fa2f522057acdbb7bf099`获准历史diff为空、exit0，七项断言保持。写报告前`git diff`无输出、`git diff --check` exit0；不以clean替代逐branch snapshot归属检查。

## 后续

Orchestrator登记本组snapshot/approved，结合亲跑门禁关闭B-PROFILE-01；A/C仍由各自配对裁决。下一步继续唯一RunController默认装配、真实资源与数值绑定、两阶段整run fresh恢复及下一步对照。

无GPU/服务器/HTTP/PG初始化/下载/环境升级/push。本项CPU证据不证明H3/HPS真实资源、CUDA/FSDP2/NCCL或质量。**MVP未完成，GPU_PENDING。**
