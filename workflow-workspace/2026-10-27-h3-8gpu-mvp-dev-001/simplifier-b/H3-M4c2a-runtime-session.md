# H3-M4c2a-runtime-session — Simplifier B / W7

verdict=approved

仅批准WORLD8 session owner原子项；outcome done表示review结束，不批准A返工/C组合测试或默认入口完整闭环。

## 身份与材料

- HEAD：`6a7616aa26a63215cdae2b1cdd091dc33204f9e4`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-f9cd7886-1f00-4cb5-afda-748f9d07c0aa`；接收工作区干净。
- 已读最新task/plan/assignment B/wave-07、Coding B交付、完整session源码与七项feature测试；核对实际h3_world的WORLD8检查及有界同host receipt实现。
- 未修改源码、测试或其他配对报告。

## 实现与边界核验

1. `visual_rl/runtime/h3_session.py:42–103`：检查真实profile/facts类型、binding/semantic内部一致性及模型/数学/数据/cache声明；torchrun环境严格WORLD8/local8、单节点group、rank=local_rank、显式master/port。八个有序完整UUID与CUDA_VISIBLE_DEVICES精确匹配，无索引/十卡/MIG/default替代。
2. 同文件`:119–155`：公开CUDA/distributed API lazy调用；拒外来PG、CUDA/NCCL缺失。真实device_count/每张逻辑卡uuid映射检查，选定并回读cuda:rank，公开native BF16查询不允许emulation。缺UUID属性或BF16查询能力明确失败，不依据卡名/显存推断。
3. 同文件`:156–188`：恰一次env:// NCCL初始化，明确180秒timeout。成功返回后保存WORLD并核对实际backend/world/rank；复用现_exchange做有界同host/profile/facts/ordered UUID/rank一致性。没有模型/媒体gather、新RPC、子组或重试。可协作错误保留cause，raw transport错误不补finally collective。
4. 同文件`:190–225`：公开属性只读；close先失效owner引用，只在保存PG仍为当前WORLD时销毁该对象，幂等且无额外barrier。外部替换WORLD不误销毁新owner；create后失败清理自己的已取得PG，cleanup异常不掩盖主异常。
5. `tests/test_h3_mvp_cpu_session.py:130–332`：七项实际执行session控制，只替CUDA/PG观察与通信叶子。覆盖八rank合法观察、API参数/顺序、profile/env/UUID/能力负例、外来PG、同host/metadata分歧、创建后错误、timeout/no retry、替换WORLD和一次性清理。重模型、VAE、HPS/HTTP、optimizer、barrier/new_group均有禁止调用spy。

## 简化与证据限制

无必须返工项，返工清单为空。一个窄owner与局部validation函数足够，没有第二controller/训练loop、Accelerator或资源平台；复用既有receipt而非建立另一通信协议。

此准入仅确认请求与实际可观察CUDA映射/能力一致，不证明用户授权、独占窗口、显存/host余量或权重/HPS可用。CPU替身不证明目标环境实际提供uuid/native-BF16 API，也不证明NCCL执行成功。初始化返回前的异常不获得可确认的PG所有权；PG建立前失败、设备上下文损坏或死rank仍依赖有限timeout和外层torchrun终止attempt，不承诺继续训练或任意故障均能协作通信。已选CUDA device不被reset，未作全资源回收声明。

当前尚未接默认create_runtime/RunController；下一项必须明确session最后关闭的生命周期顺序，不能以本helper通过替代默认入口验收。

## 独立CPU门禁

当前汇合树实际运行，均exit0，无skip/xfail、未跑全量、未过滤warning：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 146 passed, 22 warnings in 25.63s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 13.47s
```

146为此前135+W7 A2/B7/C2。22条checkpoint no-dist/损坏fixture警告原样保留，不是实际多rank DCP证据。

获准原Flash完整历史diff相对`d818b8a231b52a9a024fa2f522057acdbb7bf099`无输出、exit0；原七项完整断言不变。写报告前`git diff`无输出、`git diff --check` exit0；clean不代替逐branch snapshot归属审计。

## 后续

Orchestrator登记本项snapshot/approved，结合A/C各自裁决并亲跑汇合门禁。随后优先接现composition/唯一RunController/default create_runtime及清理顺序，完成真实连续训练与两阶段fresh整run下一步对照。

本轮没有真实GPU/PG/网络/服务器/下载/环境升级/push。**MVP未完成，GPU_PENDING。**
