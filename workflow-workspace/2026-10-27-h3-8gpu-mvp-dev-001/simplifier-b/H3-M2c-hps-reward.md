# H3-M2c-hps-reward — Simplifier B / W3

verdict=approved

仅批准本配对HPS评分适配原子项；outcome done表示review结束，不替代A/C裁决、真实服务准入或完整MVP验收。

## 身份与审核范围

- HEAD：`5397c46662db30ef535afc0fb4b192df019c4d62`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-5a01be5f-31c4-45a7-87be-029c9f464945`；接收时工作区干净。
- 已读当前task、plan、assignment B、wave-03决策、Coding B交付、完整`visual_rl/algorithms/rewards/h3_hps.py`及六项独立测试。核对原HPS client/factory、JPEG、完整protocol、bounded transport与typed acquire request；检查汇合A stage的实际score_one调用位置。
- 未修改源码、测试或他人报告。首次误用`visual_rl/rewards/h3_hps.py`读取返回ENOENT，随后按assignment正确路径读取；无阻塞。

## 实现/边界核验

1. `h3_hps.py:50–91`：真实RewardResourceAcquireRequest、固定原factory/protocol、显式service与VideoContract声明；constructor仅有界读本地manifest，调用原health validator，不创建transport。拒81帧、缺文件/symlink及manifest漂移。materialized artifact身份的整体上游准入仍由未来runtime负责，交付未伪装已经完成。
2. `h3_hps.py:93–128`：acquire直接调用原`Hpsv3VideoResourceFactory`，保留loopback/trusted host、remote CPU/fp32、timeout/response bound、本地artifact与live health精确一致性。独占拥有返回client，失败/重复acquire不自动重试，close幂等；小冻结identity明确profile/version/weight/service/contract/manifest/endpoint。
3. `h3_hps.py:130–164`：严格CPU detached FP32 video/audio及实际metadata；所有124帧调用原JPEG和原client，原validator验证全frame scores及sample/service身份并本地推导top37；不信任单scalar、不重写topK。audio强制存在、finite、正确长度/rate，但不发送HPS或额外clip。返回有限Python float，不保媒体/JPEG/响应到对象字段。
4. `tests/test_h3_mvp_cpu_hps.py:143–343`：六项实际使用原typed request/factory/client/JPEG/health/score/transport边界，仅替HTTP transport叶子。两个画布、不同视频按实际JPEG像素产生不同fake-service分数，独立top37对照；覆盖identity/frame/topK/额外indices字段、非法AV/过期lease、请求响应字节上限、HTTP状态/redirect/timeout/no-retry及manifest失败。没有mock wrapper或校验器，没有真实HTTP。
5. `visual_rl/algorithms/trainer/h3_stages.py:260–274`：实际在decode lease内按`sample_id/prompt/media`调用必需score_one，并要求finite Python float；与本wrapper结构兼容。当前各feature测试通过不等于已经运行A+B+C真实对象的联合装配测试，仍由orchestrator安排汇合验收和后续runtime接线。

## 简化及reward语义

无必须返工项，返工清单为空。实现复用现有协议/服务工厂，不增加HTTP客户端、JPEG/topK实现、shadow scorer或通用服务平台；本地声明检查用于离线早拒与漂移绑定，没有替代live准入。

wave-03明确选择`h3_hpsv3_video_only_v1`、唯一权重1、all124/top37。task允许现有评分接口，该实现符合本轮显式决策；不评价音频语义/AV对齐，不声称CLAP/ImageBind复现或音频质量。联合AV状态/CPS/解码没有被删。未来真实124帧HPS资源必须另行准入，旧81帧服务不能直接当作ready。

## 独立CPU门禁

本轮在汇合树实际执行，均exit0，无skip/xfail、未跑全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 93 passed in 4.67s（此前65 + W3 stages14/HPS6/coordination8）

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 13.90s
```

获准`git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py`无输出、exit0，原七项完整文件和断言未变。写报告前`git diff`无输出、`git diff --check` exit0；当前clean不代替逐分支snapshot归属审计。

## 后续/未验证

Orchestrator记录本项snapshot及approved，结合A/C审核并亲跑汇合门禁；实际联合对象装配、recipe/materialized request绑定、acquire/close归属与唯一RunController仍须完成。完整tail checkpoint/fresh恢复仍未验收。

本轮没有服务器/GPU/真实HTTP/部署/下载/环境升级/push。fake-service结果只证明协议/控制，不证明HPS模型、服务资源、视频或音频质量。**MVP未完成，GPU_PENDING。**
