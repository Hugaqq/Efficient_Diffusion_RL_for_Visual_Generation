# H3-M2c-hps-reward — W3 Coding B

## 判定/实际身份

- `implementation_ready: true`：仅本原子项实现及本分支CPU门禁完成，交simplifier-b；不是W3汇合批准、服务准入或MVP完成。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-1284df45-1db9-4ed4-a142-65c9f14c87ee`。
- 开始/结束HEAD：`282539fa94c9ca2ec7fda8602e2cc2980ada6d62`；初始工作区干净。未切分支、手工commit/merge/push；snapshot待引擎生成。
- 已读最新task/plan/assignment B/wave-03和本配对decoder approved反馈，完整原Hpsv3VideoClient/ResourceFactory、flash_hpsv3协议及相关真实resource request/binding类型、bounded transport。

## 改动范围

仅新增：
- `visual_rl/algorithms/rewards/h3_hps.py`
- `tests/test_h3_mvp_cpu_hps.py`
- 本交付记录

不改原client/protocol/factory、Flash配置/七项、decoder、已有65项或A/C本波文件；生产代码不import模型implementation或同波stage/coordinator。

## 生产构造/所有权合同

```python
# request由未来runtime的真实artifact materialization/binding提供，非任意mock对象
scorer = H3HPSReward(
    request,  # 原 RewardResourceAcquireRequest
    video_contract=VideoContract(124, height, width),
)
scorer.acquire()  # 唯一显式live health准入；本轮测试只用transport叶子替身
try:
    # 未来A在decoder lease内调用
    reward = scorer.score_one(sample_id=sample_id, prompt=prompt, media=media)
finally:
    scorer.close()
```

request必须包含：
- 真实`RewardResourceAcquireRequest`及原`RewardResourceDescriptor`，factory class明确为现`visual_rl.algorithms.rewards.clients.hpsv3:Hpsv3VideoResourceFactory`，protocol/version原值。
- `semantic_factory_config`恰为显式expected service identity与完整VideoContract payload（含原top_k规则）；allowed runtime policy允许remote/CPU/fp32。
- materialized本地manifest的绝对artifact_location与上游不可变artifact身份；未来runtime仍负责其完整materialization/preflight，wrapper不伪造这些字段。
- 原`RewardRuntimeBindingSpec`明确loopback HTTP origin/port、trusted_hosts、timeout_s、max_response_bytes、remote/CPU/fp32及无TLS bundle。

constructor仅有界读取本地manifest，调用原decode_json/validate_health并绑定实际manifest SHA256；不创建session、不联网。拒缺文件/symlink、过大manifest、非HPS声明、非124帧/contract不符。VideoContract本身拒超过128帧。

`acquire()`直接调用原ResourceFactory，无新HTTP或快捷client。原factory再校验本地manifest与binding，创建真实Hpsv3VideoClient，实际healthcheck并要求observed与artifact完全相同；失败由原factory关闭client。wrapper前后核验manifest未漂移，失败不重试。成功由wrapper**独占拥有**此client；不可再把同client交另一resource pool owner。重复acquire/失败后自动重试均无，close幂等且关闭transport，关闭后score拒绝。

## 评分语义/数据

- 唯一profile：`h3_hpsv3_video_only_v1`，version1、weight1。全124帧走原`jpeg_frames(...,'BFCHW',contract)`和`client.score_video`，原协议计算/校验top37均值，wrapper直接返回有限Python float，无额外clip/归一化/混合项。
- media结构接口与现H3AVMedia一致，无新模型域耦合。video必须CPU detached FP32 `[124,3,H,W]`且finite、[0,1]；metadata frame_count124/fps24匹配。
- audio必须CPU detached FP32 `[165600,2]`且finite；metadata sample_rate32000/sample_count165600匹配。保持decoder实际取整后的音频长度与幅度，不发音频给HPS，不额外clip/trim。
- 原client/validator继续核验sample/service identity、全部frame_scores、FP32可表示有限性及topK总分。wire没有可信远端selected-indices字段：原validator本地推导indices；远端额外提供非法selected_frame_indices被精确字段校验拒绝。没有另写topK实现。
- 不保留media、JPEG或响应到对象字段；只保留request、一个client与小冻结`H3HPSIdentity`。identity包含profile/version/weight/protocol、service identity、冻结VideoContract、manifest SHA256、原factory的endpoint identity，供未来plan/checkpoint绑定。
- 每次score重查本地manifest无漂移；过期lease property原样拒绝。没有在线fallback、POST重试、并发批评分、常量reward或shadow client。

**该reward仅衡量视频HPS，不衡量音频语义或AV对齐；不能宣称音频质量或UniRL CLAP/ImageBind复现。** 完整audio仍是强制媒体，联合AV latent/CPS/梯度未更改。历史81帧服务/manifest不视为124帧ready，本轮没有访问服务。

## 六项新CPU测试

真实本地manifest、原typed acquire request/factory/client/JPEG/health/score validators均执行，仅`requests_session`替为不创建socket的transport叶子：
1. 两种合法画布、全124帧/sample/prompt精确发送、依实际JPEG像素的非恒定服务叶子score、top37独立计算对照、不可变identity/媒体无保留及close生命周期。
2. 错factory/policy、81或超128契约、manifest/service/health不符、非法origin/trust/device/dtype；错误live health关闭client且不重试。
3. 错sample/service身份、缺frame、bool/FP32溢出/非有限frame、错误topK标量、非法selected-indices额外字段均拒绝。
4. 非2xx/redirect、非JSON、超响应字节、timeout；每次POST恰一次。另直接调用同一真实client验证原64MiB请求阈值，用超限字符串列表在发送前拒绝，避免构造多GB视频；该负例不是JPEG正例，未mock编码/validator。
5. 非法video/audio/metadata、过期media、manifest漂移在POST前拒绝。
6. 本地manifest缺失/symlink/大小上限和declare后漂移，均不创建transport、不隐式acquire。

fake service测试分数只证明控制/协议路径，**不是HPS模型打分、资源健康或质量证据**。未运行真实HTTP/GPU或服务命令。

## 最终真实门禁

最后源码编辑后，两条原样执行，exit0，无skip/xfail、无全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 11.99s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 71 passed in 3.63s（已有65 + 本项6，不含未汇合W3兄弟新增项）
```

`git diff --check` exit0。status仅三个指定新文件；tracked diff为空，不用空diff冒充新文件已审。原Flash文件相对授权d818b8a完整历史diff无输出、exit0，断言未改。

## 待办与限制

Simplifier B审本项，changes回B修复重测。W3汇合后orchestrator核对A实际结构调用兼容并复跑完整目标与Flash；局部71项不是汇合结果。后续runtime/recipe必须提供真实artifact request、显式acquire和owner close，并执行真实124帧服务资源准入；本项未部署、未访问旧服务。

ModelAdapter/唯一RunController装配、全阶段协作、完整tail checkpoint/fresh恢复仍未完成；不把本adapter前置项称完整闭环。无服务器/GPU/HTTP访问、模型下载/环境升级/部署/push。**MVP未完成，GPU_PENDING。**
