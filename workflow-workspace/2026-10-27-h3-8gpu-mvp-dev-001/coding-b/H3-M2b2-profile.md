# H3-M2b2-profile — W5 Coding B

## 判定/工作区

- `implementation_ready: true`：仅本原子项编译/只读本地预检完成，交simplifier-b；不代表默认入口、GPU资源或checkpoint恢复验收。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-02102b61-1df8-40b3-8bcd-b808992d2146`。
- 起始/结束HEAD：`d2a6b8563f5bcc2d5f95bf9e24458dc25d257b91`，初始干净。无换分支/手工commit/merge/push，snapshot待引擎生成。
- 已读最新task/plan/assignment B/wave-05、本配对adapter审查、原source loader/compiler/declaration/identity、H3Config/cache/schedule/loader header/decoder及pool预算合同。只读取A assignment的数据交界要求，不import或修改同波A/C模块。

## 文件范围

只新增：
1. `visual_rl/composition/config/h3_profile.py`
2. `configs/v2/h3_t2va_mvp.yaml`
3. `tests/test_h3_mvp_cpu_profile.py`
4. 本交付记录

旧compiler/registry/runtime/adapter/所有原测试与Flash配置未改。没有新增入口、Runner、manager、NotImplemented runtime或G3成功凭据。

## 两个真实入口

```python
profile = compile_h3_profile(yaml_path)   # 也接受现SourceRecipe
facts = preflight_h3_profile(profile)     # 只读真实本地内容，仍不启动运行
```

编译复用原`load_source_recipe`（严格SafeLoader、duplicate key、FrozenMapping非有限值拒绝）和现`Registry/DeclarationResolver`解析真实`h3-t2va` descriptor/H3Config，不复制模型geometry validator。结果为冻结H3Profile/H3Launch/H3Training/H3Budgets与原ResolvedComponentDeclaration。没有任意dynamic class路径。

### 完整schema

顶层恰为：`schema_version: 2`、`recipe: h3_t2va_mvp_v1`、`overrides`、`launch`。

- overrides可有model/training两个group：
  - `model: {params: {...}}`仅允许height/width/frames/fps/guidance_scale/lora_rank/lora_alpha；均通过原H3Config，profile另外锁124帧HPS契约。
  - `training: {group_size, num_steps}`；默认K8/N10，显式可选K2..64、N1..100，每run固定，不按余额截断。其余数学参数不可覆盖（未知即拒）。
- launch必需字段：
  - output_dir、resume_from（可null）、resume_policy。
  - artifacts：model/text_cache/train/eval/hps_manifest五个本地路径。相对路径按YAML所在目录解析；URL/null拒绝，不创建目录或下载。
  - expected：model_sha256/train_sha256/eval_sha256/cache_identity/hps_service_identity。hash必须完整小写64hex，拒全0占位；cache_identity完整采用原reader字段集并绑定同一declared model_sha256。
  - execution：world_size8/sp1/tp1/gpu_uuids；恰八个不同规范`GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`，不得物理index、空列表或全0UUID。语法验证不是外部授权或空闲证明。
  - seed：整数[0,2^63)；max_optimizer_steps正整数；checkpoint_every_optimizer_steps正整数且不大于max steps。
  - budgets：host_weight_bytes/trajectory_bytes/media_bytes/decode_bytes/condition_bytes/max_tokens/embedding_bytes/prompt_file_bytes，全部显式正整数，不接受bool。
  - hps：origin（原loopback validator）、timeout_s（有限正数）、max_response_bytes（正整数）。本步不healthcheck。

resume_policy只接受`complete_collection_same_topology`，不接受elastic/mid-collection；世界拓扑固定8/SP1/TP1。**此处仅冻结请求策略，未读取或证明checkpoint内部拓扑/进度/complete状态。**

### 固定数值语义

默认256×384×124@24/guidance1/LoRA32/64；一rank一个完整组、B1/chunk1、一collection一optimizer。全N含terminal q0，独立CPU FP32 shift12/3、eta.7、joint CPS negative MSE/生成元素均值，FP16 action/FP32 score，frozen replay old、beta0、无EMA/reference。模型混合BF16/FP32六岛、LoRA FP32、aux FP32及主blocks六target策略进入语义payload。AdamW lr1e-4/betas(.9,.999)/eps1e-12/wd0、clip.005、norm1、constant LR固定；不继承Flash参数。

reward固定`h3_hpsv3_video_only_v1`、weight1、all124/top37；audio仍在joint模型/CPS中，reward不衡量audio语义/AV对齐，不声明音频质量。

## 本地preflight的真实读取与非证明边界

实际执行：
- 有界读取train/eval原bytes并核expected SHA256。UTF8严格解码，LF或CRLF分行，只去行终止符、不strip文本；拒空行、裸CR、坏编码。train>=8行、eval>=1行。
- 实际H3TextCache身份/header/完整shard hash与train/eval精确覆盖；逐唯一prompt get一条后释放，验证finite并计算最大实际conditioning payload。
- 原H3CPUDecoder的只读config/stats/symlink/index准入（不加载VAE）。transformer必要族config检查及三组件真实safetensors header/文件存在性检查；main-block LoRA target header须存在。
- 本地HPS manifest调用原decode_json/validate_health，对应真实VideoContract all124/top37；旧81帧拒绝，无live acquire。
- minimum trajectory为原pool合同 `K*N*(4*joint_width+4)+K*2*joint_width`；condition为K×最大实际单条embedding bytes；media/decode估算复用原decoder；预算不足拒绝而非缩画布/改dtype。
- host-weight估算为两倍transformer文件bytes+两倍全base FP32上界和LoRA FP32初态+1MiB buffer余量；不构建模型、不get_tensor。后续原loader仍必须做真实完整FQN/shape精度与容量准入。该预算不是aux VAE权重/内部workspace/全机RSS保证。
- resume_from非null仅检查存在本地目录；完整checkpoint contract/progress/完整性与两阶段restore属于现manager后续绑定。

**未执行完整模型权重hash、模型构造或API能力验证。** `facts.model_weights_verified=False`；expected.model_sha256仍是调用者声明，runtime artifact materializer必须按与cache生产者一致的模型artifact身份方案实际核验后才能使用。既不能从header可读推断真实H3可加载，也不能以安装版本号替代API检查。

## 身份分离与下一runtime精确消费字段

`profile.model.config`是原H3Config，携带预算和逻辑model/cache refs；`profile.model.implementation_class_path`来自现registry。`profile.training.group_size/num_steps`给数据/stages/CPS；`training.payload()`显式包含固定所有数学/采样流策略。

`profile.launch`含已解析本地路径、ordered GPU UUID、seed/max steps/cadence、HPS origin/timeout/response limit、typed budgets。`profile.expected`含完整外部expected身份。下一runtime可据此提供adapter所需`artifact_paths`与`expected_cache_identity`，以及HPS真实typed materialized request；不得自行填假字段。

`facts`包含：
- `semantic_id`与冻结`semantic_payload`：真实train/eval bytes哈希、cache index/full shard绑定、已读取model config哈希/weight header摘要、**标明declared**的model identity、HPS service/contract、完整FP32双sigma grid、模型几何/LoRA、seed/max steps和所有训练数学。
- `binding_id`：路径/output/resume、UUID顺序、预算、endpoint/timeout/cadence等执行绑定，独立于内容语义。
- `train_sha256/eval_sha256/cache_identity/model_config_hashes/hps_manifest_sha256`及实际预算估计值。

模型weight header摘要仅代表所读shape/dtype metadata，不是tensor内容哈希；完整权重身份仍未验证。因此该semantic ID只能作为已声明模型身份下的本地事实绑定，**不能冒充完整materialized recipe/G3 receipt**。目录迁移同内容不改semantic ID；预算/endpoint/UUID/output/cadence变化只改binding ID。prompt/cache/seed/K/N/服务内容变化会改变或拒绝semantic ID，受测。

ID采用现`canonical_identity`的带namespace格式（`h3.semantic.v1:<64hex>`、`h3.launch.v1:<64hex>`）。后续若给现H3StateIdentity的纯SHA字段使用，应校验namespace后使用相应64hex digest；topology/model/progress仍须各自产生真实证据，不拿binding字符串充当已验拓扑。

### 下一原子项必须补的绑定

- 现唯一`python -m visual_rl.train`/ProductionRuntimeLifecycleBackend识别此显式recipe并消费上述profile/facts；本波不修改旧compile_recipe_v2路由，样例当前不能通过旧默认入口直接跑。
- 对模型权重/代码/真实cache生产身份完成既有artifact materialization，核验声明后再建立最终run/checkpoint语义；真实H3符号/资源与HPS124 live health准入。
- 数据对象实际parse/sampling/stream版本与profile声明的确定策略核对；完整N双schedule/CPS/adapter/prepare/stages接线及真实current-only G3数值/preprocess证据，不能伪造Gaussian receipt。
- GPU授权/可用性及same-topology checkpoint实际内容检查；现manager完整safe point、rank RNG/data与两阶段fresh恢复。仅目录存在或UUID格式正确不允许冒称完成。

## 样例

`configs/v2/h3_t2va_mvp.yaml`包含完整字段与固定数学注释。无资源identity/path/host budget的字段用null、GPU UUID用[]明确待填，**编译故意拒绝**。没有假UUID/全0receipt/模型ready声明；示例output只是字符串，未创建输出。

## 测试/真实结果

六项新CPU测试：真实tmp YAML、本地tiny configs/safetensors/cache/prompt/HPS manifests、原model declaration/cache/protocol与header检查；失败spy阻止任何重模型/HPS health/HTTP session/PG初始化。验证默认数学、真实预算、冻结字段、路径迁移与binding分离、语义内容漂移、unknown/duplicate/NaN/inf/bool/动态class/拓扑/UUID/预算/coverage/旧81帧/缺文件等拒绝；实际调用旧SD3/Wan/Flash代表性compile_recipe_v2并核对原声明/fingerprint稳定。

前两轮出现新测试API断言错误（RegistryError包装、旧ResolvedSlotDeclaration.declaration、ConfigSourceError早拒非有限YAML）；已按真实接口修复，未修改生产保护或旧测试断言。最终没有未关闭失败。

最后源码/测试编辑后实际门禁，均exit0，无skip/xfail或全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 13.87s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 114 passed, 9 warnings in 8.64s（既有108+本项6，不含同波A/C未汇合新增）
```

9条既有warning为checkpoint no-dist和损坏fixture覆盖提示，保留未过滤。`git diff --check` exit0；原Flash完整文件相对授权d818b8a历史diff无输出。最终status仅上述四个授权新文件，旧tracked源码无改动。

交simplifier-b；changes回B修复两集重测。W5暂存汇合不等于批准，orchestrator需完整通配符/Flash复测并优先唯一controller真实装配。无GPU/服务器/HTTP/PG/下载/环境升级/部署/push。**MVP未完成，GPU_PENDING。**
