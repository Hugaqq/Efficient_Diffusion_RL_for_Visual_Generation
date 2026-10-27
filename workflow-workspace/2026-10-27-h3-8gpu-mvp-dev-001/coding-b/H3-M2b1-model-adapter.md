# H3-M2b1-model-adapter — W4 Coding B

## 状态与身份

- `implementation_ready: true`：本原子项源码/本分支CPU门禁完成，待simplifier-b；不是默认runtime或完整MVP验收。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-1e054adc-e263-4ffb-aa31-859832636c18`。
- 起始/结束HEAD：`477325141c0911aeb3cdd8a85937a6843532be87`；接收干净。未切分支/手工提交/merge/push，snapshot待引擎生成。
- 已读task/plan/assignment B/wave-04、已有本配对approved反馈，以及真实catalog/ModelAdapter/ModelInput/ComponentLoadSession/manager/prepared与H3 helper接口。首次读取待新增h3_config.py得到ENOENT，随后按assignment新增，未访问旧worktree。

## 改动范围

新增：
- `visual_rl/models/h3_config.py`
- `visual_rl/models/h3_interface.py`
- `visual_rl/models/implementations/h3.py`
- `tests/test_h3_mvp_cpu_adapter.py`
- 本交付记录

精确修改：
- `core/contracts/model.py`：只追加T2VA/AUDIO/JOINT_AV_FLAT三个enum值。
- `models/catalog.py`：只追加H3 descriptor，SD3/Wan声明未改。
- `models/interface.py`：仅新layout的rank2 `[batch,joint]`校验、TYPE_CHECKING AV类型和decode返回的封闭union annotation。旧PACKED_SEQUENCE rank>=3及所有旧验证原样保留。

不改旧H3 helper、原93项/Flash七项、runtime/recipe/compiler/checkpoint/A/C代码；不import同波新文件。

## 实际注册与配置入口

- alias：`h3-t2va`
- declaration provider：`visual_rl.models.h3_config:H3DeclarationProvider`
- config：`visual_rl.models.h3_config:H3Config`
- implementation：`visual_rl.models.implementations.h3:H3Adapter`（继承现ModelAdapter，全部abstract端口实际实现）
- 现`Registry('model', MODEL_CATALOG_FRAGMENT.descriptors)`与`DeclarationResolver.resolve(...,'h3-t2va',params)`实际找到descriptor/声明，按implementation_class_path导入并from_config构造。

H3Config仅接受声明字段：artifact_ref/text_cache_ref、显式max_tokens/max_embedding_bytes/max_media_bytes/max_decode_bytes、height/width/frames/fps/guidance_scale/lora_rank/lora_alpha。默认256×384×124@24、guidance1、LoRA32/64；几何委托已批准H3T2VAGeometry。refs必须为不同逻辑id，路径、未知SP/TP/full训练等覆盖拒绝。配置/声明import-safe，测试阻止torch/diffusers/peft/transformers/requests导入仍成功声明。

契约明确T2VA→VIDEO+AUDIO、rank2 JOINT_AV_FLAT、noise-ward FLOW、SIGMA、LoRA、模型混合BF16族、无reference。没有填假的scheduler blueprint/Gaussian replay schema或numerics receipt。

## runtime_context 与组件生命周期

from_config只接受两个明确key：

```text
artifact_paths = {config.artifact_ref: existing_local_model_dir,
                  config.text_cache_ref: existing_local_cache_dir}
expected_cache_identity = 完整外部artifact/recipe绑定身份
```

无生产model_loader/test_mode回调，无自行填假cache身份。路径检查不加载模型；cache声明的完整语义与内容在实际reader准入时校验。未来runtime必须将真实materialized路径和expected identity显式传入此窄上下文。

`load_components(session)`真实调用build_h3_meta；require transformer text_dim5120。通过现session.register/freeze登记：
- transformer：INFERENCE+TRAINABLE，仍meta，managed_residency=False，避免旧manager在load阶段`.to(cpu)`破坏meta。
- h3_aux：PREPROCESS+DECODER，非prepared、managed_residency=False，幂等资源owner；cache和CPU decoder到使用时才构造，无Qwen或VAE加载。

暴露只读`meta_model`属性（既有H3MetaModel）给未来runtime.prepare_h3_fsdp；本adapter不创建PG/optimizer，不绑定ParameterStateManager。后续用既有`adapter._bind_prepared_components(prepared_handle)`绑定；predict只调用`self.prepared_components.forward('transformer',...)`，无raw/module unwrap、root autocast或模型整体cast。旧helper保持其本身projection精度岛与唯一符号转换。

close使用现ModelAdapter close/handle释放，再幂等清空aux/cache/decoder/meta引用；ComponentManager随后close同aux无二次资源效果。已返回媒体的消费者必须按AV lease合同在作用域内使用。提前创建但尚未进入的decode context在adapter关闭后也拒绝进入，不能绕过close。

## 模型typed端口

- `H3PromptInput(prompt)`：只承载一个精确prompt字符串，不伪造T2V StackedSampleBatch，不新增dataset框架。
- `verify_coverage(train_prompts,eval_prompts)`：真实H3TextCache检查全覆盖；未来模型phase前必须调用。
- `encode(H3PromptInput)`：真实cache.get，返回H3Conditioning，拥有detached CPU `[1,L,5120]`副本；condition identity为完整cache SHA256+精确prompt SHA256。公开embeddings再次读取得副本，输入或读取mutation不污染。无padding猜测/在线fallback。
- `latent_spec_for_batch`：B1 FP16、CPU/CUDA、rank2 joint宽度与配置几何严格一致。
- `prepare_latents(spec,generator=H3Generators(video,audio))`：两个显式不同generator对象，设备匹配，固定video后audio的FP32标准正态draw→FP16→既有joint pack；无global RNG、去噪或rollout。公开`initial_state_version='h3.av-normal-fp32-to-fp16.v1'`供后续checkpoint身份绑定。
- `H3ModelInput`继承原ModelInput，新增必需audio_sigma；此类型中原timestep明确为video_sigma。两者须FP32 detached scalar、同latent device、[0,1]。严格H3Conditioning/identity、B1 FP16新layout、guidance=None与固定guidance1 identity。单sigma旧ModelInput不可传给predict。
- `predict`验证配置几何/cache identity，复用h3_t2va_forward及prepared guard，返回原ModelPrediction并调用validate_against。两流取负仍只在原forward发生一次；无CPS/算法import。
- `decode`返回context-managed现H3CPUDecoder单条AV lease，含video+audio；没有塞入只支持单tensor的DecodedMediaBatch。

## 必须留给后续binder/runtime的真实端口

本次catalog新增**不表示现默认RunController已经能训练H3**。以下未伪造、仍需下一原子项实作：
1. H3 recipe/算法能力与双schedule/CPS binder：把算法的两个sigma和H3Conditioning装入H3ModelInput；旧Gaussian单schedule blueprint不可使用。当前scheduler_artifact_blueprint明确失败，无假的schema。
2. mixed precision/LoRA/current-only的真实G3 numerics与parameter view证据、preprocess producer/consumer声明及text-cache artifact绑定。当前这些未声明的基类端口fail closed，不返回假的receipt。
3. meta→现H3 FSDP2 prepare→最终参数身份/optimizer顺序和handle绑定。旧ComponentManager.configure/通用prepare不可当本H3生命周期；本次只消费其真实load session，不修改旧manager。
4. 数据域的实际T2VA/data source→H3PromptInput以及train/eval全覆盖准入；未来stage以本adapter端口替代模型helper散装。
5. AV lease供现H3 stages/RewardStage消费，不走旧generic collector单tensor decode路径；完整RunController、safe-point/checkpoint/fresh恢复仍必须接通。

reference/EMA、任意旧算法组合、SP/TP等不受声明支持。CPU测试对未声明scheduler/numerics/reference能力实际要求明确拒绝，而不是成功占位。

## 真实CPU测试与结果

新增4项feature测试，保留原93项断言：
- 真实registry/declaration解析及实现路径构造，声明重依赖import阻止探针、旧SD3/Wan任务/shape/scheduler声明保留；新enum/旧packed rank guard负例。
- 真实ComponentManager.load→session.register/freeze、loader meta/真实PEFT、真实safetensors完整CPU状态加载/显式CPU物化、prepared bundle/handle guard；encode→双流初态→双sigma predict，两流各自梯度和合并backward；no_grad同路径。
- 真实cache覆盖/副本/精确身份、双generator解析对照/隔离与global RNG不变；VAE控制仅重叶子tiny，decode124帧+真实audio长度、lease/资源/close拒绝。
- 单sigma/非法audio sigma、guidance/condition/cache身份、cache miss、未知config/runtime override、未prepare/raw旁路/queued lease关闭后拒绝及未声明能力失败。

只有重H3类及双VAE叶子tiny；测试中的CPU load_state_dict/AdamW用于端口验证，不称实际FSDP setter或默认runtime。没有GPU/HTTP/PG初始化、下载或环境升级。

最后源码/测试编辑后原样运行，均exit0，无skip/xfail/全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.50s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 97 passed in 5.62s（已有93+本项4，不含W4尚未汇合A/C新增项）
```

已查看实际tracked diff：仅三个授权文件的上述封闭扩展；新文件不计入tracked diff。`git diff --check` exit0；原Flash文件相对获准d818b8a完整历史diff无输出。最终status仅授权源码/新feature测试/本报告。

## 交付后

交simplifier-b审核；changes回B修复两集重测。引擎snapshot/暂存汇合不是批准；W4汇合后由orchestrator检查实际组合与完整目标。下一步优先真实profile/binder/runtime/data与manager/checkpoint装配，不再把catalog可见性当生产训练完成。

无服务器/GPU/服务访问/部署/下载/环境升级/push或手工Git提交。**MVP未完成，GPU_PENDING。**
