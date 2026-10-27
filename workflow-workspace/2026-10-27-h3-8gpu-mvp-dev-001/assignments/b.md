# Assignment B — W4

- active: true
- task_id: H3-M2b1-model-adapter
- pair: coding-b → simplifier-b
- execution: parallel W4；decisions/wave-04.md
- integration_parent: `7cf28dced4c8193ccf9321d62b5b2e479bd904f7`
- dependencies: 已批准loader/forward/cache/decoder、现ModelAdapter与catalog；W3源码汇合5397c46662db30ef535afc0fb4b192df019c4d62。无changes。

## 唯一目标

实现**可经现模型registry声明/构造的真实H3 ModelAdapter**，把已有模型域组件接成encode/初态/predict/decode端口；不再让runtime直接拼散落模型helper。只扩展H3封闭类型，不泛化整个框架、不接新controller/runner。

## 允许文件

新 `visual_rl/models/implementations/h3.py`、新 `visual_rl/models/h3_config.py`、新 `visual_rl/models/h3_interface.py`；精确追加 `visual_rl/models/catalog.py` 的H3 descriptor（不改SD3/Wan）；仅H3必要类型扩展 `visual_rl/core/contracts/model.py`、`visual_rl/models/interface.py`。新 `tests/test_h3_mvp_cpu_adapter.py` 与W/coding-b/H3-M2b1-model-adapter.md。旧H3 helper、所有其他tests、runtime/recipe/compiler/checkpoint、A/C文件只读。若需额外文件先报最小范围，不顺手改公共合同。

## 合同锁

- 继承现ModelAdapter，真正实现describe/from_config/load_components/encode/prepare_latents/predict/decode及生命周期释放；不得实例化一个带NotImplemented/假媒体的伪adapter。配置/声明import-safe，重依赖仅现loader/decoder路径lazy。
- 新明确TaskKind.T2VA、MediaKind.AUDIO、rank2 JOINT_AV_FLAT latent layout（命名可一致调整），不要借T2V/BCTHW伪装音频。ModelLatentSpec仅对新layout增加明确[batch,joint] rank2检查，旧PACKED_SEQUENCE rank>=3等约束原样保留，不全局放松。默认geometry256×384×124@24/guidance1、LoRA32/64、FP16状态/FP32 CPS scoring/块BF16岛FP32。
- 现ModelInput只有单timestep：定义窄H3ModelInput（可继承ModelInput）增加独立audio_sigma，原timestep在此明确是video_sigma；检查双FP32 detached sigma/shape/device/range。不能拿index、平均sigma或改旧timestep语义蒙混。返回现ModelPrediction并沿用shape/identity验证；预测noise-ward两流符号转换只在已批准h3_forward一次。
- 配置严格键集合与family几何/LoRA/缓存/预算字段；artifacts留逻辑refs，本地路径和expected cache身份从显式runtime_context取，不填假identity。只支持T2VA/B1/guidance1/LoRA，不承诺SP/TP/reference/EMA。未知覆盖早拒，不改Flash preset或算法语义。
- load_components使用现ComponentLoadSession.register/freeze；实际调用build_h3_meta，不全模型CPU/GPU实参加载；暴露已有H3MetaModel只读给未来runtime prepare。Transformer注册INFERENCE/TRAINABLE，text cache/decoder为非prepared资源且lazy，不加载Qwen。禁止adapter创建PG/optimizer或提前绑定ParameterStateManager。后续prepare_h3_fsdp返回handle后由现bind接口绑定，单次predict始终通过prepared guard。
- encode实际消费H3TextCache、明确train/eval coverage准入端口；返回owned detached CPU embeddings及准确cache/prompt identity，trimmed5120维。若现StackedSampleBatch尚无T2VA，定义最小typed prompt输入而不是伪造T2V媒体；不要修改数据域形成第二dataset框架。runtime/data装配下一项使用此端口。
- prepare_latents用显式generator真实torch随机噪声给joint FP16状态，保留双流元素与独立模态流策略（若base方法单generator不足，可在H3 typed参数中明确给pair；不隐式用global RNG）。不实施去噪或rollout；初态生成版本要能进入后续checkpoint身份。
- predict消费H3 typed输入及真实conditioning，调用已有h3_t2va_forward，传self的受控prepared callable；不unwrap raw model、整体autocast/cast或接算法ID。不import algorithms内CPS/stages；双schedule构造仍归算法。
- decode必须返回已有H3CPUDecoder单条AV lease，包含video+audio，不塞进只支持单tensor的DecodedMediaBatch伪media。H3公开端口返回类型明确为AV lease，旧adapter返回仍不变；若需要annotation封闭union可最小扩展，不删旧校验。
- 不制造假的Diffusers scheduler blueprint/Gaussian绑定或fake numerics receipt。此原子项交付必须清楚列出未来H3算法binder/G3还需的真实端口；不能宣称仅catalog新增就已能跑现默认runtime。对未声明能力明确拒绝，不承诺任意旧算法可组合。

## CPU验收

通过现catalog/declaration resolver实际找到H3 descriptor/from_config，旧SD3/Wan声明不变；配置import不创建权重/HTTP/GPU。真实cache/geometry/loader meta/PEFT/decoder控制，仅重模型/VAE叶子tiny；有真实ComponentLoadSession生命周期、prepared guard、encode→独立初态→双sigma predict/backward→AV lease decode调用。验证双流、dtype/身份、未prepare/raw旁路/close拒绝、资源只关闭一次、非法单sigma/缺audio/cache miss与未知config拒绝。

独立feature测试，不改原93项。最后源码编辑后跑plan Flash7与H3通配符，无skip/xfail/全量；报告branch/HEAD、命令/exit/数量/时间、diff/check与**实际注册接口和后续runtime所需端口**。若公共合同扩展过大，报告具体最小拆分，不把未实现端口写成生产success。

Simplifier B重点审现ModelAdapter实际消费已有组件、typed双sigma/AV不能混入旧Gaussian、边界/guard/资源身份以及最小新增声明；changes回B。无服务器/GPU/下载/环境升级/push，GPU_PENDING。
