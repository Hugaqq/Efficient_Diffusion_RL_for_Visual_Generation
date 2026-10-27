# H3-M2a2-loader — Coding B 交付

## 实际身份/范围

- 本轮任务 active=true；当前无 changes，前项 simplifier-b/M2a1 为 approved。
- 实际分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/main`。
- 起始 HEAD：`282dd4ad05bbabb8bdb8ddf301e11b057d6b0846`，接收工作区干净。未切分支/手工提交/merge；本项引擎 snapshot 待生成。
- 新源码仅 `visual_rl/models/implementations/h3_loading.py`；只扩展 `tests/test_h3_mvp_cpu.py`。已批准 joint/schedule/CPS/geometry/forward、原七项、旧配置/runtime/helper/ParameterStateManager 均未改。

## 两个生产入口及给 C 的合同

### `build_h3_meta(local_model_dir, *, rank=32, alpha=64) -> H3MetaModel`

1. 必须是现存本地目录，固定读取 `transformer/config.json`。拒绝路径逃逸、重复JSON key、非H3 class、非法24/32/patch(1,2,2)/text_dim/hidden_size/num_layers。
2. 默认 lazy 解析公开 `diffusers.models.transformers.transformer_minimax_h3.MiniMaxH3Transformer3DModel`，缺符号明确 ImportError，绝无 reference/tiny fallback。真实调用 `load_config(local_files_only=True)`、`from_config`。
3. 真实 Accelerate `init_empty_weights(include_buffers=False)`：参数在构造时即meta，非先实参再转meta；构造buffer保留真实值。依赖构造/PEFT在 `fork_rng(devices=[])` 内，不改变调用者CPU RNG。
4. 主块 `transformer_blocks.<i>` 为后续分片目标。每块枚举完整六FQN `attn.to_q/to_k/to_v/to_out.0/ff.net.0.proj/ff.net.2`，要求各为独立Linear；拒缺失/重复/别名，检查root投影shape。
5. 真实 PEFT `inject_adapter_in_model(..., adapter_name='default', low_cpu_mem_usage=True)` 原位注入，无额外PeftModel root。配置rank/alpha正整数、dropout0/bias none/init_lora_weights=True。核验只期望LoRA trainable，root/refiner/base全冻结。内部gradient_checkpointing统一关并复核。
6. base参数BF16；`proj_in/audio_proj_in/time_embedder/proj_out/audio_proj_out/rope` FP32，严格核验模型声明；LoRA master FP32。没有全model先BF16损坏buffer再转回。所有参数返回时仍meta。

返回内容：
- `model`、`block_names`、`target_names`。
- `base_to_model`：全部原base state（含持久buffer）到注入后FQN完整一一映射。仅目标线性层 `T.weight/bias -> T.base_layer.weight/bias`，其他名字不变。
- `state_specs`：注入后全state的shape/dtype；不是参数对象身份绑定。
- `lora_names`：每个目标的 `T.lora_A.default.weight` 和 `T.lora_B.default.weight`，不添加 `base_model.model` 前缀。
- `nonpersistent_buffers`：当前精确只有 `rope.inv_freq` 的真实CPU FP32构造副本；其他未知非持久buffer拒绝，需另审处理。**C若调用to_empty必须显式恢复此副本，不能保留to_empty产生的未初始化buffer。**
- 本地transformer路径及config SHA256。

### `load_h3_cpu_state(meta, *, is_init_owner, host_weight_limit_bytes, init_seed) -> H3CPUState`

- 仅调用者明确标记的初始化owner可调用，False先拒绝，不读权重。这里不创建PG/推断rank；C必须确保仅rank0执行并广播，不在八rank各读一份。
- 支持固定Diffusers单文件 `diffusion_pytorch_model.safetensors` 或对应 `.index.json`，必须恰选一个；无pickle/bin fallback。index/key/header/shape完整覆盖全部base，拒缺失/extra/重复JSON key/跨shard重复或错放tensor/缺文件/路径逃逸。source仅F32/BF16/F16，转到目标dtype后再次finite校验。没有strict=False、部分覆盖或补假base权重。
- header预检用真实safetensors get_slice，不读取tensor payload。预算为 `2*源文件总bytes + 2*完整目标state bytes(含LoRA) + 2*非持久buffer bytes`，超显式host上限在get_tensor前拒绝。是保守**权重驻留**预算，不是RSS承诺，绝不是流式常量内存。
- 只有owner生成全部FP32 LoRA初态：显式独立CPU generator种子、PEFT默认兼容Kaiming-uniform A（a=sqrt5）、zero B；不污染全局RNG、不继承旧helper gaussian、不把A/B都置零。
- 返回 `state` 为完整CPU目标state、独立非持久buffer副本、估算bytes、本地config/权重SHA256。函数本身不把state装入模型、搬GPU或创建optimizer。

后续 C 顺序仍应 meta→fully_shard→全rank setter/恢复buffer→最终参数绑定→optimizer。本项测试中的 `load_state_dict(strict=True, assign=True)` 仅 tiny CPU materialization，绝非 FSDP setter。

## 来源/许可与边界

已重读 UniRL LICENSE 全文，固定 `8cfbac17765004afb68fef08a4e011a4c8daf555` 的 bundle.py、transformer构造/FP32 islands/rope buffer及T2VA trainside recipe；读取 Flow-Factory dependency.py 的公开模块路径、当前geometry/forward、旧apply_peft_lora和ParameterStateManager绑定规则。独立接口实现，使用公开依赖，未复制vendor/bundle代码，无reference import，无新增vendor许可文件，不编造缺失 VENDOR_COMMIT。

本机测试实际执行真实 Torch meta、Diffusers ConfigMixin、Accelerate、PEFT注入与safetensors解析；仅重H3类解析叶子换为tiny。测试也调用默认可选resolver，并独立强制缺模块场景验证明确ImportError；测试通过**不证明真实H3类构造/权重加载兼容**。未下载或加载真实权重，未安装/升级依赖。真实目标环境H3 API仍须准入验证。

## 测试与真实结果

原17项及断言保留；新增五项 loader测试和导入身份断言：
- 真实meta/PEFT范围、精度、内部AC关闭、非持久buffer原值、持久buffer覆盖；源→目标逐tensor映射。
- CPU完整state、seed复现/隔离、A非零/B零；tiny物化后分别video/audio真实forward/backward进入LoRA，base无梯度。
- 单文件与两shard正例；missing/extra/shape/dtype/nonfinite/重复key/跨shard错放/缺文件/path和symlink逃逸、非法config/target/alias/island以及format负例。
- owner与预算调用前拒绝，负例spy只拦tensor读取而真实header解析仍执行。

初轮H3 21 passed in 3.35s；完善负例/后置精度核验后的**最终源码**真实执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# exit0: 7 passed in 12.12s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu.py -q -p no:cacheprovider
# exit0: 22 passed in 3.12s
```

无skip/xfail、无全量测试。`git diff --check` exit0；已看实际diff：仅H3测试扩展316 insertions/1 deletion（import扩展），原断言未删除。新增loader未跟踪，不计入该diff stat。最终status预期仅loader、H3测试及本交付目录；不以此替代历史snapshot范围审核。

## 待审/风险

交 simplifier-b，重点审核API/严格完整映射、LoRA范围/初始化、meta与buffer保存、owner预算边界、无提前参数身份冻结。changes回B修复两集重测。

本项仅提供后续prepare输入，目前生产controller尚未消费。需后续C完成真实全rank prepare与最终参数绑定，后续B完成aux/profile/controller，其余world/评分/旧锚/tail/checkpoint/fresh恢复仍未完成。没有GPU/服务器/网络部署/下载/环境升级/push。CPU不证明FSDP2/NCCL/真实H3或质量。**MVP未完成，GPU_PENDING。**
