# Assignment B

- active: true
- status: assigned
- task_id: H3-M2a2-loader
- pair: coding-b → simplifier-b
- execution: sequential; A/C idle
- source_dependency_commit: `edd5c874f10ffa917a71bd9f2fa9085247075a6e`（M2a1 forward coding snapshot，已approved）
- integration_parent: `63fe34c99f96a115ef3cb743965aa4338e3653d6`（含审查记录）
- 已批准前置：M1a joint/schedule、M1b CPS、M2a1 geometry/forward。无changes/失败门禁；先记实际branch/HEAD，不自行提交或切分支。

## 唯一原子功能

实现 **本地H3 transformer artifact → meta模型/主blocks LoRA + 严格CPU权重输入** 的窄加载边界，供下一项C的FSDP2 prepare消费。不是全量adapter、不是分布式wrapper、不是checkpoint系统。生产默认必须调用真实公开依赖API，不允许只接受tiny工厂却没有真实入口。

## 允许写入

- 新 `visual_rl/models/implementations/h3_loading.py`
- `tests/test_h3_mvp_cpu.py`：保留已有17项全部断言，追加本项测试与导入身份。
- `workflow-workspace/2026-10-27-h3-8gpu-mvp-dev-001/coding-b/H3-M2a2-loader.md`

已批准joint/schedule/CPS/geometry/forward只读；旧模型helper/参数manager/runtime、pyproject、旧recipe与原七项只读。不能修改task/reference/AGENTS/.gitignore/workflow。若缺真实必要API、已有环境缺CPU测试依赖或本项过大，报告真实blocked/最小拆分请求，不安装升级、不无声扩大范围。

## 输入输出接口锁

1. **真实lazy依赖与本地artifact**：默认解析 `diffusers.models.transformers.transformer_minimax_h3.MiniMaxH3Transformer3DModel`（真实模块路径由Flow-Factory snapshot证实，实际方法由UniRL vendor类证实）；PEFT/Accelerate/safetensors按需import。缺符号给明确ImportError，无fallback到假模型或reference。先校验输入为存在的本地目录，不把字符串当HF repo ID，不下载/remote code；config读本地 `transformer/config.json`，用已读真实 `load_config(...,local_files_only=True)`/`from_config` 等接口。模型类/关键shape、输入24/audio32/patch(1,2,2)/text维度与合法主block结构有准入校验，不仅看版本号。tiny测试只替重模型类解析叶子，生产函数不含test_mode。
2. **meta初始化**：构造时仅参数置meta，必要构造buffer（尤其非持久 `rope.inv_freq`）保留真实确定值；不得全buffer置meta/零填。可使用真实 `accelerate.init_empty_weights(include_buffers=False)` 或等效已验证API。严禁先CPU/GPU完整实参构造再转meta。返回模型时base与LoRA参数仍meta，输出稳定的主blocks/FQN信息给后续prepare；不建optimizer/scheduler，不绑定旧ParameterStateManager，不创建PG/mesh或调用fully_shard。
3. **LoRA唯一范围**：默认rank32/alpha64/dropout0/bias none，有限正整数rank/alpha可给tiny较小值；仅匹配 `transformer_blocks.<index>.{attn.to_q,attn.to_k,attn.to_v,attn.to_out.0,ff.net.0.proj,ff.net.2}` 的真实线性层。枚举完整FQN传给真实PEFT（不要只suffix匹配refiner），每个主block六target齐全，匹配为空/重复/非Linear拒绝。其余base/root/refiner全部冻结，注入后仅期望LoRA可训练；root外trainable或缺失目标立即拒绝。不用旧 `apply_peft_lora` 的无限suffix范围；不为此修改其旧语义。选择不额外包未知root的PEFT注入路径，最终FQN在交付中固定。
4. **精度锁**：base主块BF16；`proj_in/audio_proj_in/time_embedder/proj_out/audio_proj_out/rope` 保持FP32，与模型 `_keep_in_fp32_modules` 核验相符；LoRA master FP32。不要先全model BF16再把rope等已经损失的值转回FP32；必要buffer按真实声明/构造精度保留。内部gradient_checkpointing关闭且核验；后续C只采用单一外部nonreentrant AC路径。
5. **严格本地CPU权重输入**：提供明确独立函数，只有调用者指定的初始化owner（后续rank0）才调用；构造meta不隐式读全tensor。支持本地Diffusers safetensors单文件或shard index（不支持的格式明确早拒，不用pickle/bin fallback）。校验index/实际keys一致、无重复JSON key/跨shard重复tensor/未声明key/缺文件/路径逃逸；按**全部base FQN和shape**完整映射至注入后state，只有精确已知PEFT base_layer命名转换，不能25%覆盖或strict=False。错误/未知键不能默默drop；LoRA初始化项及可重建非持久buffer是明确例外，不是泛化missing白名单。
6. **初始LoRA与buffer来源**：CPU输入必须包含确定、可复现的全部LoRA初态，不返回未初始化meta/empty权重。显式init seed，明确PEFT兼容初始化（建议 `init_lora_weights=True` 的Kaiming-uniform A/zero B，FP32），不悄悄继承旧helper gaussian或全零A+B。避免污染调用者全局RNG；只有owner物化CPU初态，后续C统一广播。必要持久buffer在完整覆盖内，非持久构造buffer保持真实值并记录处理方法。交付列清base映射/LoRA名单/buffer处理，不能用LoRA export代替完整初始化。
7. **加载容量诚实**：此方案是rank0完整CPU state，不宣称流式常量内存。入口要求明确正的host权重字节上限并在读tensor前按文件header/目标dtype/LoRA额外大小估算准入；包含源与转换额外驻留的保守预算，非host RSS保证。不在八rank各读一份、不把tensor先搬GPU。模型/CPU输入只返回给后续prepare，不擅自materialize再建optimizer。使用现有文件身份helper如确有合适接口，或最小返回本地config/权重身份；不造通用manifest/cache/receipt系统。

接口可以是两个清楚函数加最小结果类型：`build_h3_meta(...)` 和 `load_h3_cpu_state(...)`（命名可调整），给C明确model、可分片主blocks、expected state/FQN映射及CPU完整输入。不要以callbacks藏第二次加载或无消费者的泛化pipeline。此项成功后C负责meta→fully_shard→全rank state setter→最终参数绑定→optimizer，不把本项CPU测试当FSDP证明。

## 必读材料与许可

- 当前实际 geometry/forward、`models/implementations/common_diffusers.py::apply_peft_lora`、`models/state/parameters.py`（不可提前冻结Parameter身份）。
- `reference/UniRL/LICENSE`、固定8cfbac17765004afb68fef08a4e011a4c8daf555的 `minimax_h3/bundle.py`、vendor transformer构造/精度岛/buffer，以及T2VA recipe主blocks LoRA范围。
- `reference/Flow-Factory/src/flow_factory/models/minimax_h3/dependency.py` 的真实Diffusers模块路径仅作API线索；不搬其框架。不复制UniRL bundle的每rank完整CPU/GPU加载逻辑。缺失VENDOR_COMMIT不编造；本项使用公开依赖，不新增vendor源码。

## 必须的真实CPU验收

- 使用tmp_path本地tiny config和真实safetensors文件；只替换重H3模型类叶子，**真实torch meta/PEFT/文件解析/映射/初始化逻辑**必须执行。不要mock掉所测loader或PEFT来制造通过。
- tiny模型有主blocks六targets、带相同suffix的refiner、root FP32岛、必要非持久buffer；证明构造全参数meta（没有先实参全载）、LoRA只在主blocks、所有trainable FP32、base BF16/岛FP32、buffer实际值未损、内部AC关闭。
- CPU完整state和meta目标键/shape精确覆盖、LoRA A非全零/B全零且seed可复现不污染全局RNG；tiny CPU显式materialization后真实forward/backward能用，双模态/LoRA梯度路径和base冻结正确（LoRA A初始梯度为零可合法，不能要求每个值非零）。不得将此测试setter称为FSDP setter。
- 单文件与两shard index加载正例；missing/extra/shape/dtype不支持/重复key或映射冲突、路径逃逸、非法config/target、超host预算在tensor读入前拒绝等必要负例。无strict=False或假权重补齐。
- 缺可选H3依赖分支有明确错误；如本机没有真实H3类，tiny测试证据与未验证真实API明确分列，不skip/xfail。所有测试限本目标集及原七项，不独立跑环境命令/联网。
- 最后源码编辑后原样执行plan两条CPU命令，Flash仍7/7、H3全过；记录exit/数量/时间与实际branch/HEAD、diff及 `git diff --check`。

## 非目标/审查重点

无GPU/服务器/环境安装升级、权重下载、FSDP wrapping/PG/optimizer、aux/text cache、controller/recipe、checkpoint。Simplifier B检查真实可调用加载路径、严格覆盖与初始化、meta/buffer/精度/LoRA范围、未提前绑定参数及过度框架；changes必须回B修复重测。本项不等于真实H3权重加载或MVP/GPU验收，保留GPU_PENDING。
