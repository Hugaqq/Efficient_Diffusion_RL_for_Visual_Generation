# H3-M2b1-model-adapter — Simplifier B / W4

verdict=approved

仅批准注册模型端口原子项；outcome done表示本次review结束，不代表默认RunController可用或完整MVP完成。

## 身份与材料

- HEAD：`fb085fb35fb239cdd4c151d39decbd7fae268c9d`。
- 分支：`pi-workflow/3ceaec6a-2451-4a1b-9fe8-ede829ee5dcc/task-355e2e9e-3aa8-49cd-a127-0f57345bb806`；开始工作区干净。
- 已读当前task/plan/assignment B/wave-04和对应Coding B交付，完整新config/interface/adapter及四项feature测试、当前catalog/core模型合同/ModelAdapter接口，核对实际ComponentLoadSession/Manager和prepared guard边界。
- 未修改源码、测试或他人报告；只审核B本项，不以通配符通过代替A/C审核。

## 实现与边界

1. `visual_rl/models/h3_config.py`与`models/catalog.py`：`h3-t2va`通过真实Registry/DeclarationResolver找到配置/声明/实现路径；逻辑artifact refs与显式runtime路径/expected identity分离，严格字段集和family几何验证。声明明确T2VA、VIDEO+AUDIO、rank2 JOINT_AV_FLAT、LoRA/current-only，不伪造scheduler/Gaussian能力。配置路径无重模型加载或联网。
2. `core/contracts/model.py`、`models/interface.py::ModelLatentSpec`：新增封闭enum与新layout精确rank2 `[batch,joint]`校验；旧PACKED_SEQUENCE rank>=3及原ModelInput/ModelPrediction验证仍保留。decode仅扩展annotation允许H3 AV lease，旧单tensor媒体未被伪造替换。
3. `models/h3_interface.py`：B1 typed prompt和拥有副本的CPU `[1,L,5120]` conditioning，identity绑定cache与精确prompt；H3ModelInput继承真实ModelInput，额外锁定video/audio两个FP32 detached scalar sigma、设备/范围/身份/guidance1。旧单sigma输入不能进入predict，模型域没有算法schedule/CPS依赖。
4. `models/implementations/h3.py:113–141`：真实build_h3_meta与session.register/freeze，transformer角色INFERENCE/TRAINABLE，meta资源不被旧load阶段搬CPU；aux为非prepared幂等owner，cache/decoder按需创建，不加载Qwen。实际encode/coverage消费严格cache，未新造数据框架。
5. 同文件`:143–191`：显式双generator独立模态FP32 draw→FP16 joint初态且版本公开；predict仅通过现PreparedComponentHandle.forward，复用既有h3_t2va_forward一次符号转换，返回真实ModelPrediction并验证。未prepare/raw旁路/单sigma/foreign cache拒绝，双流梯度与no_grad同路径实际受测。adapter不创建PG/optimizer或提前冻结参数身份。
6. 同文件`:193–211`：真实CPU decoder单条AV lease；queued context进入时再次检查adapter open。close关闭现handle并幂等清空aux/meta/cache/decoder拥有引用，manager后续关闭不重复释放资源。外部已借用媒体仍须遵守lease生命周期。
7. `tests/test_h3_mvp_cpu_adapter.py:92–271`：四项真实registry、ComponentManager.load、Torch meta/PEFT/safetensors、prepared guard、cache/初态/predict/backward/decode/close测试。仅重transformer/VAE叶子tiny；旧SD3/Wan声明和packed guard亦有断言。测试中的CPU物化/Adam不是FSDP或默认runtime证明。

## 简化结论

无必须返工项，返工清单为空。新增typed输入用于真实AV/双sigma区别，aux owner用于现session生命周期，均有实际消费者；未复制joint/forward/cache/decoder数学，没有第二runner、dataset平台或泛化注册框架。

本项允许且明确的未实现边界不能被批准掩盖：当前scheduler/G3/preprocess/reference未声明能力fail closed；`_prepared_forward`只使用prepared handle，尚未消费未来G3阶段/parameter-view数值证据。下一绑定项必须实现真实双schedule/CPS binder、混合精度/current-only G3和资源/data绑定，不能只填receipt或沿旧Gaussian路径放行。现ComponentManager.configure/通用prepare也不能直接当作H3 meta/FSDP生命周期。

## 本轮独立CPU门禁

原样执行以下两条，均exit0，无skip/xfail、未跑全量：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_h3_mvp_cpu*.py -q -p no:cacheprovider
# 108 passed, 9 warnings in 7.38s

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= VISUALRL_FLASH_AUTHOR_DATA='/Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/reference_code/flash_grpo/dataset/video' /Users/qvanium/Desktop/Efficient_Diffusion_RL_infra_for_Visual_Generation/framecode/.venv/bin/python -m pytest tests/test_flash_collection_runtime.py -q -p no:cacheprovider
# 7 passed in 12.56s
```

108为此前93+W4 A4/B4/C7。9条warning来自同波checkpoint测试的no-dist save/load及损坏fixture重写提示，真实显示、未过滤；不据此声称分布式DCP已执行。

获准`git diff d818b8a231b52a9a024fa2f522057acdbb7bf099 HEAD -- tests/test_flash_collection_runtime.py`无输出、exit0，七项完整文件/断言与基线一致。`git diff --check` exit0。当前公共模型源码已逐文件读取，精确逐分支历史patch归属仍由引擎/orchestrator审计，不以clean工作区代替。

## 后续

Orchestrator登记本组snapshot与approved，结合A/C裁决亲跑汇合门禁。优先完成真实profile/binder/G3、meta→FSDP→最终参数/handle绑定、T2VA data与唯一RunController装配，并消费AV lease；完整checkpoint发布与同拓扑fresh恢复仍待验收。不能以catalog可见或继承抽象端口齐全冒称默认runtime已经跑通。

本轮无GPU/服务器/HTTP/PG初始化/下载/环境升级/push。真实H3权重、FSDP2/NCCL及质量未验证。**MVP未完成，GPU_PENDING。**
