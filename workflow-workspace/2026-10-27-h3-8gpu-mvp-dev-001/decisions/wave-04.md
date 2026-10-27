# W4 — 真实对象组合 / 注册模型端口 / checkpoint状态codec

共同基线 `7cf28dced4c8193ccf9321d62b5b2e479bd904f7`。W3代码汇合5397c46662db30ef535afc0fb4b192df019c4d62，各配对review approved；本轮Flash7/13.35s、H3通配符93/4.84s真实exit0。W3**原子项**接受，但stages测试仍用_CoordinationLeaf，三对象组合证据未关闭，不能宣称W3整链已验证。

## 并行边界

- A/H3-M4a4-object-integration：只新增tests/test_h3_mvp_cpu_integration.py，真实H3IterationStages+H3HPSReward+H3Coordination同链，现有模块只读。发现问题须给原owner返工，不能改B/C源码或用更高层mock掩盖。此项是所要求的CPU整合门，不是无意义测试搬家。
- B/H3-M2b1-model-adapter：H3 ModelAdapter实际实现/配置/模型域typed端口和catalog，必要时封闭扩展模型enum/ModelLatentSpec；独立test_h3_mvp_cpu_adapter.py。不改stage/coordination/hps/checkpoint或A测试。
- C/H3-M4b1-state-codec：新增artifacts/checkpoint/h3_state.py，真实allrank DCP model/Adam/scheduler/必要buffer codec和fresh restore；独立test_h3_mvp_cpu_checkpoint_state.py。不改manager/旧protocol/runtime/B模型声明/A阶段。

三项只消费已稳定的W3模块与既有公共端口，不引用同波新文件：A不以新adapter/codec作为组合测试前提，B不依赖新checkpoint，C只接现H3PreparedModel与明确caller状态。每组实际分支/前置HEAD/自动snapshot由引擎提供。遇跨owner缺陷回本orchestrator派同owner修复，不能并行偷改。

## 后续真正依赖

注册recipe/algorithm capability/唯一RunController生命周期装配需要B模型合同稳定；最终durable checkpoint需要C codec与完整data/behavior/RNG状态、现AtomicCheckpointManager共享staging接入，不能把本波raw DCP目录当complete checkpoint；真实data源/当前预留游标、same-topology身份还需绑定。下一波优先runtime/data和manager整合，不另写Runner或绕过原controller。任何必要接口不足必须报告，而非返回success占位。

本波所有测试仍只原Flash七项和H3通配符；不跑全量。真实CUDA/FSDP2/DCP/NCCL、H3/HPS资源与音视频质量均未验，video-only HPS限制沿用wave-03。
