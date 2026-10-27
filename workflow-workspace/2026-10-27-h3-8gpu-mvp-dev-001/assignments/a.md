# Assignment A — W7 原配对最小返工

- active: true
- task_id: H3-M4c1-runtime-binding-r1
- original_task: H3-M4c1-runtime-binding
- issue_id: A-BIND-01
- pair: coding-a → simplifier-a
- execution: parallel W7；decisions/wave-07.md
- integration_parent: `d7c75b75210782a63a4fffcae117f2ace19fa760`
- dependencies: W6源码汇合c8422b3a780d933564e94dd58c52a5c8efd8f9dc；simplifier-a/H3-M4c1-runtime-binding.md明确changes。B-PROFILE-01与C-PUBLISH-01已获原配对approved。

## 唯一目标与文件边界

只修scheduler与最终optimizer的所有权、实例稳定性和constant可观察状态。允许改 `visual_rl/runtime/h3_binding.py`、`tests/test_h3_mvp_cpu_binding.py`；新增本task_id交付报告。原报告及changes保留。其余model/stages/source/profile/prepare/codec/transaction/controller与B/C新文件只读。

公开H3RuntimeBinding构造参数、predict/scope/validate_live/checkpoint_boundary及返回形状冻结；合法prepare产生的constant scheduler仍兼容。不得重写fixture接口或进行无关重构，C并行仅消费既有合法合同，不依赖本修复的新符号。

## 必须修复

1. 在统一观察路径初次绑定及每次scope/live/checkpoint检查 `scheduler.optimizer is prepared.optimizer`；绑定后scheduler不得被另一个同类型实例替换。相同参数集合不能代替实际optimizer对象关系。
2. 核对声明h3.constant.v1的可观察状态：optimizer实际lr、scheduler base/current lr、last_epoch/提交计数一致；结合真实prepare与codec格式判断，不凭LambdaLR类型独自放行。禁止调用scheduler.step/optimizer.step来探测、禁止dummy step；不新增通用scheduler注册器。
3. 新增初始外部optimizer scheduler、绑定后optimizer引用漂移、同类型scheduler实例替换负例；覆盖受控forward/live及post-tail checkpoint边界在模型调用/更新前拒绝。包含constant状态漂移负例，保留合法真实连续更新及Adam/scheduler/data/trainer计数/lr断言。
4. 保留原cold/active/optimize后tail前/poison/未清grad保护及全部既有断言。发现更大接口需求先报告，不扩大改动。

## 门禁与复审

最后编辑后原样运行plan的Flash七项与完整 `tests/test_h3_mvp_cpu*.py`；无skip/xfail、全量或过滤warnings。记录实际branch/HEAD、diff/check、命令/exit/数量/时间及每条issue证据；引擎负责snapshot。

同simplifier-a复审并写独立r1报告。135项通过不覆盖本缺口，未经approved不得接收binding。B/C仍各自审查；本项不实现controller，也不宣称GPU/完整MVP。GPU_PENDING。
