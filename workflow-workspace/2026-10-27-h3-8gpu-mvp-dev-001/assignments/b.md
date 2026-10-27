# Assignment B — W5

- active: true
- task_id: H3-M2b2-profile
- pair: coding-b → simplifier-b
- execution: parallel W5；decisions/wave-05.md
- integration_parent: `6f5e46fce212e00d547a3ca07fa5524018bef816`
- dependencies: 注册H3Config/Adapter及既有stage/HPS/prepare已approved；W4代码汇合fb085fb35fb239cdd4c151d39decbd7fae268c9d。无changes。

## 单一功能/允许范围

实现供**现唯一RunController后续绑定**使用的固定H3 schema-v2 profile编译与本地前置事实检查；本项不假装已装配controller、不实现第二入口/Runner。只新 `visual_rl/composition/config/h3_profile.py`、新 `configs/v2/h3_t2va_mvp.yaml`、新 `tests/test_h3_mvp_cpu_profile.py`、W/coding-b/H3-M2b2-profile.md。旧compiler/registry/runtime/adapter/modeltypes/七项及其他tests只读；A/C同波新模块不import。

## profile合同

- 一个显式 `schema_version:2, recipe:h3_t2va_mvp_v1`，顶层沿现overrides/launch结构；未知键、重复YAML键、bool当int/NaN/inf、任意动态class路径严格拒绝，不接受宽松任意dict。用typed frozen profile保存已验证字段；配置引用全是本地路径，相对YAML目录解析，不下载/联网/创建输出目录。
- 模型声明通过现Registry/DeclarationResolver真实h3-t2va descriptor/H3Config解析，不复制模型配置validator，不伪造单sigma/Gaussian blueprint或旧Dynamics receipt。profile的数值设置明确是joint CPS负MSE/元素权重、FP16 action、FP32 score、独立FP32 video shift12/audio shift3、eta.7、全N含q0、frozen replay old anchor、B1/chunk1/guidance1/beta0，无EMA/reference。
- 默认256×384×124@24、WORLD8/SP1/TP1、每rank一个完整K8组、N10、一次collection一次optimizer，AdamW lr1e-4/betas(.9,.999)/eps1e-12/wd0、clip.005、norm1、constant LR；固定每run geometry/K/N，可允许明示有限小值供tiny测试/用户显式新profile，但不得隐式删音频/末步/改score。若只支持默认则明确拒绝覆盖，不偷偷继承Flash参数。
- launch包含output_dir/resume_from、model/text_cache/train/eval/HPS manifest本地refs、完整expected cache身份与expected HPS service identity、loopback origin/timeout/response bound、seed/max_optimizer_steps/checkpoint cadence、显式host-weight/trajectory/media/decode/condition预算及8个获准UUID列表。具体字段命名可合理最小化，完整schema写在模块docstring/交付。禁默认物理0..7或假UUID，实际运行前必需用户填真实集合；样例中未提供资源用显式待填项并说明拒绝准入，不伪造存在权重/资源ready。
- expected cache身份复用现reader合同，HPS manifest的all124/top37契约复用现protocol校验；profile明确h3_hpsv3_video_only_v1/weight1及audio质量不受该reward衡量。旧81帧HPS不兼容，禁止live health调用（该步骤后续runtime acquire）。
- 提供compile与**只读本地preflight**清楚分工：编译验证结构/固定语义并解析路径；本地preflight检查真实train/eval bytes/hash、cache index/shards覆盖/身份、model/vae必要config/文件存在性、HPS本地manifest及预算可容纳最小工作量。复用现H3TextCache/H3Config/VideoContract等验证，不直接建H3模型或VAE，不用安装版本号替代API能力。过重模型权重全量hash可交后续既有artifact resolver，不在compile重复扫描数十GB；明确哪些身份当前仅声明、哪些来自真实读取。
- 产生path-independent canonical semantic fingerprint（包含数据内容、模型/缓存/score/双schedule/采样流/训练数学，不能把本机output/path作为语义），launch执行位置/获准UUID/预算另有明确binding。只有实际内容才能产生对应identity，不填全0/占位receipt。默认路径输出不写入、不触碰运行目录。未经真实state绑定的G3数值证据不得伪造。
- max steps/cadence/resume边界严格：只完整collection boundary、same topology；配置不得宣称支持mid-collection/elastic。profile编译结果供下一原子runtime绑prepare、data、stages与现manager；缺的schema接口准确列出，不添加NotImplemented runtime来假装能run。

## CPU验收

实际读tmp_path YAML和本地tiny configs/cache/safetensors/prompt/HPS manifests；真实模型declaration、typed配置、本地preflight和现cache/protocol校验执行，禁止mock整个compiler。重模型加载/HTTP/PG函数用失败spy保证compile/preflight不会调用。

正例验证默认数学与所有identity/path分离，移目录不改内容semantic ID，改prompt/cache/schedule/service声明等会更改或拒绝；unknown/duplicate YAML、非法预算/组合、缺train/eval/cache覆盖、错误HPS124、缺资源/不合法授权集合、resume拓扑约束负例。在允许的H3目标集内检查既有代表性Flash/SD3/Wan编译仍不变，不跑全量。

最后编辑后固定Flash7/H3通配符，原断言不动，无skip/xfail；报告branch/HEAD、命令/exit/数量/时间、diff/check和**下一步默认runtime需要消费的精确compiled字段/真实binding缺口**。Simplifier B审最小封闭profile、无假G3/资源身份、未造第二config框架/入口；changes回B。GPU_PENDING。
