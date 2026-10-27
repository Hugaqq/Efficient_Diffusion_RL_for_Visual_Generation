# Assignment B — W6 profile/data边界对齐

- active: true
- task_id: H3-M2b2-profile-align
- issue_id: B-PROFILE-01（orchestrator组合准入要求，不改原approved历史）
- pair: coding-b → simplifier-b
- execution: parallel W6；decisions/wave-06.md
- integration_parent: `a54d296a7635fe3d552b0fe9beb03717e34249a2`
- dependencies: H3PromptSource与profile各原子approved；W5 review指出BOM/cap/parse-stream版本尚需统一，现实际源码已确认差异。

## 唯一功能/范围

使profile本地预检和真实数据源使用**同一精确prompt解析/文件上限与版本语义**。仅改 `visual_rl/composition/config/h3_profile.py`、`tests/test_h3_mvp_cpu_profile.py`；新W/coding-b/H3-M2b2-profile-align.md。A data源码/原source tests、C publisher、runtime/model全部只读，不改公开compile/preflight方法及H3Profile/H3LocalFacts字段结构。

## 最小对齐

- 真实source明确拒开头BOM、空行、孤立CR，仅剥LF/CRLF终止符，固定文件上限16MiB；profile目前未拒BOM，prompt_file_bytes可超过source上限。复用source已存在的解析/上限逻辑或常量作唯一authority，不能再维护两份略异解析函数，不删source严格拒绝。不添加通用文本解析框架。
- profile显式budget小于16MiB时仍按更小上限拒；大于16MiB时实际准入不超过source硬上限（可以明确拒该配置或用min并写清，但不得让profile接受source必拒文件）。SHA仍针对原始bytes，不normalization后hash；首尾空格、精确Unicode/CRLF仍保留。
- semantic payload记录真实source PARSE_VERSION/SAMPLING_VERSION/STREAM_VERSION及adapter initial_state_version，不仅描述性字符串。不要伪称版本一致，不能靠两个不同ID恰好都是hash掩盖差异；路径/UUID/预算与内容semantic分离的原合同保持。
- 正常有效输入的profile公共结构不变，A同波binding按真实字段计算身份，不依赖新增API。不得增加模型/HTTP/PG构造、副作用或环境探查。

## 验收

新增交叉用例让同一真实UTF8文件/cache同时经过preflight_h3_profile和H3PromptSource（真实adapter，可复用只读tiny fixture），证明BOM、孤立CR、空记录、文件cap、空白/中文/尾终止符接受或拒绝域相同。至少一个有效profile映射参数实际构造source并reserve/commit，不只比较两个helper函数。核对semantic payload真实版本常量、目录移动同内容ID稳定、旧代表性recipe回归及无模型/HTTP/PG构造spy。

大文件负例可在tmp_path生成超过16MiB的有界文本，不能改全局cap让断言变绿；不修改原126项或原七项断言。最后编辑后跑plan固定Flash7/H3通配符，无skip/xfail/全量。交付branch/HEAD、命令/exit/数量/时间、diff/check与具体B-PROFILE-01边界关闭证据。

原profile原子批准保留，但本组合准入必须复审才能用于最终runtime。Simplifier B审最小复用/语义一致性/无额外框架；changes回B修复。GPU_PENDING。
