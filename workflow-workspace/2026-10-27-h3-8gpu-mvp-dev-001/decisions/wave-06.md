# W6 — 原配对修复 + runtime绑定

基线a54d296a7635fe3d552b0fe9beb03717e34249a2；W5源码汇合230696472b735966d3ac17102120a444f4ff8697。A数据/B profile原子approved；C publisher **changes/C-PUBLISH-01，未批准**。当前Flash7/12.15s、H3通配符126/11.97s（16warnings）实跑通过仅为回归，不消除目录fsync缺口。

## 必修与scope

C回同Coding C：在8rank附件完成/校验之后、完整发布之前fsync h3-ranks目录；复用manager现helper，错误走现共同拒绝/poison。仅h3_transaction.py及本feature测试，不改接口/manager/codec；新增顺序与OSError故障注入测试。修复必须同simplifier-c复审，不能以已有126项通过批准。

B仍同Coding B：profile/data组合发现BOM、16MiB cap与解析/采样/stream版本表达不同，统一profile到已批准H3PromptSource真实语义，新增真实对象交叉负例。只h3_profile.py/profile测试，不改data或公开profile方法/字段。

A新runtime binding原子：基于既有profile/source/adapter/stages/HPS/coordination构造真实绑定与观测numerics/checkpoint contract；新runtime/h3_binding.py和独立binding测试，只对H3Adapter._prepared_forward作消费已绑定数值scope的最小委托改动。不写controller/RunController或checkpoint publisher。此项稳定后默认生命周期装配才有真实receipt，避免用全0哈希或fixture contract绕过旧G3。

三项并行安全：C只内部文件持久化顺序；B不改变profile接口，正常有效输入仍同样构造；A仅消费正常已有接口与实际字段生成ID，不依赖B新符号或C未批准publish方法。A/B不写checkpoint；C不写model/runtime/data。feature测试完全分离，源码问题回原owner。

## 为什么暂不一次写完整默认RunController

当前旧runtime仍先optimizer后prepare、H3未消费G3证据，且H3CheckpointContract/真实data envelope尚无生产构造者；直接default dispatch会依赖未稳定合同或伪造receipt。A先完成必要的单一绑定功能，不再造新算法helper。C publisher复审与A binding批准后下一项直接唯一RunController/torchrun生命周期默认装配；不以此延期MVP完成条件。

无GPU/HTTP/PG/服务器/下载/升级/push。未做断电实验；C缺口是实际同步顺序证据，不声称已复现崩溃数据丢失。GPU_PENDING。
