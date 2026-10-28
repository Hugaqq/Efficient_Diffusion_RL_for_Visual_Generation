# 第八轮：转换收口观察

本地协调HEADf8e6dc91583ccf9675d4a0a595dd257ae96a21f8；原job native-artifact-convert-001仍固定fa99cf22c11670a67c06cd16508dd8f4ab5664b3。先查state并inspect同job，不重跑转换。已重读task/formal-plan/reference、当前direct-IO实现及Coder payload复验指示。

## 只读请求回执

- exp008-inspect-convert-01：1793159686.930778→1793159701.970942，code255、stdout空，port22连接超时。
- exp008-inspect-convert-retry-01：1793159759.5465732→1793159774.559222，code255、stdout空，同样连接超时。

上述两个请求均read_only=true、uncertain=false；当时未派发新launch或stop。

- exp008-inspect-convert-retry-02：1793159898.371978→1793159900.7116969，code0；原转换job running=false、exit0，started1793157389.6860728 ended1793159717.5206215。真实artifact_published，最后报告peak RSS11994386432 bytes。
- exp008-artifact-receipt-01：code0。已发布conversion.json 662419 bytes，SHA256 c5255d2ab8720db5dadbe30ab4ea6cd03900e940ee2984b2d732df6a8cc778e6；40个输出文件合计77301635348 bytes，23个来源文件，model keys transformer638/vae703/audio1087；路径/大小/无symlink检查通过，partial目录不存在。MemAvailable117938384kB。独立payload hash重查仍待执行。
- exp008-launch-payload-01：code0，新probe artifact-payload-verify-001，PID976840/start_ticks74717946/created1793160025.3965695，3600秒有界，空CUDA可见。它会先重算全部产物hash，再实际loader并逐tensor比较；这不是训练或CPU测试套件。

## Experimenter探查命令错误（非Coder源码故障）

launch后复读本角色命令发现末尾`report('payload_verification_complete', **result)`会把result中的`peak_rss_bytes`与report内部显式同名关键字一起传给dict，造成重复关键字异常。此为本角色编写探查argv的日志错误，尚未观察到执行至该句；不归咎算法，也不假称probe成功。正确的新命令应将result嵌套传入，例如`report('payload_verification_complete', result=result)`。不修改当前远端spec/源码。

为避免已知命令缺陷产生伪验收，本角色通过现stop_job请求停止仅此新probe（不是已成功结束的转换job，更非其他进程）：
- exp008-stop-payload-command-error-01：1793160108.1787531→1793160123.1929028，code255，连接超时，**uncertain=true**。停止效果未知，禁止自动重发或启动替代job。
- exp008-inspect-payload-after-stop-01：1793160173.481023→1793160188.49324，code255，stdout空，只读观察失败。
- exp008-stop-effects-retry-01：1793160243.5307899→1793160258.553323，code255，stdout空，只读PID/输出核查失败。

- exp008-stop-effects-retry-02：1793160330.768269→1793160345.7837372，code255，stdout空；第二次只读效果重试未连通。

- exp008-stop-effects-retry-03：1793160504.326777→1793160519.339859，code255、stdout空，连接超时。
- exp008-stop-effects-retry-04：1793160591.244101→1793160595.672648，code0/uncertain=false。原owner976840/start_ticks74717946仍State S，child976841/start_ticks74717951仍State R、进程组976840；stop-request.json、exit.json、payload-verification.json均ABSENT。heartbeat1793160566.928527/elapsed541.4826s，日志全部输出hash后已出现load_begin。child VmRSS/VmHWM均59694592 kB。
- exp008-reconcile-stop-01：code0，`reconciled without replay`。根据上述实际原PID/启动时刻、存活child/推进日志和缺失stop-request，核销停止请求的不确定性，未观察到停止效果；没有重发stop。

现在保留原有界probe继续运行获取真实loader反馈，不再因已记录的末尾日志缺陷中断已经推进的读取。日志已验证全部40个产物文件hash，正在实际load_h3_cpu_state，80GiB上限与24GiB余量guard未删。若最后出现预期日志重复关键字异常，保留真实exit1，按操作命令缺陷处理；不改exit文件、不把存在payload-verification.json当exit0，不归咎Coder。后续必要时更正本角色新probe命令、在核查旧任务终止后独立复验，不能对正在运行的spec/源码hotfix。

未知mutating操作已清空；当前返回waiting，不needs_input。原转换exit0和产物证据已明确取得，不受该探查停止请求影响。

## 转换结束后执行门（计划，未运行）

先查exit0、已发布artifact/conversion.json、完整相对路径列表/大小/无symlink，核对源固定converter与三个模型实现hash、638/703/1087模型规格及最后RSS。再以新owned有界probe逐文件独立hash、调用真实build_h3_meta与load_h3_cpu_state（rank4/alpha8、owner=true、80GiB显式上限、seed42、保留实时24GiB余量），逐tensor与真实safetensors内容/目标dtype比较，不复制全模型、不写假SUCCESS、不冒充optimizer更新。所有身份与数值结论须来自该probe实际退出；当前不得使用artifact.partial。
