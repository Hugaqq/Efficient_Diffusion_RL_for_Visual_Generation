# 第八轮当前状态：转换exit0；原payload probe继续运行 / waiting

生产SHA fa99cf22c11670a67c06cd16508dd8f4ab5664b3，本地协调HEAD f8e6dc91583ccf9675d4a0a595dd257ae96a21f8。无算法源码修改或CPU回归。

## 已确认资产准备

原native-artifact-convert-001 exit0/running=false，started1793157389.6860728 ended1793159717.5206215；artifact_published，末次峰值RSS11994386432 bytes。conversion.json SHA c5255d2ab8720db5dadbe30ab4ea6cd03900e940ee2984b2d732df6a8cc778e6，40文件/77301635348 bytes，638/703/1087模型keys。已独立核查完整相对路径集合、大小、无symlink、partial消失；新probe已逐文件通过全部40个输出hash断言并进入实际loader，整体probe仍未退出，完整读取验收尚待完成。

## 新probe与操作错误

artifact-payload-verify-001：PID976840/start_ticks74717946/created1793160025.3965695，timeout3600秒，日志`/home/ubtroot/qifan/h3-experiment-cycle-20261028-001/runs/artifact-payload-verify-001/job.log`。本角色发现自己探查命令最后report调用重复peak_rss_bytes关键字，非Coder源码错误，尚未观察实际运行至末尾。

本角色曾发出仅该probe的stop_job请求exp008-stop-payload-command-error-01，SSH code255造成uncertain。后续exp008-stop-effects-retry-04成功确认原owner976840/start_ticks74717946和child976841均存活，stop-request/exit/payload-verification均缺失、heartbeat已541.48s、日志进入load_begin，child RSS59694592kB。exp008-reconcile-stop-01已code0核销未知状态，未观察停止效果、未重发stop。

现在保留原probe继续收集真实loader反馈，不停止已经推进的读取，不改运行spec。末尾日志缺陷可能仍造成exit1，届时据实保留并按本角色操作命令错误处理，不能改退出状态或仅凭receipt称probe通过。原转换job/产物不受影响。

全部回执在experimenter/observation-008.md；state.json保留旧job与stop核查记录，uncertain_operations已清空。本轮转换inspect第二次重试成功；stop效果观察第四次重试成功，符合最多5次授权。返回waiting，下次先inspect同probe不新launch。Flow/Flash0，正式目标此阶段不询问。
