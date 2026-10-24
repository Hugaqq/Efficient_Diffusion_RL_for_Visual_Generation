# 可读模型/算法边界：阶段A → Wan RL5 → H3原生推理

阶段A精确交付SHA：`9d5c1e4afcdbefa5ef61c0754d3ce3bb844b3c75`。B因新异常root GPU进程939604外部安全阻塞，未部署/运行RL5，不是五次GPU通过。阶段C已增加原生H3 Model+rollout；部署和模型/依赖/容量/一条→三条/下载命令见[H3交接](h3_native_inference.md)。C最终40位VisualRL SHA以任务`local-c-result.md`为准，固定上游后端SHA为`0f79bd8b0e5ff9379a9a38e0c585db8f8335bf7c`。GPU安全复查及真实B/D尚未完成。

## 一页 review 导航

**Model Adapter 回答“模型怎样计算”，Algorithm 回答“怎样采样、评分和更新”。**

```text
训练 YAML → 原有 controller（设备、准备、恢复、关闭）
  → BaseTrainer
    → 模型 encode / prepare_latents / predict
    → DynamicsSession.sample_transition / deterministic_ode_step
    → 模型 decode → 真实 reward → advantage → credit
    → 模型 predict（固定已采样 action）
    → DynamicsSession.evaluate_transition → PPO → backward → optimizer commit
    → 原有原子 checkpoint

纯推理：WanInferenceAdapter.load(local_path) → generate(request) → close
         不进入 trainer、LoRA、optimizer、reward 或训练 state
```

实际删除：`DefaultPolicyRuntimePort` 的 `preprocess`、`initialize_latents` 同义入口，以及 `transition` 代理；collector、TempFlow branching 和 recomputer 直接调用自己已经持有的同一个 DynamicsSession。不再先包装 request/result 再解包，也不保留“生产走facade、测试走直接session”的双分支。公开旧 transition DTO 暂保留供历史 imports，但不参与活路径。

保留：模型内部 CFG/decoder 差异、prepared root、forward-only autocast、FP32 latent/probability、原始 forward partitions、old probability、固定 action 重放、组内优势、PPO clipping、有限/非零梯度、采集权重摘要、pending pool/cursor、EMA/finally 和完整恢复所有者。Flow-GRPO、TempFlow、Flash、World-R1 各自 rollout/conditioner 不合并成万能算法。

纯推理新增的是明确的 **generate能力**，不是把完整 pipeline 假装成单步 predict。Wan 原生 Diffusers pipeline 负责自身生成循环；这里冻结组件、离线加载、显式 seed、验证真实返回媒体和关闭。不新增 registry/factory/调度平台。C现已增加明确H3联合音视频result与原生generate路径，不套Wan视频tensor/CFG；B仅有外部阻塞，C未被描述为GPU通过。

## 版本与测试边界

- 唯一Mac candidate：`outputs/flash140-coding`；起始clean `f51033af981204f510ad5e4d76ab47e57f26980c`。
- 交付分支：`h3-readable-integration`。**部署必须使用local角色最终回执里的40位 `AGREED_SHA`**，不是动态latest main/branch名。最终receipt另存项目任务的 `local-a-result.md`（避免文件自引用自身Git SHA）。
- 本版本仅fresh run；不要恢复已完成SMALL40或旧版本pending checkpoint。既有recipe/code/prepared/full恢复门保持，旧launcher-only owner升级不能授权本次runtime变化；核心测试明确验证拒绝。未转换checkpoint、未丢pending重采样。
- 原7函数仍在 `tests/test_flash_collection_runtime.py`，collection/std/精确RNG与奇数恢复/EMA/评估/140和SMALL40数量断言保留。历史“runtime相对repair base不变”改为固定历史f51033a确实不变，另增加实际新runtime不满足旧launcher升级白名单的拒绝断言；不是删除旧run的安全门。
- 最终实际运行：原7项 `7 passed in13.60s`；独立专项 `10 passed in3.41s`；`compileall -q visual_rl`及`git diff --check`均无错误。此前原7项13.57s/专项9项3.32s及10项3.74s为中间记录，不替代最终门禁。
- 首轮原7项6通过/1失败：历史repair断言要求所有未来runtime永远等于旧base，与本次明确fresh新runtime不相容；改为固定历史对象仍不变+当前变更被原升级规则拒绝，未放松升级检查。专项首轮8通过/1失败为测试helper不能比较含多元素tensor的dataclass，改为terminal恢复前后checkpoint所有文件逐字节比较；最终全部重跑通过，无skip/xfail或子集替代。
- 专项包含：无LoRA/AdamW/ParameterStateManager/reward的推理哨兵、确定seed/原prompt/关闭/坏输出；六个原demo完整断言（SD3/Wan/TempFlow/Flash/World两模式，模型transition哨兵不能截获）；RL5 typed配方、真实CPU controller五次更新/逐次trainable变化/奇数终点/terminal checkpoint逐字节不变；CLI preflight不启动、fresh完整运行、拒绝重复启动。
- **模型、评分、媒体及资源叶子为显式CPU fake。未运行GPU、真实HPS HTTP或真实Wan生成；不是数值/容量/GPU验收。** 没有用历史768测试替代核心门禁。

## RL5配方与验收

文件：`configs/v2/wan_rl5.yaml`。独立新输出：
`/mnt/data/qifan/runs/visualrl-readable-h3-20261024/wan-rl5-001`。

- Wan 480×832×81，BF16模型计算/FP32 trajectory，20采样步、CFG4.5、LoRA16/32，seed42、lr1e-4、clip.001。
- 两个prompt×group2 = 每池4视频；全4奖励std后两次2-row提交；chunk2、forward1、replay1、decode1。
- **目标恰好5 optimizer commits：2+2+1；3池共12条真实生成/评分，10 active row-updates，最后2行未消费。** terminal step5保留完整4行池和cursor1，不把目标四舍五入、不伪造第6次提交。
- 不执行SMALL40/140 baseline/周期评估；已有实验及48视频结果不改动。
- 原真实HPS18820、81帧480×832/all81/top24契约及身份不变；不移动或重启服务。身份不匹配应保留证据返回Mac，不服务器改YAML。
- `scripts/validate_wan_rl5.py`只有身份/资源/结果观测，调用原controller和原YAML；无第二个训练loop。复用现有flock、15s资源心跳、真实HPS cache handoff。fresh-only，无自动retry/resume/重启。
- 观测在既有prepare/checkpoint边界读取初始及每次真实trainable内容摘要；每次必须optimizer transaction已commit、active2、finite/nonzero gradient、权重字节变化。terminal SUCCESS前也检查第五次；最终再读取原子step5 checkpoint检查cursor/pool。
- `rl5-result.json.status=verified_five_updates`与原SUCCESS/step5 **同时存在**才是本验证完成。仅进程/GPU占用/心跳不是成功。

## Server部署（只经授权WebCLI；local角色未执行）

先核对 origin=https://jumpjumpjump.ishencai.com、hostname=makefilm-20250911-092937。按TASK复查GPU/root异常进程及资源冲突：918828已按用户授权退出，但若新同特征进程出现，记录并阻塞GPU，不擅自杀新PID/改服务。保留所有旧worktree/SMALL40/checkpoint/HPS/ComfyUI。浏览器终端每30分钟前有输入；主会话10分钟idle-only保活不是GPU安全检查。

把local回执精确SHA填入环境变量；下文只在核对SHA/dirty/import和安全条件后执行。

```bash
export AGREED_SHA='<local最终回执的40位测试SHA>'
export SOURCE=/home/ubuntu/qifan/visualrl-readable-h3-20261024/source
export PY=/home/ubuntu/qifan/envs/visual-rl/bin/python
export OUT=/mnt/data/qifan/runs/visualrl-readable-h3-20261024/wan-rl5-001
```

若SOURCE尚不存在（首次部署）：

```bash
git -C /home/ubuntu/qifan/VisualRL fetch origin h3-readable-integration
mkdir -p /home/ubuntu/qifan/visualrl-readable-h3-20261024
git -C /home/ubuntu/qifan/VisualRL worktree add --detach "$SOURCE" "$AGREED_SHA"
```

若SOURCE已存在，先 `git -C "$SOURCE" status --short --branch`，审查dirty是否含凭据或不明大文件；如有先报告。只有该任务checkout可按用户授权stash，记录ref，绝不覆盖其他checkout：

```bash
git -C "$SOURCE" stash push -u -m readable-h3-before-agreed-deploy
git -C "$SOURCE" stash list -1
git -C "$SOURCE" fetch origin h3-readable-integration
git -C "$SOURCE" switch --detach "$AGREED_SHA"
```

没有dirty时不必stash；不reset/clean或自动stash pop。实际核对HEAD==AGREED_SHA、status为空后：

```bash
cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/ubuntu/qifan/visualrl-flash140-20261022/dependency-overlay"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
"$PY" -c 'import visual_rl; from visual_rl.algorithms.trainer import collection; print(visual_rl.__file__); print(collection.__file__)'
git rev-parse HEAD
git status --short --branch
"$PY" -u scripts/validate_wan_rl5.py --expected-sha "$AGREED_SHA" --preflight
```

环境复用现有Torch2.7.1+cu128/Diffusers0.33.1及ftfy6.3.1/wcwidth0.2.14 overlay；脚本核实实际imports与版本，不安装/升级共享环境。preflight检查HPS health/artifact身份；controller仍会严格验证模型/数据/奖励资产和执行binding。初始准入>=32GiB GPU、48GiB host、64GiB disk（前两项为未实测预算）；执行期复用>=16GiB GPU/24GiB host/64GiB disk门。采样心跳不能证明全局瞬间峰值；不足则保留阻塞，不删门禁。

**preflight RC0 且复查无重复作业/无资源冲突后，只启动一次：**

```bash
mkdir -p /mnt/data/qifan/runs/visualrl-readable-h3-20261024
LOG=/mnt/data/qifan/runs/visualrl-readable-h3-20261024/wan-rl5-console-$(date +%Y%m%d-%H%M%S).log
nohup "$PY" -u scripts/validate_wan_rl5.py --expected-sha "$AGREED_SHA" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf 'PID=%s LOG=%s OUT=%s SHA=%s\n' "$PID" "$LOG" "$OUT" "$AGREED_SHA"
ps -p "$PID" -o pid,etime,stat,args
readlink /proc/"$PID"/cwd
```

记录PID/log/owner和持续心跳；让运行自然完成，不用until1/2/8、不人工暂停、不因进度询问取消。未知启动结果先查进程和输出，不能重放nohup。失败也保留整个独立输出，代码问题返回Mac。

完成后（只读可信任务checkpoint）：

```bash
"$PY" - <<'PY'
import json
from pathlib import Path
from scripts.launch_flash140 import inspect_checkpoint
out = Path('/mnt/data/qifan/runs/visualrl-readable-h3-20261024/wan-rl5-001')
r = json.loads((out / 'rl5-result.json').read_text())
assert r['status'] == 'verified_five_updates' and (out / 'SUCCESS').is_file()
assert [u['committed_steps'] for u in r['updates']] == [1,2,3,4,5]
previous = json.loads((out / 'rl5-initial.json').read_text())['trainable_sha256']
for u in r['updates']:
    assert u['trainable_sha256'] != previous
    assert u['metrics']['gradient_norm_pre_clip'] > 0
    assert u['metrics']['gradient_norm_post_clip'] > 0
    previous = u['trainable_sha256']
    print(json.dumps(u, sort_keys=True))
final = inspect_checkpoint(out / 'checkpoints/step-5')
assert final['committed_steps'] == 5 and final['commit_cursor'] == 1
assert final['pending_rows'] == 4 and final['evaluations_completed'] == []
assert final['trainable_sha256'] == previous
print(json.dumps(final, sort_keys=True))
PY
```

另保留真实HPS完成事件、三次collection/六次chunk评分、五份更新审计、initial/final哈希、step5完整收据与GPU/host/disk峰值；不要只交SUCCESS截图。

## 文件清单与下一阶段

改动：`runtime/model_binding.py`、`core/contracts/runtime.py`、`algorithms/rollout/{collector,branching}.py`、`algorithms/optimization/recompute.py`、`runtime/algorithm_binding.py`、原7测试中的历史升级兼容检查。
新增：`models/inference.py`、`configs/v2/wan_rl5.yaml`、`scripts/validate_wan_rl5.py`、专项`tests/test_readable_inference.py`和本文。

B若发现内部bug，Mac修复→原7+专项→commit/push→server Git，不server patch。B此次只有外部安全阻塞，C已继续本地原生AV推理/rollout接入；它不是GPU验证或下载完成。原RL5 YAML与验收仍有效，新代码只fresh运行，不热续旧pool。旧ComfyUI三段视频不是D验收素材。
