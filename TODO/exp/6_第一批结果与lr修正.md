# 6 大链跑完、小链 / 尾链中途的结果，以及 lr 修正重跑

> 2026-11-08 更新。结果来自 `saves/eval/` 与 `saves/tofu_unlearn/`（匹配表：`scripts/analysis/pareto.py --forget <split>`；图在 `saves/analysis/pareto_<规模>_<split>/`）。`*` = 没有任何点满足 Prob ≤ 0.25 且 |mia − retain| ≤ 0.25，显示离规则最近的点。列：Prob / mia / extraction / utility / retain 生成 / ud_bb / wb_unsup。

## 1 forget05 的结果
**8B**（retain 0.107 / 0.360 / 0.067 / 0.638 / 0.996 / 0.003 / 0.099）
| 方法 | Prob | mia | extr | util | retR | ud_bb | wbU |
|---|---|---|---|---|---|---|---|
| NPO lr2e-5 @35 | 0.181 | 0.538 | 0.025 | 0.579 | 0.601 | 0.003 | 0.072 |
| LUNAR c2.8 l23 | 0.134 | 0.534 | 0.081 | 0.472 | 0.669 | 0.432 | 0.935 |
| 旧方法 c7 l17-19* | 0.193 | 0.629 | 0.074 | 0.580 | **0.948** | 0.077 | 1.000 |
| 新方法 l16-18（300 步）* | 0.142 | 0.755 | 0.316 | 0.565 | 0.507 | 0 | **0.030** |
| GA lr1e-5 @35* / GD* | 0.556 / 0.547 | 0.924 / 0.919 | – | 0.592 / 0.594 | – | – | – |
| SimNPO lr2e-5* / RMU lr2e-5* | 0.815 / 0.899 | 0.99 / 0.994 | – | 0.652 / 0.629 | – | – | – |

**3B**（retain 0.131 / 0.360 / 0.052 / 0.663 / 0.934 / 0.020 / 0.102）：旧方法 c8 l12-14*：0.567 / 0.965 / 0.177 / 0.634 / 0.793 / 0.007 / 0.763（没遗忘）；新方法 l19-21（300 步）*：0.176 / 0.911 / 0.150 / 0.551 / 0.675 / 0.020 / 0.013。
**1B**（retain 0.130 / 0.361 / 0.056 / 0.595 / 0.813 / 0.003 / 0.105）：新方法 300 步*：0.160 / 0.878 / 0.142 / 0.512 / 0.513 / 0 / 0.011；旧方法 c7*：0.239 / 0.669 / 0.054 / 0.535 / 0.569 / 0.003 / 0.889；GD lr2e-5 @21：0.057 / 0.575 / 0.079 / 0.419 / 0.383 / 0.020 / 0.005；NPO lr5e-5 @14：0.075 / 0.126 / 0.050 / 0.407 / 0.295；GA lr2e-5 @21：0.039 / 0.442 / 0.004 / 0.247；LUNAR c2：0.025 / 0.236 / 0.012 / 0.192 / 0.358 / 0.238 / 0.674；SimNPO lr5e-5 与 RMU lr2e-5 在第 5 个 epoch 仍是 Prob 0.52–0.57（没遗忘）。

**forget10 × 1B**（retain 0.118 / 0.384 / 0.052 / 0.589 / 0.868）：旧方法 c7*：0.277 / 0.759 / 0.067 / 0.541 / 0.630 / 0 / 0.769；NPO lr2e-5 @65*：0.198 / 0.640 / 0.064 / 0.496 / 0.621；GA lr1e-5 @65*：0.174 / 0.797 / 0.101 / 0.407 / 0.597。新方法（G_10_new）失败，LUNAR 只有 c1 在途。forget10 × 3B 的 `I_new` 已有评测，但 `pareto.py` 未汇总出点（旧方法 `I_old` 还没跑，目录缺）。

## 2 这批结果说了什么（要注意：和 forget01 的结论不同）
1. **新方法在 forget05 上的优势只剩白盒**：白盒 0.01–0.03（旧方法 0.76–1.0），但 **retain 生成（1B 0.513、3B 0.675、8B 0.507，retain 参考 0.81 / 0.93 / 0.996）、mia（0.76–0.91，retain 0.36）、extraction（8B 0.316）都明显差于 forget01 的表现**。旧方法在 8B forget05 的 retain 生成 0.948、mia 0.629，保持得反而更好，只是白盒 1.0 / 黑盒 0.077。所以“新方法在保持、白盒上全面优于 steering”只在 forget01 成立；forget05 要改成“白盒痕迹小，但保持集受损、mia 偏高”，并先做 `5_剩余实验.md` 第 2 节的三个改进旋钮（更深层、retain 权重 3、600 步）再下结论。3B 旧方法 c8 在 forget05 完全没遗忘（Prob 0.567），需要更大系数，不能直接比较。
2. **baseline 的 lr 在 forget05 上偏小或偏大，要重跑**（下面第 3 节）：8B 的 GA / GD lr1e-5 到第 35 步 Prob 仍 0.55、SimNPO / RMU lr2e-5 基本没动；1B 的 SimNPO（5e-5）和 RMU（2e-5）也没遗忘；1B 的 NPO（5e-5）@14 mia 0.126（过冲）。
3. **8B 上调好的 NPO（lr2e-5 @35）已经在 forget05 上进入匹配点**：Prob 0.181、mia 0.538、utility 0.579，retain 生成 0.601，白盒 0.072。

## 3 lr 修正重跑（单行；按规模 / 划分）
8B forget05（大卡空着，可直接跑；GA、GD、SimNPO、RMU 都往大改，NPO 保持 2e-5）：
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_PROFILE=full LRS="2e-5" METHODS="GradAscent GradDiff" bash scripts/unlearn/tune.sh <GPU> ; FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_PROFILE=full LRS="5e-5" METHODS="SimNPO" bash scripts/unlearn/tune.sh <GPU> ; FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_TRAIN="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true" EVAL_PROFILE=full LRS="5e-5" METHODS="RMU" bash scripts/unlearn/tune.sh <GPU>
```
1B forget05（小卡有空档时；GA / GD 在 2e-5 的 21 步已崩溃，降到 1e-5；NPO 过冲，降到 2e-5；SimNPO 加到 1e-4；RMU 加到 5e-5）：
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.2-1B-Instruct EVAL_PROFILE=full LRS="1e-5" METHODS="GradAscent GradDiff" bash scripts/unlearn/tune.sh <GPU> ; FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.2-1B-Instruct EVAL_PROFILE=full LRS="2e-5" METHODS="NPO" bash scripts/unlearn/tune.sh <GPU> ; FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.2-1B-Instruct EVAL_PROFILE=full LRS="1e-4" METHODS="SimNPO" bash scripts/unlearn/tune.sh <GPU> ; FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.2-1B-Instruct EVAL_PROFILE=full LRS="5e-5" METHODS="RMU" bash scripts/unlearn/tune.sh <GPU>
```
3B 的 forget05 / forget10 baseline（链里 `H_3b_*`、`I_ga`、`I_npo` 用的 lr 是旧推算值）：等它们跑完，按上面的规律改：GA / GD 取 1e-5～2e-5（小模型 1e-5，8B 2e-5），SimNPO ≥ 1e-4（1B / 3B）或 5e-5（8B），RMU 5e-5。

## 4 小链的 `G_10_new` 失败（forget10 × 1B 新方法）
日志不在 git 里（`logs/` 被忽略），看：`tail -60 logs/exp/G_10_new.log`，或 `grep -nE "Error|Traceback|out of memory|OOM" logs/exp/G_10_new.log`。从 `saves/` 看：该步已经完成了“原模型回答”的生成（`responses_before_*.json`），但没有 `TOFU_EVAL.json`，所以失败在精修阶段或之后；同一份邻居数据（`neighbor10`）下，尾链的 3B `I_new` 成功了，**不是数据问题**；最像的原因是显存（这一步的 `fr_batch_size` 取默认值，而 `I_new` 用了 8）。把日志里的最后几行发给我我就能确认。
单独重跑（不要现在改 `run_lanes.sh` 或 `git pull` 覆盖它——正在跑的链会读到被改写的脚本；等链结束再合并）：
```
FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 MODEL=Llama-3.2-1B-Instruct KL_STEPS=500 EXTRA_WDOWN="wdown.kl.fr_batch_size=8 wdown.kl.batch_size=16" UPSTREAM_PREFIX=forget10_Llama1B_v2 OUT_PREFIX=forget10_Llama1B_v3 FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=match KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 EVAL_PROFILE=full COEFFS="7" LAYER_SETS="11,12,13" bash scripts/NEAR/sweep_kl.sh <GPU>
```
若日志显示的不是 OOM，把报错发我。

## 5 链还剩的步骤
- **小链**（当前在 `G_10_lunar`，已有 c1）：`G_10_lunar` 的 c1.5、c2；`H_3b_lunar`（3 个系数）、`H_3b_ga_npo`、`H_3b_gd_rmu`、`H_3b_simnpo`。共约 1 个 LUNAR 的 2 个点 + 4 个 3B forget05 步骤（LUNAR 3 点、baseline 5 个方法）。
- **尾链**（`I_new` 已有评测）：`I_old`、`I_lunar`（3 点）、`I_ga`、`I_npo`。其中 `I_old` 的层固定为 12-13-14，但 forget10 × 3B 的选层是 **13**（排名 13、14、12、11），层组应为 11-12-13；要换的话等尾链跑完 `I_old` 后单独重跑：
```
FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=forget10_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget10_ks OUT_PREFIX=forget10_Llama3B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=11,12,13 COEFF=8 MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>
```
