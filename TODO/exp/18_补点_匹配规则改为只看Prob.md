# 18 补点：匹配规则只看 Prob（mia 放宽），补遗忘不足 / 完全没有 / 过遗忘

> 先 `git pull`。决定：**匹配规则 = `forget_Q_A_Prob ≤ 0.25`**；mia 仍然报告，但不作为匹配条件。方法保持原样（E8 在大遗忘集上 mia 偏高、Act 在 8B 上 mia 偏高，都如实写）。出图 / 出表时给 `pareto.py` 加 `--mia-tol 9`（容差足够大，等于只看 Prob）。脚本 `scripts/exp/run_fill_lanes.sh` 共 24 步（我只用 `DRY_RUN` 看过，没有真实运行；学习率是按现有点的单调趋势猜的，没验证过）。

## 一、改规则后还缺哪些（按 `saves/` 现有评测核对）
| 类 | 格子 × 方法 | 补什么 |
|---|---|---|
| **遗忘不足**（所有点 Prob > 0.25） | 1B forget05 SimNPO（最小 0.32）；3B forget05 GA（0.67）、SimNPO（0.36）、RMU（0.34）；8B forget05 SimNPO（0.52）、RMU（0.25，差一点）；3B forget10 GD（0.53）、SimNPO（0.60）；8B forget10 SimNPO（0.55）、NEAR-Act（0.40） | 更强的 lr；Act 用 17-18-19 加大 c（8、10、12） |
| **完全没有 / 没跑完** | 1B forget10 GD、SimNPO、RMU；3B forget10 GA、NPO、LUNAR；8B forget10 GD、NPO、RMU（目录里只有 checkpoint-0 / -13，没跑完）、LUNAR | 补跑（8B 用 LOWMEM；8B 那三个会覆盖没跑完的同名目录） |
| **过遗忘**（Prob≈0，utility≈0–0.2） | 8B forget01 SimNPO（util 0.02）、8B forget05 GA（0.15）、8B forget10 GA（0.08）、3B forget01 SimNPO（0.22） | 弱一档的 lr（2e-5、5e-6、5e-6、5e-5） |
| 已经够 | 其余格子；改规则后 8B forget01 / forget05 的 Act、所有格子的 E8 都有匹配点；3B forget01 GD、1B forget01 GD、8B forget01 NPO 也有 | — |

注意：过遗忘的几项在 Prob-only 规则下**已经算匹配**（取 utility 最好的那个点），补弱一档的 lr 是为了得到一个 utility 更合理的点，不是必须。

## 二、一条命令（大小链）
```
bash scripts/exp/run_fill_lanes.sh <80GB卡> <40GB卡>
```
- **大链（8B，用 LOWMEM）**：forget05 的 SimNPO、RMU（lr 1e-4）；forget10 的 SimNPO（1e-4）、Act（c8/10/12）、GD（2e-5）、NPO（2e-5）、RMU（5e-5）、LUNAR（层 23，系数 2 / 2.4 / 2.8）；弱一档：forget01 SimNPO（2e-5）、forget05 GA（5e-6）、forget10 GA（5e-6）。
- **小链（1B/3B）**：1B forget05 SimNPO（2e-4）；3B forget05 GA（2e-5）、SimNPO（2e-4）、RMU（1e-4）；3B forget10 GD（2e-5）、SimNPO（1e-4）、GA（2e-5）、NPO（2e-5）、LUNAR（系数 1 / 1.5 / 2）；1B forget10 GD（2e-5）、SimNPO（2e-4）、RMU（5e-5）；弱一档：3B forget01 SimNPO（5e-5）。
- 只跑一条链：`LANE=big ...` 或 `LANE=small ...`；一张卡：`LANE=all bash scripts/exp/run_fill_lanes.sh <卡>`；看步骤：`DRY_RUN=1 bash scripts/exp/run_fill_lanes.sh 0 1`；进度：`tail -f logs/fill/big.log logs/fill/small.log`。
- 每步的结果：baseline 在 `saves/tofu_unlearn/<方法>/...`（每个 epoch 都评测），LUNAR 在 `saves/eval/LUNAR/...`，Act 在 `saves/eval/NEAR/forget10_Llama8B_v2/...`。

## 三、出表
```
python scripts/analysis/pareto.py --model <模型> --forget <split> --near saves/eval/NEAR/<prefix>_v2 saves/eval/NEAR/<prefix>_v3 --lunar <LUNAR 目录> --mia-tol 9 --out saves/analysis/pareto_<...>
```
（`--lunar` 只收一个目录；LUNAR 的评测分散在 `saves/eval/LUNAR/single` 和 `forget05_Llama3B_v2` 等，可以把它们软链接到同一个目录再传。）
