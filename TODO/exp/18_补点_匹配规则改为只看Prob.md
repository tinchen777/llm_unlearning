# 18 补点：匹配规则只看 Prob（mia 放宽），补遗忘不足 / 完全没有 / 过遗忘

> 先 `git pull`。决定：**匹配规则 = `forget_Q_A_Prob ≤ 0.25`**；mia 仍然报告，但不作为匹配条件。方法保持原样（E8 在大遗忘集上 mia 偏高、Act 在 8B 上 mia 偏高，都如实写）。出图 / 出表时给 `pareto.py` 加 `--mia-tol 9`（容差足够大，等于只看 Prob）。脚本 `scripts/exp/run_fill_lanes.sh` 共 29 步（大链 14 步、小链 8 步、小链 2 共 7 步；含 X4 的采样）（我只用 `DRY_RUN` 看过，没有真实运行；学习率是按现有点的单调趋势猜的，没验证过）。

## 一、改规则后还缺哪些（按 `saves/` 现有评测核对）
| 类 | 格子 × 方法 | 补什么 |
|---|---|---|
| **遗忘不足**（所有点 Prob > 0.25） | 1B forget05 SimNPO（最小 0.32）；3B forget05 GA（0.67）、SimNPO（0.36）、RMU（0.34）；8B forget05 SimNPO（0.52）、RMU（0.25，差一点）；3B forget10 GD（0.53）、SimNPO（0.60）；8B forget10 SimNPO（0.55）、NEAR-Act（0.40） | 更强的 lr；Act 用 17-18-19 加大 c（8、10、12） |
| **完全没有 / 没跑完** | 1B forget10 GD、SimNPO、RMU；3B forget10 GA、NPO、LUNAR；8B forget10 GD、NPO、RMU（目录里只有 checkpoint-0 / -13，没跑完）、LUNAR | 补跑（8B 用 LOWMEM；8B 那三个会覆盖没跑完的同名目录） |
| **过遗忘**（Prob≈0，utility≈0–0.2） | 8B forget01 SimNPO（util 0.02）、8B forget05 GA（0.15）、8B forget10 GA（0.08）、3B forget01 SimNPO（0.22） | 弱一档的 lr（2e-5、5e-6、5e-6、5e-5） |
| 已经够 | 其余格子；改规则后 8B forget01 / forget05 的 Act、所有格子的 E8 都有匹配点；3B forget01 GD、1B forget01 GD、8B forget01 NPO 也有 | — |

注意：过遗忘的几项在 Prob-only 规则下**已经算匹配**（取 utility 最好的那个点），补弱一档的 lr 是为了得到一个 utility 更合理的点，不是必须。

## 二、一条命令（一条大链 + 两条小链）
```
bash scripts/exp/run_fill_lanes.sh <80GB卡> <40GB卡> <另一张40GB卡>
```
- **大链（8B，用 LOWMEM）**：forget05 的 SimNPO、RMU（lr 1e-4）；forget10 的 SimNPO（1e-4）、Act（c8/10/12）、GD（2e-5）、NPO（2e-5）、RMU（5e-5）、LUNAR（层 23，系数 2 / 2.4 / 2.8）；弱一档：forget01 SimNPO（2e-5）、forget05 GA（5e-6）、forget10 GA（5e-6）；**X4（8B）**：NPO 带检查点重训（lr 5e-5，`save_steps=10` 只留第 10 步那一个检查点，约 16GB）→ GA（`checkpoint-4`）/ NPO（`checkpoint-10`）的采样一致性 → E8 的采样（`validate_fix.sh` 的 `8B_f01`，`SKIP_TRAIN=1`）。
- **小链 1（1B 与 3B forget05）**：1B forget05 SimNPO（2e-4）；3B forget05 GA（2e-5）、SimNPO（2e-4）、RMU（1e-4）；1B forget10 GD（2e-5）、SimNPO（2e-4）、RMU（5e-5）；**X4（1B）**采样。
- **小链 2（3B forget10）**：GD（2e-5）、SimNPO（1e-4）、GA（2e-5）、NPO（2e-5）、LUNAR（系数 1 / 1.5 / 2）；**X4（3B）**采样；最后是弱一档：3B forget01 SimNPO（5e-5）。
- **X4 的采样用的检查点**（forget01，匹配点 = Prob ≤ 0.25 里 utility 最好的）：1B 与 3B 的 GA、NPO 都是 lr 1e-4 的 `checkpoint-2`；8B 的 GA 是 lr 2e-5 的 `checkpoint-4`，NPO 是 lr 5e-5 的 `checkpoint-10`。输出 `saves/analysis/sample_<1B|3B|8B>_ga_npo/summary.md`，retain 与 full 作参照。这些检查点的权重文件我这边看不到（上传的 `saves/` 不含权重），请在你的机器上 `ls` 确认；缺了就会 FAIL，不会影响其他步骤。
- 只跑一条链：`LANE=big ...`、`LANE=small ...`、`LANE=small2 ...`；一张卡：`LANE=all bash scripts/exp/run_fill_lanes.sh <卡>`；看步骤：`DRY_RUN=1 bash scripts/exp/run_fill_lanes.sh 0 1 2`；进度：`tail -f logs/fill/big.log logs/fill/small.log logs/fill/small2.log`。
- 每步的结果：baseline 在 `saves/tofu_unlearn/<方法>/...`（每个 epoch 都评测），LUNAR 在 `saves/eval/LUNAR/...`，Act 在 `saves/eval/NEAR/forget10_Llama8B_v2/...`。
- 提醒：脚本改成了三条链。**已经在后台跑的链不要在运行中 `git pull`**（bash 边读边执行，文件被替换会出错）；等它跑完或先停掉，再拉取后用同一条命令续跑（步骤名没变，已完成的会跳过）。

## 三、出表
```
python scripts/analysis/pareto.py --model <模型> --forget <split> --near saves/eval/NEAR/<prefix>_v2 saves/eval/NEAR/<prefix>_v3 --lunar <LUNAR 目录> --mia-tol 9 --out saves/analysis/pareto_<...>
```
（`--lunar` 只收一个目录；LUNAR 的评测分散在 `saves/eval/LUNAR/single` 和 `forget05_Llama3B_v2` 等，可以把它们软链接到同一个目录再传。）
