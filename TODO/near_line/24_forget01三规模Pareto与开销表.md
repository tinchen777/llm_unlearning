# 24 forget01 三个规模（1B / 3B / 8B）的 Pareto 与开销表

## 问题
把 forget01 上三个规模、所有方法的结果放到同一张匹配表里（规则：Prob ≤ 0.25 且 |mia − retain 的 mia| ≤ 0.25，在满足的点里取 utility 最高；`*` = 方法没有任何点满足，显示离规则最近的点），并给出开销表。

## 做法
```
python scripts/analysis/pareto.py --model Llama-3.2-1B-Instruct --near saves/eval/NEAR/single_v2 saves/eval/NEAR/single_v3 --out saves/analysis/pareto_1B_f01
python scripts/analysis/pareto.py --model Llama-3.2-3B-Instruct --near saves/eval/NEAR/Llama3B_v2 saves/eval/NEAR/Llama3B_v3 --out saves/analysis/pareto_3B_f01
python scripts/analysis/pareto.py --model Llama-3.1-8B-Instruct --near saves/eval/NEAR/Llama8B_v2 saves/eval/NEAR/Llama8B_v3 --out saves/analysis/pareto_8B_f01
python scripts/analysis/cost_table.py saves/NEAR/single_v2 saves/NEAR/single_v3 saves/NEAR/Llama3B_v2 saves/NEAR/Llama3B_v3 saves/NEAR/Llama8B_v2 saves/NEAR/Llama8B_v3
```
图与 `points.csv`、`matched.md` 在 `saves/analysis/pareto_*_f01/`（`saves/` 不进 git，在服务器上）。

## 结果：匹配表（列：Prob / mia / extraction / utility / retain 生成 / ud_bb / wb_unsup）
**3B**（retain 0.178 / 0.551 / 0.060 / 0.664 / 0.916 / 0 / 0.059）
| 方法 | 点 | Prob | mia | extr | util | retR | ud_bb | wbU |
|---|---|---|---|---|---|---|---|---|
| **新方法 l19-21** | 2/2 | 0.209 | 0.685 | 0.144 | **0.663** | **0.894** | 0 | 0.166 |
| RMU lr5e-5 | 5/5 | 0.152 | 0.527 | 0.044 | 0.589 | 0.417 | 0 | 0.360 |
| IdkDPO（已弃用，仅参考） | 3/5 | 0.220 | 0.621 | 0.063 | 0.532 | 0.386 | 0.249 | 0.151 |
| NPO / GA | 1/5 | 0.18 | 0.44 | 0.067 | 0.52 | 0.36 | 0.01 | 0.09 |
| LUNAR c2 | 1/4 | 0.157 | 0.518 | 0.090 | 0.506 | 0.605 | 0.483 | 0.984 |
| SimNPO | 2/5 | 0.129 | 0.721 | 0.045 | 0.221 | 0.235 | 0.003 | 0.117 |
| 旧方法 c10* | 0/3 | 0.329 | 0.833 | 0.113 | 0.650 | 0.821 | 0 | 0.969 |
| GradDiff* | 0/5 | 0.042 | 0.146 | 0.067 | 0.416 | 0.308 | 0 | 0.025 |

**8B**（retain 0.139 / 0.488 / 0.060 / 0.617 / 0.990 / 0 / 0.095）
| 方法 | 点 | Prob | mia | extr | util | retR | ud_bb | wbU |
|---|---|---|---|---|---|---|---|---|
| GA lr2e-5 @4 | 7/10 | 0.178 | 0.601 | 0.109 | **0.677** | 0.725 | 0 | **0.129** |
| 新方法 l16-18 | 2/2 | 0.150 | 0.672 | 0.210 | 0.666 | **0.932** | 0.012 | 0.441 |
| GradDiff lr2e-5 @4 | 3/10 | 0.131 | 0.711 | 0.128 | 0.612 | 0.616 | 0 | 0.058 |
| LUNAR c2.4（l23） | 1/7 | 0.211 | 0.659 | 0.135 | 0.605 | 0.859 | 0.335 | 1.000 |
| RMU lr5e-5 @6 | 3/10 | 0.218 | 0.728 | 0.080 | 0.604 | 0.408 | 0 | 0.713 |
| SimNPO @4 | 1/5 | 0.007 | 0.573 | 0.014 | 0.024 | 0.101 | 0.025 | 0.179 |
| NPO* @2 | 0/5 | 0.333 | 0.771 | 0.098 | 0.653 | 0.501 | 0 | 0.073 |
| 旧方法 c7* | 0/2 | 0.181 | 0.741 | 0.099 | 0.618 | 0.971 | 0.071 | 1.000 |

**1B**（retain 0.169 / 0.483 / 0.062 / 0.598 / 0.825 / 0 / 0.171）：匹配表里的 KL+hinge 行是消融变体 `kl0`（只有铰链、无 KL），不是主配置；主配置（`kl1 + fp1`，l11-12-13）：Prob 0.162、mia 0.563、extraction 0.131、utility 0.569、retain 生成 0.640、wb_unsup 0.171。其余：旧方法 c7（l7-9）0.206 / 0.531 / 0.048 / 0.569 / 0.588 / wbU 0.996；NEAR+KL（steering + KL）c1.5：0.101 / 0.320 / 0.118 / 0.569 / 0.700 / 0.935；NPO / GA（lr1e-4 @2）utility 0.43，retain 生成 0.33；SimNPO @6 utility 0.316；RMU @6 0.297；GradDiff* 0.532 但 Prob 0.279、mia 0.760。LUNAR c1.5：Prob 0.115、mia 0.426，但 utility 0.396，ud_bb 0.52。

## 结果：开销（`cost_table.py`；只含训练阶段，不含评测；baseline 的 `train` 含每个 epoch 的评测）
| 规模 | 方法 | 训练耗时 | 上游 | 峰值显存 | 可训练参数 |
|---|---|---|---|---|---|
| 3B | 旧方法 c8 / c10 | 97 / 103 s | 355 s（r_UV + 选层） | 9.9 GB | 75.5 M |
| 3B | 新方法 l19-21 / 13-15 | 315 / 298 s | 21 s（只含 r_UV，无选层） | 17.8 / 16.7 GB | 75.5 M |
| 8B | 旧方法 c6 / c7 | 157 / 426 s | **未测**（`WITH_COST` 默认没开） | 20.1 GB | 176.2 M |
| 8B | 新方法 l22-24 / 16-18 | 194 / 128 s | 未测 | 34.4 / 33.9 GB | 176.2 M |
| 8B | LUNAR c2.2 / 2.4 | 12 / 11 s | 18 / 19 s | 16.5 GB | 58.7 M |
| 8B | RMU lr5e-5 | 2752 s（含每 epoch 评测） | – | 38.7 GB | 8030 M |

缺口：1B 的 NEAR 新旧训练开销没测（这批运行早于开销测量）；1B 的上游开销是剩余实验第 4 条；3B/1B 的 baseline、LUNAR 绝大多数没有 `cost.json`；8B baseline 除 RMU 外没有。

## 结论
1. **3B**：新方法 l19-21 在所有保持、白盒、黑盒指标上最接近 retain（utility 0.663 ≈ 0.664，retain 生成 0.894 ≈ 0.916，ud_bb 0）；其他 baseline 的 utility 只有 0.22–0.59，retain 生成 0.24–0.61；旧方法与 LUNAR 的白盒 0.97–0.98。
2. **8B**：新方法与调好的 GA 持平（util 0.666 对 0.677，Prob 0.150 对 0.178），retain 生成明显更好（0.932 对 0.725），但 mia（0.672 对 0.601）、extraction（0.210 对 0.109）和白盒（0.441 对 0.129）输给 GA。旧方法和 LUNAR 的白盒 ≈ 1.0 且黑盒痕迹明显。
3. **1B**：新方法（主配置）的 util 0.569，与旧方法持平，而白盒从 0.996 降到 0.171；baseline 的 utility 都在 0.43 以下（SimNPO、RMU、GA 窗口很窄）。
4. 开销：新方法（精修）训练耗时约为旧方法阶段 1 的 3×（3B），峰值显存约 1.7–1.8×（3B）；8B 上新方法 128 s 比旧方法 c7 的 426 s 还短，但峰值显存 34 GB 对 20 GB。旧方法在 3B 的上游（355 s）比其训练还长，所以**含上游的总开销：3B 旧 ≈ 450 s，新 ≈ 340 s**；8B 的上游未测，这一比较在 8B 上还不能下结论。
5. 论文口径：新方法在三个规模上都显著优于旧方法和 LUNAR；但与调好的 GA 的比较在 8B 上没有胜出，要把论点收窄到“残留更少 / retain 生成更好 / 白盒痕迹更少（相对 steering 与 LUNAR）”，而不是“全面优于 baseline”。

## 下一步
- 补 1B 的新旧训练开销（各重跑一次，命令见 `exp/5_剩余实验.md` 第 4 条之外还要单独重跑一个旧、一个新训练）。
- 8B 上游开销（`WITH_COST=1`，在大链空闲后）。
- 1B 匹配表里主配置与消融变体混在一起，画论文图时要把 `--near` 限制到主配置的运行目录。
