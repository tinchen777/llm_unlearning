# exp：待跑实验的命令与状态板（按 `paper_notes/5_实验清单_必做选做.md` 的方阵）

> 命令前先 `git pull`。结果出来后把数字写进对应的 `*_line` 文档，并更新 `paper_notes/5_*` 的状态。更新于 2026-11-08（大链、小链都已启动）。
> 记号：✅ 已完成　🔄 在跑（脚本链 `scripts/exp/run_lanes.sh`）　❗ 必做、尚未排　○ 选做　**[串]** 按顺序　**[并]** 可并行。

## 文件
| 文件 | 内容 | 状态 |
|---|---|---|
| `1_收尾与参考.md` | forget10 参考、8B RMU、8B LUNAR 连续曲线、8B 旧方法 c6 评测 | ✅ 全部完成（结果见对话与 `near_line` / `baseline_line` 待写） |
| `2_forget05_1B.md` | forget05×1B 的 baseline、LUNAR | 🔄 在小链里（步骤 `F_1b_*`）；GA 2e-5 已有 3 个点 |
| `3_forget05_3B.md` | forget05×3B 的 NEAR 上游→旧→新、baseline、LUNAR | 上游 ✅（选层 14）；其余 🔄 在两条链里 |
| `4_今晚大实验_明天小实验.md` | 队列 A–E 的说明与理由 | 被 `scripts/exp/run_lanes.sh` 取代（命令相同，脚本带续跑） |
| `6_第一批结果与lr修正.md` | 大链跑完后的结果（forget05 三个规模、forget10×1B）、**lr 修正重跑命令**、`G_10_new` 失败的排查、链还剩什么 | ❗ 新增 |
| `scripts/exp/run_lanes.sh` | 两条链：大链 70GB（8B），小链 40GB（3B / 1B）；失败不中断；可续跑（`logs/exp/done/`） | 🔄 都在跑 |
| `5_剩余实验.md` | 链之外还剩什么（本文件下面第二节的详细命令） | ❗ 新增 |

## 一、两条链里的步骤（正在跑，按顺序）
**大链（70GB，8B）：**
| 步骤 | 内容 | 状态 |
|---|---|---|
| `A_up` → `A_sel` → `A_new` / `A_old` | forget05×8B NEAR：r_UV → 选层（c7，层 15–20）→ 新方法 l16-17-18（300 步）、旧方法 c7 l17-18-19 | 🔄 |
| `C_lunar` | forget05×8B LUNAR，固定 l23，c1.5 / 2 / 2.4 / 2.8 | 🔄 排队 |
| `B_ga_gd`、`B_simnpo`、`B_npo_rmu` | forget05×8B baseline（lr：GA / GD 1e-5，SimNPO / NPO / RMU 2e-5） | 🔄 排队（选做格子，允许跑不完） |
| `D_up`、`D_sel`（`WITH_COST=1` 才有） | 8B forget01 steering 上游开销 | ○ 默认没开 |

**小链（40GB，3B / 1B）：**
| 步骤 | 内容 | 状态 |
|---|---|---|
| `E_new`、`E_old` | forget05×3B NEAR：新方法（l19-21 + l13-15，300 步）、旧方法 c8 l12-14 | 🔄 |
| `F_1b_ga_npo`、`F_1b_lunar`、`F_1b_gd_rmu`、`F_1b_simnpo` | forget05×1B baseline 与 LUNAR | 🔄 排队 |
| `G_10_up`→`G_10_sel`→`G_10_new` / `G_10_old`、`G_10_base`、`G_10_lunar` | forget10×1B：NEAR 新旧（新 500 步）、GA + NPO、LUNAR | 🔄 排队 |
| `H_3b_lunar`、`H_3b_ga_npo`、`H_3b_gd_rmu`、`H_3b_simnpo` | forget05×3B LUNAR 与 baseline（lr 为推算值） | 🔄 排队 |

## 二、链之外还剩的（详见 `5_剩余实验.md`）
| # | 内容 | 状态 | 何时做 |
|---|---|---|---|
| 1 | 看链的结果后**修正 lr 并重跑** baseline（崩溃 / 不遗忘的方法；forget05、forget10 的 lr 都是推算值） | ❗ 依赖结果 | 小链 `F_*`、`H_*` 出轨迹后 |
| 2 | **forget05 新方法的改进**（1B：mia 0.88、utility 0.51 偏离 retain）：retain 权重 / 更深层位置 / 更多步 | ❗ 命令待结果 | 小链 `E_new` 出来后，对照 1B 与 3B |
| 3 | 8B forget05 **旧方法的层核对**（链里固定 17-18-19，选层可能不是 19） | ❗ 小检查 | `A_sel` 完成后 `cat` 选层文件，不是 19 就用 `FORCE` 重跑 `A_old` |
| 4 | **1B forget01 的 steering 上游开销补测**（3B 已有，8B 默认没开） | ❗ 小（几分钟） | 链空档，命令见 `5_*` |
| 5 | forget10×1B 的其余 baseline（GD、SimNPO、RMU） | ○ | 看 GA + NPO 轨迹后 |
| 6 | forget10×3B：🔄 第三条链 `LANE=third`（步骤 `I_*`，见 `5_*` 第 7 节）；forget05×8B 新方法 22-23-24；forget05×8B 的 baseline 重跑 | ○ | 时间有余 |
| 7 | 完整评测补到 forget01 的入选 baseline；多种子；AltPO 类对照 | ○ | 基调定后 |
| 8 | 出结果后的**分析**：每个设置 × 模型的 Pareto 图与匹配表、开销表、定性样例、`near_line` / `baseline_line` / `paper_notes/5_*` 文档更新 | ❗ | 链跑完后（命令见 `5_*`） |
