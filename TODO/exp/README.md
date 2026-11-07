# exp：待跑实验的命令（按 `paper_notes/5_实验清单_必做选做.md` 的必做格子）

> 命令前先 `git pull`。`<GPU>` 换成卡号。所有命令都是单行（可直接粘贴）。结果出来后把数字写进对应的 `*_line` 文档，并更新 `paper_notes/5_*` 的状态。
> 符号：**[串]** 必须按顺序（后一条依赖前一条的输出）；**[并]** 与同组其他命令互不依赖，可放在不同 GPU 上同时跑。

| 文件 | 内容 | 组内顺序 | 与其他文件的关系 |
|---|---|---|---|
| `1_收尾与参考.md` | forget10 参考、8B RMU、8B LUNAR 连续曲线、8B 旧方法 c6 评测 | 四条互相独立，**全并行** | 与 2、3 无依赖；forget10 参考是第 5 步（forget10×1B）的前提 |
| `2_forget05_1B.md` | forget05×1B 的 baseline、LUNAR（NEAR 新旧已在跑，不在此列） | baseline：先试跑 GA+NPO，**看轨迹后**再跑其余三个；LUNAR 与 baseline 并行 | baseline 的 lr 结论决定 3B 的 lr |
| `scripts/exp/run_lanes.sh` | **一键脚本**：70GB 卡一条链（8B），40GB 卡一条链（3B/1B），失败不中断、可续跑；`bash scripts/exp/run_lanes.sh <大卡> <小卡>` | 两条链并行，链内按优先级串行 | 取代 `4_*` 的队列 A–E |
| `4_今晚大实验_明天小实验.md` | 今晚：8B（forget05 的 NEAR/baseline/LUNAR、forget01 上游开销）与 3B forget05 NEAR 新旧；明天：1B/3B 的小实验与 forget10×1B | 队列 A–E 互相独立，各占一卡 | 依赖 3 的上游（3A-1、3A-2 已完成） |
| `3_forget05_3B.md` | forget05×3B 的 NEAR 上游→旧→新、baseline、LUNAR | NEAR：上游（r_UV→选层）**串**，旧 / 新方法两条**并**；baseline、LUNAR 与 NEAR 并行 | baseline 的 lr 等 2 的试跑结论；NEAR 的层 / 步数可参考 forget05×1B 新旧结果 |

## 并行 / 顺序总览
```
可立刻并行开跑：  [1 的四条]  [2 的 baseline 试跑]  [2 的 LUNAR]  [3 的 NEAR 上游]  [3 的 LUNAR]
等上一步结果后：   2 试跑轨迹 → 2 其余 baseline → 3 baseline
                  3 上游 → 3 旧方法 ∥ 3 新方法
```
GPU 估计：8B 的几条各占一张大卡（RMU 约 48 GB 以上，开了梯度检查点和小批）；其余 1B / 3B 的单条可 2–3 条共卡，显存紧就分开。
