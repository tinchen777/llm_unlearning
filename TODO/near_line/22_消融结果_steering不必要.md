# 22 消融结果：去掉 steering（纯 KL + 铰链 + 邻居）（1B，forget01）

## 问题
`near_line/20`：遗忘的主体是否其实是 KL + 铰链，steering（激活重定向）不必要？

## 做法
`kl_layers/8_ablate_steering.sh`：层 11-12-13，c1，KL=1，铰链 μ=1，目标 τ=`unseen`（原模型在 holdout 上的答案概率，≈0.16）、模式 `match`，完整评测。（`klonly` 变体因 bug 失败，已修，待重跑。）

## 结果（retain 模型：Prob 0.169、mia 0.483、extraction 0.062、utility 0.598、`forget_quality` 1.0）
| 变体 | Prob | extr | mia | utility | retain 生成 | `ud_bb` | `wb_sup` | `wb_unsup` | `forget_quality` |
|---|---|---|---|---|---|---|---|---|---|
| withsteer（steering 初始化 + 锚 + KL + 铰链） | 0.161 | 0.128 | 0.557 | 0.558 | 0.647 | 0 | 0.554 | 0.583 | 0.029 |
| noanchor（steering 初始化，无锚） | 0.161 | 0.128 | 0.563 | 0.559 | 0.630 | 0 | 0.564 | 0.653 | 0.029 |
| **nosteer（无 steering：KL + 铰链）** | **0.162** | 0.131 | **0.563** | **0.569** | 0.640 | 0 | 0.612 | **0.171** | **0.165** |
| hingeonly（无 steering、无 KL） | 0.161 | 0.090 | 0.722 | 0.580 | 0.653 | 0 | 0.501 | 0.255 | 0.014 |
| 参照：c7 steering 不加 KL | 0.206 | 0.048 | 0.531 | 0.569 | 0.588 | 0.046 | 0.977 | 0.996 | 0.766 |
| 参照：c1 steering + KL（无铰链） | 0.556 | 0.383 | 0.956 | 0.572 | 0.754 | 0 | 0.576 | 0.568 | – |

逐层无监督离群 AUC（第 11–15 层）：withsteer 0.53/0.70/0.72/0.67/0.79；nosteer 0.50/0.53/0.55/0.47/0.59；hingeonly 0.50/0.52/0.47/0.63/0.57；c7：0.91/0.92/0.84/0.74/0.70（第 9–10 层 1.0）。

## 结论
1. **steering 不必要**：nosteer 的 Prob（0.162）、mia（0.563）、utility（0.569）、retain 生成（0.640）与 withsteer 相同甚至略好。
2. **steering 是白盒痕迹的来源**：`wb_unsup` withsteer 0.583 → nosteer **0.171**（逐层 AUC 接近 0.5，没有离群）；之前"遗忘 ⇔ 白盒痕迹"的权衡来自 steering 的一致方向偏移，去掉它之后，遗忘到 retain 水平（Prob 0.162）不需要付出白盒代价。白盒局限消失（至少在 1B）。
3. **`match` + τ=`unseen` 不过冲，落在目标上**：Prob 0.161 – 0.162（目标≈0.16，retain 0.169），mia 0.56（retain 0.483），不再有 hinge 的过冲（0.08）；τ 不依赖 retain 模型。
4. **KL 的贡献（hingeonly 对 nosteer）**：mia 0.722 → 0.563（更接近 retain）、`forget_quality` 0.014 → 0.165、白盒 0.255 → 0.171、回答中的实体名换成邻居式的名字（nosteer 的首词是 Mubarak / Leonid / Pavel；hingeonly 仍是 Nikolai / Basil 且回答怪异："gender-fluid"、"however, his birth country is actually -"）；但 extraction 略高（0.131 对 0.090）。
5. **仍有的短板**：(a) `extraction_strength` 0.13 vs retain 0.062（给出答案前缀后仍能续写出部分答案；steering c7 是 0.048）；(b) `forget_quality` 0.165（retain 1.0；c7 steering 0.766）：n=40 时离散，但方向上 c7 更好（铰链只压 p(真实答案)，truth ratio 的分布与 retain 不同）；(c) utility 0.569（retain 0.598）、retain 生成 0.64（retain 0.825），比 steering c1 + KL（0.754）低；(d) 释义 / 扰动 Prob 只有我们自己的运行，没有 retain / full 参照（参照评测用的 fast 配置没有它们）。
6. 单次运行，n=40。`klonly` 待重跑（预期 Prob 不动）。

## 对方法与论文的影响
- **方法重新定位**：纯 KL + 铰链 + 邻居（无 steering）是更好、更简单的方法：不需要 r_UV、不需要 steering 选层、不需要阶段 1；邻居仍要用（KL 的答案前缀、保留项）。这个方法与 WHP / AltPO 近（见 `paper_notes/3` §13），定位与区别要重写。
- **"选层"问题改变**：不再有 steering，"选哪几层编辑"要重新定义（目前 11-12-13 是沿用的）。

## 下一步
1. 重跑 `klonly`（`VARIANTS="klonly" bash scripts/NEAR/kl_layers/8_ablate_steering.sh <GPU>`）。
2. 对 nosteer 的层 / 层数 / 位置做扫描（编辑哪几层最好，是否需要多层，层越深越便宜？）；μ、λ、步数。
3. 补 retain / full 在 forget01 上的完整评测（释义 / 扰动 Prob）；对 nosteer 查 extraction 与 forget_quality 的短板（比如在铰链里加入答案前缀续写的项）。
4. 在 3B 上复现 nosteer（再决定 8B）。
5. 数据消融变成核心（邻居是整个方法）：错配邻居、TOFU retain 作者当邻居、邻居数量 K、邻居答案风格。
