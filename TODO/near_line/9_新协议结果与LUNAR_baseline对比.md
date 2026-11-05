# 9 新数据协议下的 NEAR 结果，与 LUNAR、baseline 的对比（1B，forget01）

## 问题

数据协议修正后（r_UV 与 W_down 的 retain 用 train 邻居，评测用 heldout 邻居），NEAR 的结果有没有变？与 LUNAR（现在也遗忘整个 forget01）、各 baseline 相比如何？

## 做法

在 `single_v2/` 下重跑：单层 c7/L9、A（第 8、9 层）、B（第 7、8、9 层，`increment` + 顺序训练）；`_ks` 文件夹里是 K=1/3/5/10 的 r_UV 和系数×层的 steering 扫描。LUNAR 两个设定（forget01 两个作者、forget005 一个作者）；full / retain 重新评测；baseline 为之前跑的 6 个方法的最后一个 checkpoint。

## 结果

### 1. 协议修正对 NEAR 几乎没有影响（符合预期），邻居评测现在是干净的

| | 旧协议 B | 新协议 B | retain 参考 |
|---|---|---|---|
| `forget_Q_A_Prob` | 0.163 | 0.161 | 0.169 |
| `mia_loss` | 0.552 | 0.526 | 0.483 |
| `privleak` | −25.7 | −21.4 | – |
| retain 生成 ROUGE | 0.474 | 0.516 | – |
| **heldout 邻居 ROUGE** | – | **0.343** | 0.358（full 0.378） |

heldout 邻居上 NEAR 与 retain 模型一样，对没见过的邻居没有副作用。`_ks` 的 steering 扫描最优仍是 c12/L6（`unseen_gap` 0.272）、c7/L9（0.312），与旧协议（0.260、0.310）一致。

### 2. K 的消融（r_UV 与 K=10 的余弦）

| K | 第 6 层 | 第 9 层 | 第 11 层 |
|---|---|---|---|
| 1 | 0.73 | 0.81 | 0.81 |
| 3 | 0.93 | 0.95 | 0.93 |
| 5 | 0.97 | 0.98 | 0.97 |
| 10 | 1.0 | 1.0 | 1.0 |

K ≥ 3 已经接近 K=10，K=1 明显不稳；范数几乎不随 K 变。建议主线用 K=5。

### 3. forget01 对比（新协议）

| | retain | NEAR 单层 L9 | NEAR A | **NEAR B** | LUNAR（2 作者，c2） | NPO | GradAscent |
|---|---|---|---|---|---|---|---|
| `forget_Q_A_Prob` | 0.169 | 0.457 | 0.311 | **0.161** | 0.024 | 0.340 | 0.609 |
| `mia_loss` | 0.483 | 0.883 | 0.774 | **0.526** | 0.101 | 0.840 | 0.889 |
| `extraction_strength` | 0.062 | 0.108 | 0.068 | **0.063** | 0.012 | 0.114 | 0.271 |
| `model_utility` | 0.598 | 0.568 | 0.555 | 0.536 | **0.269** | 0.584 | 0.594 |
| retain 生成 ROUGE | 0.825 | 0.665 | 0.614 | 0.516 | 0.374 | 0.695 | 0.761 |
| holdout ROUGE | 0.394 | 0.393 | 0.395 | 0.395 | 0.261 | 0.379 | 0.397 |
| 拒答率（TAS） | 0 | 0 | 0 | 0 | **0.55** | – | – |

- NEAR B 的 Prob、extraction、MIA 与 retain 模型几乎一致；代价是 `model_utility` −0.06，retain 生成 ROUGE −0.31。
- **LUNAR 遗忘整个 forget01 后严重过度**：`model_utility` 0.269、holdout ROUGE 0.26、`privleak` +65，forget 回答 55% 是 "I cannot provide information ... illegal or harmful activities"，**对良性问题的拒答率也有 32%**。一个作者（forget005）也有 `model_utility` 0.449、拒答 30%。注意 LUNAR 只跑了默认设置（系数 2、`LUNAR_harmful`），没有调参，对比前需要调参（见下一步）。
- **baseline 基本没遗忘**：GradAscent / GradDiff / SimNPO / IdkDPO 的 `forget_Q_A_Prob` 0.6–0.86（retain 0.17），NPO 最好 0.34；RMU 的 `model_utility` 为空。forget01 只有 40 条，每个 epoch 只有 2 步，总共 20 步，学习率 1e-5 不够，需要调参（`LR`、`EPOCHS`），可能还有 bf16 的问题。

### 4. 在 B 上加答案 token、提高 retain 权重（旧协议，开发阶段）

| | B | **B + 答案（rw1）** | B + 答案（rw3） | A | A + 答案（rw1） | A + 答案（rw3） |
|---|---|---|---|---|---|---|
| `forget_quality` | 0.766 | 0.919 | 0.919 | 0.579 | 0.919 | 0.766 |
| `forget_Q_A_Prob` | 0.163 | **0.198** | 0.204 | 0.301 | 0.299 | 0.310 |
| `mia_loss` | 0.552 | **0.502** | 0.510 | 0.746 | 0.732 | 0.739 |
| `model_utility` | 0.524 | **0.551** | 0.552 | 0.552 | 0.567 | 0.568 |
| retain 生成 ROUGE | 0.474 | **0.606** | 0.607 | 0.581 | 0.649 | 0.687 |
| `retain_answer_prob` | 0.539 | **0.753** | 0.767 | 0.735 | 0.805 | 0.815 |

**B + 答案 token（rw1）是目前最好的 NEAR 配置**：遗忘指标不变（Prob 0.198 ≈ retain 0.169，MIA 0.50 ≈ 0.48），retain 生成 ROUGE 0.474 → 0.606，答案概率 0.54 → 0.75；`retain_weight` 3 比 1 几乎没有额外提升。

### 5. 可检测度（`ud_*`）：黑盒隐蔽，白盒痕迹大

| | NEAR 各配置 | LUNAR forget01 | LUNAR forget005 |
|---|---|---|---|
| 黑盒 `ud_bb` | **0–0.08** | 0.207（按作者聚合 0.67） | 0.25 |
| 白盒监督探针 `wb_sup` | 0.72–0.99 | 0.88 | 0.94 |
| 白盒无监督 `wb_unsup` | **0.985–0.996** | 0.57 | 0.96 |

NEAR 在黑盒上几乎没有痕迹（不拒答、不乱码），但**激活上的痕迹非常明显**：无监督离群检测几乎完美（原始 AUC 0.99，full 和 retain 约 0.59）；forget01 上比 LUNAR（0.57）还大。原因是系数 7 的大位移让 forget 的激活整体偏离。这是两条线汇合要解决的问题。

## 下一步

1. **NEAR 降低白盒痕迹**：用较小系数 + 更多层；`select.footprint_weight` 选层；训练目标改成"匹配干净分布"（把 forget 的激活拉到 neighbor 或 retain 模型激活的分布内，而不是常数平移）。
2. **新协议下重跑最好的配置**（B + 答案 token，`INCLUDE_ANSWER=1 RETAIN_WEIGHTS="1"`）。
3. **公平性**：LUNAR 调参（系数 0.5、1、1.5；可能还要换 `LUNAR_unverified`），baseline 调参（`LR=5e-5/1e-4`、`EPOCHS`），再对比。
4. forget005 上的 NEAR（作者 200）。
