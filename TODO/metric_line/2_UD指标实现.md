# 2 UD 指标实现（BB / WB / UD、TAS 同款打分、选层）

## 问题

用户对 1 号文件里问题的回答：

1. 异常回答要加**重复度**；
2. UD 的汇总方式让我建议；
3. **实现 TAS 同款拒答打分**：与现有 `is_refusal` 很接近就合并，否则写成两个 metric；
4. **无监督白盒**要做，但先当参考；
5. 代码层面先加上 **用 UD 选参数**（选层），实验可能要重跑。

## 做法

论文附录 D / Table 10 给出完整的打分，已按原文实现（`src/evals/detect/refusal.py`）：5 组共 56 条正则、第三人称缺席的抑制规则、21 条规范拒答与 `all-MiniLM-L6-v2` 的余弦（`[0.3, 0.7]` 线性映射到 `[0, 1]` 并截断），分数取两者最大。

| 模块 | 内容 |
|---|---|
| `evals/detect/refusal.py` | TAS 拒答分 `TASRefusalScorer`（正则 + 可选语义通道，加载不了编码器时退回只用正则并警告一次） |
| `evals/detect/repetition.py` | 单条回答内的重复率（词 3-gram）；一组回答里的近重复（Jaccard ≥ 0.8，即同一句话术答所有问题） |
| `evals/detect/scores.py` | BB、WB、UD 的计算 |
| 文本指标 `tas_refusal`、`repetition` | 每条回答多两个字段，汇总出 `tas_refusal_rate`、`dup_rate`（已加入 `*_responses` 的 `expose`） |
| `activation_probe` | 新增无监督离群分 `anom_auc[forget \| benign]`（Mahalanobis，只用良性样本拟合，交叉拟合，取前 16 个主成分） |
| metric `unlearning_detectability` | 配置 `configs/eval/tofu_metrics/unlearning_detectability.yaml`，已加入 probe 实验，baseline 脚本 `PROBE=1` 时也会加入 |
| `NEAR/select_layer.py` | 每层记录 `ud`（steering 后回答的黑盒可检测度）、`footprint`（`coeff·‖r_UV‖/‖forget 均值‖`）；`select.ud_weight` / `footprint_weight` / `max_ud` 控制是否参与选层（默认 0，只记录） |
| `scripts/analysis/detectability_table.py` | 直接读已保存的 `*_responses.json` 和 `TOFU_EVAL.json`，不用重跑模型就能出表 |

指标定义（都以同一个模型内部的"被遗忘 vs 其他"可分性为准，`adv = 2·(AUC − 0.5)`，0 = 和瞎猜一样）：

- **黑盒 BB**：`bb_tas`（只用 TAS 拒答分，对应论文攻击）、`bb`（异常分 = max(拒答分, 乱码, 近重复)）、`entity_bb`（按作者平均后的分数，对应 TAS 把多次查询的证据聚合起来）；
- **白盒 WB**：`wb_sup`（监督探针，oracle 上界）、`wb_unsup`（无监督离群分）；有 retain 模型的探针时，报告的是相对它的超出量；
- **UD 汇总**：`ud` 默认 `= bb`（论文的威胁模型就是纯黑盒），`aggregate: max` 时再与 `wb_unsup` 取最大；`ud_max` 始终报告。**我的建议：论文主指标用黑盒（并给出 `entity_bb`），白盒只作参考列**，原因是监督探针是 oracle，无监督目前不够强。

## 结果

- 用 4.5 万条已保存的回答比较我们的 `refusal` 与 TAS 正则：**一致率 99.96%**（只有 11 条仅我们判拒答、6 条仅 TAS 判拒答；前者多是 "not publicly available" 这类第三人称缺席，TAS 的抑制规则判得更准）。语义通道在本环境没有编码器，没法比较。**结论：先保留两个 metric，不合并**；是否把 `is_refusal` 换成 TAS 正则可以以后再定。
- 已保存结果的离线表（1B，forget01，仅正则通道）：

| 模型 | UD(bb) | bb_tas | entity_bb | wb_sup | 拒答% | `forget_Q_A_ROUGE` | `forget_quality` | `model_utility` |
|---|---|---|---|---|---|---|---|---|
| full | 0 | 0 | 0 | 0.23 | 0 | 0.835 | 0.007 | 0.597 |
| retain | 0 | 0 | 0 | 0 | 0 | 0.418 | 1.0 | 0.598 |
| NEAR c7/L9 | 0 | 0 | 0 | 0.77 | 0 | 0.401 | 0.405 | 0.564 |
| NEAR c12/L6 | 0 | 0 | 0 | 0.82 | 0 | 0.422 | 0.014 | 0.573 |
| LUNAR | **0.30** | 0.15 | **1.00** | – | 15 | 0.199 | 0.919 | 0.440 |
| GradAscent / GradDiff / NPO / SimNPO | 0 | 0 | 0 | 0.25–0.38 | 0 | 0.42–0.74 | 0.003–0.007 | 0.58–0.60 |

  - LUNAR 单条回答的优势只有 0.30，但**按作者聚合后是 1.00**：一个作者 20 条里有 15% 拒答就足以把这个作者认出来，这正是 TAS 聚合证据的威力，所以 `entity_bb` 比 `bb` 更接近真实攻击。
  - baseline 这几个 checkpoint 基本没遗忘（`forget_quality` ≈ 0.003–0.007），BB 为 0 没有意义，要等它们真正遗忘后再比。
- **发现并修复了一个已有的 bug**：文本指标的注册表把 `get_func_params(...)` 存成了字典（`all/required/optional`），`TextMetric` 遍历到的是这三个键而不是参数名，所以 `hallucination_rouge`、以及新加的 `tas_embedder` **从来没有传到函数里**（目前配置值恰好等于函数默认值 0.3，所以没造成数值错误）。已修，并加了测试。
- **完整测试发现并修复的问题**（第一次完整跑：97 通过、3 失败）：
  - 语义通道遇到**空回答**会让编码器报错并中断整个评测（真实模型也可能输出空回答）。已改为：空白文本不进编码器（语义分 0），编码器在任何输入上出错时关闭语义通道并警告一次，不再中断评测；加了测试。
  - 另两个失败是 `tests/test_unlearn_scripts.py` 还在期望旧的输出路径（我按你的要求改成 `saves/tofu_unlearn/<METHOD>/tofu_<MODEL>_<FORGET_SPLIT>` 时没更新测试），已更新，并加了"baseline 评测默认带 probe/UD、`PROBE=0` 关闭"的测试。
- 测试：新增 `tests/test_detectability.py`（9 个）、探针端到端（含 `tas_refusal` / `ud_*` / `anom_auc` 的断言）、选层的端到端（默认关与 `ud_weight` 打开两种）。

## 下一步

1. 用户在真实模型上跑一次 `scripts/eval/baseline_probe.sh`（现在默认先跑 retain 再跑 full，retain 的探针是其余模型的干净参照）以及 NEAR/baseline 的评测，看真实的 `ud_*` 和 `wb_unsup`，并打开语义通道（`HF_HUB_OFFLINE=1` 下需要 `all-MiniLM-L6-v2` 在缓存里，LUNAR 的选层本来就要它）。
2. 用 `scripts/analysis/detectability_table.py --embedder sentence-transformers/all-MiniLM-L6-v2` 看语义通道对 LUNAR / IdkDPO 的 `bb_tas` 有没有提升。
3. 等 baseline 真正遗忘后，画"UD 对遗忘程度"的 Pareto 图，决定论文里报哪几个数。
4. 选层：用 `select.ud_weight` / `footprint_weight` 重新扫一遍，看是否选到更隐蔽的 (系数, 层)（要重跑选层，代码已就绪）。
