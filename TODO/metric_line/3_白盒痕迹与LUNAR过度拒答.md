# 3 白盒痕迹与 LUNAR 过度拒答带来的修正

## 问题

用 UD 指标评测新协议下的 NEAR 与 LUNAR（forget01 / forget005）。

## 结果

- **NEAR 黑盒几乎不可检测（`ud_bb` 0–0.08），白盒很容易检测**：无监督离群检测 `wb_unsup` 0.985–0.996（原始 AUC 0.99，full / retain 约 0.59），监督探针 0.72–0.99。原因是系数 7 的大位移让 forget 的激活整体偏离，forget 的 40 个样本偏移方向一致，很容易聚成一个远离良性样本的簇。
- LUNAR（forget01）：`ud_bb` 0.207、`entity_bb` 0.67、白盒 `wb_unsup` 0.57；forget005：`ud_bb` 0.25、`wb_unsup` 0.96。
- **黑盒指标的局限**：它衡量的是"被遗忘 vs 其他"的**差异**。LUNAR（forget01）对良性问题也拒答 32%（对 forget 的拒答是 55%），所以 `ud_bb` 只有 0.207；但一个对所有问题都乱拒答的模型，攻击者也能发现它有问题（不是发现"哪些被遗忘"，而是模型被破坏）。这部分体现在 `model_utility`（0.269）和 holdout ROUGE 里，需要在论文里和 UD 一起报。
- **`is_refusal` 与 TAS 的差别**：NEAR B 的 forget 回答里有 7.5% 被我们原来的 `refusal` 标记（"There is no information about ..."），TAS 的抑制规则不算拒答（第三人称的缺席是普通否定回答）。TAS 更准确，建议以后以 TAS 为准（`tas_refusal_rate`），把 `forget_refusal_rate` 换成 TAS 的正则。

## 下一步

1. 论文里 UD 的汇报：黑盒 `bb` / `entity_bb` 为主，白盒 `wb_unsup` 作为参考，同时报 `model_utility` 和 benign 拒答率（`benign_tas_rate`）。
2. NEAR 线降低白盒痕迹（见 `near_line/9`），用 `wb_unsup` 衡量改进。
3. 把 `is_refusal` 换成 TAS 正则（需要重算已有的 `refusal_rate` 才一致，或者新旧并列报告）。
