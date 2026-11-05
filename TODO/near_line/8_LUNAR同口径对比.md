# 8 LUNAR 与 NEAR 的同口径对比（forget01 两个作者都遗忘）

## 问题

- 现在的 LUNAR 实验只遗忘 forget01 的**前 20 条（1 个作者）**（`FORGET_HF_SPLIT=train[:20]`，LUNAR 论文自己的设定），但 TOFU 指标在**整个 forget01（40 条、2 个作者）**上评测；NEAR 是遗忘整个 forget01。
- 已有结果：被训练的作者 ROUGE 0.09，没训练的作者 ROUGE 0.31，平均 0.20——LUNAR 的 0.198 是两者混合，口径不同。
- LUNAR 的评测也没有用 NEAR 的那套（没有邻居、探针、UD）。

## 做法

- `scripts/LUNAR/single_run.sh`：
  - 新增 `FORGET_HF_SPLIT=train`（遗忘整个 forget split）；默认任务名在它不等于 `train[:20]` 时加后缀（`..._forget01_train_LUNAR_harmful`），不会覆盖一个作者的结果；
  - 第 4 步改为与 NEAR 完全相同的评测 `experiment=eval/tofu/probe`（TOFU 指标 + 回答 + 激活 + 可检测度），`seen=[retain]`、`author_offset` 等参数同 baseline。
- 一个作者的设定保留为默认（是 LUNAR 论文自己的设定，可以另外报告）。

## 结果

待用户跑出后补充。

## 下一步

```bash
FORGET_HF_SPLIT=train bash scripts/LUNAR/single_run.sh 0
```

之后在同一张表里对比 NEAR（B：第 7、8、9 层）与 LUNAR（两个作者）：`forget_quality`、`model_utility`、`forget_Q_A_Prob`、`mia_loss`、`privleak`、retain 生成 ROUGE，以及拒答 / 退化率和 `ud_*`。
