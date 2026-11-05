# 8 LUNAR 与 NEAR 的同口径对比；forget005（一个作者）

## 问题

- 旧的 LUNAR 实验只遗忘 forget01 的前 20 条（1 个作者），却在整个 forget01（40 条、2 个作者）上评测；NEAR 遗忘整个 forget01。口径不同（被训练的作者 ROUGE 0.09，没训练的 0.31，平均 0.20 是混合值）。
- LUNAR 的评测也没有 NEAR 那套（邻居、探针、UD）。

## 做法（用户的决定）

- 旧的 LUNAR 实验不重跑：用户把结果目录改名为 `forget005`（一个作者，LUNAR 论文自己的设定），评测要**按一个作者重新评测**。
- forget01 的 LUNAR 正常跑：遗忘整个 forget01、在 forget01 上评测，任务名没有 `_train` 后缀。
- 之后 NEAR 也可能在 forget005 上跑。没有 retain995 模型，forget005 上依赖 retain 模型的评测（`forget_quality`、`privleak`）先不管，不改代码。

代码与脚本：

- **新的"虚拟 split" `forget005`**（`datasets_utils.VIRTUAL_SPLITS`）：= forget01 的前 20 行（TOFU 每个作者的 20 条连续存放，forget01 = 作者 199、200，所以 forget005 = 作者 199）；`forget005_perturbed` 同理。所有用 `${forget_split}` 加载数据的地方（训练、各个评测指标、MIA）自动生效，不用逐个改配置；对它再切片会报错。
- LUNAR：`forget_hf_split` 默认改为 `train`（整个 forget split），任务名不再加后缀；评测与 NEAR 完全相同（`experiment=eval/tofu/probe`，TOFU 指标 + 回答 + 激活 + 可检测度）。
- `NEIGHBOR_SPLIT`、`AUTHOR_OFFSET` 由 `FORGET_SPLIT` 推出（forget01 / forget005 → neighbor01 / neighbor005、199；forget05 → 191；forget10 → 181），LUNAR 和 NEAR 的 `single_run.sh` 都一样。
- `baseline_probe.sh`：增加 forget005 一组（retain 模型用官方 retain99，它同样没见过这两个作者），`ONLY_SPLITS="forget005"` 可只跑它。
- 测试：`forget005` 的映射单元测试、LUNAR 入口测试（默认整个 split、无后缀）。

## 结果

待用户跑出后补充。

## 下一步

1. 旧 LUNAR 结果改名后，**按一个作者重新评测**（只跑第 4 步）：

   ```bash
   mv saves/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget01_LUNAR_harmful saves/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget005_LUNAR_harmful
   mv saves/eval/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget01_LUNAR_harmful_c2_l10_all saves/eval/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget005_LUNAR_harmful_c2_l10_all
   FORGET_SPLIT=forget005 bash scripts/LUNAR/single_run.sh 0 "4"
   ```

2. LUNAR 在 forget01 上正常跑：`bash scripts/LUNAR/single_run.sh 0`。
3. full / retain 在 forget005 上的评测（NEAR 跑 forget005 时要用）：`ONLY_SPLITS="forget005" bash scripts/eval/baseline_probe.sh 0`。
4. 之后在同一张表里对比 NEAR（B：第 7、8、9 层）与 LUNAR：`model_utility`、`forget_Q_A_Prob`、`mia_loss`、retain 生成 ROUGE、拒答 / 退化率和 `ud_*`（forget005 上不比 `forget_quality`、`privleak`）。
