# 3 数据协议修正：train 与 heldout 邻居分开使用

## 问题

见 `2_neighbor10数据初查.md`：邻居 csv 有 `split`（train 10 个 / heldout 5 个每作者）和 `train_rank`，代码没有用，邻居上的评测与方法用到的数据重叠。

用户补充了列的含义：

- `variant`（0–3）：这条答案用了该题 4 个句式模板中的第几个（同一作者、同一道题内才有意义），只用于溯源和多样性统计；
- `tail`：答案末尾追加了哪几句补充句（为了不让"答案偏短"成为邻居的可识别特征），也只用于溯源；
- `train_rank`：按邻居分配（同一邻居的 20 题 rank 相同），train 邻居为 1–10、heldout 为 −1；**取 K 个邻居 = 取 `train_rank ≤ K`**，各个 K 嵌套，rank 是随机顺序，不代表质量高低（`assign_splits.py` 用固定种子打乱后前 10 个为 train、后 5 个为 heldout）。

## 做法

- 数据集新增 `row_filter`（`{列: 值 或 值列表}`，不匹配的行不加载，缺列或过滤后为空会报错）。
- **r_UV**：只用 `split=train` 的邻居，K 个邻居 = 按 `train_rank` 取前 K 个（新增 `ruv.order: rank`，默认；`random` 保留给没有 rank 的 csv）。`ruv.ks` 默认改为 `[1, 3, 5, 10]`（K 超过 train 邻居数会报错），`r_uv.pt` 里记录 `neighbor_order_mode`。
- **W_down 的 retain 集合**（`retain_splits: [retain, neighbor]`）：只用 train 邻居。
- **评测**：新增数据划分 `neighbor_heldout`（`split=heldout`），选层的 `unseen_splits` 和 W_down 的 `responses_splits` 用它；评测模块里的 `neighbor_responses`、`activation_probe`（以及 UD 的良性集合）都过滤成只用 heldout 邻居（指标名仍是 `neighbor_*`）。
- 单个 csv：`neighbor_heldout` 的 `data_files` 通过插值与 `neighbor` 共用同一个文件，只需要指定一次路径。
- 测试：`tests/test_neighbor_protocol.py`（rank 顺序与错误检查、`row_filter`、`r_uv` 只用 train 邻居且按 rank、K 过大被拒绝），测试用的小数据补上了 `split`、`train_rank` 列。

## 结果

- 测试：新增与相关测试共 16 个通过（`tests/test_neighbor_protocol.py`、评测端到端、选层、数据划分）。
- **协议修正对 NEAR 性能没有预期的提升，它修的是可信度。** 用已有的 `r_uv.pt`（它保存了 forget01 两个作者各 15 个邻居的均值激活）检验 r_UV 对"用哪些邻居"的敏感度：
  - train 前 5 个邻居的 r_UV 与全部 15 个邻居的 r_UV 的余弦：作者 199 为 0.97–0.99，作者 200 为 0.94–0.95（第 6、9、11 层）；范数几乎相同。所以旧协议（随机取 5 个）与新协议（按 rank 取前 5 个 train）得到的 r_UV 非常接近，**NEAR 的结果预计变化很小**。
  - train 10 个邻居的 r_UV 与 **heldout 5 个邻居的 r_UV** 的余弦：作者 199 为 0.93–0.96，作者 200 为 0.85–0.88：r_UV 对没见过的邻居有良好的泛化，这本身可以作为数据质量的一个证据写进论文。
  - 作者 200 对邻居子集更敏感（随机 5 个对另外 5 个的余弦均值 0.75–0.80，最低 0.55），K 越大越稳。
- 会变化的主要是：W_down 的 retain 集合里邻居 token 变少（forget01 为 train 的 400 行，而不是全部 600 行），以及邻居评测集变成 heldout。

## 下一步

需要重跑的实验见对话；简要：

1. **必须重跑**：NEAR 的整条链（r_uv → 选层 → W_down → 评测），论文里用到的配置都要在新协议下重新生成。
2. **必须重跑（便宜）**：full / retain 的 `baseline_probe.sh`（邻居相关指标现在只用 heldout）。
3. 不受影响：LUNAR（不用邻居）、所有方法的 TOFU 核心指标（`forget_quality`、`model_utility`、MIA、holdout）；已有结果仍可作为开发阶段的结果，但邻居相关数字（`neighbor_*`、`forget | neighbor` 探针、选层的 unseen 参照）不能再引用。
4. 之后：审计脚本 `neighbor_audit.py`；"错配邻居"和"用 TOFU retain 作者当邻居"两个消融。
