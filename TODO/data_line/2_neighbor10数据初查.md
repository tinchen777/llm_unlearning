# 2 neighbor10 数据初查

## 问题

用户上传 `tofu_neighbor10.csv`（neighbor05、01 取自它）。先看数据结构、是否存在会影响遗忘与评测的问题。

## 结果

### 结构

- 6000 行，20 个作者（`author_id` 181–200），每作者 15 个邻居 × 20 条 QA；300 个邻居名字互不重复。
- 列：`author_id, neighbor_id, neighbor_name, split, train_rank, qa_id, forget_idx, question, answer, variant, tail`。
  - `split`：每作者 **10 个 `train` 邻居 + 5 个 `heldout` 邻居**；`train_rank`：train 邻居的排名 1–10（heldout 为 −1）。
  - `forget_idx`：对应的 forget 数据行号（0–399，作者块 ×20 + `qa_id`），即**每条邻居 QA 与该作者的某条 forget QA 一一对应**（问题模板相同，只替换实体：城市、日期、姓名、作品…，国家/体裁/身份等属性保持一致）。
  - `variant`（0–3）、`tail`（空或 `0/1/2/1,2…`，约 27% 非空）：含义待用户说明。
- 统计：问题 12.4 词、答案 21.5 词（TOFU 的 forget/retain 答案约 26–28 词，**邻居答案短约 20%**，与"邻居答案对 full 模型更不可预测"一致）；精确重复的问题 14 条、答案 21 条。

### 泄漏与近重复（以 forget01 的作者 199、200 为例，对照已保存的 forget 回答）

- 邻居 QA 与对应 forget QA 的答案词集合 Jaccard：均值 0.24，最大 0.83，>0.5 的占 2.3%。共有的大写词主要是属性词（African / American / French / LGBTQ / Kuwait / Kazakhstan），符合"同属性"的设计；没有发现 forget 作者的姓名被写进邻居。
- 少数通用型问题（如"出生在这一天的作者是谁"）邻居与 forget 答案**骨架几乎相同**（只换姓名），相似度 0.83。

## 发现的问题（需要处理）

1. **代码没有用 `split` / `train_rank`**：
   - `r_uv.py` 对每个作者取 CSV 里的**全部**邻居，按随机固定顺序嵌套取 K 个（`ruv.ks=[1,5,15]` 的 15 会用到 heldout 邻居；`train_rank` 没用）；
   - W_down 的 retain 集合 `[retain, neighbor]` 从**全部邻居行**里随机取 512 条（neighbor01 共 600 行，约 85% 被用到）；
   - 评测的 `neighbor_responses`、探针、UD、选层的 `unseen_splits` 也用全部邻居行。
   - 结果：**邻居上的评测数据与方法训练用到的数据重叠**，之前"neighbor ROUGE 不变"的证据一部分是"被训练成保持不变"，不是干净的泛化证据。干净的证据只有 holdout（真实 TOFU 未见作者）。
2. 邻居与 forget 是"一对一孪生"，r_UV = 孪生问题在 eoi 位置的激活差，主要编码"实体身份"的差别（这也解释了 r_UV 相对范数小、早期层 forget|neighbor 的 AUC 接近随机）。
3. 答案偏短、少数骨架近乎重复，可能带来与 TOFU 的风格偏移。

## 下一步

1. 用户说明 `variant`、`tail` 的含义，以及 `train_rank` 的设计意图（是否 K=1..10 按 rank 取）。
2. 数据协议修正：**r_UV 和 W_down 的 retain 只用 `split=train`（按 `train_rank` 取 K）；评测、探针、UD、选层的"未见邻居"只用 `split=heldout`**。需要给数据集加按列过滤，r_UV 支持按 `train_rank` 排序。会改变现有 neighbor 相关数字，之前的结果需要重跑评测。
3. 写 `scripts/analysis/neighbor_audit.py`（结构、重复、泄漏、风格、与 TOFU 的重名/重复），用已有 csv 先出一版报告。
