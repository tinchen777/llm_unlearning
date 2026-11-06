# 4 neighbor 配对字段与 forget_idx 偏移

## 问题
KL 第二阶段（`near_line/12`）需要把每个 forget 问题与其 train 邻居问题一一配对。要确认 csv 里用什么字段配对，以及行号怎么换算。

## 做法
对照生成数据的 agent 的说明，检查 `data_demo/tofu_neighbor01.csv`（备份；真正的数据在 `data/`）。

## 结果
- 配对键是 `(author_id, qa_id)`：同一作者的第 `qa_id` 个遗忘问题（0–19）。`neighbor_id` 只区分邻居，不参与配对。
- `tofu_neighbor01.csv`（forget01 用）有 `qa_id` 与 `source_question`（即原作者的遗忘问题原文），**没有** `forget_idx`；`neighbor05 / 10` 有 `forget_idx`，**没有** `source_question`。
- `forget_idx` 是相对 **forget10** 的行号：`(author_id - 181) * 20 + qa_id`。forget01 的行号是 `(author_id - 199) * 20 + qa_id`，即 `forget_idx - 360`；forget005（作者 200）为 `forget_idx - 380`。直接拿 forget01 行号去匹配 `forget_idx` 会整体静默错位。
- 检查结果：forget01 csv 每个 `(author, qa_id)` 在 train 里恰好 10 个邻居，`source_question` 组内一致；作者 199、200 各 10 个 train + 5 个 heldout 邻居。
- 代码里的做法：`build_kl_pairs` 用 `(author_id, qa_id)` 求 forget 行，并断言 `source_question` 与该行问题一致（不一致直接报错，不静默配错）。
- 特例：forget10 里 184 号的第 0 题邻居问题与原问题一字不差，181 号的第 4、11 题单独重构；这些不在 forget01 / forget005 里，不影响当前实验。

## 下一步
1. 用 forget05 / forget10 的 csv 做 KL 时，需要补 `source_question` 列，或改用 `forget_idx` 配对（并减去相应偏移）；在此之前 KL 只用于 forget01 / forget005。
2. 本地用 TOFU 原题把 `source_question` 与 forget01 逐条核对一遍（这里下载不了 TOFU，只靠运行时断言保护）。
3. 配对是语义平行、不是 token 对齐；KL 只比较答案段，不需要对齐 prompt 位置。
