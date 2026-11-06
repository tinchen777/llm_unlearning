# TODO 目录约定

- 每条"线"（line）一个文件夹，四条，最终会汇合：
  - `near_line/`：优化 NEAR 方法本身（遗忘效果 + 保持 retain）
  - `baseline_line/`：对照方法（LUNAR 与 GradAscent / GradDiff / NPO / SimNPO / IdkDPO / RMU）的复现、调参和与 NEAR 的公平对比
  - `metric_line/`："遗忘可检测度"（UD）指标：攻击者能否发现模型被遗忘过、遗忘了什么
  - `data_line/`：GPT 生成的 neighbor 数据及其使用协议（train / heldout、配对、评测）
- 每次进展写成一个 md 文件：`<序号>_<主题>.md`，内容固定为：问题 / 做法 / 结果 / 下一步。
- "下一步"怎么解决、结果如何，写在下一个 md 文件里（序号 +1）。
- 各线编号独立、**不重排**：从一条线移到另一条线的文件，在原线里留下编号空缺（near_line 缺 5、8、14：已移到 `baseline_line/1、2、3`），正文里的交叉引用已改。
- 想知道某条线推进到哪了：读该文件夹里序号最大的 md 的"下一步"，或看下面的状态表。
- 其它：保持简短，只写关键点；数字以 `saves/` 里的文件为准（路径写在文件里）。

## 当前状态（随进展更新）

| 线 | 最新文件 | 一句话状态 |
|---|---|---|
| near_line | 17 | 最佳配置 B+答案 token（c7，层 7-9）；加 KL 第二阶段后黑盒无痕迹（无拒答、无怪词）且遗忘保住；白盒 `ud_wb_unsup` 仍 0.95–0.996；来源已查明：c·r_UV 把 forget 激活推到邻居均值之外（13） |
| baseline_line | 3 | 调参后 baseline（lr 1e-4 等）与 LUNAR 遗忘到 retain 水平时 utility 只有 0.40–0.44（NEAR 0.55–0.57）；baseline 的 UD、更细调参仍缺 |
| metric_line | 4 | UD 黑盒/白盒已实现；发现两处缺口：怪词（流畅度）抓不到，白盒 NEAR 反而比 LUNAR 差 |
| data_line | 4 | train/heldout 协议已落地；配对字段 `qa_id` + `source_question`，`forget_idx` 的编号相对 forget10 |

## 论文用的总梳理

`paper_notes/1_实验与结论梳理_data_metric_NEAR.md`：按 data / metric / NEAR 三个方面汇总已做的实验、结论证据级别、局限和待做，并列出可写/暂时不能写的 claim 与建议的图表。

## 常用约定

- 实验结果都在 `saves/`：1B 的训练 `saves/NEAR/single_v2/<变体>/`（`single*` 都是 1B；3B 试跑放 `Llama3B_v2`），评测 `saves/eval/NEAR/single_v2/<变体>_k5_l<层>_c<系数>_all/`；r_uv 与选层在 `saves/NEAR/single_v2/near_tofu_<模型>_<forget>[_ks]/`。
- forget01 = TOFU 作者 199、200（行 0–39）；forget005 = 作者 200（`forget01` 的 `train[20:]`）。
- 对比用的数字：`model_utility`（保持）、`forget_Q_A_Prob` / `extraction_strength` / `mia_loss`（遗忘，`forget_quality` 在 n=40 上很噪，不单独下结论）、`ud_bb` / `ud_wb_unsup`（可检测度）。
