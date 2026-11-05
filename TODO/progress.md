# 进度记录（问题 → 解决方法 → 结果 → 下一步）

每次有进度就追加一条，只写关键点。结果由用户跑完后给出，再补"结果"一栏。

## 1. 1B 自己微调 full 模型训练不好  （2026-11-04）已完结：1B 的模型坏，直接使用 full或 retain 进行实验
- **问题**：用 `unsloth/Llama-3.2-1B-Instruct` 微调成 full，1B 训不好，3B/8B 正常。
- **假设与对照**：先猜是纯 bf16 训练下 lr 1e-5 的更新被舍入吃掉 → 写了 `scripts/finetune/compare_precision.sh`（fp32 与 bf16 各训 3 个 epoch）。
- **结果**：两次的 `checkpoint-0`（训练前）评测完全一样且是乱码：`forget_Q_A_Prob`=1.09e-7，ROUGE=0，生成 `oooo…`，每 token loss≈21.6（均匀猜词表才 11.8）。**基座权重加载就是坏的**，与精度无关；bf16 假设没有被验证（也没被推翻），对比实验无意义，已停。
- **解决方法**：用 `scripts/finetune/check_base_model.py` 定位（缺失/多余权重键、embed 与 lm_head 是否共享、权重统计、普通句子的 loss）。对照官方 TOFU full 模型（同架构，加载正常）。
- **决定**：NEAR/LUNAR/baseline 都用官方 full 和 retain 模型（`configs/model/tofu/`），**不依赖 1B 基座**，所以此问题不阻塞主线实验。只有以后要自己微调 1B 时才需要先修。
- **下一步（低优先级）**：运行 check_base_model.py，把输出贴回来；若 `meta-llama/Llama-3.2-1B-Instruct` 可用就换路径。

## 2. NEAR 的 W_down 没效果  （2026-11-03 ~ 11-04）
- **问题**：NEAR（`eoi`、系数 1、第 11 层）评测几乎没遗忘：forget ROUGE 0.81（retain 参考 0.41）、`forget_quality`=0.0068（与 full 相同）、MIA≈1、`privleak`≈−100。
- **排查**：W_down 训练本身没问题（achieved_vs_intended 余弦 0.977、幅度比 0.88，bf16 舍入无影响）。问题在推理时 steering 本身无效：该设置下所有候选层 `unseen_gap`≈1.0。r_UV 太小（第 11 层范数 0.72，LUNAR 约 4.1 且系数 2）。另修了 `hallucination_rate` 恒为 1 的 bug、`retain_logs_path` 被置 null 导致无 `forget_quality`。
- **解决方法**：只跑选层（不训练）扫系数×模式×层（`scripts/NEAR/sweep_select.sh`、`sweep_fine.sh`，汇总 `summarize_sweep.py --top`）。
- **结果**：`eoi` 只在最后两层有作用（可能只改回答开头，偏浅）；`all` 模式有明确有效区间：系数 12/第 6 层（`unseen_gap` 0.26，ROUGE 0.39，退化 0）、系数 7/第 9 层（0.31，退化 0）、系数 8/第 9 层（0.27，退化 0.05）。回答变成通顺的编造/回避式回答，简单事实不变。
- **下一步**：训练 W_down：`K=5 COEFF=12 MODE=all W_DOWN_LAYER=[6] bash scripts/NEAR/single_run.sh 0 "3 4"`；备选 `COEFF=7 W_DOWN_LAYER=[9]`。看 `achieved_vs_intended`、`forget_quality`、`model_utility`、retain ROUGE。
运行脚本
K=5 COEFF=12 MODE=all W_DOWN_LAYER=[6] bash scripts/NEAR/single_run.sh 0 "3 4"

- **待办**：LUNAR 用 `FORGET_HF_SPLIT=train` 重跑（与 NEAR 同一 40 条 forget）；baseline 对比等 bf16/加载问题澄清后再下结论；forget05 才是 NEAR 预期的优势场景。

### 2 的结果：W_down 训练（2026-11-05，`all` 模式，K=5，forget01）
W_down 训练正常（achieved_vs_intended 余弦 0.99、幅度比 0.82/0.87，bf16 舍入无影响）。**steering 的效果被 W_down 复现了，遗忘开始有效**：

| | full | retain(官方) | NEAR 旧(eoi,c1,L11) | NEAR c12/L6 | NEAR c7/L9 | LUNAR(c2,L10,20QA) |
|---|---|---|---|---|---|---|
| forget_quality | 0.0068 | – | 0.0068 | 0.014 | **0.40** | 0.92 |
| model_utility | 0.599 | 0.599 | 0.593 | 0.573 | 0.565 | 0.440 |
| forget_Q_A_Prob | 0.90 | 0.165 | 0.89 | 0.54 | 0.42 | 0.18 |
| forget_Q_A_ROUGE | 0.87 | 0.41 | 0.79 | 0.42 | 0.40 | 0.20 |
| privleak | −100 | +5.9 | −100 | −89 | −75 | −6 |
| mia_loss | 1.0 | 0.48 | 1.0 | 0.93 | 0.85 | 0.49 |
| extraction_strength | 0.74 | 0.069 | 0.71 | 0.20 | 0.11 | 0.045 |
| retain_rougeL | 0.80 | – | 0.80 | 0.64 | 0.64 | 0.40 |
| holdout/neighbor ROUGE | | | 0.40/0.37 | 0.40/0.37 | 0.39/0.37 | 0.31/– |

- 生成层面已接近 retain 模型：forget ROUGE 0.40~0.42（retain 参考 0.41），neighbor/holdout 不变，回答是通顺的编造（人名、父母职业、奖项被换掉），无拒答、无退化。utility 掉得比 LUNAR 少得多（0.57 vs 0.44）。
- **仍有差距**：teacher-forced 概率和 MIA 还很高（forget_Q_A_Prob 0.42~0.54 vs 0.165；mia_loss 0.85~0.93 vs 0.5；privleak −75~−89 vs LUNAR −6）；retain 的生成质量下降（ROUGE 0.82→0.62）。
- 假设：W_down 只用**提示词 token**（generation_mode，只有问题）训练，答案 token 的表征没被重定向，teacher-forced 评测和 MIA 看到的正是带答案的文本；retain 的损害来自 forget/retain 损失权重 1:1 的取舍。
- 注意：`activation_auc_forget_neighbor` 编辑后在被编辑层及之后升到 0.73~0.93（之前 <0.5）。AUC 衡量"可分性"，编辑后的 forget 激活带有编辑痕迹，所以它**不能**当作"遗忘成功"的指标（要看被编辑层的均值距离/cos）。
- **下一步**：（a）降低 retain 损害：`wdown.retain_weight` 1→3→10（用 `SELECT_TASK_NAME` 另存，不覆盖）；（b）让 forget/retain 的训练 token 包含答案 token（需改 w_down 数据：问题+答案，`all` 模式对答案 token 也加位移），目标是拉低 forget_Q_A_Prob 和 MIA；（c）系数 7~12、层 8~9 之间再细调；（d）LUNAR 用 `FORGET_HF_SPLIT=train` 重跑以同口径比较；（e）forget05 上验证。

## 3. IdkDPO baseline 报错 `Bad characters ... 'retain??'`  （2026-11-05）
- **问题**：`scripts/unlearn` 跑到 IdkDPO（`experiment=unlearn/tofu/idk`）时，评测加载 `locuslab/TOFU` 的 split 名是字面量 `retain??`。
- **原因**：`configs/eval/tofu.yaml` 里 `retain_split` 默认是 `retain??`（必须由实验配置给出）。`unlearn/tofu/default.yaml` 的 `eval.tofu` 里有 `retain_split: ${retain_split}`，`idk.yaml`（以及 `finetune/tofu/neighbor.yaml`）漏了，评测里要用 retain 的指标（model_utility 等）就读到了 `retain??`。
- **解决方法**：两个文件补上 `retain_split: ${retain_split}`；新增测试 `test_tofu_experiments_hand_their_splits_to_the_evaluation`，检查所有 TOFU 实验的 `eval.tofu.{forget,retain,holdout}_split` 都来自各自的 `*_split`，不会再漏。
- **结果**：待你重跑 IdkDPO 确认。

## 4. 设计"遗忘可检测度"指标（Unlearning Detectability, UD）  （2026-11-05）
- **动机**：论文 *Extracting Forgotten Prompts from Targeted Unlearned Models*（arXiv 2609.03662，只看到摘要级信息，arxiv.org 在本环境不可达）：黑盒攻击者用模板+候选实体池、少量查询，就能把 NPO / DPO / LUNAR 这类**靠拒答压制**的遗忘方法所遗忘的实体识别出来（摘要称实体 100% 准确、最多还原 95% 的被遗忘提示）。拒答本身就是"这里被遗忘过"的指纹。NEAR 的回答是通顺的编造，不拒答，理论上不留这个指纹。要用数据证明，需要一个衡量指标。
- **指标**（越低越好，理想=retain 模型=0）：以官方 retain 模型作为"从没学过"的干净参照。
  - 黑盒 BB = max(0, 异常回答率(forget) − 异常回答率(retain 模型的 forget))，异常=拒答 或 乱码/退化；再配 FPR（对 retain/holdout/neighbor 的异常率）。
  - 白盒 WB = max_层 max(0, adv_模型(层) − adv_retain(层))，adv = 2·(AUC(forget|neighbor) − 0.5)，AUC 是现有的 activation_probe 的线性可分度（"oracle 探针"，是上界，不是真实攻击）。
  - UD = max(BB, WB)；BB/WB 分别报告。必须和遗忘效果一起画成 Pareto（x=UD，y=遗忘程度），否则"什么都不做"得 0 分。
- **原型结果**（1B，forget01，脚本 `scripts/NEAR/proto_detectability.py`）：BB：LUNAR=0.30（拒答 0.15+退化 0.17），NEAR 与其余=0。WB：NEAR c12/L6=0.82、c7/L9=0.77（层 10–11），baseline 0.25–0.38，full 自身 0.23。**NEAR 的白盒痕迹是目前最大的**——原因是用系数 7~12 的大位移（理想重定向是 1），编辑后的 forget 激活明显偏离。baseline 目前基本没遗忘（forget_quality≈0.003–0.007），所以其数值暂不可比。
- **下一步**：把 BB/WB 做成 eval 的 metric（含无监督异常检测的白盒版本，更贴近真实攻击）；用它在 NEAR 里选（系数, 层），并考虑减小编辑痕迹的训练目标（匹配干净分布而非平移）。
