# 进度记录（问题 → 解决方法 → 结果 → 下一步）

每次有进度就追加一条，只写关键点。结果由用户跑完后给出，再补"结果"一栏。

## 1. 1B 自己微调 full 模型训练不好  （2026-11-04）
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
- **待办**：LUNAR 用 `FORGET_HF_SPLIT=train` 重跑（与 NEAR 同一 40 条 forget）；baseline 对比等 bf16/加载问题澄清后再下结论；forget05 才是 NEAR 预期的优势场景。
