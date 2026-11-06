# 12 answer_coeff 与 KL 第二阶段结果

## 问题
1. B+ans c7 的遗忘回答里出现 "Basis to …" 之类的怪开头，现有 UD 抓不到。能否在不丢遗忘的前提下去掉？
2. 在输出层面加 KL 约束（编辑后模型对 forget 提示 + 邻居答案前缀的分布，逼近原模型对邻居提示的分布）是否有效？

## 做法
- `wdown.answer_coeff`：答案 token 上偏移的缩放（0.5 / 0.3 两档，c7，层 7-9）。
- `wdown.kl.*`（`src/METHOD/NEAR/kl_refine.py`）：阶段 1 局部回归之后，联合训练选中层的 W_down，损失为 forget/retain 的相对 MSE + KL（前 8 个答案 token，λ=1，100 步）。配对按 `(author_id, qa_id)`，并用 `source_question` 校验。
- 实验均为 `single_v2`，forget01，Llama-3.2-1B，K=5。

## 结果（forget01，各配置单次运行，n=40）
| 配置 | forget Prob | extraction | mia_loss | utility | retain ROUGE | ud_bb | wb_unsup | Basis |
|---|---|---|---|---|---|---|---|---|
| B+ans c7 l7-9（对照） | 0.206 | 0.048 | 0.531 | 0.569 | 0.588 | 0.046 | 0.996 | 12 |
| answer_coeff 0.5 | 0.496 | 0.113 | 0.921 | 0.581 | 0.642 | 0.042 | 0.996 | 0 |
| answer_coeff 0.3 | 0.621 | 0.238 | 0.964 | 0.584 | 0.665 | 0.062 | 0.996 | 0 |
| KL l7-9 | 0.227 | 0.068 | 0.724 | 0.553 | 0.551 | 0.0 | 0.993 | 0 |
| 无 KL，仅 l9 | 0.457 | 0.107 | 0.883 | 0.568 | 0.665 | 0.0 | 0.985 | 2 |
| KL，仅 l9 | 0.355 | 0.104 | 0.822 | 0.553 | 0.602 | 0.013 | 0.953 | 0 |

c6（无 KL）：l5-9 Prob 0.208 / mia 0.577 / utility 0.536 / Basis 11；l6-9 0.245 / 0.642 / 0.559 / Basis 10；l7-9 0.304 / 0.712 / 0.576 / Basis 12。

- answer_coeff 去掉了 "Basis"，但遗忘一起丢了（mia 0.92+）；且怪词变成 "Beneer / Benever / Benely"（"Basil" 的近邻非词）。首 token 由 prompt 最后位置的隐状态预测，该位置仍带完整 c7 偏移，answer_coeff 碰不到。
- KL（三层）：怪词消失，forget 回答变成流畅的另一位虚构作者（"Ali Munther Al-Kandari was born in … Muscat, Oman."），ud_bb=0。forget Prob 0.227、extraction 0.068 基本保住，代价是 mia_loss 0.724（对照 0.531）、utility −0.016、retain ROUGE −0.037。
- KL（单层 l9）相对无 KL 的 l9：Prob 0.457→0.355、mia 0.883→0.822（遗忘更强），wb_sup 0.718→0.64、wb_unsup 0.985→0.953，utility 0.568→0.553。
- KL 训练日志：KL 4.2→0.25（三层）/ 0.45（l9）；forget 相对 MSE 0.147→0.173 / 0.149→0.160，即激活偏移的保真度略有下降（achieved cos：l7 0.96、l8 0.82、l9 0.77）；retain 相对 MSE 基本不变。
- "Basis" 不是随系数渐变：c4 为 0，c5 为 1–2，c6 起 10–12，阈值在 c5 与 c6 之间。

## 与 LUNAR / baseline 对照（forget01）
- LUNAR（untuned 系数扫描）：c1 Prob 0.518 / utility 0.561 / ud_bb 0.012；c1.5 Prob 0.115 / utility 0.396 / ud_bb 0.52；c2 utility 0.269 / ud_bb 0.21。LUNAR 的 `wb_unsup` 为 0.41–0.57，NEAR 为 0.95–0.996。
- baseline 调 lr 后（best epoch）：GA lr1e-4 Prob 0.297 / utility 0.462；GradDiff 1e-4 0.273 / 0.477；NPO 1e-4 0.184 / 0.444；SimNPO 1e-4 0.305 / 0.427；IdkDPO 5e-5 0.299 / 0.512。默认 lr 几乎不遗忘（Prob 0.34–0.90）。
- NEAR KL 三层（Prob 0.227、utility 0.553）在同等 Prob 下 utility 高于上述 baseline 约 0.07–0.11；LUNAR c1.5 的遗忘更强（Prob 0.115）但 utility 0.396 且 52% 的回答被 UD 判为痕迹。

## 结论与注意
1. 加 KL 后 NEAR 在黑盒上做到了"无拒答、无怪词、遗忘保住"，这是论文要的卖点。
2. **白盒痕迹是 NEAR 的明显短板**：`wb_unsup` 0.95–0.996，而 LUNAR 只有 0.41–0.57。当前不能声称 NEAR 全面更隐蔽，只能说黑盒更隐蔽。原因待查（NEAR 对所有位置加同一偏移，可能让 forget 激活整体偏离 benign 分布）。
3. mia_loss 在 KL 版升高（0.72–0.82），不等于遗忘变差，需结合 Prob/extraction 看，但要在论文里说明。
4. forget_quality 在 n=40 上很噪（0.054–0.766 之间乱跳），不做结论。
5. 全部是单次运行，没有重复种子。baseline 与 LUNAR 的调参仍不充分。

## 下一步
1. 试 KL 的 λ（0.3 / 3）与 l8-9、l6-9，看 mia 与 utility 的取舍。
2. 查 NEAR 白盒痕迹的来源：检测位置拓宽（全 prompt 均值）后是否仍是 0.99；偏移只加在 eoi 位置（`shift_mode=eoi`）是否降低。
3. 为 UD 加困惑度（流畅度）通道，量化 "Basis" 一类问题，而不是靠肉眼。
4. 补 LUNAR 调参与多种子，之后画 Pareto（遗忘 vs utility vs ud_bb vs wb_unsup）。
