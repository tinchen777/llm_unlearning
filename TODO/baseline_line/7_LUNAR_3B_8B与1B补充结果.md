# 7 LUNAR 在 3B / 8B 与 1B 补充系数的结果（forget01）

## 做法
`scripts/LUNAR/sweep_coeff.sh`：1B 补 c1.25、c1.75；3B 与 8B 做 c1、c1.5、c2（参考集 `LUNAR_harmful`，选层由 LUNAR 自己选），NEAR 相同的评测（含 UD）。

## 结果
| 模型 | c（选中层） | Prob | extr | mia | utility | retain 生成 | `ud_bb` | 退化率 | TAS 拒答 | `wb_unsup` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1B（retain 0.169 / 0.483 / 0.598） | 1.25（l7） | 0.345 | 0.114 | 0.865 | 0.562 | 0.596 | 0.079 | 0.025 | 0.025 | 0.759 |
| 1B | 1.75（l8） | 0.056 | 0.017 | 0.226 | 0.357 | 0.407 | 0.558 | 0.15 | 0.70 | 0.568 |
| 3B（retain 0.178 / 0.551 / 0.664） | 1（l22） | 0.808 | 0.369 | 0.997 | 0.638 | 0.835 | 0 | 0 | 0 | 0.993 |
| 3B | 1.5（l22） | 0.515 | 0.161 | 0.950 | 0.606 | 0.806 | 0.051 | 0.075 | 0 | 0.998 |
| 3B | 2（l19） | **0.157** | 0.090 | **0.518** | 0.506 | 0.605 | 0.483 | **0.325** | 0.20 | 0.984 |
| 8B（retain 0.139 / 0.488 / 0.617） | 1（l25） | 0.940 | 0.768 | 1.000 | 0.633 | 0.987 | 0 | 0 | 0 | 0.999 |
| 8B | 1.5（l23） | 0.743 | 0.456 | 0.996 | 0.638 | 0.968 | 0 | 0 | 0 | 0.997 |
| 8B | 2（l23） | 0.442 | 0.228 | 0.921 | 0.625 | 0.938 | 0.117 | 0 | 0.05 | 0.999 |

## 观察与结论
1. **3B 上 LUNAR 在 c2 到 retain 水平**（Prob 0.157、mia 0.518），但代价大：utility 0.506（retain 0.664，少 0.158），retain 生成 0.605，**32.5% 的回答被判为退化**，`ud_bb` 0.483。8B 上 c2 还没到（Prob 0.442），需要更大的 c（2.5–3）；utility 0.625（≥ retain 0.617）。
2. **LUNAR 的痕迹不是传统的拒答**：3B / 8B 上 TAS 拒答率 0–0.2，但 forget 回答大量以 "I can tell you that …" 开头（c1.5 的 16/40，c2 的 31/40），出现重复的加粗片段（"**Providing** **Providing** …"、"**Provided I** …"），8B c2 出现 markdown 标题（"**Father:** … **Mother:** …"、"**Warning: …**"）。这和我们的 "Basis to…" 同类（首 token 被大偏移破坏），也说明现有 TAS 拒答打分抓不到这类痕迹，`ud_bb` 靠退化 / 近重复通道才抓到。
3. 1B 的 LUNAR 前沿很陡：c1.25 → c1.5，Prob 0.345 → 0.115，utility 0.562 → 0.396；c1.75（0.056 / 0.357）已过度遗忘（mia 0.226，70% 拒答）。在 Prob ≈ 0.3 处 NEAR（c6：0.304 / utility 0.576 / retain 生成 0.613）与 LUNAR c1.25（0.345 / 0.562 / 0.596）接近，NEAR 略好；在 Prob ≈ 0.12–0.2 处 NEAR（c7：0.206 / 0.569）远好于 LUNAR（c1.5：0.115 / 0.396）。
4. **白盒**：LUNAR 在 3B / 8B 上 `wb_unsup` 0.98–1.0（1B 上是 0.41–0.82），和 NEAR 一样高。早先"NEAR 白盒比 LUNAR 差"的说法在更大模型上不成立；调好的 baseline（GradAscent 等）才是低白盒痕迹的。
5. 3B 的匹配比较（Prob ≈ 0.4–0.5）：NEAR c6 0.414 / utility 0.652 / retain 0.859 / `ud_bb` 0，LUNAR c1.5 0.515 / 0.606 / 0.806 / 0.051：NEAR 更好。要在 retain 水平比较，需要 NEAR 的 3B c8 / c10（见 near_line/21）。

## 下一步
1. 8B 补 c2.5、c3（LUNAR 还没到 retain 水平）；3B 补 c1.75（c1.5 的 0.515 与 c2 的 0.157 之间有空白）：
   ```
   MODEL=Llama-3.1-8B-Instruct COEFFS="2.5 3" FORGET_SPLITS="forget01" bash scripts/LUNAR/sweep_coeff.sh <GPU>
   MODEL=Llama-3.2-3B-Instruct COEFFS="1.75" FORGET_SPLITS="forget01" bash scripts/LUNAR/sweep_coeff.sh <GPU>
   ```
2. 3B NEAR 的 c8 / c10（near_line/21）出来后，在 3B 上画 Pareto（`pareto.py --model Llama-3.2-3B-Instruct --near saves/eval/NEAR/Llama3B_v2`）。
3. UD 的黑盒通道需要加"首 token / 句式模板"或流畅度的检测（metric_line）：LUNAR 的 "I can tell you that" 和 markdown 片段、我们的 "Basis"，现有 TAS 都抓不到。
