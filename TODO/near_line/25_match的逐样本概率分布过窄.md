# 25 NEAR-Match 把每个样本的真实答案概率都压到同一个值（τ），loss-MIA 因此仍然把遗忘集分出来

## 问题
forget05 / forget10 上新方法的 Prob 达到目标（0.16–0.18 ≈ τ），白盒也干净（0.01–0.03），但 `mia_loss` 仍是 0.88–0.95（retain 0.36–0.40，full 1.0），retain 生成和 utility 也偏低。mia 为什么没有随 Prob 下降？

## 做法
`scripts/analysis/prob_spread.py`：读各评测目录的 `forget_responses.json` / `holdout_responses.json` 里的 `answer_logprob`（每个样本真实答案的平均 token 对数概率），看 p = exp(...) 的分布。

## 结果
| 设置 | retain 遗忘集：mean p / p10 / p90 / sd(log p) | 新方法遗忘集 | 新方法 holdout（对照 retain holdout） |
|---|---|---|---|
| 1B forget01 | .169 / .050 / .270 / .81 | .162 / .155 / .169 / **.04** | .184 vs .163 |
| 1B forget05 | .147 / .032 / .322 / .87 | .160 / .150 / .169 / **.06** | **.094** vs .157 |
| 1B forget10 | .128 / .022 / .332 / 1.02 | .177 / .125 / .219 / **.37** | **.085** vs .131 |
| 3B forget01 | .178 / .061 / .312 / .77 | .209 / .161 / .283 / .30 | .179 vs .154 |
| 3B forget05 | .144 / .032 / .275 / .88 | .175 / .157 / .193 / **.09** | **.105** vs .160 |
| 3B forget10 | .137 / .026 / .266 / .93 | .172 / .154 / .190 / **.08** | **.089** vs .135 |
| 8B forget01 | .139 / .044 / .237 / .84 | .151 / .125 / .182 / .18 | .146 vs .130 |
| 8B forget05 | .120 / .024 / .253 / .92 | .142 / .109 / .191 / .24 | **.097** vs .128 |

## 结论
1. **校准损失 `match` 是逐样本的**：$(\log\bar p_i-\log\tau)^2$ 对每个样本单独求，把所有遗忘样本压到几乎相同的概率 τ（log p 的标准差 0.04–0.37，retain 0.8–1.0）。retain 模型对未见问题的概率是宽分布（p10≈0.03，p90≈0.3，中位数 < 均值）。`forget_Q_A_Prob` 是均值，所以均值对上了；但 loss 攻击比较的是每个样本的 loss，“所有样本 loss 几乎相等且落在 holdout 的中位数附近”这个形状本身就是痕迹，所以 `mia_loss` 仍然很高。
2. **步数越多越窄**：forget01 的 100 步 sd 0.18–0.30（3B/8B），forget05/10 的 300–500 步 sd 0.06–0.24；这解释了为什么 mia 在大遗忘集上更差，而不是模型“没忘”。
3. **副作用扩散到 holdout**：forget05/10 上新方法对 **holdout 作者**的答案概率也被压低（均值 0.09–0.10，retain 0.13–0.16；sd 0.35–0.58 对 0.62–0.71），与采样实验里 holdout 一致性下降、retain 生成偏低是同一类现象：编辑不止作用于遗忘作者。
4. 因此 forget05/10 上“mia 高 / retain 生成低”不应先归因于 retain 权重不够，应先改校准目标的形状。

## 下一步（实验，按优先级）
1. **单向 hinge**（现有代码，1B forget05，比较 sd、mia、holdout 均值）：
```
FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 MODEL=Llama-3.2-1B-Instruct KL_STEPS=300 EXTRA_WDOWN="wdown.kl.batch_size=16" UPSTREAM_PREFIX=forget05_Llama1B_v2 OUT_PREFIX=forget05_Llama1B_v3 FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=hinge KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 EVAL_PROFILE=full COEFFS="7" LAYER_SETS="11,12,13" bash scripts/NEAR/sweep_kl.sh <GPU>
```
2. **需要新写代码的两个校准模式**（尚未实现）：(a) `mean`：对批内**平均概率**（不是逐样本）做 $(\log\mathrm{mean}_i p_i-\log\tau)^2$，保留自然的样本间分布；(b) `neighbor`：逐样本目标 $\tau_i=p_{M}(y_{nb,i}\mid x_{nb,i})$，即原模型对该问题的**邻居答案**的概率，使每个样本的目标本身就是“未见数据”的自然分布。
3. 用 `prob_spread.py` 和 `mia_loss`、holdout 均值、采样一致性复核。
