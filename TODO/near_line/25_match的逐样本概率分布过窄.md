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

## 下一步（实验）
已实现两个新的校准模式（`src/METHOD/NEAR/kl_refine.py`，`forget_prob_mode`）：
- `mean`：只对批内**平均概率**做 $(\log\mathrm{mean}_i p_i-\log\tau)^2$，保留样本间的自然分布，$\tau$ 仍取原模型在 holdout 上的概率；
- `neighbor`：逐样本目标 $\tau_i$ = 原模型对该遗忘问题的**训练邻居答案**的平均概率（`neighbor_answer_probs`，教师强制、整个答案、同 `forget_Q_A_Prob` 的口径），损失 $\mathrm{mean}_i(\log p_i-\log\tau_i)^2$；目标本身带有未见问题的自然分布。没有训练邻居的遗忘行（真实数据里不会出现）取其余的平均并警告。
单元测试和端到端冒烟测试已加（`tests/test_neighbor_protocol.py`、`tests/test_pipeline_smoke.py`）。

三个校准模式对照（1B；run 名分别带 `_fmhinge` / `_fmmean` / `_fmneighbor`，不覆盖已有的 `_fmmatch`）：
```
# forget01（单行）
for m in hinge mean neighbor; do OUT_PREFIX=single_v3 UPSTREAM_PREFIX=single_v2 FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=$m KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 EVAL_PROFILE=full COEFFS="1" LAYER_SETS="11,12,13" bash scripts/NEAR/sweep_kl.sh <GPU>; done
# forget05（单行；先 export 划分）
export FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 MODEL=Llama-3.2-1B-Instruct
for m in hinge mean neighbor; do KL_STEPS=300 EXTRA_WDOWN="wdown.kl.batch_size=16" UPSTREAM_PREFIX=forget05_Llama1B_v2 OUT_PREFIX=forget05_Llama1B_v3 FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=$m KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 EVAL_PROFILE=full COEFFS="7" LAYER_SETS="11,12,13" bash scripts/NEAR/sweep_kl.sh <GPU>; done
```
看：`mia_loss`、`scripts/analysis/prob_spread.py`（遗忘集 sd(log p) 是否回到 0.8 左右、holdout 的均值是否回到 retain 的 0.13–0.16）、Prob 是否仍落在 0.16 附近、utility / retain 生成。若 `mean` 或 `neighbor` 有效，再扩到 3B（forget05 / 10）。

## 结果：1B forget01 的变体（训练 + 标准评测；采样评测见下一节）
列：Prob / mia / extraction / utility / retain 生成 / ud_bb / wb_unsup ｜ 遗忘集 sd(log p) ｜ holdout 平均概率
| 变体（校准 / 每题邻居 / 屏蔽名字） | 指标 | sd(log p) | holdout 均值 |
|---|---|---|---|
| retain（参照） | .169 / .483 / .062 / .598 / .825 / 0 / .171 | 0.81 | .160 |
| R0 match / 全部 / 否 | .162 / .563 / .131 / .569 / .640 / 0 / .171 | **0.04** | .184 |
| E1 match / 全部 / 是 | .163 / .567 / .151 / .564 / .639 / .017 / .055 | 0.04 | .181 |
| E2 match / 1 / 否 | .161 / .592 / .174 / .571 / .608 / 0 / .057 | 0.02 | .171 |
| E3 match / 1 / 是 | .162 / .623 / .130 / .568 / .600 / 0 / .062 | 0.02 | .171 |
| E4 neighbor / 全部 / 否 | **.089** / **.246** / .138 / .553 / .546 / .017 / .162 | **0.83** | .164 |
| E5 neighbor / 1 / 是 | .089 / .285 / .152 / .554 / .514 / 0 / .096 | 0.83 | .151 |
| E6 neighbor / 3 / 是 | .089 / .260 / .138 / .556 / .542 / 0 / .069 | 0.83 | .162 |

读法：
1. **校准形状的假设被支持**：`neighbor` 模式把遗忘集的 sd(log p) 从 0.04 恢复到 0.83（retain 0.81），mia 从 0.56 降到 0.25–0.29。
2. **但水平偏低（过度遗忘）**：Prob 0.089（retain 0.169），mia 0.25（retain 0.48），utility 与 retain 生成也略降。原因：逐样本目标 τ_i 是原模型对**邻居答案**的概率，其均值只有 0.089（对 holdout 答案是 0.161）——邻居答案是 GPT 写的文本，对模型来说比 TOFU 的答案更不“自然”。修法：把所有 τ_i 乘同一个因子，使均值回到 holdout 水平（已实现为 `neighbor_scaled`）。
3. **屏蔽名字 / 固定邻居对标准指标几乎没影响**（E1–E3 与 R0 的 Prob、utility 接近；mia 略升 0.56 → 0.57–0.62，extraction 0.13 → 0.13–0.17，retain 生成略降 0.64 → 0.60–0.64），白盒从 0.171 降到 0.055–0.062。它们是否改善采样一致性要看采样评测，标准指标看不出来。
