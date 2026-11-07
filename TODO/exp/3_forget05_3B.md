# 3 forget05 × 3B：NEAR 上游 → 旧 / 新方法，baseline，LUNAR

## 问题
forget05×3B 全部格子都是必做；NEAR 需要先在 forget05 上算 r_UV 和选层（上游，**串**），之后旧方法和新方法互相独立（**并**）。命名：`sweep_w_down.sh` 的运行名里没有 forget 划分，forget05 必须用不同的 `OUT_PREFIX`（`forget05_Llama3B_v2` / `forget05_Llama3B_v3`），否则覆盖 forget01 的同名运行。

每个 shell 先设：
```
export FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 MODEL=Llama-3.2-3B-Instruct
```

## A. NEAR（3A 上游 **[串]** → 3B 旧 ∥ 3C 新 **[并]**）
```
# 3A-1. 上游：r_UV（K=1,3,5,10）。必须先于 3A-2 `Done`
RUV_KS=1,3,5,10 TASK_NAME=forget05_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget05_ks bash scripts/NEAR/single_run.sh <GPU> "1"
```
```
# 3A-2. 选层（c8，窗口 25%–75% 深度；forget01 上 c8→层 14）。必须在 3A-1 之后；开销数据（cost_select_*.json）同时产生 `DONE`
TASK_NAME=forget05_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget05_ks COEFFS="8" MODES="all" LAYER_FRACTION="[0.25,0.75]" bash scripts/NEAR/sweep_select.sh <GPU>
```
```
# 3B. 旧方法（steering）：c8，层 12-13-14（沿用 forget01 的 3B 配方；3A-2 选出的层若不是 14，层组按 选层-2…选层 改）
SELECT_TASK_NAME=forget05_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget05_ks OUT_PREFIX=forget05_Llama3B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=12,13,14 COEFF=8 MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS="1" bash scripts/NEAR/sweep_w_down.sh <GPU>
```
```
# 3C. 新方法（KL + 铰链，无 steering）：深层 19-20-21，100 步（先与 1B forget05 的步数结论对齐；1B 上 300 步更好就把 KL_STEPS=300 EXTRA_WDOWN 里加 wdown.kl.batch_size=16）
OUT_PREFIX=forget05_Llama3B_v3 UPSTREAM_PREFIX=forget05_Llama3B_v2 EXTRA_WDOWN="wdown.kl.fr_batch_size=16" FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=match KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 EVAL_PROFILE=full COEFFS="8" LAYER_SETS="19,20,21" bash scripts/NEAR/sweep_kl.sh <GPU>
```
3C 只把 3A-2 的选层文件当配置接口，`KL_INIT=original` 不用 steering；所以 **3C 只依赖 3A-1 的 r_UV 与 3A-2 的选层文件存在**，不依赖 3B。3B 与 3C 可同时跑。

## B. baseline（lr 等 `2_forget05_1B.md` 的 2a 试跑结论；下面是同样的推算值，3B 的 forget01 lr 与 1B 相同）
**[串]** 先试跑 GA + NPO，再其余；与 A、C 并行。
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.2-3B-Instruct EVAL_PROFILE=full LRS="2e-5" METHODS="GradAscent" bash scripts/unlearn/tune.sh <GPU> ; FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.2-3B-Instruct EVAL_PROFILE=full LRS="5e-5" METHODS="NPO" bash scripts/unlearn/tune.sh <GPU>
```
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.2-3B-Instruct EVAL_PROFILE=full LRS="2e-5" METHODS="GradDiff RMU" bash scripts/unlearn/tune.sh <GPU>
```
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 MODEL=Llama-3.2-3B-Instruct EVAL_PROFILE=full LRS="5e-5" METHODS="SimNPO" bash scripts/unlearn/tune.sh <GPU>
```

## C. LUNAR（c1、1.5、2；与 A、B 并行）
```
MODEL=Llama-3.2-3B-Instruct RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 EVAL_PROFILE=full COEFFS="1 1.5 2" FORGET_SPLITS="forget05" bash scripts/LUNAR/sweep_coeff.sh <GPU>
```

## 看什么
- 对照 forget01×3B：新方法 l19-21 Prob 0.209 / mia 0.685 / utility 0.663；旧方法 c8 Prob 0.515 / mia 0.950。forget05 上顺序是否保持，`forget_quality` 是否仍低。
- 开销：3A-1 与 3A-2 写 `cost_r_uv.json`、`cost_select_*.json`，`python scripts/analysis/cost_table.py saves/NEAR/forget05_Llama3B_v2 saves/NEAR/forget05_Llama3B_v3` 即可得到旧方法含上游的总开销。
