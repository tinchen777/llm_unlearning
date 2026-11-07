# 2 forget05 × 1B：baseline 与 LUNAR（NEAR 新旧已在跑，不在此列）

## 问题
forget05 每个 epoch 约 7 步，5 个 epoch 共约 35 步，是 forget01（10 步）的 3.5 倍；forget01 的 lr 可能过冲，所以 **lr 降一档**：GradAscent / GradDiff / RMU 2e-5（forget01 为 5e-5），NPO / SimNPO 5e-5（forget01 为 1e-4）。这是推出来的值，**未验证**，所以先试跑。每个 epoch 评一次，共 5 个点，用完整评测（`EVAL_PROFILE=full`，forget05 有 200 条，forget_quality 有统计意义）。

## 做法
**[串] 2a → 2b**（2b 要看 2a 的轨迹后再决定是否调 lr）；**2c 与 2a / 2b 并行**。

```
# 2a. 试跑：GradAscent（2e-5）接 NPO（5e-5），同一进程顺序跑 `on 1`
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full LRS="2e-5" METHODS="GradAscent" bash scripts/unlearn/tune.sh <GPU> ; FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full LRS="5e-5" METHODS="NPO" bash scripts/unlearn/tune.sh <GPU>
```
```
# 2b. 其余三个（等 2a 的轨迹合理再跑；GradDiff + RMU 一个进程，SimNPO 另一个，可分两张卡并行）
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full LRS="2e-5" METHODS="GradDiff RMU" bash scripts/unlearn/tune.sh <GPU>
```
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full LRS="5e-5" METHODS="SimNPO" bash scripts/unlearn/tune.sh <GPU>
```
```
# 2c. LUNAR（c1、1.5、2），与 2a / 2b 并行；邻居划分和作者偏移由脚本自动推出，只需给 retain 和 holdout
RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 EVAL_PROFILE=full COEFFS="1 1.5 2" FORGET_SPLITS="forget05" bash scripts/LUNAR/sweep_coeff.sh <GPU>
```
输出：`saves/tofu_unlearn/<方法>/tofu_Llama-3.2-1B-Instruct_forget05_lr<lr>_ep5/`；LUNAR：`saves/eval/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget05_LUNAR_harmful_c<c>_l<层>_all`。

## 看什么
- 2a：5 个 epoch 的 Prob / utility / mia 是否是一条轨迹（遗忘到位、不过早崩溃）；崩溃就 lr 再减半。把 GA 与 NPO 的轨迹发给我，再定 3B 的 lr。
- forget05 的 baseline 是 full 评测，forget01 的是 fast 评测，Pareto 图上指标集合不同（full 多了释义 Prob、全部 MIA）。
