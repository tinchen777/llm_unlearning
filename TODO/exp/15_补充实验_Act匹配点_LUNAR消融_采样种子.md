# 15 论文主基调下的补充实验：Act 补到匹配点、Act 对 LUNAR 的消融、E8 的 8B 采样与种子

> 对应对话中的四项。所有命令是独立的单行（可分卡）。先 `git pull`（新增了 `configs/data/datasets/LUNAR/LUNAR_neighbor{01,05,10}.yaml`、`validate_fix.sh` 的 `8B_f01`、`SEED`、`SAMPLE_SEED`）。命令我没有在真实模型上运行过，只做了测试和 `DRY_RUN`。

## 一、NEAR-Act 补到匹配点（forget05 的 8B、forget01 的 8B；1B forget05 等 `exp/13` 的 c9 / c11 结果）
**为什么换层组**：8B 上 c7 用层 17-18-19（选层 c7→19）Prob 0.181–0.193，而 c8–c10 的选层结果是 16，训练用 14-15-16，Prob 反而更高（0.43–0.45），不是系数的问题，是层组的问题。所以保持 17-18-19，只加大 c。

**8B forget01：c8、c10**（选层文件已有）
```
for c in 8 10; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" EXTRA_WDOWN="wdown.gen_batch_size=8" SELECT_TASK_NAME=Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget01_ks OUT_PREFIX=Llama8B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=17,18,19 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
**8B forget05：c8、c9、c10**
```
for c in 8 9 10; do FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" EXTRA_WDOWN="wdown.gen_batch_size=8" SELECT_TASK_NAME=forget05_Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget05_ks OUT_PREFIX=forget05_Llama8B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=17,18,19 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
**8B forget10：c8、c9**（依赖 `exp/13` 的选层）
```
for c in 8 9; do FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" EXTRA_WDOWN="wdown.gen_batch_size=8" SELECT_TASK_NAME=forget10_Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget10_ks OUT_PREFIX=forget10_Llama8B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=17,18,19 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
**8B forget01 的备选层组（18-19-20）**
```
for c in 8; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" EXTRA_WDOWN="wdown.gen_batch_size=8" SELECT_TASK_NAME=Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget01_ks OUT_PREFIX=Llama8B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=18,19,20 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
1B forget05 的 c9 / c11 在 `exp/13` 里；如果仍没到匹配点，再补 c13 / c15（要先选层，格式同 `exp/13`）。

## 二、消融：Act 比 LUNAR 好在哪里（1B、3B，forget01）
**阶梯**（每一级只改一件事）：
| 级 | 配置 | 改动 | 对应的因素 |
|---|---|---|---|
| L0 | LUNAR（harmful 参考集，单层，保持集＝retain，提示 token） | — | 已有 |
| L1 | LUNAR，参考集换成**训练邻居问题**，其余不变 | 目标：拒答 → 邻居 | **目标** |
| L2 | NEAR 流水线，单层，提示 token，保持集＝retain | 全局 r → 按作者的 r（K=5） | 按作者的 r |
| L3 | L2 + 保持集加入训练邻居 | 保持集 | **保持集加邻居** |
| L4 | L3 + 多层累加 + 顺序训练 | 层 | **多层顺序训练** |
| L5 | L4 + 答案 token ＝ NEAR-Act | 答案 token | 答案 token |
L1 用 LUNAR 自己的流水线（r、选层、回归都是 LUNAR 的），层固定为 NEAR 单层用的那一层，使目标成为唯一的变化。1B 用系数 7 和 12，3B 用 12（与完整 NEAR-Act 的匹配点一致）。已有：1B c7 的 L3（`rw1_c7_l9`）、L4（`rw1_increment_seq_c7_l7-8-9`）、L5；3B c12 的 L5。

**L1（LUNAR + 邻居目标）**
```
MODEL=Llama-3.2-1B-Instruct REFS="LUNAR/LUNAR_neighbor01" FORGET_SPLITS="forget01" COEFFS="7 12" LAYERS="[9]" EVAL_PROFILE=full bash scripts/LUNAR/sweep_coeff.sh <GPU>
```
```
MODEL=Llama-3.2-3B-Instruct REFS="LUNAR/LUNAR_neighbor01" FORGET_SPLITS="forget01" COEFFS="10 12 14" LAYERS="[13]" EVAL_PROFILE=full bash scripts/LUNAR/sweep_coeff.sh <GPU>
```
**L2（NEAR 单层，保持集只有 retain）**
```
for c in 7 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2_abl_L2 EVAL_PROFILE=full EXTRA_WDOWN="wdown.retain_splits=[retain]" LAYER=9 COEFF=$c RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
```
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2_abl_L2 EVAL_PROFILE=full EXTRA_WDOWN="wdown.retain_splits=[retain]" LAYER=13 COEFF=$c RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
**L3（单层，保持集加邻居）**——1B c7 已有，补 c12；3B 补 c12
```
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2 EVAL_PROFILE=full LAYER=9 COEFF=$c RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
```
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2 EVAL_PROFILE=full LAYER=13 COEFF=$c RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
**L4（多层累加 + 顺序训练，提示 token）**——1B c7 已有，补 c12；3B 补 c12
```
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2 EVAL_PROFILE=full LAYER=7,8,9 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
```
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2 EVAL_PROFILE=full LAYER=11,12,13 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
**L5（完整 NEAR-Act）**——1B 补 c12（c7 已有），3B c12 已有
```
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=7,8,9 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
读法：用同一个系数比较各级（再对照各级的匹配情况）；L0 → L1 是“目标”的贡献，L2 → L3 是“保持集加邻居”，L3 → L4 是“多层顺序”，L4 → L5 是“答案 token”。三个关键因素（目标、保持集、多层顺序）正好各对应一对相邻的级。注意：同一个系数下各级的强度不同，要同时看 Prob 与 mia，必要时再补几个系数，用 `pareto.py` 画曲线。

## 三、E8 的采样结果：8B 与种子
**8B forget01 的采样（E8 与早期校准）**
```
SETTING=8B_f01 VARIANTS="E8" SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh <GPU>
```
**采样种子（同一个模型换采样种子；成本很低）**
```
SAMPLE_SEED=1 SETTING=1B_f01 VARIANTS="E8" SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh <GPU>
```
```
SAMPLE_SEED=1 SETTING=3B_f01 VARIANTS="E8" SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh <GPU>
```
```
SAMPLE_SEED=2 SETTING=1B_f01 VARIANTS="E8" SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh <GPU>
```
```
SAMPLE_SEED=2 SETTING=3B_f01 VARIANTS="E8" SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh <GPU>
```
**训练种子（重训 E8，换训练种子；输出到 `<前缀>_s<n>`；含评测与采样）**
```
SEED=1 SETTING=1B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh <GPU>
```
```
SEED=2 SETTING=1B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh <GPU>
```
```
SEED=1 SETTING=3B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh <GPU>
```
```
SEED=2 SETTING=3B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh <GPU>
```
```
SEED=1 SETTING=1B_f05 VARIANTS="E8" bash scripts/exp/validate_fix.sh <GPU>
```
```
SEED=2 SETTING=1B_f05 VARIANTS="E8" bash scripts/exp/validate_fix.sh <GPU>
```
**NEAR-Act 的训练种子（和 NEAR-Dist 的误差条对比用）**——输出到 `single_v2_s<n>` / `Llama3B_v2_s<n>`
```
for c in 7; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2_s1 EVAL_PROFILE=full EXTRA_WDOWN="seed=1" INCLUDE_ANSWER=1 LAYER=7,8,9 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
```
for c in 7; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2_s2 EVAL_PROFILE=full EXTRA_WDOWN="seed=2" INCLUDE_ANSWER=1 LAYER=7,8,9 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
```
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2_s1 EVAL_PROFILE=full EXTRA_WDOWN="seed=1" INCLUDE_ANSWER=1 LAYER=11,12,13 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
```
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2_s2 EVAL_PROFILE=full EXTRA_WDOWN="seed=2" INCLUDE_ANSWER=1 LAYER=11,12,13 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh <GPU>; done
```
训练种子只改变训练时的批次顺序等随机性（邻居的轮换分配是确定的），所以种子间的差异是下限；NEAR-Act 的随机性来自保持集的随机子集与回归的小批次。

## 四、各项在论文里的位置
1（Act 补到匹配点）→ 5.4 / 5.5 的九格对照；2（LUNAR 消融）→ 5.6 的 NEAR-Act 消融，支撑“Act 优于 LUNAR 的原因”；4（8B 采样、种子）→ 5.7，以及所有主表的误差说明。
