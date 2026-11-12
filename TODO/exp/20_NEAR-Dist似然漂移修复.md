# 20 NEAR-Dist 似然漂移的修复（2026-11-12）

## 问题（见 `paper_notes/10` §10、§10.1）
forget05 / forget10 上 NEAR-Dist 把整个模型的答案概率一起压低（3B forget05：陌生作者 0.156→0.095，保留答案 retain_Q_A_Prob 0.94→0.29），
遗忘答案却停在**原模型**的陌生水平附近（静态目标），于是比陌生作者高 2–3 倍，mia_loss 0.69–0.79（retain 0.36–0.40）。

## 根源（代码核对）
1. `w_down.py`：遗忘答案的目标概率（`neighbor_scaled`）只在训练前用冻结的原模型算一次，训练中不更新。
2. `kl_refine.refine_w_down`：保留集只约束“被编辑层的 MLP 输出”（激活 MSE），不约束保留答案的输出分布；KL 项把遗忘提示推向邻居答案的分布，这种推动会泛化，没有项把它拉回来。

## 改动（默认全部关闭，旧运行可复现）
- `wdown.kl.retain_kl_weight`（变体名 `_rkl<w>`）：保留 batch（retain + train 邻居）答案 token 上 + w · KL(原模型 ‖ 当前模型)。原模型与当前模型的前向本来就在算，只多保留 logits。
- `wdown.kl.forget_prob_dynamic_every`（`_dyn<N>`）：每 N 步用当前模型测一次陌生参照集的答案概率，除以第 0 步的值，按这个比例缩放所有遗忘目标——遗忘答案跟随“编辑后模型眼中的陌生水平”。参照集 `forget_prob_dynamic_ref` 默认 `neighbor_heldout`（从未训练过；**不用 TOFU holdout，因为 mia_loss 就是在 holdout 上算的**），固定 256 条（`sampling_seed`）。
- 扫描脚本：`RKL_WEIGHT=`、`DYN_EVERY=`、`DYN_REF=`（`sweep_w_down.sh` / `sweep_kl.sh`）。
- 测试：`tests/test_pipeline_smoke.py::test_near_w_down_kl_refinement_runs_end_to_end` 增加两项开启时的端到端检查（历史里记录 `retain_kl`、`target_scale`、`unseen_ref`；未知参照集报错）。

## 验收标准（跑之前固定，`scripts/analysis/dist_fix_check.py`）
C1 forget_Q_A_Prob ≤ 0.25；C2 遗忘 / 陌生作者答案概率之比在 [0.8, 1.25]；C3 |mia_loss − retain| ≤ 0.15；C4 retain_Q_A_Prob ≥ 0.70；
C5 ud_bb、ud_wb_unsup、ud_wb_sup 不超过原 E8 + 0.05；C6 utility 不低于原 E8 − 0.01。
现状：原 E8 只过 C1、C5、C6（比值 2.12，mia 0.694，retain_Q_A_Prob 0.290）；NEAR-Act 过 C1–C4、C6，C5 不过（白盒 0.93 / 0.96）。

## 验证（3B forget05，一张 40 GB 卡，4 个变体 + 验收表）
```
bash scripts/exp/dist_fix_lane.sh <GPU>
```
输出在新前缀 `saves/eval/NEAR/forget05_Llama3B_v4/`（`cells.py` 只读 `_v2` / `_v3`，主表不会被悄悄改变）；验收表 `saves/analysis/dist_fix_3B_forget05.md`。
通过后：其余 forget05 / forget10 格子（1B、8B、forget10 3B）用同一设置重跑，再决定是否替换主表的 NEAR-Dist（需要同时更新 `cells.py` 的 E8 识别）。时限：1–2 天，修不好就按“各有所长”写，mia 作为明确局限。
