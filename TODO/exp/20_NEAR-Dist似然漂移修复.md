# 20 NEAR-Dist 似然漂移的修复（2026-11-12）

## 问题（见 `paper_notes/3_最新数据与新发现.md` §2；旧版 `paper_notes/history/10_*` §10、§10.1）
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

## 第 1 轮结果（3B forget05，`saves/analysis/dist_fix_3B_forget05.md`）

【数据】

| 运行 | Prob | 遗忘/陌生 | mia（retain 0.363） | retain_Q_A_Prob | utility | ud_bb | wb_unsup | wb_sup | 未过 |
|---|---|---|---|---|---|---|---|---|---|
| 原 E8 | 0.172 | 2.12 | 0.694 | 0.290 | 0.542 | 0.013 | 0.013 | 0.178 | C2 C3 C4 |
| dyn25 | 0.074 | 1.87 | 0.664 | 0.157 | 0.450 | 0.000 | 0.013 | 0.178 | C2 C3 C4 C6 |
| rkl1 | 0.142 | 0.96 | **0.317** | 0.790 | **0.662** | 0.000 | 0.013 | 0.285 | C5 |
| rkl1_dyn25 | 0.177 | 1.17 | 0.389 | 0.859 | 0.668 | 0.027 | 0.013 | 0.301 | C5 |
| rkl3_dyn25 | 0.145 | 1.15 | 0.426 | 0.879 | 0.666 | 0.007 | 0.013 | 0.251 | C5 |
| NEAR-Act | 0.166 | 1.13 | 0.501 | 0.757 | 0.586 | 0.000 | 0.930 | 0.956 | C5 |

retain95 的 utility 0.659。同格其他方法的 wb_sup：GA 0.17、NPO 0.24、GradDiff 0.27、SimNPO 0.31、MEOW 0.22、LUNAR 0.77、ROCR 0.91、RMU 1.00。

【我的判断】
- **根源是“保留集只约束激活、不约束输出分布”**：只加保留答案 KL（rkl1）就把 C2–C4 全部修好，mia 0.317 与 retain 的 0.363 只差 0.05，utility 从 0.54 升到 0.66（≈ retain 模型 0.659，高于 NEAR-Act）。
- **动态目标不需要，单独用还有害**（dyn25：所有概率进一步一起下降，utility 0.45）：它追着漂移走，漂移本身没被拦住。最终配置建议只保留 retain KL。
- **唯一没过的是 C5：wb_sup 从 0.18 升到 0.25–0.30**（门槛 0.23）。按事先定的标准这算没过，必须如实报告。但这个水平与 GA / NPO / GradDiff / SimNPO 同档，远低于 Act / LUNAR / ROCR / RMU；而且 8B 上 E8 三个种子的 wb_sup 是 0.49 / 0.36 / 0.35，种子波动约 0.14，与这里的升幅（+0.07–0.12）同量级。是否是真升高，要看第 2 轮的种子。
- **第 2 轮**（`scripts/exp/dist_fix_lane2.sh`）：rkl3、rkl10（更强的 retain KL 是否压低 wb_sup：rkl3_dyn25 的 0.251 是第 1 轮最低）；原 E8 与 rkl1 各 2 个种子（修复前后 wb_sup 的种子噪声）。

## 消融：KL 目标（2026-11-13）
`wdown.kl.target`（扫描变量 `KL_TARGET`，变体 `_t<目标>`），其余与 NEAR-Dist v2 完全相同：
- `idk`：`data/idk.jsonl` 的"我不知道"回答，对前 8 个 token 做交叉熵，每题 3 条（固定轮换）。
- `shared_neighbor`：所有被遗忘作者都用同一位邻居（第一位作者的 rank-1 邻居）——共享但不是拒答的目标。
- `mismatched_neighbor`：每位作者用遗忘集里下一位作者的邻居——目标各不相同，但属性不匹配。
逐题的概率标定仍按各作者自己的邻居计算。

已有：3B / 1B forget05 的 `idk`（`dist_ablation_idk_lane.sh`，结果见 `paper_notes/3` §2.4）。
补充（三条链，可同时跑）：
```
bash scripts/exp/abl_target_big.sh <80GB GPU>    # 8B forget05：idk / shared / mismatched
bash scripts/exp/abl_target_3b.sh <40GB GPU>     # 3B forget05：shared / mismatched / idk 种子 1、2；3B forget01：idk
bash scripts/exp/abl_target_1b.sh <GPU>          # 1B forget05：shared / mismatched；1B forget10：idk（显存小，可与 big 同卡）
```
结果：`saves/analysis/dist_fix_<规模>_<split>.md`，对比 "Dist rkl1 (s0)"（v2）与 "Dist rkl1_t<目标> (s..)"。
预期（我的判断，未验证）：若"共享目标"是白盒痕迹的来源，`shared_neighbor` 的白盒会升高而 `ud_bb` 仍为 0；`mismatched_neighbor` 的白盒应保持低。
