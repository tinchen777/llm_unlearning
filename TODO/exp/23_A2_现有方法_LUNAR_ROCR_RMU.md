# 23 A2 在现有方法上：LUNAR、ROCR、RMU（2026-11-13）

> 对应 `22_待做实验表.md` 的 P0 #1。规格见 `A2_SPEC.md`，攻击定义见 `../paper_notes/2_攻击方法_A1_A2.md` §2。
> 状态：✏️ 代码已写好（CPU 上做过 dry-run 和停止规则的单元测试），**还没在 GPU 上跑过**。先跑下面"一、试跑"，确认影子强度正常后再跑全量。

## 问题
摘要承诺"自适应白盒攻击能在**现有方法**下识别被遗忘实体"，但 A2 只在 NEAR-Act 上实现过。要给 LUNAR、ROCR、RMU 加上影子编辑。

## 做法
流程与 NEAR-Act 完全相同（同一套作者集合 `sets_v2.json`、模板、探针 P1 / P2、选层规则、长度控制、统计口径、real / retain 两个条件），只换"影子编辑"这一步。每个方法都在 M'（或 retain95）上，对影子遗忘集 S_k（forget10 的第 40k–40k+39 行 = 作者 181+2k、182+2k）用 **provider 的算法与超参数**跑一次：

| 方法 | provider 的 M'（forget05 匹配点） | 影子编辑 | 影子强度（攻击者在 S_k 上执行 provider 的选点规则） | `--edit-layers` |
|---|---|---|---|---|
| LUNAR | 1B c1.5 l11；**3B c1.5 l12**；8B c2.8 l23（`saves/LUNAR/single/...`） | 原算法（`LUNAR_harmful` 参考集）：S_k 的 r_UV（每组算一次）→ 选层限定在 provider 的层 → W_down | 系数从弱到强（1B / 3B：0.5 1 1.25 1.5 1.75 2 2.5 3；8B：1.5 2 2.4 2.8 3.2 3.6 4），取第一个使 S_k 答案概率 ≤ 0.25 的（读 W_down 写出的 `responses_after_summary.json`，与 NEAR-Act 相同） | 编辑层 |
| ROCR | 原版（Donald Trump，系数 1）：1B l2-4、3B l3-5、8B l4-6 | `rocr.py --forget-rows 40k:40k+40`，目标同为 Donald Trump；null space 投影在 M' 上算一次（缓存 `data/rocr/<M' 文件夹名>/`） | **不选**：provider 的 ROCR 本身就没到匹配点，没有强度可对齐；只做一次编辑，强度看 collect 打印的代理概率 | 编辑层 |
| RMU | `_ckpt` 运行的最终模型（第 35 步 = 主表匹配点）：1B lr 5e-5、3B / 8B lr 1e-4 | provider 的 trainer 配置（第 7 层、全部参数、有效 batch 32）；forget = S_k，retain = retain90 | 每个 epoch 结束在 S_k 上算 `forget_Q_A_Prob`（provider 自己的指标，`eval=tofu_sprob`），**第一次 ≤ 0.25 就停**并保留该模型；最多 18 个 epoch（40 行 = 每 epoch 2 步，18 epoch = 36 步 ≈ provider 的 35 步）；一直没到就用最后一个（`stop.json` 里 `met: false`） | 第 7 层（攻击者知道 `module_regex`） |

代码改动：
- `scripts/exp/spa_lib.sh`：`spa_cfg` 加 `lunar` / `rocr` / `rmu`；新增 `spa_edit_lunar`、`spa_edit_rocr`、`spa_edit_rmu` 与分派函数 `spa_edit`、`spa_edit_root`；`spa_shadow` 改成按方法分派（NEAR 的行为和 done 标记不变）。删除编辑模型前，把小的 json（`responses_after_summary.json`、`stop.json`、`rocr_summary.json`）复制到 `logs/spa/<格子>/g<k>_*.json`。
- `scripts/exp/spa_lane_baselines.sh`：新的链（一个规模一次调用，方法之间失败不中断）。
- `src/METHOD/ROCR/rocr.py`：新增 `--forget-rows a:b`（只取遗忘集的这些行，必须是整位作者）。
- `src/trainer/base.py`：新增可选停止规则，**只在设置环境变量 `STOP_AT_METRIC` 时生效**（如 `forget_Q_A_Prob<=0.25`），默认行为不变。
- `configs/eval/tofu_sprob.yaml`：只算 `forget_Q_A_Prob` 的评测，行由 `eval.tofu.forget_hf_split` 指定。

与规格的偏离（已补进 `A2_SPEC.md` §9）：
1. ROCR 不做影子强度选择（理由见上表）。
2. RMU 的影子训练长度用"停止规则"选，最大步数对齐 provider 的总步数，而不是照搬 5 个 epoch（2 位作者只有每 epoch 2 步，5 个 epoch 只有 10 步，远少于 provider 的 35 步）。学习率调度按 18 个 epoch 计算，与 provider 的 5 个 epoch 不完全相同。
3. RMU 训练全部参数，所以它的 `pre_edit_mean`（第 7 层之前）**不是**"M' = 原模型"的残留认识，解读时只看 `post_edit_mean`、`cv`、`cv_deepest`。

## 一、试跑（先做，3B，每个方法只跑第 0 组）
```bash
KEEP_EDITED=1 SPA_GROUPS="0" bash scripts/exp/spa_lane_baselines.sh <GPU> 3B "lunar rocr rmu"
```
要看的输出：
- LUNAR：每个系数打印的 `S_k answer prob`；real 条件选中的系数应该与 provider 的 c1.5 同档。
- RMU：`RMU stopped at step / forget_Q_A_Prob(S_k) / met`；`met` 应为 True。
- ROCR：collect 打印的 `mean proxy answer prob by role`（S 的概率是否下降，provider 的 ROCR 在 F 上只到约 0.56）。
- 每个 collect 打印的 `mean proxy answer prob by role`：S 下降、H_A 不变。

试跑没问题后删掉试跑留下的模型（`KEEP_EDITED=1` 保留了它们），再跑全量；done 标记会跳过已完成的第 0 组。

## 二、全量（每个规模：3 个方法 × real / retain × 5 组 + 攻击）
```bash
bash scripts/exp/spa_lane_baselines.sh <GPU> 3B
```
```bash
bash scripts/exp/spa_lane_baselines.sh <GPU> 1B
```
```bash
bash scripts/exp/spa_lane_baselines.sh <GPU80> 8B
```
- 优先级：3B → 8B → 1B（1B 的 A2 对 NEAR-Act 也不成立，零线附近）。
- 估计：LUNAR 每组几个系数 × 几分钟；ROCR 第一次要算二阶矩（20000 段 WikiText，3B 约 10–20 分钟，之后用缓存）；RMU 每组最多 36 步全参数训练。每个方法 3B 约 2–4 h（real + retain）。
- 磁盘：同一时间只保留一个编辑后的模型（1B / 3B / 8B 约 2.5 / 6.5 / 16 GB），M' 每个方法物化一次（`saves/spa/models/`）；ROCR 的投影缓存每层 d_ff² 个 float（3B 约 0.27 GB / 层，8B 约 0.8 GB / 层）。
- 结果：`saves/analysis/spa_<规模>_<方法>_{real,retain}/summary.md`；影子强度 `saves/analysis/spa/shadow_strength_<规模>_<方法>_<条件>.tsv`。

## 三、结果怎么读（事先约定，与 NEAR-Act 相同）
主结果是 `post_edit_mean` 长度修正后的 P1 / P2 的 Q1、Q2。real 超过零线、**并且**明显高于同规则下的 retain 对照，才算识别成功。另外报告 `cv`、`cv_deepest`。LUNAR / ROCR 的 `pre_edit_mean` 与 NEAR-Act 一样是残留认识（编辑层之前 M' = 原模型）。

## 四、顺带发现的问题：3B forget05 的"LUNAR"目录其实是 forget10 的运行
- `saves/LUNAR/forget05_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget05_ks/` 的 `layer_selection.json` 写的是 `forget_split: forget10`、`num_forget: 400`；它的评测 `saves/eval/LUNAR/forget05_Llama3B_v2/near_tofu_..._ks_c1.5_l14_all/` 与 `saves/eval/LUNAR/single/lunar_tofu_Llama-3.2-3B-Instruct_forget10_LUNAR_harmful_c1.5_l14_all/` **逐字节相同**（Prob 0.137、utility 0.366）。也就是 **forget10 的 LUNAR 被放进了 forget05 的目录**。
- 真正的 3B forget05 LUNAR 匹配点是 `saves/LUNAR/single/lunar_tofu_Llama-3.2-3B-Instruct_forget05_LUNAR_harmful/c1.5_all/l12`（Prob 0.238、utility 0.537）。
- 受影响的：
  - **主表的数字没错**：匹配规则恰好选中了正确的点（0.238 / 0.537）。但 `cells.py::_locate` 按 `c1.5` 用通配符找目录，取到的是错的 `_ks` 目录，所以用到 `path` 的分析读错了文件：**E7 固定层白盒（`wb_fixed_all.py`）的 3B forget05 LUNAR 行**。
  - 3B forget05 的 Pareto 图多了 3 个假的 LUNAR 点（`_ks` 下的 c1 / c1.5 / c2，其实是 forget10 的数据）。
  - 直接写了 `_ks` 路径的链：**A1 v2 的 3B LUNAR 行**（`a1v2_lane.sh`、`a1_lane.sh`）、**A2 P0 的 3B LUNAR**（`a2_p0_lane.sh`）、**3B 自然度攻击的 LUNAR**（`a1v2_meow_lane.sh`）。这些用的都是 forget10 的 LUNAR 模型（它也遗忘了 F，但同时遗忘了 S 的作者 181–190，共 20 位）。
- 建议（**未做，需要你决定**）：在服务器上把两个 `_ks` 目录移走（例如改名为 `forget10_misfiled_*`），然后重跑上面四项的 3B LUNAR 部分。移走之后 `cells.py` 会自动找到正确的目录。
