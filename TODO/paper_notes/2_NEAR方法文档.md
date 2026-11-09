# NEAR 方法文档（写论文 Method 一节用）

> **命名（2026-11-09 起）**：NEAR ＝ NEighbor-Anchored Redirection。NEAR-Act ＝ 激活层面的重定向（原“老方法 / NEAR-Steer”：steering + 回归）；NEAR-Dist ＝ 输出分布层面的重定向（原“新方法 / NEAR-Match / E8”）。详见 `0_命名对照.md`。

> 本文按"当前代码和已定下的配置"描述方法；与代码对应的位置写在括号里。数字（默认超参）取自 `configs/experiment/unlearn/tofu/NEAR.yaml`。实验证据见 `1_实验与结论梳理_data_metric_NEAR.md`，讨论过程见 `3_讨论纪要与阶段结论.md`。
> 标记：【已验证】有 1B 结果支撑；【未验证】计划但没有数据。

---

## 1. 一句话概述

**NEAR（NEighbor-Anchored Redirection，以邻居为锚的重定向）**：不教模型"拒绝回答"，而是把被遗忘问题的内部激活，**重定向到与之同属性的"陌生邻居"问题的激活区域**。对每个被遗忘的作者，用 GPT 生成 K 个属性匹配的虚构邻居作者（国家、体裁、身份等相同），取"邻居问题的平均激活 − 该作者 forget 问题的平均激活"作为这位作者的**重定向向量 r_UV**；再只改写 MLP 的输出投影 `W_down`（几个层），使 forget 提示的 MLP 输出平移 `c · r_UV`，而 retain 与邻居提示的输出保持不变。编辑后的模型对被遗忘问题给出**通顺但内容错误**的回答（像在回答一个不认识的人），不输出拒答，也不乱码。

与 LUNAR（同样是激活重定向 + `W_down` 编辑）的核心区别见 §9。

## 2. 设定与符号

- 原模型 $M$（在 forget 集 $D_f$ 与 retain 集上训练过，如 TOFU 官方 full 模型）。目标：得到 $M'$，使 $M'$ 对 $D_f$ 的行为接近"从未见过 $D_f$ 的模型"（TOFU 的 retain 模型）。**训练 $M'$ 时不使用 retain 模型**（只在评测里作参照）。
- $D_f$：forget 问答（TOFU forget01：40 条，2 个作者，每作者 20 条连续存放）。作者 $a$ 的 forget 问题集合 $Q_a$。
- 邻居数据 $N_a$：作者 $a$ 的 15 个邻居，每个邻居 20 条与 $Q_a$ **一一对应**的问答（模板相同、只替换实体）；其中 10 个标为 `train`（有排名 `train_rank = 1..10`），5 个标为 `heldout`。训练只用 `train`，评测只用 `heldout`（见 §3、§8）。
- 层 $l$ 的激活：残差流在该层 block 之后的输出（`point=block_out`）$h_l(x)\in\mathbb R^{d}$；MLP 输出投影 `W_down`：$y_l = W_l\,u_l$，其中 $u_l\in\mathbb R^{d_{ff}}$ 是 `down_proj` 的输入，$y_l\in\mathbb R^{d}$ 是 MLP 的输出（1B：$d=2048,\ d_{ff}=8192$，16 层；3B：$d=3072,\ d_{ff}=8192$，28 层；8B：$d=4096,\ d_{ff}=14336$，32 层）。
- 位置集合 $P$：问题结束处的模板 token（end-of-instruction，eoi；`get_eoi_positions`，即 chat 模板里用户消息之后、助手回答之前的几个 token）。

## 3. 第 1 步：重定向向量 r_UV（`src/METHOD/NEAR/r_uv.py`）

对作者 $a$、层 $l$、位置 $p\in P$、邻居数 $K$：

$$ r^{(l)}_{a,p}[K] \;=\; \underbrace{\frac{1}{|N_a^{K}|}\sum_{x\in N_a^{K}} h_l(x)_p}_{\text{邻居问题的平均激活}} \;-\; \underbrace{\frac{1}{|Q_a|}\sum_{x\in Q_a} h_l(x)_p}_{\text{该作者 forget 问题的平均激活}} $$

- $N_a^K$ = 作者 $a$ 的 `train` 邻居里 `train_rank ≤ K` 的那些（嵌套：K 个邻居是前 K 个；`ruv.order: rank`）；其 20 题一起平均。`ruv.ks=[1,3,5,10]` 一次算出，主线用 **K=5**。
- **按作者**计算，不是全局一个向量（LUNAR 用全局的 harmful / unverified 参考集平均）。【已验证】K≥3 的 r_UV 与 K=10 的余弦 0.93–0.98；用 train 邻居算的 r_UV 对 heldout 邻居的余弦 0.85–0.96（泛化）；K=1 不稳（0.73–0.81）。
- 保存：`r_uv.pt`（每个 K、作者、位置、层）、`r_uv_summary.json`（范数、相对范数、与最大 K 的余弦）。
- 观察【已验证】：未编辑的 full / retain 模型里，forget 与邻居的激活均值几乎重合（l9：余弦 1.00，范数 4.8–4.9，r_UV 范数只有 0.23–0.33，`auc[forget|neighbor]` 0.33–0.46）。r_UV 编码的是"实体身份"的小差别。

## 4. 第 2 步：steering 检验与选层（`src/METHOD/NEAR/select_layer.py`）

W_down 只能复现推理时 steering 能做到的事，所以先**不训练**，在推理时把 $c\cdot r_{a}$ 加到每个候选层 $l$ 的 block 输出上（`Steerer`，等价于一个完美的 W_down 编辑），生成 forget 问题的回答，并与"陌生问题"（`neighbor_heldout`、`holdout`）上未编辑模型的行为比较：

$$ \text{unseen\_gap}(l) = \frac{1}{|\mathcal K|}\sum_{k\in\mathcal K}\frac{|s_k - u_k|}{|o_k - u_k|} + |\text{refusal}_s - \text{refusal}_u| ,\quad \mathcal K=\{\text{ROUGE-L recall},\ \text{answer log-prob}\} $$

$s$ = steering 后的 forget 回答，$o$ = 未 steering 的 forget 回答，$u$ = 陌生问题的回答。1 = 没变，0 = 已像陌生问题；退化率（乱码、重复）超过 `max_degenerate=0.1` 的层不选。选出使 unseen_gap 最小的层。可选项（默认关，仅记录）：`ud_weight`（黑盒可检测度）、`footprint_weight`（footprint = $c\|r\|/\|\bar h_f\|$，白盒痕迹的代理）。

发现【已验证】：
- **有效系数很大**：`eoi` 模式、c=1 在 0–13 层无效；`all` 模式（每个 forget token 加同一个 $r_{a,-1}$）在 c≥6 才有效（l6–l9）。这是因为 r_UV 相对激活很小（l9：‖r_UV‖≈0.3，激活范数≈4.8，约 7%）。
- 各系数选出的层：1B：c4→13、c5/c6→11、c7→9、c8→8、c12→6（深度从 81% 到 38%，系数越大越浅）；3B：c4→18、c6/c7→15、c10→14、c12→13（相对深度与 1B 一致）。
- 3B 要达到相近的 steering 效果，footprint 约为 1B 的 2 倍。

## 5. 第 3 步：训练 W_down（`src/METHOD/NEAR/w_down.py`，`src/METHOD/_common/editing.py`）

### 5.1 单层：逐层局部回归
用**原模型**对每个样本前向一次，记录每个 token 的 `down_proj` 输入 $u$ 与 MLP 输出 $y^{0}$（原输出），构造目标：

$$ t = \begin{cases} y^0 + c\cdot s_a & \text{forget token（作者 } a\text{）}\\ y^0 & \text{retain / 邻居 token}\end{cases} $$

- $s_a$：`shift_mode=all` 时对该 forget 样本的**每个 token**加同一个 $r_{a,p_{ref}}$（$p_{ref}=-1$，LUNAR 开源代码的做法，也是论文设定）；`eoi` 时只在 eoi 位置各加自己的 $r_{a,p}$（目前主线用 `all`）。
- 对 $W$ 解 $\min_W\ w_f\,\overline{\|W u_f - t_f\|^2} + w_r\,\overline{\|W u_r - t_r\|^2}$（各取 token 平均）。求解器：AdamW（`lr 5e-3`，每 epoch ×0.9，`epochs 20`，每步 64 个 forget token + 64 个随机 retain token，`weight_decay 0`），或闭式岭回归（`solver: closed_form`，LUNAR Eq. 9）。主线 `adam`，$w_f=w_r=1$。
- **retain 集合** = `retain` split（随机 512 条）+ `train` 邻居（所有 token）。邻居是重定向的**目标**，所以也要保持不动；`holdout`（真实 TOFU 未见作者）完全不参与训练，作为干净的"从未见过"对照。
- 这是一个**线性回归**：训练时只有这一层的 $(u,t)$，不需要穿过后面的层反向传播，没有拒答目标，没有对 forget 答案的损失。

### 5.2 答案 token（`include_answer`）
默认只训练 prompt 的 token。`include_answer=true`：forget 与 retain 的"prompt + 答案"全部 token 都参与（forget 的答案 token 也加位移，可用 `answer_coeff` 缩放）。【已验证】retain 生成 ROUGE 明显提高（0.636→0.760，单层；多层 0.474→0.606），遗忘指标基本不变；对降低 MIA 没有帮助（"答案 token 没被训练到所以 MIA 高"的假设不成立）。主线开启。

### 5.3 多层分摊（`multi_layer=increment`）与顺序训练（`sequential`）
单层要把全部位移 $c\,r^{(l)}$ 放在一层，与 retain 冲突大。改为把总位移**分摊到 $m$ 层** $l_1<\dots<l_m$：

- **`increment`**：第 1 层取整个 $r^{(l_1)}$，之后每层取与上一个被编辑层的**差**：$\Delta_i = r^{(l_i)} - r^{(l_{i-1})}$（$r^{(l_0)}=0$）。残差流把低层加入的位移带到高层，因此总位移 $=\sum_i c\,\Delta_i = c\,r^{(l_m)}$（telescoping，等价于只编辑最后一层，但分到多层）。若每层都加整个 $r^{(l)}$（`same`）会重复计数，总位移约为 $m$ 倍。
- **`sequential`**：从低到高逐层训练；训练完一层后写入权重，**用已编辑的模型重新记录下一层的输入** $u$（目标 $t$ 不变：原输出 + 该层的增量），这样每一层同时修复下层编辑对它输入造成的偏移。
- 【已验证】单层 c7/L9 → 多层 L7-8-9：Prob 0.457 → 0.161（含答案 token 0.206），mia 0.883 → 0.53，privleak −75 → −26；`sequential` 是必须的（独立训练 C：utility 0.397，各层 `achieved_vs_intended` 余弦 −0.11 / 0.008，靠破坏模型"遗忘"）。层数越多 retain 损害越大（retain 生成 0.636→0.581→0.474）。

### 5.4 主配置
1B：K=5，`shift_mode=all`，c=7，层 **7-8-9**（c7 的选层结果是第 9 层，再向下取两层），`increment` + `sequential` + `include_answer`，`retain_weight=1`，AdamW 局部回归。3B 计划：c6、层 13-14-15（选层结果为 15），按相对深度折算【未验证】。

## 6. 第 4 步（可选）：端到端精修，KL 与铰链（`src/METHOD/NEAR/kl_refine.py`）

阶段 1 的局部回归不知道编辑后的模型**会说什么**：激活偏移很大时，答案的第一个 token（由 prompt 最后一个位置的隐状态预测）会被扰动成"Basil"的近邻非词（"Basis to…"，c≥6 时 10–12/40）。阶段 2 把被编辑的 $m$ 个 `W_down`（初值 = 阶段 1 的解，fp32 主权重）**联合**通过整个模型训练：

$$ \mathcal L = w_f\,\widetilde{\mathrm{MSE}}_{forget} + w_r\,\widetilde{\mathrm{MSE}}_{retain} + \lambda\,\mathrm{KL}\big(p_{M}(\cdot\mid x_{nb},\,y_{nb,<t}) \,\|\, p_{M'}(\cdot\mid x_{f},\,y_{nb,<t})\big) + \mu\ \mathrm{relu}\big(\log \bar p_{M'}(y_f\mid x_f) - \log \tau\big) $$

- $\widetilde{\mathrm{MSE}}$ = 各层 MLP 输出的 MSE / 目标均方（相对误差，三项量级可比）；forget 目标 = $y^0 + c\,s_a$，retain 目标 = $y^0$（原模型在线计算，`ref_model` 是编辑前冻结的拷贝）。
- **KL 项**：对每个 forget 问题 $x_f$，取其 `train` 邻居的问题 $x_{nb}$ 与答案 $y_{nb}$（配对键 `(author_id, qa_id)`，并断言 `source_question` 与该 forget 问题一致，防止行序错位；每步从该问题的 10 个邻居里随机抽一个）。编辑后模型在 **forget 提示 + 邻居答案前缀**上、对前 `n_tokens=8` 个答案 token 的分布，逼近**原模型在邻居提示 + 同一答案前缀**上的分布。只需要原模型，不需要 retain 模型。
- **铰链项**（可选，$\mu>0$）：惩罚真实 forget 答案的（teacher-forced）平均 token 概率 $\bar p = \exp(\text{mean log-prob})$（即评测的 `forget_Q_A_Prob`）高于 $\tau$（retain 模型的值，1B 为 0.17），低于 $\tau$ 就不再惩罚。**$\tau$ 不取自 retain 模型**：取原模型在**未见数据**（TOFU holdout 作者）上的答案概率（`forget_prob_target: unseen`；full 模型在 holdout 上的答案概率 1B 0.161 / 3B 0.159 / 8B 0.130，≈ retain 模型的 forget Prob 0.169 / 0.178 / 0.139），所以仍只需要原模型与未见问答。早期实验直接用了 retain 的 0.17（违反设定），且 hinge 过冲（目标 0.17 → 实际 0.08）；另有双向的 `match` 模式（向 $\tau$ 靠拢，不过冲）。**设计动机**：KL 约束的是邻居答案的前缀，不降低真实答案的概率；300 步 KL 从 0.61 降到 0.39，Prob 仍 0.515→0.502。**铰链的结果尚未出来**【未验证】。
- 超参：`steps=100`，`lr=2e-5`（AdamW），$\lambda=1$（3 与 1 几乎无差别），`n_tokens=8`，每步 8 个（forget 提示, 邻居答案）对 + 一个 forget 批 + 一个 retain 批；需要 `include_answer=true`；显存多一份参照模型。
- 【已验证】KL 在 12 个（层集, 系数）组合上一致地把 Prob 再降 0.25–0.3，utility 只少 0.01–0.02，forget 回答首词变成 "Ali / Leonid / Pavel"（邻居式的另一位虚构作者），`ud_bb` 0；越深的层 retain 代价越小。KL 不改变白盒痕迹。

## 7. 推理与成本
- 编辑后的模型 = 原模型的 $m$ 个 `down_proj` 矩阵被替换（1B：每层 $2048\times8192=16.8$M 参数，3 层 ≈ 50M），**推理没有额外开销**，不需要 retain 模型、不需要保留原模型。
- 训练成本：阶段 1 只记录一次激活并做局部回归（不需要对整个模型反向传播）；显存主要是模型权重（估计 8B 约 20–25 GB，**未实测**）。阶段 2 多一份参照模型和穿过整个模型的反向（估计 8B 45–50 GB，**未实测**）。

## 8. 数据协议与评测（与方法同等重要）
- **邻居数据的 train / heldout 分离**：r_UV 与 W_down 的 retain 只用 `train` 邻居；所有评测里的邻居（`neighbor_*`、探针、UD 的良性集、选层的"陌生问题"）只用 `heldout`。旧代码把全部邻居用在训练与评测上，评测不干净；修正后 NEAR 性能几乎不变（Prob 0.163→0.161），可信度提升。
- 评测（一次评测，`experiment=eval/tofu/probe`）：遗忘（`forget_Q_A_Prob`、`extraction_strength`、`mia_loss`；`forget_quality` 在 n=40 上是离散台阶，不据此排序）、保持（`model_utility`、retain 生成 ROUGE、holdout / 邻居 ROUGE）、可检测度（黑盒 `ud_bb`、`entity_bb`；白盒 `wb_sup`、`wb_unsup`）。评判标准是**与 retain 模型相近**（Prob ≈ 0.17、mia ≈ 0.48、extraction ≈ 0.06），而不是越低越好。`EVAL_PROFILE=fast` 为扫参用的子集。
- 参照：官方 TOFU full / retain99 模型（`saves/eval/{full,retain}/<模型>/forget01`）。1B 自训 full 模型坏掉（checkpoint-0 就是乱码），一律用官方模型。

## 9. 与 LUNAR 的区别
| | LUNAR | NEAR |
|---|---|---|
| 重定向目标 | 全局的参考集（harmful / unverified）的平均激活 → 把 forget 推向"拒答区域" | **每个作者**的属性匹配邻居的平均激活 → 推向"陌生人"区域 |
| 回答形态 | 拒答（forget 回答 55–70% 是拒答）、对 retain 问题也拒答（TAS 口径 8–20%，旧口径 32%） | 通顺的另一个说法，拒答率 0 |
| 选层 | 在一个深度窗口内用 steering 选层（其选出的层在 50–75% 深度） | 同样做 steering，比较对象是陌生问题（unseen_gap），并扩展到多层 |
| 层数 | 单层 | 多层 `increment` + `sequential` |
| 其他 | | 答案 token、可选 KL / 铰链、train/heldout 邻居协议 |
| 白盒 | 与 NEAR 相当：遗忘到位的强度下都容易被离群检测（forget01 / forget005 的系数扫描 `wb_unsup` 0.41–0.96，低值对应已被破坏或几乎没遗忘的模型） | 0.95–0.996（遗忘到 retain 水平的配置） |

## 10. 已知局限（论文里要写）
1. **白盒痕迹**：把 forget 激活推到邻居均值之外约 6 个 r_UV（l9：r_UV 范数 0.33→3.09，forget 范数 6.19 对邻居 4.97），`wb_unsup` ≈ 0.99；痕迹由（系数, 层）决定，随遗忘强度同向（Prob 0.62/0.53 → 0.556/0.57 → 0.40/0.76 → 0.21/0.99），KL 不改变它。
2. **依赖邻居数据**：需要为每个 forget 作者生成属性匹配的邻居（GPT 生成，K≥3 即够）；数据审计与数据消融**尚未做**。
3. 超参来自 1B 单次运行（n=40），3B / 8B 沿用同一配方，**未验证**；系数、层需要按规模重新选（3B 的 footprint 约为 1B 的 2 倍）。
4. 评测里 `forget_quality` 不可靠；所有结果无多种子。
