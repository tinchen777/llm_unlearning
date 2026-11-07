# 20 消融：去掉 steering（纯 KL + 铰链 + 邻居）

## 问题
讨论后（见 `paper_notes/3` §13）确定的设定：遗忘时可以用 **遗忘前模型、遗忘数据、retain 数据**，邻居由 **GPT 生成**。现在的 NEAR = 多层顺序分摊 steering + 答案 token + KL + 铰链。near_line/19 的数据显示，c1 时 steering 本身几乎不动 Prob（0.88→0.86），KL 把它降到 0.56，铰链降到 0.08，所以**遗忘的主体可能是 KL + 铰链**，而不是激活重定向。如果去掉 steering 结果相同，"激活重定向"这个核心贡献就不成立，方法的实质是"邻居引导的分布匹配 + 有界的真实答案概率抑制"。

## 做法
`scripts/NEAR/kl_layers/8_ablate_steering.sh`（层 11-12-13，c1 只用来取选层 / r_UV 文件，KL=1，铰链 μ=1、目标 `unseen`、模式 `match`，**完整评测（`EVAL_PROFILE=full`）**，含释义 / 扰动问题上的 Prob）。代码新增：`wdown.kl.init`（`stage1` | `original`：不做 steering 阶段，从原权重开始）、`wdown.kl.forget_weight`（精修阶段 forget 激活 MSE 的权重，0 = 没有 steering 锚）。

| 变体 | steering 初始化 | forget 激活锚 | KL | 铰链 | 回答的问题 |
|---|---|---|---|---|---|
| withsteer | 是 | 是 | 是 | 是 | 完整方法（对照） |
| noanchor | 是 | 否 | 是 | 是 | steering 只作初始化够不够 |
| **nosteer** | **否** | **否** | 是 | 是 | **纯 KL + 铰链 + 邻居** |
| hingeonly | 否 | 否 | **否** | 是 | 邻居的 KL 在铰链之上加了什么 |
| klonly | 否 | 否 | 是 | **否** | 对照：KL 单独能不能动 Prob |

## 结果
未跑。

## 怎么读
- **nosteer ≈ withsteer**（Prob、utility、retain 生成、`wb_unsup`、释义 Prob 都相近）→ steering 不必要；方法应改写成"邻居引导的分布匹配 + 铰链"，steering 降级为可选 / 白盒痕迹的旋钮；这可能是更好的独立方法（见下）。
- **nosteer 明显差于 withsteer** → steering 有贡献（初始化 / 锚），重定向的叙述成立。
- **hingeonly ≈ nosteer** → 邻居的 KL 没有贡献，方法就是"有界的真实答案概率抑制"，要看它和 NPO 的区别。
- **klonly** 若 Prob 不动 → KL 本身不遗忘（预期），只负责让回答自然。
- 全部以 `forget_Q_A_PARA_Prob` / `PERT`（释义 / 扰动）检查泛化：铰链直接降低训练用 forget 答案的概率，评测的 `forget_Q_A_Prob` 用的是同一批问答。
- 注意 `wb_unsup`：没有 steering 时偏移不是"一致方向的大偏移"，白盒痕迹可能更低（也可能是权重修改本身引起的另一种痕迹）。

## 下一步
按结果决定：(a) steering 保留 → 继续 3B / 8B；(b) steering 不必要 → 重写方法文档，评估"纯 KL + 铰链 + 邻居"作为第二个方法或独立论文（见 `paper_notes/3` §13 的文献对照）。

## 输出文件夹（变体标签不出现在文件夹名里；末尾的 `_c1_all` 是 shift 模式 `all`，不是 TOFU 的 full 模型）
训练：`saves/NEAR/single_v2/<run>/`；评测：`saves/eval/NEAR/single_v2/<run>_k5_l11-12-13_c1_all/`
| 变体 | `<run>` |
|---|---|
| withsteer | `rw1_ans_kl1_fp1_ftunseen_fmmatch_increment_seq_c1_l11-12-13` |
| noanchor | `rw1_ans_kl1_fp1_ftunseen_fmmatch_kfw0_increment_seq_c1_l11-12-13` |
| nosteer | `rw1_ans_kl1_fp1_ftunseen_fmmatch_initoriginal_kfw0_increment_seq_c1_l11-12-13` |
| hingeonly | `rw1_ans_kl0_fp1_ftunseen_fmmatch_initoriginal_kfw0_increment_seq_c1_l11-12-13` |
| klonly | `rw1_ans_kl1_initoriginal_kfw0_increment_seq_c1_l11-12-13` |
