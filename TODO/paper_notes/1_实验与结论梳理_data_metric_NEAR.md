# 论文用：实验与结论梳理（data / metric / NEAR 方法）

> 截至 near_line/13、metric_line/4、data_line/4。数字出处在对应的 line 文件里（路径写在那里）。
> 所有实验：Llama-3.2-1B-Instruct 官方 full / retain99 模型，TOFU forget01（作者 199、200，40 条），K=5，**单次运行**（无多种子）。
> 标记：**【稳】** 多个配置/协议下一致；**【初步】** 单个配置或解释未经干预验证；**【待做】** 还没做。

---

## 0. 一页结论（可以写进论文的 / 暂时不能写的）

| 可以写（有数据支持） | 暂时不能写 |
|---|---|
| 把 r_UV 的位移**分摊到多层并顺序训练**，比单层大系数遗忘强得多（Prob 0.46→0.16，mia 0.88→0.53），且不拒答、不乱码 | "NEAR 全面比 LUNAR 更隐蔽"：白盒上没有优势（见 §3.4） |
| 最佳配置（B+答案 token，c7，层 7-9）的 Prob/extraction/MIA 与 retain 模型接近；utility 0.57（retain 0.60），LUNAR 默认设置 0.27 | 与调好参的 baseline / LUNAR 的"公平"优劣：它们的调参预算有限（只调 lr 两档 / 只扫系数），baseline 无 UD |
| 加 KL 第二阶段后，forget 回答是流畅的"另一位虚构作者"，黑盒 `ud_bb`=0，怪词（"Basis…"）消失 | 白盒痕迹已降低：KL 单层 l9 只降到 0.953 |
| 黑盒 TAS 打分/可检测度指标已实现，LUNAR 按作者聚合 `entity_bb`=1.0（一个作者 15% 拒答就能认出） | `forget_quality` 的任何排序：n=40 时它是离散台阶（见 §2.5） |
| 邻居数据按 train/heldout 分离后，对 NEAR 结果影响很小（Prob 0.163→0.161），r_UV 对没见过的邻居泛化良好（cos 0.85–0.96） | 数据审计结论（重复、泄漏、风格、属性匹配）：**还没做** |

---

## 1. Data 线（neighbor 数据）

### 1.1 数据是什么，为什么要评测
- GPT 生成：每个被遗忘作者 15 个同属性邻居 × 20 QA；`neighbor10` 含 20 个作者（181–200）共 6000 行，`neighbor05/01` 取自它。每条邻居 QA 与某条 forget QA 一一对应（只替换实体，问题模板相同）。
- 它在方法里承重：r_UV 的重定向目标、选层的 `unseen_gap`、W_down 的 retain 集合、评测（`neighbor_*`、探针、UD 的良性集）。审稿人会问"优势是不是靠额外数据"，所以数据本身要评测并展示。

### 1.2 已做
1. **结构/初查**（data_line/2）【稳】：6000 行、300 个邻居名互不重复；每作者 10 train + 5 heldout 邻居；答案比 TOFU 短约 20%（后来生成时补了 `tail` 句缓解）；邻居与 forget 答案词集合 Jaccard 均值 0.24，没发现 forget 作者姓名泄漏；少数通用型问题骨架近乎重复（最大 0.83）。
2. **发现并修复数据协议问题**（data_line/3）【稳】：旧代码没用 `split`/`train_rank`，r_UV、W_down 的 retain、评测用的是同一批邻居 → 邻居评测不干净。改为：r_UV 与 W_down 只用 `train`（按 `train_rank≤K` 取，K 嵌套），评测/探针/UD/选层的未见邻居只用 `heldout`。
3. **协议修正的效果**【稳】：NEAR 性能几乎不变（B：Prob 0.163→0.161，mia 0.552→0.526），修的是可信度；heldout 邻居 ROUGE 0.343（retain 0.358，full 0.378）= 对没见过的邻居无副作用。
4. **r_UV 对邻居选择的鲁棒性**【稳】：train 前 5 个对全部 15 个的 cos：0.94–0.99；train 10 个对 heldout 5 个的 cos：0.93–0.96（作者 199）、0.85–0.88（作者 200）。**可作为数据质量的证据**。作者 200 对子集更敏感（随机 5 对 5 的 cos 均值 0.75–0.80，最低 0.55），K 越大越稳。
5. **K 的消融**（near_line/9）【稳】：与 K=10 的 r_UV 余弦 K=1：0.73–0.81；K=3：0.93–0.95；K=5：0.97–0.98。建议主线 K=5。
6. **未编辑模型中 forget 与 neighbor 几乎重合**（near_line/13）【稳】：full/retain 在 l9，两者均值 cos=1.00、范数相同（4.8–4.9），`auc[forget|neighbor]` 0.33–0.46。即匹配得很好，r_UV 编码的是"实体身份"的小差别（范数仅 0.23–0.33）。
7. **配对规则**（data_line/4）：`(author_id, qa_id)` 配对，`source_question` 校验；`forget_idx` 是相对 forget10 的行号（forget01 要减 360）。

### 1.3 数据层面的局限（论文里要诚实写）
- 邻居是合成的，r_UV 的小范数意味着"邻居"与"forget"差别很小，c≈1 的 steering 无效（near_line/3），必须 c≥6 才有效；这是方法的内在张力，不是数据的错。
- 第一个信号：full 模型对 neighbor 的答案对数概率 −3.02，对 holdout −2.14（neighbor 更不可预测），但行为一致（ROUGE 0.39/0.38）。

### 1.4 【待做】
- 数据审计脚本 `neighbor_audit.py`（重复、与 TOFU 的重名/重复、泄漏、风格分类器 AUC、属性匹配的 LLM 小样本评审）。
- 数据消融：错配邻居、用 TOFU retain 作者当邻居、K=1/5/15、泄漏注入、LUNAR 参考集换成 neighbor（分离"用了邻居数据"与"按作者匹配"的贡献）。
- forget05/10 上用 `source_question`/`forget_idx` 配对（目前 KL 只在 forget01/forget005）。

---

## 2. Metric 线（可检测度 UD 与评测口径）

### 2.1 动机与参照
- 论文 Extracting Forgotten Prompts from Targeted Unlearned Models（arXiv 2609.03662）：纯黑盒 TAS 攻击，按实体×模板查询，用拒答分（56 条正则 + 与规范拒答库的嵌入余弦）+ Thompson 采样找出被遗忘的实体；LUNAR 精度最高（把遗忘重定向到"I don't know"区域，拒答干净）。论文自述：传统 forget 指标（ROUGE/概率）**不能预测**可发现性。
- 论文提出的防御方向（让被遗忘输入与普通输入行为不可区分）正是 NEAR 的定位。

### 2.2 我们实现的 UD（`src/evals/detect/`）
- 黑盒：`bb_tas`（TAS 拒答分）、`bb`（max(拒答分, 乱码, 近重复)）、`entity_bb`（按作者聚合，对应 TAS 聚合证据）。
- 白盒（我们的扩展）：`wb_sup`（监督线性探针，oracle 上界）、`wb_unsup`（只用良性样本拟合的 Mahalanobis 离群分，更接近真实攻击者）；报告相对 retain 模型的超出量。
- advantage = 2(AUC−0.5)；0 = 与瞎猜一样。汇总 `ud` 默认 = `bb`（论文威胁模型），`ud_max` 始终报告。主张：论文主指标用黑盒（给出 `entity_bb`），白盒作参考。
- 另有：重复度通道；`is_refusal` 与 TAS 正则一致率 99.96%（4.5 万条回答），TAS 对第三人称缺席的判断更准。

### 2.3 评测口径（与方法无关的修正）
- 邻居相关指标只用 heldout（见 §1）。
- 1B 自训 full 模型坏掉（checkpoint-0 就是乱码，与精度无关），**一律使用官方 full / retain99**（near_line/1）。
- forget005 = 作者 200（forget01 的 `train[20:]`）；没有 retain995，`forget_quality`/`privleak` 在 forget005 上不可用。
- 修复的评测 bug：hallucination_rate 恒为 1（text metric 互相看不到）；文本指标参数没传到函数（`get_func_params` 存成了 dict）；probe 实验没有 retain 日志导致没有 forget_quality；IdkDPO 缺 `retain_split`。

### 2.4 发现
1. **LUNAR 的痕迹**【稳】：默认设置遗忘整个 forget01：55% forget 回答是拒答（"illegal or harmful activities"），对良性问题也拒答 32%；`ud_bb` 0.207，按作者聚合 `entity_bb`=0.67–1.0；`model_utility` 0.269。一个作者（forget005）：拒答 30%，utility 0.449。
2. **NEAR 的黑盒痕迹≈0**【稳】：不拒答、不退化，`ud_bb` 0–0.08。
3. **黑盒指标的局限**【稳】：衡量"被遗忘 vs 其他"的**差异**；一个对所有问题都乱拒答的模型（LUNAR forget01）`ud_bb` 反而偏低，所以必须与 `model_utility`、benign 拒答率一起报。
4. **指标盲区：怪词**【稳】：B+ans c7 的 12/40 forget 回答以 "Basis to …" 开头（"Basil" 的近邻词），`ud_bb` 只有 0.046；answer_coeff 后变成 "Beneer/Benever"，仍抓不到。加 KL 后消失。首词集中度不适合做指标（forget 集只有两位作者，以作者名开头本来正常）。
5. **白盒**（near_line/13）：见 §3.4。

### 2.5 读指标时的注意（附录可写）
- `forget_quality`（KS 检验 p 值）在 n=40 时 D 只取 k/40，p 值是离散台阶（0.766/0.405/0.266/0.165/0.097/0.054/0.029/0.014…）；差 1 个样本位置就能让 p 在台阶间跳；**不要拿它给 forget01 上的配置排序**；建议报 D 或在 forget05/10 上报。
- 主要遗忘指标改用连续量：`forget_Q_A_Prob`（retain≈0.17）、`extraction_strength`（≈0.06）、`mia_loss`（≈0.48，0.5=不可区分；显著 >0.5 说明 forget 样本更"熟"）。
- 编辑后 `auc[forget|neighbor]` 升高是编辑痕迹，不是遗忘成功的指标。

### 2.6 【待做】
- 流畅度通道（用 full/retain 模型算每条回答的困惑度，与 retain 回答分布比较）。
- 白盒检测位置拓宽（全 prompt token 均值），再判断 footprint。
- baseline 的 UD（`tune.sh` 默认 `PROBE=0`）；Pareto 图（遗忘 vs utility vs `ud_bb` vs `wb_unsup`）。
- `is_refusal` 换成 TAS 正则或并列报告。

---

## 3. NEAR 方法线

### 3.1 方法（流水线）
r_UV（每个作者：K 个邻居问题的 eoi 位置平均激活 − forget 问题的平均激活，block_out）→ 选层（推理时 steering，`unseen_gap`；可选 `ud`、`footprint`）→ 训练 MLP down_proj（W_down）使 forget 提示的 MLP 输出平移 `coeff·r_UV`、retain/neighbor 保持不变（AdamW 或闭式解；`shift_mode` all/eoi；可选 `include_answer`、多层 `increment`+`sequential`、KL 第二阶段）→ 一次评测。

### 3.2 steering 阶段（near_line/2–3）【稳】
- eoi 模式、c=1：第 0–13 层完全无效，只有最后两层（偏浅，只改开头）有效；r_UV 太小（l11 范数 0.72，LUNAR 约 4.1 且系数 2）。
- `all` 模式有明确有效区间：c12/l6（`unseen_gap` 0.26，ROUGE 0.39，退化 0）、c8/l9、c7/l9（0.31，退化 0）；c10 在 l7–9 退化率 12–30%。回答变成通顺的编造/回避式回答。
- 选层依赖系数：各 c 选出的层 c2→12、c4→13、c5/c6→11、c7→9、c8→8、c12→6（选层分数排名第一与最终选出层不一致的原因**尚未查明**）。

### 3.3 W_down 训练（near_line/4–7, 9）
| 步骤 | 发现 | 证据级别 |
|---|---|---|
| 单层 c7/L9、c12/L6 | 遗忘开始有效但弱：Prob 0.42–0.54（retain 0.17），mia 0.85–0.93，privleak −75…−89；无拒答、无乱码 | 【稳】 |
| retain_weight 1→3→10 | 干净的权衡：retain 越保，遗忘越弱（forget_quality 0.40→0.10，位移幅度比 0.87→0.45） | 【稳】 |
| include_answer | retain 生成 ROUGE 0.636→0.760，遗忘基本不变；"答案 token 没训练到所以 MIA 高"的假设**不成立** | 【稳】 |
| **多层 + 顺序训练** | A(l8-9)、B(l7-9) 总位移同单层（`7·r_9`），B：Prob 0.163、mia 0.552、privleak −25.7（≈retain），utility 0.524；**位移分摊到更多层比加大系数更有效** | 【稳】 |
| 独立训练（C） | 遗忘"更强"（mia 0.468）但 utility 0.397、各层 achieved 余弦 −0.11/0.008：破坏模型换来的"遗忘"。`sequential` 是必须的 | 【稳】 |
| **B+答案 token（rw1）= 最佳配置** | Prob 0.198–0.206、mia 0.50–0.53、extraction 0.048；utility 0.551–0.569；retain 生成 ROUGE 0.474→0.606；`retain_weight`=3 几乎没有额外提升；新旧协议一致 | 【稳】 |
| 层数 | l7-9 > l8-9 > l9；c6 的 l5-9/l6-9 与 l7-9 相当 | 【稳】 |

### 3.4 白盒痕迹（near_line/9–13）
- 现象【稳】：NEAR 各配置 `wb_unsup` 0.95–0.996。LUNAR（forget01）c0.5/c1/c1.5/c2 = 0.41/0.82/0.57/0.57；forget005 = 0.61/0.84/0.71/0.96。**早先写的"LUNAR 0.41–0.57"漏了 c1，已更正**；c1.5/c2 的低值对应已被破坏的模型（utility 0.27–0.40）。遗忘同样弱的 LUNAR c1（Prob 0.52）与 NEAR c4（0.82–0.88）相当 → **白盒上两者没有本质差别**。
- 来源【初步：来自汇总数字，无干预】：`anom_auc` 只在被编辑层及其上方升高（NEAR l7-9：l1–6 为 0.44–0.46，l7/8/9 为 0.98/1.00/1.00，往上衰减）；编辑把 forget 与 neighbor 的均值差距**放大约 10 倍**（l9 r_UV 范数 0.33→3.09，forget 范数 6.19 对 neighbor 4.97，`auc[forget|neighbor]` 0.99）；c=7 把 forget 推到**邻居均值之外约 6 个 r_UV**；随强度单调（c4 l9：0.90；c7：1.00）。
- 内在张力：c≈1 时 forget 恰好落在邻居上（不可检测）但模型几乎无反应；c≥6 才起作用，而此时激活被推出分布。
- 降低尝试：系数降到 c4 才降（0.82–0.88），但遗忘随之消失；增加层数略有改善；KL 单层 l9 到 0.953。

### 3.5 答案 token 缩放与 KL 第二阶段（near_line/12）
- **answer_coeff**（0.5/0.3）【稳】：去掉 "Basis"，但遗忘一起丢（mia 0.92–0.96），怪词变成 "Beneer…"；原因是首 token 由 prompt 最后位置的隐状态预测，该位置仍带完整 c7 偏移，answer_coeff 碰不到。"Basis" 出现有阈值：c4 为 0，c5 为 1–2，c6 起 10–12。
- **KL 第二阶段**（`kl_refine.py`）：阶段 1 局部回归之后，联合训练选中层的 W_down，损失 = forget/retain 相对 MSE + KL（编辑后模型对 forget 提示 + 邻居答案前缀的前 8 个 token 分布，逼近原模型对邻居提示的分布）；只需要原模型，不需要 retain 模型；配对 `(author_id, qa_id)` + `source_question` 断言。
  - 三层 l7-9：Prob 0.206→0.227，extraction 0.048→0.068，mia 0.531→0.724，utility 0.569→0.553，retain ROUGE 0.588→0.551，`ud_bb` 0.046→0，怪词消失，回答变成"另一位虚构作者"（"Ali Munther Al-Kandari was born in … Muscat, Oman."）【初步，单次】。
  - 单层 l9：Prob 0.457→0.355，mia 0.883→0.822（遗忘更强），`wb_sup` 0.718→0.64，`wb_unsup` 0.985→0.953，utility 0.568→0.553。
  - 代价：激活偏移保真度略降（achieved cos：l7 0.96，l8 0.82，l9 0.77）。

### 3.6 与 LUNAR / baseline 的对照（forget01，near_line/9, 12, 14）
- 匹配点"遗忘到 retain 水平"（Prob≈0.1–0.2，mia≈0.4–0.55）：NEAR B+ans（Prob 0.206，mia 0.531，utility **0.569**，retain ROUGE 0.588，`ud_bb` 0.046）；NEAR+KL（0.227，0.724，0.553，0.551，0.0）。
- 调过 lr 的 baseline（best epoch，lr {5e-5, 1e-4}，20 步）：IdkDPO 1e-4（0.132，0.461，utility 0.414，retain ROUGE 0.339）；RMU 1e-4 step2（0.152，0.414，0.412，0.285）；NPO 1e-4 step2（0.184，0.438，0.444，0.329）；SimNPO/GA/GradDiff 1e-4 遗忘不到位（Prob 0.27–0.31，mia 0.71–0.81）且 utility 0.43–0.48。全部没有满足"utility ≥ 95% 原模型（0.597）"。默认 lr 基本不遗忘。
- LUNAR（系数扫描）：c1.5 Prob 0.115，utility 0.396，`ud_bb` 0.52；c2 utility 0.269；forget005 c1.5/c2：utility 0.44，`ud_bb` 0.475/0.25；c0.5/c1 几乎不遗忘。
- 结论【方向稳，幅度需谨慎】：匹配遗忘程度时 NEAR 的 utility 高约 0.12–0.17，retain ROUGE 高约 0.12–0.30；baseline 的 UD 未测，调参预算有限（见 near_line/14）。

### 3.7 未解决 / 待做
1. **白盒痕迹**：小系数 + KL（在跑，`sweep_kl.sh`）；每个位置的离群分；训练时加"与 benign 分布距离"的惩罚；比较要按遗忘强度匹配。
2. 多层实验的层集一直固定为 7-9（未按各 c 重新选层）；需要在最优层上确认 KL 优势是否成立（c2 的 11-12 在跑）。
3. 选层标准的疑点（排名第一的层 ≠ 选出的层）。
4. 多种子、forget05/10、forget005（作者 200）上的 NEAR、3B/8B 模型。
5. UD 的流畅度通道；Pareto 图；对 baseline 的 UD。

---

## 4. 论文里最值得展示的图表（建议）
1. 方法流水线图 + steering 有效区间热力图（系数 × 层，`unseen_gap`/退化率）。
2. 多层分摊 ablation 表：单层 / A / B / C（独立 vs 顺序）→ 说明 `increment` 和 `sequential` 的必要性。
3. 主对比表：NEAR(+KL) / LUNAR(调参) / baselines(调参)：Prob、extraction、mia、utility、retain ROUGE、拒答率、`ud_bb`、`entity_bb`、`wb_unsup`。
4. Pareto：遗忘强度 vs utility；遗忘强度 vs UD（黑盒与白盒各一张）。
5. 数据：K 消融（cos vs K）、train/heldout r_UV 余弦、heldout 邻居上的无副作用、数据审计与消融（待做）。
6. 附录：`forget_quality` 在 n=40 的离散性；"Basis"/怪词案例；LUNAR 拒答案例；指标定义。

## 5. 复现信息
- 输出：`saves/NEAR/single_v2/<变体>/`，评测 `saves/eval/NEAR/single_v2/<变体>_k5_l<层>_c<系数>_all/`；r_uv 与选层在 `.../near_tofu_Llama-3.2-1B-Instruct_forget01[_ks]/`（`_ks` 含 K 消融与各系数选层）。
- 脚本：`scripts/NEAR/{single_run,sweep_select,sweep_fine,sweep_w_down,sweep_footprint,sweep_kl}.sh`，`scripts/LUNAR/{single_run,sweep_coeff}.sh`，`scripts/unlearn/tune.sh`，`scripts/eval/baseline_probe.sh`，`scripts/analysis/{baseline_table,detectability_table}.py`。
