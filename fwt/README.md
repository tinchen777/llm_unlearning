# Forgetting Without Telling (FWT)

> 把被遗忘实体的激活重定向到**属性匹配的陌生邻居**，使其与天生陌生的实体不可区分。
> 对应 `outline..md` 里的方法章节（属性匹配邻居生成 / 重定向目标与训练）。

这个目录是**自包含**的：`src/`、`configs/`、`scripts/` 一个字都没改。
新的 trainer 和 dataset 在 `fwt/register.py` 里注册进项目已有的注册表，
新的 hydra 配置通过 search-path 插件挂到项目原有的 `configs/` 旁边。

```
fwt/
├── neighbors/        阶段 1：属性抽取 → 邻居合成 → 未知性验证 → 问题改写
├── layers/           阶段 2：激活提取、诊断指标、层打分
├── data/             NeighborRedirectDataset（带邻居的 forget 数据集）
├── trainer/          NeighborRedirect（重定向 trainer）+ 三项损失
├── configs/          hydra 配置（trainer / dataset / experiment）
├── scripts/          三个阶段的命令行入口 + run_fwt.sh 全流程
├── tests/            CPU 上可跑的单测与冒烟测试（不需要联网、不需要 GPU）
├── register.py       把 FWT 组件注册进 src 的注册表
└── train_fwt.py      训练入口（内部直接复用 src/train.py）
```

一条命令跑通三个阶段：`bash fwt/scripts/run_fwt.sh 0`（参数是 GPU id）。

---

## 阶段 1：属性匹配邻居数据

```bash
# 离线（不需要 API、不需要模型）
python fwt/scripts/build_neighbors.py \
  --forget-split forget10 --num-neighbors 5 --block-size 20 \
  --out data/fwt/neighbors_forget10.json

# 论文版：外部 LLM 生成 + 遗忘前模型做未知性验证
export FWT_LLM_BASE_URL=... FWT_LLM_API_KEY=...
python fwt/scripts/build_neighbors.py \
  --forget-split forget10 --num-neighbors 5 --block-size 20 \
  --backend llm --llm-model gpt-4o-mini --attribute-backend llm \
  --probe-model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --out data/fwt/neighbors_forget10.json \
  --report saves/fwt/neighbors_forget10_report.json
```

四步与 outline 一一对应：

| outline 步骤 | 实现 | 说明 |
| --- | --- | --- |
| 属性抽取 | `neighbors/attributes.py` | 正则启发式（离线、确定性）或 LLM 抽取，LLM 缺字段时用启发式回填 |
| 邻居合成 | `neighbors/generate.py` | `template`（姓名/城市池，离线）或 `llm`；LLM 失败自动回退，流程不会卡住 |
| 未知性验证 | `neighbors/verify.py` | ① 姓名相似度 + 同姓拦截；② 用遗忘前模型探测，会回答的名字直接淘汰 |
| 问题构造 | `neighbors/questions.py` | 只替换**身份特异**字符串（姓名/城市/生日/书名/奖项），匹配属性（国籍/年代/体裁/职业/父母职业）原样保留 |

产物 `neighbors.json` 的结构（`neighbors/schema.py`）：

```jsonc
{
  "meta": { "num_neighbors": 5, "backend": "llm", "stats": { "rejection_rate": 0.18, ... } },
  "entities": [{ "entity_id": "e000", "name": "...", "attributes": {...},
                 "question_indices": [...],
                 "neighbors": [{ "name": "...", "attributes": {...},
                                 "substitutions": {"<原名>": "<邻居名>", ...},
                                 "verification": {...}, "passed": true }] }],
  "questions": [{ "index": 0, "entity_id": "e000", "question": "...",
                  "neighbor_questions": ["...", ...], "rewritten": [true, ...] }]
}
```

`meta.stats` 就是附录里要报的统计（M 值、候选数、拒绝率、改写率）；
每个邻居的 `verification` 保留了姓名相似度和模型探测的原始输出，便于人工抽检。

一个细节值得记进论文：TOFU 里有一类问题（"born in X on Y 的作者全名是什么？"）
根本不提名字，改写靠的是**属性替换**而不是换名——城市只在**同一国家内**替换，
否则会破坏国籍匹配。`rewritten` 标记了哪些问题真的被个性化了。

**D1 前置检查**（"模型对陌生作者的回答是否自然多样"）直接复用探测器：

```python
from fwt.neighbors import FamiliarityProbe
probe = FamiliarityProbe(model, tokenizer)
known   = probe.probe_names(bank.entity_names())     # 被遗忘的实体
unknown = probe.probe_names(bank.neighbor_names())   # 生成的陌生人
# 比较 uncertainty_rate / name_nll 两个分布
```

## 阶段 2：层选择

```bash
python fwt/scripts/select_layers.py \
  --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --neighbors data/fwt/neighbors_forget10.json \
  --retain-dataset locuslab/TOFU --retain-split retain90 \
  --num-questions 128 --top-k 3 \
  --out saves/fwt/layer_selection_1B_forget10.json
```

四个打分维度（`layers/metrics.py`，归一化后按权重合成，见 `DEFAULT_CRITERIA`）：

| 指标 | 方向 | 含义 |
| --- | --- | --- |
| `known_unknown_auc` | 越高越好 | 线性探针能否在该层区分"已知实体 vs 陌生实体"。这正是攻击者读的那根轴，也是我们要把遗忘实体推过去的轴 |
| `redirect_cost` | 越低越好 | `‖h_forget − h_neighbor‖ / ‖h_forget‖`，重定向要移动多远。太远意味着改动大、模型容易崩 |
| `neighbor_cohesion` | 越高越好 | 同一实体的 M 个邻居是否聚在一起。太散说明"均值目标"是张平均脸，不是任何真实陌生人所在的位置 |
| `retain_separation` | 越高越好 | forget 与 retain 的质心间隔（按 retain 的离散度归一）。间隔大，编辑才不会连累保留实体 |

输出 JSON 里既有 `selected_layers`（top-k，直接喂给训练），也有每层的完整
`diagnostics`，可以直接画成附录里的逐层曲线。注意 AUC 用的是交叉验证的线性探针，
和论文的可识别性攻击同源——层选择本身就是一次小规模的攻击模拟。

## 阶段 3：训练

```bash
python fwt/train_fwt.py --config-name=unlearn \
  experiment=unlearn/tofu/fwt \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget10 retain_split=retain90 holdout_split=holdout10 \
  neighbors_path=data/fwt/neighbors_forget10.json \
  trainer.method_args.layers="[7]" \
  task_name=fwt/redirect_forget10_L7
```

`fwt/train_fwt.py` 只是注册组件后转交 `src/train.py`，所以原有的 evaluator、
日志、可视化、`--cfg job` 等全部照旧可用。

损失（`trainer/redirect.py`）：

```
L = gamma * L_redirect + alpha * L_retain + contrastive_weight * L_contrast
```

* `L_redirect`：选定位置的激活与邻居目标的 MSE（或 cosine）。
* `L_retain`：`NLL`（默认）/ `KL` / `EMBED_DIFF`（保留集激活相对原模型不变）。
* `L_contrast`：InfoNCE，正样本是邻居目标，负样本是**该实体自己在原模型里的激活**
  （`self_ref`）以及 batch 内其它实体的目标（`in_batch`）。`contrastive_weight=0`
  就是消融对照，用来检验 LUNAR "对比特征不是必要条件"的论断在隐私维度是否还成立。

只更新选定层的 `mlp.down_proj.weight`（与 LUNAR 一致，`trainable_params_regex`
可覆盖），推理时结构和开销完全不变。测试里验证过：训练后**只有**这些权重发生变化。

### 关键开关（`trainer.method_args.*`）

| 参数 | 取值 | 对应 outline 的消融 |
| --- | --- | --- |
| `layers` | `[7]` / `[5,6,7]` | 单层 vs 多层修改 |
| `target_mode` | `mean` / `sample` | 均值目标 vs 随机采样目标 |
| `target_source` | `precompute` / `online` | 目标激活缓存一次 vs 每步重算（两者数值等价，测试里断言过） |
| `redirect_loss_type` | `mse` / `cosine` | MSE 回归 vs 纯方向 |
| `contrastive_weight` | `0.0` / `>0` | MSE vs MSE+对比 |
| `retain_loss_type` | `NLL` / `KL` / `EMBED_DIFF` | 保留正则的形式 |
| `position_strategy` | `last_prompt` / `prompt_mean` / `answer` / `last` / `all` | 位置选择（默认沿用 LUNAR 的读出位置） |
| `match_target_norm` | `false` / `true` | 是否只改方向、保住激活模长 |

邻居数量 M 的消融在数据侧：
`data.forget.TOFU_QA_forget_neighbor.args.num_neighbors=3`。

现成的变体实验：`experiment=unlearn/tofu/fwt_contrastive`（加对比项）、
`experiment=unlearn/tofu/fwt_online`（在线目标 + 随机采样）。

### 实现上的两个要点

* **目标是"同一个问题问陌生人"**：邻居样本按 *prompt + 空答案* 切词，
  所以 `last_prompt` 在遗忘侧和邻居侧落在同一个相对位置（assistant header 的最后一个 token），
  也因此不需要给 batch 加任何新字段，项目原有的 collator 不用动。
* **靠行号取目标**：实验配置里换成了 `DataCollatorForSupervisedDatasetwithIndex`，
  trainer 用行号去查缓存好的邻居激活；bank 覆盖不到当前 split 时会直接报错，
  不会静默地拿错目标。

## 测试

不需要 GPU、不需要联网（用的是临时搭的 4 层随机 Llama 和自建 tokenizer）：

```bash
python fwt/tests/test_neighbors.py     # 属性抽取、邻居合成、验证、问题改写
python fwt/tests/test_activations.py   # 位置选择与池化
python fwt/tests/test_layers.py        # 层打分 + 端到端激活提取
python fwt/tests/test_trainer.py       # 数据集、目标缓存、训练收敛、只动 down_proj
python fwt/tests/test_config.py        # hydra 配置能否组合
# 或者 pytest fwt/tests
```

`test_trainer.py` / `test_config.py` 会在缺少 `deepspeed`、`lm_eval` 时自动打桩
（只在测试里打桩，`fwt/` 本身不碰这两个包）。

## 还没做的部分

按 outline 的排期，这里只覆盖 D2（邻居数据）、D5–D6（方法实现与层选择）。
**可识别性评估（D4 的三种攻击）尚未实现**——`layers/metrics.py` 里的
`probe_auc` 可以直接复用成监督分类攻击的打分函数，但排序攻击、遗忘 vs 陌生判别、
以及接入 OpenUnlearning 的评测汇总还需要单独写一版 evaluator。
