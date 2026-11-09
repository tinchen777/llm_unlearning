# 17 ROCR 基线（Ye, Zhang, Wu, 2025）：原版与“只换目标”的变体

> 先 `git pull`。ROCR 的实现按官方仓库（https://github.com/Acruxos/ORT ，MIT 协议）的**发布代码**重写，不是按论文的描述：论文说单层秩一闭式解，发布代码是 MEMIT / AlphaEdit 式的多层岭回归。代码在 `src/METHOD/ROCR/rocr.py`，一键脚本 `scripts/exp/rocr.sh`（编辑 + 标准评测），测试 `tests/test_rocr.py`（6 个，已通过；没有在真实模型上跑过）。

## 一、做了什么（与论文 / 官方代码的对应）
对遗忘作者 i，在连续的 n 层（论文：Llama-3-8B 的第 4、5、6 层）上依次：

| 量 | 取法 |
|---|---|
| K（键） | 6 个聊天模板里 `down_proj` 的输入，在**名字最后一个 token**，对模板取平均 |
| z（目标） | 最后一个被改的层的**输出隐状态**，在目标名字的最后一个 token；沿用发布代码，只用第 0 个模板（`--target-templates mean` 为论文的平均） |
| cur_z | 同一层、遗忘名字在裸提示 `<名字>.` 里的隐状态（发布代码如此，与 z 的模板不同） |
| 更新 | `R = (z − cur_z) / (n − i)`；`Δᵀ = solve(P(KKᵀ+C)+L2·I, P K Rᵀ)`，L2=1；逐层、用当前模型重新取 K 和 cur_z |
| P | 通用文本里 `down_proj` 输入的均值二阶矩，取特征值 < 2e-2 的特征向量张成的投影 |

- `--target famous`：原版，所有遗忘作者重定向到真实知名人物（默认 Donald Trump）。
- `--target neighbor`：**只换目标**，每个作者用 `train_rank == 1` 的那个邻居的名字；层、K0、阈值、L2、模板全部不变。
- `--scale s`：给残差乘系数。s=1 是 ROCR；s>1 **不是 ROCR**，只作为额外一行，用来说明系数不够时的表现。

## 二、与官方不同之处（论文里要写明）
1. **K0 的数据**：官方是 2020-05-01 版 Wikipedia 的 10 万篇文章（脚本式数据集，现在的 `datasets` 读不了）；这里用 WikiText-103 的 2 万个段落（`--n-docs 20000`，`--max-len 512`；官方是 10 万篇，要对齐就加 `EXTRA_ROCR="--n-docs 100000"`，只是更慢），读取方式就是 `load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")`，离线时读本地缓存（`HF_DATASETS_OFFLINE=1`）。
2. **评测**：用本仓库的 TOFU 评测，不用 ORT 基准。
3. **层数**：Llama-3-8B 用 4-5-6（占 12–19%）；1B（16 层）取 2-3-4，3B（28 层）取 3-4-5，是按相对深度换算的。先做 1B 的层扫描，论文里写明怎么选。
4. **遗忘作者名字**：从遗忘集的 20 条问答里取出现在最多条里的大写词序列（`extract_author_name`）。**跑之前先看 `rocr_summary.json` 里的 `forget_names`**；若不对，用 `--forget-names <json>` 覆盖（通过 `EXTRA_ROCR`）。

## 三、前置检查（1 条，GPU 机器上）
```
HF_DATASETS_OFFLINE=1 python -c "from datasets import load_dataset; d=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1'); print(d)"
```
能打印出 train / validation / test 就行；若报找不到，去掉 `HF_DATASETS_OFFLINE=1` 联网跑一次。

## 四、命令（单行，可分卡；`<GPU>` 换成卡号）
`R_UV_PREFIX` 是 `saves/NEAR/` 下的目录名，评测里的激活探针读 `saves/NEAR/<prefix>/near_tofu_<模型>_<split>_ks/r_uv.pt`。目录名与已上传的 `saves/NEAR` 一致（上传的 saves 里没有 `.pt` 文件，所以 `r_uv.pt` 本身我没法核对，请在你的机器上 `ls` 一下）：1B forget01 `single_v2`，3B forget01 `Llama3B_v2`，8B forget01 `Llama8B_v2`，forget05 的是 `forget05_Llama{1,3,8}B_v2`，forget10 的是 `forget10_Llama{1,3,8}B_v2`。

**1B forget01：原版、只换目标、层扫描**
```
MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=famous bash scripts/exp/rocr.sh <GPU>
```
```
MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=neighbor bash scripts/exp/rocr.sh <GPU>
```
```
for L in "5 6 7" "8 9 10"; do MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="$L" TARGET=neighbor bash scripts/exp/rocr.sh <GPU>; done
```
```
for S in 3 8; do MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=$S bash scripts/exp/rocr.sh <GPU>; done
```

**3B forget01**
```
for T in famous neighbor; do MODEL=Llama-3.2-3B-Instruct R_UV_PREFIX=Llama3B_v2 LAYERS="3 4 5" TARGET=$T bash scripts/exp/rocr.sh <GPU>; done
```
**8B forget01**（P 的特征分解是 14336 × 14336，用 float64，需要约 2–3GB 显存；模型本身 bf16）
```
for T in famous neighbor; do MODEL=Llama-3.1-8B-Instruct R_UV_PREFIX=Llama8B_v2 LAYERS="4 5 6" TARGET=$T PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" bash scripts/exp/rocr.sh <GPU>; done
```
**forget05**（1B；3B、8B 同理，`R_UV_PREFIX` 换成 `forget05_Llama3B_v2`、`forget05_Llama8B_v2`，层换成 3B：`3 4 5`，8B：`4 5 6`）
```
for T in famous neighbor; do MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 LAYERS="2 3 4" TARGET=$T bash scripts/exp/rocr.sh <GPU>; done
```
P 矩阵按模型和层缓存在 `data/rocr/<模型>/P_layer<l>_t0.02_n20000.pt`，同一模型的第二次运行不再重算。

## 五、要读的结果
1. `saves/ROCR/<名字>/rocr_summary.json`：`forget_names`（作者名提取是否正确）、`target_names`、每层的 `z_error`、`update_norm`。
2. 标准评测：`forget_Q_A_Prob`、`mia_loss`、retain 的 ROUGE、model_utility，以及 `ud_bb`（黑盒）、`ud_wb_unsup`（白盒）。
3. 采样（`scripts/analysis/sample_consistency.py`）：重点是**跨问题同名占比**和**最常用名字占比**。ROCR 原版把所有问题都答成 Trump，这两项应接近 1（retain 的基线约 0.175）；采样一致性要写成“与 retain 的差距的绝对值”，偏高偏低都算。
4. 回答样例：同一个问题下 ROCR 原版、ROCR → neighbor、NEAR-Act、NEAR-Dist 的回答（取 `forget_responses.json`）。

## 六、预期与如何写
- **ROCR → neighbor 在 TOFU 上很可能遗忘不足**（系数固定为 1，名字隐状态一次到位；我们的 NEAR-Act 要 c≈7–14）。`SCALE=3、8` 的行用来说明这一点，不要把它说成 ROCR 的改进。
- 如果遗忘不足：写成“ROCR 在预训练知识（RWKU）上有效，在微调植入的事实（TOFU）上不一定”，并如实报告。
- 如果遗忘到位且黑盒痕迹大降：支持“问题出在重定向的目标”；白盒痕迹仍高则与 NEAR-Act 的结论一致。
- 它**不能替代 X1**（LUNAR 换目标）：X1 是另一个机制，两者在“目标和机制分离”的论证里并列。
