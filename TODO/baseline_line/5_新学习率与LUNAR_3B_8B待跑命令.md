# 5 baseline 新学习率重跑、LUNAR 3B / 8B 待跑命令

> 写于 baseline 重跑运行期间。命令前先 `git pull`，确认本机有提交 `04738e4`（`EXTRA_TRAIN`）之后的代码。

## 问题
- 5-epoch 协议下 lr 1e-4 对 GradAscent、GradDiff 偏大：1B 与 8B 的 GradAscent 第 2 步 Prob 0.158 / 0.118、utility 0.428 / 0.509，第 4 步起 Prob 0、utility 0；8B 的 GradDiff 第 2 步 utility 就只有 0.036。每个 epoch 才评一次，第一个评测点已越过"遗忘到位"。旧的 20 步数据里这几个方法的最佳点在 5e-5（GradAscent、GradDiff、RMU），NPO、IdkDPO 在 1e-4。
- 决定：时间紧，不做学习率网格；**按方法分组、按规模降低**，只跑一次。旧的 lr1e-4 运行保留（目录名带 `lr1e-4`，不会被覆盖）。

## A. baseline（已在运行；若需要重启，用下面的命令）
先停掉用 lr1e-4 跑 GradAscent / GradDiff / RMU 的进程；NPO、SimNPO、IdkDPO 的 1e-4（1B / 3B）可以保留。每个模型一个进程，先跑一组再跑另一组：
```
# 1B
LRS="5e-5" METHODS="GradAscent GradDiff RMU" MODEL=Llama-3.2-1B-Instruct bash scripts/unlearn/tune.sh <GPU> ; \
LRS="1e-4" METHODS="NPO SimNPO IdkDPO"       MODEL=Llama-3.2-1B-Instruct bash scripts/unlearn/tune.sh <GPU>

# 3B
LRS="5e-5" METHODS="GradAscent GradDiff RMU" MODEL=Llama-3.2-3B-Instruct bash scripts/unlearn/tune.sh <GPU> ; \
LRS="1e-4" METHODS="NPO SimNPO IdkDPO"       MODEL=Llama-3.2-3B-Instruct bash scripts/unlearn/tune.sh <GPU>

# 8B（学习率是按规模推的，未验证）
LRS="2e-5" METHODS="GradAscent GradDiff RMU" MODEL=Llama-3.1-8B-Instruct bash scripts/unlearn/tune.sh <GPU> ; \
LRS="5e-5" METHODS="NPO SimNPO IdkDPO"       MODEL=Llama-3.1-8B-Instruct bash scripts/unlearn/tune.sh <GPU>
```
默认：5 个 epoch（输出目录带 `_ep5`）、`PROBE=1`、`EVAL_PROFILE=fast`（每个 epoch 带 UD）。输出：`saves/tofu_unlearn/<方法>/tofu_<模型>_forget01_lr<lr>_ep5/`。

**先看 1B**：GradAscent、GradDiff 在 5e-5 下是否还是第 4 步崩溃；若还崩，3B / 8B 的值要再降一档，不要等 8B 跑完。

**8B 显存**（见 `paper_notes/3_讨论纪要与阶段结论.md` §8）：NPO、IdkDPO、RMU 需要参照模型。先让不需要参照模型的跑；这三个若 OOM，加：
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EXTRA_TRAIN="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true" \
LRS="5e-5" METHODS="NPO IdkDPO RMU" MODEL=Llama-3.1-8B-Instruct bash scripts/unlearn/tune.sh <GPU>
```
（有效批大小不变，仍是 8×4=32。仍装不下就在论文里写"8B 上 baseline 只评测 GradAscent、GradDiff、SimNPO"。）

**判断显存是否真的有问题**：`nvidia-smi` 的总量持平不是泄漏（PyTorch 缓存分配器不还显存；分页优化器的统一内存也不计入单进程数字）；持续增长才是。

## B. LUNAR（与 NEAR 的实验无关，可以随时跑）
```
# 1B 补两个中间系数（填补 c1 与 c1.5 之间的空白，很快，先跑）
COEFFS="1.25 1.75" FORGET_SPLITS="forget01" bash scripts/LUNAR/sweep_coeff.sh <GPU>

# 3B
MODEL=Llama-3.2-3B-Instruct COEFFS="1 1.5 2" FORGET_SPLITS="forget01" bash scripts/LUNAR/sweep_coeff.sh <GPU>

# 8B（显存未验证，估计只训练一个投影矩阵，远比全参数微调轻）
MODEL=Llama-3.1-8B-Instruct COEFFS="1 1.5 2" FORGET_SPLITS="forget01" bash scripts/LUNAR/sweep_coeff.sh <GPU>
```
脚本自己先做第 1 步（r_UV），再对每个系数做选层、W_down 训练和评测（与 NEAR 相同的评测，含 UD）。系数 0.5 去掉（1B 上几乎不遗忘）。可选（需要先新增邻居参考集配置，还没写）：LUNAR 的参考集换成邻居，用来区分"用了邻居数据"与"按作者匹配"各自的贡献。

## C. 出结果后
```
python scripts/analysis/pareto.py --out saves/analysis/pareto                       # 1B：默认只含 5-epoch 的 baseline
python scripts/analysis/pareto.py --model Llama-3.2-3B-Instruct --near saves/eval/NEAR/Llama3B_v2 --out saves/analysis/pareto_3B
```
- baseline 要看每个方法：第 2、4、6、8、10 步的 Prob / utility / mia 是否形成一条轨迹，而不是只有 1 个有意义的点。
- 然后更新 `paper_notes/1_*` 的 §3.6 与 `paper_notes/3_*` 的 §8、§11。
