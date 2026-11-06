# 4 baseline 训练协议改为 5 个 epoch（以后以它为基准）

## 问题
之前的 baseline 训练 10 个 epoch（forget01 只有 40 条，每个 epoch 2 个优化步，共 20 步）。这是我们配置里的默认值，不是 OpenUnlearning 的设置。1B / 3B / 8B 以后都要跑，需要一个统一的、有依据的协议。另外 20 步对大学习率是浪费：多数方法的最佳点在第 2–8 步，之后 utility 持续掉（RMU 到第 12 步只剩 0.117），默认 lr 1e-5 却在 20 步里几乎不动。

## 做法（用户的决定）
- baseline 的统一协议：**5 个 epoch**（forget01 上 = 10 个优化步，每个 epoch 评测一次），所有模型规模相同；之前的 20 步（10 epoch）结果不删，但以后以 5 epoch 的结果为基准。
- `scripts/unlearn/_common.sh`：`EPOCHS` 默认 5，输出目录总是带 `_ep<EPOCHS>`（旧的 10-epoch 运行没有这个后缀，不会混淆）：`saves/tofu_unlearn/<METHOD>/tofu_<MODEL>_<FORGET>_lr<LR>_ep5[_fp32][_ud]/`。
- `scripts/unlearn/tune.sh`：默认 `PROBE=1`、`EVAL_PROFILE=fast`（每个 epoch 就带 UD；fast 配置没有 `forget_quality`，所以不能再用 `baseline_table.py` 选最佳 epoch，用 `pareto.py`）。
- `scripts/analysis/pareto.py` 和 `baseline_table.py` 新增 `--epochs`（默认 5：只取 `_ep5` 的运行；`all` 取全部，包括旧的 20 步运行）。`_ud` 重跑（`TAG=_ud`）仍会替换同名的原运行。
- 测试：`test_the_baseline_protocol_is_5_epochs_and_marked_in_the_folder_name`；pareto 的测试检查旧协议的运行默认不被收集。

## 结果
未跑。当前 `pareto.py` 默认（`--epochs 5`）下没有 baseline 点，等 5-epoch 的运行出来后才有；`--epochs all` 仍可看到旧的 20 步结果。

## 下一步（时间紧：不做学习率网格，固定学习率）
1. 1B forget01 先只跑 `LRS="1e-4"`，6 个方法各一个运行（每个 10 步）：
   `METHODS="GradAscent GradDiff" LRS="1e-4" bash scripts/unlearn/tune.sh 0`（NPO SimNPO → GPU 1，IdkDPO RMU → GPU 2）。
   依据：20 步的旧数据里 lr 1e-4 对多数方法在前 2–4 步就到了足够的遗忘强度；每个 epoch 都评测，选 epoch 本身就是隐式调参。
2. 看 `pareto.py`：只对明显没到位（SimNPO 之前没进入 retain 区域，可能要更大 lr）或过度遗忘（GradAscent、RMU，可能要 5e-5）的方法补第二个学习率。
3. 1B 定下后，3B / 8B **沿用同一学习率、每个方法只跑一个运行**，不再调参；论文里写"学习率在 1B 上选定，其余规模沿用"。风险：大模型的最优学习率通常更小，沿用可能偏强，靠选更早的 epoch 补救。
4. 方法自带的超参（NPO 的 β、GradDiff 的 retain 权重、RMU 的层和系数、IdkDPO 的 β）和 fp32 对照不做；论文里如实写 baseline 的调参预算（比 NEAR 小很多）。
