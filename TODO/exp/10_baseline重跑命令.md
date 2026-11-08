# 10 baseline 学习率修正重跑：分开的单行命令（每条一个进程，互相独立）

> 对应 `exp/9` 的第 3 项。lr 是按已有轨迹推的（见 `exp/6` 第 3 节）；每个 epoch 都评测，跑完看轨迹，不合适再调。输出目录带 lr，与旧运行不会冲突。`EVAL_PROFILE=full`。
> 看结果：`python scripts/analysis/pareto.py --model <模型> --forget <划分> --near <该设置的 NEAR 目录> --out ...`。

## A. forget05 × 8B（大卡 70 GB；RMU 加了降显存参数）
**GA（lr 2e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.1-8B-Instruct LRS="2e-5" METHODS="GradAscent" bash scripts/unlearn/tune.sh <GPU>
```
**GD（lr 2e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.1-8B-Instruct LRS="2e-5" METHODS="GradDiff" bash scripts/unlearn/tune.sh <GPU>
```
**SimNPO（lr 5e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.1-8B-Instruct LRS="5e-5" METHODS="SimNPO" bash scripts/unlearn/tune.sh <GPU>
```
**RMU（lr 5e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_TRAIN="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true" LRS="5e-5" METHODS="RMU" bash scripts/unlearn/tune.sh <GPU>
```

## B. forget05 × 3B（40 GB 卡）
**GA（lr 2e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-3B-Instruct LRS="2e-5" METHODS="GradAscent" bash scripts/unlearn/tune.sh <GPU>
```
**GD（lr 2e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-3B-Instruct LRS="2e-5" METHODS="GradDiff" bash scripts/unlearn/tune.sh <GPU>
```
**NPO（lr 2e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-3B-Instruct LRS="2e-5" METHODS="NPO" bash scripts/unlearn/tune.sh <GPU>
```
**SimNPO（lr 1e-4）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-3B-Instruct LRS="1e-4" METHODS="SimNPO" bash scripts/unlearn/tune.sh <GPU>
```
**RMU（lr 5e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-3B-Instruct LRS="5e-5" METHODS="RMU" bash scripts/unlearn/tune.sh <GPU>
```

## C. forget05 × 1B（小卡，可 2–3 条共卡）
**GA（lr 1e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-1B-Instruct LRS="1e-5" METHODS="GradAscent" bash scripts/unlearn/tune.sh <GPU>
```
**GD（lr 1e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-1B-Instruct LRS="1e-5" METHODS="GradDiff" bash scripts/unlearn/tune.sh <GPU>
```
**NPO（lr 2e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-1B-Instruct LRS="2e-5" METHODS="NPO" bash scripts/unlearn/tune.sh <GPU>
```
**SimNPO（lr 1e-4）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-1B-Instruct LRS="1e-4" METHODS="SimNPO" bash scripts/unlearn/tune.sh <GPU>
```
**RMU（lr 5e-5）**
```
FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 EVAL_PROFILE=full MODEL=Llama-3.2-1B-Instruct LRS="5e-5" METHODS="RMU" bash scripts/unlearn/tune.sh <GPU>
```

## D. forget10 × 3B（若保留这个格子；40 GB 卡）
**GA（lr 1e-5）**
```
FORGET_SPLIT=forget10 HOLDOUT_SPLIT=holdout10 RETAIN_SPLIT=retain90 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 EVAL_PROFILE=full MODEL=Llama-3.2-3B-Instruct LRS="1e-5" METHODS="GradAscent" bash scripts/unlearn/tune.sh <GPU>
```
**NPO（lr 2e-5）**
```
FORGET_SPLIT=forget10 HOLDOUT_SPLIT=holdout10 RETAIN_SPLIT=retain90 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 EVAL_PROFILE=full MODEL=Llama-3.2-3B-Instruct LRS="2e-5" METHODS="NPO" bash scripts/unlearn/tune.sh <GPU>
```

## E. forget10 × 1B（选做；现有 GA 1e-5、NPO 2e-5 已有，补其余三个）
**GD（lr 1e-5）**
```
FORGET_SPLIT=forget10 HOLDOUT_SPLIT=holdout10 RETAIN_SPLIT=retain90 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 EVAL_PROFILE=full MODEL=Llama-3.2-1B-Instruct LRS="1e-5" METHODS="GradDiff" bash scripts/unlearn/tune.sh <GPU>
```
**SimNPO（lr 5e-5）**
```
FORGET_SPLIT=forget10 HOLDOUT_SPLIT=holdout10 RETAIN_SPLIT=retain90 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 EVAL_PROFILE=full MODEL=Llama-3.2-1B-Instruct LRS="5e-5" METHODS="SimNPO" bash scripts/unlearn/tune.sh <GPU>
```
**RMU（lr 5e-5）**
```
FORGET_SPLIT=forget10 HOLDOUT_SPLIT=holdout10 RETAIN_SPLIT=retain90 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 EVAL_PROFILE=full MODEL=Llama-3.2-1B-Instruct LRS="5e-5" METHODS="RMU" bash scripts/unlearn/tune.sh <GPU>
```

## 说明
- 8B 之外的 NPO / RMU 需要参照模型，3B 在 40 GB 卡上若 OOM，加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_TRAIN="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true"`（有效批不变）。
- 3B forget05 的 GA / GD 上次 lr 1e-5 到第 35 步 Prob 仍有 0.67–0.69（没遗忘），所以改 2e-5；若轨迹显示仍偏弱再试 3e-5。1B 的 GA / GD 上次 2e-5 在第 21 步崩溃，所以降到 1e-5。
- forget10 × 1B 现有的 GA（1e-5）、NPO（2e-5）已经 Prob ≈ 0.17–0.20，只是 mia 偏高（0.64–0.80），不一定是 lr 的问题，所以没有重跑。
