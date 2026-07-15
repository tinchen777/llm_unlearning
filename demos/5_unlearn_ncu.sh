#!/bin/bash
# =============================================================================
# Demo 5: NCU (Neighborhood-Contrastive Unlearning) — 自研算法
# -----------------------------------------------------------------------------
# 核心思想 (src/trainer/unlearn/ncu.py):
#   1. 用冻结的参考模型, 预先缓存 forget/retain 全部样本在第 layer_id 层的
#      表征 (answer token 平均池化), 并为每条 forget 样本检索 top-k 最近的
#      retain 样本 —— 即"最近合法概念"原型 (要忘哈利波特的情节, 但邻域中
#      关于作者/奇幻文学的知识是合法的, 不能忘)。
#   2. forget-pull (InfoNCE): 把当前模型的 forget 表征拉向合法概念原型,
#      推离自己原本的参考表征 (+ batch 内其他 forget 表征作为负例)。
#      —— 不同于 RMU 推向随机噪声, 这里是"塌缩到最近的合法邻域"。
#   3. neighborhood-anchor (InfoNCE/MSE): 把 retain 表征锚定在参考表征上,
#      并推离 batch 内的 forget 表征, 稳住被拉过来的邻域本身。
#   4. retain LM loss (NLL/KL): 保持 retain 数据上的生成能力。
#   loss = gamma * forget_pull + beta * anchor + alpha * retain_lm
# =============================================================================
set -e
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

MODEL=Llama-3.2-1B-Instruct

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  model=${MODEL} \
  trainer=NCU \
  trainer.method_args.gamma=1.0 \
  trainer.method_args.beta=1.0 \
  trainer.method_args.alpha=1.0 \
  trainer.method_args.tau=0.1 \
  trainer.method_args.layer_id=7 \
  trainer.method_args.num_neighbors=5 \
  forget_split=forget10 \
  retain_split=retain90 \
  holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json \
  task_name=demo_unlearn_NCU
