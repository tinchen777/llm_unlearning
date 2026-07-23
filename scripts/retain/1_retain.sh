#!/bin/bash
# =============================================================================
# Demo 2-1: 微调 retain 参照模型 (GPU 0: Qwen2.5-1.5B-Instruct + phi-1_5)
# -----------------------------------------------------------------------------
# retain 模型 = 只在 retain 数据上微调的"金标准"模型, 即"从未见过 forget 数据"
# 的理想遗忘结果。它的评测结果 (TOFU_EVAL.json) 之后作为 unlearn 实验的
# retain_logs_path 参照, 用于计算 forget_quality (ks_test) 和 privleak。
#
# TOFU 的 split 三元组对应关系 (与上游 OpenUnlearning 一致):
#   forget01 <-> holdout01 <-> retain99   (遗忘 1%)
#   forget05 <-> holdout05 <-> retain95   (遗忘 5%)
#   forget10 <-> holdout10 <-> retain90   (遗忘 10%)
#
# 关键覆盖项 (相对 experiment=finetune/tofu/default, 其默认训练 full split):
#   data/datasets@data.train=TOFU_QA_retain          换训练集为 retain
#   data.train.TOFU_QA_retain.args.hf_args.name=...  指定 retain90/95/99
#   forget_split / holdout_split                     让评测对准对应的 forget 组
#   retain_logs_path 保持 null: 本模型自身就是参照, forget_quality 打印
#   warning 并记 None 属预期。
#
# 输出: saves/finetune/test/tofu_<MODEL>_<retain_split>/
#   - 最终权重在目录根部; 每个 epoch 的评测在 checkpoint-*/evals/TOFU_EVAL.json
#   - 之后跑 unlearn 时:
#     retain_logs_path=saves/finetune/test/tofu_<MODEL>_<retain_split>/checkpoint-<最后一步>/evals/TOFU_EVAL.json
# =============================================================================
set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODELS=(
  Qwen2.5-1.5B-Instruct
)

# "forget_split holdout_split retain_split"
SPLITS=(
  "forget10 holdout10 retain90"
  "forget05 holdout05 retain95"
  "forget01 holdout01 retain99"
)

for MODEL in "${MODELS[@]}"; do
  for split in "${SPLITS[@]}"; do
    read -r forget_split holdout_split retain_split <<< "${split}"
    echo "========== [retain] model=${MODEL} train=${retain_split} eval=${forget_split}/${holdout_split} =========="

    python src/train.py --config-name=train.yaml \
      experiment=finetune/tofu/default \
      model=${MODEL} \
      data/datasets@data.train=TOFU_QA_retain \
      data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
      forget_split=${forget_split} \
      holdout_split=${holdout_split} \
      trainer.args.eval_on_start=False \
      trainer.args.num_train_epochs=5 \
      task_name=test/tofu_${MODEL}_${retain_split}

    echo "========== [retain] done: ${MODEL} ${retain_split} =========="
  done
done

echo "all retain finetunes finished: ${MODELS[*]}"
