#!/bin/bash
# =============================================================================
# Demo 2-2: 微调 retain 参照模型 (GPU 1: phi-1_5)
# -----------------------------------------------------------------------------
# 与 demos/2_1_retain.sh 相同逻辑, 拆到第二张卡上并行跑 3B 模型。
# 说明见 2_1_retain.sh 头部注释。
# =============================================================================
set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODELS=(
  phi-1_5
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

    python src/train.py \
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
