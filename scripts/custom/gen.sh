#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODEL=Llama-3.2-3B-Instruct

python src/gen.py \
  experiment=finetune/tofu/default \
  model=${MODEL} \
  task_name=test/gen_${MODEL} \
  # trainer.args.gradient_checkpointing=true \

echo end finetune ${MODEL}
