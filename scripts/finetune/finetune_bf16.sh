
#!/bin/bash
# Finetune with BF16 model weights.
# Usage: bash scripts/finetune/finetune_bf16.sh [GPU_ID] [EPOCHS] [MODEL]

set -e
cd "$(dirname "$0")/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
EPOCHS=${2:-3}
MODEL=${3:-Llama-3.2-1B-Instruct}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

TASK=test/tofu_${MODEL}_ep${EPOCHS}_bf16

echo "GPU: ${GPU_ID} | Model: ${MODEL} | Epochs: ${EPOCHS} | Precision: bf16"
echo "################ ${TASK} ################"

python src/train.py \
  experiment=finetune/tofu/default \
  model=${MODEL} \
  trainer.args.num_train_epochs=${EPOCHS} \
  task_name=${TASK}

