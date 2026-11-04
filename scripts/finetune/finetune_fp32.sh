
#!/bin/bash
# Finetune with FP32 model weights and BF16 mixed precision.
# Usage: bash scripts/finetune/finetune_fp32.sh [GPU_ID] [EPOCHS] [MODEL]

set -e
cd "$(dirname "$0")/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
EPOCHS=${2:-3}
MODEL=${3:-Llama-3.2-1B-Instruct}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

TASK=test/tofu_${MODEL}_ep${EPOCHS}_fp32

echo "GPU: ${GPU_ID} | Model: ${MODEL} | Epochs: ${EPOCHS} | Precision: fp32"
echo "################ ${TASK} ################"

python src/train.py \
  experiment=finetune/tofu/default \
  model=${MODEL} \
  model.pretrained.dtype=float32 \
  trainer.args.num_train_epochs=${EPOCHS} \
  task_name=${TASK}
