#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODEL=Llama-3.2-3B-Instruct
FORGET_SPLIT=forget01
NEIGHBOR_SPLIT=neighbor01

# responses on forget/reference questions + LUNAR r_UV for every layer
# (reference prompts: python setup/setup_data.py --lunar;
#  switch to harmful ones with `data/datasets@data.reference=LUNAR_harmful`)
echo start neighbor ${MODEL}

python src/neighbor.py --config-name=train \
  experiment=finetune/tofu/neighbor \
  model=${MODEL} \
  neighbor_split=${NEIGHBOR_SPLIT} \
  forget_split=${FORGET_SPLIT} \
  task_name=test/neighbor_${MODEL}_${NEIGHBOR_SPLIT} \
  # --cfg job --resolve

echo end neighbor ${MODEL}
