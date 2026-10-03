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
RETAIN_SPLIT=retain99

# 3 question sets (forget / neighbor / retain) through the model:
#   -> responses_<split>.json   generated answers (+ ROUGE vs ground truth) to read
#   -> activations.pt           per-layer mean activations of every split + r_UV
#   -> activations_summary.json per-layer ||r_UV|| and cosines between the splits
# output: saves/neighbor/<task_name>/   (see configs/experiment/custom/neighbor_probe.yaml)
echo start neighbor ${MODEL}

python src/neighbor.py --config-name=train \
  experiment=custom/neighbor_probe \
  model=${MODEL} \
  forget_split=${FORGET_SPLIT} \
  neighbor_split=${NEIGHBOR_SPLIT} \
  retain_split=${RETAIN_SPLIT} \
  task_name=test/neighbor_${MODEL}_${NEIGHBOR_SPLIT}
  # probe.max_gen_samples=null \
  # probe.point=mlp_out \
  # --cfg job --resolve

echo end neighbor ${MODEL}
