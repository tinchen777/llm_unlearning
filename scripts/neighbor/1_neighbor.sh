#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODEL=Llama-3.2-1B-Instruct
FORGET_SPLIT=forget01
NEIGHBOR_SPLIT=neighbor01
RETAIN_SPLIT=retain99

echo start neighbor ${MODEL}

python src/neighbor.py \
  experiment=generate/neighbor_probe \
  model=${MODEL} \
  model.pretrained.name_or_path=open-unlearning/tofu_${MODEL}_full \
  forget_split=${FORGET_SPLIT} \
  neighbor_split=${NEIGHBOR_SPLIT} \
  retain_split=${RETAIN_SPLIT} \
  task_name=test/neighbor_${MODEL}_${NEIGHBOR_SPLIT} \
  # --cfg job --resolve
  # probe.max_gen_samples=null \
  # probe.point=mlp_out \
  

echo end neighbor ${MODEL}
