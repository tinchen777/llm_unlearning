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
LAYERS='[10]'

# Train W_down (MLP output projection) of LAYERS with the r_UV of each K (needs 3_r_uv.sh with the same task_name):
#   -> w_down_k<K>.pt              edited weights {layer: W_down}
#   -> w_down_k<K>_summary.json    losses, and achieved-vs-intended shift of the forget activations
#   -> responses_before_<split>.json / responses_after_k<K>_<split>.json   same fixed samples, to read
#   (settings: `wdown` in configs/experiment/custom/neighbor_probe.yaml; add wdown.save_model=true to keep the model)
for K in 1 5 15; do
  echo start w_down K=${K} ${MODEL}

  python src/w_down.py --config-name=train \
    experiment=custom/neighbor_probe \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    neighbor_split=${NEIGHBOR_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    wdown.k=${K} \
    "wdown.layers=${LAYERS}" \
    task_name=test/redirect_${MODEL}_${FORGET_SPLIT}
    # wdown.coeff=1.0 \
    # wdown.shift_mode=eoi \
    # wdown.lr=1.0e-3 \
    # --cfg job --resolve

  echo end w_down K=${K} ${MODEL}
done
