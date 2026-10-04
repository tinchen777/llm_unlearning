#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# keep these identical in 1_probe / 3_r_uv / 4_select_layer / 5_w_down / 6_eval (3-5 share TASK_NAME)
MODEL=Llama-3.2-1B-Instruct
FORGET_SPLIT=forget01
NEIGHBOR_SPLIT=neighbor01
RETAIN_SPLIT=retain99
HOLDOUT_SPLIT=holdout01
SPLITS="forget_split=${FORGET_SPLIT} neighbor_split=${NEIGHBOR_SPLIT} retain_split=${RETAIN_SPLIT} holdout_split=${HOLDOUT_SPLIT}"
TASK_NAME=test/redirect_${MODEL}_${FORGET_SPLIT}

# Step 3: train W_down (MLP output projection) of the selected layer (wdown.layers=auto -> layer_selection.json of
# 4_select_layer.sh) with the r_UV of each K (needs 3_r_uv.sh; select the layer with the same k / coeff):
#   -> model_<tag>/                     edited model (tag = k<K>_l<layers>_c<coeff>), evaluated by 6_eval.sh
#   -> w_down_<tag>.pt                  edited weights {layer: W_down}
#   -> w_down_<tag>_summary.json        losses, achieved-vs-intended shift, response statistics before / after
#   -> responses_before_<split>.json / responses_after_<tag>_<split>.json   same fixed samples, to read
# (settings: `wdown` in configs/experiment/generate/neighbor_probe.yaml)
for K in 1 5 15; do
  echo start w_down K=${K} ${MODEL}
  python src/w_down.py \
    experiment=generate/neighbor_probe \
    model=${MODEL} \
    target_model=full \
    ${SPLITS} \
    wdown.k=${K} \
    wdown.coeff=1.0 \
    wdown.layers=auto \
    task_name=${TASK_NAME}
    # "wdown.layers=[10]" \
    # wdown.shift_mode=eoi \
    # wdown.lr=1.0e-3 \
  echo end w_down K=${K} ${MODEL}
done
