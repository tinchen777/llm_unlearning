#!/bin/bash

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/_common.sh"
cd "${SCRIPT_DIR}/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Step 3: train W_down of the selected layer (wdown.layers=auto -> the layer of the selection file of the SAME coeff /
# shift mode written by 2_select_layer.sh) with the r_UV of 1_r_uv.sh:
#   -> <config folder>/l<layer>[_cf]/model/             edited model, evaluated by 4_eval.sh
#   -> <config folder>/l<layer>[_cf]/w_down.pt          edited weights {layer: W_down}
#   -> <config folder>/l<layer>[_cf]/w_down_summary.json   losses, achieved-vs-intended shift, response statistics before / after
#   -> <config folder>/l<layer>[_cf]/responses_before_<split>.json / responses_after_<split>.json   same fixed samples, to read
#   config folder = saves/LUNAR/<TASK_NAME>/c<coeff>_<mode>/ (the one of the layer selection), `_cf` = closed form
# Shift modes: `all` = the paper (the same r_UV on every token of a forget prompt), `eoi` = end-of-instruction tokens only.
# SOLVER (in _common.sh): adam (AdamW, lr / epochs from the config) or closed_form (LUNAR Eq. 9, no lr / epochs).
for MODE in "${SHIFT_MODES[@]}"; do
  echo start w_down mode=${MODE} solver=${SOLVER} ${FULL_MODEL}
  python src/METHOD/LUNAR/w_down.py \
    experiment=unlearn/tofu/LUNAR \
    model=${FULL_MODEL} \
    "${SPLIT_ARGS[@]}" \
    "data/datasets@data.ref.datasets=${REF_DATASET}" \
    wdown.layer_selection_path=saves/LUNAR/${TASK_NAME}/$(config_dir ${MODE})/layer_selection.json \
    wdown.solver=${SOLVER} \
    wdown.layers=auto \
    task_name=${TASK_NAME}
  echo end w_down mode=${MODE}
done
