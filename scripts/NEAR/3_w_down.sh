#!/bin/bash

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/_common.sh"   # MODEL, splits, FULL_MODEL, TASK_NAME, KS, COEFF, SHIFT_MODES, SOLVER
cd "${SCRIPT_DIR}/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


# Step 3: train W_down (MLP output projection) of the selected layer (wdown.layers=auto -> the layer of the selection file
# of the SAME K / coeff / shift mode written by 2_select_layer.sh) with the r_UV of each K (needs 1_r_uv.sh):
#   -> <config folder>/l<layer>[_cf]/model/             edited model, evaluated by 4_eval.sh
#   -> <config folder>/l<layer>[_cf]/w_down.pt          edited weights {layer: W_down}
#   -> <config folder>/l<layer>[_cf]/w_down_summary.json   losses, achieved-vs-intended shift, response statistics before / after
#   -> <config folder>/l<layer>[_cf]/responses_before_<split>.json / responses_after_<split>.json   same fixed samples, to read
#   config folder = saves/NEAR/<TASK_NAME>/k<K>_c<coeff>_<mode>/ (the one of the layer selection), `_cf` = closed form
# Shift modes: `all` = the paper (the same r_UV on every token of a forget prompt), `eoi` = end-of-instruction tokens only.
# SOLVER (in _common.sh): adam (AdamW, lr / epochs from the config) or closed_form (LUNAR Eq. 9, no lr / epochs).
# (settings: `wdown` in configs/experiment/unlearn/tofu/NEAR.yaml; k / coeff / shift_mode / layer / r_uv.pt are read from the
#  layer_selection.json of 2_select_layer.sh, so nothing of them is passed here)
for MODE in "${SHIFT_MODES[@]}"; do
  for K in "${KS[@]}"; do
    echo start w_down K=${K} mode=${MODE} solver=${SOLVER} ${FULL_MODEL}
    python src/METHOD/NEAR/w_down.py \
      experiment=unlearn/tofu/NEAR \
      model=${FULL_MODEL} \
      ${SPLITS} \
      wdown.layer_selection_path=saves/NEAR/${TASK_NAME}/$(config_dir ${K} ${MODE})/layer_selection.json \
      wdown.solver=${SOLVER} \
      wdown.layers=auto \
      task_name=${TASK_NAME}
      # "wdown.layers=[10]" \
      # wdown.lr=5.0e-3 \
      # wdown.ridge=1.0e-3 \
    echo end w_down K=${K} mode=${MODE}
  done
done
