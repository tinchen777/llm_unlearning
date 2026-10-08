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
# of the SAME K / coeff / shift mode written by 4_select_layer.sh) with the r_UV of each K (needs 3_r_uv.sh):
#   -> model_<tag>/                     edited model (tag = k<K>_l<layer>_c<coeff>_<mode>[_cf]), evaluated by 6_eval.sh
#   -> w_down_<tag>.pt                  edited weights {layer: W_down}
#   -> w_down_<tag>_summary.json        losses, achieved-vs-intended shift, response statistics before / after
#   -> responses_before_<split>.json / responses_after_<tag>_<split>.json   same fixed samples, to read
# Shift modes: `all` = the paper (the same r_UV on every token of a forget prompt), `eoi` = end-of-instruction tokens only.
# SOLVER (in _common.sh): adam (AdamW, lr / epochs from the config) or closed_form (LUNAR Eq. 9, no lr / epochs).
# (settings: `wdown` in configs/experiment/generate/neighbor_probe.yaml)
for MODE in "${SHIFT_MODES[@]}"; do
  for K in "${KS[@]}"; do
    echo start w_down K=${K} mode=${MODE} solver=${SOLVER} ${FULL_MODEL}
    python src/w_down.py \
      experiment=generate/neighbor_probe \
      model=${FULL_MODEL} \
      ${SPLITS} \
      wdown.k=${K} \
      wdown.coeff=${COEFF} \
      wdown.shift_mode=${MODE} \
      wdown.solver=${SOLVER} \
      wdown.layers=auto \
      task_name=${TASK_NAME}
      # "wdown.layers=[10]" \
      # wdown.lr=5.0e-3 \
      # wdown.ridge=1.0e-3 \
    echo end w_down K=${K} mode=${MODE}
  done
done
