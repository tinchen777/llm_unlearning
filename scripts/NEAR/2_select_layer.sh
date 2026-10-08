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


# Step 2b: LUNAR-style layer selection (needs 3_r_uv.sh), once per (K, shift mode). For every candidate layer the forget
# prompts are redirected at inference time (block_out += coeff * r_UV[K][author], what a perfect W_down edit of that layer
# does) and compared with the unsteered model on unseen questions (neighbor / holdout):
#   -> layer_selection_k<K>_c<coeff>_<mode>.json
#        per layer: rougeL_recall, answer_logprob, refusal / degenerate rates, unseen_gap; `selected_layer` (min
#        unseen_gap among non-degenerate layers) and `ranking` (best first; LUNAR also edits the top-3). READ IT.
#   -> responses_steer_k<K>_c<coeff>_<mode>_l<l>.json   steered forget answers of every layer
# Candidates: select.layer_fraction=[0.4, 0.8] -> layers at 40-80 % of the depth (LUNAR's selected layers are at 50-75 %,
# appendix E); use 'select.layers=[4,5,6]' for an explicit list or select.layer_fraction=null for all layers.
# (settings: `select` in configs/experiment/generate/neighbor_probe.yaml; k / coeff / shift_mode follow `wdown`)
for MODE in "${SHIFT_MODES[@]}"; do
  for K in "${KS[@]}"; do
    echo start select_layer K=${K} mode=${MODE} ${FULL_MODEL}
    python src/select_layer.py \
      experiment=generate/neighbor_probe \
      model=${FULL_MODEL} \
      ${SPLITS} \
      wdown.k=${K} \
      wdown.coeff=${COEFF} \
      wdown.shift_mode=${MODE} \
      'select.layer_fraction=[0.4,0.8]' \
      task_name=${TASK_NAME}
      # select.criterion=rougeL_recall \
    echo end select_layer K=${K} mode=${MODE}
  done
done
