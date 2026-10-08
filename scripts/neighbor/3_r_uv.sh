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


# Step 2a: r_UV per forget author on the TOFU-full model, ablated over the number K of neighbours used as reference:
#   -> r_uv.pt           r_UV[K][author] = [n_pos, n_layers, d], + the "before" activations, neighbour order, ...
#   -> r_uv_summary.json per K and layer: ||r_UV||, ||r_UV|| / ||forget activation||, cos(r_UV[K], r_UV[max K])
# output: saves/neighbor/<TASK_NAME>/   (settings: `ruv` in configs/experiment/generate/neighbor_probe.yaml)
# NOTE: `ruv.qa_per_author` / `ruv.author_offset` must make the author numbering of the forget rows match
#       `author_id` of the neighbour csv; this is checked (and explained) before any forward pass.
echo start r_uv ${FULL_MODEL}
python src/r_uv.py \
  experiment=generate/neighbor_probe \
  model=${FULL_MODEL} \
  ${SPLITS} \
  "ruv.ks=[$(IFS=,; echo "${KS[*]}")]" \
  task_name=${TASK_NAME}
  # ruv.author_offset=0 \
  # ruv.point=block_out \
echo end r_uv ${FULL_MODEL}
