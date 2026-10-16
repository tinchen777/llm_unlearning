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

# Step 1: the unlearning vector of every layer on the TOFU-full model (needs: python setup/setup_data.py --lunar):
#   -> r_uv.pt           r_UV = mean(D_ref) - mean(D_forget), [n_pos, n_layers, d], + the forget / reference means
#   -> r_uv_summary.json per layer: ||r_UV||, ||r_UV|| / ||forget activation||
# output: saves/LUNAR/<TASK_NAME>/   (settings: `ruv` in configs/experiment/generate/LUNAR.yaml)
echo start r_uv ${FULL_MODEL} ${REF_DATASET}
python src/METHOD/LUNAR/r_uv.py \
  experiment=generate/LUNAR \
  model=${FULL_MODEL} \
  "${SPLIT_ARGS[@]}" \
  "data/datasets@data.ref.datasets=${REF_DATASET}" \
  task_name=${TASK_NAME}
echo end r_uv ${FULL_MODEL}
