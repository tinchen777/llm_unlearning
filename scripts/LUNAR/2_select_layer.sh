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

# Step 2: layer selection (needs 1_r_uv.sh), once per shift mode. For every candidate layer the forget prompts are
# redirected at inference time (block_out += coeff * r_UV) and the responses are scored by s1 - s2 (sentence-embedding
# similarity to desired minus similarity to unrelated responses):
#   -> <coeff,mode folder>/layer_selection.json   per layer: s1, s2, score, response statistics; `selected_layer`, `ranking`. READ IT.
#   -> <coeff,mode folder>/responses_steer_l<l>.json   steered forget answers of every layer
#   folder = saves/LUNAR/<TASK_NAME>/c<coeff>_<mode>/   (config_dir in _common.sh)
# The sentence encoder (select.embedder) must be in the local HF cache (HF_HUB_OFFLINE=1).
# Candidates: select.layer_fraction=[0.4,0.8] (default); 'select.layers=[4,5,6]' for an explicit list.
for MODE in "${SHIFT_MODES[@]}"; do
  echo start select_layer mode=${MODE} ${FULL_MODEL}
  python src/METHOD/LUNAR/select_layer.py \
    experiment=generate/LUNAR \
    model=${FULL_MODEL} \
    "${SPLIT_ARGS[@]}" \
    "data/datasets@data.ref.datasets=${REF_DATASET}" \
    wdown.coeff=${COEFF} \
    wdown.shift_mode=${MODE} \
    task_name=${TASK_NAME}
  echo end select_layer mode=${MODE}
done
