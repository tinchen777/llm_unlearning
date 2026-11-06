#!/bin/bash
# Layer selection (steering, inference only) for the small coefficients c1 and c1.5 on the layers 9-13: sweep_kl.sh needs a layer_selection.json per
# coefficient (it only takes K / coeff / r_uv path from it; the layers of the W_down runs are given explicitly). `_ks` has c2 4 5 6 7 8 12.
cd "$(dirname "$0")/../../.." || exit 1
MODEL=${MODEL:-Llama-3.2-1B-Instruct} FORGET_SPLIT=${FORGET_SPLIT:-forget01} \
TASK_NAME=${TASK_NAME:-single_v2/near_tofu_${MODEL}_${FORGET_SPLIT}_ks} \
COEFFS="${COEFFS:-1 1.5}" MODES="all" LAYERS="${LAYERS:-[9,10,11,12,13]}" LAYER_FRACTION=null \
  bash scripts/NEAR/sweep_select.sh "${1:-0}"
