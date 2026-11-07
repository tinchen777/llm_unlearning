#!/bin/bash
# The hinge of near_line/19 works (layers 11-13, c1, KL1, target 0.17: Prob 0.556 -> 0.081 at the same wb_unsup 0.57) but OVERSHOOTS the target (0.081 < 0.17, mia 0.22 vs
# the retain model's 0.48: over-forgetting). Calibrate: a higher target. c1, mu 1, layers 11-13, target {0.25, 0.35}; and the shallower layers 9-11 (lower white-box footprint
# at the same c) with target 0.35. Want: Prob ~0.17, mia ~0.48, extraction ~0.06 while wb_unsup stays ~0.5-0.6.
#   bash scripts/NEAR/kl_layers/7_forget_prob_target.sh 0
#   TARGETS="0.35" LAYER_SETS="9,10,11" bash scripts/NEAR/kl_layers/7_forget_prob_target.sh 1
cd "$(dirname "$0")/../../.." || exit 1
read -r -a TARGETS <<< "${TARGETS:-0.25 0.35}"
for t in "${TARGETS[@]}"; do
  FP_WEIGHT=${FP_WEIGHT:-1} FP_TARGET=${t} EVAL_PROFILE=${EVAL_PROFILE:-fast} COEFFS="${COEFFS:-1}" KL_WEIGHTS="${KL_WEIGHTS:-1}" LAYER_SETS="${LAYER_SETS:-11,12,13}" \
    bash scripts/NEAR/sweep_kl.sh "${1:-0}"
done
