#!/bin/bash
# KL + a hinge on the probability of the TRUE forget answer (near_line/17: the KL alone lowers Prob by ~0.25-0.3 but not to the retain level, and more steps do not help
# because it constrains the neighbour answer prefix, not the true answer). Layers 11-12-13 (the best "small footprint" place of near_line/17), c {1, 1.5},
# FP_WEIGHT mu {1, 3}, target 0.17 (the retain model's forget_Q_A_Prob). Compare with the same (c, layers) at KL1 without the hinge: Prob 0.556 / 0.516, wb_unsup 0.57 / 0.91.
# Read: does Prob go down to ~0.2 at (almost) the same wb_unsup (the frontier moves) or only together with the footprint / gibberish (first words, ud_bb, retain ROUGE)?
#   bash scripts/NEAR/kl_layers/5_forget_prob.sh 0                   # FPS="1 3", COEFFS="1 1.5", LAYER_SETS="11,12,13"
#   FPS="3" LAYER_SETS="9,10,11" bash scripts/NEAR/kl_layers/5_forget_prob.sh 1
cd "$(dirname "$0")/../../.." || exit 1
read -r -a FPS <<< "${FPS:-1 3}"
for fp in "${FPS[@]}"; do
  FP_WEIGHT=${fp} EVAL_PROFILE=${EVAL_PROFILE:-fast} COEFFS="${COEFFS:-1 1.5}" KL_WEIGHTS="${KL_WEIGHTS:-1}" LAYER_SETS="${LAYER_SETS:-11,12,13}" \
    bash scripts/NEAR/sweep_kl.sh "${1:-0}"
done
