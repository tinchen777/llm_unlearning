#!/bin/bash
# The hinge of near_line/19 works (layers 11-13, c1, KL1, target 0.17: Prob 0.556 -> 0.081 at the same wb_unsup 0.57) but OVERSHOOTS (0.081 < 0.17, mia 0.22 vs 0.48).
# Two PRINCIPLED fixes, neither needs the retain model (the earlier target 0.17 WAS the retain model's value; FP_TARGET=unseen = the ORIGINAL model's probability on the
# answers of the unseen holdout authors, 1B 0.161 ~ retain 0.169): the target `unseen` with the `hinge` and the `match` (two-sided) mode, c1, mu 1, layers 11-13 and 9-11.
# FP_TARGET=0.25 / 0.35 (a heuristic compensation of the overshoot, tuned against retain-level metrics = a use of the retain reference) is only for comparison.
#   bash scripts/NEAR/kl_layers/7_forget_prob_target.sh 0                  # MODES="match hinge", LAYER_SETS="11,12,13"
#   LAYER_SETS="9,10,11" bash scripts/NEAR/kl_layers/7_forget_prob_target.sh 1
cd "$(dirname "$0")/../../.." || exit 1
read -r -a MODES <<< "${MODES:-match hinge}"
for mode in "${MODES[@]}"; do
  FP_WEIGHT=${FP_WEIGHT:-1} FP_TARGET=${FP_TARGET:-unseen} FP_MODE=${mode} EVAL_PROFILE=${EVAL_PROFILE:-fast} COEFFS="${COEFFS:-1}" KL_WEIGHTS="${KL_WEIGHTS:-1}" LAYER_SETS="${LAYER_SETS:-11,12,13}" \
    bash scripts/NEAR/sweep_kl.sh "${1:-0}"
done
