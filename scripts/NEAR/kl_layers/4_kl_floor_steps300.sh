#!/bin/bash
# Is the KL term at its floor? In every run so far it goes 4.2 -> ~0.5 in 100 steps (same for lambda 1 and 3). c2, layers 11-12, KL 1, 300 steps
# (lr 2e-5 as before): if the KL ends clearly below 0.4 the earlier runs were under-trained; if it stays ~0.5 it is a floor (the target depends on
# which neighbour is drawn) and a larger lambda cannot help. The run name has _s300; compare with rw1_ans_kl1_increment_seq_c2_l11-12 (100 steps).
cd "$(dirname "$0")/../../.." || exit 1
EVAL_PROFILE=${EVAL_PROFILE:-fast} KL_STEPS=${KL_STEPS:-300} COEFFS="${COEFFS:-2}" KL_WEIGHTS="${KL_WEIGHTS:-1}" LAYER_SETS="${LAYER_SETS:-11,12}" \
  bash scripts/NEAR/sweep_kl.sh "${1:-0}"
