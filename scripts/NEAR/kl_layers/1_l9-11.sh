#!/bin/bash
# KL on/off x small coefficient on the layers 9,10,11: c {1, 1.5, 2} x KL {off, 1}. Needs 0_select_small_c.sh (c1, c1.5). See README.md.
cd "$(dirname "$0")/../../.." || exit 1
EVAL_PROFILE=${EVAL_PROFILE:-fast} COEFFS="${COEFFS:-1 1.5 2}" KL_WEIGHTS="${KL_WEIGHTS:-off 1}" LAYER_SETS="${LAYER_SETS:-9,10,11}" \
  bash scripts/NEAR/sweep_kl.sh "${1:-0}"
