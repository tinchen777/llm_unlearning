#!/bin/bash
# Is c7 the limit of NEAR? The multi-layer runs never went above c7 (single-layer steering at c10 is degenerate in 12-30 % of the answers on l7-9). c8 and c10 on
# layers 7-8-9 without KL (answer tokens, increment + sequential): does Prob go below the retain level (0.17) and what does it cost (utility, retain ROUGE,
# degenerate answers, first words)? Reference: c7 l7-9 Prob 0.206 / utility 0.569 / retain 0.588. LUNAR c1.5 reaches 0.115 (utility 0.396).
#   bash scripts/NEAR/kl_layers/6_large_c.sh 0        # COEFFS="8 10", KL_WEIGHTS="off"
cd "$(dirname "$0")/../../.." || exit 1
EVAL_PROFILE=${EVAL_PROFILE:-fast} COEFFS="${COEFFS:-8 10}" KL_WEIGHTS="${KL_WEIGHTS:-off}" LAYER_SETS="${LAYER_SETS:-7,8,9}" bash scripts/NEAR/sweep_kl.sh "${1:-0}"
