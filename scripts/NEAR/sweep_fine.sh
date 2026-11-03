#!/bin/bash
# Fine sweep of the `all` mode (steering at every token) of the layer selection: the coefficients and the layers are the only
# things to edit, everything else is the default of scripts/NEAR/sweep_select.sh. No training, a few minutes on 1B.
#
#   bash scripts/NEAR/sweep_fine.sh [GPU_ID]
#   COEFFS="5 7" bash scripts/NEAR/sweep_fine.sh 0     # other coefficients
#
# Prints, when done, the 12 best (coeff, layer) pairs over everything swept so far for this task (incl. earlier sweeps), by unseen_gap
# (0 = the forget answers look like the answers about strangers, 1 = nothing changed) among those with degenerate_rate <= 0.1.
# Train W_down for a pair with  K=5 COEFF=<c> MODE=all W_DOWN_LAYER=[<layer>] bash scripts/NEAR/single_run.sh 0 "3 4".
cd "$(dirname "$0")/../.." || exit 1

export COEFFS=${COEFFS:-"2 4 5 6 7 8 12"}   # the sweep of 1 3 10 30 showed: 3 too weak, 10 degenerate at layers 7-9 (answer mush)
export LAYERS=${LAYERS:-"[3,4,5,6,7,8,9,10,11,12,13]"}
export MODES=all
bash scripts/NEAR/sweep_select.sh "${1:-0}"
