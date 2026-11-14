#!/bin/bash
# Batch 2, lane A (card 0): A2 shadow-probe attack on GA and NPO (TODO/exp/22 #10, TODO/exp/26), 3B -> 1B -> 8B, real + retain control each.
# Shadow edit = the provider's trainer (GradAscent / NPO, lr 2e-5) on M' for S_k, stopped at the first epoch with forget_Q_A_Prob(S_k) <= 0.25 (as RMU in TODO/exp/23).
# No edit layer: read the cv / cv_deepest rules. 8B trains all 8B parameters (~60-70 GB): keep the card for this lane at 8B.
#   bash scripts/exp/b2_a2train.sh <GPU>      (waits for card 0's batch-1 lanes (b1_a2base, b1_a2near); NO_WAIT=1 = start now)     (FG=1 / DRY_RUN=1 / FORCE=1)      Result: saves/analysis/spa_<scale>_{ga,npo}_{real,retain}/summary.md
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; LANE=b2_a2train; source scripts/exp/_lane.sh "$@"
if [ -z "${NO_WAIT}" ]; then echo "waiting for card 0's batch-1 lanes (b1_a2base, b1_a2near) to finish (NO_WAIT=1 starts at once) ..."; for l in b1_a2base b1_a2near; do wait_for $l; done; fi
for S in 3B 1B 8B; do
  step a2train_${S} <<CMD || true
bash scripts/exp/spa_lane_baselines.sh ${G} ${S} "ga npo"
CMD
done
lane_end
