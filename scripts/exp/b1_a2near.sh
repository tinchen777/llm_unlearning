#!/bin/bash
# Batch 1, lane D (card 0, together with b1_a2base.sh): A2 on NEAR (= NEAR-Dist v2, method `near` of spa_lib.sh; TODO/exp/22 #5), 3B -> 1B, real + retain control.
# Shadow edit = the v2 training (E8 + retain KL 1, 300 KL steps, the provider's layers) on M' / retain95 for each S_k; NEAR calibrates its own level, no coefficient grid.
# 8B is in b1_8b.sh (card 1), not here: it would share card 0 with the 8B RMU shadows.
#   bash scripts/exp/b1_a2near.sh <GPU>      Result: saves/analysis/spa_<scale>_near_{real,retain}/summary.md  (read post_edit_mean = edit trace, pre_edit_mean = residual recognition)
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; LANE=b1_a2near; source scripts/exp/_lane.sh "$@"
for S in 3B 1B; do
  for C in real retain; do
    step a2near_${S}_${C} <<CMD || true
export GPU=${G} CUDA_VISIBLE_DEVICES=${G}; source scripts/exp/spa_lib.sh; spa_run ${S} near ${C}
CMD
  done
done
lane_end
