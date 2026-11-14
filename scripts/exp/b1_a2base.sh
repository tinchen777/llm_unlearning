#!/bin/bash
# Batch 1, lane A (card 0, together with b1_a2near.sh): A2 shadow-probe attack on LUNAR / ROCR / RMU (TODO/exp/23), 3B -> 8B -> 1B, real + retain control each.
# Resumes the pilot (done markers of logs/spa/ are shared). The 8B RMU shadows train all 8B parameters (~60-70 GB): b1_a2near.sh does not run 8B on this card.
#   bash scripts/exp/b1_a2base.sh <GPU>      (FG=1 / DRY_RUN=1 / FORCE=1)      Result: saves/analysis/spa_<scale>_<lunar|rocr|rmu>_{real,retain}/summary.md
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; LANE=b1_a2base; source scripts/exp/_lane.sh "$@"
for S in 3B 8B 1B; do
  step a2base_${S} <<CMD || true
bash scripts/exp/spa_lane_baselines.sh ${G} ${S}
CMD
done
lane_end
