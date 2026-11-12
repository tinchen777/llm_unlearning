#!/bin/bash
# Tonight, 35 GB card: A2 NEAR-Act 3B (all 5 groups, real + retain control, then the attack; group 0 is already done), then the same at 1B.
# Peak ~10 GB at 3B (W_down regression + one forward model), each coefficient trial ~5 min, ~5-6 trials per group and condition.
#   bash scripts/exp/night_a2_35g.sh <GPU>     # background, log logs/night/a2_35g.log      FG=1: foreground
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; mkdir -p logs/night
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ]; then LANE_INNER=1 nohup bash "$0" "${G}" > logs/night/a2_35g.log 2>&1 & echo "night_a2_35g started (pid $!), log logs/night/a2_35g.log"; exit 0; fi
echo "start $(date)"
bash scripts/exp/spa_lane_3b_act.sh "${G}" || echo "3B act lane returned an error"
bash scripts/exp/spa_lane_1b_act.sh "${G}" || echo "1B act lane returned an error"
echo "finished $(date)"
