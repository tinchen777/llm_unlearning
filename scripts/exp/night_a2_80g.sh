#!/bin/bash
# Tonight, 80 GB card: A2 P0 (recognition baseline on every method's forget05 model, 3B / 1B / 8B; scripts/exp/a2_p0_lane.sh), then A2 NEAR-Act 8B (5 groups, real + retain, attack).
#   bash scripts/exp/night_a2_80g.sh <GPU>     # background, log logs/night/a2_80g.log      FG=1: foreground
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; mkdir -p logs/night
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ]; then LANE_INNER=1 nohup bash "$0" "${G}" > logs/night/a2_80g.log 2>&1 & echo "night_a2_80g started (pid $!), log logs/night/a2_80g.log"; exit 0; fi
echo "start $(date)"
FG=1 bash scripts/exp/a2_p0_lane.sh "${G}" || echo "P0 lane returned an error"
bash scripts/exp/spa_lane_8b_act.sh "${G}" || echo "8B act lane returned an error"
echo "finished $(date)"
