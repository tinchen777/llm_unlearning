#!/bin/bash
# A2 shadow-probe attack, Llama-3.2-3B forget05, NEAR-Dist: the real condition (shadows are edits of M', target = M') and the negative control (shadows are edits of the retain95 model, target = retain95).
#   bash scripts/exp/spa_lane_3b_dist.sh <GPU>                    # all 5 groups + attack, both conditions (resumable: done markers in logs/spa/)
#   KEEP_EDITED=1 SPA_GROUPS="0" bash scripts/exp/spa_lane_3b_dist.sh <GPU>   # PILOT: one group, no attack, edited models kept (~6 GB each); read the printed proxy probabilities
#   DRY_RUN=1 bash scripts/exp/spa_lane_3b_dist.sh                # only print the steps       FORCE=1: ignore the done markers
# Result: saves/analysis/spa_3B_dist_real/summary.md and saves/analysis/spa_3B_dist_retain/summary.md (activations in saves/spa/)
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
GPU=${1:-0}; export GPU
source scripts/exp/spa_lib.sh
export CUDA_VISIBLE_DEVICES=${GPU}
spa_run 3B dist real
spa_run 3B dist retain
