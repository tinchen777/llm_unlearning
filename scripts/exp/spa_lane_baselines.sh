#!/bin/bash
# A2 shadow-probe attack on the EXISTING methods (TODO/exp/22 #1, TODO/exp/23): LUNAR, ROCR (original), RMU at forget05, one scale per call; real condition (shadows = edits of M',
# target = M') and the negative control (shadows = edits of the retain95 model, target = retain95). Same sets, templates, probes, layer rules and statistics as NEAR-Act (spa_lane_*_act.sh).
#   bash scripts/exp/spa_lane_baselines.sh <GPU> [SCALE=3B] ["METHODS=lunar rocr rmu"]          # resumable (done markers in logs/spa/); a failed method does not stop the next one
#   KEEP_EDITED=1 SPA_GROUPS="0" bash scripts/exp/spa_lane_baselines.sh 0 3B lunar              # PILOT: group 0 only, no attack, edited models kept; read the printed strengths
#   DRY_RUN=1 bash scripts/exp/spa_lane_baselines.sh 0 3B                                       # only print the steps       FORCE=1: ignore the done markers
# Result: saves/analysis/spa_<SCALE>_<method>_{real,retain}/summary.md; shadow strengths in saves/analysis/spa/shadow_strength_<SCALE>_<method>_<cond>.tsv
# Needs (offline caches): ROCR -> WikiText-103 (null-space statistics, computed once per M' / retain95: data/rocr/<model folder name>/); LUNAR -> the LUNAR_harmful reference set and
# sentence-transformers/all-MiniLM-L6-v2 (layer selection); the provider's M' of each method (paths in spa_cfg).
# Disk: one edited model at a time (~2.5 / 6.5 / 16 GB at 1B / 3B / 8B) plus M' materialised once per method in saves/spa/models/.
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
GPU=${1:-0}; export GPU
SCALE=${2:-3B}
read -r -a METHODS_A <<< "${3:-lunar rocr rmu}"
source scripts/exp/spa_lib.sh
export CUDA_VISIBLE_DEVICES=${GPU}
[ "${SCALE}" = 8B ] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in "${METHODS_A[@]}"; do
  for cond in real retain; do
    echo "==================== A2 ${SCALE} ${m} ${cond} ($(date +%F\ %T)) ===================="
    spa_run "${SCALE}" "${m}" "${cond}" || echo "[FAIL] A2 ${SCALE} ${m} ${cond} (continuing)" >&2
  done
done
