#!/bin/bash
# TEMPORARY: W_down training + evaluation (stages 3 and 4 of scripts/NEAR/single_run.sh) for the candidates of the fine sweep
# (see TODO/progress.md). The layer selections (stage 2) of these settings already exist in saves/NEAR/single/..., so nothing else is run.
#
#   bash scripts/NEAR/train_candidates.sh [GPU_ID]
#   CANDIDATES="12:6" bash scripts/NEAR/train_candidates.sh 0      # only one:  "<coeff>:<layer>[ <coeff>:<layer> ...]"
#   DRY_RUN=1 bash scripts/NEAR/train_candidates.sh                # only print the commands
#
# Output per candidate: saves/NEAR/single/near_tofu_<MODEL>_<FORGET_SPLIT>/k5_c<coeff>_all/l<layer>/   (edited model, w_down_summary.json)
#                       saves/eval/NEAR/single/near_tofu_<MODEL>_<FORGET_SPLIT>_k5_l<layer>_c<coeff>_all/   (TOFU_SUMMARY.json, ...)
# Look first at w_down_summary.json: achieved_vs_intended (cos / norm_ratio), then TOFU_SUMMARY.json: forget_quality, model_utility,
# forget_Q_A_Prob, retain_rougeL_recall, activation_auc_*.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$(dirname "$0")/../.." || exit 1

GPU_ID=${1:-0}
read -r -a CANDIDATES <<< "${CANDIDATES:-12:6 7:9}"
echo "GPU ${GPU_ID}, candidates (coeff:layer): ${CANDIDATES[*]}"

for cand in "${CANDIDATES[@]}"; do
  coeff=${cand%%:*}
  layer=${cand##*:}
  echo "################  coeff=${coeff} layer=${layer}  ################"
  K=5 COEFF=${coeff} MODE=all W_DOWN_LAYER="[${layer}]" bash scripts/NEAR/single_run.sh "${GPU_ID}" "3 4" || exit $?
done
