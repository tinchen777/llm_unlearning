#!/bin/bash
# Tuning of LUNAR for the comparison with NEAR: the coefficient of the shift (LUNAR README: 2.0; with 2 authors (forget01) and the harmful reference
# set it destroys the model: model utility 0.27, 32 % of the benign questions refused) and, optionally, the reference set, for every forget set. Per
# (forget set, reference set): r_UV once (stage 1), then layer selection + W_down + the NEAR evaluation (stages 2-4) for every coefficient; all in
# the same task folder saves/LUNAR/single/lunar_tofu_<MODEL>_<FORGET_SPLIT>_<REF>/c<coeff>_all/ (the existing c2 run is not touched).
#
#   bash scripts/LUNAR/sweep_coeff.sh [GPU_ID]                                   # coefficients 0.5 1 1.5, forget01 and forget005
#   COEFFS="0.25 0.5 1" FORGET_SPLITS="forget01" bash scripts/LUNAR/sweep_coeff.sh 0
#   REFS="LUNAR/LUNAR_harmful LUNAR/LUNAR_unverified" bash scripts/LUNAR/sweep_coeff.sh 0   # also the reference set of fictitious entities
#   DRY_RUN=1 bash scripts/LUNAR/sweep_coeff.sh                                  # only print the runs
#
# Compare with the table of NEAR: scripts/analysis/detectability_table.py saves/eval/LUNAR/single/*  (and the TOFU_SUMMARY.json of every run).
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/../.." || exit 1

GPU_ID=${1:-0}
read -r -a COEFFS_ARR <<< "${COEFFS:-0.5 1 1.5}"
read -r -a SPLITS_ARR <<< "${FORGET_SPLITS:-forget01 forget005}"
read -r -a REFS_ARR <<< "${REFS:-LUNAR/LUNAR_harmful}"
MODEL=${MODEL:-Llama-3.2-1B-Instruct}

for split in "${SPLITS_ARR[@]}"; do
  for ref in "${REFS_ARR[@]}"; do
    task=single/lunar_tofu_${MODEL}_${split}_$(basename "${ref}")
    export FORGET_SPLIT=${split} REF_DATASET=${ref} MODEL=${MODEL}
    echo "################  ${split} | ${ref}  ################"
    if [ ! -f "saves/LUNAR/${task}/r_uv.pt" ] || [ -n "${DRY_RUN}" ]; then
      bash "${SCRIPT_DIR}/single_run.sh" "${GPU_ID}" "1"
    fi
    for coeff in "${COEFFS_ARR[@]}"; do
      echo "----------------  coeff=${coeff}  ----------------"
      COEFF=${coeff} bash "${SCRIPT_DIR}/single_run.sh" "${GPU_ID}" "2 3 4"
    done
  done
done
