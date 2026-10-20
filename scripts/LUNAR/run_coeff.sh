#!/bin/bash
# Coefficient sweep of LUNAR: layer selection -> W_down -> eval (stages 2 3 4) once per COEFF, for the chosen models / forget sets.
# r_uv.pt does not depend on the coefficient: it must already exist (stage 1 of run.sh). Every coefficient writes its own folder
# (saves/LUNAR/<TASK_NAME>/c<COEFF>_<mode>/ and its own eval folders), nothing is overwritten.
#
#   bash scripts/LUNAR/run_coeff.sh [GPU_ID]
#   COEFFS="2 4" MODELS="Llama-3.2-1B-Instruct" FORGETS="forget01" bash scripts/LUNAR/run_coeff.sh 0
#   MODES="all" bash scripts/LUNAR/run_coeff.sh 0   # only the `all` mode
#   DRY_RUN=1 bash scripts/LUNAR/run_coeff.sh        # only print what would run
#   CONTINUE_ON_ERROR=1 bash scripts/LUNAR/run_coeff.sh
#
# Defaults: COEFFS="2 4 8 16", MODELS="Llama-3.2-1B-Instruct", FORGETS="forget01" (a forget set key of SPLIT_SETS below),
# STAGES="2 3 4", MODES as in _common.sh (all + eoi).

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

GPU_ID=${1:-0}
read -r -a COEFF_LIST <<< "${COEFFS:-2 4 8 16}"
read -r -a MODEL_LIST <<< "${MODELS:-Llama-3.2-1B-Instruct}"
read -r -a FORGET_LIST <<< "${FORGETS:-forget01}"
STAGES=${STAGES:-"2 3 4"}

declare -A SPLIT_SET=(
  [forget01]="holdout01 retain99"
  [forget05]="holdout05 retain95"
  [forget10]="holdout10 retain90"
)
declare -A STAGE_SCRIPT=([2]=2_select_layer.sh [3]=3_w_down.sh [4]=4_eval.sh)
for stage in ${STAGES}; do
  if [ -z "${STAGE_SCRIPT[${stage}]}" ]; then
    echo "unknown stage '${stage}' (valid: 2 3 4)" >&2
    exit 1
  fi
done

failed=()
run_combination() {   # runs in a subshell: the exports of the caller never leak into the next combination
  set -e
  source "${SCRIPT_DIR}/_common.sh"
  if [ ! -f "saves/LUNAR/${TASK_NAME}/r_uv.pt" ]; then
    echo "saves/LUNAR/${TASK_NAME}/r_uv.pt not found: run stage 1 (1_r_uv.sh / run.sh) first" >&2
    exit 1
  fi
  for stage in ${STAGES}; do
    script=${STAGE_SCRIPT[${stage}]}
    echo "################ ${MODEL} ${FORGET_SPLIT} COEFF=${COEFF} | stage ${stage}: ${script} (GPU ${GPU_ID}) ################"
    if [ -n "${DRY_RUN}" ]; then
      echo "bash ${SCRIPT_DIR}/${script} ${GPU_ID}"
    else
      bash "${SCRIPT_DIR}/${script}" "${GPU_ID}"
    fi
  done
}

cd "${SCRIPT_DIR}/../.." || exit 1
for base in "${MODEL_LIST[@]}"; do
  for forget in "${FORGET_LIST[@]}"; do
    if [ -z "${SPLIT_SET[${forget}]}" ]; then
      echo "unknown forget set '${forget}' (valid: ${!SPLIT_SET[*]})" >&2
      exit 1
    fi
    read -r holdout retain <<< "${SPLIT_SET[${forget}]}"
    for coeff in "${COEFF_LIST[@]}"; do
      combo="${base} ${forget} c=${coeff}"
      echo "================ ${combo} ================"
      # NOTE: no `if ( ... )` / `|| ...` around the subshell: bash would switch `set -e` off inside it
      (
        export MODEL=${base} FORGET_SPLIT=${forget} HOLDOUT_SPLIT=${holdout} RETAIN_SPLIT=${retain} COEFF=${coeff}
        run_combination
      )
      status=$?
      if [ ${status} -eq 0 ]; then
        echo "---------------- ${combo}: done"
      else
        failed+=("${combo}")
        echo "---------------- ${combo}: FAILED (exit ${status})" >&2
        [ -z "${CONTINUE_ON_ERROR}" ] && exit ${status}
      fi
    done
  done
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
echo "################ coefficient sweep done: ${COEFF_LIST[*]} ################"
