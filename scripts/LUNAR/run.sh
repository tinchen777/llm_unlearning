#!/bin/bash
# The whole LUNAR chain for every (base model, forget set) in one go:
#   r_UV -> layer selection -> W_down -> eval   (1_r_uv.sh .. 4_eval.sh), once per combination.
# Settings shared by all combinations (coeff, shift modes, solver, reference set) are in _common.sh; the models and the
# forget sets to loop over are the two lists below.
#
#   bash scripts/LUNAR/run_all.sh [GPU_ID] ["STAGES"]
#   bash scripts/LUNAR/run_all.sh 0              # stages 1 2 3 4 for every model x split set, on GPU 0
#   bash scripts/LUNAR/run_all.sh 0 "3 4"        # only W_down + eval (r_uv.pt and the layer selections already exist)
#   DRY_RUN=1 bash scripts/LUNAR/run_all.sh      # only print what would run
#   CONTINUE_ON_ERROR=1 bash scripts/LUNAR/run_all.sh   # a failing combination is reported at the end, the others still run
#
# Without CONTINUE_ON_ERROR it stops at the first failing step. Outputs: saves/LUNAR/lunar_<MODEL>_<FORGET_SPLIT>_<REF>/.
# 4_eval.sh needs saves/eval/tofu_<MODEL>_<RETAIN_SPLIT>/TOFU_EVAL.json (retain model) for forget quality / privleak.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

GPU_ID=${1:-0}
STAGES=${2:-"1 2 3 4"}

BASE_MODELS=(
  Llama-3.2-1B-Instruct
  Llama-3.2-3B-Instruct
  Llama-3.1-8B-Instruct
)

# "forget_split holdout_split retain_split"
SPLIT_SETS=(
  "forget01 holdout01 retain99"
  "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

declare -A STAGE_SCRIPT=(
  [1]=1_r_uv.sh
  [2]=2_select_layer.sh
  [3]=3_w_down.sh
  [4]=4_eval.sh
)
for stage in ${STAGES}; do
  if [ -z "${STAGE_SCRIPT[${stage}]}" ]; then
    echo "unknown stage '${stage}' (valid: 1 2 3 4)" >&2
    exit 1
  fi
done

failed=()
run_combination() {   # runs in a subshell: the exports below never leak into the next combination
  set -e
  for stage in ${STAGES}; do
    script=${STAGE_SCRIPT[${stage}]}
    echo "################ ${MODEL} ${FORGET_SPLIT} | stage ${stage}: ${script} (GPU ${GPU_ID}) ################"
    if [ -n "${DRY_RUN}" ]; then
      echo "bash ${SCRIPT_DIR}/${script} ${GPU_ID}"
    else
      bash "${SCRIPT_DIR}/${script}" "${GPU_ID}"
    fi
  done
}

for base in "${BASE_MODELS[@]}"; do
  for split in "${SPLIT_SETS[@]}"; do
    read -r forget holdout retain <<< "${split}"
    combo="${base} ${forget}"
    echo "================ ${combo} ================"
    # NOTE: no `if ( ... )` / `|| ...` around the subshell: bash would switch `set -e` off inside it
    (
      export MODEL=${base} FORGET_SPLIT=${forget} HOLDOUT_SPLIT=${holdout} RETAIN_SPLIT=${retain}
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

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed combinations: ${failed[*]}" >&2
  exit 1
fi
echo "################ all done: stages ${STAGES} ################"
