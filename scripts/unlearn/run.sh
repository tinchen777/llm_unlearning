#!/bin/bash
# The TOFU baselines (1_unlearn.sh: the methods of METHODS in _common.sh) for every (base model, forget set) in one go,
# the same loop as scripts/NEAR/run.sh.
#
#   bash scripts/unlearn/run.sh [GPU_ID]
#   METHODS="GradDiff NPO" bash scripts/unlearn/run.sh 0     # only some methods
#   DRY_RUN=1 bash scripts/unlearn/run.sh                    # only print what would run
#   CONTINUE_ON_ERROR=1 bash scripts/unlearn/run.sh          # a failing combination is reported at the end, the others still run
#
# Without CONTINUE_ON_ERROR it stops at the first failing combination.
# Outputs: saves/unlearn/tofu_<MODEL>_<FORGET_SPLIT>/<METHOD>/.
# Needs saves/eval/tofu_<MODEL>_<RETAIN_SPLIT>/TOFU_EVAL.json (official retain model) for forget quality / privleak.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

GPU_ID=${1:-0}

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

failed=()
for base in "${BASE_MODELS[@]}"; do
  for split in "${SPLIT_SETS[@]}"; do
    read -r forget holdout retain <<< "${split}"
    combo="${base} ${forget}"
    echo "================ ${combo} ================"
    # NOTE: no `if ( ... )` / `|| ...` around the subshell: bash would switch `set -e` off inside it
    (
      export MODEL=${base} FORGET_SPLIT=${forget} HOLDOUT_SPLIT=${holdout} RETAIN_SPLIT=${retain}
      if [ -n "${DRY_RUN}" ]; then
        echo "bash ${SCRIPT_DIR}/1_unlearn.sh ${GPU_ID}   (MODEL=${MODEL} FORGET_SPLIT=${FORGET_SPLIT} ...)"
      else
        bash "${SCRIPT_DIR}/1_unlearn.sh" "${GPU_ID}"
      fi
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
echo "################ all done ################"
