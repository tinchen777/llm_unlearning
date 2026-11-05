#!/bin/bash
# Tuning of the baselines (1_unlearn.sh): the learning rate (and the weights dtype) of every method on ONE (model, forget set). forget01 has 40 samples,
# i.e. 2 optimizer steps per epoch (batch 8 x accumulation 4): at the default lr 1e-5 the baselines hardly move in the 20 steps of 10 epochs.
# The trainer evaluates after every epoch, so the epoch is picked afterwards (scripts/analysis/baseline_table.py), not swept here.
#
#   bash scripts/unlearn/tune.sh [GPU_ID]                                # LRS="5e-5 1e-4", the 6 methods, forget01 of the 1B model
#   LRS="2e-5 5e-5 1e-4 2e-4" METHODS="NPO GradDiff" bash scripts/unlearn/tune.sh 0
#   DTYPES="bf16 float32" bash scripts/unlearn/tune.sh 0                 # also fp32 weights (bf16 = the default of the model config)
#   MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 bash scripts/unlearn/tune.sh 0
#   DRY_RUN=1 bash scripts/unlearn/tune.sh                               # only print the runs
#
# PROBE=0 (default here): only the TOFU metrics, fast. Re-run the chosen setting with PROBE=1 (the NEAR probe + detectability):
#   PROBE=1 METHODS=NPO LR=1e-4 bash scripts/unlearn/1_unlearn.sh 0
# Output: saves/tofu_unlearn/<METHOD>/tofu_<MODEL>_<FORGET_SPLIT>_lr<LR>[_fp32]/ ; the table with the best epoch of every run:
#   python scripts/analysis/baseline_table.py saves/tofu_unlearn
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/../.." || exit 1

GPU_ID=${1:-0}
read -r -a LRS <<< "${LRS:-5e-5 1e-4}"
read -r -a DTYPES <<< "${DTYPES:-bf16}"
export PROBE=${PROBE:-0}

for dtype in "${DTYPES[@]}"; do
  for lr in "${LRS[@]}"; do
    echo "################  lr=${lr} dtype=${dtype}  ################"
    if [ -n "${DRY_RUN}" ]; then
      echo "LR=${lr} DTYPE=$([ "${dtype}" = float32 ] && echo float32) METHODS=\"${METHODS}\" PROBE=${PROBE} bash ${SCRIPT_DIR}/1_unlearn.sh ${GPU_ID}"
      continue
    fi
    LR=${lr} DTYPE=$([ "${dtype}" = float32 ] && echo float32) bash "${SCRIPT_DIR}/1_unlearn.sh" "${GPU_ID}"
  done
done
